# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mcp[cli]",
#     "httpx",
#     "sqlite-vec",
#     "sentence-transformers",
#     "numpy",
#     "einops",
#     "torch",
# ]
# ///
"""Mechanical batch acquisition for academic papers.

Given a chunk JSON file (from /acquire's partition-sources.py), downloads each
paper via OA PDF URL, arXiv, or direct URL; processes with Docling; and
enhances arXiv papers with TeX. Papers needing LLM fallback (web search,
abstract generation) are written to a separate output file for agent handling.

The chain hits three sources by default: URLs explicitly flagged as
open access in Unpaywall / S2 metadata, arXiv PDFs by arXiv ID, and
direct URLs supplied in the chunk's paper metadata (the caller is
responsible for not pointing it at sources they shouldn't download
from). PMC, sci-hub, Anna's Archive, and other gray-license sources
are out of scope for v0.1.0 — to add one for your own use, see
docs/DOWNLOAD_BACKENDS.md.

Shares the same dependency set as server.py so that imported functions
(process_pdf, download_arxiv_tex, etc.) work correctly and papers get
embedded immediately.

Usage:
    uv run acquire_batch.py <chunk.json>
    uv run acquire_batch.py <chunk.json> --dry-run
"""

import argparse
import asyncio
import json
import os
import re
import signal
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# Import from the MCP server (same directory).
# server.py's heavy imports (torch, sentence-transformers) are lazy-loaded
# behind _get_embed_model() — they only fire on the first process_pdf call.
sys.path.insert(0, str(Path(__file__).parent.parent))
from server import (
    process_pdf,
    download_arxiv_tex,
    fix_orphan_paper,
    _store_papers,
    _normalize_id,
    _Throttle,
    _init_db,
    _verify_paper_text_match,
    _clear_paper_text,
    _parse_stored_pid,
    _parse_chars,
    _process_pdf_and_verify,
    _try_claim_paper,
    _release_paper_claim,
    _refresh_paper_claim,
    S2_BASE,
    S2_API_KEY,
    DEFAULT_FIELDS,
    BROWSER_UA,
    PAPERS_DIR,
    DB_PATH,
    _s2_search_throttle,
)

# Back-compat local alias: the helper lives in server.py now so process_inbox.py
# and any other caller can share the single implementation. Keep the local name
# pointing at the shared impl so existing acquire_batch call sites don't need
# to change.
_process_and_verify = _process_pdf_and_verify


# Per-run identifier used as a staging-filename suffix so two concurrent
# acquire_batch processes working on the same paper_id never collide on the
# same path under staging/. pid+uuid guards against pid reuse across
# short-lived workers on the same host.
_WORKER_TAG = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"


def _staging_name(prefix: str, paper_id: str, fallback: str = "") -> str:
    """Build a cross-worker-unique staging filename.

    prefix is the source tag (oa/arxiv/url). `paper_id` is sanitized
    and suffixed with _WORKER_TAG so concurrent workers downloading the
    same paper_id do not overwrite each other's staging PDF. `fallback`
    is used when paper_id is empty.
    """
    base = paper_id or fallback or "paper"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)[:80]
    return f"{prefix}_{safe}_{_WORKER_TAG}.pdf"

import sqlite3

_shutdown = False


def _on_signal(sig, frame):
    global _shutdown
    _shutdown = True
    print("\n[signal] Finishing current paper, then stopping...", file=sys.stderr)


signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


# ---------------------------------------------------------------------------
# S2 API helpers (raw JSON, not the MCP-tool string wrappers)
# ---------------------------------------------------------------------------

async def _s2_batch_lookup(client: httpx.AsyncClient, ids: list[str]) -> list[dict | None]:
    """POST /paper/batch — returns raw JSON list aligned with input IDs."""
    await _s2_search_throttle.wait()
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    resp = await client.post(
        f"{S2_BASE}/paper/batch",
        params={"fields": DEFAULT_FIELDS},
        json={"ids": ids},
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


async def _s2_title_match(client: httpx.AsyncClient, title: str) -> tuple[dict | None, float]:
    """GET /paper/search/match — returns (paper_dict, score) or (None, 0)."""
    await _s2_search_throttle.wait()
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    try:
        resp = await client.get(
            f"{S2_BASE}/paper/search/match",
            params={"query": title, "fields": DEFAULT_FIELDS},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None, 0.0
        raise
    data = resp.json()
    paper = data.get("data", [{}])[0] if data.get("data") else data
    score = float(data.get("matchScore") or paper.get("matchScore") or 0)
    return paper, score


def _title_similarity(query: str, candidate: str) -> float:
    """Bidirectional Jaccard-like similarity between two titles.

    Returns the minimum of (forward containment, reverse containment) so a
    query matches only when both directions have substantial overlap.

    One-sided containment was unsafe: a short query trivially scored 1.0
    against any longer candidate that contained it, including distinct
    papers in the same series.
    """
    def words(t: str) -> set[str]:
        return set(re.sub(r"[^\w\s]", "", t.lower()).split()) - {
            "the", "a", "an", "of", "and", "in", "on", "for", "to", "with",
        }
    wq, wc = words(query), words(candidate)
    if not wq or not wc:
        return 0.0
    shared = len(wq & wc)
    fwd = shared / len(wq)
    rev = shared / len(wc)
    return min(fwd, rev)


async def _resolve_doi_by_title(
    client: httpx.AsyncClient,
    title: str,
    year: int | None = None,
) -> tuple[str | None, str | None]:
    """Resolve a title to a DOI via three cascading sources.

    Returns (doi, source_name) where source_name is "s2", "openalex", "crossref",
    or None if nothing is found with sufficient confidence.
    """
    normalized_title = (title or "").strip()
    if len(normalized_title) < 10:
        return None, None

    try:
        paper, score = await _s2_title_match(client, normalized_title)
        if paper and score >= 0.5:
            doi = _normalize_doi_value((paper.get("externalIds") or {}).get("DOI"))
            if doi:
                return doi, "s2"
    except Exception:
        pass

    try:
        params: dict[str, str | int] = {"search": normalized_title, "per_page": 3}
        if year is not None:
            params["filter"] = f"publication_year:{year}"
        await _openalex_throttle.wait()
        resp = await client.get("https://api.openalex.org/works", params=params, timeout=30)
        resp.raise_for_status()
        work = (resp.json().get("results") or [None])[0]
        if isinstance(work, dict):
            candidate_title = (work.get("title") or "").strip()
            doi = _normalize_doi_value(work.get("doi"))
            if doi and _title_similarity(normalized_title, candidate_title) >= 0.85:
                return doi, "openalex"
    except Exception:
        pass

    try:
        params = {"query.title": normalized_title, "rows": 3}
        if year is not None:
            params["filter"] = f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31"
        await _crossref_throttle.wait()
        resp = await client.get("https://api.crossref.org/works", params=params, timeout=30)
        resp.raise_for_status()
        item = ((resp.json().get("message") or {}).get("items") or [None])[0]
        if isinstance(item, dict):
            titles = item.get("title") or []
            candidate_title = titles[0].strip() if titles else ""
            doi = _normalize_doi_value(item.get("DOI"))
            if doi and _title_similarity(normalized_title, candidate_title) >= 0.85:
                return doi, "crossref"
    except Exception:
        pass

    return None, None


# ---------------------------------------------------------------------------
# Result parsing helpers
# ---------------------------------------------------------------------------
# _parse_chars is imported from server.py (single source of truth).

def _parse_orphan_id(text: str) -> str | None:
    m = re.search(r"METADATA_NEEDED.*?stored as `(local:[^`]+)`", text, re.DOTALL)
    return m.group(1) if m else None


def _normalize_doi_value(value: str | None) -> str | None:
    if not value:
        return None
    doi = value.strip()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    return doi or None


def _coerce_year(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"\d{4}", stripped):
            return int(stripped)
    return None


def _paper_meta_defaults(source: dict) -> dict:
    return {
        "paper_id": "",
        "oa_url": None,
        "arxiv_id": None,
        "doi": _normalize_doi_value(source.get("doi")),
        "title": (source.get("title") or "").strip(),
        "year": _coerce_year(source.get("year")),
        # 2026-04-26: authors carried alongside title for the post-DOI-gate
        # verifier. Source spec may already provide a list[str] or string;
        # otherwise paths populate from S2/OpenAlex response or fall back
        # to fetching from papers.db at verify time.
        "authors": source.get("authors") or None,
    }


def _authors_for_verify(paper_id: str, meta_authors: Any) -> Any:
    """Return a usable target_authors value for _verify_paper_text_match.

    Prefer meta-supplied authors when available. Otherwise look up the
    `authors` column for paper_id from papers.db (S2/OpenAlex responses
    are already persisted there by _store_papers).
    """
    if meta_authors:
        return meta_authors
    if not paper_id:
        return None
    try:
        conn = _init_db()
    except Exception:
        return None
    try:
        row = conn.execute(
            "SELECT authors FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        return row[0] if row and row[0] else None
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


_METHOD_DETAIL_LIMIT = 200


def _method_detail(text: str, *, first_line: bool = False) -> str:
    if not text:
        return ""
    detail = text.splitlines()[0] if first_line else text
    return detail.replace("\r", " ").replace("\n", " ").strip()[:_METHOD_DETAIL_LIMIT]


def _record_method(
    methods_attempted: list[dict], method: str, status: str, detail: str,
    *, first_line: bool = False,
) -> None:
    methods_attempted.append({
        "method": method,
        "status": status,
        "detail": _method_detail(detail, first_line=first_line),
    })


def _classify_unavailability(
    doi: str | None, paper_id: str | None, title: str, methods_attempted: list[dict],
) -> str:
    if not doi and not paper_id and not title:
        return "no_identifiers"
    statuses = [attempt["status"] for attempt in methods_attempted]
    if any(status == "cloudflare_block" for status in statuses):
        return "cloudflare_block"
    download_methods = [
        attempt
        for attempt in methods_attempted
        if attempt.get("method") != "doi_resolver"
    ]
    if not statuses or (
        download_methods
        and all((attempt.get("status") or "").startswith("skipped_") for attempt in download_methods)
    ):
        return "skipped_all"
    return "exhausted_chain"


# ---------------------------------------------------------------------------
# Shared throttles for DOI resolution
# ---------------------------------------------------------------------------

_openalex_throttle = _Throttle(0.125)  # 8 RPS for OpenAlex
_crossref_throttle = _Throttle(0.1)  # 10 RPS for Crossref


# ---------------------------------------------------------------------------
# METADATA_NEEDED handler (mechanical: S2 title match → fix_orphan_paper)
# ---------------------------------------------------------------------------

async def _handle_metadata_needed(
    client: httpx.AsyncClient, orphan_id: str, title: str, doi: str | None
) -> None:
    paper, score = await _s2_title_match(client, title)
    if paper and score >= 0.85:
        s2_id = paper.get("paperId", "")
        _store_papers([paper])
        if s2_id:
            await fix_orphan_paper(orphan_id, canonical_paper_id=s2_id)
            return
    # No good match — in-place update with what we have from the chunk
    await fix_orphan_paper(orphan_id, title=title, doi=doi or "")


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_partial(project_dir: Path, chunk_id: str, entries: list[dict]) -> None:
    out = project_dir / f"acquisition-partial-{chunk_id}.json"
    out.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "chunk_id": chunk_id,
        "papers": entries,
    }, indent=2))


def _write_needs_llm(
    project_dir: Path, chunk_id: str,
    web_search: list[dict], abstracts: list[dict],
) -> None:
    out = project_dir / f"acquisition-needs-llm-{chunk_id}.json"
    out.write_text(json.dumps({
        "chunk_id": chunk_id,
        "needs_web_search": web_search,
        "needs_abstract": abstracts,
    }, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    ap = argparse.ArgumentParser(description="Mechanical batch paper acquisition")
    ap.add_argument("chunk_file", help="Path to chunk-paper-N.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    chunk_path = Path(args.chunk_file).resolve()
    if not chunk_path.exists():
        print(f"[error] Not found: {chunk_path}", file=sys.stderr)
        sys.exit(1)

    chunk = json.loads(chunk_path.read_text())
    chunk_id = chunk["chunk_id"]
    sources = [s for s in chunk["sources"] if s.get("type") == "paper"]
    # Detect project_dir: if file is in acquire-chunks/ subdir, go up two levels;
    # otherwise (researcher agents write directly to project_dir) go up one level.
    if chunk_path.parent.name == "acquire-chunks":
        project_dir = chunk_path.parent.parent
    else:
        project_dir = chunk_path.parent

    print(f"[start] chunk={chunk_id}  papers={len(sources)}  dry_run={args.dry_run}", file=sys.stderr)

    # ── Phase 1: Batch metadata lookup ─────────────────────────────────────
    # Separate papers with IDs from title-only papers.
    id_pairs: list[tuple[int, str]] = []   # (source_index, normalized_id)
    title_pairs: list[tuple[int, str]] = []

    openalex_pairs: list[tuple[int, str]] = []  # (source_index, openalex_id)

    for i, s in enumerate(sources):
        if s.get("s2_id"):
            id_pairs.append((i, s["s2_id"]))
        elif s.get("doi"):
            id_pairs.append((i, _normalize_id(s["doi"])))
        elif s.get("openalex_id"):
            openalex_pairs.append((i, s["openalex_id"]))
        elif s.get("title"):
            title_pairs.append((i, s["title"]))

    # paper_meta[source_index] → {paper_id, oa_url, arxiv_id, doi, title, year, ...}
    paper_meta: dict[int, dict] = {
        i: _paper_meta_defaults(s)
        for i, s in enumerate(sources)
    }

    async with httpx.AsyncClient(headers={"User-Agent": BROWSER_UA}, timeout=30) as api:
        # Batch lookup (chunk into 500-ID batches per S2 API limit)
        if id_pairs:
            ids = [nid for _, nid in id_pairs]
            print(f"  S2 batch: {len(ids)} papers with IDs", file=sys.stderr)
            try:
                # S2 /paper/batch accepts max 500 IDs per call
                all_results: list[dict | None] = []
                for batch_start in range(0, len(ids), 500):
                    batch_ids = ids[batch_start:batch_start + 500]
                    batch_results = await _s2_batch_lookup(api, batch_ids)
                    all_results.extend(batch_results)
                results = all_results
                found = [p for p in results if p]
                _store_papers(found)
                print(f"  → found {len(found)}/{len(ids)}", file=sys.stderr)
                for (idx, _), paper in zip(id_pairs, results):
                    if not paper:
                        continue
                    ext = paper.get("externalIds") or {}
                    oa = paper.get("openAccessPdf") or {}
                    arxiv = ext.get("ArXiv")
                    doi_val = _normalize_doi_value(ext.get("DOI")) or ""
                    # Fallback: extract arXiv ID from DOI (10.48550/arXiv.YYMM.NNNNN)
                    if not arxiv and doi_val.startswith("10.48550/arXiv."):
                        arxiv = doi_val.split("arXiv.", 1)[1]
                    meta = paper_meta.setdefault(idx, _paper_meta_defaults(sources[idx]))
                    meta.update({
                        "paper_id": paper.get("paperId", ""),
                        "oa_url": oa.get("url"),
                        "arxiv_id": arxiv,
                        "doi": doi_val,
                    })
            except Exception as e:
                print(f"  [error] batch lookup: {e}", file=sys.stderr)

        # Title matching (sequential, 1 RPS)
        for idx, title in title_pairs:
            if _shutdown:
                break
            try:
                paper, score = await _s2_title_match(api, title)
                if paper and score >= 0.5:
                    _store_papers([paper])
                    ext = paper.get("externalIds") or {}
                    oa = paper.get("openAccessPdf") or {}
                    arxiv = ext.get("ArXiv")
                    doi_val = _normalize_doi_value(ext.get("DOI")) or ""
                    if not arxiv and doi_val.startswith("10.48550/arXiv."):
                        arxiv = doi_val.split("arXiv.", 1)[1]
                    meta = paper_meta.setdefault(idx, _paper_meta_defaults(sources[idx]))
                    meta.update({
                        "paper_id": paper.get("paperId", ""),
                        "oa_url": oa.get("url"),
                        "arxiv_id": arxiv,
                        "doi": doi_val,
                    })
                    print(f"  title [{score:.2f}]: {title[:55]}", file=sys.stderr)
                else:
                    print(f"  no match: {title[:55]}", file=sys.stderr)
            except Exception as e:
                print(f"  [error] title match: {e}", file=sys.stderr)

        # OpenAlex ID resolution (get DOI from OpenAlex API)
        if openalex_pairs:
            print(f"  OpenAlex resolve: {len(openalex_pairs)} papers", file=sys.stderr)
            for idx, oa_id in openalex_pairs:
                if _shutdown:
                    break
                try:
                    await _openalex_throttle.wait()
                    resp = await api.get(
                        f"https://api.openalex.org/works/{oa_id}",
                        params={"select": "id,doi,title,open_access"},
                    )
                    resp.raise_for_status()
                    work = resp.json()
                    doi_val = _normalize_doi_value(work.get("doi")) or ""
                    oa_info = work.get("open_access") or {}
                    oa_url = oa_info.get("oa_url")
                    title = (work.get("title") or sources[idx].get("title") or "").strip()
                    meta = paper_meta.setdefault(idx, _paper_meta_defaults(sources[idx]))
                    if doi_val or oa_url:
                        meta.update({
                            "oa_url": oa_url,
                            "doi": doi_val,
                            "title": title or meta.get("title", ""),
                        })
                        print(f"  OA resolved {oa_id}: DOI={doi_val or 'none'} OA={'yes' if oa_url else 'no'}", file=sys.stderr)
                    else:
                        print(f"  OA resolved {oa_id}: no DOI, no OA -> DOI resolver", file=sys.stderr)
                except Exception as e:
                    print(f"  [error] OpenAlex resolve {oa_id}: {e}", file=sys.stderr)

        for idx, meta in paper_meta.items():
            if _shutdown:
                break
            if meta.get("doi"):
                continue
            title = meta.get("title")
            if not title:
                continue
            year = meta.get("year")
            doi, source = await _resolve_doi_by_title(api, title, year)
            if doi:
                meta["doi"] = doi
                meta.setdefault(
                    "_resolver_event",
                    {
                        "method": "doi_resolver",
                        "status": f"resolved_{source}",
                        "detail": _method_detail(doi),
                    },
                )
            else:
                meta.setdefault(
                    "_resolver_event",
                    {
                        "method": "doi_resolver",
                        "status": "not_found",
                        "detail": _method_detail(title),
                    },
                )

    # ── Phase 1.5: Library dedup ─────────────────────────────────────────
    already_have: set[int] = set()
    dedup_conn = sqlite3.connect(str(DB_PATH), timeout=10)
    for i, src in enumerate(sources):
        meta = paper_meta.get(i, {})
        pid = meta.get("paper_id", "")
        doi_val = meta.get("doi") or src.get("doi")
        row = None
        if pid:
            # Consider BOTH text sources: a TeX-only paper (processed_text
            # empty, tex_text populated) is already cite-ready and should
            # not be redownloaded. Mirrors the DOI fallback below.
            row = dedup_conn.execute(
                "SELECT MAX(LENGTH(COALESCE(processed_text,'')), LENGTH(COALESCE(tex_text,''))) AS body_len "
                "FROM papers WHERE paper_id = ?",
                (pid,),
            ).fetchone()
        if not row and doi_val:
            # Case-insensitive DOI match (mixed-case DOIs like SICI exist
            # in the DB). MAX(LENGTH, LENGTH) is the 2-arg SCALAR max —
            # one row at a time — so we order by it and take the best
            # match, so duplicate-DOI rows always report the largest
            # text. Matches server.py's other DOI lookups.
            row = dedup_conn.execute(
                "SELECT MAX(LENGTH(COALESCE(processed_text,'')), LENGTH(COALESCE(tex_text,''))) AS body_len "
                "FROM papers WHERE LOWER(doi) = LOWER(?) "
                "ORDER BY body_len DESC LIMIT 1",
                (doi_val,),
            ).fetchone()
        if row and row[0] and row[0] > 500:
            already_have.add(i)
    dedup_conn.close()
    if already_have:
        print(f"  Dedup: skipping {len(already_have)} papers already in library", file=sys.stderr)

    # ── Phase 2: Download loop ─────────────────────────────────────────────
    entries: list[dict] = []
    needs_web_search: list[dict] = []
    needs_abstract: list[dict] = []
    n_downloaded = 0

    staging = PAPERS_DIR.parent / "downloads"
    staging.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(sources):
        if _shutdown:
            print(f"  [signal] stopped at {i}/{len(sources)}", file=sys.stderr)
            break

        # Skip papers already in library with full text
        if i in already_have:
            meta = paper_meta.get(i, {})
            entries.append({
                "title": src.get("title", "Unknown"), "type": "paper",
                "paper_id": meta.get("paper_id") or None,
                "doi": meta.get("doi") or src.get("doi") or None,
                "url": src.get("url"),
                "status": "already_in_library", "unavailability_reason": None,
                "chars_extracted": 0, "quality_flag": None,
                "source": src.get("source_file"), "web_file": None,
                "jstor_url": None, "retrieval_instructions": None,
                "methods_attempted": [],
            })
            continue

        meta = paper_meta.get(i, {})
        source_title = (meta.get("title") or src.get("title") or "").strip()
        title = source_title or "Unknown"
        paper_id = meta.get("paper_id", "")
        oa_url = meta.get("oa_url")
        arxiv_id = meta.get("arxiv_id")
        doi = meta.get("doi") or src.get("doi")
        source_url = (src.get("url") or "").strip()
        # Fallback: extract arXiv ID from DOI if S2 didn't provide one
        if not arxiv_id and doi and doi.startswith("10.48550/arXiv."):
            arxiv_id = doi.split("arXiv.", 1)[1]

        status = "not_available"
        chars = 0
        quality_flag = None
        unavail_reason = None
        result_text = ""
        methods_attempted: list[dict] = []
        resolver_event = meta.get("_resolver_event")
        if resolver_event:
            methods_attempted.append(resolver_event.copy())
        claim_lost = False

        label = f"[{i+1}/{len(sources)}]"
        print(f"  {label} {title[:60]}", file=sys.stderr)

        # Claim the paper_id so parallel acquire_batch workers cannot
        # race on the same row. Empty paper_id has no row yet to claim;
        # in that case we skip the claim entirely (collision is vanishingly
        # rare for empty-pid rows since they resolve server-side inside
        # process_pdf on a per-PDF basis).
        claim_conn = None
        claimed_pid = None
        if paper_id and not args.dry_run:
            claim_conn = _init_db()
            try:
                if not _try_claim_paper(claim_conn, paper_id, _WORKER_TAG):
                    claim_conn.close()
                    claim_conn = None
                    print(
                        f"        claimed_by_other_worker: skipping {paper_id}",
                        file=sys.stderr,
                    )
                    entries.append({
                        "title": title, "type": "paper",
                        "paper_id": paper_id or None,
                        "doi": doi or None,
                        "url": source_url or None,
                        "status": "skipped_claimed_by_other",
                        "unavailability_reason": "claim_contention",
                        "chars_extracted": 0, "quality_flag": None,
                        "source": src.get("source_file"), "web_file": None,
                        "jstor_url": None, "retrieval_instructions": None,
                        "methods_attempted": [],
                    })
                    continue
                claimed_pid = paper_id
            except Exception as e:
                # Claim machinery itself failed (schema missing, DB locked, etc.).
                # Fall through without claim so the run doesn't silently stall;
                # a row-level overwrite is strictly less bad than zero downloads.
                print(f"        claim error ({e}); proceeding without lock", file=sys.stderr)
                if claim_conn is not None:
                    claim_conn.close()
                    claim_conn = None

        if not args.dry_run:
            # ── Try OA download ────────────────────────────────────────
            if oa_url:
                try:
                    async with httpx.AsyncClient(
                        timeout=60, follow_redirects=True,
                        headers={"User-Agent": BROWSER_UA},
                    ) as dl:
                        resp = await dl.get(oa_url)
                    if resp.status_code == 200 and resp.content[:5] == b"%PDF-":
                        tmp = staging / _staging_name("oa", paper_id, fallback="oa")
                        tmp.write_bytes(resp.content)
                        try:
                            result_text, chars_oa, ok = await _process_and_verify(
                                str(tmp), paper_id, target_doi=doi, target_title=title,
                                target_authors=_authors_for_verify(paper_id, meta.get("authors")),
                            )
                            if ok:
                                status = "downloaded"
                                chars = chars_oa
                                _record_method(
                                    methods_attempted, "oa", "success", result_text, first_line=True,
                                )
                                print(f"        OA → {chars:,} chars", file=sys.stderr)
                            else:
                                _record_method(methods_attempted, "oa", "failed", result_text)
                                print(f"        OA {result_text.splitlines()[0]}", file=sys.stderr)
                        finally:
                            # Clean up tmp on any exit — verify-fail or
                            # exception. On verify-pass tmp was moved to
                            # canonical so unlink is a no-op.
                            try:
                                tmp.unlink(missing_ok=True)
                            except Exception:
                                pass
                    else:
                        _record_method(
                            methods_attempted,
                            "oa",
                            f"http_{resp.status_code}" if resp.status_code != 200 else "not_pdf",
                            f"OA: HTTP {resp.status_code} or not PDF",
                        )
                        print(f"        OA: HTTP {resp.status_code} or not PDF", file=sys.stderr)
                except Exception as e:
                    _record_method(methods_attempted, "oa", "failed", f"OA error: {e}")
                    print(f"        OA error: {e}", file=sys.stderr)
            else:
                _record_method(methods_attempted, "oa", "skipped_no_url", "No OA URL")

            # Refresh the claim lease so multi-fallback acquisitions (OA +
            # arXiv + URL) that push past 1 hour don't lose their claim
            # mid-flight. If refresh returns False the claim has been
            # stolen or cleared — abandon the remaining fallbacks so we
            # don't write text under a row another worker has since
            # claimed.
            if claim_conn is not None and claimed_pid:
                try:
                    if not _refresh_paper_claim(claim_conn, claimed_pid, _WORKER_TAG):
                        claim_lost = True
                        print(
                            f"        claim expired/stolen mid-acquisition; "
                            f"stopping fallbacks for {claimed_pid}",
                            file=sys.stderr,
                        )
                except Exception:
                    pass

            # ── Try arXiv direct (S2 often misses arXiv OA flag) ──────
            if status == "downloaded":
                _record_method(
                    methods_attempted, "arxiv", "skipped_after_success", "Downloaded earlier in chain",
                )
            elif claim_lost:
                _record_method(methods_attempted, "arxiv", "skipped_claim_lost", "Claim lost")
            elif arxiv_id:
                arxiv_url = f"https://arxiv.org/pdf/{arxiv_id}"
                try:
                    async with httpx.AsyncClient(
                        timeout=60, follow_redirects=True,
                        headers={"User-Agent": BROWSER_UA},
                    ) as dl:
                        resp = await dl.get(arxiv_url)
                    if resp.status_code == 200 and resp.content[:5] == b"%PDF-":
                        tmp = staging / _staging_name("arxiv", paper_id, fallback="arxiv")
                        tmp.write_bytes(resp.content)
                        # arXiv IDs are unambiguous, but upstream metadata can
                        # still associate a wrong arxiv_id with the paper_id.
                        # Verify against title (DOI often absent on preprints).
                        try:
                            result_text, chars_ax, ok = await _process_and_verify(
                                str(tmp), paper_id, target_doi=doi, target_title=title,
                                target_authors=_authors_for_verify(paper_id, meta.get("authors")),
                            )
                            if ok:
                                status = "downloaded"
                                chars = chars_ax
                                _record_method(
                                    methods_attempted, "arxiv", "success", result_text, first_line=True,
                                )
                                print(f"        arXiv → {chars:,} chars", file=sys.stderr)
                            else:
                                _record_method(methods_attempted, "arxiv", "failed", result_text)
                                print(f"        arXiv {result_text.splitlines()[0]}", file=sys.stderr)
                        finally:
                            try:
                                tmp.unlink(missing_ok=True)
                            except Exception:
                                pass
                    else:
                        _record_method(
                            methods_attempted,
                            "arxiv",
                            f"http_{resp.status_code}" if resp.status_code != 200 else "not_pdf",
                            f"arXiv: HTTP {resp.status_code} or not PDF",
                        )
                        print(f"        arXiv: HTTP {resp.status_code} or not PDF", file=sys.stderr)
                except Exception as e:
                    _record_method(methods_attempted, "arxiv", "failed", f"arXiv error: {e}")
                    print(f"        arXiv error: {e}", file=sys.stderr)
            else:
                _record_method(methods_attempted, "arxiv", "skipped_no_arxiv", "No arXiv ID")

            if claim_conn is not None and claimed_pid and not claim_lost:
                try:
                    if not _refresh_paper_claim(claim_conn, claimed_pid, _WORKER_TAG):
                        claim_lost = True
                        print(
                            f"        claim expired/stolen; stopping fallbacks for {claimed_pid}",
                            file=sys.stderr,
                        )
                except Exception:
                    pass

            # ── Try direct URL (bepress, institutional repos, author sites) ─
            if status == "downloaded":
                _record_method(
                    methods_attempted, "url", "skipped_after_success", "Downloaded earlier in chain",
                )
            elif claim_lost:
                _record_method(methods_attempted, "url", "skipped_claim_lost", "Claim lost")
            elif source_url:
                try:
                    async with httpx.AsyncClient(
                        timeout=60, follow_redirects=True,
                        headers={"User-Agent": BROWSER_UA},
                    ) as dl:
                        resp = await dl.get(source_url)
                    if resp.status_code == 200 and resp.content[:5] == b"%PDF-":
                        tmp = staging / _staging_name("url", paper_id, fallback="url")
                        tmp.write_bytes(resp.content)
                        try:
                            result_text, chars_url, ok = await _process_and_verify(
                                str(tmp), paper_id, target_doi=doi, target_title=title,
                                target_authors=_authors_for_verify(paper_id, meta.get("authors")),
                            )
                            if ok:
                                status = "downloaded"
                                chars = chars_url
                                _record_method(
                                    methods_attempted, "url", "success", result_text, first_line=True,
                                )
                                print(f"        URL direct → {chars:,} chars", file=sys.stderr)
                            else:
                                _record_method(methods_attempted, "url", "failed", result_text)
                                print(f"        URL direct: {result_text.splitlines()[0]}", file=sys.stderr)
                        finally:
                            try:
                                tmp.unlink(missing_ok=True)
                            except Exception:
                                pass
                    elif resp.status_code == 200:
                        _record_method(
                            methods_attempted, "url", "not_pdf", "URL direct: response is not a PDF",
                        )
                    else:
                        _record_method(
                            methods_attempted,
                            "url",
                            f"http_{resp.status_code}",
                            f"URL direct: HTTP {resp.status_code}",
                        )
                        print(f"        URL direct: HTTP {resp.status_code}", file=sys.stderr)
                except Exception as e:
                    _record_method(methods_attempted, "url", "failed", f"URL direct error: {e}")
                    print(f"        URL direct error: {e}", file=sys.stderr)
            else:
                _record_method(methods_attempted, "url", "skipped_no_url", "No source URL")

            # ── arXiv TeX enhancement (after success) ──────────────────
            if status == "downloaded" and arxiv_id:
                try:
                    tex_r = await download_arxiv_tex(arxiv_id, paper_id=paper_id)
                    if tex_r.startswith("Downloaded arXiv TeX:"):
                        tc = re.search(r"(\d[\d,]*)\s*chars", tex_r)
                        print(f"        TeX → {tc.group(1) if tc else '?'} chars", file=sys.stderr)
                    elif tex_r.startswith("Already has TeX:"):
                        print(f"        TeX: already present", file=sys.stderr)
                    else:
                        print(f"        TeX: {tex_r.split(chr(10))[0][:60]}", file=sys.stderr)
                except Exception as e:
                    print(f"        TeX error: {e}", file=sys.stderr)

            # ── Handle METADATA_NEEDED ─────────────────────────────────
            orphan_id = _parse_orphan_id(result_text) if result_text else None
            if orphan_id:
                try:
                    async with httpx.AsyncClient(
                        headers={"User-Agent": BROWSER_UA}, timeout=30,
                    ) as mc:
                        await _handle_metadata_needed(mc, orphan_id, title, doi)
                    print(f"        fixed orphan {orphan_id}", file=sys.stderr)
                except Exception as e:
                    print(f"        orphan fix error: {e}", file=sys.stderr)

            # ── Flag ABSTRACT_NEEDED for LLM ───────────────────────────
            if result_text and "ABSTRACT_NEEDED" in result_text:
                pid = paper_id or orphan_id or f"unknown_{i}"
                needs_abstract.append({"paper_id": pid, "title": title})

        # ── Classify failures for LLM fallback ────────────────────────
        if status != "downloaded" and not args.dry_run:
            unavail_reason = _classify_unavailability(doi, paper_id, source_title, methods_attempted)
            # Include resolved OA URL if the source URL is empty
            resolved_url = source_url or meta.get("oa_url")
            needs_web_search.append({
                "title": title,
                "paper_id": paper_id or "",
                "s2_id": src.get("s2_id"),
                "doi": doi,
                "url": resolved_url,
                "type": "paper",
                "authority_tier": src.get("authority_tier"),
                "source_file": src.get("source_file"),
                "reason": result_text.split("\n")[0][:200] if result_text else "no_oa_pdf",
                "unavail_reason": unavail_reason,
                "methods_attempted": [attempt.copy() for attempt in methods_attempted],
            })

        if status == "downloaded":
            n_downloaded += 1

        # Release the paper claim so other workers can proceed. Uses a fresh
        # try/except so a release-time error never blocks entries.append.
        if claim_conn is not None and claimed_pid:
            try:
                _release_paper_claim(claim_conn, claimed_pid, _WORKER_TAG)
            except Exception as e:
                print(f"        claim release error ({e})", file=sys.stderr)
            finally:
                try:
                    claim_conn.close()
                except Exception:
                    pass
                claim_conn = None
                claimed_pid = None

        entries.append({
            "title": title,
            "type": "paper",
            "paper_id": paper_id or None,
            "doi": doi or None,
            "url": source_url or None,
            "status": status if not args.dry_run else "dry_run",
            "unavailability_reason": unavail_reason,
            "chars_extracted": chars,
            "quality_flag": quality_flag,
            "source": src.get("source_file"),
            "web_file": None,
            "jstor_url": None,
            "retrieval_instructions": None,
            "methods_attempted": [attempt.copy() for attempt in methods_attempted],
        })

        # Incremental write: after first result, then every 5
        if len(entries) == 1 or len(entries) % 5 == 0:
            _write_partial(project_dir, chunk_id, entries)

    # ── Final output ───────────────────────────────────────────────────────
    _write_partial(project_dir, chunk_id, entries)
    _write_needs_llm(project_dir, chunk_id, needs_web_search, needs_abstract)

    n_failed = len(needs_web_search)
    print(
        f"\n[done] chunk={chunk_id}  downloaded={n_downloaded}  "
        f"needs_llm={n_failed}  needs_abstract={len(needs_abstract)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    asyncio.run(main())
