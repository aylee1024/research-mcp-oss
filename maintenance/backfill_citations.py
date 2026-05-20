# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx",
# ]
# ///
"""One-time citation-graph backfill for ~17k papers in papers.db.

Three phases run in sequence (each resumable via backfill_citations_progress.json):

  A. OpenAlex backfill — for `oa:*` papers (12,955). Direct W-ID lookup via
     OpenAlex /works/{id}?select=id,referenced_works. Free, 8 RPS throttle.

  B. S2 batch backfill — for native S2 SHA-1 paper IDs with DOIs (1,395).
     POST /paper/batch with fields=references.externalIds. 1 RPS with API key.

  C. local: title-match + reference fetch — for local: papers (1,610). Each one
     gets run through S2 /paper/search/match to find a canonical S2 ID. Matches
     receive a follow-up reference fetch via the same batch endpoint.

Runs as a separate process against the live papers.db in WAL mode, so the MCP
server stays responsive. Short commits (50 papers per transaction). Conservative
throttles leave headroom for live MCP traffic.

Usage:
    uv run backfill_citations.py [--phase A|B|C|all] [--dry-run] [--limit N]
"""

import argparse
import asyncio
import json
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_mcp.paths import PAPERS_DB_PATH as DB_PATH, RESEARCH_MCP_HOME  # noqa: E402

PROGRESS_PATH = RESEARCH_MCP_HOME / "backfill_citations_progress.json"

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_API_KEY = os.environ.get("S2_API_KEY", "")
OA_BASE = "https://api.openalex.org"
OA_MAILTO = os.environ.get("OPENALEX_MAILTO", "")

# Conservative throttles — leave headroom for live MCP traffic.
# S2 with API key = 1 RPS. We use 0.5 RPS for safety (2 second minimum interval).
# OpenAlex documented max = 10 RPS. We use 8 RPS.
S2_INTERVAL = 2.0
OA_INTERVAL = 0.125

# Short commit interval to avoid locking the DB for too long
COMMIT_EVERY = 50

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[signal] Shutdown requested — will commit current batch and exit", file=sys.stderr)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


class Throttle:
    """Simple async rate limiter."""
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last = time.monotonic()


def _load_progress() -> dict:
    if PROGRESS_PATH.exists():
        try:
            return json.loads(PROGRESS_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {"openalex_last": "", "s2_last": "", "local_last": ""}


def _save_progress(progress: dict) -> None:
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2))


def _init_db() -> sqlite3.Connection:
    """Open a connection to papers.db. Expects schema v6 to be in place
    (created by the running MCP server on its next startup)."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    # Verify paper_references table exists
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='paper_references'"
    ).fetchone()
    if not row:
        raise RuntimeError(
            "paper_references table missing. Start the MCP server once to run the "
            "schema v6 migration, then retry the backfill."
        )
    return conn


def _store_references(
    conn: sqlite3.Connection,
    citing_paper_id: str,
    references: list[dict],
    source: str,
) -> int:
    """Mirror of server.py::_store_references. Inserts edges where the cited
    paper resolves to a local paper_id. Papers not in the library are stored
    with placeholder cited_paper_id prefixed __doi:{doi}.
    """
    if not references:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    for ref in references:
        cited_pid: str | None = None
        cited_doi = (ref.get("doi") or "").strip()
        s2_id = (ref.get("paperId") or "").strip()
        oa_id = (ref.get("openalex_id") or "").strip()

        if s2_id:
            row = conn.execute(
                "SELECT paper_id FROM papers WHERE paper_id = ?", (s2_id,)
            ).fetchone()
            if row:
                cited_pid = row[0]
        if not cited_pid and oa_id:
            short = oa_id.split("/")[-1]
            row = conn.execute(
                "SELECT paper_id FROM papers WHERE paper_id = ?", (f"oa:{short}",)
            ).fetchone()
            if row:
                cited_pid = row[0]
        if not cited_pid and cited_doi:
            row = conn.execute(
                "SELECT paper_id FROM papers WHERE lower(doi) = lower(?)", (cited_doi,)
            ).fetchone()
            if row:
                cited_pid = row[0]

        if not cited_pid and not cited_doi:
            continue

        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO paper_references
                (citing_paper_id, cited_paper_id, cited_doi, source, is_influential, first_seen)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    citing_paper_id,
                    cited_pid or f"__doi:{cited_doi}",
                    cited_doi or None,
                    source,
                    1 if ref.get("is_influential") else 0,
                    now,
                ),
            )
            count += 1
        except sqlite3.Error:
            pass
    return count


# -----------------------------------------------------------------------------
# Phase A: OpenAlex backfill
# -----------------------------------------------------------------------------

async def backfill_openalex(conn: sqlite3.Connection, progress: dict, dry_run: bool, limit: int | None) -> dict:
    """Fetch referenced_works for each oa:* paper via OpenAlex single-entity GET."""
    start = progress.get("openalex_last", "")
    query = """
        SELECT paper_id, openalex_id
        FROM papers
        WHERE paper_id LIKE 'oa:%'
          AND paper_id > ?
        ORDER BY paper_id
    """
    if limit:
        query += f" LIMIT {int(limit)}"
    oa_papers = conn.execute(query, (start,)).fetchall()

    total = len(oa_papers)
    print(f"[Phase A] OpenAlex backfill: {total} papers to process", file=sys.stderr)
    if total == 0:
        return {"processed": 0, "edges": 0, "failed": 0}

    throttle = Throttle(OA_INTERVAL)
    edges_added = 0
    failed = 0
    processed = 0
    last_processed_pid = start  # track the actual last paper we touched

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, (pid, oa_id) in enumerate(oa_papers):
            if _shutdown_requested:
                break
            await throttle.wait()

            # Derive W ID — prefer openalex_id column, fallback to stripping 'oa:' prefix
            w_id = ""
            if oa_id:
                w_id = oa_id.split("/")[-1]
            if not w_id:
                w_id = pid.split(":", 1)[1] if ":" in pid else ""
            if not w_id or not w_id.startswith("W"):
                failed += 1
                last_processed_pid = pid
                continue

            params = {"select": "id,referenced_works"}
            if OA_MAILTO:
                params["mailto"] = OA_MAILTO

            try:
                resp = await client.get(
                    f"{OA_BASE}/works/{w_id}",
                    params=params,
                )
                if resp.status_code == 404:
                    processed += 1
                    last_processed_pid = pid
                    continue
                resp.raise_for_status()
                data = resp.json()
                ref_works = data.get("referenced_works") or []
                if ref_works and not dry_run:
                    refs = [{"openalex_id": rw} for rw in ref_works]
                    edges_added += _store_references(conn, pid, refs, source="openalex_backfill")
                processed += 1
                last_processed_pid = pid
            except httpx.HTTPError as e:
                print(f"  [OA fail] {pid}: {e}", file=sys.stderr)
                failed += 1
                last_processed_pid = pid
            except Exception as e:
                print(f"  [OA error] {pid}: {e}", file=sys.stderr)
                failed += 1
                last_processed_pid = pid

            if (i + 1) % COMMIT_EVERY == 0:
                if not dry_run:
                    conn.commit()
                progress["openalex_last"] = last_processed_pid
                _save_progress(progress)
                if (i + 1) % 500 == 0:
                    print(
                        f"  [Phase A] {i + 1}/{total} papers, {edges_added} edges added, {failed} failed",
                        file=sys.stderr,
                    )

    if not dry_run:
        conn.commit()
    progress["openalex_last"] = last_processed_pid
    _save_progress(progress)
    print(f"[Phase A] Done: {processed}/{total} processed, {edges_added} edges, {failed} failed", file=sys.stderr)
    return {"processed": processed, "edges": edges_added, "failed": failed}


# -----------------------------------------------------------------------------
# Phase B: S2 batch backfill
# -----------------------------------------------------------------------------

async def _s2_batch_with_retry(
    client: httpx.AsyncClient,
    ids: list[str],
    headers: dict,
    throttle: Throttle,
    max_retries: int = 5,
) -> list | None:
    """POST /paper/batch with exponential backoff on 429/5xx/network errors.
    Returns the parsed JSON results list, or None if all retries exhausted.
    """
    delay = 2.0
    for attempt in range(max_retries):
        if _shutdown_requested:
            return None
        await throttle.wait()
        try:
            resp = await client.post(
                f"{S2_BASE}/paper/batch",
                params={"fields": "paperId,references.paperId,references.externalIds"},
                headers=headers,
                json={"ids": ids},
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                print(
                    f"    [retry] S2 batch HTTP {resp.status_code} (attempt {attempt + 1}/{max_retries}), "
                    f"sleeping {delay:.1f}s",
                    file=sys.stderr,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            print(
                f"    [retry] S2 batch network error (attempt {attempt + 1}/{max_retries}): {e}",
                file=sys.stderr,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)
    return None


async def backfill_s2_batch(conn: sqlite3.Connection, progress: dict, dry_run: bool, limit: int | None) -> dict:
    """Fetch references for native-S2 papers via POST /paper/batch.

    Safe-by-default semantics: progress is only advanced for successfully-processed
    batches. Failed batches (after retries) are recorded in a failed_batches list
    and retried at the end of the phase. If a batch still fails after the final
    retry pass, those paper_ids remain unprocessed — the next run will pick them
    up from the same progress checkpoint.
    """
    start = progress.get("s2_last", "")
    query = """
        SELECT paper_id, doi
        FROM papers
        WHERE paper_id NOT LIKE 'oa:%'
          AND paper_id NOT LIKE 'local:%'
          AND paper_id NOT LIKE '__doi:%'
          AND paper_id > ?
        ORDER BY paper_id
    """
    if limit:
        query += f" LIMIT {int(limit)}"
    papers = conn.execute(query, (start,)).fetchall()

    total = len(papers)
    print(f"[Phase B] S2 batch backfill: {total} papers to process", file=sys.stderr)
    if total == 0:
        return {"processed": 0, "edges": 0, "failed": 0}

    throttle = Throttle(S2_INTERVAL)
    BATCH_SIZE = 500
    edges_added = 0
    failed = 0
    processed = 0
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    failed_batches: list[list] = []  # batches to retry at end

    def process_batch(batch_papers, results_json) -> tuple[int, int]:
        """Process a batch result and store edges. Returns (processed, edges)."""
        local_proc = 0
        local_edges = 0
        ids_local = [p[0] for p in batch_papers]
        for source_pid, result in zip(ids_local, results_json):
            if result is None:
                local_proc += 1
                continue
            refs = result.get("references") or []
            refs_for_store = [
                {
                    "paperId": r.get("paperId", "") or "",
                    "doi": (r.get("externalIds") or {}).get("DOI", "") or "",
                }
                for r in refs
            ]
            if refs_for_store and not dry_run:
                local_edges += _store_references(
                    conn, source_pid, refs_for_store, source="s2_batch_backfill"
                )
            local_proc += 1
        return local_proc, local_edges

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Track the highest contiguous-successful batch end for progress updates.
        # We can only advance progress past a batch if all preceding batches also
        # succeeded, otherwise resuming will skip the gap.
        contiguous_success_end = start
        has_failure_below = False

        for batch_start in range(0, total, BATCH_SIZE):
            if _shutdown_requested:
                break
            batch = papers[batch_start:batch_start + BATCH_SIZE]
            ids = [p[0] for p in batch]

            results_json = await _s2_batch_with_retry(client, ids, headers, throttle)
            if results_json is None:
                print(
                    f"  [Phase B] batch starting at {batch_start}: FAILED after retries, "
                    f"queued for end-of-phase retry",
                    file=sys.stderr,
                )
                failed_batches.append(batch)
                has_failure_below = True
                continue

            try:
                b_proc, b_edges = process_batch(batch, results_json)
                processed += b_proc
                edges_added += b_edges
                if not dry_run:
                    conn.commit()
                # Only advance progress if there have been no failures at or
                # before this batch — otherwise the resume would skip the gap.
                if not has_failure_below:
                    contiguous_success_end = batch[-1][0]
                    progress["s2_last"] = contiguous_success_end
                    _save_progress(progress)
                print(
                    f"  [Phase B] batch {batch_start // BATCH_SIZE + 1}/{(total + BATCH_SIZE - 1) // BATCH_SIZE}, "
                    f"{processed} processed, {edges_added} edges",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"  [S2 process fail] batch starting at {batch_start}: {e}", file=sys.stderr)
                failed_batches.append(batch)
                has_failure_below = True

        # End-of-phase retry pass for any failed batches. One more shot each.
        if failed_batches and not _shutdown_requested:
            print(
                f"[Phase B] Retry pass: {len(failed_batches)} failed batch(es)",
                file=sys.stderr,
            )
            still_failed = []
            for batch in failed_batches:
                if _shutdown_requested:
                    still_failed.append(batch)
                    continue
                ids = [p[0] for p in batch]
                results_json = await _s2_batch_with_retry(client, ids, headers, throttle)
                if results_json is None:
                    still_failed.append(batch)
                    failed += len(batch)
                    continue
                try:
                    b_proc, b_edges = process_batch(batch, results_json)
                    processed += b_proc
                    edges_added += b_edges
                    if not dry_run:
                        conn.commit()
                except Exception as e:
                    print(f"  [retry process fail]: {e}", file=sys.stderr)
                    still_failed.append(batch)
                    failed += len(batch)

            if still_failed:
                # Save their IDs so the user can retry manually.
                unrecovered_ids = [p[0] for batch in still_failed for p in batch]
                print(
                    f"[Phase B] WARNING: {len(unrecovered_ids)} papers unrecovered after retry pass. "
                    f"Their progress checkpoint is NOT advanced; next run will retry.",
                    file=sys.stderr,
                )

    if not dry_run:
        conn.commit()
    print(f"[Phase B] Done: {processed}/{total} processed, {edges_added} edges, {failed} failed", file=sys.stderr)
    return {"processed": processed, "edges": edges_added, "failed": failed}


# -----------------------------------------------------------------------------
# Phase C: local: title-match + reference fetch
# -----------------------------------------------------------------------------

async def backfill_local(conn: sqlite3.Connection, progress: dict, dry_run: bool, limit: int | None) -> dict:
    """Try to upgrade local:* papers via S2 title-match, then fetch their
    references. Papers with no match stay edge-less."""
    start = progress.get("local_last", "")
    query = """
        SELECT paper_id, title
        FROM papers
        WHERE paper_id LIKE 'local:%'
          AND title != ''
          AND title IS NOT NULL
          AND paper_id > ?
        ORDER BY paper_id
    """
    if limit:
        query += f" LIMIT {int(limit)}"
    local_papers = conn.execute(query, (start,)).fetchall()

    total = len(local_papers)
    print(f"[Phase C] local: title-match + ref fetch: {total} papers", file=sys.stderr)
    if total == 0:
        return {"matched": 0, "processed": 0, "edges": 0}

    # S2 paper/search/match also falls under the 1 RPS with-key throttle
    throttle = Throttle(S2_INTERVAL)
    matched = 0
    edges_added = 0
    processed = 0
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    last_processed_pid = start

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, (pid, title) in enumerate(local_papers):
            if _shutdown_requested:
                break
            await throttle.wait()

            try:
                # Step 1: title-match to find a canonical S2 record
                resp = await client.get(
                    f"{S2_BASE}/paper/search/match",
                    params={"query": title, "fields": "paperId,title,references.paperId,references.externalIds"},
                    headers=headers,
                )
                if resp.status_code == 404:
                    processed += 1
                    last_processed_pid = pid
                    continue
                resp.raise_for_status()
                data = resp.json()
                matches = data.get("data", [])
                if not matches:
                    processed += 1
                    last_processed_pid = pid
                    continue
                match = matches[0]
                raw_score = data.get("matchScore")
                if raw_score is None:
                    raw_score = match.get("matchScore", 0)
                # Coerce score to float; fail-closed on unparseable values
                try:
                    match_score = float(raw_score) if raw_score is not None else 0.0
                except (TypeError, ValueError):
                    match_score = 0.0
                if match_score < 0.85:
                    processed += 1
                    last_processed_pid = pid
                    continue
                matched += 1

                refs = match.get("references") or []
                refs_for_store = [
                    {
                        "paperId": r.get("paperId", "") or "",
                        "doi": (r.get("externalIds") or {}).get("DOI", "") or "",
                    }
                    for r in refs
                ]
                # Store edges under the LOCAL paper_id (citing), so when search_local
                # returns this local: paper, its hub boost kicks in against whatever
                # cited papers resolve to library entries.
                if refs_for_store and not dry_run:
                    edges_added += _store_references(
                        conn, pid, refs_for_store, source="s2_title_match_backfill"
                    )
                processed += 1
                last_processed_pid = pid
            except httpx.HTTPError as e:
                print(f"  [local fail] {pid}: {e}", file=sys.stderr)
                processed += 1
                last_processed_pid = pid
            except Exception as e:
                print(f"  [local error] {pid}: {e}", file=sys.stderr)
                processed += 1
                last_processed_pid = pid

            if (i + 1) % COMMIT_EVERY == 0:
                if not dry_run:
                    conn.commit()
                progress["local_last"] = last_processed_pid
                _save_progress(progress)
                if (i + 1) % 200 == 0:
                    print(
                        f"  [Phase C] {i + 1}/{total}: {matched} matched, {edges_added} edges",
                        file=sys.stderr,
                    )

    if not dry_run:
        conn.commit()
    progress["local_last"] = last_processed_pid
    _save_progress(progress)
    print(
        f"[Phase C] Done: {matched}/{total} matched, {edges_added} edges added",
        file=sys.stderr,
    )
    return {"matched": matched, "processed": processed, "edges": edges_added}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Citation-graph backfill for papers.db")
    parser.add_argument("--phase", choices=["A", "B", "C", "all"], default="all",
                        help="Which phase(s) to run. Default: all")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch references but do not write to DB")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to process per phase (for testing)")
    parser.add_argument("--reset", action="store_true",
                        help="Ignore progress file and start from scratch")
    args = parser.parse_args()

    progress = {} if args.reset else _load_progress()
    if args.reset:
        print("[reset] Starting from scratch", file=sys.stderr)

    if not S2_API_KEY and args.phase in ("B", "C", "all"):
        print(
            "[warn] No S2_API_KEY in env. Phases B and C will fall back to the "
            "shared unauthenticated pool (5000 req / 5 min globally).",
            file=sys.stderr,
        )

    conn = _init_db()
    try:
        start_time = time.monotonic()
        totals = {"phase_a": None, "phase_b": None, "phase_c": None}

        if args.phase in ("A", "all") and not _shutdown_requested:
            totals["phase_a"] = await backfill_openalex(conn, progress, args.dry_run, args.limit)
        if args.phase in ("B", "all") and not _shutdown_requested:
            totals["phase_b"] = await backfill_s2_batch(conn, progress, args.dry_run, args.limit)
        if args.phase in ("C", "all") and not _shutdown_requested:
            totals["phase_c"] = await backfill_local(conn, progress, args.dry_run, args.limit)

        elapsed = time.monotonic() - start_time
        print(
            f"\n[done] Backfill complete in {elapsed:.1f}s",
            file=sys.stderr,
        )
        for phase, stats in totals.items():
            if stats:
                print(f"  {phase}: {stats}", file=sys.stderr)

        final_count = conn.execute("SELECT COUNT(*) FROM paper_references").fetchone()[0]
        print(f"[done] Total edges in paper_references: {final_count:,}", file=sys.stderr)
    finally:
        conn.close()


if __name__ == "__main__":
    asyncio.run(main())
