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
#     "pymupdf",
# ]
# ///
"""Re-Docling cleared papers from their source PDFs.

Strategy:
  Phase 1: for every cleared local: paper with a valid local_pdf_path, re-run Docling
           via server.process_pdf. This is ground truth — text matches the PDF.
  Phase 2: for cleared papers with NULL local_pdf_path, try to find a matching PDF in
           $PAPERS_DIR by paper_id stem and DOI. If found, link and re-Docling.

Usage:
    uv run redocling_cleared.py              # dry-run (report what would be done)
    uv run redocling_cleared.py --apply      # execute
"""

import argparse
import asyncio
import re
import signal
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import (
    _init_db,
    process_pdf,
    PAPERS_DIR,
)


def _default_search_roots() -> list[Path]:
    """Default directories to scan for orphan PDFs when local_pdf_path is NULL.

    Single root by default: $PAPERS_DIR. Pass additional roots via the
    `--root` CLI flag if PDFs are scattered across other locations
    (e.g., an external archive).
    """
    return [PAPERS_DIR]


SEARCH_ROOTS = _default_search_roots()

_shutdown = False


def _on_signal(sig, frame):
    global _shutdown
    _shutdown = True
    print("\n[signal] Finishing current paper...", file=sys.stderr)


signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


_DOI_IN_ID_PAT = re.compile(r"10[._\-/](\d{4,5})[._\-/](.+)$")


def _doi_from_paper_id(pid: str) -> str | None:
    """Extract DOI substring embedded in a local: paper_id, if any."""
    stem = pid[len("local:"):] if pid.startswith("local:") else pid
    m = _DOI_IN_ID_PAT.search(stem)
    if m:
        doi = f"10.{m.group(1)}/{m.group(2)}"
        return re.sub(r"[/._\-]+$", "", doi)
    return None


def _stem_from_paper_id(pid: str) -> str:
    """Return the natural-name portion of a local: paper_id (drop leading local: and
    the trailing DOI block after `__`)."""
    s = pid[len("local:"):] if pid.startswith("local:") else pid
    return s.split("__", 1)[0]


def _find_pdf_for_paper(pid: str, pdf_index: dict[str, list[Path]]) -> Path | None:
    """Try to locate a PDF in SEARCH_ROOTS for a paper_id with no local_pdf_path.

    Match strategies (in order):
      1. Full-stem filename match (no truncation)
      2. DOI substring match with collision check (return None on ambiguous match)
    """
    stem = _stem_from_paper_id(pid).lower()
    stem_normalized = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")

    # Strategy 1: exact full-stem match (no truncation, no first-wins collision)
    hits = pdf_index.get(stem_normalized, [])
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return None  # ambiguous collision — refuse to guess

    # Strategy 2: DOI substring match. Require exactly one match.
    doi = _doi_from_paper_id(pid)
    if doi:
        doi_tokens = re.sub(r"[^a-z0-9]+", "_", doi.lower()).strip("_")
        matches = [paths[0] for fname_key, paths in pdf_index.items()
                   if doi_tokens and doi_tokens in fname_key and len(paths) == 1]
        if len(matches) == 1:
            return matches[0]
    return None


def _build_pdf_index() -> dict[str, list[Path]]:
    """Walk all SEARCH_ROOTS and build a full-normalized-filename → [Path] index.
    Stores a list (not single Path) so collisions are visible and can be refused."""
    index: dict[str, list[Path]] = {}
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for pdf in root.rglob("*.pdf"):
            fname_normalized = re.sub(r"[^a-z0-9]+", "_", pdf.stem.lower()).strip("_")
            if fname_normalized:
                index.setdefault(fname_normalized, []).append(pdf)
    return index


async def _reprocess(paper_id: str, pdf_path: Path) -> tuple[bool, str]:
    """Run process_pdf on a path with the in-transaction verify gate.

    Routes through process_pdf(verify_doi=..., verify_title=...) so the verify
    happens INSIDE the same transaction as the store. A verify failure rolls
    the store back atomically, which is stronger than the previous "commit,
    then verify on a second connection, then clear" pattern that left a
    window where wrong text could be durably visible.
    """
    # Look up target metadata to use as the verify targets.
    meta_conn = _init_db()
    try:
        meta = meta_conn.execute(
            "SELECT title, doi, authors FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
    finally:
        meta_conn.close()
    if not meta:
        return False, "no row for paper_id"
    title, doi, authors = meta

    try:
        # move_to_canonical=False: we're scavenging the user's research
        # tree for a PDF matching a cleared row. Pass the file path
        # through to process_pdf but do NOT relocate the file from its
        # original location. Earlier iterations moved user-owned PDFs
        # into Papers/, which was an unintended destructive side effect.
        result = await process_pdf(
            str(pdf_path),
            paper_id=paper_id,
            verify_doi=(doi or None),
            verify_title=(title or None),
            verify_authors=(authors or None),
            move_to_canonical=False,
        )
    except Exception as e:
        return False, f"exception: {e}"

    if result.startswith("VERIFY_REJECTED"):
        # process_pdf rolled back atomically; the store never committed so
        # there is no need to issue a separate clear.
        return False, result[:200]
    if not result.startswith("Processed:"):
        return False, result[:80]

    m = re.search(r"Extracted:\s*([\d,]+)\s*chars", result)
    chars = m.group(1) if m else "?"
    return True, f"{chars} chars (verified inline)"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument(
        "--root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Additional directory to search for orphan PDFs (may be passed "
            "multiple times). Defaults to $PAPERS_DIR only."
        ),
    )
    args = ap.parse_args()

    if args.root:
        global SEARCH_ROOTS
        # Preserve order, dedupe with a set check
        seen = set()
        roots: list[Path] = []
        for r in (*_default_search_roots(), *[p.expanduser() for p in args.root]):
            key = str(r.resolve())
            if key in seen:
                continue
            seen.add(key)
            roots.append(r)
        SEARCH_ROOTS = roots

    print(f"[start] redocling_cleared apply={args.apply}", file=sys.stderr)
    conn = _init_db()

    # Phase 1: papers with existing local_pdf_path and no usable text.
    # "No usable text" = neither processed_text nor tex_text exceeds 500
    # chars. Previously only processed_text was checked, which treated
    # healthy TeX-only papers as needing re-Docling.
    phase1 = conn.execute("""
        SELECT paper_id, local_pdf_path FROM papers
        WHERE paper_id LIKE 'local:%'
        AND LENGTH(COALESCE(processed_text, '')) <= 500
        AND LENGTH(COALESCE(tex_text, '')) <= 500
        AND local_pdf_path IS NOT NULL
    """).fetchall()
    phase1 = [(pid, path) for pid, path in phase1 if Path(path).is_file()]
    print(f"[phase 1] {len(phase1)} cleared papers with valid PDF path", file=sys.stderr)

    # Phase 2: papers without local_pdf_path AND no usable text → search
    # Research tree. Same "no usable text" definition as Phase 1.
    phase2_ids = [r[0] for r in conn.execute("""
        SELECT paper_id FROM papers
        WHERE paper_id LIKE 'local:%'
        AND LENGTH(COALESCE(processed_text, '')) <= 500
        AND LENGTH(COALESCE(tex_text, '')) <= 500
        AND local_pdf_path IS NULL
    """).fetchall()]

    if phase2_ids:
        print(f"[phase 2] building PDF index for {len(SEARCH_ROOTS)} roots...", file=sys.stderr)
        pdf_index = _build_pdf_index()
        print(f"          indexed {len(pdf_index)} PDFs", file=sys.stderr)
    else:
        pdf_index = {}

    phase2_matches: list[tuple[str, Path]] = []
    phase2_unmatched: list[str] = []
    for pid in phase2_ids:
        found = _find_pdf_for_paper(pid, pdf_index)
        if found:
            phase2_matches.append((pid, found))
        else:
            phase2_unmatched.append(pid)
    print(f"[phase 2] matched {len(phase2_matches)}, unmatched {len(phase2_unmatched)}", file=sys.stderr)

    if not args.apply:
        print("\n-- Phase 1 samples --", file=sys.stderr)
        for pid, path in phase1[:5]:
            print(f"  {pid}: {path}", file=sys.stderr)
        print("\n-- Phase 2 samples --", file=sys.stderr)
        for pid, path in phase2_matches[:5]:
            print(f"  {pid}\n    -> {path}", file=sys.stderr)
        print("\n-- Phase 2 unmatched samples --", file=sys.stderr)
        for pid in phase2_unmatched[:10]:
            print(f"  {pid}", file=sys.stderr)
        conn.close()
        return

    conn.close()  # process_pdf manages its own connection

    # Execute
    succeeded = 0
    failed = 0
    for i, (pid, pdf_path) in enumerate(phase1):
        if _shutdown:
            break
        ok, msg = await _reprocess(pid, Path(pdf_path))
        status = "OK" if ok else "FAIL"
        print(f"  [phase1 {i+1}/{len(phase1)}] {status} {pid}: {msg}", file=sys.stderr)
        if ok:
            succeeded += 1
        else:
            failed += 1

    for i, (pid, pdf_path) in enumerate(phase2_matches):
        if _shutdown:
            break
        # Do NOT persist local_pdf_path BEFORE verify. _sync_store_pdf_text
        # will write it as part of the atomic store-verify-commit flow and
        # roll it back on verify-fail. Persisting pre-verify left a wrong
        # PDF path linked to the row whenever _find_pdf_for_paper matched
        # a DOI-substring false positive and verify rejected it.
        ok, msg = await _reprocess(pid, pdf_path)
        status = "OK" if ok else "FAIL"
        print(f"  [phase2 {i+1}/{len(phase2_matches)}] {status} {pid}: {msg}", file=sys.stderr)
        if ok:
            succeeded += 1
        else:
            failed += 1

    print(f"\n[done] succeeded={succeeded} failed={failed} unmatched={len(phase2_unmatched)}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
