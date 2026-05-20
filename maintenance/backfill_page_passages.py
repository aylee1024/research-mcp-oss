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
"""Backfill `paper_passages` from existing processed_text for cite-ready papers.

Schema v18 adds paper_passages (one row per page) but only auto-populates it
for NEW ingestions via _store_processed_text. This script backfills the
2,889-ish rows already in the DB whose processed_text has Docling page
markers.

Safe to run multiple times. For each paper, existing paper_passages rows
are deleted and rebuilt from the current processed_text.

Usage:
    uv run backfill_page_passages.py              # dry-run (report counts)
    uv run backfill_page_passages.py --apply      # execute
"""

import argparse
import signal
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from server import _init_db, _build_paper_passages, DB_PATH

_shutdown = False


def _on_signal(sig, frame):
    global _shutdown
    _shutdown = True
    print("\n[signal] Finishing current paper...", file=sys.stderr)


signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Execute (default: dry-run)")
    args = ap.parse_args()

    print(f"[start] backfill_page_passages  apply={args.apply}", file=sys.stderr)
    conn = _init_db()

    # Candidates: every paper with has_full_text=1 and processed_text present.
    # has_full_text can be 1 via tex_text alone; tex_text has no page markers
    # so we require processed_text>500 explicitly here.
    candidates = conn.execute("""
        SELECT paper_id, LENGTH(processed_text)
        FROM papers
        WHERE has_full_text = 1
          AND processed_text IS NOT NULL
          AND LENGTH(processed_text) > 500
    """).fetchall()
    print(f"[found] {len(candidates)} candidate papers with processed_text", file=sys.stderr)

    existing = conn.execute(
        "SELECT COUNT(DISTINCT paper_id) FROM paper_passages"
    ).fetchone()[0]
    print(f"[found] {existing} papers already have paper_passages rows", file=sys.stderr)

    if not args.apply:
        print("\n[dry-run] pass --apply to execute. Sample of first 3:", file=sys.stderr)
        for pid, txt_len in candidates[:3]:
            title = conn.execute("SELECT title FROM papers WHERE paper_id = ?", (pid,)).fetchone()[0]
            print(f"  {pid}: '{title[:60]}' ({txt_len:,} chars processed_text)", file=sys.stderr)
        conn.close()
        return 0

    done = 0
    total_passages = 0
    for pid, _ in candidates:
        if _shutdown:
            break
        text = conn.execute(
            "SELECT processed_text FROM papers WHERE paper_id = ?", (pid,)
        ).fetchone()[0]
        n = _build_paper_passages(conn, pid, text)
        total_passages += n
        done += 1
        if done % 100 == 0:
            conn.commit()
            print(f"  progress: {done}/{len(candidates)} papers, {total_passages:,} passages", file=sys.stderr)
    conn.commit()

    # Report
    by_page = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT paper_id) FROM paper_passages"
    ).fetchone()
    print(f"\n[done] processed {done} papers, built {total_passages:,} passages", file=sys.stderr)
    print(f"[verify] paper_passages now holds {by_page[0]:,} rows across {by_page[1]:,} papers", file=sys.stderr)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
