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
"""Data integrity repair sweep for papers.db.

Operates on papers.db in place. Run after any large re-Docling sweep or
schema upgrade to detect and repair common drift modes.

Phases:
  1. Schema migration runs automatically via _init_db at startup.
  2. Orphan-chunk embedding backfill: chunks without a vec_chunks row.
  3. Dedup-merged re-embed: re-embed papers whose paper-level embedding
     may be stale after dedup_papers.py merged duplicates.
  4. Optional: re-run dedup_papers.py to catch any remaining exact-text
     duplicate groups.

Usage:
    uv run maintenance/repair_db_v2.py             # dry-run (report counts)
    uv run repair_db_v2.py --apply             # execute all three steps
    uv run repair_db_v2.py --apply --only 2    # execute just step 2
"""

import argparse
import signal
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from server import (
    _init_db,
    _embed_texts,
    _paper_embed_text,
    DB_PATH,
)

_shutdown = False


def _on_signal(sig, frame):
    global _shutdown
    _shutdown = True
    print("\n[signal] Stopping after current batch...", file=sys.stderr)


signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


def step_schema_v8(apply: bool) -> None:
    """Step 1: schema v8 migration is automatic via _init_db. Verify + report."""
    conn = _init_db()  # runs migration if needed
    row = conn.execute(
        "SELECT has_full_text, COUNT(*) FROM papers GROUP BY has_full_text"
    ).fetchall()
    print("[step 1] schema v8 counts:", file=sys.stderr)
    for flag, count in row:
        print(f"         has_full_text={flag}: {count:,}", file=sys.stderr)
    conn.close()


def step_backfill_orphan_chunks(apply: bool, batch_size: int = 500) -> None:
    """Step 2: re-embed chunks that exist in paper_chunks but not in vec_chunks."""
    conn = _init_db()
    # Chunks where no vec_chunks row has rowid = chunk_id.
    todo = conn.execute("""
        SELECT c.chunk_id, c.chunk_text
        FROM paper_chunks c
        JOIN papers p ON c.paper_id = p.paper_id
        WHERE p.verified = 1
        AND c.chunk_id NOT IN (SELECT rowid FROM vec_chunks)
    """).fetchall()
    print(f"[step 2] {len(todo):,} orphan chunks to embed", file=sys.stderr)
    if not apply or not todo:
        conn.close()
        return

    done = 0
    for i in range(0, len(todo), batch_size):
        if _shutdown:
            break
        batch = todo[i:i + batch_size]
        texts = [c[1] for c in batch]
        embeddings = _embed_texts(texts)
        for (chunk_id, _), emb in zip(batch, embeddings):
            try:
                conn.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                    (chunk_id, emb),
                )
            except sqlite3.IntegrityError:
                # Concurrent backfill already inserted. Skip.
                pass
        done += len(batch)
        conn.commit()
        print(f"  progress: {done:,}/{len(todo):,}", file=sys.stderr)
    conn.close()
    print(f"[step 2] embedded {done:,} chunks", file=sys.stderr)


def step_reembed_possibly_stale(apply: bool, batch_size: int = 100) -> None:
    """Step 3: re-embed papers whose paper-level embedding may be out of sync with
    their current processed_text / tex_text. We don't have a log of which papers
    changed text during past dedup merges, so conservatively re-embed every
    has_full_text=1 paper whose current embedding was generated from a text of
    different length than the current text. Since we don't store the source
    length, we simply re-embed ALL has_full_text=1 papers that have vec_papers
    rows — ~45 minutes CPU.

    Transactional safety: this script does NOT wrap DELETE+INSERT in a SAVEPOINT.
    Each batch commits after completion. If the process dies mid-batch, the
    current batch's partial work is rolled back by SQLite's connection-close
    behavior (autocommit off), but batches that committed earlier persist.
    Result: a random subset of papers end up with no vec_papers row — those
    get re-backfilled on next run (idempotent) or by backfill_embeddings.py.
    Not wrapped in a savepoint because the cost of one failed batch is small
    and backfill coverage is idempotent."""
    conn = _init_db()
    todo = conn.execute("""
        SELECT p.paper_id, p.rowid, p.title, p.abstract, p.processed_text, p.tex_text
        FROM papers p
        WHERE p.has_full_text = 1
        AND p.rowid IN (SELECT rowid FROM vec_papers)
    """).fetchall()
    print(f"[step 3] {len(todo):,} has_full_text papers to re-embed", file=sys.stderr)
    if not apply or not todo:
        conn.close()
        return

    done = 0
    skipped = 0
    for i in range(0, len(todo), batch_size):
        if _shutdown:
            break
        batch = todo[i:i + batch_size]
        texts = []
        rowids = []
        for pid, rowid, title, abstract, pt, tt in batch:
            text = _paper_embed_text(title or "", abstract, pt, tt)
            if not text:
                skipped += 1
                continue
            texts.append(text)
            rowids.append(rowid)
        if texts:
            embeddings = _embed_texts(texts)
            for rowid, emb in zip(rowids, embeddings):
                conn.execute("DELETE FROM vec_papers WHERE rowid = ?", (rowid,))
                conn.execute(
                    "INSERT INTO vec_papers(rowid, embedding) VALUES (?, ?)",
                    (rowid, emb),
                )
        done += len(batch)
        conn.commit()
        print(f"  progress: {done:,}/{len(todo):,}", file=sys.stderr)
    conn.close()
    print(f"[step 3] re-embedded {done:,} papers (skipped {skipped:,})", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only", type=int, choices=[1, 2, 3],
                    help="Run only step N (default: all)")
    args = ap.parse_args()

    steps = [args.only] if args.only else [1, 2, 3]

    print(f"[start] apply={args.apply} steps={steps}", file=sys.stderr)

    if 1 in steps:
        step_schema_v8(args.apply)
    if 2 in steps and not _shutdown:
        step_backfill_orphan_chunks(args.apply)
    if 3 in steps and not _shutdown:
        step_reembed_possibly_stale(args.apply)

    print("[done]", file=sys.stderr)


if __name__ == "__main__":
    main()
