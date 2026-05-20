# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "sqlite-vec",
# ]
# ///
"""One-off cleanup: delete orphan rows in vec_papers and vec_chunks.

Context: prior to schema v15, referential integrity between papers/paper_chunks
and the vec0 virtual tables was entirely manual. Historical delete paths
(especially build_reacquire_list.py before the 2026-04-18 fix) omitted the
vec_chunks cleanup step, leaving orphan embedding rows whose rowid no longer
matches any paper_chunks.chunk_id. Similarly vec_papers can outlive papers.

After v15, these orphans cannot recur (triggers cascade the cleanup). This
script deletes the historical accumulation.

Usage:
    uv run cleanup_orphans.py              # dry-run: report counts + samples
    uv run cleanup_orphans.py --apply      # execute deletion
"""

import argparse
import sqlite3
import sqlite_vec
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_mcp.paths import PAPERS_DB_PATH as DB_PATH  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Execute deletion (default: dry-run)")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA busy_timeout = 120000")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    orphan_vec_papers = conn.execute("""
        SELECT v.rowid FROM vec_papers v
        LEFT JOIN papers p ON v.rowid = p.rowid
        WHERE p.rowid IS NULL
    """).fetchall()
    orphan_vec_chunks = conn.execute("""
        SELECT v.rowid FROM vec_chunks v
        LEFT JOIN paper_chunks c ON v.rowid = c.chunk_id
        WHERE c.chunk_id IS NULL
    """).fetchall()

    print(f"[found] orphan_vec_papers: {len(orphan_vec_papers)}", file=sys.stderr)
    print(f"[found] orphan_vec_chunks: {len(orphan_vec_chunks)}", file=sys.stderr)

    if not args.apply:
        sample_p = [r[0] for r in orphan_vec_papers[:10]]
        sample_c = [r[0] for r in orphan_vec_chunks[:10]]
        print(f"[sample] first 10 orphan vec_papers rowids: {sample_p}", file=sys.stderr)
        print(f"[sample] first 10 orphan vec_chunks rowids: {sample_c}", file=sys.stderr)
        print("\n[dry-run] pass --apply to execute", file=sys.stderr)
        conn.close()
        return 0

    if orphan_vec_papers:
        ids = [r[0] for r in orphan_vec_papers]
        ph = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM vec_papers WHERE rowid IN ({ph})", ids)
        print(f"[delete] vec_papers: {len(ids)} rows", file=sys.stderr)

    if orphan_vec_chunks:
        ids = [r[0] for r in orphan_vec_chunks]
        # Chunk the delete to avoid SQLite parameter limits (default ~32K).
        BATCH = 5000
        deleted = 0
        for i in range(0, len(ids), BATCH):
            batch = ids[i:i + BATCH]
            ph = ",".join("?" * len(batch))
            conn.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({ph})", batch)
            deleted += len(batch)
        print(f"[delete] vec_chunks: {deleted} rows", file=sys.stderr)

    conn.commit()

    # Verify clean
    remaining_p = conn.execute("""
        SELECT COUNT(*) FROM vec_papers v
        LEFT JOIN papers p ON v.rowid = p.rowid
        WHERE p.rowid IS NULL
    """).fetchone()[0]
    remaining_c = conn.execute("""
        SELECT COUNT(*) FROM vec_chunks v
        LEFT JOIN paper_chunks c ON v.rowid = c.chunk_id
        WHERE c.chunk_id IS NULL
    """).fetchone()[0]

    print(f"[verify] orphan_vec_papers remaining: {remaining_p}", file=sys.stderr)
    print(f"[verify] orphan_vec_chunks remaining: {remaining_c}", file=sys.stderr)

    conn.close()
    return 0 if (remaining_p == 0 and remaining_c == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
