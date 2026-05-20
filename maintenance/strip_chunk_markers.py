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
"""Strip Docling `<!-- page N -->` and `<!-- image -->` markers from paper_chunks.chunk_text.

The markers are useful for provenance (which page a passage came from) and stay in
papers.processed_text. But they contaminate retrieval results: 66% of chunks contain
a page marker, many appear mid-sentence. A user pasting a chunk as a verbatim quote
ends up with `<!-- page 3 -->` breaking the sentence.

Usage:
    uv run strip_chunk_markers.py              # dry-run (report)
    uv run strip_chunk_markers.py --apply      # execute
"""

import argparse
import re
import signal
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from server import _embed_chunks, _init_db, DB_PATH  # noqa: E402

# Include adjacent whitespace so marker removal doesn't leave trailing/leading spaces
# around the removed marker. Replacement is \n\n (the existing paragraph separator),
# since Docling puts markers on their own paragraphs. Subsequent collapse of 3+ newlines
# to 2 handles cases where a marker sat between paragraphs.
PAGE_MARKER = re.compile(r"[ \t]*<!--\s*page\s+-?\d+\s*-->[ \t]*\n?")
IMAGE_MARKER = re.compile(r"[ \t]*<!--\s*image\s*-->[ \t]*\n?")
BLANK_LINE_RUN = re.compile(r"\n{3,}")

_shutdown = False


def _on_signal(sig, frame):
    global _shutdown
    _shutdown = True
    print("\n[signal] Finishing current batch...", file=sys.stderr)


signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


def _strip(text: str) -> str:
    cleaned = PAGE_MARKER.sub("", text)
    cleaned = IMAGE_MARKER.sub("", cleaned)
    cleaned = BLANK_LINE_RUN.sub("\n\n", cleaned)
    return cleaned.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    # Use _init_db so sqlite-vec is loaded on the connection. Direct
    # sqlite3.connect produces a connection that can't touch vec_chunks,
    # and the re-embed step then fails silently per-paper (observed
    # 2026-04-18: 3,860 chunks had text updated but embeddings were not
    # refreshed because vec0 module was unavailable).
    conn = _init_db()

    rows = conn.execute("""
        SELECT c.chunk_id, c.chunk_text, c.paper_id
        FROM paper_chunks c
        WHERE c.chunk_text LIKE '%<!-- page%'
           OR c.chunk_text LIKE '%<!-- image%'
    """).fetchall()

    print(f"[start] {len(rows)} chunks contain markers", file=sys.stderr)

    changes: list[tuple[int, str, int, int, str]] = []  # (chunk_id, new_text, old_len, new_len, paper_id)
    touched_paper_ids: set[str] = set()
    for chunk_id, text, paper_id in rows:
        cleaned = _strip(text)
        if cleaned != text:
            changes.append((chunk_id, cleaned, len(text), len(cleaned), paper_id))
            touched_paper_ids.add(paper_id)

    print(f"[diff] {len(changes)} chunks will change", file=sys.stderr)

    if not args.apply:
        print("\n-- Sample before/after --", file=sys.stderr)
        for cid, new_text, old_len, new_len, _ in changes[:3]:
            old_text = conn.execute(
                "SELECT chunk_text FROM paper_chunks WHERE chunk_id = ?", (cid,)
            ).fetchone()[0]
            print(f"\nchunk {cid}: {old_len} -> {new_len} chars", file=sys.stderr)
            print(f"  BEFORE: {old_text[:200]!r}", file=sys.stderr)
            print(f"  AFTER:  {new_text[:200]!r}", file=sys.stderr)
        conn.close()
        return

    # Apply
    done = 0
    for cid, new_text, _, _, _ in changes:
        if _shutdown:
            break
        conn.execute("UPDATE paper_chunks SET chunk_text = ? WHERE chunk_id = ?", (new_text, cid))
        done += 1
        if done % 1000 == 0:
            conn.commit()
            print(f"  progress: {done}/{len(changes)}", file=sys.stderr)
    conn.commit()
    print(f"[update] {done} chunks updated", file=sys.stderr)

    # Re-embed touched chunks so vec_chunks stays in sync with chunk_text
    print(f"[re-embed] re-embedding chunks across {len(touched_paper_ids)} papers", file=sys.stderr)
    reembed_done = 0
    for pid in touched_paper_ids:
        if _shutdown:
            break
        try:
            _embed_chunks(conn, pid)
        except Exception as e:
            print(f"  WARN re-embed failed for {pid}: {e}", file=sys.stderr)
        reembed_done += 1
        if reembed_done % 100 == 0:
            conn.commit()
            print(f"  re-embed: {reembed_done}/{len(touched_paper_ids)}", file=sys.stderr)
    conn.commit()

    # Rebuild FTS on paper_chunks_fts (external-content FTS5 auto-syncs? Depends on triggers.)
    # Check if triggers exist; otherwise rebuild manually.
    trigger_rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='paper_chunks'"
    ).fetchall()
    if trigger_rows:
        trigger_names = [r[0] for r in trigger_rows]
        print(f"[fts] triggers present: {trigger_names}; updates propagated automatically", file=sys.stderr)
    else:
        print("[fts] no triggers found; rebuilding paper_chunks_fts", file=sys.stderr)
        try:
            conn.execute("INSERT INTO paper_chunks_fts(paper_chunks_fts) VALUES('rebuild')")
            conn.commit()
            print("[fts] rebuild complete", file=sys.stderr)
        except sqlite3.OperationalError as e:
            print(f"[fts] rebuild failed: {e}", file=sys.stderr)

    conn.close()
    print(f"\n[done] updated={done}", file=sys.stderr)


if __name__ == "__main__":
    main()
