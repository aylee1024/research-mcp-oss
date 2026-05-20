# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mcp[cli]",
#     "httpx",
#     "sentence-transformers[onnx]",
#     "onnxruntime",
#     "sqlite-vec",
#     "numpy",
#     "einops",
#     "torch",
# ]
# ///
"""Standalone embedding backfill for papers.db.

Uses ONNX Runtime on CPU with all cores. NOT MPS GPU.
Why: For 137M param models, CPU with AMX is 2x faster than MPS (PyTorch #77799).
sentence-transformers.encode() handles internal length-sorted batching.

Usage:
    # Check status:
    uv run backfill_embeddings.py --status

    # Run the backfill (chunks only, papers already done):
    caffeinate -i nice -n 10 uv run backfill_embeddings.py --backfill

    # Re-chunk papers with missing chunks:
    uv run backfill_embeddings.py --repair-chunks
"""

import argparse
import json
import os
import re
import signal
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import sqlite_vec

# Force unbuffered output so progress shows in log files
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_mcp.paths import PAPERS_DB_PATH as DB_PATH  # noqa: E402

PROGRESS_FILE = DB_PATH.parent / "backfill_progress.json"
BATCH_SIZE = 32  # internal encode() batch; 32 is sweet spot for ~650-token chunks on CPU
CHUNK_SPLIT = 5000  # process in groups to bound memory (tokenizer + sorting happens per group)

# Use 16 of 18 cores, leave 2 for system
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

_shutdown = False
def _handle_signal(sig, frame):
    global _shutdown
    _shutdown = True
    print("\n>>> Shutdown requested. Will exit after current encode() completes...")
signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def connect_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def load_model():
    import torch
    # Align with server.py (8 threads). Different thread counts silently produce
    # slightly different embeddings because PyTorch CPU reduction order depends
    # on thread schedule.
    torch.set_num_threads(8)
    from sentence_transformers import SentenceTransformer
    print("Loading nomic-embed-text-v1.5 (PyTorch CPU, 8 threads)...")
    t0 = time.time()
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        device="cpu",
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model


def paper_text(title, abstract, processed_text, tex_text=None):
    """Build the embedding input text for a paper.

    Delegates to server._paper_embed_text so this backfill path produces
    byte-identical embedding inputs to the live server path. Previously this
    reimplemented the logic with divergent semantics (tex_text preferred over
    abstract), which created silent drift between embeddings produced by this
    script vs. those produced by process_pdf / set_abstract / download_paper.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from server import _paper_embed_text
    text = _paper_embed_text(title or "", abstract, processed_text, tex_text)
    if not text:
        return f"search_document: {title or ''}"
    # _paper_embed_text already returns prefixed text; ensure the document prefix.
    if text.startswith("search_document:") or text.startswith("search_query:"):
        return text
    return f"search_document: {text}"


def cmd_status():
    conn = connect_db()
    # Count the embed-eligible cohort (wave-3): verified=1 OR has_full_text=1.
    # Previously verified=1 only, which hid merge-keepers.
    tp = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE verified = 1 OR has_full_text = 1"
    ).fetchone()[0]
    ep = conn.execute("SELECT COUNT(*) FROM vec_papers").fetchone()[0]
    tc = conn.execute("""
        SELECT COUNT(*) FROM paper_chunks c
        JOIN papers p ON c.paper_id = p.paper_id
        WHERE p.verified = 1 OR p.has_full_text = 1
    """).fetchone()[0]
    ec = conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
    # Unchunked: any paper with non-empty processed_text OR tex_text but no
    # chunks yet. Previously processed_text-only; TeX-only papers were
    # invisible to the repair flow even though _store_tex_text is the source
    # of their chunks in the live server.
    uc = conn.execute("""
        SELECT COUNT(*) FROM papers
        WHERE ((processed_text IS NOT NULL AND processed_text != '')
            OR (tex_text IS NOT NULL AND tex_text != ''))
        AND paper_id NOT IN (SELECT DISTINCT paper_id FROM paper_chunks)
    """).fetchone()[0]
    conn.close()
    print(f"Database: {DB_PATH}")
    print(f"Papers:   {ep:,} / {tp:,} embedded")
    print(f"Chunks:   {ec:,} / {tc:,} embedded ({tc-ec:,} remaining)")
    print(f"Unchunked papers: {uc}")


def cmd_repair_chunks():
    # Preflight server import BEFORE any DB mutation. If sentence-transformers
    # or torch cannot load, we abort here instead of after committing fresh
    # paper_chunks rows that would then have no vec_chunks.
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from server import _embed_chunks  # noqa: F401 — used below after preflight

    conn = connect_db()
    # Prefer tex_text as the chunk source when present (math-clean, same as
    # server._store_tex_text's behavior), else fall back to processed_text.
    # Filter verified = 1 because `_embed_chunks` returns early on verified = 0,
    # and chunk-level search paths gate on has_full_text which tracks verified.
    # Without this filter, unverified rows get fresh paper_chunks but zero
    # vec_chunks, the same silent partial-repair M3 was meant to eliminate.
    rows = conn.execute("""
        SELECT paper_id,
               CASE
                   WHEN tex_text IS NOT NULL AND tex_text != '' THEN tex_text
                   ELSE processed_text
               END AS chunk_text
        FROM papers
        WHERE (verified = 1 OR has_full_text = 1)
        AND ((processed_text IS NOT NULL AND processed_text != '')
            OR (tex_text IS NOT NULL AND tex_text != ''))
        AND paper_id NOT IN (SELECT DISTINCT paper_id FROM paper_chunks)
    """).fetchall()
    if not rows:
        print("No papers need chunk repair.")
        conn.close()
        return
    print(f"Re-chunking {len(rows)} papers...")
    # Track paper_ids so we can re-embed chunks at the end. Previously
    # cmd_repair_chunks recreated paper_chunks but never embedded them,
    # leaving the semantic-leg (vec_chunks) empty until a separate
    # --backfill run. The repair was silently partial.
    repaired_pids: list[str] = []
    for paper_id, text in rows:
        conn.execute("DELETE FROM paper_chunks WHERE paper_id = ?", (paper_id,))
        paragraphs = text.split("\n\n")
        chunks = []
        current_parts, current_words = [], 0
        current_section, page_start, page_end = None, None, None
        def emit():
            nonlocal current_parts, current_words, page_start
            if current_parts:
                chunks.append({"text": "\n\n".join(current_parts), "section": current_section,
                               "page_start": page_start, "page_end": page_end})
                current_parts, current_words = [], 0
                page_start = page_end
        for para in paragraphs:
            stripped = para.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                current_section = stripped.lstrip("#").strip()
            page_match = re.search(r"<!-- page (\d+) -->", stripped)
            if page_match:
                pn = int(page_match.group(1))
                if page_start is None:
                    page_start = pn
                page_end = pn
            wc = len(stripped.split())
            if current_words + wc > 500 and current_parts:
                emit()
            current_parts.append(para)
            current_words += wc
        emit()
        for i, chunk in enumerate(chunks):
            conn.execute(
                "INSERT INTO paper_chunks (paper_id, chunk_index, chunk_text, section_header, page_start, page_end) VALUES (?, ?, ?, ?, ?, ?)",
                (paper_id, i, chunk["text"], chunk["section"], chunk["page_start"], chunk["page_end"]))
        title = conn.execute("SELECT title FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()[0]
        print(f"  {title[:60]} -> {len(chunks)} chunks")
        repaired_pids.append(paper_id)
    conn.commit()

    # Re-embed the repaired chunks so the semantic leg is in sync with the
    # new chunk text. Without this, the papers remain keyword-searchable
    # via FTS but return zero hits from the chunk-level vector leg.
    if repaired_pids:
        print(f"Re-embedding chunks for {len(repaired_pids)} papers...")
        for i, pid in enumerate(repaired_pids, 1):
            try:
                _embed_chunks(conn, pid)
            except Exception as e:
                print(f"  WARN embed failed for {pid}: {e}")
            if i % 100 == 0:
                conn.commit()
                print(f"  re-embed: {i}/{len(repaired_pids)}")
        conn.commit()

    conn.close()
    print("Done.")


def cmd_backfill():
    model = load_model()
    conn = connect_db()

    # --- Paper embeddings ---
    existing_p = {r[0] for r in conn.execute("SELECT rowid FROM vec_papers").fetchall()}
    papers = conn.execute(
        "SELECT rowid, title, abstract, processed_text, tex_text FROM papers "
        "WHERE verified = 1 OR has_full_text = 1"
    ).fetchall()
    todo_p = [
        (r[0], paper_text(r[1] or "", r[2], r[3], r[4]))
        for r in papers if r[0] not in existing_p
    ]

    if todo_p:
        print(f"\n=== Papers: {len(todo_p)} to embed ===")
        ids, texts = zip(*todo_p)
        embeddings = model.encode(list(texts), batch_size=BATCH_SIZE, show_progress_bar=True)
        for rowid, emb in zip(ids, embeddings):
            conn.execute("INSERT OR REPLACE INTO vec_papers(rowid, embedding) VALUES (?, ?)",
                         (rowid, emb.astype(np.float32).tobytes()))
        conn.commit()
        print(f"  Stored {len(todo_p)} paper embeddings.")
    else:
        print("Papers: all embedded.")

    if _shutdown:
        _save_progress(conn)
        conn.close()
        return

    # --- Chunk embeddings ---
    existing_c = {r[0] for r in conn.execute("SELECT rowid FROM vec_chunks").fetchall()}
    chunks = conn.execute("""
        SELECT c.chunk_id, c.chunk_text
        FROM paper_chunks c
        JOIN papers p ON c.paper_id = p.paper_id
        WHERE p.verified = 1 OR p.has_full_text = 1
    """).fetchall()
    todo_c = [(r[0], f"search_document: {r[1]}") for r in chunks if r[0] not in existing_c]

    if not todo_c:
        print("Chunks: all embedded.")
        _save_progress(conn)
        conn.close()
        return

    total = len(todo_c)
    print(f"\n=== Chunks: {total} to embed ===")

    # Process in halves to avoid tokenizer corruption at ~60K items
    stored = 0
    for split_start in range(0, total, CHUNK_SPLIT):
        if _shutdown:
            break
        split_end = min(split_start + CHUNK_SPLIT, total)
        batch = todo_c[split_start:split_end]
        batch_ids, batch_texts = zip(*batch)

        print(f"  Encoding items {split_start+1}-{split_end} of {total}...")
        t0 = time.time()
        embeddings = model.encode(list(batch_texts), batch_size=BATCH_SIZE, show_progress_bar=True)
        elapsed = time.time() - t0
        rate = len(batch_ids) / elapsed
        print(f"  Encoded {len(batch_ids)} in {elapsed:.1f}s ({rate:.1f}/s)")

        print(f"  Storing in database...")
        for cid, emb in zip(batch_ids, embeddings):
            conn.execute("INSERT OR REPLACE INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                         (cid, emb.astype(np.float32).tobytes()))
        conn.commit()
        stored += len(batch_ids)
        print(f"  Committed. Total stored: {stored}/{total}")

    _save_progress(conn)
    conn.close()
    print(f"\n=== Backfill complete. {stored} chunk embeddings created. ===")


def _save_progress(conn):
    p = conn.execute("SELECT COUNT(*) FROM vec_papers").fetchone()[0]
    c = conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
    PROGRESS_FILE.write_text(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                         "vec_papers": p, "vec_chunks": c}, indent=2))
    print(f"Progress saved to {PROGRESS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding backfill for papers.db")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true")
    group.add_argument("--backfill", action="store_true")
    group.add_argument("--repair-chunks", action="store_true")
    args = parser.parse_args()

    if args.status:
        cmd_status()
    elif args.repair_chunks:
        cmd_repair_chunks()
    elif args.backfill:
        cmd_backfill()
