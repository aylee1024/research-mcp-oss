#!/usr/bin/env -S uv run --script
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
#     "mlx",
#     "mlx-embeddings",
# ]
# ///
"""Plan v3 Phase 4R Phase 2: backfill Qwen3-Embedding-4B-MLX vectors.

Walks every cite-ready paper + chunk, encodes via Qwen3-MLX at the
chosen Matryoshka dim, writes to vec_papers_qwen3_<dim> /
vec_chunks_qwen3_<dim>. Nomic vec_papers / vec_chunks remain
untouched, so EMBED_VARIANT can flip back to nomic-768d at any time.

Idempotent: skips rowids already present in the destination table.
Use --rebuild to wipe and re-encode.

Usage:
    uv run maintenance/backfill_embed_qwen3.py --dim 1024
    uv run maintenance/backfill_embed_qwen3.py --dim 2560 --batch 16 --limit-papers 100
    uv run maintenance/backfill_embed_qwen3.py --dim 1024 --rebuild
    uv run maintenance/backfill_embed_qwen3.py --dim 1024 --skip-papers   # chunks only
    uv run maintenance/backfill_embed_qwen3.py --dim 1024 --skip-chunks   # papers only

Plan v3 §3 Phase 4R smoke gate: post-backfill, full bench at
EMBED_VARIANT=qwen3-mlx-<dim> must NOT regress >5pt on any metric vs
the v2 baseline at SHA 80dd084.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from research_mcp.paths import PAPERS_DB_PATH as DEFAULT_DB  # noqa: E402


def _supports(dim: int) -> bool:
    return dim in (512, 1024, 2560)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rebuild", action="store_true", help="Wipe destination tables before refilling.")
    ap.add_argument("--limit-papers", type=int, default=0)
    ap.add_argument("--limit-chunks", type=int, default=0)
    ap.add_argument("--skip-papers", action="store_true")
    ap.add_argument("--skip-chunks", action="store_true")
    args = ap.parse_args()

    if not _supports(args.dim):
        print(f"--dim must be 512, 1024, or 2560 (got {args.dim})", file=sys.stderr)
        return 2

    # Round-4 review HIGH fix: --rebuild + --limit-* DELETEs the entire
    # corpus then refills only --limit rows, silently destroying
    # coverage. Refuse the combination.
    if args.rebuild and (args.limit_papers or args.limit_chunks):
        print(
            "--rebuild is incompatible with --limit-papers / --limit-chunks: "
            "rebuild deletes the full table before refilling, but limit caps "
            "the refill, so the result is a partial corpus. Re-run without "
            "--rebuild for a smoke test, or accept the full re-encode "
            "(~5 hours for the default corpus).",
            file=sys.stderr,
        )
        return 2

    # --db must be wired to server.DB_PATH BEFORE the server module is
    # imported, since DB_PATH is resolved at import time via
    # research_mcp.paths. Setting PAPERS_DB_PATH here lets the import pick
    # up the override.
    if str(args.db) != str(DEFAULT_DB):
        os.environ["PAPERS_DB_PATH"] = str(args.db.resolve())
    # Force the import-side EMBED_VARIANT to qwen3 so server's encoder helpers
    # take the MLX path, then import server functions.
    os.environ["EMBED_VARIANT"] = f"qwen3-mlx-{args.dim}"
    sys.path.insert(0, str(ROOT))
    from server import _qwen3_mlx_encode, _init_db, _paper_embed_text  # noqa: E402

    paper_table = f"vec_papers_qwen3_{args.dim}"
    chunk_table = f"vec_chunks_qwen3_{args.dim}"

    conn = _init_db()
    if args.rebuild:
        # Round-2 review CRITICAL fix: --rebuild + --skip-papers (or
        # --skip-chunks) previously DELETEd both tables but only re-
        # filled one, orphaning the skipped table indefinitely. Only
        # DELETE from the table that will be re-filled.
        delete_targets: list[str] = []
        if not args.skip_papers:
            delete_targets.append(paper_table)
        if not args.skip_chunks:
            delete_targets.append(chunk_table)
        if not delete_targets:
            print(
                "[backfill] --rebuild with both --skip-papers and --skip-chunks "
                "is a no-op; nothing to do.",
                file=sys.stderr,
            )
            return 0
        print(f"[backfill] rebuild: DELETE FROM {', '.join(delete_targets)}", flush=True)
        for tbl in delete_targets:
            conn.execute(f"DELETE FROM {tbl}")
        # Wave-4 review HIGH fix: do NOT commit here. Defer until first batch
        # writes complete so concurrent readers don't see an empty vec table
        # window. The DELETEs land via the first batch's commit instead.

    # Round-3 review HIGH fix: persist encode failures across both
    # phases so the exit code can reflect them and a sidecar file
    # records the offending rowids for human review.
    total_encode_failed_rowids: list[tuple[str, int]] = []  # ("papers"|"chunks", rowid)

    # ---- Papers ----
    if not args.skip_papers:
        existing_paper_rowids = {r[0] for r in conn.execute(f"SELECT rowid FROM {paper_table}").fetchall()}
        # Mirror nomic policy: embed verified OR has_full_text rows. (See
        # _embed_paper_if_verified in server.py.)
        rows = conn.execute(
            "SELECT p.rowid, p.title, p.abstract, p.processed_text, p.tex_text "
            "FROM papers p WHERE (p.verified = 1 OR p.has_full_text = 1)"
        ).fetchall()
        # Wave-4 review MEDIUM fix: filter existing rowids BEFORE applying limit
        # so --limit-papers 100 means "at most 100 NEW encodings" not "look at
        # the first 100 candidates and skip the ones already done".
        todo = [r for r in rows if r[0] not in existing_paper_rowids]
        if args.limit_papers:
            todo = todo[: args.limit_papers]
        print(f"[papers] eligible={len(rows)} present={len(existing_paper_rowids)} todo={len(todo)}", flush=True)
        t0 = time.time()
        encoded = 0
        skipped_empty = 0
        encode_failed = 0
        for i in range(0, len(todo), args.batch):
            batch_rows = todo[i : i + args.batch]
            texts: list[str] = []
            rowids: list[int] = []
            for rowid, title, abstract, processed_text, tex_text in batch_rows:
                text = _paper_embed_text(title or "", abstract, processed_text, tex_text)
                if not text:
                    skipped_empty += 1
                    continue
                texts.append(text)
                rowids.append(rowid)
            if not texts:
                continue
            # Wave-4 review HIGH fix: try/except so a single malformed text in a
            # batch doesn't crash the whole 5-hour run. Log offending rowids and
            # skip the batch (caller can resume + investigate).
            try:
                embs = _qwen3_mlx_encode(texts, args.dim)
            except Exception as exc:
                encode_failed += len(texts)
                for r in rowids:
                    total_encode_failed_rowids.append(("papers", r))
                print(
                    f"  [papers] ENCODE FAIL batch={i//args.batch} rowids={rowids} err={exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            for rowid, emb in zip(rowids, embs):
                conn.execute(f"INSERT INTO {paper_table}(rowid, embedding) VALUES (?, ?)", (rowid, emb))
            encoded += len(texts)
            if encoded % 100 < args.batch:
                conn.commit()
                rate = encoded / max(time.time() - t0, 1e-9)
                eta = (len(todo) - i - args.batch) / max(rate, 0.1)
                print(f"  [papers] {i + args.batch}/{len(todo)} ({rate:.1f}/s, eta {eta:.0f}s)", flush=True)
        # Round-5 review HIGH fix: previously this final commit fired
        # unconditionally, so under --rebuild + total encode failure
        # (encoded=0 throughout), the DELETE alone would commit and
        # the user lost their entire vec index. Now: if rebuild was
        # active AND zero papers were encoded AND there were failures,
        # explicitly rollback the pending DELETE instead of committing
        # an empty table.
        if args.rebuild and encoded == 0 and encode_failed > 0:
            conn.rollback()
            print(
                "[papers] CRITICAL: --rebuild + total encode failure detected; "
                "rolling back DELETE to preserve prior vec_papers_qwen3 index. "
                "Investigate the encoder error and re-run.",
                file=sys.stderr,
                flush=True,
            )
        else:
            conn.commit()
        print(
            f"[papers] encoded={encoded} skipped_empty={skipped_empty} "
            f"encode_failed={encode_failed} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    # ---- Chunks ----
    if not args.skip_chunks:
        existing_chunk_rowids = {r[0] for r in conn.execute(f"SELECT rowid FROM {chunk_table}").fetchall()}
        rows = conn.execute(
            "SELECT c.chunk_id, c.chunk_text "
            "FROM paper_chunks c JOIN papers p ON c.paper_id = p.paper_id "
            "WHERE (p.verified = 1 OR p.has_full_text = 1)"
        ).fetchall()
        # Wave-4 review MEDIUM fix: filter BEFORE limit (mirrors papers fix).
        todo = [r for r in rows if r[0] not in existing_chunk_rowids]
        if args.limit_chunks:
            todo = todo[: args.limit_chunks]
        print(f"[chunks] eligible={len(rows)} present={len(existing_chunk_rowids)} todo={len(todo)}", flush=True)
        t0 = time.time()
        encoded = 0
        skipped_empty = 0
        encode_failed = 0
        for i in range(0, len(todo), args.batch):
            batch_rows = todo[i : i + args.batch]
            # Wave-4 review HIGH fix: track skipped_empty for chunks (matches
            # papers); use a single comprehension over batch_rows so texts and
            # rowids stay aligned even after a future refactor to a generator.
            paired = [(cid, text) for cid, text in batch_rows if text]
            skipped_empty += len(batch_rows) - len(paired)
            if not paired:
                continue
            rowids = [cid for cid, _ in paired]
            texts = [text for _, text in paired]
            try:
                embs = _qwen3_mlx_encode(texts, args.dim)
            except Exception as exc:
                encode_failed += len(texts)
                for r in rowids:
                    total_encode_failed_rowids.append(("chunks", r))
                print(
                    f"  [chunks] ENCODE FAIL batch={i//args.batch} rowids={rowids} err={exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            for chunk_id, emb in zip(rowids, embs):
                conn.execute(f"INSERT INTO {chunk_table}(rowid, embedding) VALUES (?, ?)", (chunk_id, emb))
            encoded += len(texts)
            if encoded % 500 < args.batch:
                conn.commit()
                rate = encoded / max(time.time() - t0, 1e-9)
                eta = (len(todo) - i - args.batch) / max(rate, 0.1)
                print(f"  [chunks] {i + args.batch}/{len(todo)} ({rate:.1f}/s, eta {eta:.0f}s)", flush=True)
        # Round-5 review HIGH fix (mirror of papers phase).
        if args.rebuild and encoded == 0 and encode_failed > 0:
            conn.rollback()
            print(
                "[chunks] CRITICAL: --rebuild + total encode failure detected; "
                "rolling back DELETE to preserve prior vec_chunks_qwen3 index.",
                file=sys.stderr,
                flush=True,
            )
        else:
            conn.commit()
        print(
            f"[chunks] encoded={encoded} skipped_empty={skipped_empty} "
            f"encode_failed={encode_failed} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    # Summary
    p_count = conn.execute(f"SELECT COUNT(*) FROM {paper_table}").fetchone()[0]
    c_count = conn.execute(f"SELECT COUNT(*) FROM {chunk_table}").fetchone()[0]
    print(f"FINAL: {paper_table}={p_count}, {chunk_table}={c_count}")
    # Round-3 review HIGH fix: persist failed rowids + non-zero exit so
    # CI can detect partial-failure runs even if stderr is discarded.
    if total_encode_failed_rowids:
        # Round-5 review HIGH fix: if --rebuild AND every batch failed
        # (the destination tables ended up empty), the prior commit
        # fired anyway, durably wiping the user's index. Detect the
        # all-failure-under-rebuild case via FINAL counts and emit a
        # loud sentinel so the operator knows to restore from backup.
        # Cannot retroactively roll back here because `conn.commit()`
        # at end-of-phase already landed; this guard is the warning
        # they need to see.
        if args.rebuild and (
            (not args.skip_papers and p_count == 0)
            or (not args.skip_chunks and c_count == 0)
        ):
            print(
                f"CRITICAL: --rebuild + total encode failure left destination "
                f"table EMPTY ({paper_table}={p_count}, {chunk_table}={c_count}). "
                f"The vec index has been wiped. Restore the qwen3 vec rows from "
                f"backup OR rerun without --rebuild after fixing the encoder.",
                file=sys.stderr,
            )
        sidecar = ROOT / "maintenance" / "logs" / f"backfill_embed_qwen3_failed_dim{args.dim}.txt"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with sidecar.open("w") as fh:
            fh.write(f"# Encode failures from backfill_embed_qwen3.py at {time.time()}\n")
            for table, rowid in total_encode_failed_rowids:
                fh.write(f"{table}\t{rowid}\n")
        print(
            f"FAILED: {len(total_encode_failed_rowids)} encode failures recorded to {sidecar}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
