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
"""One-off database repair: fix misassigned text, text-hash dedup, orphan cleanup, backfill embeddings.

Usage:
    uv run repair_db.py              # dry-run (default): report only
    uv run repair_db.py --apply      # execute repairs
"""

import argparse
import hashlib
import signal
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from server import (
    _init_db,
    _recompute_has_full_text,
    _chunk_processed_text,
    _build_paper_passages,
    _embed_paper_if_verified,
    _embed_chunks,
    DB_PATH,
)

_shutdown = False


def _on_signal(sig, frame):
    global _shutdown
    _shutdown = True
    print("\n[signal] Finishing current operation...", file=sys.stderr)


signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


def _word_set(text: str) -> set[str]:
    return set(text.lower().split())


def _paper_id_priority(pid: str) -> int:
    if pid.startswith("web:"):
        return 0
    if pid.startswith("local:"):
        return 1
    if pid.startswith("oa:"):
        return 2
    return 3


def _richness(conn, pid: str) -> int:
    row = conn.execute(
        "SELECT LENGTH(COALESCE(processed_text,'')), LENGTH(COALESCE(tex_text,'')), "
        "LENGTH(COALESCE(abstract,'')), local_pdf_path IS NOT NULL, citation_count "
        "FROM papers WHERE paper_id = ?",
        (pid,),
    ).fetchone()
    if not row:
        return 0
    pt, tt, ab, has_pdf, cites = row
    return (pt or 0) + (tt or 0) + (ab or 0) * 10 + (1000 if has_pdf else 0) + (cites or 0)


def _pick_keeper(conn, pids: list[str]) -> tuple[str, list[str]]:
    scored = [(pid, _paper_id_priority(pid), _richness(conn, pid)) for pid in pids]
    scored.sort(key=lambda x: (-x[1], -x[2]))
    keep = scored[0][0]
    delete = [pid for pid, _, _ in scored[1:]]
    return keep, delete


def _merge_into(conn, keep_id: str, delete_id: str) -> str:
    # Self-merge guard. Without this, DELETE FROM papers WHERE paper_id =
    # delete_id would delete the survivor.
    if keep_id == delete_id:
        return f"  SKIP: keep_id == delete_id ({keep_id})"

    keep = conn.execute(
        "SELECT title, doi, processed_text, tex_text, local_pdf_path, abstract, authors, year "
        "FROM papers WHERE paper_id = ?", (keep_id,)
    ).fetchone()
    delete = conn.execute(
        "SELECT title, doi, processed_text, tex_text, local_pdf_path, abstract, authors, year "
        "FROM papers WHERE paper_id = ?", (delete_id,)
    ).fetchone()
    if not keep or not delete:
        return "  SKIP: one entry not found"

    k_title, k_doi, k_pt, k_tt, k_pdf, k_abs, k_auth, k_year = keep
    d_title, d_doi, d_pt, d_tt, d_pdf, d_abs, d_auth, d_year = delete

    changes = []
    updates = []
    params = []

    if d_doi and not k_doi:
        updates.append("doi = ?")
        params.append(d_doi)
        changes.append("doi")
    keep_prio = _paper_id_priority(keep_id)
    delete_prio = _paper_id_priority(delete_id)
    if d_pt and not k_pt:
        updates.append("processed_text = ?")
        params.append(d_pt)
        changes.append(f"processed_text ({len(d_pt):,} chars)")
    elif d_pt and k_pt:
        if delete_prio > keep_prio and len(d_pt) >= len(k_pt) * 0.7:
            updates.append("processed_text = ?")
            params.append(d_pt)
            changes.append(f"processed_text (higher-source, {len(d_pt):,} chars)")
        elif delete_prio == keep_prio and len(d_pt) > len(k_pt) * 1.1:
            updates.append("processed_text = ?")
            params.append(d_pt)
            changes.append(f"processed_text ({len(d_pt):,} chars)")
    if d_tt and (not k_tt or len(d_tt) > len(k_tt or "")):
        updates.append("tex_text = ?")
        params.append(d_tt)
        changes.append("tex_text")
    if d_pdf and not k_pdf:
        updates.append("local_pdf_path = ?")
        params.append(d_pdf)
        changes.append("local_pdf_path")
    if d_abs and (not k_abs or len(d_abs) > len(k_abs or "")):
        updates.append("abstract = ?")
        params.append(d_abs)
        changes.append("abstract")
    if d_auth and not k_auth:
        updates.append("authors = ?")
        params.append(d_auth)
        changes.append("authors")
    if d_year and not k_year:
        updates.append("year = ?")
        params.append(d_year)
        changes.append("year")

    if updates:
        params.append(keep_id)
        conn.execute(f"UPDATE papers SET {', '.join(updates)} WHERE paper_id = ?", params)

    # Drop delete-side chunks entirely. Previously this redirected chunks
    # via `UPDATE OR IGNORE paper_chunks SET paper_id = keep_id`, but
    # paper_chunks has no UNIQUE(paper_id, chunk_index) so the UPDATE never
    # conflicts and the keeper ended up with two chunk sets. The keeper's
    # own chunks are authoritative; delete-side chunks are redundant. v15
    # triggers would handle this automatically on the final DELETE FROM
    # papers below, but we pre-empt for clarity and pre-v15 safety.
    try:
        delete_chunk_ids = [r[0] for r in conn.execute(
            "SELECT chunk_id FROM paper_chunks WHERE paper_id = ?", (delete_id,)
        ).fetchall()]
        if delete_chunk_ids:
            ph = ",".join("?" * len(delete_chunk_ids))
            conn.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({ph})", delete_chunk_ids)
    except sqlite3.OperationalError:
        pass
    conn.execute("DELETE FROM paper_chunks WHERE paper_id = ?", (delete_id,))

    try:
        row = conn.execute("SELECT rowid FROM papers WHERE paper_id = ?", (delete_id,)).fetchone()
        if row:
            conn.execute("DELETE FROM vec_papers WHERE rowid = ?", (row[0],))
    except sqlite3.OperationalError:
        pass

    conn.execute("UPDATE OR IGNORE paper_references SET citing_paper_id = ? WHERE citing_paper_id = ?", (keep_id, delete_id))
    conn.execute("UPDATE OR IGNORE paper_references SET cited_paper_id = ? WHERE cited_paper_id = ?", (keep_id, delete_id))

    # Normalize cited_doi on retargeted edges so a later DELETE on the
    # keeper produces a self-consistent __doi placeholder via the v17
    # trigger (which reads cited_doi first).
    keep_doi_now = conn.execute(
        "SELECT doi FROM papers WHERE paper_id = ?", (keep_id,)
    ).fetchone()
    if keep_doi_now and keep_doi_now[0]:
        conn.execute(
            "UPDATE paper_references SET cited_doi = ? "
            "WHERE cited_paper_id = ? AND (cited_doi IS NULL OR cited_doi != ?)",
            (keep_doi_now[0], keep_id, keep_doi_now[0]),
        )

    conn.execute("DELETE FROM paper_references WHERE citing_paper_id = ? OR cited_paper_id = ?", (delete_id, delete_id))
    conn.execute("DELETE FROM paper_references WHERE citing_paper_id = ? AND cited_paper_id = ?", (keep_id, keep_id))

    conn.execute("DELETE FROM papers WHERE paper_id = ?", (delete_id,))

    # Restore the invariant set that dedup_papers._merge_into already
    # honors: has_full_text must reflect the text actually on the keeper,
    # and a text-copy must re-chunk + re-embed. Previously this function
    # omitted all three steps, leaving merge survivors with stale
    # has_full_text, stale chunks for the old body text, or no chunks at
    # all when the keeper had been metadata-only.
    _recompute_has_full_text(conn, keep_id)

    changes_joined = " ".join(changes)
    processed_changed = "processed_text" in changes_joined
    tex_changed = "tex_text" in changes_joined
    chunk_source_changed = processed_changed or tex_changed
    embed_inputs_changed = any(
        k in changes_joined for k in ("processed_text", "abstract", "tex_text")
    )

    # Keeper-has-no-chunks guard: if the keeper now carries cite-ready text
    # but no paper_chunks rows (because the keeper had no chunks pre-merge
    # and we dropped the delete-side chunks), force a re-chunk so the
    # semantic + FTS passage legs are not silently empty.
    kept_now = conn.execute(
        "SELECT processed_text, tex_text FROM papers WHERE paper_id = ?",
        (keep_id,),
    ).fetchone()
    if kept_now:
        kept_pt_now, kept_tt_now = kept_now
        kept_has_cite_text = (
            (kept_tt_now and len(kept_tt_now) > 500)
            or (kept_pt_now and len(kept_pt_now) > 500)
        )
        has_chunks = conn.execute(
            "SELECT 1 FROM paper_chunks WHERE paper_id = ? LIMIT 1", (keep_id,)
        ).fetchone() is not None
        if kept_has_cite_text and not has_chunks:
            chunk_source_changed = True
            embed_inputs_changed = True

    if embed_inputs_changed:
        kept = conn.execute(
            "SELECT processed_text, tex_text FROM papers WHERE paper_id = ?",
            (keep_id,),
        ).fetchone()
        if kept:
            kept_pt, kept_tt = kept
            chunk_text = None
            if kept_tt and len(kept_tt) > 500:
                chunk_text = kept_tt
            elif kept_pt and len(kept_pt) > 500:
                chunk_text = kept_pt
            if chunk_text is not None:
                try:
                    if chunk_source_changed:
                        _chunk_processed_text(conn, keep_id, chunk_text)
                        _build_paper_passages(conn, keep_id, chunk_text)
                    _embed_paper_if_verified(conn, keep_id)
                    if chunk_source_changed:
                        _embed_chunks(conn, keep_id)
                except Exception as e:
                    # repair_db runs as a one-off batch without a dedup-
                    # style merge savepoint; re-embed failure is logged and
                    # moves on. The keeper is in a committed text-only
                    # state; the next run of backfill_embeddings --backfill
                    # will pick it up.
                    print(f"  WARN merge re-embed failed for {keep_id}: {e}", file=sys.stderr)

    desc = f"  merged {delete_id} -> {keep_id}"
    if changes:
        desc += f" (copied: {', '.join(changes)})"
    return desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Execute repairs (default: dry-run)")
    args = ap.parse_args()
    apply = args.apply

    print(f"[start] repair_db  apply={apply}", file=sys.stderr)
    conn = _init_db()

    # ── 9a. Detect misassigned text ──────────────────────────────────
    print("\n=== 9a: Misassigned text detection ===", file=sys.stderr)
    misassigned = []
    rows = conn.execute("""
        SELECT paper_id, title, SUBSTR(processed_text, 1, 2000)
        FROM papers
        WHERE paper_id LIKE 'local:%'
        AND processed_text IS NOT NULL
        AND LENGTH(processed_text) > 500
        AND title IS NOT NULL
        AND LENGTH(title) > 5
    """).fetchall()

    for pid, db_title, text_head in rows:
        if _shutdown:
            break
        title_words = _word_set(db_title)
        if not title_words:
            continue
        text_words = _word_set(text_head[:500])
        overlap = len(title_words & text_words) / len(title_words) if title_words else 0
        if overlap < 0.3 and len(title_words) >= 3:
            misassigned.append(pid)

    print(f"  Found {len(misassigned)} likely misassigned papers", file=sys.stderr)
    for pid in misassigned[:10]:
        row = conn.execute("SELECT title FROM papers WHERE paper_id = ?", (pid,)).fetchone()
        print(f"    {pid}: {row[0][:60] if row else '?'}", file=sys.stderr)
    if len(misassigned) > 10:
        print(f"    ... and {len(misassigned) - 10} more", file=sys.stderr)

    if apply and misassigned:
        for pid in misassigned:
            if _shutdown:
                break
            # Clear text, chunks, embeddings
            chunk_ids = [r[0] for r in conn.execute(
                "SELECT chunk_id FROM paper_chunks WHERE paper_id = ?", (pid,)
            ).fetchall()]
            if chunk_ids:
                ph = ",".join("?" * len(chunk_ids))
                try:
                    conn.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({ph})", chunk_ids)
                except sqlite3.OperationalError:
                    pass
            conn.execute("DELETE FROM paper_chunks WHERE paper_id = ?", (pid,))
            try:
                row = conn.execute("SELECT rowid FROM papers WHERE paper_id = ?", (pid,)).fetchone()
                if row:
                    conn.execute("DELETE FROM vec_papers WHERE rowid = ?", (row[0],))
            except sqlite3.OperationalError:
                pass
            conn.execute("UPDATE papers SET processed_text = NULL WHERE paper_id = ?", (pid,))
            _recompute_has_full_text(conn, pid)
        conn.commit()
        print(f"  Cleared text for {len(misassigned)} papers", file=sys.stderr)

    # ── 9b. Text-hash dedup ──────────────────────────────────────────
    print("\n=== 9b: Text-hash dedup ===", file=sys.stderr)
    text_rows = conn.execute("""
        SELECT paper_id, processed_text FROM papers
        WHERE processed_text IS NOT NULL AND LENGTH(processed_text) > 500
    """).fetchall()

    hash_groups: dict[str, list[str]] = defaultdict(list)
    for pid, text in text_rows:
        h = hashlib.md5(text.encode()).hexdigest()
        hash_groups[h].append(pid)

    dup_groups = {h: pids for h, pids in hash_groups.items() if len(pids) > 1}
    excess = sum(len(pids) - 1 for pids in dup_groups.values())
    print(f"  Found {len(dup_groups)} groups with identical text, {excess} excess copies", file=sys.stderr)

    merge_count = 0
    for h, pids in dup_groups.items():
        if _shutdown:
            break
        keep, deletes = _pick_keeper(conn, pids)
        for d in deletes:
            exists = conn.execute("SELECT 1 FROM papers WHERE paper_id = ?", (d,)).fetchone()
            if not exists:
                continue
            if apply:
                desc = _merge_into(conn, keep, d)
                print(f"  text-hash: {desc}", file=sys.stderr)
            else:
                print(f"  would merge {d} -> {keep}", file=sys.stderr)
            merge_count += 1
        if apply and merge_count % 100 == 0:
            conn.commit()

    if apply:
        conn.commit()
    print(f"  Total merges: {merge_count}", file=sys.stderr)

    # ── 9c. Clean orphaned chunks ────────────────────────────────────
    print("\n=== 9c: Orphaned chunks ===", file=sys.stderr)
    orphan_chunks = conn.execute("""
        SELECT pc.chunk_id FROM paper_chunks pc
        LEFT JOIN papers p ON pc.paper_id = p.paper_id
        WHERE p.paper_id IS NULL
    """).fetchall()
    orphan_ids = [r[0] for r in orphan_chunks]
    print(f"  Found {len(orphan_ids)} orphaned chunks", file=sys.stderr)

    if apply and orphan_ids:
        # Delete vec_chunks first
        batch_size = 500
        for i in range(0, len(orphan_ids), batch_size):
            batch = orphan_ids[i:i + batch_size]
            ph = ",".join("?" * len(batch))
            try:
                conn.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({ph})", batch)
            except sqlite3.OperationalError:
                pass
            conn.execute(f"DELETE FROM paper_chunks WHERE chunk_id IN ({ph})", batch)
        conn.commit()
        print(f"  Deleted {len(orphan_ids)} orphaned chunks", file=sys.stderr)

    # ── 9d. Self-referencing citations ───────────────────────────────
    print("\n=== 9d: Self-referencing citations ===", file=sys.stderr)
    self_refs = conn.execute(
        "SELECT COUNT(*) FROM paper_references WHERE citing_paper_id = cited_paper_id"
    ).fetchone()[0]
    print(f"  Found {self_refs} self-referencing citations", file=sys.stderr)

    if apply and self_refs > 0:
        conn.execute("DELETE FROM paper_references WHERE citing_paper_id = cited_paper_id")
        conn.commit()
        print(f"  Deleted {self_refs} self-referencing citations", file=sys.stderr)

    # ── 9e. Re-chunk papers with text but no chunks ──────────────────
    print("\n=== 9e: Papers with text but no chunks ===", file=sys.stderr)
    no_chunk_rows = conn.execute("""
        SELECT p.paper_id, p.processed_text FROM papers p
        LEFT JOIN paper_chunks pc ON p.paper_id = pc.paper_id
        WHERE p.processed_text IS NOT NULL
        AND LENGTH(p.processed_text) > 500
        AND pc.chunk_id IS NULL
        GROUP BY p.paper_id
    """).fetchall()
    print(f"  Found {len(no_chunk_rows)} papers with text but no chunks", file=sys.stderr)

    if apply:
        for pid, text in no_chunk_rows:
            if _shutdown:
                break
            count = _chunk_processed_text(conn, pid, text)
            _build_paper_passages(conn, pid, text)
            _embed_paper_if_verified(conn, pid)
            _embed_chunks(conn, pid)
            print(f"  re-chunked {pid}: {count} chunks", file=sys.stderr)
        conn.commit()

    # ── 9f. Backfill missing embeddings ──────────────────────────────
    print("\n=== 9f: Missing embeddings ===", file=sys.stderr)
    missing_emb = conn.execute("""
        SELECT paper_id FROM papers
        WHERE verified = 1
        AND (LENGTH(COALESCE(abstract,'')) > 50 OR LENGTH(COALESCE(processed_text,'')) > 100)
        AND rowid NOT IN (SELECT rowid FROM vec_papers)
    """).fetchall()
    missing_pids = [r[0] for r in missing_emb]
    print(f"  Found {len(missing_pids)} verified papers missing vec_papers embeddings", file=sys.stderr)

    if apply and missing_pids:
        done = 0
        for pid in missing_pids:
            if _shutdown:
                break
            _embed_paper_if_verified(conn, pid)
            done += 1
            if done % 100 == 0:
                conn.commit()
                print(f"  progress: {done}/{len(missing_pids)}", file=sys.stderr)
        conn.commit()
        print(f"  Embedded {done} papers", file=sys.stderr)

    conn.close()
    print(f"\n[done] apply={apply}", file=sys.stderr)


if __name__ == "__main__":
    main()
