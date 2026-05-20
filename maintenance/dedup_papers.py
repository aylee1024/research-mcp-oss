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
"""Deduplicate papers.db: find and merge duplicate entries for the same work.

Discovery strategies (candidates go into a union-find):
  1a. Same DOI, different paper_id
  1b. Same normalized title + same year (or one NULL year)
  1c. web: entries that duplicate a paper entry

Discovery is separated from merge. All strategies contribute edges to a
union-find; connected components are the canonical dup groups. One keeper
per component is selected via `_pick_keeper` and all other component
members are merged into it. This eliminates the pre-2026-04-18 bug where
a row could be the keeper in strategy 1a and then a delete target in
strategy 1b of the same run, making final survivors depend on strategy
order.

Usage:
    uv run dedup_papers.py              # dry-run (default): report only
    uv run dedup_papers.py --apply      # actually merge and delete
"""

import argparse
import json
import re
import signal
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from server import (
    _init_db,
    _chunk_processed_text,
    _build_paper_passages,
    _embed_paper_if_verified,
    _embed_chunks,
    _recompute_has_full_text,
    _delete_vec_chunks_for_paper,
    _delete_vec_papers_for_paper,
    DB_PATH,
    PAPERS_DIR,
)

_shutdown = False


def _on_signal(sig, frame):
    global _shutdown
    _shutdown = True
    print("\n[signal] Finishing current merge...", file=sys.stderr)


signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


def _paper_id_priority(pid: str) -> int:
    """Higher = better paper_id to keep. S2 SHA > oa: > local: > web:"""
    if pid.startswith("web:"):
        return 0
    if pid.startswith("local:"):
        return 1
    if pid.startswith("oa:"):
        return 2
    return 3  # S2 SHA-1 or other canonical IDs


def _richness(conn, pid: str) -> int:
    """Score how much content a paper entry has. Higher = richer."""
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
    """Pick the best paper_id to keep. Returns (keep_id, delete_ids)."""
    scored = [(pid, _paper_id_priority(pid), _richness(conn, pid)) for pid in pids]
    # Sort: highest priority first, then richest
    scored.sort(key=lambda x: (-x[1], -x[2]))
    keep = scored[0][0]
    delete = [pid for pid, _, _ in scored[1:]]
    return keep, delete


class _UnionFind:
    """Minimal union-find over paper_ids. Used to compose multiple dup
    strategies into connected components so keeper selection runs once per
    component rather than once per strategy. That removes the ordering bug
    where a keeper in strategy A could become a delete target in strategy B
    during the same run."""

    def __init__(self):
        self._parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self._parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # path compression
        cur = x
        while self._parent[cur] != root:
            nxt = self._parent[cur]
            self._parent[cur] = root
            cur = nxt
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def components(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for pid in self._parent:
            groups[self.find(pid)].append(pid)
        return groups


def merge_into(conn, keep_id: str, delete_id: str, apply: bool) -> str:
    """Merge delete_id into keep_id. Returns description of what happened.

    When apply=True, wraps the whole merge (row UPDATE, chunks/vec delete,
    row DELETE, re-embed) in a SAVEPOINT so a re-embed failure rolls the
    entire merge back. Previously the merge committed unconditionally and
    a re-embed raise only emitted a WARN, leaving a keep_id with stale
    vectors until some later backfill noticed.
    """
    # Self-merge guard. Helper must be safe to call independently of the
    # _pick_keeper contract; nothing inside this function otherwise blocks
    # keep_id == delete_id, and a self-merge would DELETE the survivor.
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
        return f"  SKIP: one entry not found"

    k_title, k_doi, k_pt, k_tt, k_pdf, k_abs, k_auth, k_year = keep
    d_title, d_doi, d_pt, d_tt, d_pdf, d_abs, d_auth, d_year = delete

    changes = []

    if not apply:
        return f"  would merge {delete_id} -> {keep_id}"

    # Open a savepoint so we can roll the whole merge back if the re-embed
    # step raises. The savepoint name includes the delete_id so nested
    # merges in the same connection don't collide.
    sp_name = "merge_" + re.sub(r"[^A-Za-z0-9]", "_", delete_id)[:40]
    conn.execute(f"SAVEPOINT {sp_name}")

    # Copy richer fields from delete into keep (only if keep's field is empty/short)
    updates = []
    params = []

    if d_doi and not k_doi:
        updates.append("doi = ?")
        params.append(d_doi)
        changes.append("doi")
    # Source-priority tiebreaker: higher-authority paper_id wins regardless of length.
    # Only take delete_id's text if (a) keep has none, OR (b) delete has higher authority
    # AND delete's text is at least 70% as long as keep's (guards against truncation).
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
            changes.append(f"processed_text (higher-source {delete_id[:10]}..., {len(d_pt):,} chars)")
        elif delete_prio == keep_prio and len(d_pt) > len(k_pt) * 1.1:
            # Same source tier, materially longer (10%+) → take delete's text.
            # Threshold was 30%; lowered so the richer delete-side text is
            # adopted before we drop its chunk corpus. Under the higher bar,
            # deletes in the 10-30% range silently lost both text and chunks.
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

    # Drop the delete-side paper's chunks entirely. Previously this path
    # did `UPDATE OR IGNORE paper_chunks SET paper_id = keep_id` to redirect
    # chunks onto the keeper. But `paper_chunks` has no UNIQUE(paper_id,
    # chunk_index), so the UPDATE never conflicts and the keeper ended up
    # carrying two chunk sets. Worse, delete-side vec_chunks were dropped
    # before the redirect, leaving the moved chunks with no corresponding
    # embeddings. The keeper's own chunk set (from its own processed_text /
    # tex_text) is authoritative; the delete-side chunks are redundant.
    # If processed_text or tex_text was merged below, _chunk_processed_text
    # further down rebuilds the chunk set from the newly-merged text.
    #
    # v15 trigger chunks_ad_vec automatically removes the corresponding
    # vec_chunks rows when we delete paper_chunks, so the manual vec_chunks
    # DELETE here is no longer required but is kept as a belt-and-suspenders
    # for pre-v15 DBs that haven't yet migrated.
    # Round-2 review CRITICAL fix: previously only cleared vec_chunks
    # (nomic). Once the qwen3 backfill landed, every dedup-merge left
    # orphan rows in vec_chunks_qwen3_<dim>. Use the centralized helpers
    # from server.py so all variants are cleaned uniformly.
    _delete_vec_chunks_for_paper(conn, delete_id)
    conn.execute("DELETE FROM paper_chunks WHERE paper_id = ?", (delete_id,))

    # Round-2 review CRITICAL fix: clear paper-level vec rows for ALL
    # variants (was only nomic vec_papers).
    try:
        row = conn.execute("SELECT rowid FROM papers WHERE paper_id = ?", (delete_id,)).fetchone()
        if row:
            _delete_vec_papers_for_paper(conn, row[0])
    except sqlite3.OperationalError:
        pass

    # Redirect paper_references (both directions)
    conn.execute(
        "UPDATE OR IGNORE paper_references SET citing_paper_id = ? WHERE citing_paper_id = ?",
        (keep_id, delete_id),
    )
    conn.execute(
        "UPDATE OR IGNORE paper_references SET cited_paper_id = ? WHERE cited_paper_id = ?",
        (keep_id, delete_id),
    )
    # Normalize cited_doi on edges we just retargeted. Without this, a
    # later `DELETE FROM papers WHERE paper_id = keep_id` could produce a
    # __doi placeholder row where cited_paper_id names the keeper DOI but
    # cited_doi carries the stale delete-side DOI. The v17 trigger reads
    # cited_doi first, so the resulting placeholder would be self-
    # contradictory. Fetch the keeper's DOI as it stands post-merge.
    keep_doi_now = conn.execute(
        "SELECT doi FROM papers WHERE paper_id = ?", (keep_id,)
    ).fetchone()
    if keep_doi_now and keep_doi_now[0]:
        conn.execute(
            "UPDATE paper_references SET cited_doi = ? "
            "WHERE cited_paper_id = ? AND (cited_doi IS NULL OR cited_doi != ?)",
            (keep_doi_now[0], keep_id, keep_doi_now[0]),
        )
    # Clean up any remaining refs pointing to deleted ID
    conn.execute("DELETE FROM paper_references WHERE citing_paper_id = ? OR cited_paper_id = ?", (delete_id, delete_id))
    # Clean self-referencing citations created by redirect
    conn.execute(
        "DELETE FROM paper_references WHERE citing_paper_id = ? AND cited_paper_id = ?",
        (keep_id, keep_id),
    )

    # Delete the duplicate entry
    conn.execute("DELETE FROM papers WHERE paper_id = ?", (delete_id,))

    # Keep has_full_text in sync with whatever text fields the keeper now holds.
    # Earlier versions of this merge path copied processed_text/tex_text into
    # the keeper without updating has_full_text, so a row with >500 chars of
    # text could stay stuck at has_full_text=0 and get excluded from chunk-mode
    # search_local.
    _recompute_has_full_text(conn, keep_id)

    # Re-chunk and re-embed the kept paper if any embed-affecting field changed.
    # _paper_embed_text in server.py consumes title, abstract, processed_text,
    # and tex_text — a change to any of them should trigger re-embedding.
    # Previously this guard only checked processed_text, leaving the paper-level
    # embedding stale after abstract/tex_text-only merges.
    #
    # Chunking policy: prefer tex_text when present (math-clean source), else
    # processed_text. Server-side _store_tex_text does the same, so dedup-
    # merged keepers end up with the same chunking as a fresh TeX ingest.
    #
    # Atomicity (H7): a re-embed failure ROLLBACKs the merge via the savepoint
    # opened above so the duplicate isn't already deleted while the canonical
    # keep_id is left with stale vectors.
    changes_joined = " ".join(changes)
    processed_changed = "processed_text" in changes_joined
    tex_changed = "tex_text" in changes_joined
    chunk_source_changed = processed_changed or tex_changed
    embed_inputs_changed = any(
        k in changes_joined
        for k in ("processed_text", "abstract", "tex_text")
    )
    # Guard: if the keeper has cite-ready body text but no chunks (which
    # happens when (a) the keeper never had chunks pre-merge, or (b) we
    # just dropped the delete-side chunks and did not re-chunk because
    # no text was copied), force a re-chunk. Without this, dedup can
    # silently strip the only chunk set off a paper.
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
                        # Rebuild passages from the merged text so passage-
                        # mode search and verify_claim reflect the new body.
                        _build_paper_passages(conn, keep_id, chunk_text)
                    # raise_on_error=True so an embed failure actually
                    # propagates up into the ROLLBACK branch below. Without
                    # it, the helpers swallow their own exceptions, the
                    # savepoint gets RELEASEd with stale vectors, and the
                    # merge commits anyway — defeating Phase 4's atomicity.
                    _embed_paper_if_verified(conn, keep_id, raise_on_error=True)
                    if chunk_source_changed:
                        _embed_chunks(conn, keep_id, raise_on_error=True)
                except Exception as e:
                    # Undo the merge so keep_id doesn't end up with stale
                    # vectors after the duplicate was already deleted.
                    try:
                        conn.execute(f"ROLLBACK TO {sp_name}")
                    except sqlite3.OperationalError:
                        pass
                    try:
                        conn.execute(f"RELEASE {sp_name}")
                    except sqlite3.OperationalError:
                        pass
                    return f"  ROLLED BACK merge {delete_id} -> {keep_id}: re-embed failed ({type(e).__name__}: {e})"

    try:
        conn.execute(f"RELEASE {sp_name}")
    except sqlite3.OperationalError:
        pass

    desc = f"  merged {delete_id} -> {keep_id}"
    if changes:
        desc += f" (copied: {', '.join(changes)})"
    return desc


_GENERIC_TITLES = frozenset({
    "introduction", "book reviews", "preface", "editorial", "conclusion",
    "discussion", "abstract", "acknowledgments", "methods", "results",
    "review", "commentary", "errata", "corrigendum", "reply", "response",
    "foreword", "appendix", "supplementary material", "letter to the editor",
    "letter", "correction", "retraction", "erratum", "note", "announcement",
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Actually merge (default: dry-run)")
    args = ap.parse_args()
    apply = args.apply

    print(f"[start] dedup papers.db  apply={apply}", file=sys.stderr)

    conn = _init_db()
    uf = _UnionFind()
    reasons: dict[tuple[str, str], list[str]] = defaultdict(list)

    def _union_with_reason(pids: list[str], reason: str) -> None:
        for p in pids:
            uf.add(p)
        for p in pids[1:]:
            uf.union(pids[0], p)
            key = tuple(sorted((pids[0], p)))
            reasons[key].append(reason)

    # ── Discovery 1a: Same DOI ───────────────────────────────────────
    print("\n=== Discovery 1a: Same DOI ===", file=sys.stderr)
    doi_rows = conn.execute("""
        SELECT LOWER(doi), GROUP_CONCAT(paper_id, '|||')
        FROM papers WHERE doi IS NOT NULL AND doi != ''
        GROUP BY LOWER(doi) HAVING COUNT(*) > 1
    """).fetchall()
    print(f"  {len(doi_rows)} DOI groups", file=sys.stderr)
    for doi, pid_str in doi_rows:
        _union_with_reason(pid_str.split("|||"), f"doi={doi}")

    # ── Discovery 1b: Same normalized title + compatible year ────────
    print("\n=== Discovery 1b: Same title + year ===", file=sys.stderr)
    title_rows = conn.execute("""
        SELECT LOWER(TRIM(title)), GROUP_CONCAT(paper_id, '|||'), GROUP_CONCAT(COALESCE(year,''), '|||')
        FROM papers WHERE title IS NOT NULL AND title != ''
        GROUP BY LOWER(TRIM(title)) HAVING COUNT(*) > 1
    """).fetchall()
    print(f"  {len(title_rows)} title groups", file=sys.stderr)
    for title, pid_str, year_str in title_rows:
        if title.lower() in _GENERIC_TITLES:
            continue
        pids = pid_str.split("|||")
        years = year_str.split("|||")
        unique_years = set(y for y in years if y)
        if len(unique_years) > 1:
            # Different editions; don't union title-only.
            continue
        _union_with_reason(pids, f"title='{title[:50]}'")

    # ── Discovery 1c: web: duplicates of paper entries ──────────────
    print("\n=== Discovery 1c: web: duplicates ===", file=sys.stderr)
    web_rows = conn.execute("""
        SELECT w.paper_id, LOWER(TRIM(w.title)), p.paper_id
        FROM papers w
        JOIN papers p ON LOWER(TRIM(w.title)) = LOWER(TRIM(p.title))
        WHERE w.paper_id LIKE 'web:%'
        AND p.paper_id NOT LIKE 'web:%'
        AND w.paper_id != p.paper_id
    """).fetchall()
    print(f"  {len(web_rows)} web duplicates", file=sys.stderr)
    for web_pid, title, paper_pid in web_rows:
        if title in _GENERIC_TITLES:
            continue
        _union_with_reason([paper_pid, web_pid], f"web-dup '{(title or '')[:50]}'")

    # ── Merge: one keeper per connected component ────────────────────
    components = uf.components()
    multi = {root: members for root, members in components.items() if len(members) > 1}
    print(f"\n=== Merge: {len(multi)} dup components ===", file=sys.stderr)
    total_merges = 0
    merge_count = 0
    for root, members in multi.items():
        if _shutdown:
            break
        # Re-verify every member still exists (something earlier in this
        # run could have deleted it, e.g. if we're re-entering after a
        # crash). `merge_into` also guards, but skipping up front keeps
        # log noise down.
        surviving = [
            pid for pid in members
            if conn.execute("SELECT 1 FROM papers WHERE paper_id = ?", (pid,)).fetchone()
        ]
        if len(surviving) < 2:
            continue
        keep, deletes = _pick_keeper(conn, surviving)
        comp_reasons = {
            r for a in surviving for b in surviving if a < b
            for r in reasons.get((a, b), [])
        }
        reason_summary = "; ".join(sorted(comp_reasons)[:3])
        for d in deletes:
            desc = merge_into(conn, keep, d, apply)
            print(f"  [{reason_summary}] {desc}", file=sys.stderr)
            merge_count += 1
        total_merges += len(deletes)
        if apply and merge_count % 50 == 0:
            conn.commit()

    if apply:
        conn.commit()

    # ── Report orphaned PDFs ──────────────────────────────────────────
    print(f"\n=== Orphaned PDFs ===", file=sys.stderr)
    db_paths = set()
    for row in conn.execute("SELECT local_pdf_path FROM papers WHERE local_pdf_path IS NOT NULL").fetchall():
        db_paths.add(row[0])

    orphaned = 0
    if PAPERS_DIR.exists():
        for pdf in PAPERS_DIR.glob("*.pdf"):
            if str(pdf) not in db_paths:
                orphaned += 1
    print(f"  {orphaned} PDFs in {PAPERS_DIR} not referenced by any DB entry", file=sys.stderr)

    conn.close()

    print(f"\n[done] total_merges={total_merges} apply={apply}", file=sys.stderr)
    if not apply and total_merges > 0:
        print(f"  Run with --apply to execute merges", file=sys.stderr)


if __name__ == "__main__":
    main()
