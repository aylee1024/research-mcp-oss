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
"""DISABLED: Fix Docling's broken-ligature pattern.

!!! DO NOT RUN. The regex pattern `(word) (fi|fl) (word)` corrupts legitimate
    compound constructions like "orbital flight" -> "orbitalflight" or
    "methodological flaws" -> "methodologicalflaws". This was reverted 2026-04-16.

A safe ligature fix would require a dictionary-based approach: only merge if the
joined result is a known English word. Left here as a reference for what not to do.
Ligature-corrupt data is now in the DB with "word fi word" / "word fl word" form
(e.g., "scienti fi c", "bene fi ts", "re fl ect") until a dictionary-based fixer is
written.

Also handles Docling's letter-spaced small-caps author names like `C HARLES A. C ZEISLER`.

Usage:
    uv run fix_ligatures.py              # dry-run (report + diff samples)
    uv run fix_ligatures.py --apply      # execute
"""

import argparse
import re
import signal
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import _embed_chunks
from research_mcp.paths import PAPERS_DB_PATH as DB_PATH  # noqa: E402

# Broken ligature patterns:
# word1 fi word2  -> word1+fi+word2  (e.g., "scienti fi c" -> "scientific")
# word1 fl word2  -> word1+fl+word2  (e.g., "re fl ect" -> "reflect")
# fi word         -> fi+word         (e.g., "fi rst", "fi nding", "fi fty")
LIGATURE_PAT = re.compile(r"\b([a-z]+) (fi|fl) ([a-z]+)\b")
LIGATURE_START = re.compile(r"(?<![a-zA-Z])(fi|fl) ([a-z]{2,})\b")

# Common English words that legitimately have a space between "fi" or "fl" and adjacent
# letters. We reconstruct only if the joined result is in our affirm-list OR follows
# a plausible-English shape. For conservatism we require minimum lengths.
_SAFE_JOIN_MIN_PREFIX = 2  # 'sci' from 'scientific' -> wait that's 3. 're' is 2, from 'reflect'. OK 2.
_SAFE_JOIN_MIN_SUFFIX = 1  # 'c' from 'scientific' is 1.
_SAFE_JOIN_MIN_TOTAL = 5   # reconstructed word must be at least 5 chars to reduce false positives.

# Blacklist: combinations that should NOT be joined (risk of false positive).
_JOIN_BLACKLIST = {
    "afit",  # e.g., "a fit" as two words shouldn't collapse
    "afile",  # "a file" shouldn't collapse
    "ofit",  # "o fit"
    "ofile",
    "iflu",  # may appear in short forms
    "ofi",
}

_shutdown = False


def _on_signal(sig, frame):
    global _shutdown
    _shutdown = True
    print("\n[signal] Stopping...", file=sys.stderr)


signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


def _fix_ligature(text: str) -> tuple[str, int]:
    """Return (fixed_text, replacements_count)."""
    changes = 0

    def repl_mid(m: re.Match) -> str:
        nonlocal changes
        prefix, lig, suffix = m.group(1), m.group(2), m.group(3)
        if len(prefix) < _SAFE_JOIN_MIN_PREFIX or len(suffix) < _SAFE_JOIN_MIN_SUFFIX:
            return m.group(0)
        joined = prefix + lig + suffix
        if len(joined) < _SAFE_JOIN_MIN_TOTAL:
            return m.group(0)
        if joined.lower() in _JOIN_BLACKLIST:
            return m.group(0)
        changes += 1
        return joined

    def repl_start(m: re.Match) -> str:
        nonlocal changes
        lig, suffix = m.group(1), m.group(2)
        joined = lig + suffix
        changes += 1
        return joined

    fixed = LIGATURE_PAT.sub(repl_mid, text)
    fixed = LIGATURE_START.sub(repl_start, fixed)
    return fixed, changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA busy_timeout = 120000")

    # papers.processed_text
    rows_processed = conn.execute("""
        SELECT paper_id, processed_text FROM papers
        WHERE processed_text IS NOT NULL
        AND (processed_text LIKE '% fi %' OR processed_text LIKE '% fl %')
    """).fetchall()
    print(f"[papers] {len(rows_processed)} candidates for processed_text fix", file=sys.stderr)

    paper_changes = []
    total_paper_reps = 0
    for pid, text in rows_processed:
        fixed, count = _fix_ligature(text)
        if count > 0:
            paper_changes.append((pid, fixed, count))
            total_paper_reps += count
    print(f"[papers] {len(paper_changes)} papers will change, {total_paper_reps} replacements total", file=sys.stderr)

    # paper_chunks.chunk_text
    rows_chunks = conn.execute("""
        SELECT chunk_id, chunk_text, paper_id FROM paper_chunks
        WHERE chunk_text LIKE '% fi %' OR chunk_text LIKE '% fl %'
    """).fetchall()
    print(f"[chunks] {len(rows_chunks)} candidates for chunk_text fix", file=sys.stderr)

    chunk_changes = []
    total_chunk_reps = 0
    touched_paper_ids: set[str] = set()
    for cid, text, cpid in rows_chunks:
        fixed, count = _fix_ligature(text)
        if count > 0:
            chunk_changes.append((cid, fixed, count))
            total_chunk_reps += count
            touched_paper_ids.add(cpid)
    print(f"[chunks] {len(chunk_changes)} chunks will change, {total_chunk_reps} replacements total", file=sys.stderr)

    if not args.apply:
        # Sample some diffs
        print("\n-- Sample paper diffs --", file=sys.stderr)
        for pid, fixed, count in paper_changes[:3]:
            original = conn.execute("SELECT processed_text FROM papers WHERE paper_id = ?", (pid,)).fetchone()[0]
            # Find and print 3 example substitutions
            for m in LIGATURE_PAT.finditer(original):
                prefix, lig, suffix = m.group(1), m.group(2), m.group(3)
                if len(prefix) >= _SAFE_JOIN_MIN_PREFIX and len(suffix) >= _SAFE_JOIN_MIN_SUFFIX:
                    joined = prefix + lig + suffix
                    if len(joined) >= _SAFE_JOIN_MIN_TOTAL and joined.lower() not in _JOIN_BLACKLIST:
                        print(f"  {pid}: '{m.group(0)}' -> '{joined}'", file=sys.stderr)
                        break
        conn.close()
        return

    done_papers = 0
    for pid, fixed, _ in paper_changes:
        if _shutdown:
            break
        conn.execute("UPDATE papers SET processed_text = ? WHERE paper_id = ?", (fixed, pid))
        conn.execute(
            "UPDATE papers SET has_full_text = CASE"
            " WHEN LENGTH(COALESCE(processed_text,''))>500"
            " OR LENGTH(COALESCE(tex_text,''))>500 THEN 1 ELSE 0 END"
            " WHERE paper_id = ?"
            " AND has_full_text != CASE"
            " WHEN LENGTH(COALESCE(processed_text,''))>500"
            " OR LENGTH(COALESCE(tex_text,''))>500 THEN 1 ELSE 0 END",
            (pid,),
        )
        done_papers += 1
        if done_papers % 100 == 0:
            conn.commit()
            print(f"  papers: {done_papers}/{len(paper_changes)}", file=sys.stderr)
    conn.commit()

    done_chunks = 0
    for cid, fixed, _ in chunk_changes:
        if _shutdown:
            break
        conn.execute("UPDATE paper_chunks SET chunk_text = ? WHERE chunk_id = ?", (fixed, cid))
        done_chunks += 1
        if done_chunks % 1000 == 0:
            conn.commit()
            print(f"  chunks: {done_chunks}/{len(chunk_changes)}", file=sys.stderr)
    conn.commit()

    # Re-embed touched chunks so vec_chunks stays in sync with chunk_text
    print(f"[re-embed] re-embedding chunks across {len(touched_paper_ids)} papers", file=sys.stderr)
    reembed_done = 0
    for cpid in touched_paper_ids:
        if _shutdown:
            break
        try:
            _embed_chunks(conn, cpid)
        except Exception as e:
            print(f"  WARN re-embed failed for {cpid}: {e}", file=sys.stderr)
        reembed_done += 1
        if reembed_done % 100 == 0:
            conn.commit()
            print(f"  re-embed: {reembed_done}/{len(touched_paper_ids)}", file=sys.stderr)
    conn.commit()
    conn.close()

    print(f"\n[done] papers_updated={done_papers} chunks_updated={done_chunks} papers_reembedded={reembed_done}", file=sys.stderr)


if __name__ == "__main__":
    main()
