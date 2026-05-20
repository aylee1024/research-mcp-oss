# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Build an acquire_batch.py input list from local: papers with NULL processed_text.

Extracts DOI and title from each orphaned entry. Also deletes the empty stub records
so acquire_batch.py creates fresh entries without collisions.

Usage:
    uv run build_reacquire_list.py                 # dry-run (report)
    uv run build_reacquire_list.py --apply         # delete stubs, write JSON
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_mcp.paths import PAPERS_DB_PATH as DB_PATH, RESEARCH_MCP_HOME  # noqa: E402

OUT_JSON = RESEARCH_MCP_HOME / "acquire-reacquire-cleared.json"

_DOI_IN_ID_PAT = re.compile(r"10[._\-/](\d{4,5})[._\-/](.+)$")


def _doi_from_paper_id(pid: str) -> str | None:
    stem = pid[len("local:"):] if pid.startswith("local:") else pid
    m = _DOI_IN_ID_PAT.search(stem)
    if m:
        doi = f"10.{m.group(1)}/{m.group(2)}"
        return re.sub(r"[/._\-]+$", "", doi)
    return None


def _clean_title(t: str) -> str:
    return t.replace("_", " ").replace("-", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA busy_timeout = 120000")

    rows = conn.execute("""
        SELECT paper_id, title, doi
        FROM papers
        WHERE paper_id LIKE 'local:%'
        AND processed_text IS NULL
        AND (tex_text IS NULL OR LENGTH(tex_text) < 500)
    """).fetchall()

    print(f"[start] found {len(rows)} NULL-text local: papers", file=sys.stderr)

    sources = []
    for pid, title, doi in rows:
        extracted_doi = doi or _doi_from_paper_id(pid)
        clean_title = _clean_title(title) if title else ""
        # Skip entries with no resolvable identifier AND no meaningful title
        if not extracted_doi and len(clean_title) < 15:
            continue
        sources.append({
            "type": "paper",
            "title": clean_title or pid,
            "s2_id": None,
            "doi": extracted_doi,
            "source_file": "reacquire-cleared",
            "original_paper_id": pid,
        })

    print(f"[build] {len(sources)} papers resolved for re-acquisition", file=sys.stderr)

    if not args.apply:
        print(f"\nWould write: {OUT_JSON}", file=sys.stderr)
        for s in sources[:5]:
            print(f"  {s['original_paper_id']}", file=sys.stderr)
            print(f"    title: {s['title'][:70]}", file=sys.stderr)
            print(f"    doi: {s['doi']}", file=sys.stderr)
        if len(sources) > 5:
            print(f"  ... and {len(sources) - 5} more", file=sys.stderr)
        conn.close()
        return

    # Delete empty stubs so acquire creates fresh entries
    pids_to_delete = [s["original_paper_id"] for s in sources]
    for pid in pids_to_delete:
        conn.execute("DELETE FROM paper_references WHERE citing_paper_id = ? OR cited_paper_id = ?", (pid, pid))
        try:
            conn.execute(
                "DELETE FROM vec_chunks WHERE rowid IN ("
                "SELECT chunk_id FROM paper_chunks WHERE paper_id = ?)",
                (pid,),
            )
        except sqlite3.OperationalError:
            pass
        conn.execute("DELETE FROM paper_chunks WHERE paper_id = ?", (pid,))
        row = conn.execute("SELECT rowid FROM papers WHERE paper_id = ?", (pid,)).fetchone()
        if row:
            try:
                conn.execute("DELETE FROM vec_papers WHERE rowid = ?", (row[0],))
            except sqlite3.OperationalError:
                pass
        conn.execute("DELETE FROM papers WHERE paper_id = ?", (pid,))
    conn.commit()
    print(f"[delete] removed {len(pids_to_delete)} empty stub entries", file=sys.stderr)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "chunk_id": "reacquire-cleared",
        "sources": sources,
    }, indent=2))
    print(f"[write] {OUT_JSON}", file=sys.stderr)

    conn.close()


if __name__ == "__main__":
    main()
