# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""One-time migration: consolidate all PDFs into $PAPERS_DIR with canonical naming.

Use after upgrading from an earlier layout, or to normalize an existing library
where PDFs are scattered across multiple directories. Reads paths from
research_mcp.paths so you can target a different PAPERS_DIR via env var.
"""

import json
import os
import re
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_mcp.paths import PAPERS_DB_PATH as DB_PATH, PAPERS_DIR  # noqa: E402


def _canonical_pdf_name(title: str, authors_json: str | None, year: int | None) -> str:
    author = "Unknown"
    if authors_json:
        try:
            authors = json.loads(authors_json)
            if authors and isinstance(authors[0], str) and authors[0]:
                parts = authors[0].split()
                author = parts[-1] if parts else "Unknown"
        except (json.JSONDecodeError, TypeError):
            pass
    author = re.sub(r'[^\w]', '', author)[:30] or "Unknown"
    yr = str(year) if year else "undated"
    safe_title = re.sub(r'[^\w\s]', '', title or "untitled")[:80].strip()
    safe_title = re.sub(r'\s+', '_', safe_title)
    if not safe_title:
        safe_title = "untitled"
    return f"{author}_{yr}_{safe_title}.pdf"


def main():
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT paper_id, local_pdf_path, title, authors, year FROM papers "
        "WHERE local_pdf_path IS NOT NULL AND local_pdf_path != ''"
    ).fetchall()

    print(f"Total papers with local_pdf_path: {len(rows)}")

    copied = 0
    orphaned = 0
    already_there = 0
    updates = []  # (new_path_or_None, paper_id)

    for paper_id, old_path, title, authors_json, year in rows:
        old_file = Path(old_path)

        # Already in canonical dir
        if old_file.parent.resolve() == PAPERS_DIR.resolve() and old_file.exists():
            already_there += 1
            continue

        if not old_file.exists():
            updates.append((None, paper_id))
            orphaned += 1
            continue

        # Generate canonical name
        filename = _canonical_pdf_name(title or "", authors_json, year)
        new_path = PAPERS_DIR / filename

        counter = 1
        while new_path.exists():
            new_path = PAPERS_DIR / f"{filename[:-4]}_{counter}.pdf"
            counter += 1

        shutil.copy2(str(old_file), str(new_path))
        updates.append((str(new_path), paper_id))
        copied += 1

        if copied % 100 == 0:
            print(f"  Copied {copied}...")

    # Apply DB updates in single transaction
    for new_path, paper_id in updates:
        if new_path is None:
            conn.execute("UPDATE papers SET local_pdf_path = NULL WHERE paper_id = ?", (paper_id,))
        else:
            conn.execute(
                "UPDATE papers SET local_pdf_path = ?, last_updated = datetime('now') WHERE paper_id = ?",
                (new_path, paper_id),
            )
    conn.commit()
    conn.close()

    print(f"\nDone:")
    print(f"  Copied: {copied}")
    print(f"  Already in place: {already_there}")
    print(f"  Orphaned (set to NULL): {orphaned}")
    print(f"  Total files in {PAPERS_DIR}: {len(list(PAPERS_DIR.glob('*.pdf')))}")


if __name__ == "__main__":
    main()
