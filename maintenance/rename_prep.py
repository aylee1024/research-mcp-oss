"""Prep script: split Unknown_undated papers into 20 batches for parallel renaming agents."""

import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_mcp.paths import PAPERS_DB_PATH as DB_PATH  # noqa: E402

OUTPUT_DIR = Path("/tmp/rename_batches")
OUTPUT_DIR.mkdir(exist_ok=True)

conn = sqlite3.connect(str(DB_PATH))
rows = conn.execute("""
    SELECT paper_id, local_pdf_path, title,
           substr(processed_text, 1, 2000)
    FROM papers
    WHERE local_pdf_path LIKE '%Unknown_undated%'
    ORDER BY paper_id
""").fetchall()
conn.close()

print(f"Total papers: {len(rows)}")

# Split into 20 batches
num_batches = 20
batch_size = (len(rows) + num_batches - 1) // num_batches

for i in range(num_batches):
    start = i * batch_size
    end = min(start + batch_size, len(rows))
    if start >= len(rows):
        break
    batch = [
        {
            "paper_id": r[0],
            "local_pdf_path": r[1],
            "db_title": r[2] or "",
            "text_snippet": r[3] or "",
        }
        for r in rows[start:end]
    ]
    out_file = OUTPUT_DIR / f"batch_{i:02d}.json"
    with open(out_file, "w") as f:
        json.dump(batch, f, indent=2)
    print(f"Batch {i:02d}: {len(batch)} papers -> {out_file}")

print(f"\nReady: {OUTPUT_DIR}")
