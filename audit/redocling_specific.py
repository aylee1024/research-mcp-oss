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
#     "pymupdf",
# ]
# ///
"""Re-Docling specific paper_ids from their local_pdf_path."""

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from server import _init_db, process_pdf


PAPER_IDS = [
    "oa:W2510708214",
    "oa:W2792021696",
    "oa:W4413043087",
    "local:Bidirectional_relationship_between_sleep_and_Alzheimers__10.1038_s41386-019-0478-5",
    "local:Asif Siddiqi - Challenge to Apollo - The Soviet Union and the Space Race, 1945-1974-NASA (2000)",
    "local:Byers_Boley_2023_Who_Owns_Outer_Space",
    "local:Michael Byers, Aaron Boley - Who Owns Outer Space__ International Law, Astrophysics, and the Sustainable Development of Space (Cambridge Studies in International and Comparative Law, Series Number 176-1",
]


async def main():
    conn = _init_db()
    for pid in PAPER_IDS:
        # Round-3 verifier-review fix (codex 1 LOW): fetch title/doi/authors
        # for the verify gate. Previously this script bypassed the verifier.
        row = conn.execute(
            "SELECT local_pdf_path, title, doi, authors FROM papers WHERE paper_id = ?", (pid,)
        ).fetchone()
        if not row or not row[0] or not Path(row[0]).is_file():
            print(f"SKIP {pid}: no valid PDF", file=sys.stderr)
            continue
        local_pdf_path, t, d, au = row
        try:
            result = await process_pdf(
                str(local_pdf_path), paper_id=pid,
                verify_doi=(d or None),
                verify_title=(t or None),
                verify_authors=(au or None),
            )
            if result.startswith("Processed:"):
                m = re.search(r"Extracted:\s*([\d,]+)\s*chars", result)
                chars = m.group(1) if m else "?"
                print(f"OK {pid}: {chars} chars", file=sys.stderr)
            else:
                print(f"FAIL {pid}: {result[:80]}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR {pid}: {e}", file=sys.stderr)
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
