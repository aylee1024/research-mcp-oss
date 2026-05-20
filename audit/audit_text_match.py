# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Find local: papers whose processed_text does not match their title.

Uses a STRICT test: for each paper, extract distinctive title words (length >= 5,
lowercase, not common stopwords). If none of them appear in the first 2000 chars
of the text, flag as likely misassigned.

Scope: ALL local: papers with processed_text, regardless of whether they also
have a local_pdf_path. Earlier versions of this docstring claimed the script
was scoped to no-PDF papers; the SQL never enforced that. The current scope
is intentional — misassigned text can exist in either cohort, and --clear
preserves local_pdf_path so a recovery path remains.

Usage:
    uv run audit_text_match.py                # report
    uv run audit_text_match.py --clear        # clear flagged papers' text
                                              # (local_pdf_path is preserved)
"""

import argparse
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import _clear_paper_text
from research_mcp.paths import PAPERS_DB_PATH as DB_PATH  # noqa: E402

STOPWORDS = {
    "about", "above", "after", "again", "against", "among", "around", "because",
    "before", "being", "below", "between", "both", "could", "doing", "during",
    "each", "from", "further", "having", "here", "into", "itself", "more",
    "most", "myself", "once", "only", "other", "ourselves", "over", "same",
    "should", "some", "such", "than", "that", "their", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "under",
    "until", "upon", "very", "were", "what", "when", "where", "which", "while",
    "with", "within", "would", "your", "yours", "yourself", "above", "also",
    "many", "much", "like", "ones", "make", "just", "know", "take", "time",
    "year", "work", "study", "studies", "paper", "article", "review",
    "using", "used", "based", "among", "another", "between", "towards",
    "across", "being", "across", "including", "toward", "methods", "results",
    "data", "analysis", "research",
}


def _distinctive_words(title: str) -> list[str]:
    """Return title words of length >= 5, lowercase, alphanumeric, not stopwords."""
    tokens = re.findall(r"[A-Za-z]+", title.lower())
    out = []
    for t in tokens:
        if len(t) >= 5 and t not in STOPWORDS:
            out.append(t)
    return out


def _contains_any(text_lower: str, words: list[str]) -> list[str]:
    return [w for w in words if w in text_lower]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA busy_timeout = 120000")

    rows = conn.execute("""
        SELECT paper_id, title, processed_text
        FROM papers
        WHERE paper_id LIKE 'local:%'
        AND processed_text IS NOT NULL
        AND LENGTH(processed_text) > 500
        AND title IS NOT NULL
        AND LENGTH(title) > 5
    """).fetchall()

    print(f"[start] audit_text_match checking {len(rows)} papers", file=sys.stderr)

    flagged: list[tuple[str, str, str]] = []

    for pid, title, text in rows:
        distinctive = _distinctive_words(title)
        if len(distinctive) < 2:
            continue  # title too short / too common to verify
        text_head_lower = text[:2000].lower()
        matches = _contains_any(text_head_lower, distinctive)
        if not matches:
            # Extra strict: look for EXACT title phrase too
            title_phrase = title.lower().strip()
            if title_phrase not in text_head_lower:
                flagged.append((pid, title, text[:200]))

    print(f"\nFlagged {len(flagged)} papers where title words don't appear in text:\n", file=sys.stderr)
    for pid, title, head in flagged:
        print(f"  {pid}", file=sys.stderr)
        print(f"    title: {title[:80]}", file=sys.stderr)
        print(f"    text:  {head[:140].replace(chr(10), ' ')}", file=sys.stderr)
        print("", file=sys.stderr)

    if args.clear and flagged:
        pids_to_clear = [f[0] for f in flagged]
        # Route through _clear_paper_text so the clear is savepoint-wrapped,
        # has_full_text is recomputed from remaining text state, and the
        # chunks/vec/paper_chunks deletions all happen inside a single atomic
        # helper. The selective kwargs preserve any TeX-sourced text and the
        # local_pdf_path so the PDF can be reprocessed later.
        for pid in pids_to_clear:
            try:
                _clear_paper_text(
                    conn, pid,
                    clear_processed_text=True,
                    clear_tex_text=False,
                    clear_pdf_path=False,
                    clear_tex_path=False,
                )
            except Exception as e:
                print(f"  WARN clear failed for {pid}: {e}", file=sys.stderr)
        conn.commit()
        print(f"Cleared {len(pids_to_clear)} papers", file=sys.stderr)

    conn.close()


if __name__ == "__main__":
    main()
