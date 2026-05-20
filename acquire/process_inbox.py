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
"""Process PDFs from $INBOX_DIR into the research library.

For each PDF:
  1. Try to match against papers.db (by DOI from filename, title from filename,
     DOI from first 3 pages). Avoids Docling for papers already in the library.
  2. If matched + already has full text: just move PDF to Papers/ folder.
  3. If matched but no full text: run Docling, store text.
  4. If no match: run Docling, create new entry.

Usage:
    uv run process_inbox.py                   # process all
    uv run process_inbox.py --limit 50        # process first 50
    uv run process_inbox.py --dry-run         # report only
    uv run process_inbox.py --skip-docling    # only move matched PDFs, skip unmatched
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import sqlite3
import sys
from pathlib import Path

import fitz  # pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import (
    process_pdf,
    _process_pdf_and_verify,
    _init_db,
    _canonical_pdf_path,
    _canonical_pdf_name,
    DB_PATH,
    PAPERS_DIR,
)
from research_mcp.paths import INBOX_DIR as INBOX  # noqa: E402

_shutdown = False


def _on_signal(sig, frame):
    global _shutdown
    _shutdown = True
    print("\n[signal] Finishing current PDF...", file=sys.stderr)


signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_DOI_FILENAME_PAT = re.compile(
    r"10[._\-](\d{4,5})[._\-/\s](.+?)(?:\.pdf$|$)", re.IGNORECASE
)

# Generic dash-separated catalog filename: "Author - Title (Year[, extra])".
# The author/title/year span is captured up to the first ")" closing the
# year-extra group. Anything after that ")" (a trailing source tag, a
# .pdf extension, hashes, etc.) is ignored. The pattern is intentionally
# loose: when it doesn't match, downstream code falls back to
# whole-filename-as-title and the METADATA_NEEDED protocol asks an LLM
# sub-agent to recover author/year from the PDF text.
_CATALOG_FILENAME_PAT = re.compile(
    r"^(?P<author>.+?)\s*-\s*(?P<title>.+?)\s*\((?P<year>\d{4})(?:,\s*[^)]+)?\)",
)


def _doi_from_filename(stem: str) -> str | None:
    """Extract DOI from a PDF filename."""
    m = _DOI_FILENAME_PAT.search(stem)
    if not m:
        return None
    prefix = m.group(1)
    suffix = m.group(2).strip()
    # Replace filename separators with DOI separators
    suffix = re.sub(r"[_\s]", "/", suffix)
    doi = f"10.{prefix}/{suffix}"
    # Clean trailing junk
    doi = re.sub(r"[/._\-]+$", "", doi)
    return doi


def _title_from_filename(stem: str) -> tuple[str, str, str | None]:
    """Extract (author, title, year) from a dash-separated catalog filename.

    Matches the common pattern `Author - Title (Year[, extra])`.
    Anything after the closing `)` is ignored, so trailing source tags
    like ` - some-source` do not corrupt the title. Returns
    `("", whole_stem, None)` when the filename doesn't conform; the
    caller's METADATA_NEEDED path handles recovery via the PDF text.
    """
    m = _CATALOG_FILENAME_PAT.match(stem)
    if m:
        author = m.group("author").strip()
        title = m.group("title").strip().replace("_", " ")
        year = m.group("year")
        return author, title, year
    return "", stem.replace("_", " ").replace("-", " "), None


# ---------------------------------------------------------------------------
# DOI extraction from PDF pages (lightweight, no Docling)
# ---------------------------------------------------------------------------

_DOI_LABELED_PAT = re.compile(
    r"(?:doi|DOI|https?://(?:dx\.)?doi\.org/)[:\s]*(10\.\d{4,9}/\S+)"
)
_DOI_BARE_PAT = re.compile(r"\b(10\.\d{4,9}/\S+)")
# See server._clean_doi_suffix: trailing set omits `;` `,` `:` so SICI DOIs
# keep their internal punctuation. `>` is a trailing-only strip (SICI has
# internal `>` followed by more chars, so it's never rstripped).
_DOI_TRAILING_PUNCT = ".\"')>]"


def _clean_doi_suffix(doi: str) -> str:
    return doi.rstrip(_DOI_TRAILING_PUNCT)


def _doi_from_pdf_pages(path: Path, max_pages: int = 3) -> str | None:
    """Extract the paper's own DOI from the first N pages of a PDF (fast, no Docling).

    Prefers labeled DOIs (`doi:`, `DOI:`, `doi.org/`) because those almost always
    mark the paper's own DOI. Falls back to most-frequent DOI in text (paper's
    DOI typically recurs in header/footer while citation DOIs appear once).
    Last resort: last unlabeled DOI in text (paper's own usually precedes
    references).

    Stop-set on the DOI regex uses only whitespace + `<`/`>`/`"` so SICI-style
    DOIs like 10.1130/0091-7613(1999)027<0359 and DOIs containing `;` or
    `,` aren't truncated. Trailing sentence punctuation is peeled off after
    extraction.

    Window is 50K chars to match server._verify_paper_text_match — a paper
    whose own DOI lives past page 1 frontmatter (book-length or heavy preprint
    title pages) otherwise missed matching here."""
    try:
        doc = fitz.open(str(path))
        text = ""
        for i in range(min(max_pages, len(doc))):
            text += doc[i].get_text()
        doc.close()
    except Exception:
        return None

    head = text[:50000]
    m = _DOI_LABELED_PAT.search(head)
    if m:
        return _clean_doi_suffix(m.group(1))

    unlabeled = _DOI_BARE_PAT.findall(head)
    if not unlabeled:
        return None
    cleaned = [_clean_doi_suffix(d) for d in unlabeled if len(d) > 10]
    if not cleaned:
        return None
    from collections import Counter
    counts = Counter(cleaned)
    most_common, count = counts.most_common(1)[0]
    if count > 1:
        return most_common
    return cleaned[-1]


# ---------------------------------------------------------------------------
# DB matching
# ---------------------------------------------------------------------------

def _find_in_db(conn: sqlite3.Connection, doi: str | None, title: str, author: str) -> tuple[str | None, bool]:
    """Try to find a matching paper in the DB. Returns (paper_id, has_full_text).

    has_full_text is True when the paper has cite-ready body text in EITHER
    processed_text or tex_text (>500 chars) OR the has_full_text flag is set
    (covers schema-consistent rows that Phase 2 may have written after a
    repair). Previously only processed_text was checked, which forced a
    needless re-Docling on tex-only papers and, combined with the pre-Phase-2
    H8 bug in _store_tex_text, could also flip has_full_text from 1 to 0.
    """
    # By DOI (exact). If multiple rows share the same DOI (duplicates pending
    # dedup_papers.py), prefer the one that already has full text so the
    # inbox path doesn't flip between SKIP and PROCESS depending on which
    # duplicate SQLite happens to return first.
    if doi:
        row = conn.execute(
            """
            SELECT paper_id,
                   CASE WHEN LENGTH(COALESCE(processed_text,'')) > 500
                         OR LENGTH(COALESCE(tex_text,'')) > 500
                         OR has_full_text = 1
                   THEN 1 ELSE 0 END AS has_text
            FROM papers
            WHERE LOWER(doi) = LOWER(?)
            ORDER BY has_text DESC, paper_id ASC
            LIMIT 1
            """,
            (doi,),
        ).fetchone()
        if row:
            return row[0], bool(row[1])

    # By title — require BIDIRECTIONAL containment on a substantial substring, plus
    # author verification when an author was extracted from the filename.
    # A one-way LIKE (`DB CONTAINS inbox title`) false-matches when a short inbox
    # filename is a substring of an unrelated longer DB title.
    # ORDER BY has_text DESC so duplicate-title matches prefer the full-text
    # row — same fix as the DOI path above.
    if title and len(title) > 30:
        search_title = title[:60].strip()
        candidates = conn.execute(
            """
            SELECT paper_id, title, authors,
                   CASE WHEN LENGTH(COALESCE(processed_text,'')) > 500
                         OR LENGTH(COALESCE(tex_text,'')) > 500
                         OR has_full_text = 1
                   THEN 1 ELSE 0 END AS has_text
            FROM papers
            WHERE LOWER(title) LIKE LOWER(?)
            ORDER BY has_text DESC, paper_id ASC
            LIMIT 10
            """,
            (f"%{search_title}%",),
        ).fetchall()
        inbox_words = set(re.findall(r"[a-z]{4,}", title.lower()))
        author_lower = (author or "").lower().strip()
        for pid, db_title, db_authors, has_text in candidates:
            db_title_lower = (db_title or "").lower()
            db_words = set(re.findall(r"[a-z]{4,}", db_title_lower))
            if not inbox_words:
                continue
            # Bidirectional Jaccard: require high overlap in both directions to
            # avoid false positives like
            # "Introduction to International Law" matching
            # "Introduction to International Law and Human Rights"
            # (fwd=1.0, rev=0.6 — was accepted under the 0.6/0.6 floor).
            shared = inbox_words & db_words
            if not shared:
                continue
            fwd = len(shared) / len(inbox_words)
            rev = len(shared) / len(db_words) if db_words else 0.0
            if fwd < 0.75 or rev < 0.75:
                continue
            # Author check.
            # - If filename yielded an author, require at least one token to
            #   appear in db.authors. (Also require db.authors non-empty.)
            # - If no author from filename (pattern didn't match), still require db.authors
            #   non-empty and the title match to be nearly exact (>=0.85 both
            #   directions) — prevents accepting bare-title matches on
            #   potentially-distinct papers.
            if author_lower:
                if not db_authors:
                    continue
                author_tokens = set(re.findall(r"[a-z]{3,}", author_lower))
                db_authors_lower = db_authors.lower()
                if author_tokens and not any(tok in db_authors_lower for tok in author_tokens):
                    continue
            else:
                if fwd < 0.85 or rev < 0.85:
                    continue
            return pid, bool(has_text)

    return None, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Max PDFs to process (0=all)")
    ap.add_argument("--skip-docling", action="store_true", help="Only move matched PDFs, skip unmatched")
    args = ap.parse_args()

    if not INBOX.exists():
        print(f"[error] Inbox not found: {INBOX}", file=sys.stderr)
        sys.exit(1)

    pdfs = sorted(INBOX.glob("*.pdf"))
    if args.limit > 0:
        pdfs = pdfs[:args.limit]

    print(f"[start] inbox={len(pdfs)} PDFs  dry_run={args.dry_run}  skip_docling={args.skip_docling}", file=sys.stderr)

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    conn = _init_db()

    matched_skip = 0      # Already in DB with full text — just move
    matched_process = 0   # In DB but no text — run Docling
    new_process = 0       # Not in DB — run Docling
    skipped_docling = 0   # Skipped because --skip-docling
    errors = 0

    for i, pdf in enumerate(pdfs):
        if _shutdown:
            print(f"  [signal] stopped at {i}/{len(pdfs)}", file=sys.stderr)
            break

        stem = pdf.stem
        label = f"[{i+1}/{len(pdfs)}]"

        # Step 1: Try DOI from filename
        doi = _doi_from_filename(stem)
        author, title, year = _title_from_filename(stem)

        # Step 2: Check DB
        paper_id, has_text = _find_in_db(conn, doi, title, author)

        # Step 3: If no match from filename, try DOI from PDF pages
        if not paper_id:
            pdf_doi = _doi_from_pdf_pages(pdf, max_pages=3)
            if pdf_doi:
                doi = pdf_doi
                paper_id, has_text = _find_in_db(conn, doi, "", "")

        if paper_id and has_text:
            # Already in DB with full text — just move PDF
            print(f"  {label} SKIP (in DB): {stem[:60]}", file=sys.stderr)
            if not args.dry_run:
                # Move to Papers/ if not already there
                if pdf.parent != PAPERS_DIR:
                    canonical = _canonical_pdf_path(conn, paper_id)
                    if not canonical.exists():
                        shutil.move(str(pdf), str(canonical))
                        conn.execute(
                            "UPDATE papers SET local_pdf_path = ? WHERE paper_id = ?",
                            (str(canonical), paper_id),
                        )
                        conn.commit()
                    else:
                        # Canonical path already has a file — just delete inbox copy
                        pdf.unlink()
            matched_skip += 1

        elif paper_id and not has_text:
            # In DB but no text — needs Docling. Route through the shared
            # verify gate so _doi_from_pdf_pages returning a foreign cited DOI
            # cannot silently bind this PDF to the wrong row. The verify step
            # checks the stored processed_text against the filename-derived
            # doi + title; on mismatch, text and local_pdf_path are cleared
            # from the wrongly-bound row.
            #
            # Round-1 verifier-review MED fix (codex 1): when matched to a
            # DB row, prefer DB authors over filename-extracted author.
            # Filename gives one surname; DB has full author list and is
            # more reliable, especially when the filename pattern didn't
            # yield an author (`author` is None).
            db_authors = None
            try:
                row = conn.execute(
                    "SELECT authors FROM papers WHERE paper_id = ?", (paper_id,)
                ).fetchone()
                if row and row[0]:
                    db_authors = row[0]
            except Exception:
                pass
            target_authors = db_authors or (author or None)
            print(f"  {label} PROCESS (no text): {stem[:60]}", file=sys.stderr)
            if args.skip_docling:
                skipped_docling += 1
                continue
            if not args.dry_run:
                try:
                    result, chars, ok = await _process_pdf_and_verify(
                        str(pdf), paper_id=paper_id,
                        target_doi=doi or None,
                        target_title=title or None,
                        target_authors=target_authors,
                    )
                    if ok:
                        print(f"         → {chars} chars", file=sys.stderr)
                    else:
                        print(f"         → {result[:80]}", file=sys.stderr)
                        errors += 1 if result.startswith("VERIFY_REJECTED") else 0
                except Exception as e:
                    print(f"         error: {e}", file=sys.stderr)
                    errors += 1
            matched_process += 1

        else:
            # Not in DB — new paper. Still run verify when the filename gave
            # us a title/doi; process_pdf's _resolve_paper_id can bind to an
            # existing row via DOI-from-text, and we want to catch misbinds.
            print(f"  {label} NEW: {stem[:60]}", file=sys.stderr)
            if args.skip_docling:
                skipped_docling += 1
                continue
            if not args.dry_run:
                try:
                    result, chars, ok = await _process_pdf_and_verify(
                        str(pdf), paper_id="",
                        target_doi=doi or None,
                        target_title=title or None,
                        target_authors=author or None,
                    )
                    if ok:
                        print(f"         → {chars} chars", file=sys.stderr)
                    else:
                        print(f"         → {result[:80]}", file=sys.stderr)
                        errors += 1 if result.startswith("VERIFY_REJECTED") else 0
                except Exception as e:
                    print(f"         error: {e}", file=sys.stderr)
                    errors += 1
            new_process += 1

        # Progress
        if (i + 1) % 50 == 0:
            conn.commit()
            print(
                f"  progress: {i+1}/{len(pdfs)}  skip={matched_skip} "
                f"process={matched_process} new={new_process} err={errors}",
                file=sys.stderr,
            )

    conn.commit()
    conn.close()

    print(
        f"\n[done] skip={matched_skip} process_existing={matched_process} "
        f"new={new_process} skipped_docling={skipped_docling} errors={errors}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    asyncio.run(main())
