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
"""Detect and populate `pdf_page_offset` + `pages_verified` for cite-ready papers.

For each `has_full_text=1` paper, split processed_text on Docling's
<!-- page N --> markers. For each of the first 40 physical pages,
scan the tail of the page's text for a trailing standalone integer
(the printed page number in the footer).

Calibrate the offset: offset k = physical_page - printed_page, chosen
as the value consistent across at least 3 consecutive pages. Set
pages_verified = 1 only when that consistency check passes; otherwise
leave pages_verified = 0 (the existing default) so the search layer
labels results as pdf_page rather than printed_page.

Usage:
    uv run backfill_page_offsets.py              # dry-run: report coverage
    uv run backfill_page_offsets.py --apply      # execute writes
"""

import argparse
import re
import signal
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_mcp.paths import PAPERS_DB_PATH as DB_PATH  # noqa: E402

PAGE_MARKER = re.compile(r"<!--\s*page\s+(-?\d+)\s*-->")
# Standalone integer at end of text. Allow some trailing punctuation.
# Avoid years (4 digits 1800-2099) and very large numbers.
TRAILING_NUM = re.compile(r"(?<!\d)(\d{1,4})\s*[.)\]]*\s*$")

MIN_CONSECUTIVE_RUN = 3
MAX_PHYSICAL_PAGES_TO_SCAN = 40

_shutdown = False


def _on_signal(sig, frame):
    global _shutdown
    _shutdown = True
    print("\n[signal] Finishing current paper...", file=sys.stderr)


signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


def _page_tail_number(page_text: str) -> int | None:
    """Return the trailing standalone integer in the last ~200 chars, or None.
    Excludes years (common false positive)."""
    tail = page_text.rstrip()[-200:]
    # Strip obvious post-number garbage (like PDF artifacts); look at last ~3 non-empty lines.
    last_lines = [ln.strip() for ln in tail.split("\n") if ln.strip()]
    for line in reversed(last_lines[-3:]):
        m = TRAILING_NUM.search(line)
        if m:
            n = int(m.group(1))
            # Reject obvious year candidates (1800-2099) and zero.
            if 1800 <= n <= 2099:
                continue
            if n == 0:
                continue
            # Reject absurdly-large footer numbers (>2000 would be unusual).
            if n > 2000:
                continue
            return n
    return None


def _detect_offset(processed_text: str) -> tuple[int | None, int]:
    """Return (offset, confidence_count) where offset = physical_page - printed_page.

    offset is None if no consistent run of >= MIN_CONSECUTIVE_RUN pages was
    found. confidence_count is the length of the longest consistent run.
    """
    parts = PAGE_MARKER.split(processed_text)
    # parts[0] = pre-first-marker; then alternating [page_num_str, page_text]
    samples: list[tuple[int, int]] = []  # (physical_num, printed_num)
    i = 1
    scanned = 0
    while i < len(parts) and scanned < MAX_PHYSICAL_PAGES_TO_SCAN:
        try:
            physical = int(parts[i])
        except (ValueError, IndexError):
            i += 2
            continue
        page_text = parts[i + 1] if i + 1 < len(parts) else ""
        trailing = _page_tail_number(page_text)
        if trailing is not None:
            samples.append((physical, trailing))
        i += 2
        scanned += 1

    if len(samples) < MIN_CONSECUTIVE_RUN:
        return None, 0

    # Find the longest run of consecutive samples with the same (physical -
    # printed) delta AND printed increasing by 1 per physical step.
    best_offset: int | None = None
    best_run = 0
    current_offset: int | None = None
    current_run = 0
    last_sample: tuple[int, int] | None = None

    for phys, pr in samples:
        offset = phys - pr
        # Require the pair (phys, pr) to continue the run:
        # same offset AND physical advances by exactly 1 AND printed
        # advances by exactly 1 since last sample.
        if last_sample is not None:
            phys_prev, pr_prev = last_sample
            if (offset == current_offset
                and phys == phys_prev + 1
                and pr == pr_prev + 1):
                current_run += 1
            else:
                current_offset = offset
                current_run = 1
        else:
            current_offset = offset
            current_run = 1
        if current_run > best_run:
            best_run = current_run
            best_offset = current_offset
        last_sample = (phys, pr)

    if best_run >= MIN_CONSECUTIVE_RUN:
        return best_offset, best_run
    return None, best_run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Execute writes")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of papers to process")
    args = ap.parse_args()

    print(f"[start] backfill_page_offsets  apply={args.apply}", file=sys.stderr)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA busy_timeout = 120000")

    # Only papers with body text AND Docling page markers. Skip TeX-only rows
    # (no reliable page markers in TeX source).
    rows = conn.execute("""
        SELECT paper_id, processed_text
        FROM papers
        WHERE has_full_text = 1
          AND processed_text IS NOT NULL
          AND LENGTH(processed_text) > 500
          AND processed_text LIKE '%<!-- page %'
    """).fetchall()
    if args.limit:
        rows = rows[: args.limit]
    print(f"[found] {len(rows)} candidate papers", file=sys.stderr)

    detected = 0
    no_run = 0
    offset_dist: Counter = Counter()
    updates: list[tuple[int, str]] = []  # (offset, paper_id)

    for pid, txt in rows:
        if _shutdown:
            break
        offset, run = _detect_offset(txt)
        if offset is None:
            no_run += 1
        else:
            detected += 1
            offset_dist[offset] += 1
            updates.append((offset, pid))

    print(f"\n[results] detected={detected}, no_consistent_run={no_run}", file=sys.stderr)
    print("[results] offset distribution (top 10):", file=sys.stderr)
    for off, n in offset_dist.most_common(10):
        print(f"   offset={off:>4}  papers={n}", file=sys.stderr)

    if not args.apply:
        print("\n[dry-run] pass --apply to write pdf_page_offset and pages_verified=1", file=sys.stderr)
        conn.close()
        return 0

    # Apply
    written = 0
    for offset, pid in updates:
        conn.execute(
            "UPDATE papers SET pdf_page_offset = ?, pages_verified = 1 WHERE paper_id = ?",
            (offset, pid),
        )
        written += 1
        if written % 200 == 0:
            conn.commit()
            print(f"  progress: {written}/{len(updates)} applied", file=sys.stderr)
    conn.commit()
    print(f"\n[done] wrote pdf_page_offset + pages_verified=1 on {written} papers", file=sys.stderr)

    n_verified_now = conn.execute("SELECT COUNT(*) FROM papers WHERE pages_verified = 1").fetchone()[0]
    print(f"[verify] pages_verified=1 total now: {n_verified_now}", file=sys.stderr)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
