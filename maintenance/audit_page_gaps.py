#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Plan v3 Phase A.1 pre-backfill audit: page-marker gap detection.

Scans `papers.processed_text` for non-sequential `<!-- page N -->`
markers. A paper has a "gap" when any consecutive marker pair
(N_k, N_{k+1}) violates N_{k+1} == N_k + 1. Gaps break the assumption
that page numbers advance monotonically by 1, which silently corrupts
page-offset math across the body.

Usage:
    uv run maintenance/audit_page_gaps.py [--db papers.db] [--detail]
    uv run maintenance/audit_page_gaps.py --detail --limit 20

Emits:
    Total cite-ready papers (has_full_text=1)
    Papers with gaps
    Papers with duplicate markers (same page N twice)
    Papers with no markers at all
    Papers with first-marker != 1 (probable offset fraction)
    Papers with descending markers (strong extraction-failure signal)
    CSV per-paper detail with --detail

The audit is READ-ONLY. Use the results to size the Phase A.1
backfill (maintenance/backfill_page_gaps.py).
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from research_mcp.paths import PAPERS_DB_PATH as DEFAULT_DB  # noqa: E402

PAGE_MARKER = re.compile(r"<!--\s*page\s+(-?\d+)\s*-->")


def classify(markers: list[int]) -> dict[str, int]:
    """Return flags describing the marker sequence for one paper."""
    if not markers:
        return {
            "no_markers": 1,
            "gap_count": 0,
            "duplicate_count": 0,
            "descending_count": 0,
            "first_is_one": 0,
            "min_page": 0,
            "max_page": 0,
            "marker_count": 0,
        }
    gaps = 0
    duplicates = 0
    descending = 0
    for a, b in zip(markers, markers[1:]):
        diff = b - a
        if diff == 0:
            duplicates += 1
        elif diff < 0:
            descending += 1
        elif diff > 1:
            gaps += 1
    return {
        "no_markers": 0,
        "gap_count": gaps,
        "duplicate_count": duplicates,
        "descending_count": descending,
        "first_is_one": 1 if markers[0] == 1 else 0,
        "min_page": min(markers),
        "max_page": max(markers),
        "marker_count": len(markers),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--detail", action="store_true", help="Emit per-paper CSV")
    ap.add_argument("--limit", type=int, default=0, help="Cap per-paper detail rows")
    ap.add_argument("--only-gaps", action="store_true", help="Per-paper detail: only gap-having papers")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT paper_id, title, processed_text FROM papers WHERE has_full_text = 1 AND processed_text IS NOT NULL AND length(processed_text) > 0"
    )

    stats = {
        "total": 0,
        "no_markers": 0,
        "with_gaps": 0,
        "with_duplicates": 0,
        "with_descending": 0,
        "first_not_one": 0,
        "clean": 0,
    }
    detail_rows: list[tuple] = []
    for row in rows:
        stats["total"] += 1
        markers = [int(m.group(1)) for m in PAGE_MARKER.finditer(row["processed_text"])]
        info = classify(markers)
        if info["no_markers"]:
            stats["no_markers"] += 1
        if info["gap_count"] > 0:
            stats["with_gaps"] += 1
        if info["duplicate_count"] > 0:
            stats["with_duplicates"] += 1
        if info["descending_count"] > 0:
            stats["with_descending"] += 1
        if info["marker_count"] > 0 and not info["first_is_one"]:
            stats["first_not_one"] += 1
        if info["gap_count"] == 0 and info["duplicate_count"] == 0 and info["descending_count"] == 0 and info["marker_count"] > 0 and info["first_is_one"]:
            stats["clean"] += 1
        if args.detail:
            if args.only_gaps and info["gap_count"] == 0 and info["descending_count"] == 0:
                continue
            detail_rows.append(
                (
                    row["paper_id"],
                    info["marker_count"],
                    info["gap_count"],
                    info["duplicate_count"],
                    info["descending_count"],
                    info["min_page"],
                    info["max_page"],
                    (row["title"] or "")[:70],
                )
            )

    print(f"# Page-marker gap audit @ {args.db}")
    print("## Summary")
    print(f"- cite-ready papers with body: {stats['total']}")
    print(f"- no markers at all          : {stats['no_markers']} ({stats['no_markers']/max(stats['total'],1)*100:.1f}%)")
    print(f"- with gaps (diff > 1)       : {stats['with_gaps']} ({stats['with_gaps']/max(stats['total'],1)*100:.1f}%)")
    print(f"- with duplicates (diff == 0): {stats['with_duplicates']} ({stats['with_duplicates']/max(stats['total'],1)*100:.1f}%)")
    print(f"- with descending (diff < 0) : {stats['with_descending']} ({stats['with_descending']/max(stats['total'],1)*100:.1f}%)")
    print(f"- first marker != 1          : {stats['first_not_one']} ({stats['first_not_one']/max(stats['total'],1)*100:.1f}%)")
    print(f"- clean (monotone +1 from 1) : {stats['clean']} ({stats['clean']/max(stats['total'],1)*100:.1f}%)")

    if args.detail:
        print()
        print("## Per-paper detail")
        print("paper_id\tmarker_count\tgap_count\tduplicate_count\tdescending_count\tmin_page\tmax_page\ttitle")
        detail_rows.sort(key=lambda r: (-r[2], -r[3]))
        for idx, row in enumerate(detail_rows):
            if args.limit and idx >= args.limit:
                break
            print("\t".join(str(x) for x in row))

    return 0


if __name__ == "__main__":
    sys.exit(main())
