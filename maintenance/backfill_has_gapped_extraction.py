#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Plan v3 Phase A.1: backfill papers.has_gapped_extraction (schema v19).

One-pass over has_full_text=1 papers, classifying page-marker quality
via the same heuristic as server.py:_classify_page_markers:
  None = no body  (skip)
  0    = clean markers
  1    = no markers OR descending markers

Idempotent: only writes rows whose current value differs from the
classification. Safe to re-run.

Usage:
    uv run maintenance/backfill_has_gapped_extraction.py
    uv run maintenance/backfill_has_gapped_extraction.py --dry-run
    uv run maintenance/backfill_has_gapped_extraction.py --force-revisit  # re-evaluate all rows even if already non-NULL
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from research_mcp.paths import PAPERS_DB_PATH as DEFAULT_DB  # noqa: E402

PAGE_MARKER = re.compile(r"<!--\s*page\s+(-?\d+)\s*-->")


def classify(processed_text: str) -> int | None:
    if not processed_text:
        return None
    markers = [int(m.group(1)) for m in PAGE_MARKER.finditer(processed_text)]
    if not markers:
        return 1
    for a, b in zip(markers, markers[1:]):
        if b - a < 0:
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--force-revisit",
        action="store_true",
        help="Re-evaluate all rows including those already non-NULL.",
    )
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    cols = {r[1] for r in conn.execute("PRAGMA table_info(papers)").fetchall()}
    if "has_gapped_extraction" not in cols:
        print("ERROR: schema v19 not present (no has_gapped_extraction column)", file=sys.stderr)
        print("Run server.py once to migrate, OR ALTER TABLE manually.", file=sys.stderr)
        return 2

    where = "WHERE has_full_text = 1"
    if not args.force_revisit:
        where += " AND has_gapped_extraction IS NULL"
    rows = conn.execute(
        f"SELECT paper_id, processed_text, tex_text, has_gapped_extraction FROM papers {where}"
    ).fetchall()
    print(f"Candidates: {len(rows)} (force_revisit={args.force_revisit})")

    counts = {"clean": 0, "problematic": 0, "skipped": 0, "unchanged": 0, "updated": 0, "tex_only_problematic": 0}
    t0 = time.perf_counter()
    BATCH = 500
    pending: list[tuple[int, str]] = []
    for row in rows:
        # TeX-only papers (processed_text empty, tex_text populated) lack
        # the <!-- page N --> scaffold entirely. Pincite at page-level
        # requires processed_text page markers — TeX section structure
        # alone is insufficient. Classify as problematic so downstream
        # pincite callers can filter.
        proc = row["processed_text"] or ""
        tex = row["tex_text"] or ""
        if not proc and tex:
            cls = 1
            counts["tex_only_problematic"] += 1
            counts["problematic"] += 1
        else:
            cls = classify(proc)
            if cls is None:
                counts["skipped"] += 1
                continue
            if cls == 0:
                counts["clean"] += 1
            else:
                counts["problematic"] += 1
        if row["has_gapped_extraction"] == cls:
            counts["unchanged"] += 1
            continue
        pending.append((cls, row["paper_id"]))
        if not args.dry_run and len(pending) >= BATCH:
            conn.executemany("UPDATE papers SET has_gapped_extraction = ? WHERE paper_id = ?", pending)
            conn.commit()
            counts["updated"] += len(pending)
            pending = []

    if pending and not args.dry_run:
        conn.executemany("UPDATE papers SET has_gapped_extraction = ? WHERE paper_id = ?", pending)
        conn.commit()
        counts["updated"] += len(pending)

    if args.dry_run:
        # In dry-run mode, "updated" is the proposed-write count, not actual writes.
        counts["updated"] = (counts["clean"] + counts["problematic"]) - counts["unchanged"]

    dt = time.perf_counter() - t0
    print(f"Counts: {counts}")
    print(f"Wallclock: {dt:.2f}s")

    # Verification query (mirrors plan v3 §3 Phase A.1 audit gate):
    null_count = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE has_full_text = 1 AND has_gapped_extraction IS NULL"
    ).fetchone()[0]
    print(f"VERIFY: papers WHERE has_full_text=1 AND has_gapped_extraction IS NULL: {null_count}")
    # Round-2 review HIGH fix: --dry-run was returning exit 1 whenever
    # the column had any NULL rows (which is the *normal* state before
    # the first real run). CI/CD gates checking exit code would loop
    # forever. Dry-run is a preview and should always exit 0; the
    # null_count is informational, surfaced via stdout.
    if args.dry_run:
        return 0
    return 0 if null_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
