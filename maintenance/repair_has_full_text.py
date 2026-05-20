# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""One-off DB repair: recompute has_full_text for every row.

Schema v8 introduced has_full_text as an additive flag, and the v8 migration
backfilled it once. But several mutation paths (pre-Phase-2 _store_*, dedup
merges, selective clears) could flip the flag away from the actual state of
processed_text + tex_text. Phase 2 hardened those paths via
_recompute_has_full_text; this script is the one-time repair for rows that
drifted before the hardening landed.

Invariant after apply:
    has_full_text = 1
    iff LENGTH(COALESCE(processed_text, '')) > 500
     OR LENGTH(COALESCE(tex_text, '')) > 500

Usage:
    uv run repair_has_full_text.py              # dry-run: report mismatches
    uv run repair_has_full_text.py --apply      # fix them
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_mcp.paths import PAPERS_DB_PATH as DB_PATH  # noqa: E402

MISMATCH_SQL = """
SELECT
    SUM(CASE
        WHEN has_full_text = 0
         AND (LENGTH(COALESCE(processed_text, '')) > 500
              OR LENGTH(COALESCE(tex_text, '')) > 500)
        THEN 1 ELSE 0 END)                                   AS false_negatives,
    SUM(CASE
        WHEN has_full_text = 1
         AND LENGTH(COALESCE(processed_text, '')) <= 500
         AND LENGTH(COALESCE(tex_text, '')) <= 500
        THEN 1 ELSE 0 END)                                   AS false_positives,
    COUNT(*)                                                 AS total
FROM papers
"""

REPAIR_SQL = """
UPDATE papers
SET has_full_text = CASE
    WHEN LENGTH(COALESCE(processed_text, '')) > 500
      OR LENGTH(COALESCE(tex_text, '')) > 500
    THEN 1 ELSE 0 END
WHERE has_full_text != CASE
    WHEN LENGTH(COALESCE(processed_text, '')) > 500
      OR LENGTH(COALESCE(tex_text, '')) > 500
    THEN 1 ELSE 0 END
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA busy_timeout = 120000")

    fneg, fpos, total = conn.execute(MISMATCH_SQL).fetchone()
    fneg = fneg or 0
    fpos = fpos or 0
    print(f"[start] total papers: {total:,}", file=sys.stderr)
    print(
        f"        has_full_text=0 but text>500: {fneg:,} (false negatives)",
        file=sys.stderr,
    )
    print(
        f"        has_full_text=1 but no text:  {fpos:,} (false positives)",
        file=sys.stderr,
    )

    if fneg == 0 and fpos == 0:
        print("[done] nothing to repair", file=sys.stderr)
        conn.close()
        return

    if not args.apply:
        print("[done] dry-run. Re-run with --apply to fix.", file=sys.stderr)
        conn.close()
        return

    cur = conn.execute(REPAIR_SQL)
    conn.commit()
    print(f"[done] repaired {cur.rowcount:,} rows", file=sys.stderr)

    # Verify post-repair invariant.
    fneg2, fpos2, _ = conn.execute(MISMATCH_SQL).fetchone()
    if (fneg2 or 0) + (fpos2 or 0) != 0:
        print(
            f"[WARN] post-repair residual mismatches: fneg={fneg2} fpos={fpos2}",
            file=sys.stderr,
        )
        sys.exit(1)
    print("[verified] invariant holds", file=sys.stderr)
    conn.close()


if __name__ == "__main__":
    main()
