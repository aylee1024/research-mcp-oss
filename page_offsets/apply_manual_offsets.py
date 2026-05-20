# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Apply manual pdf_page_offset edits from a reviewed CSV.

Usage:
    uv run apply_manual_offsets.py --csv /path/to/review.csv
    uv run apply_manual_offsets.py --csv /path/to/review.csv --apply
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_mcp.paths import PAPERS_DB_PATH as DB_PATH, RESEARCH_MCP_HOME  # noqa: E402

DEFAULT_CSV = RESEARCH_MCP_HOME / "manual_review_offsets.csv"
YES_VALUES = {"yes", "y"}
SAFETY_ABS_OFFSET = 500


@dataclass(slots=True)
class RowDecision:
    row_number: int
    paper_id: str
    status: str
    offset: int | None = None
    raw_confirmed: str = ""
    raw_offset: str = ""


def _pages_verified_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM papers WHERE pages_verified = 1").fetchone()[0])


def _parse_offset(raw_value: str) -> int | None:
    text = (raw_value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _evaluate_rows(
    rows: list[dict[str, str]],
    conn: sqlite3.Connection,
) -> list[RowDecision]:
    decisions: list[RowDecision] = []
    for row_number, row in enumerate(rows, start=2):
        paper_id = (row.get("paper_id") or "").strip()
        raw_confirmed = (row.get("manual_confirmed") or "").strip()
        raw_offset = (row.get("manual_offset") or "").strip()
        confirmed = raw_confirmed.lower()
        if confirmed not in YES_VALUES:
            decisions.append(
                RowDecision(
                    row_number=row_number,
                    paper_id=paper_id,
                    status="skipped_unconfirmed",
                    raw_confirmed=raw_confirmed,
                    raw_offset=raw_offset,
                )
            )
            continue

        offset = _parse_offset(raw_offset)
        if offset is None:
            decisions.append(
                RowDecision(
                    row_number=row_number,
                    paper_id=paper_id,
                    status="skipped_bad_offset",
                    raw_confirmed=raw_confirmed,
                    raw_offset=raw_offset,
                )
            )
            continue

        exists = conn.execute(
            "SELECT 1 FROM papers WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()
        if exists is None:
            decisions.append(
                RowDecision(
                    row_number=row_number,
                    paper_id=paper_id,
                    status="paper_not_found",
                    offset=offset,
                    raw_confirmed=raw_confirmed,
                    raw_offset=raw_offset,
                )
            )
            continue

        decisions.append(
            RowDecision(
                row_number=row_number,
                paper_id=paper_id,
                status="applied",
                offset=offset,
                raw_confirmed=raw_confirmed,
                raw_offset=raw_offset,
            )
        )
    return decisions


def _warn_safety_violations(decisions: list[RowDecision]) -> list[RowDecision]:
    violations = [
        decision
        for decision in decisions
        if decision.status == "applied"
        and decision.offset is not None
        and abs(decision.offset) > SAFETY_ABS_OFFSET
    ]
    if not violations:
        return violations

    print(
        f"[warning] {len(violations)} confirmed rows exceed the safety threshold "
        f"(|offset| > {SAFETY_ABS_OFFSET}).",
        file=sys.stderr,
    )
    for decision in violations[:10]:
        print(
            f"[warning] row={decision.row_number} paper_id={decision.paper_id} "
            f"offset={decision.offset}",
            file=sys.stderr,
        )
    if len(violations) > 10:
        print(
            f"[warning] ... plus {len(violations) - 10} more large offsets",
            file=sys.stderr,
        )
    return violations


def _log_decision(decision: RowDecision, apply_mode: bool) -> None:
    if decision.status == "applied":
        mode = "apply" if apply_mode else "dry_run"
        print(
            f"applied row={decision.row_number} paper_id={decision.paper_id} "
            f"offset={decision.offset} mode={mode}"
        )
        return
    if decision.status == "skipped_unconfirmed":
        print(
            f"skipped_unconfirmed row={decision.row_number} paper_id={decision.paper_id} "
            f"manual_confirmed={decision.raw_confirmed!r}"
        )
        return
    if decision.status == "skipped_bad_offset":
        print(
            f"skipped_bad_offset row={decision.row_number} paper_id={decision.paper_id} "
            f"manual_offset={decision.raw_offset!r}"
        )
        return
    print(
        f"paper_not_found row={decision.row_number} paper_id={decision.paper_id} "
        f"manual_offset={decision.raw_offset!r}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply manual pdf_page_offset edits from a reviewed CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"input CSV path (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write updates to papers.db (default: dry-run)",
    )
    parser.add_argument(
        "--safety-check",
        action="store_true",
        help=(
            f"warn on confirmed manual_offset values whose absolute value exceeds {SAFETY_ABS_OFFSET}; "
            "with --apply this blocks writes unless --force is also set"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="bypass --safety-check blocking",
    )
    args = parser.parse_args()

    csv_path = args.csv.expanduser()
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    rows = _read_rows(csv_path)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA busy_timeout = 120000")

    before_count = _pages_verified_count(conn)
    print(f"[verify] pages_verified=1 before: {before_count}")

    decisions = _evaluate_rows(rows, conn)
    safety_violations: list[RowDecision] = []
    if args.safety_check:
        safety_violations = _warn_safety_violations(decisions)

    write_blocked = bool(
        args.apply and args.safety_check and safety_violations and not args.force
    )
    if write_blocked:
        print(
            "[warning] Aborting apply because --safety-check found large offsets. "
            "Re-run with --force if the values are intentional.",
            file=sys.stderr,
        )

    writes_committed = 0
    if args.apply and not write_blocked:
        for decision in decisions:
            if decision.status != "applied" or decision.offset is None:
                continue
            conn.execute(
                """
                UPDATE papers
                SET pdf_page_offset = ?, pages_verified = 1
                WHERE paper_id = ?
                """,
                (decision.offset, decision.paper_id),
            )
            writes_committed += 1
        conn.commit()

    for decision in decisions:
        _log_decision(decision, apply_mode=args.apply and not write_blocked)

    counts = Counter(decision.status for decision in decisions)
    after_count = _pages_verified_count(conn)
    print(f"[verify] pages_verified=1 after: {after_count}")
    print(
        "[summary] "
        f"rows_seen={len(decisions)} "
        f"applied={counts['applied']} "
        f"skipped_unconfirmed={counts['skipped_unconfirmed']} "
        f"skipped_bad_offset={counts['skipped_bad_offset']} "
        f"paper_not_found={counts['paper_not_found']} "
        f"writes_committed={writes_committed}"
    )
    if not args.apply:
        print("[dry-run] no database changes written")
    elif write_blocked:
        print("[apply] blocked by safety-check; no database changes written")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
