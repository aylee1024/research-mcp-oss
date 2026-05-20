# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pdfplumber>=0.11.0",
#     "pypdf>=5.0.0",
# ]
# ///
"""Export a CSV for manual review of unresolved PDF page offsets.

Usage:
    uv run export_manual_review.py
    uv run export_manual_review.py --limit 25 --output /tmp/manual_review.csv
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import multiprocessing as mp
import signal
import sqlite3
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_mcp.paths import RESEARCH_MCP_HOME  # noqa: E402

import backfill_page_offsets_pdf as pass1


DB_PATH = pass1.DB_PATH
DEFAULT_OUTPUT = RESEARCH_MCP_HOME / "manual_review_offsets.csv"
DEFAULT_WORKERS = 9
DEFAULT_TIMEOUT = 10.0
FOOTER_SAMPLE_RATIO = 0.15
FOOTER_SAMPLE_PAGES = 5
FOOTER_SAMPLE_CHARS = 80
IMPLAUSIBLE_ABS_OFFSET = 100
MIN_ACCEPTED_RUN = pass1.MIN_CONSECUTIVE_RUN

CSV_COLUMNS = [
    "paper_id",
    "title",
    "year",
    "local_pdf_path",
    "doi",
    "authors",
    "why_unverified",
    "heuristic_detected_offset",
    "run_length",
    "rejected_reason",
    "sample_footer_p1",
    "sample_footer_p2",
    "sample_footer_p3",
    "sample_footer_p4",
    "sample_footer_p5",
    "manual_offset",
    "manual_confirmed",
]

_shutdown_requested = False


@dataclass(slots=True)
class Candidate:
    paper_id: str
    title: str
    year: int | None
    local_pdf_path: str
    doi: str
    authors: str
    has_markers: bool


@dataclass(slots=True)
class WorkerResult:
    paper_id: str
    heuristic_offset: int | None
    run_length: int
    sample_footers: tuple[str, str, str, str, str]
    rejected_reason: str
    error: str | None = None


def _on_signal(sig: int, frame: object) -> None:
    del frame
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n[signal] Received {signal.Signals(sig).name}. Finishing in-flight work...", file=sys.stderr)


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split())


def _parse_authors(raw_authors: str | None) -> str:
    if not raw_authors:
        return ""
    text = raw_authors.strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        names = [str(item).strip() for item in parsed if str(item).strip()]
        return "; ".join(names[:2])
    return text


def _build_candidate(row: sqlite3.Row) -> Candidate:
    return Candidate(
        paper_id=row["paper_id"],
        title=row["title"] or "",
        year=row["year"],
        local_pdf_path=row["local_pdf_path"] or "",
        doi=row["doi"] or "",
        authors=_parse_authors(row["authors"]),
        has_markers=bool(row["has_markers"]),
    )


def _load_candidates(
    conn: sqlite3.Connection,
    *,
    limit: int,
    include_rejected_implausible: bool,
) -> list[Candidate]:
    base_sql = """
        SELECT
            paper_id,
            title,
            year,
            local_pdf_path,
            doi,
            authors,
            CASE WHEN COALESCE(processed_text, '') LIKE '%<!-- page %' THEN 1 ELSE 0 END AS has_markers
        FROM papers
        WHERE has_full_text = 1
          AND COALESCE(pages_verified, 0) = 0
          AND local_pdf_path IS NOT NULL
          AND local_pdf_path != ''
    """
    rows_by_id: dict[str, Candidate] = {}
    for row in conn.execute(base_sql):
        candidate = _build_candidate(row)
        rows_by_id[candidate.paper_id] = candidate

    if include_rejected_implausible:
        # The extra flag is a union, not a replacement. On the current DB this
        # is expected to be a no-op because those rows also match the base
        # filter, but the explicit union keeps the CLI aligned with the intent.
        rejected_sql = """
            SELECT
                paper_id,
                title,
                year,
                local_pdf_path,
                doi,
                authors,
                CASE WHEN COALESCE(processed_text, '') LIKE '%<!-- page %' THEN 1 ELSE 0 END AS has_markers
            FROM papers
            WHERE has_full_text = 1
              AND COALESCE(pages_verified, 0) = 0
              AND local_pdf_path IS NOT NULL
              AND local_pdf_path != ''
              AND ABS(COALESCE(pdf_page_offset, 0)) > ?
        """
        for row in conn.execute(rejected_sql, (IMPLAUSIBLE_ABS_OFFSET,)):
            candidate = _build_candidate(row)
            rows_by_id.setdefault(candidate.paper_id, candidate)

    ordered = [rows_by_id[paper_id] for paper_id in sorted(rows_by_id)]
    if limit:
        ordered = ordered[:limit]
    return ordered


@contextmanager
def _detect_any_run() -> object:
    original_min_run = pass1.MIN_CONSECUTIVE_RUN
    pass1.MIN_CONSECUTIVE_RUN = 1
    try:
        yield
    finally:
        pass1.MIN_CONSECUTIVE_RUN = original_min_run


def _detect_best_offset(
    page_candidates: list[tuple[int, dict[int, int]]],
) -> tuple[int | None, int]:
    with _detect_any_run():
        return pass1._detect_offset(page_candidates)


def _remaining_budget(deadline: float, total_budget: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise pass1.PDFTimeoutError(f"timed out after {total_budget:.1f}s")
    return remaining


def _derive_rejected_reason(offset: int | None, run_length: int) -> str:
    if offset is None:
        return ""
    reasons: list[str] = []
    if abs(offset) > IMPLAUSIBLE_ABS_OFFSET:
        reasons.append(f"|offset|>{IMPLAUSIBLE_ABS_OFFSET}")
    if run_length < MIN_ACCEPTED_RUN:
        reasons.append(f"run<{MIN_ACCEPTED_RUN}")
    return "; ".join(reasons)


def _extract_footer_samples(pdf_path: Path, timeout_seconds: float) -> tuple[str, str, str, str, str]:
    import pdfplumber

    samples = [""] * FOOTER_SAMPLE_PAGES
    with pass1._time_limit(timeout_seconds):
        with pdfplumber.open(str(pdf_path)) as pdf:
            page_limit = min(len(pdf.pages), FOOTER_SAMPLE_PAGES)
            for idx in range(page_limit):
                page = pdf.pages[idx]
                width = float(page.width)
                height = float(page.height)
                footer_text = page.crop(
                    (0, height * (1.0 - FOOTER_SAMPLE_RATIO), width, height)
                ).extract_text() or ""
                samples[idx] = _normalize_text(footer_text)[:FOOTER_SAMPLE_CHARS]
    return tuple(samples)  # type: ignore[return-value]


def _process_candidate(candidate: Candidate, timeout_seconds: float) -> WorkerResult:
    pdf_path = Path(candidate.local_pdf_path).expanduser()
    if not pdf_path.exists():
        return WorkerResult(
            paper_id=candidate.paper_id,
            heuristic_offset=None,
            run_length=0,
            sample_footers=("", "", "", "", ""),
            rejected_reason="",
            error=f"missing_pdf:{pdf_path}",
        )

    try:
        deadline = time.monotonic() + timeout_seconds
        page_candidates, _engine = pass1._extract_page_candidates(
            pdf_path,
            timeout_seconds=_remaining_budget(deadline, timeout_seconds),
        )
        heuristic_offset, run_length = _detect_best_offset(page_candidates)
        rejected_reason = _derive_rejected_reason(heuristic_offset, run_length)
        sample_footers = _extract_footer_samples(
            pdf_path,
            timeout_seconds=_remaining_budget(deadline, timeout_seconds),
        )
        return WorkerResult(
            paper_id=candidate.paper_id,
            heuristic_offset=heuristic_offset,
            run_length=run_length,
            sample_footers=sample_footers,
            rejected_reason=rejected_reason,
        )
    except pass1.PDFTimeoutError as exc:
        return WorkerResult(
            paper_id=candidate.paper_id,
            heuristic_offset=None,
            run_length=0,
            sample_footers=("", "", "", "", ""),
            rejected_reason="",
            error=str(exc),
        )
    except Exception as exc:
        return WorkerResult(
            paper_id=candidate.paper_id,
            heuristic_offset=None,
            run_length=0,
            sample_footers=("", "", "", "", ""),
            rejected_reason="",
            error=f"{type(exc).__name__}: {exc}",
        )


def _build_csv_row(candidate: Candidate, result: WorkerResult) -> dict[str, str | int | None]:
    sample_p1, sample_p2, sample_p3, sample_p4, sample_p5 = result.sample_footers
    return {
        "paper_id": candidate.paper_id,
        "title": candidate.title,
        "year": candidate.year,
        "local_pdf_path": candidate.local_pdf_path,
        "doi": candidate.doi,
        "authors": candidate.authors,
        "why_unverified": "markers_no_run" if candidate.has_markers else "no_markers",
        "heuristic_detected_offset": result.heuristic_offset,
        "run_length": result.run_length,
        "rejected_reason": result.rejected_reason,
        "sample_footer_p1": sample_p1,
        "sample_footer_p2": sample_p2,
        "sample_footer_p3": sample_p3,
        "sample_footer_p4": sample_p4,
        "sample_footer_p5": sample_p5,
        "manual_offset": "",
        "manual_confirmed": "",
    }


def _sort_key(row: dict[str, str | int | None]) -> tuple[int, int, str]:
    run_length_raw = row["run_length"]
    offset_raw = row["heuristic_detected_offset"]
    run_length = int(run_length_raw) if run_length_raw not in (None, "") else 0
    if isinstance(offset_raw, int):
        offset_rank = abs(offset_raw)
    else:
        offset_rank = 10**9
    return (-run_length, offset_rank, str(row["paper_id"]))


def _write_csv(output_path: Path, rows: list[dict[str, str | int | None]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    parser = argparse.ArgumentParser(
        description="Export unresolved pdf_page_offset papers to a manual review CSV."
    )
    parser.add_argument("--limit", type=int, default=0, help="export at most N papers")
    parser.add_argument(
        "--workers",
        type=pass1._positive_int,
        default=DEFAULT_WORKERS,
        help=f"parallel workers to use (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--timeout",
        type=pass1._positive_float,
        default=DEFAULT_TIMEOUT,
        help=f"per-paper timeout in seconds (default: {DEFAULT_TIMEOUT:g})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--include-rejected-implausible",
        action="store_true",
        help=(
            "also union pages_verified=0 rows with an existing rejected "
            f"pdf_page_offset whose absolute value exceeds {IMPLAUSIBLE_ABS_OFFSET}"
        ),
    )
    args = parser.parse_args()

    started = time.monotonic()
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")

    candidates = _load_candidates(
        conn,
        limit=args.limit,
        include_rejected_implausible=args.include_rejected_implausible,
    )
    conn.close()

    print(
        f"[start] candidates={len(candidates)} workers={args.workers} "
        f"timeout={args.timeout:g}s output={args.output}",
        file=sys.stderr,
    )

    if not candidates:
        _write_csv(args.output, [])
        print(f"[done] wrote empty CSV to {args.output}", file=sys.stderr)
        return 0

    results_by_id: dict[str, WorkerResult] = {}
    max_workers = min(args.workers, len(candidates))
    mp_context = mp.get_context("spawn")

    with cf.ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
        futures = {
            executor.submit(_process_candidate, candidate, args.timeout): candidate.paper_id
            for candidate in candidates
        }

        completed = 0
        for future in cf.as_completed(futures):
            paper_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = WorkerResult(
                    paper_id=paper_id,
                    heuristic_offset=None,
                    run_length=0,
                    sample_footers=("", "", "", "", ""),
                    rejected_reason="",
                    error=f"worker_crash:{type(exc).__name__}: {exc}",
                )

            results_by_id[paper_id] = result
            completed += 1
            if result.error:
                print(f"[error] {paper_id} {result.error}", file=sys.stderr)
            elif completed % 25 == 0 or completed == len(candidates):
                print(f"[progress] completed={completed}/{len(candidates)}", file=sys.stderr)
            if _shutdown_requested:
                break

    export_rows = [
        _build_csv_row(candidate, results_by_id.get(
            candidate.paper_id,
            WorkerResult(
                paper_id=candidate.paper_id,
                heuristic_offset=None,
                run_length=0,
                sample_footers=("", "", "", "", ""),
                rejected_reason="",
                error="not_processed",
            ),
        ))
        for candidate in candidates
    ]
    export_rows.sort(key=_sort_key)
    _write_csv(args.output, export_rows)

    elapsed = time.monotonic() - started
    print(
        f"[done] wrote {len(export_rows)} rows to {args.output} in {elapsed:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
