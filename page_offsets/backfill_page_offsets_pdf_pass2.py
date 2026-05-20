# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pdfplumber>=0.11.0",
#     "pypdf>=5.0.0",
# ]
# ///
"""Second-pass `pdf_page_offset` backfill for Docling-marker PDFs.

This pass targets papers that already have Docling page markers and a local PDF
but were missed by `backfill_page_offsets_pdf.py`. It tries three heuristics in
order and stops at the first validated hit:

1. Wider header/footer crops with the original 3-page run threshold.
2. Original crop sizes with a 2-page run threshold, but only if a wider-crop
   2-page run agrees on the same offset.
3. Left/right side-margin crops, but only if a wider-crop 2-page run agrees on
   the same offset.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import multiprocessing as mp
import signal
import sqlite3
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import backfill_page_offsets_pdf as pass1


DB_PATH = pass1.DB_PATH

MAX_PHYSICAL_PAGES_TO_SCAN = pass1.MAX_PHYSICAL_PAGES_TO_SCAN
DEFAULT_TIMEOUT = 10.0
DEFAULT_WORKERS = pass1.DEFAULT_WORKERS

BASE_HEADER_RATIO = pass1.HEADER_RATIO
BASE_FOOTER_RATIO = pass1.FOOTER_RATIO
WIDER_HEADER_RATIO = 0.20
WIDER_FOOTER_RATIO = 0.25
SIDE_MARGIN_RATIO = 0.10

DEFAULT_MIN_RUN = pass1.MIN_CONSECUTIVE_RUN
RELAXED_MIN_RUN = 2

_shutdown_requested = False
_WORKER_DB: sqlite3.Connection | None = None


@dataclass(slots=True)
class PaperResult:
    paper_id: str
    status: str
    offset: int | None = None
    run_length: int = 0
    path: str | None = None
    error: str | None = None
    engine: str | None = None
    method: str | None = None


def _on_signal(sig: int, frame: object) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n[signal] Received {signal.Signals(sig).name}. Finishing in-flight work...", file=sys.stderr)


@contextmanager
def _pass1_overrides(
    *,
    header_ratio: float | None = None,
    footer_ratio: float | None = None,
    min_run: int | None = None,
) -> Iterable[None]:
    original_header = pass1.HEADER_RATIO
    original_footer = pass1.FOOTER_RATIO
    original_min_run = pass1.MIN_CONSECUTIVE_RUN
    if header_ratio is not None:
        pass1.HEADER_RATIO = header_ratio
    if footer_ratio is not None:
        pass1.FOOTER_RATIO = footer_ratio
    if min_run is not None:
        pass1.MIN_CONSECUTIVE_RUN = min_run
    try:
        yield
    finally:
        pass1.HEADER_RATIO = original_header
        pass1.FOOTER_RATIO = original_footer
        pass1.MIN_CONSECUTIVE_RUN = original_min_run


def _offset_is_acceptable(offset: int | None, max_abs_offset: int) -> bool:
    return offset is not None and abs(offset) <= max_abs_offset


def _merge_candidate_scores(*score_maps: dict[int, int]) -> dict[int, int]:
    merged: dict[int, int] = {}
    for score_map in score_maps:
        for number, weight in score_map.items():
            merged[number] = max(merged.get(number, 0), weight)
    return merged


def _side_margin_candidate_scores(left_text: str, right_text: str) -> dict[int, int]:
    scores: dict[int, int] = {}
    for region_text in (left_text, right_text):
        lines = [line for line in region_text.splitlines() if line.strip()]
        if not lines:
            continue
        scores = _merge_candidate_scores(scores, pass1._extract_candidates_from_lines(lines))
    return scores


def _extract_side_margin_pdfplumber(pdf_path: Path) -> list[tuple[int, dict[int, int]]]:
    import pdfplumber

    results: list[tuple[int, dict[int, int]]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        page_limit = min(len(pdf.pages), MAX_PHYSICAL_PAGES_TO_SCAN)
        for physical_page in range(1, page_limit + 1):
            page = pdf.pages[physical_page - 1]
            width = float(page.width)
            height = float(page.height)
            left_text = page.crop((0, 0, width * SIDE_MARGIN_RATIO, height)).extract_text() or ""
            right_text = page.crop((width * (1.0 - SIDE_MARGIN_RATIO), 0, width, height)).extract_text() or ""
            results.append((physical_page, _side_margin_candidate_scores(left_text, right_text)))
    return results


def _page_width_pypdf(page: object) -> float:
    mediabox = page.mediabox
    try:
        return float(mediabox.right) - float(mediabox.left)
    except Exception:
        return float(mediabox.width)


def _extract_side_margin_pypdf(pdf_path: Path) -> list[tuple[int, dict[int, int]]]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise RuntimeError("encrypted PDF") from exc

    results: list[tuple[int, dict[int, int]]] = []
    page_limit = min(len(reader.pages), MAX_PHYSICAL_PAGES_TO_SCAN)
    for physical_page in range(1, page_limit + 1):
        page = reader.pages[physical_page - 1]
        width = _page_width_pypdf(page)
        height = pass1._page_height_pypdf(page)
        left_fragments: list[tuple[float, float, str]] = []
        right_fragments: list[tuple[float, float, str]] = []

        def visitor(text: str, cm: list[float], tm: list[float], font_dict: object, font_size: float) -> None:
            del font_dict, font_size
            if not text or not text.strip():
                return
            try:
                x_value = float(tm[4])
                y_value = float(tm[5])
            except Exception:
                try:
                    x_value = float(cm[4])
                    y_value = float(cm[5])
                except Exception:
                    return
            top_value = height - y_value
            if x_value <= width * SIDE_MARGIN_RATIO:
                left_fragments.append((top_value, x_value, text))
            elif x_value >= width * (1.0 - SIDE_MARGIN_RATIO):
                right_fragments.append((top_value, x_value, text))

        page.extract_text(visitor_text=visitor)
        left_text = pass1._fragments_to_text(left_fragments)
        right_text = pass1._fragments_to_text(right_fragments)
        results.append((physical_page, _side_margin_candidate_scores(left_text, right_text)))
    return results


def _extract_side_margin_candidates(pdf_path: Path, timeout_seconds: float) -> tuple[list[tuple[int, dict[int, int]]], str]:
    total_budget = max(1.0, timeout_seconds)
    pdfplumber_budget = max(1.0, total_budget * 0.75)
    started = time.monotonic()
    last_error: Exception | None = None

    try:
        with pass1._time_limit(pdfplumber_budget):
            return _extract_side_margin_pdfplumber(pdf_path), "pdfplumber"
    except Exception as exc:
        last_error = exc

    remaining = total_budget - (time.monotonic() - started)
    if remaining <= 0:
        if last_error is not None:
            raise last_error
        raise pass1.PDFTimeoutError(f"timed out after {timeout_seconds:.1f}s")

    try:
        with pass1._time_limit(remaining):
            return _extract_side_margin_pypdf(pdf_path), "pypdf"
    except Exception as exc:
        if isinstance(exc, pass1.PDFTimeoutError):
            raise exc
        if isinstance(last_error, pass1.PDFTimeoutError):
            raise last_error
        if last_error is not None:
            raise RuntimeError(f"pdfplumber failed: {last_error}; pypdf failed: {exc}") from exc
        raise


def _extract_header_footer_candidates(
    pdf_path: Path,
    timeout_seconds: float,
    *,
    header_ratio: float,
    footer_ratio: float,
) -> tuple[list[tuple[int, dict[int, int]]], str]:
    with _pass1_overrides(header_ratio=header_ratio, footer_ratio=footer_ratio):
        return pass1._extract_page_candidates(pdf_path, timeout_seconds=timeout_seconds)


def _detect_offset_with_min_run(
    page_candidates: list[tuple[int, dict[int, int]]],
    *,
    min_run: int,
) -> tuple[int | None, int]:
    with _pass1_overrides(min_run=min_run):
        return pass1._detect_offset(page_candidates)


def _remaining_budget(deadline: float, total_budget: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise pass1.PDFTimeoutError(f"timed out after {total_budget:.1f}s")
    return remaining


def _worker_init(db_path: str) -> None:
    global _WORKER_DB
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    db_uri = f"file:{db_path}?mode=ro"
    _WORKER_DB = sqlite3.connect(db_uri, uri=True)
    _WORKER_DB.execute("PRAGMA busy_timeout = 30000")


def _process_paper(paper_id: str, timeout_seconds: float, max_abs_offset: int) -> PaperResult:
    if _WORKER_DB is None:
        raise RuntimeError("worker database was not initialized")

    row = _WORKER_DB.execute(
        "SELECT local_pdf_path FROM papers WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    if row is None:
        return PaperResult(paper_id=paper_id, status="parse_error", error="paper not found in DB")

    pdf_path = Path((row[0] or "")).expanduser()
    if not row[0] or not pdf_path.exists():
        return PaperResult(paper_id=paper_id, status="missing_pdf", path=str(pdf_path))

    wider_run = 0
    wider_relaxed_run = 0
    base_run = 0
    side_run = 0
    wider_relaxed_offset: int | None = None

    try:
        deadline = time.monotonic() + timeout_seconds
        wider_candidates, wider_engine = _extract_header_footer_candidates(
            pdf_path,
            _remaining_budget(deadline, timeout_seconds),
            header_ratio=WIDER_HEADER_RATIO,
            footer_ratio=WIDER_FOOTER_RATIO,
        )
        wider_offset, wider_run = _detect_offset_with_min_run(wider_candidates, min_run=DEFAULT_MIN_RUN)
        if _offset_is_acceptable(wider_offset, max_abs_offset):
            return PaperResult(
                paper_id=paper_id,
                status="detected",
                offset=wider_offset,
                run_length=wider_run,
                path=str(pdf_path),
                engine=wider_engine,
                method="wider_crop",
            )

        wider_relaxed_offset, wider_relaxed_run = _detect_offset_with_min_run(
            wider_candidates,
            min_run=RELAXED_MIN_RUN,
        )
        if not _offset_is_acceptable(wider_relaxed_offset, max_abs_offset):
            wider_relaxed_offset = None

        base_candidates, base_engine = _extract_header_footer_candidates(
            pdf_path,
            _remaining_budget(deadline, timeout_seconds),
            header_ratio=BASE_HEADER_RATIO,
            footer_ratio=BASE_FOOTER_RATIO,
        )
        base_offset, base_run = _detect_offset_with_min_run(base_candidates, min_run=RELAXED_MIN_RUN)
        if (
            _offset_is_acceptable(base_offset, max_abs_offset)
            and wider_relaxed_offset is not None
            and base_offset == wider_relaxed_offset
        ):
            return PaperResult(
                paper_id=paper_id,
                status="detected",
                offset=base_offset,
                run_length=base_run,
                path=str(pdf_path),
                engine=base_engine,
                method="relaxed_run",
            )

        side_candidates, side_engine = _extract_side_margin_candidates(
            pdf_path,
            timeout_seconds=_remaining_budget(deadline, timeout_seconds),
        )
        side_offset, side_run = _detect_offset_with_min_run(side_candidates, min_run=DEFAULT_MIN_RUN)
        if (
            _offset_is_acceptable(side_offset, max_abs_offset)
            and wider_relaxed_offset is not None
            and side_offset == wider_relaxed_offset
        ):
            return PaperResult(
                paper_id=paper_id,
                status="detected",
                offset=side_offset,
                run_length=side_run,
                path=str(pdf_path),
                engine=side_engine,
                method="side_margin",
            )

        return PaperResult(
            paper_id=paper_id,
            status="no_consistent_run",
            run_length=max(wider_run, wider_relaxed_run, base_run, side_run),
            path=str(pdf_path),
        )
    except pass1.PDFTimeoutError as exc:
        return PaperResult(paper_id=paper_id, status="timeout", path=str(pdf_path), error=str(exc))
    except Exception as exc:
        return PaperResult(paper_id=paper_id, status="parse_error", path=str(pdf_path), error=str(exc))


def _load_candidate_ids(conn: sqlite3.Connection, limit: int, skip_verified: bool) -> list[str]:
    sql = """
        SELECT paper_id
        FROM papers
        WHERE has_full_text = 1
          AND COALESCE(processed_text, '') LIKE '%<!-- page %'
          AND local_pdf_path IS NOT NULL
          AND local_pdf_path != ''
    """
    params: list[int] = []
    if skip_verified:
        sql += " AND COALESCE(pages_verified, 0) != 1"
    sql += " ORDER BY paper_id"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return [row[0] for row in conn.execute(sql, params).fetchall()]


def main() -> int:
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    parser = argparse.ArgumentParser(
        description="Backfill pdf_page_offset for Docling-marker PDFs missed by the first pass."
    )
    parser.add_argument("--apply", action="store_true", help="write updates to papers.db")
    parser.add_argument("--limit", type=int, default=0, help="process at most N candidate papers")
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
        help=f"per-PDF timeout in seconds (default: {DEFAULT_TIMEOUT:g})",
    )
    parser.add_argument(
        "--skip-verified",
        dest="skip_verified",
        action="store_true",
        default=True,
        help="skip papers already at pages_verified = 1 (default: True)",
    )
    parser.add_argument(
        "--max-abs-offset",
        type=int,
        default=100,
        help=(
            "reject detected offsets whose absolute value exceeds this "
            "threshold (default: 100). Guards against false-positive "
            "detections where the scanner latched onto a year, ISBN, or "
            "article id that produced a spurious run."
        ),
    )
    args = parser.parse_args()

    started = time.monotonic()
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA busy_timeout = 120000")

    candidate_ids = _load_candidate_ids(conn, limit=args.limit, skip_verified=args.skip_verified)
    total = len(candidate_ids)
    print(
        f"[start] candidates={total} apply={args.apply} workers={args.workers} "
        f"timeout={args.timeout:g}s skip_verified={args.skip_verified}",
        file=sys.stderr,
    )
    if total == 0:
        print("[done] no candidate papers", file=sys.stderr)
        conn.close()
        return 0

    processed = 0
    detected = 0
    no_run = 0
    missing_pdf = 0
    timeout_count = 0
    parse_errors = 0
    worker_crashes = 0
    offset_dist: Counter[int] = Counter()
    method_counts: Counter[str] = Counter()
    pending_writes = 0
    applied_detected = 0
    applied_rejected = 0

    max_workers = min(args.workers, total)
    mp_context = mp.get_context("spawn")
    futures: dict[cf.Future[PaperResult], str] = {}
    next_index = 0

    with cf.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_context,
        initializer=_worker_init,
        initargs=(str(DB_PATH),),
    ) as executor:
        while next_index < total and len(futures) < max_workers and not _shutdown_requested:
            paper_id = candidate_ids[next_index]
            futures[executor.submit(_process_paper, paper_id, args.timeout, args.max_abs_offset)] = paper_id
            next_index += 1

        while futures:
            done, _ = cf.wait(futures, timeout=0.5, return_when=cf.FIRST_COMPLETED)
            if not done:
                continue

            for future in done:
                paper_id = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = PaperResult(
                        paper_id=paper_id,
                        status="worker_crash",
                        error=str(exc),
                    )

                processed += 1
                if result.status == "detected":
                    detected += 1
                    if result.offset is not None:
                        offset_dist[result.offset] += 1
                    if result.method:
                        method_counts[result.method] += 1
                    print(
                        f"[detected] {result.paper_id} offset={result.offset} "
                        f"run={result.run_length} method={result.method} engine={result.engine}",
                        file=sys.stderr,
                    )
                elif result.status == "no_consistent_run":
                    no_run += 1
                elif result.status == "missing_pdf":
                    missing_pdf += 1
                    print(f"[missing] {result.paper_id} {result.path}", file=sys.stderr)
                elif result.status == "timeout":
                    timeout_count += 1
                    print(f"[timeout] {result.paper_id} {result.error}", file=sys.stderr)
                elif result.status == "parse_error":
                    parse_errors += 1
                    print(f"[error] {result.paper_id} {result.error}", file=sys.stderr)
                else:
                    worker_crashes += 1
                    print(f"[worker-crash] {result.paper_id} {result.error}", file=sys.stderr)

                if args.apply:
                    action = pass1._apply_result(
                        conn,
                        result,
                        redo_verified=False,
                        max_abs_offset=args.max_abs_offset,
                    )
                    if action not in ("noop", "rejected_implausible_offset"):
                        pending_writes += 1
                    if action == "detected":
                        applied_detected += 1
                    elif action == "rejected_implausible_offset":
                        applied_rejected += 1
                    if pending_writes >= 100:
                        conn.commit()
                        pending_writes = 0

                if processed % 100 == 0 or processed == total or _shutdown_requested:
                    elapsed = time.monotonic() - started
                    rate = detected / processed if processed else 0.0
                    per_paper = elapsed / processed if processed else 0.0
                    eta = per_paper * max(total - processed, 0)
                    print(
                        f"[progress] {processed}/{total} processed "
                        f"detected={detected} rate={rate:.1%} eta={pass1._format_seconds(eta)}",
                        file=sys.stderr,
                    )

                if not _shutdown_requested and next_index < total:
                    next_paper_id = candidate_ids[next_index]
                    futures[
                        executor.submit(_process_paper, next_paper_id, args.timeout, args.max_abs_offset)
                    ] = next_paper_id
                    next_index += 1

        if _shutdown_requested:
            print("[signal] No new work submitted. Exiting after in-flight tasks.", file=sys.stderr)

    if args.apply and pending_writes:
        conn.commit()

    elapsed = time.monotonic() - started
    error_count = missing_pdf + timeout_count + parse_errors + worker_crashes
    print(
        f"[summary] processed={processed} detected={detected} no_consistent_run={no_run} "
        f"errors={error_count} elapsed={pass1._format_seconds(elapsed)}",
        file=sys.stderr,
    )
    print(
        f"[summary] error_breakdown missing_pdf={missing_pdf} timeout={timeout_count} "
        f"parse_error={parse_errors} worker_crash={worker_crashes}",
        file=sys.stderr,
    )
    print(
        "[summary] "
        f"method_wider_crop={method_counts['wider_crop']} "
        f"method_relaxed_run={method_counts['relaxed_run']} "
        f"method_side_margin={method_counts['side_margin']}",
        file=sys.stderr,
    )
    print(f"[summary] offset_distribution={offset_dist.most_common()}", file=sys.stderr)
    if args.apply:
        print(
            f"[summary] applied_detected={applied_detected} "
            f"applied_rejected_implausible={applied_rejected} "
            f"(max_abs_offset={args.max_abs_offset})",
            file=sys.stderr,
        )

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
