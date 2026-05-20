# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pdfplumber>=0.11.0",
#     "pypdf>=5.0.0",
# ]
# ///
"""Detect and populate `pdf_page_offset` using the source PDFs directly.

This script scans the first 50 physical PDF pages, extracts text from the
header and footer bands, and looks for a consistent run where:

    physical_page - printed_page = constant

Only runs of length >= 3 are treated as high confidence. In `--apply` mode the
script writes `pdf_page_offset` and `pages_verified = 1` for detected papers.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import multiprocessing as mp
import os
import re
import signal
import sqlite3
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_mcp.paths import PAPERS_DB_PATH as DB_PATH  # noqa: E402

MAX_PHYSICAL_PAGES_TO_SCAN = 50
HEADER_RATIO = 0.10
FOOTER_RATIO = 0.15
MIN_CONSECUTIVE_RUN = 3
DEFAULT_TIMEOUT = 30.0
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) // 2)

TRAILING_NUMBER_RE = re.compile(r"(?:^|\s)(\d{1,4})\s*[.)\]]*\s*$")
SECTION_TRAILING_RE = re.compile(r"(?<!\d)\d{1,4}\s*[-\u2010-\u2015]\s*(\d{1,4})\s*[.)\]]*\s*$")
OF_TOTAL_RE = re.compile(r"(?<!\d)(\d{1,4})\s+of\s+\d{1,4}(?!\d)", re.IGNORECASE)
INTEGER_RE = re.compile(r"(?<!\d)(\d{1,4})(?!\d)")
ISOLATED_NUMBER_RE = re.compile(r"^[\[(]?\d{1,4}[.)\]]?$")
MULTISPACE_RE = re.compile(r"\s+")

ROMAN_NUMERALS = {
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii",
    "xix", "xx", "xxi", "xxii", "xxiii", "xxiv", "xxv",
}

_shutdown_requested = False
_WORKER_DB: sqlite3.Connection | None = None


class PDFTimeoutError(TimeoutError):
    """Raised when a single PDF exceeds the configured processing budget."""


@dataclass(slots=True)
class PaperResult:
    paper_id: str
    status: str
    offset: int | None = None
    run_length: int = 0
    path: str | None = None
    error: str | None = None
    engine: str | None = None


@dataclass(slots=True)
class RunState:
    length: int
    score: int


def _on_signal(sig: int, frame: object) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n[signal] Received {signal.Signals(sig).name}. Finishing in-flight work...", file=sys.stderr)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _normalize_line(line: str) -> str:
    return MULTISPACE_RE.sub(" ", line).strip()


def _looks_like_year(value: int) -> bool:
    return 1800 <= value <= 2099


def _add_candidate(scores: dict[int, int], value: int | str, weight: int) -> None:
    number = int(value)
    if number == 0:
        return
    if number < 0 or number > 9999:
        return
    if _looks_like_year(number):
        weight = max(1, weight - 2)
    scores[number] = max(scores.get(number, 0), weight)


def _extract_candidates_from_lines(lines: Iterable[str]) -> dict[int, int]:
    scores: dict[int, int] = {}
    for raw_line in lines:
        line = _normalize_line(raw_line)
        if not line:
            continue
        lowered = line.strip("[](){}.,;:").lower()
        if lowered in ROMAN_NUMERALS:
            continue

        if ISOLATED_NUMBER_RE.match(line):
            match = INTEGER_RE.search(line)
            if match:
                _add_candidate(scores, match.group(1), 6)

        match = SECTION_TRAILING_RE.search(line)
        if match:
            _add_candidate(scores, match.group(1), 5)

        match = OF_TOTAL_RE.search(line)
        if match:
            _add_candidate(scores, match.group(1), 5)

        match = TRAILING_NUMBER_RE.search(line)
        if match:
            _add_candidate(scores, match.group(1), 4)

        for token in INTEGER_RE.findall(line):
            _add_candidate(scores, token, 2)
    return scores


def _page_candidate_scores(header_text: str, footer_text: str) -> dict[int, int]:
    scores: dict[int, int] = {}
    for region_text in (header_text, footer_text):
        lines = [line for line in region_text.splitlines() if line.strip()]
        if not lines:
            continue
        focus_lines = lines[:4] + lines[-4:]
        for number, weight in _extract_candidates_from_lines(focus_lines).items():
            scores[number] = max(scores.get(number, 0), weight)
    return scores


def _detect_offset(page_candidates: list[tuple[int, dict[int, int]]]) -> tuple[int | None, int]:
    best_offset: int | None = None
    best_length = 0
    best_score = -1
    previous_states: dict[int, RunState] = {}
    previous_physical: int | None = None

    for physical_page, candidates in page_candidates:
        current_states: dict[int, RunState] = {}
        if previous_physical is None or physical_page != previous_physical + 1:
            previous_states = {}

        for printed_page, page_score in sorted(candidates.items()):
            prior = previous_states.get(printed_page - 1)
            if prior is None:
                state = RunState(length=1, score=page_score)
            else:
                state = RunState(length=prior.length + 1, score=prior.score + page_score)

            existing = current_states.get(printed_page)
            if existing is None or state.length > existing.length or (
                state.length == existing.length and state.score > existing.score
            ):
                current_states[printed_page] = state

            offset = physical_page - printed_page
            if (
                state.length > best_length
                or (state.length == best_length and state.score > best_score)
                or (
                    state.length == best_length
                    and state.score == best_score
                    and best_offset is not None
                    and abs(offset) < abs(best_offset)
                )
            ):
                best_offset = offset
                best_length = state.length
                best_score = state.score

        previous_states = current_states
        previous_physical = physical_page

    if best_length >= MIN_CONSECUTIVE_RUN:
        return best_offset, best_length
    return None, best_length


def _extract_text_pdfplumber(pdf_path: Path) -> list[tuple[int, dict[int, int]]]:
    import pdfplumber

    results: list[tuple[int, dict[int, int]]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        page_limit = min(len(pdf.pages), MAX_PHYSICAL_PAGES_TO_SCAN)
        for physical_page in range(1, page_limit + 1):
            page = pdf.pages[physical_page - 1]
            width = float(page.width)
            height = float(page.height)
            header = page.crop((0, 0, width, height * HEADER_RATIO)).extract_text() or ""
            footer = page.crop((0, height * (1.0 - FOOTER_RATIO), width, height)).extract_text() or ""
            results.append((physical_page, _page_candidate_scores(header, footer)))
    return results


def _page_height_pypdf(page: object) -> float:
    mediabox = page.mediabox
    try:
        return float(mediabox.top) - float(mediabox.bottom)
    except Exception:
        return float(mediabox.height)


def _fragments_to_text(fragments: list[tuple[float, float, str]]) -> str:
    if not fragments:
        return ""
    fragments.sort(key=lambda item: (round(item[0] / 3.0), item[1]))
    lines: list[str] = []
    current_y: float | None = None
    current_parts: list[str] = []
    for y_value, x_value, text in fragments:
        if current_y is None or abs(y_value - current_y) <= 3.0:
            current_parts.append(text)
            if current_y is None:
                current_y = y_value
        else:
            line = "".join(current_parts).strip()
            if line:
                lines.append(line)
            current_parts = [text]
            current_y = y_value
    trailing = "".join(current_parts).strip()
    if trailing:
        lines.append(trailing)
    return "\n".join(lines)


def _split_full_page_text(page_text: str) -> tuple[str, str]:
    lines = [line for line in page_text.splitlines() if line.strip()]
    if not lines:
        return "", ""
    head = "\n".join(lines[:4])
    tail = "\n".join(lines[-4:])
    return head, tail


def _extract_text_pypdf(pdf_path: Path) -> list[tuple[int, dict[int, int]]]:
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
        height = _page_height_pypdf(page)
        header_fragments: list[tuple[float, float, str]] = []
        footer_fragments: list[tuple[float, float, str]] = []

        def visitor(text: str, cm: list[float], tm: list[float], font_dict: object, font_size: float) -> None:
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
            if y_value >= height * (1.0 - HEADER_RATIO):
                header_fragments.append((top_value, x_value, text))
            elif y_value <= height * FOOTER_RATIO:
                footer_fragments.append((top_value, x_value, text))

        full_text = page.extract_text(visitor_text=visitor) or ""
        header = _fragments_to_text(header_fragments)
        footer = _fragments_to_text(footer_fragments)
        if not header and not footer:
            header, footer = _split_full_page_text(full_text)
        results.append((physical_page, _page_candidate_scores(header, footer)))
    return results


@contextmanager
def _time_limit(seconds: float) -> Iterable[None]:
    if seconds <= 0:
        yield
        return

    def _raise_timeout(sig: int, frame: object) -> None:
        raise PDFTimeoutError(f"timed out after {seconds:.1f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _extract_page_candidates(pdf_path: Path, timeout_seconds: float) -> tuple[list[tuple[int, dict[int, int]]], str]:
    total_budget = max(1.0, timeout_seconds)
    pdfplumber_budget = max(1.0, total_budget * 0.75)
    started = time.monotonic()
    last_error: Exception | None = None

    try:
        with _time_limit(pdfplumber_budget):
            return _extract_text_pdfplumber(pdf_path), "pdfplumber"
    except Exception as exc:
        last_error = exc

    remaining = total_budget - (time.monotonic() - started)
    if remaining <= 0:
        raise last_error if last_error is not None else PDFTimeoutError(f"timed out after {timeout_seconds:.1f}s")

    try:
        with _time_limit(remaining):
            return _extract_text_pypdf(pdf_path), "pypdf"
    except Exception as exc:
        if isinstance(exc, PDFTimeoutError):
            raise exc
        if isinstance(last_error, PDFTimeoutError):
            raise last_error
        if last_error is not None:
            raise RuntimeError(f"pdfplumber failed: {last_error}; pypdf failed: {exc}") from exc
        raise


def _worker_init(db_path: str) -> None:
    global _WORKER_DB
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    db_uri = f"file:{db_path}?mode=ro"
    _WORKER_DB = sqlite3.connect(db_uri, uri=True)
    _WORKER_DB.execute("PRAGMA busy_timeout = 30000")


def _process_paper(paper_id: str, timeout_seconds: float) -> PaperResult:
    if _WORKER_DB is None:
        raise RuntimeError("worker database was not initialized")

    row = _WORKER_DB.execute(
        "SELECT local_pdf_path FROM papers WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    if row is None:
        return PaperResult(paper_id=paper_id, status="parse_error", error="paper not found in DB")

    pdf_path = Path(row[0]).expanduser()
    if not pdf_path.exists():
        return PaperResult(paper_id=paper_id, status="missing_pdf", path=str(pdf_path))

    try:
        page_candidates, engine = _extract_page_candidates(pdf_path, timeout_seconds=timeout_seconds)
        offset, run_length = _detect_offset(page_candidates)
        if offset is None:
            return PaperResult(
                paper_id=paper_id,
                status="no_consistent_run",
                run_length=run_length,
                path=str(pdf_path),
                engine=engine,
            )
        return PaperResult(
            paper_id=paper_id,
            status="detected",
            offset=offset,
            run_length=run_length,
            path=str(pdf_path),
            engine=engine,
        )
    except PDFTimeoutError as exc:
        return PaperResult(paper_id=paper_id, status="timeout", path=str(pdf_path), error=str(exc))
    except Exception as exc:
        return PaperResult(paper_id=paper_id, status="parse_error", path=str(pdf_path), error=str(exc))


def _load_candidate_ids(conn: sqlite3.Connection, limit: int, skip_verified: bool) -> list[str]:
    sql = """
        SELECT paper_id
        FROM papers
        WHERE has_full_text = 1
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


def _apply_result(
    conn: sqlite3.Connection,
    result: PaperResult,
    redo_verified: bool,
    max_abs_offset: int,
) -> str:
    if result.status == "detected" and result.offset is not None:
        # Sanity filter: implausibly-large absolute offsets almost always
        # come from the detector latching onto a year (e.g. "2019" in a
        # footer), ISBN fragment, or article identifier and building a
        # spurious 3-page run. A legitimate offset > 100 would require a
        # book excerpt whose first physical page is deep into a large
        # volume; rare in this library and worth rejecting over admitting
        # false positives that would label pages wrong.
        if abs(result.offset) > max_abs_offset:
            return "rejected_implausible_offset"
        conn.execute(
            """
            UPDATE papers
            SET pdf_page_offset = ?, pages_verified = 1
            WHERE paper_id = ?
            """,
            (result.offset, result.paper_id),
        )
        return "detected"

    # The previous `redo_verified + no_consistent_run` branch cleared
    # pages_verified back to 0. That violated the session-wide invariant
    # that pages_verified must be monotone non-decreasing across
    # detection passes: a noisier rerun could silently clear a row that
    # a stronger source (PyMuPDF labels, manual review) had already
    # verified. Even under --redo-verified, a negative rerun result
    # is not evidence of incorrectness; it is evidence of the new
    # detector missing what the old one caught. We now treat it as
    # a no-op and log rather than clear. Use apply_manual_offsets.py
    # plus an explicit UPDATE to clear a specific row if ever needed.
    return "noop"


def main() -> int:
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    parser = argparse.ArgumentParser(
        description="Backfill pdf_page_offset from original PDF headers and footers."
    )
    parser.add_argument("--apply", action="store_true", help="write updates to papers.db")
    parser.add_argument("--limit", type=int, default=0, help="process at most N candidate papers")
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=DEFAULT_WORKERS,
        help=f"parallel workers to use (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_TIMEOUT,
        help=f"per-PDF timeout in seconds (default: {DEFAULT_TIMEOUT:g})",
    )
    verified_group = parser.add_mutually_exclusive_group()
    verified_group.add_argument(
        "--skip-verified",
        dest="skip_verified",
        action="store_true",
        help="skip papers already at pages_verified = 1 (default)",
    )
    verified_group.add_argument(
        "--redo-verified",
        dest="skip_verified",
        action="store_false",
        help="re-run papers already at pages_verified = 1",
    )
    parser.set_defaults(skip_verified=True)
    parser.add_argument(
        "--max-abs-offset",
        type=int,
        default=100,
        help=(
            "reject detected offsets whose absolute value exceeds this "
            "threshold (default: 100). Guards against false-positive "
            "detections where the scanner latched onto a year, ISBN, or "
            "article id that produced a spurious 3-page run."
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
    pending_writes = 0
    applied_detected = 0
    applied_cleared = 0
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
            futures[executor.submit(_process_paper, paper_id, args.timeout)] = paper_id
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
                    action = _apply_result(
                        conn, result,
                        redo_verified=not args.skip_verified,
                        max_abs_offset=args.max_abs_offset,
                    )
                    if action not in ("noop", "rejected_implausible_offset"):
                        pending_writes += 1
                    if action == "detected":
                        applied_detected += 1
                    elif action == "cleared":
                        applied_cleared += 1
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
                        f"detected={detected} rate={rate:.1%} eta={_format_seconds(eta)}",
                        file=sys.stderr,
                    )

                if not _shutdown_requested and next_index < total:
                    next_paper_id = candidate_ids[next_index]
                    futures[executor.submit(_process_paper, next_paper_id, args.timeout)] = next_paper_id
                    next_index += 1

        if _shutdown_requested:
            print("[signal] No new work submitted. Exiting after in-flight tasks.", file=sys.stderr)

    if args.apply and pending_writes:
        conn.commit()

    elapsed = time.monotonic() - started
    error_count = missing_pdf + timeout_count + parse_errors + worker_crashes
    print(
        f"[summary] processed={processed} detected={detected} no_consistent_run={no_run} "
        f"errors={error_count} elapsed={_format_seconds(elapsed)}",
        file=sys.stderr,
    )
    print(
        f"[summary] error_breakdown missing_pdf={missing_pdf} timeout={timeout_count} "
        f"parse_error={parse_errors} worker_crash={worker_crashes}",
        file=sys.stderr,
    )
    print(f"[summary] offset_distribution={offset_dist.most_common()}", file=sys.stderr)
    if args.apply:
        print(
            f"[summary] applied_detected={applied_detected} applied_cleared={applied_cleared} "
            f"applied_rejected_implausible={applied_rejected} "
            f"(max_abs_offset={args.max_abs_offset})",
            file=sys.stderr,
        )

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
