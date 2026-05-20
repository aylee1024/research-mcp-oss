# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pymupdf>=1.24.0",
# ]
# ///
"""Detect and populate `pdf_page_offset` using PDF-embedded page labels.

This script reads the PDF `/PageLabels` tree via PyMuPDF. If labels are absent,
or if no arabic-labeled page 1 appears in the first 80 physical pages, the
paper is skipped. In `--apply` mode the script writes `pdf_page_offset` and
`pages_verified = 1` for detected papers.
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
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_mcp.paths import PAPERS_DB_PATH as DB_PATH  # noqa: E402

MAX_PHYSICAL_PAGES_TO_SCAN = 80
DEFAULT_TIMEOUT = 10.0
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) // 2)

ROMAN_LABEL_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
TRAILING_NUMBER_RE = re.compile(r"(\d+)\s*$")

_shutdown_requested = False
_WORKER_DB: sqlite3.Connection | None = None


@dataclass(slots=True)
class PaperResult:
    paper_id: str
    status: str
    offset: int | None = None
    path: str | None = None
    error: str | None = None
    existing_offset: int | None = None
    existing_verified: bool = False


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


def _parse_label(label: str) -> int | None:
    """Return the arabic printed page integer if parseable, else None."""
    text = label.strip()
    if not text:
        return None
    if ROMAN_LABEL_RE.fullmatch(text):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    match = TRAILING_NUMBER_RE.search(text)
    if match:
        return int(match.group(1))
    return None


def _extract_offset_from_labels(pdf_path: Path) -> tuple[str, int | None]:
    import fitz

    doc = fitz.open(str(pdf_path))
    try:
        rules = doc.get_page_labels()
        if not rules:
            return "no_labels", None

        page_limit = min(doc.page_count, MAX_PHYSICAL_PAGES_TO_SCAN)
        for physical_idx in range(page_limit):
            label = doc[physical_idx].get_label()
            printed_page = _parse_label(label)
            if printed_page == 1:
                offset = (physical_idx + 1) - printed_page
                return "detected", offset

        return "no_arabic_page_1", None
    finally:
        doc.close()


def _label_probe_child(pdf_path: str, child_conn: Connection) -> None:
    try:
        try:
            import fitz
        except Exception as exc:
            child_conn.send(("parse_error", None, f"{type(exc).__name__}: {exc}"))
            return

        try:
            status, offset = _extract_offset_from_labels(Path(pdf_path))
            child_conn.send((status, offset, None))
        except FileNotFoundError as exc:
            child_conn.send(("missing_pdf", None, f"{type(exc).__name__}: {exc}"))
        except fitz.EmptyFileError as exc:
            child_conn.send(("parse_error", None, f"{type(exc).__name__}: {exc}"))
        except fitz.FileDataError as exc:
            child_conn.send(("parse_error", None, f"{type(exc).__name__}: {exc}"))
        except Exception as exc:
            child_conn.send(("parse_error", None, f"{type(exc).__name__}: {exc}"))
    finally:
        child_conn.close()


def _extract_with_timeout(pdf_path: Path, timeout_seconds: float) -> tuple[str, int | None, str | None]:
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_label_probe_child, args=(str(pdf_path), child_conn))
    process.start()
    child_conn.close()

    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.terminate()
                process.join(1.0)
                if process.is_alive():
                    process.kill()
                    process.join()
                return "timeout", None, f"timed out after {timeout_seconds:.1f}s"

            if parent_conn.poll(min(0.1, remaining)):
                try:
                    status, offset, error = parent_conn.recv()
                except EOFError:
                    status, offset, error = None, None, None
                process.join(1.0)
                if process.is_alive():
                    process.kill()
                    process.join()
                if status is None:
                    return (
                        "parse_error",
                        None,
                        f"label probe exited without returning a result (exitcode={process.exitcode})",
                    )
                return status, offset, error

            if not process.is_alive():
                process.join(1.0)
                if parent_conn.poll():
                    try:
                        return parent_conn.recv()
                    except EOFError:
                        pass
                return (
                    "parse_error",
                    None,
                    f"label probe exited without returning a result (exitcode={process.exitcode})",
                )
    finally:
        parent_conn.close()


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
        """
        SELECT local_pdf_path, pdf_page_offset, COALESCE(pages_verified, 0)
        FROM papers
        WHERE paper_id = ?
        """,
        (paper_id,),
    ).fetchone()
    if row is None:
        return PaperResult(paper_id=paper_id, status="parse_error", error="paper not found in DB")

    pdf_path_raw, existing_offset, existing_verified = row
    pdf_path = Path((pdf_path_raw or "")).expanduser()
    if not pdf_path_raw or not pdf_path.exists():
        return PaperResult(
            paper_id=paper_id,
            status="missing_pdf",
            path=str(pdf_path),
            existing_offset=existing_offset,
            existing_verified=bool(existing_verified),
        )

    status, offset, error = _extract_with_timeout(pdf_path, timeout_seconds=timeout_seconds)
    return PaperResult(
        paper_id=paper_id,
        status=status,
        offset=offset,
        path=str(pdf_path),
        error=error,
        existing_offset=existing_offset,
        existing_verified=bool(existing_verified),
    )


def _load_candidate_ids(
    conn: sqlite3.Connection,
    limit: int,
    skip_verified: bool,
    include_rejected_implausible: bool,
) -> list[str]:
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
    if not include_rejected_implausible:
        # The prior heuristic pass may have left some rows with an implausible
        # placeholder offset. Keep those opt-in so the default run stays focused
        # on rows that have never received a large rejected value.
        sql += """
          AND NOT (
                COALESCE(pages_verified, 0) = 0
            AND pdf_page_offset IS NOT NULL
            AND ABS(pdf_page_offset) > 100
          )
        """
    sql += " ORDER BY paper_id"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return [row[0] for row in conn.execute(sql, params).fetchall()]


def _apply_result(conn: sqlite3.Connection, result: PaperResult) -> str:
    if result.status != "detected" or result.offset is None:
        return "noop"

    conn.execute(
        """
        UPDATE papers
        SET pdf_page_offset = ?, pages_verified = 1
        WHERE paper_id = ?
        """,
        (result.offset, result.paper_id),
    )

    had_prior_offset = result.existing_verified or (
        result.existing_offset is not None and abs(result.existing_offset) > 100
    )
    return "overwritten" if had_prior_offset else "detected"


def main() -> int:
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    parser = argparse.ArgumentParser(
        description="Backfill pdf_page_offset from PDF-embedded /PageLabels metadata."
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
        "--include-rejected-implausible",
        action="store_true",
        help=(
            "also process rows with pages_verified = 0 and an existing "
            "pdf_page_offset whose absolute value exceeds 100"
        ),
    )
    args = parser.parse_args()

    started = time.monotonic()
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA busy_timeout = 120000")

    candidate_ids = _load_candidate_ids(
        conn,
        limit=args.limit,
        skip_verified=args.skip_verified,
        include_rejected_implausible=args.include_rejected_implausible,
    )
    total = len(candidate_ids)
    print(
        f"[start] candidates={total} apply={args.apply} workers={args.workers} "
        f"timeout={args.timeout:g}s skip_verified={args.skip_verified} "
        f"include_rejected_implausible={args.include_rejected_implausible}",
        file=sys.stderr,
    )
    if total == 0:
        print("[done] no candidate papers", file=sys.stderr)
        conn.close()
        return 0

    processed = 0
    detected = 0
    no_labels = 0
    no_arabic_page_1 = 0
    missing_pdf = 0
    timeout_count = 0
    parse_errors = 0
    worker_crashes = 0
    offset_dist: Counter[int] = Counter()
    pending_writes = 0
    applied_detected = 0
    applied_overwritten = 0

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
                elif result.status == "no_labels":
                    no_labels += 1
                elif result.status == "no_arabic_page_1":
                    no_arabic_page_1 += 1
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
                    action = _apply_result(conn, result)
                    if action != "noop":
                        pending_writes += 1
                    if action == "detected":
                        applied_detected += 1
                    elif action == "overwritten":
                        applied_overwritten += 1
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
        f"[summary] processed={processed} detected={detected} no_labels={no_labels} "
        f"no_arabic_page_1={no_arabic_page_1} errors={error_count} "
        f"elapsed={_format_seconds(elapsed)}",
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
            f"[summary] applied_detected={applied_detected} "
            f"applied_overwritten={applied_overwritten}",
            file=sys.stderr,
        )

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
