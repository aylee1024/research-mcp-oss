# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pymupdf>=1.24.0",
#     "pytesseract>=0.3.10",
#     "Pillow>=10.0.0",
# ]
# ///
"""OCR fallback detector for image-based PDFs missed by text extraction passes.

This pass targets papers with a local PDF and full text, but without Docling
page markers. It rasterizes the first 40 physical pages with PyMuPDF, crops the
bottom 20 percent of each page, OCRs the footer crop with Tesseract, and feeds
the extracted candidate numbers into the same consecutive-run detector used by
`backfill_page_offsets_pdf.py`.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import multiprocessing as mp
import shutil
import signal
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import backfill_page_offsets_pdf as pass1


DB_PATH = pass1.DB_PATH

MAX_PHYSICAL_PAGES_TO_SCAN = 40
FOOTER_RATIO = 0.20
DEFAULT_TIMEOUT = 60.0
DEFAULT_WORKERS = pass1.DEFAULT_WORKERS
DEFAULT_DPI = 200
DEFAULT_PSM = 6

PaperResult = pass1.PaperResult

_shutdown_requested = False
_TESSERACT_BIN = shutil.which("tesseract")


class TesseractMissingError(RuntimeError):
    """Raised when the tesseract binary is unavailable to pytesseract."""


def _on_signal(sig: int, frame: object) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n[signal] Received {signal.Signals(sig).name}. Finishing in-flight work...", file=sys.stderr)


def _load_candidate_ids(conn: sqlite3.Connection, limit: int, skip_verified: bool) -> list[str]:
    sql = """
        SELECT paper_id
        FROM papers
        WHERE has_full_text = 1
          AND local_pdf_path IS NOT NULL
          AND local_pdf_path != ''
          AND (
                processed_text IS NULL
                OR processed_text NOT LIKE '%<!-- page %'
              )
    """
    params: list[int] = []
    if skip_verified:
        sql += " AND COALESCE(pages_verified, 0) != 1"
    sql += " ORDER BY paper_id"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return [row[0] for row in conn.execute(sql, params).fetchall()]


def _pixmap_to_image(pix: object) -> object:
    import fitz
    from PIL import Image

    if getattr(pix, "alpha", 0) or getattr(pix, "n", 0) not in (1, 3):
        pix = fitz.Pixmap(fitz.csRGB, pix)

    mode = "L" if pix.n == 1 else "RGB"
    return Image.frombytes(mode, [pix.width, pix.height], pix.samples)


def _footer_candidate_scores(footer_text: str) -> dict[int, int]:
    return pass1._page_candidate_scores("", footer_text)


def _extract_ocr_page_candidates(
    pdf_path: Path,
    *,
    dpi: int,
    psm: int,
) -> tuple[list[tuple[int, dict[int, int]]], int, int, str | None]:
    import fitz
    import pytesseract

    if _TESSERACT_BIN is None:
        raise TesseractMissingError("tesseract binary not found on PATH")

    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_BIN
    config = f"--psm {psm}"

    page_candidates: list[tuple[int, dict[int, int]]] = []
    successful_ocr_pages = 0
    failed_ocr_pages = 0
    last_ocr_error: str | None = None

    doc = fitz.open(str(pdf_path))
    try:
        if doc.needs_pass and not doc.authenticate(""):
            raise RuntimeError("encrypted PDF")

        page_limit = min(doc.page_count, MAX_PHYSICAL_PAGES_TO_SCAN)
        for physical_page in range(1, page_limit + 1):
            page = doc[physical_page - 1]
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            image = _pixmap_to_image(pix)
            width, height = image.size
            footer_top = int(height * (1.0 - FOOTER_RATIO))
            footer_image = image.crop((0, footer_top, width, height))

            try:
                footer_text = pytesseract.image_to_string(footer_image, config=config)
                successful_ocr_pages += 1
                page_candidates.append((physical_page, _footer_candidate_scores(footer_text)))
            except pytesseract.pytesseract.TesseractNotFoundError as exc:
                raise TesseractMissingError(str(exc)) from exc
            except pytesseract.TesseractError as exc:
                failed_ocr_pages += 1
                last_ocr_error = f"page {physical_page}: {type(exc).__name__}: {exc}"
                page_candidates.append((physical_page, {}))
            except RuntimeError as exc:
                failed_ocr_pages += 1
                last_ocr_error = f"page {physical_page}: {type(exc).__name__}: {exc}"
                page_candidates.append((physical_page, {}))
    finally:
        doc.close()

    return page_candidates, successful_ocr_pages, failed_ocr_pages, last_ocr_error


def _process_paper(
    paper_id: str,
    timeout_seconds: float,
    dpi: int,
    psm: int,
) -> PaperResult:
    if pass1._WORKER_DB is None:
        raise RuntimeError("worker database was not initialized")

    row = pass1._WORKER_DB.execute(
        "SELECT local_pdf_path FROM papers WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    if row is None:
        return PaperResult(paper_id=paper_id, status="parse_error", error="paper not found in DB")

    pdf_path = Path((row[0] or "")).expanduser()
    if not row[0] or not pdf_path.exists():
        return PaperResult(paper_id=paper_id, status="missing_pdf", path=str(pdf_path))

    try:
        with pass1._time_limit(timeout_seconds):
            page_candidates, successful_pages, failed_pages, last_ocr_error = _extract_ocr_page_candidates(
                pdf_path,
                dpi=dpi,
                psm=psm,
            )

        if successful_pages == 0 and failed_pages > 0:
            error = last_ocr_error or "OCR failed on every scanned page"
            return PaperResult(
                paper_id=paper_id,
                status="parse_error",
                path=str(pdf_path),
                error=f"OCR failed on all scanned pages ({failed_pages}): {error}",
                engine=f"ocr:dpi={dpi}:psm={psm}",
            )

        offset, run_length = pass1._detect_offset(page_candidates)
        if offset is None:
            return PaperResult(
                paper_id=paper_id,
                status="no_consistent_run",
                run_length=run_length,
                path=str(pdf_path),
                engine=f"ocr:dpi={dpi}:psm={psm}",
            )
        return PaperResult(
            paper_id=paper_id,
            status="detected",
            offset=offset,
            run_length=run_length,
            path=str(pdf_path),
            engine=f"ocr:dpi={dpi}:psm={psm}",
        )
    except TesseractMissingError as exc:
        return PaperResult(paper_id=paper_id, status="tesseract_missing", path=str(pdf_path), error=str(exc))
    except pass1.PDFTimeoutError as exc:
        return PaperResult(paper_id=paper_id, status="timeout", path=str(pdf_path), error=str(exc))
    except Exception as exc:
        return PaperResult(paper_id=paper_id, status="parse_error", path=str(pdf_path), error=str(exc))


def main() -> int:
    if not shutil.which("tesseract"):
        print(
            "FATAL: tesseract binary not found on PATH. "
            "Install via `brew install tesseract`.",
            file=sys.stderr,
        )
        return 1

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    parser = argparse.ArgumentParser(
        description="Backfill pdf_page_offset via OCR for image-based PDF footers."
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
            "detections where the OCR latched onto a year, ISBN, or article id."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=pass1._positive_int,
        default=DEFAULT_DPI,
        help=f"rasterization DPI for footer OCR (default: {DEFAULT_DPI})",
    )
    parser.add_argument(
        "--psm",
        type=pass1._positive_int,
        default=DEFAULT_PSM,
        help=f"Tesseract page segmentation mode (default: {DEFAULT_PSM})",
    )
    args = parser.parse_args()

    started = time.monotonic()
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA busy_timeout = 120000")

    candidate_ids = _load_candidate_ids(conn, limit=args.limit, skip_verified=args.skip_verified)
    total = len(candidate_ids)
    print(
        f"[start] candidates={total} apply={args.apply} workers={args.workers} "
        f"timeout={args.timeout:g}s skip_verified={args.skip_verified} "
        f"dpi={args.dpi} psm={args.psm}",
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
    tesseract_missing = 0
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
        initializer=pass1._worker_init,
        initargs=(str(DB_PATH),),
    ) as executor:
        while next_index < total and len(futures) < max_workers and not _shutdown_requested:
            paper_id = candidate_ids[next_index]
            futures[executor.submit(_process_paper, paper_id, args.timeout, args.dpi, args.psm)] = paper_id
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
                    print(
                        f"[detected] {result.paper_id} offset={result.offset} "
                        f"run={result.run_length} engine={result.engine}",
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
                elif result.status == "tesseract_missing":
                    tesseract_missing += 1
                    print(f"[tesseract-missing] {result.paper_id} {result.error}", file=sys.stderr)
                else:
                    worker_crashes += 1
                    print(f"[worker-crash] {result.paper_id} {result.error}", file=sys.stderr)

                if args.apply:
                    action = pass1._apply_result(
                        conn,
                        result,
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

                if processed % 10 == 0 or processed == total or _shutdown_requested:
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
                        executor.submit(_process_paper, next_paper_id, args.timeout, args.dpi, args.psm)
                    ] = next_paper_id
                    next_index += 1

        if _shutdown_requested:
            print("[signal] No new work submitted. Exiting after in-flight tasks.", file=sys.stderr)

    if args.apply and pending_writes:
        conn.commit()

    elapsed = time.monotonic() - started
    error_count = missing_pdf + timeout_count + parse_errors + worker_crashes
    print(
        f"[summary] processed={processed} detected={detected} no_run={no_run} "
        f"errors={error_count} tesseract_missing={tesseract_missing} "
        f"applied_detected={applied_detected} elapsed={pass1._format_seconds(elapsed)}",
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
            f"[summary] applied_cleared={applied_cleared} "
            f"applied_rejected_implausible={applied_rejected} "
            f"(max_abs_offset={args.max_abs_offset})",
            file=sys.stderr,
        )

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
