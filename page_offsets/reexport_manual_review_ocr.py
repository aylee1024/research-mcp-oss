#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pymupdf>=1.24.0",
#     "pytesseract>=0.3.10",
#     "Pillow>=10.0.0",
# ]
# ///
"""Re-export manual page-offset review rows with OCR footer samples.

This is a one-off CSV-to-CSV rewrite. It does not touch papers.db.

For each non-YES row in the source CSV, the script:
1. Opens the PDF with PyMuPDF.
2. Rasterizes the first 5 physical pages at 200 DPI.
3. Crops the bottom 10 percent of each page.
4. OCRs the crop with Tesseract using `--psm 7`.
5. Replaces sample_footer_p1..p5 with normalized OCR output.
6. Clears manual_offset/manual_confirmed for re-classification.

Rows already marked YES are preserved unchanged.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_mcp.paths import RESEARCH_MCP_HOME  # noqa: E402

DEFAULT_INPUT = RESEARCH_MCP_HOME / "manual_review_offsets.filled.csv"
DEFAULT_OUTPUT = RESEARCH_MCP_HOME / "manual_review_offsets_ocr.csv"
DEFAULT_WORKERS = 9
DEFAULT_TIMEOUT = 60.0
DEFAULT_DPI = 200
DEFAULT_PSM = 7
FOOTER_RATIO = 0.10
FOOTER_SAMPLE_PAGES = 5
FOOTER_SAMPLE_CHARS = 80

_TESSERACT_BIN = shutil.which("tesseract")
_shutdown_requested = False


class PaperTimeoutError(TimeoutError):
    """Raised when a single PDF exceeds its processing budget."""


@dataclass(slots=True)
class OCRTask:
    row_index: int
    paper_id: str
    pdf_path: str


@dataclass(slots=True)
class OCRResult:
    row_index: int
    paper_id: str
    sample_footers: tuple[str, str, str, str, str]
    error: str | None = None


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


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split())


def _remaining_budget(deadline: float, total_budget: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise PaperTimeoutError(f"timed out after {total_budget:.1f}s")
    return remaining


def _on_signal(sig: int, frame: object) -> None:
    del frame
    global _shutdown_requested
    _shutdown_requested = True
    name = signal.Signals(sig).name
    print(f"\n[signal] Received {name}. Finishing in-flight OCR...", file=sys.stderr)


def _pixmap_to_image(pix: object) -> object:
    import fitz
    from PIL import Image

    if getattr(pix, "alpha", 0) or getattr(pix, "n", 0) not in (1, 3):
        pix = fitz.Pixmap(fitz.csRGB, pix)

    mode = "L" if pix.n == 1 else "RGB"
    return Image.frombytes(mode, [pix.width, pix.height], pix.samples)


def _extract_footer_samples_ocr(pdf_path: Path, timeout_seconds: float) -> OCRResult:
    import fitz
    import pytesseract

    if _TESSERACT_BIN is None:
        raise RuntimeError("tesseract binary not found on PATH")

    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_BIN

    samples = [""] * FOOTER_SAMPLE_PAGES
    page_errors: list[str] = []
    deadline = time.monotonic() + timeout_seconds

    doc = fitz.open(str(pdf_path))
    try:
        if doc.needs_pass and not doc.authenticate(""):
            raise RuntimeError("encrypted PDF")

        page_limit = min(doc.page_count, FOOTER_SAMPLE_PAGES)
        config = f"--psm {DEFAULT_PSM}"

        for page_index in range(page_limit):
            _remaining_budget(deadline, timeout_seconds)
            page = doc[page_index]
            pix = page.get_pixmap(dpi=DEFAULT_DPI, alpha=False)
            image = _pixmap_to_image(pix)
            width, height = image.size
            footer_top = int(height * (1.0 - FOOTER_RATIO))
            footer_image = image.crop((0, footer_top, width, height))

            try:
                text = pytesseract.image_to_string(
                    footer_image,
                    config=config,
                    timeout=max(1.0, _remaining_budget(deadline, timeout_seconds)),
                )
                samples[page_index] = _normalize_text(text)[:FOOTER_SAMPLE_CHARS]
            except pytesseract.pytesseract.TesseractNotFoundError as exc:
                raise RuntimeError(str(exc)) from exc
            except pytesseract.TesseractError as exc:
                page_errors.append(
                    f"page {page_index + 1}: {type(exc).__name__}: {exc}"
                )
            except RuntimeError as exc:
                page_errors.append(
                    f"page {page_index + 1}: {type(exc).__name__}: {exc}"
                )
    finally:
        doc.close()

    error = "; ".join(page_errors) if page_errors else None
    return OCRResult(
        row_index=-1,
        paper_id="",
        sample_footers=tuple(samples),  # type: ignore[arg-type]
        error=error,
    )


def _process_task(task: OCRTask, timeout_seconds: float) -> OCRResult:
    pdf_path = Path(task.pdf_path).expanduser()
    if not task.pdf_path:
        return OCRResult(
            row_index=task.row_index,
            paper_id=task.paper_id,
            sample_footers=("", "", "", "", ""),
            error="missing_pdf_path",
        )
    if not pdf_path.exists():
        return OCRResult(
            row_index=task.row_index,
            paper_id=task.paper_id,
            sample_footers=("", "", "", "", ""),
            error=f"missing_pdf:{pdf_path}",
        )

    try:
        extracted = _extract_footer_samples_ocr(pdf_path, timeout_seconds)
        return OCRResult(
            row_index=task.row_index,
            paper_id=task.paper_id,
            sample_footers=extracted.sample_footers,
            error=extracted.error,
        )
    except PaperTimeoutError as exc:
        return OCRResult(
            row_index=task.row_index,
            paper_id=task.paper_id,
            sample_footers=("", "", "", "", ""),
            error=str(exc),
        )
    except Exception as exc:
        return OCRResult(
            row_index=task.row_index,
            paper_id=task.paper_id,
            sample_footers=("", "", "", "", ""),
            error=str(exc),
        )


def _load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("CSV is missing a header row.")
        return list(reader), fieldnames


def _build_tasks(rows: list[dict[str, str]], limit: int) -> list[OCRTask]:
    tasks: list[OCRTask] = []
    for row_index, row in enumerate(rows):
        confirmed = (row.get("manual_confirmed") or "").strip().upper()
        if confirmed == "YES":
            continue
        tasks.append(
            OCRTask(
                row_index=row_index,
                paper_id=(row.get("paper_id") or "").strip(),
                pdf_path=(row.get("local_pdf_path") or "").strip(),
            )
        )
    if limit:
        return tasks[:limit]
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-export manual review CSV rows with OCR footer samples."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"source CSV path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"destination CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="process at most N non-YES rows",
    )
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
        help=f"per-paper timeout in seconds (default: {DEFAULT_TIMEOUT:g})",
    )
    return parser.parse_args()


def main() -> int:
    if _TESSERACT_BIN is None:
        print(
            "FATAL: tesseract binary not found on PATH. Install via `brew install tesseract`.",
            file=sys.stderr,
        )
        return 1

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    args = parse_args()
    started = time.monotonic()

    rows, fieldnames = _load_rows(args.input_csv)
    tasks = _build_tasks(rows, args.limit)
    kept_yes = sum(
        1
        for row in rows
        if (row.get("manual_confirmed") or "").strip().upper() == "YES"
    )

    print(
        f"[start] rows={len(rows)} kept_yes={kept_yes} "
        f"ocr_targets={len(tasks)} workers={args.workers} timeout={args.timeout:g}s",
        file=sys.stderr,
    )

    if tasks:
        completed = 0
        error_count = 0
        with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_process_task, task, args.timeout): task for task in tasks
            }
            for future in cf.as_completed(futures):
                task = futures[future]
                result = future.result()
                row = rows[result.row_index]
                for sample_index, sample in enumerate(result.sample_footers, start=1):
                    row[f"sample_footer_p{sample_index}"] = sample
                row["manual_offset"] = ""
                row["manual_confirmed"] = ""
                if result.error:
                    error_count += 1
                    print(
                        f"[warn] {task.paper_id} {result.error}",
                        file=sys.stderr,
                    )
                completed += 1
                if completed == len(tasks) or completed % 25 == 0:
                    elapsed = time.monotonic() - started
                    print(
                        f"[progress] {completed}/{len(tasks)} done "
                        f"errors={error_count} elapsed={_format_seconds(elapsed)}",
                        file=sys.stderr,
                    )
                if _shutdown_requested:
                    break

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.monotonic() - started
    print(
        f"[done] wrote {args.output} in {_format_seconds(elapsed)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
