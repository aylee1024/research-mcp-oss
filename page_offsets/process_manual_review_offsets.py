#!/usr/bin/env python3
"""Fill manual offset review columns from sampled footer text."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])([0-9OoIl]{1,4})(?![A-Za-z0-9])")
TRANSLATION = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify detected page offsets from footer samples."
    )
    parser.add_argument("input_csv", type=Path, help="Path to the source CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the updated CSV. Defaults to the input path.",
    )
    return parser.parse_args()


def is_decimal_fragment(text: str, start: int, end: int) -> bool:
    prev_char = text[start - 1] if start > 0 else ""
    next_char = text[end] if end < len(text) else ""
    prev_prev = text[start - 2] if start > 1 else ""
    next_next = text[end + 1] if end + 1 < len(text) else ""
    return (prev_char == "." and prev_prev.isdigit()) or (
        next_char == "." and next_next.isdigit()
    )


def iter_numeric_tokens(text: str):
    for match in TOKEN_RE.finditer(text):
        token = match.group(1)
        start, end = match.span(1)
        if is_decimal_fragment(text, start, end):
            continue
        normalized = token.translate(TRANSLATION)
        if not normalized.isdigit():
            continue
        value = int(normalized)
        if value <= 0:
            continue
        yield value, token, start, end


def letters_only(text: str) -> str:
    return re.sub(r"[^A-Za-z]+", "", text)


def salient_tokens(text: str):
    sample = text or ""
    stripped = sample.strip()
    sample_len = len(sample)
    tokens = []
    for value, token, start, end in iter_numeric_tokens(sample):
        before = sample[:start]
        after = sample[end:]
        whole_token = stripped == token or (
            re.fullmatch(rf"[^A-Za-z0-9]*{re.escape(token)}[^A-Za-z0-9]*", stripped)
            is not None
        )
        after_page_label = re.search(r"Page\s*:?\s*$", before, re.IGNORECASE) is not None
        near_start = start <= 4 and not letters_only(before)
        near_end = sample_len - end <= 4 and not letters_only(after)
        if whole_token or after_page_label or near_start or near_end:
            tokens.append((value, token, start, end))
    return tokens


def longest_consecutive_run(values: list[int]) -> int:
    best = 0
    current = 0
    previous = None
    for value in values:
        if previous is not None and value == previous + 1:
            current += 1
        else:
            current = 1
        best = max(best, current)
        previous = value
    return best


def classify_row(row: dict[str, str]) -> tuple[str, str]:
    samples = [(row.get(f"sample_footer_p{i}") or "") for i in range(1, 6)]
    if all(not sample.strip() for sample in samples):
        return "", "NO"

    salient_by_page: dict[int, list[tuple[int, str, int, int]]] = {}
    supports: dict[int, set[int]] = defaultdict(set)
    for page_index, sample in enumerate(samples, start=1):
        tokens = salient_tokens(sample)
        salient_by_page[page_index] = tokens
        for value, *_ in tokens:
            supports[page_index - value].add(page_index)

    if not supports:
        return "", "NO"

    heuristic_offset_raw = (row.get("heuristic_detected_offset") or "").strip()
    heuristic_offset = int(heuristic_offset_raw) if heuristic_offset_raw else None

    scored_offsets = []
    for offset, page_set in supports.items():
        pages = sorted(page_set)
        scored_offsets.append(
            (
                len(pages),
                longest_consecutive_run(pages),
                int(offset == heuristic_offset) if heuristic_offset is not None else 0,
                -abs(offset),
                offset,
                pages,
            )
        )

    support_count, longest_run, _, _, best_offset, best_pages = max(scored_offsets)
    run_length_raw = (row.get("run_length") or "").strip()
    run_length = int(run_length_raw) if run_length_raw else 0
    other_salient_pages = [
        page_index
        for page_index, tokens in salient_by_page.items()
        if tokens and page_index not in best_pages
    ]

    if abs(best_offset) <= 2000:
        if support_count >= 4 and (run_length >= 3 or best_pages == [1, 2, 3, 4, 5]):
            return str(best_offset), "YES"
        if (
            support_count == 3
            and longest_run == 3
            and run_length >= 3
            and not other_salient_pages
        ):
            return str(best_offset), "YES"
        if run_length < 3 and support_count == 5 and best_pages == [1, 2, 3, 4, 5]:
            return str(best_offset), "YES"

    if abs(best_offset) > 2000 and support_count >= 4:
        return "", "NO"

    if support_count >= 2:
        return "", ""

    return "", "NO"


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv
    output_csv = args.output or input_csv

    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("CSV is missing a header row.")
        rows = list(reader)

    yes_count = 0
    no_count = 0
    blank_count = 0

    for row in rows:
        manual_offset, manual_confirmed = classify_row(row)
        row["manual_offset"] = manual_offset
        row["manual_confirmed"] = manual_confirmed
        if manual_confirmed == "YES":
            yes_count += 1
        elif manual_confirmed == "NO":
            no_count += 1
        else:
            blank_count += 1

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Total rows processed: {len(rows)} | YES: {yes_count} | "
        f"NO: {no_count} | blank/deferred: {blank_count}"
    )


if __name__ == "__main__":
    main()
