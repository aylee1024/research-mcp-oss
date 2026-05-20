# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx",
# ]
# ///
"""Periodic retraction refresh for the local paper library.

Iterates over every paper in papers.db that has an openalex_id, batches them
into groups of 50, asks OpenAlex `/works?filter=openalex:W1|W2|...` for the
current `is_retracted` flag, and updates the local row when the upstream
status differs.

Two transitions are logged loudly to stderr:
  0->1  newly retracted upstream — verify any work that cites it
  1->0  retraction reversed (rare) — flag clear

Why: papers can be retracted *after* we ingest them. The local
`_store_openalex_work()` only sets `is_retracted` on fresh fetches, so existing
rows drift out of sync silently. This script closes that loop.

Schedule: launchd weekly (Sunday 03:00). Manual run also fine.

Concurrency: the MCP server writes to the same papers.db. SQLite WAL mode lets
the server keep reading while we write, and serializes the two writers via the
busy timeout (30s). To minimize the time we hold the write lock, we commit
immediately after any batch that contained at least one UPDATE rather than
batching commits across many batches.

Progress JSON at retraction_refresh_progress.json records the last successful
run timestamp, used for observability. The script does NOT resume from a
partial run — at ~38 seconds for a full pass, just rerun from the start.

Usage:
    uv run retraction_refresh.py            # full pass, writes to DB
    uv run retraction_refresh.py --dry-run  # check + log only, no UPDATE
    uv run retraction_refresh.py --limit 100  # cap papers checked (testing)
"""

import argparse
import asyncio
import json
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_mcp.paths import PAPERS_DB_PATH as DB_PATH, RESEARCH_MCP_HOME  # noqa: E402

PROGRESS_PATH = RESEARCH_MCP_HOME / "retraction_refresh_progress.json"

OA_BASE = "https://api.openalex.org"
OA_API_KEY = os.environ.get("OPENALEX_API_KEY", "")
OA_MAILTO = os.environ.get("OPENALEX_MAILTO", os.environ.get("UNPAYWALL_EMAIL", ""))

# OpenAlex documented max = 10 RPS post Feb 2026 pricing change. Use 8 RPS.
OA_INTERVAL = 0.125

# OpenAlex per_page max for /works.
BATCH_SIZE = 50

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[signal] Shutdown requested — committing current batch and exiting", file=sys.stderr)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


class Throttle:
    """Async rate limiter — copied from server.py:_Throttle to keep this script
    independent of the MCP module's heavy import graph (torch, sentence-transformers)."""
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last = time.monotonic()


def _load_progress() -> dict:
    if PROGRESS_PATH.exists():
        try:
            return json.loads(PROGRESS_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {"last_run": None, "last_batch_index": 0, "total_checked": 0}


def _save_progress(progress: dict) -> None:
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2))


def _connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _extract_w_id(openalex_id: str) -> str:
    """Extract short W-ID from a stored openalex_id field.

    Stored values come in two shapes:
        https://openalex.org/W2890058952  (canonical, what _store_openalex_work writes)
        W2890058952                       (already-short, defensive)

    Returns the short form. Returns "" if the input doesn't look like a W-ID
    (caller should skip the row)."""
    if not openalex_id:
        return ""
    short = openalex_id.rsplit("/", 1)[-1].strip()
    if short.startswith("W") and short[1:].isdigit():
        return short
    return ""


async def fetch_batch(
    client: httpx.AsyncClient, throttle: Throttle, w_ids: list[str]
) -> tuple[dict[str, bool], bool]:
    """Look up `is_retracted` for a batch of W-IDs.

    Returns (results, ok) where results is {W-ID: bool} and ok is True iff the
    HTTP request succeeded. Missing IDs in a successful response (paper deleted
    or merged at OpenAlex) are silently omitted from the dict — caller treats
    those as "no signal".

    On 429 the function honors the Retry-After header (or sleeps 60s) and
    retries once. On any other HTTP error or timeout, returns ({}, False) so
    the caller can track the batch as failed and surface it in the summary."""
    if not w_ids:
        return {}, True
    params: dict = {
        "filter": "openalex:" + "|".join(w_ids),
        "per_page": len(w_ids),
        "select": "id,is_retracted",
    }
    if OA_API_KEY:
        params["api_key"] = OA_API_KEY
    elif OA_MAILTO:
        params["mailto"] = OA_MAILTO

    for attempt in (1, 2):
        await throttle.wait()
        try:
            resp = await client.get(f"{OA_BASE}/works", params=params, timeout=30)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            if attempt == 1:
                print(
                    f"  [warn] OpenAlex transport error ({len(w_ids)} ids), retrying once: {e}",
                    file=sys.stderr,
                )
                await asyncio.sleep(5)
                continue
            print(f"  [warn] OpenAlex batch failed after retry ({len(w_ids)} ids): {e}", file=sys.stderr)
            return {}, False

        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after", "")
            try:
                wait_s = max(1, int(retry_after)) if retry_after else 60
            except ValueError:
                wait_s = 60
            if attempt == 1:
                print(
                    f"  [warn] OpenAlex 429 rate-limited, sleeping {wait_s}s before retry",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait_s)
                continue
            print(
                f"  [warn] OpenAlex 429 persisted after retry, marking batch failed",
                file=sys.stderr,
            )
            return {}, False

        if resp.status_code >= 400:
            print(
                f"  [warn] OpenAlex HTTP {resp.status_code} ({len(w_ids)} ids): {resp.text[:200]}",
                file=sys.stderr,
            )
            return {}, False

        # Success.
        break
    else:
        # Loop exhausted without break (shouldn't happen given the explicit returns above).
        return {}, False

    out: dict[str, bool] = {}
    try:
        for w in resp.json().get("results", []):
            oa_url = w.get("id", "")
            short = _extract_w_id(oa_url)
            if short:
                out[short] = bool(w.get("is_retracted", False))
    except (ValueError, KeyError) as e:
        print(f"  [warn] OpenAlex response parse error: {e}", file=sys.stderr)
        return {}, False
    return out, True


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Check + log only; do not UPDATE the DB.")
    ap.add_argument("--limit", type=int, default=0, help="Cap papers checked (0 = all).")
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"[error] papers.db not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    started = datetime.now(timezone.utc).isoformat()
    progress = _load_progress()
    print(f"[start] retraction refresh @ {started}  (dry_run={args.dry_run})", file=sys.stderr)
    if progress.get("last_run"):
        print(f"  last successful run: {progress['last_run']}", file=sys.stderr)

    conn = _connect_db()

    # Build the working set. Skip papers with no openalex_id and rows where the
    # ID can't be parsed into a W-ID.
    rows = conn.execute("""
        SELECT paper_id, openalex_id, is_retracted, title, doi, year
        FROM papers
        WHERE openalex_id IS NOT NULL AND openalex_id != ''
        ORDER BY paper_id
    """).fetchall()

    work_items: list[tuple[str, str, int, str, str, int]] = []  # (pid, w_id, current_retracted, title, doi, year)
    for pid, oa_id, is_retracted, title, doi, year in rows:
        w_id = _extract_w_id(oa_id or "")
        if not w_id:
            continue
        work_items.append((pid, w_id, int(is_retracted or 0), title or "", doi or "", year))

    if args.limit > 0:
        work_items = work_items[: args.limit]

    total = len(work_items)
    print(f"  papers eligible: {total} (with parseable openalex W-ID)", file=sys.stderr)
    if total == 0:
        print("[done] nothing to check", file=sys.stderr)
        return

    throttle = Throttle(OA_INTERVAL)
    newly_retracted: list[tuple[str, str, str, object]] = []  # (pid, title, doi, year)
    de_retracted: list[tuple[str, str, str, object]] = []
    checked = 0
    updated = 0
    failed_batches = 0
    missing_from_oa = 0
    batch_count = (total + BATCH_SIZE - 1) // BATCH_SIZE

    async with httpx.AsyncClient() as client:
        for batch_idx in range(batch_count):
            if _shutdown_requested:
                print(f"  [signal] stopping at batch {batch_idx}/{batch_count}", file=sys.stderr)
                break

            batch = work_items[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
            w_ids = [item[1] for item in batch]
            results, ok = await fetch_batch(client, throttle, w_ids)
            checked += len(batch)
            if not ok:
                failed_batches += 1
                # Skip comparison/update for this batch — we have no data.
                if (batch_idx + 1) % 20 == 0 or batch_idx + 1 == batch_count:
                    print(
                        f"  progress: batch {batch_idx + 1}/{batch_count}  checked={checked}/{total}  "
                        f"newly_retracted={len(newly_retracted)}  reversed={len(de_retracted)}  "
                        f"failed_batches={failed_batches}",
                        file=sys.stderr,
                    )
                continue

            # One timestamp per batch — every UPDATE in this batch shares it.
            now_iso = datetime.now(timezone.utc).isoformat()
            batch_had_updates = False

            for pid, w_id, current, title, doi, year in batch:
                if w_id not in results:
                    missing_from_oa += 1
                    continue  # OpenAlex omitted this paper (deleted/merged) — leave alone
                upstream = 1 if results[w_id] else 0
                if upstream == current:
                    continue
                year_str = str(year) if year is not None else "?"
                if upstream == 1 and current == 0:
                    newly_retracted.append((pid, title, doi, year))
                    print(
                        f"  [RETRACTED 0->1] {title[:80]} ({year_str}) doi={doi or '-'} pid={pid}",
                        file=sys.stderr,
                    )
                else:  # upstream == 0 and current == 1
                    de_retracted.append((pid, title, doi, year))
                    print(
                        f"  [REVERSED 1->0] {title[:80]} ({year_str}) doi={doi or '-'} pid={pid}",
                        file=sys.stderr,
                    )
                if not args.dry_run:
                    conn.execute(
                        "UPDATE papers SET is_retracted = ?, last_updated = ? WHERE paper_id = ?",
                        (upstream, now_iso, pid),
                    )
                    updated += 1
                    batch_had_updates = True

            # Commit immediately after any batch that wrote, so we don't hold the
            # WAL write lock across multiple batches and starve the MCP server.
            if batch_had_updates and not args.dry_run:
                conn.commit()

            if (batch_idx + 1) % 20 == 0 or batch_idx + 1 == batch_count:
                print(
                    f"  progress: batch {batch_idx + 1}/{batch_count}  checked={checked}/{total}  "
                    f"newly_retracted={len(newly_retracted)}  reversed={len(de_retracted)}  "
                    f"failed_batches={failed_batches}",
                    file=sys.stderr,
                )

    if not args.dry_run:
        conn.commit()
    conn.close()

    finished = datetime.now(timezone.utc).isoformat()
    if not args.dry_run and not _shutdown_requested:
        progress = {
            "last_run": finished,
            "total_checked": checked,
            "failed_batches": failed_batches,
        }
        _save_progress(progress)

    # End-of-run summary.
    print("", file=sys.stderr)
    print(
        f"[done] checked={checked} updated={updated} failed_batches={failed_batches} "
        f"missing_from_oa={missing_from_oa} dry_run={args.dry_run}",
        file=sys.stderr,
    )
    print(f"  newly retracted (0->1): {len(newly_retracted)}", file=sys.stderr)
    for pid, title, doi, year in newly_retracted:
        year_str = str(year) if year is not None else "?"
        print(f"    - {title[:90]} ({year_str})  doi={doi or '-'}  pid={pid}", file=sys.stderr)
    print(f"  reversed (1->0): {len(de_retracted)}", file=sys.stderr)
    for pid, title, doi, year in de_retracted:
        year_str = str(year) if year is not None else "?"
        print(f"    - {title[:90]} ({year_str})  doi={doi or '-'}  pid={pid}", file=sys.stderr)
    print(f"[finished] {finished}", file=sys.stderr)

    # Exit non-zero if more than 5% of batches failed. launchd will record the
    # failure, and if many runs in a row fail it'll back off the schedule.
    if batch_count > 0 and failed_batches / batch_count > 0.05:
        print(
            f"[error] {failed_batches}/{batch_count} batches failed (>5%) — exiting non-zero",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
