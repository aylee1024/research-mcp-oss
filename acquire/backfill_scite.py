# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx>=0.28,<1",
# ]
# ///
"""Backfill scite citation-reception tallies into the local paper library.

Walks papers.db rows that have a DOI and fetches scite's Smart-Citation tally
(supporting / contradicting / mentioning counts) from the token-less
`https://api.scite.ai/tallies/{doi}` endpoint, caching the raw counts on the
row. Discovery (search_papers/search_openalex) and verify_claim READ this cache
so interactive calls never block on scite's 40 req/min limit.

Why a separate script: scite is rate-limited to 40 req/min unauthenticated, so a
full pass over a large library is slow. This runs out-of-band and is incremental
and rerunnable — by default it only touches rows never fetched before
(scite_fetched_at IS NULL), so successive runs converge. Every fetch attempt
stamps scite_fetched_at (even for DOIs scite has no record of, whose counts stay
NULL) so the same DOI is never re-queried in the "never fetched" pass.

Schema: relies on the v22 columns added by server.py
(scite_supporting/contradicting/mentioning/total/scite_fetched_at). Run the MCP
server (or any tool) once first so the migration applies.

Concurrency: the MCP server writes the same papers.db. WAL mode lets it keep
reading while we write; we commit after every write so we don't hold the write
lock across the (throttled) network gaps.

Usage:
    uv run backfill_scite.py                  # fetch all never-fetched DOIs
    uv run backfill_scite.py --dry-run        # fetch + log, no DB write
    uv run backfill_scite.py --limit 200      # cap rows this run (testing / chunking)
    uv run backfill_scite.py --max-age-days 90 # also refresh rows older than 90 days
"""

import argparse
import asyncio
import json
import signal
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_mcp.paths import PAPERS_DB_PATH as DB_PATH, RESEARCH_MCP_HOME  # noqa: E402

PROGRESS_PATH = RESEARCH_MCP_HOME / "backfill_scite_progress.json"

SCITE_TALLIES_BASE = "https://api.scite.ai/tallies"
# scite's token-less limit is 40 req/min (and 10 req/s); 40/min is binding.
# 1.6s interval -> ~37 req/min, safely under.
SCITE_INTERVAL = 1.6

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[signal] Shutdown requested — finishing current row and exiting", file=sys.stderr)


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
    return {"last_run": None, "total_checked": 0}


def _save_progress(progress: dict) -> None:
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2))


def _connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _is_contested(supporting: int, contradicting: int) -> bool:
    """Mirror of server.py:_scite_is_contested (counts only; for the run summary)."""
    return contradicting >= 2 and contradicting >= 0.5 * max(supporting, 1)


async def fetch_tally(
    client: httpx.AsyncClient, throttle: Throttle, doi: str
) -> tuple[dict | None, bool]:
    """Fetch scite tallies for one DOI.

    Returns (tally, ok):
      tally = {supporting, contradicting, mentioning, total} on a 200,
              None when scite has no record (404) — a real, cacheable "no data".
      ok    = False only on a transient failure (429/5xx/transport) where the
              row should be left UNstamped so a later run retries it.

    On 429 honors Retry-After (or 30s) and retries once.
    """
    params_url = f"{SCITE_TALLIES_BASE}/{quote(doi, safe='/')}"
    for attempt in (1, 2):
        await throttle.wait()
        try:
            resp = await client.get(params_url, timeout=30)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            if attempt == 1:
                print(f"  [warn] transport error for {doi}, retrying once: {e}", file=sys.stderr)
                await asyncio.sleep(5)
                continue
            print(f"  [warn] transport error after retry for {doi}: {e}", file=sys.stderr)
            return None, False

        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after", "")
            try:
                wait_s = max(1, int(retry_after)) if retry_after else 30
            except ValueError:
                wait_s = 30
            if attempt == 1:
                print(f"  [warn] 429 rate-limited, sleeping {wait_s}s before retry", file=sys.stderr)
                await asyncio.sleep(wait_s)
                continue
            print("  [warn] 429 persisted after retry — leaving row unstamped", file=sys.stderr)
            return None, False

        if resp.status_code == 404:
            return None, True  # scite genuinely has no record — cacheable

        if resp.status_code >= 400:
            print(f"  [warn] HTTP {resp.status_code} for {doi}: {resp.text[:160]}", file=sys.stderr)
            return None, False

        try:
            d = resp.json()
        except ValueError as e:
            print(f"  [warn] bad JSON for {doi}: {e}", file=sys.stderr)
            return None, False
        if not isinstance(d, dict):
            return None, False
        return {
            "supporting": int(d.get("supporting") or 0),
            "contradicting": int(d.get("contradicting") or 0),
            "mentioning": int(d.get("mentioning") or 0),
            "total": int(d.get("total") or 0),
        }, True

    return None, False


async def main():
    ap = argparse.ArgumentParser(description="Backfill scite citation-reception tallies")
    ap.add_argument("--dry-run", action="store_true", help="Fetch + log only; do not write the DB.")
    ap.add_argument("--limit", type=int, default=0, help="Cap rows fetched this run (0 = all).")
    ap.add_argument(
        "--max-age-days", type=int, default=0,
        help="Also refresh rows whose scite_fetched_at is older than this many days (0 = only never-fetched).",
    )
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"[error] papers.db not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    conn = _connect_db()
    # Guard: the v22 columns must exist (run the server once to migrate).
    cols = {r[1] for r in conn.execute("PRAGMA table_info(papers)").fetchall()}
    required = {"scite_supporting", "scite_contradicting", "scite_mentioning", "scite_total", "scite_fetched_at"}
    missing = required - cols
    if missing:
        print(
            f"[error] papers table missing scite columns {sorted(missing)}. "
            f"Run the MCP server once to apply the v22 migration first.",
            file=sys.stderr,
        )
        conn.close()
        sys.exit(1)

    started = datetime.now(timezone.utc).isoformat()
    print(f"[start] scite backfill @ {started}  (dry_run={args.dry_run}, max_age_days={args.max_age_days})", file=sys.stderr)

    if args.max_age_days > 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=args.max_age_days)).isoformat()
        rows = conn.execute(
            "SELECT paper_id, doi FROM papers "
            "WHERE doi IS NOT NULL AND doi != '' "
            "AND (scite_fetched_at IS NULL OR scite_fetched_at < ?) "
            "ORDER BY scite_fetched_at IS NOT NULL, paper_id",
            (cutoff,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT paper_id, doi FROM papers "
            "WHERE doi IS NOT NULL AND doi != '' AND scite_fetched_at IS NULL "
            "ORDER BY paper_id"
        ).fetchall()

    if args.limit > 0:
        rows = rows[: args.limit]

    total = len(rows)
    print(f"  rows to fetch: {total}", file=sys.stderr)
    if total == 0:
        print("[done] nothing to fetch", file=sys.stderr)
        conn.close()
        return

    throttle = Throttle(SCITE_INTERVAL)
    checked = updated = no_record = failed = contested = 0

    async with httpx.AsyncClient() as client:
        for i, (paper_id, doi) in enumerate(rows, 1):
            if _shutdown_requested:
                print(f"  [signal] stopping at row {i}/{total}", file=sys.stderr)
                break

            tally, ok = await fetch_tally(client, throttle, doi)
            checked += 1
            if not ok:
                failed += 1
                continue  # leave unstamped so a later run retries

            now_iso = datetime.now(timezone.utc).isoformat()
            if tally is None:
                no_record += 1
                if not args.dry_run:
                    conn.execute(
                        "UPDATE papers SET scite_fetched_at = ? WHERE paper_id = ?",
                        (now_iso, paper_id),
                    )
                    conn.commit()
            else:
                if _is_contested(tally["supporting"], tally["contradicting"]):
                    contested += 1
                    print(
                        f"  [CONTESTED] {doi}  "
                        f"{tally['supporting']}+/{tally['contradicting']}- / {tally['mentioning']} mentions",
                        file=sys.stderr,
                    )
                if not args.dry_run:
                    conn.execute(
                        "UPDATE papers SET scite_supporting = ?, scite_contradicting = ?, "
                        "scite_mentioning = ?, scite_total = ?, scite_fetched_at = ? WHERE paper_id = ?",
                        (tally["supporting"], tally["contradicting"], tally["mentioning"],
                         tally["total"], now_iso, paper_id),
                    )
                    conn.commit()
                    updated += 1

            if i % 50 == 0 or i == total:
                print(
                    f"  progress: {i}/{total}  updated={updated}  no_record={no_record}  "
                    f"failed={failed}  contested={contested}",
                    file=sys.stderr,
                )

    conn.close()

    finished = datetime.now(timezone.utc).isoformat()
    if not args.dry_run and not _shutdown_requested:
        _save_progress({"last_run": finished, "total_checked": checked, "failed": failed})

    print("", file=sys.stderr)
    print(
        f"[done] checked={checked} updated={updated} no_record={no_record} "
        f"failed={failed} contested={contested} dry_run={args.dry_run}",
        file=sys.stderr,
    )
    print(f"[finished] {finished}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
