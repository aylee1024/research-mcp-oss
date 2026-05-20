# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Build JSTOR lookup SQLite DB from metadata JSONL export.

Usage: uv run maintenance/build_jstor_db.py <path_to_jsonl>

Creates the JSTOR sidecar at $JSTOR_DB_PATH (defaults to
$RESEARCH_MCP_HOME/jstor.db) with FTS5 on titles and an index on ISSNs.
Only ingests articles (skips books, reports, contributed content).

A JSTOR sidecar lets `check_jstor` report whether a paper is in JSTOR
(useful for users with institutional access). It is fully optional;
research-mcp works without it.
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_mcp.paths import JSTOR_DB_PATH as DB_PATH  # noqa: E402
BATCH_SIZE = 50_000


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS jstor_articles (
            rowid INTEGER PRIMARY KEY,
            item_id TEXT UNIQUE,
            ithaka_doi TEXT,
            title TEXT,
            journal TEXT,
            creators TEXT,
            print_issn TEXT,
            online_issn TEXT,
            published_date TEXT,
            disciplines TEXT,
            content_subtype TEXT,
            url TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_jstor_print_issn ON jstor_articles(print_issn);
        CREATE INDEX IF NOT EXISTS idx_jstor_online_issn ON jstor_articles(online_issn);
        CREATE INDEX IF NOT EXISTS idx_jstor_doi ON jstor_articles(ithaka_doi);

        CREATE VIRTUAL TABLE IF NOT EXISTS jstor_fts USING fts5(
            title, journal,
            content=jstor_articles, content_rowid=rowid
        );

        CREATE TRIGGER IF NOT EXISTS jstor_ai AFTER INSERT ON jstor_articles BEGIN
            INSERT INTO jstor_fts(rowid, title, journal)
            VALUES (new.rowid, new.title, new.journal);
        END;

        CREATE TRIGGER IF NOT EXISTS jstor_ad AFTER DELETE ON jstor_articles BEGIN
            INSERT INTO jstor_fts(jstor_fts, rowid, title, journal)
            VALUES ('delete', old.rowid, old.title, old.journal);
        END;

        CREATE TRIGGER IF NOT EXISTS jstor_au AFTER UPDATE ON jstor_articles BEGIN
            INSERT INTO jstor_fts(jstor_fts, rowid, title, journal)
            VALUES ('delete', old.rowid, old.title, old.journal);
            INSERT INTO jstor_fts(rowid, title, journal)
            VALUES (new.rowid, new.title, new.journal);
        END;
    """)


def ingest(jsonl_path: str) -> None:
    path = Path(jsonl_path)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Building JSTOR DB at {DB_PATH}")
    print(f"Reading {path} ({path.stat().st_size / 1e9:.1f} GB)")

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-200000")  # 200MB cache
    create_schema(conn)

    batch = []
    total = 0
    skipped = 0
    t0 = time.monotonic()

    with open(path) as f:
        for line in f:
            total += 1
            r = json.loads(line)

            if r.get("content_type") != "article":
                skipped += 1
                continue

            title = r.get("title")
            if not title:
                skipped += 1
                continue

            ids = r.get("identifiers") or {}
            creators = r.get("creators_string") or ""
            disciplines = r.get("discipline_names") or []

            batch.append((
                r.get("item_id", ""),
                r.get("ithaka_doi", ""),
                title,
                r.get("is_part_of", ""),
                creators,
                ids.get("print_issn") or "",
                ids.get("online_issn") or "",
                r.get("published_date", ""),
                json.dumps(disciplines) if disciplines else "[]",
                r.get("content_subtype", ""),
                r.get("url", ""),
            ))

            if len(batch) >= BATCH_SIZE:
                conn.executemany("""
                    INSERT OR IGNORE INTO jstor_articles
                    (item_id, ithaka_doi, title, journal, creators, print_issn,
                     online_issn, published_date, disciplines, content_subtype, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
                conn.commit()
                elapsed = time.monotonic() - t0
                rate = total / elapsed
                print(f"  {total:,} read, {total - skipped:,} articles, "
                      f"{rate:,.0f} lines/sec, {elapsed:.0f}s elapsed")
                batch = []

    if batch:
        conn.executemany("""
            INSERT OR IGNORE INTO jstor_articles
            (item_id, ithaka_doi, title, journal, creators, print_issn,
             online_issn, published_date, disciplines, content_subtype, url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)
        conn.commit()

    elapsed = time.monotonic() - t0
    articles = total - skipped
    print(f"\nDone: {total:,} records read, {articles:,} articles ingested, "
          f"{skipped:,} skipped, {elapsed:.0f}s")

    # Stats
    row = conn.execute("SELECT COUNT(*) FROM jstor_articles").fetchone()
    print(f"DB rows: {row[0]:,}")
    db_size = DB_PATH.stat().st_size / 1e6
    print(f"DB size: {db_size:.0f} MB")

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run build_jstor_db.py <path_to_jsonl>", file=sys.stderr)
        sys.exit(1)
    ingest(sys.argv[1])
