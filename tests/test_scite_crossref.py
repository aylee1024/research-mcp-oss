# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pytest",
#     "pytest-asyncio",
#     "respx",
#     "httpx",
#     "mcp[cli]",
#     "sqlite-vec",
#     "numpy",
# ]
# ///
"""Tests for the scite citation-reception + shared title->DOI resolver work
(2026-06-23). Self-contained PEP-723 script — run with:

    uv run tests/test_scite_crossref.py

Points PAPERS_DB/JSTOR_DB at a throwaway temp DB *before* importing server so
the v22 migration runs against scratch state and the live library is untouched.
External HTTP is mocked with respx; no network is hit. server.py's heavy deps
(torch/mlx/sentence-transformers) are lazy, so importing it here is cheap.
"""

import os
import sys
import tempfile
from pathlib import Path

# Redirect the whole research-mcp tree to scratch BEFORE importing server (it
# reads research_mcp.paths at import). Never touch the real library. The public
# repo honors RESEARCH_MCP_HOME / PAPERS_DB_PATH / JSTOR_DB_PATH (research_mcp/paths.py).
_TMP = Path(tempfile.mkdtemp(prefix="scite_test_"))
os.environ["RESEARCH_MCP_HOME"] = str(_TMP)
os.environ["PAPERS_DB_PATH"] = str(_TMP / "papers.db")
os.environ["JSTOR_DB_PATH"] = str(_TMP / "jstor.db")

import httpx
import pytest
import respx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import server  # noqa: E402


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_contested_rule_boundaries():
    # Needs >=2 contradictions AND contradictions >= half the supports.
    assert server._scite_is_contested(0, 2) is True
    assert server._scite_is_contested(4, 2) is True          # 2 >= 0.5*4
    assert server._scite_is_contested(605, 18) is False      # AlphaFold: not contested
    assert server._scite_is_contested(10, 1) is False        # only 1 contradiction
    assert server._scite_is_contested(0, 1) is False         # below absolute floor
    assert server._scite_is_contested(0, 0) is False


def test_normalize_doi_value_strips_prefixes():
    assert server._normalize_doi_value("https://doi.org/10.1/x") == "10.1/x"
    assert server._normalize_doi_value("http://dx.doi.org/10.1/x") == "10.1/x"
    assert server._normalize_doi_value("doi:10.1/x") == "10.1/x"
    assert server._normalize_doi_value("10.1/x") == "10.1/x"
    assert server._normalize_doi_value("") is None
    assert server._normalize_doi_value(None) is None


def test_title_similarity_rejects_derivative_record():
    # The exact Crossref trap: a "Faculty Opinions recommendation of <paper>"
    # record must score below the 0.85 gate so the resolver won't accept its DOI.
    q = "Highly accurate protein structure prediction with AlphaFold"
    derivative = "Faculty Opinions recommendation of Highly accurate protein structure prediction with AlphaFold."
    assert server._title_similarity(q, derivative) < 0.85
    assert server._title_similarity(q, q) == 1.0


# ---------------------------------------------------------------------------
# _fetch_scite_tally (network mocked)
# ---------------------------------------------------------------------------

@respx.mock
async def test_fetch_scite_tally_parses_200():
    respx.get(url__regex=r"https://api\.scite\.ai/tallies/.*").mock(
        return_value=httpx.Response(200, json={
            "total": 42050, "supporting": 605, "contradicting": 18,
            "mentioning": 41343, "unclassified": 84,
        })
    )
    async with httpx.AsyncClient() as client:
        tally, ok = await server._fetch_scite_tally(client, "10.1038/s41586-021-03819-2")
    assert ok is True
    assert tally == {"supporting": 605, "contradicting": 18, "mentioning": 41343, "total": 42050}


@respx.mock
async def test_fetch_scite_tally_404_definitive_vs_transient():
    # 404 = definitive no-record (cacheable: ok=True). 429/5xx/bad-json = transient
    # (ok=False) so callers must NOT cache them. Regression for gemini-2.
    route = respx.get(url__regex=r"https://api\.scite\.ai/tallies/.*")
    async with httpx.AsyncClient() as client:
        route.mock(return_value=httpx.Response(404, json={"detail": "DOI not found"}))
        assert await server._fetch_scite_tally(client, "10.1/missing") == (None, True)
        route.mock(return_value=httpx.Response(500, text="oops"))
        assert await server._fetch_scite_tally(client, "10.1/err") == (None, False)
        route.mock(return_value=httpx.Response(429, text="slow down"))
        assert await server._fetch_scite_tally(client, "10.1/rl") == (None, False)
        route.mock(return_value=httpx.Response(200, text="not json"))
        assert await server._fetch_scite_tally(client, "10.1/badjson") == (None, False)
    assert await server._fetch_scite_tally(httpx.AsyncClient(), "") == (None, False)


@respx.mock
async def test_fetch_scite_tally_zeros_ok():
    respx.get(url__regex=r"https://api\.scite\.ai/tallies/.*").mock(
        return_value=httpx.Response(200, json={"total": 0, "supporting": 0, "contradicting": 0, "mentioning": 0})
    )
    async with httpx.AsyncClient() as client:
        tally, ok = await server._fetch_scite_tally(client, "10.1/zero")
    assert ok is True and tally == {"supporting": 0, "contradicting": 0, "mentioning": 0, "total": 0}


# ---------------------------------------------------------------------------
# _resolve_doi_by_title cascade (network mocked)
# ---------------------------------------------------------------------------

_S2_MATCH = r"https://api\.semanticscholar\.org/graph/v1/paper/search/match.*"
_OA_WORKS = r"https://api\.openalex\.org/works.*"
_CR_WORKS = r"https://api\.crossref\.org/works.*"
TITLE = "Highly accurate protein structure prediction with AlphaFold"


@respx.mock
async def test_resolve_doi_s2_short_circuits():
    s2 = respx.get(url__regex=_S2_MATCH).mock(return_value=httpx.Response(
        200, json={"data": [{"externalIds": {"DOI": "10.1038/s2hit"}, "matchScore": 0.95}]}))
    oa = respx.get(url__regex=_OA_WORKS).mock(return_value=httpx.Response(200, json={"results": []}))
    async with httpx.AsyncClient() as client:
        doi, src = await server._resolve_doi_by_title(client, TITLE)
    assert (doi, src) == ("10.1038/s2hit", "s2")
    assert s2.called and not oa.called  # OpenAlex never reached


@respx.mock
async def test_resolve_doi_openalex_when_s2_misses():
    respx.get(url__regex=_S2_MATCH).mock(return_value=httpx.Response(404, json={}))
    respx.get(url__regex=_OA_WORKS).mock(return_value=httpx.Response(
        200, json={"results": [{"title": TITLE, "doi": "https://doi.org/10.1038/oahit"}]}))
    async with httpx.AsyncClient() as client:
        doi, src = await server._resolve_doi_by_title(client, TITLE)
    assert (doi, src) == ("10.1038/oahit", "openalex")  # prefix stripped


@respx.mock
async def test_resolve_doi_crossref_last():
    respx.get(url__regex=_S2_MATCH).mock(return_value=httpx.Response(404, json={}))
    respx.get(url__regex=_OA_WORKS).mock(return_value=httpx.Response(200, json={"results": []}))
    respx.get(url__regex=_CR_WORKS).mock(return_value=httpx.Response(
        200, json={"message": {"items": [{"title": [TITLE], "DOI": "10.1038/crhit"}]}}))
    async with httpx.AsyncClient() as client:
        doi, src = await server._resolve_doi_by_title(client, TITLE)
    assert (doi, src) == ("10.1038/crhit", "crossref")


@respx.mock
async def test_resolve_doi_rejects_low_similarity_everywhere():
    # All three return a derivative/wrong title -> below the 0.85 gate -> no DOI.
    respx.get(url__regex=_S2_MATCH).mock(return_value=httpx.Response(404, json={}))
    respx.get(url__regex=_OA_WORKS).mock(return_value=httpx.Response(
        200, json={"results": [{"title": "A completely unrelated paper about cats", "doi": "10.1/wrong"}]}))
    respx.get(url__regex=_CR_WORKS).mock(return_value=httpx.Response(
        200, json={"message": {"items": [{
            "title": ["Faculty Opinions recommendation of Highly accurate protein structure prediction with AlphaFold."],
            "DOI": "10.3410/f.derivative"}]}}))
    async with httpx.AsyncClient() as client:
        doi, src = await server._resolve_doi_by_title(client, TITLE)
    assert (doi, src) == (None, None)


@respx.mock
async def test_resolve_doi_short_title_skips():
    async with httpx.AsyncClient() as client:
        assert await server._resolve_doi_by_title(client, "short") == (None, None)


# ---------------------------------------------------------------------------
# Schema migration + cache read/write (scratch DB)
# ---------------------------------------------------------------------------

def test_migration_creates_v22_columns_and_version():
    conn = server._init_db()
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(papers)").fetchall()}
        for c in ("scite_supporting", "scite_contradicting", "scite_mentioning",
                  "scite_total", "scite_fetched_at"):
            assert c in cols, f"missing column {c}"
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 22
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 22
    finally:
        conn.close()


def test_migration_is_idempotent():
    conn = server._init_db()
    try:
        # Re-running the raw migration SQL on the same connection must not raise.
        server._run_schema_migrations(conn)
        server._run_schema_migrations(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(papers)").fetchall()}
        assert "scite_total" in cols
    finally:
        conn.close()


def test_update_and_cached_scite_tally_roundtrip():
    conn = server._init_db()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO papers (paper_id, doi, title) VALUES (?, ?, ?)",
            ("pidA", "10.1/contested", "A"),
        )
        conn.commit()
        server._update_scite_tally(conn, "pidA", {
            "supporting": 1, "contradicting": 5, "mentioning": 10, "total": 16})

        by_pid = server._cached_scite_tally(conn, paper_id="pidA")
        assert by_pid == {"supporting": 1, "contradicting": 5, "mentioning": 10,
                          "total": 16, "contested": True}
        by_doi = server._cached_scite_tally(conn, doi="10.1/CONTESTED")  # case-insensitive
        assert by_doi == by_pid
    finally:
        conn.close()


def test_cached_scite_tally_none_when_unfetched_or_empty():
    conn = server._init_db()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO papers (paper_id, doi, title) VALUES (?, ?, ?)",
            ("pidUnfetched", "10.1/never", "B"),
        )
        conn.commit()
        # Never fetched (scite_fetched_at NULL) -> None.
        assert server._cached_scite_tally(conn, paper_id="pidUnfetched") is None

        # Fetched but scite had no record (None tally) -> stamped, counts NULL -> None.
        server._update_scite_tally(conn, "pidUnfetched", None)
        assert server._cached_scite_tally(conn, paper_id="pidUnfetched") is None

        # Fetched, all-zero counts -> None (nothing worth showing).
        conn.execute(
            "INSERT OR REPLACE INTO papers (paper_id, doi, title) VALUES (?, ?, ?)",
            ("pidZero", "10.1/zero", "C"),
        )
        conn.commit()
        server._update_scite_tally(conn, "pidZero", {
            "supporting": 0, "contradicting": 0, "mentioning": 0, "total": 0})
        assert server._cached_scite_tally(conn, paper_id="pidZero") is None
    finally:
        conn.close()


def test_update_scite_tally_none_clears_stale_counts():
    # Regression for codex-integration-2: a later no-record fetch must CLEAR prior
    # counts, not leave them stale behind a fresh scite_fetched_at.
    conn = server._init_db()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO papers (paper_id, doi, title, scite_supporting, "
            "scite_contradicting, scite_mentioning, scite_total, scite_fetched_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("pidStale", "10.1/stale", "S", 3, 2, 10, 15, "old"),
        )
        conn.commit()
        server._update_scite_tally(conn, "pidStale", None)
        row = conn.execute(
            "SELECT scite_supporting, scite_contradicting, scite_mentioning, scite_total "
            "FROM papers WHERE paper_id='pidStale'").fetchone()
        assert row == (None, None, None, None)
        assert server._cached_scite_tally(conn, paper_id="pidStale") is None
    finally:
        conn.close()


def test_init_db_v21_to_v22_upgrade_does_not_raise():
    # Regression for the CRITICAL panel finding: a clean v21 DB (schema_version
    # holds a single row {21}, no scite_* columns) must migrate to v22 WITHOUT the
    # strict-guard mismatch. Pre-fix, _init_db read `SELECT version ... LIMIT 1`,
    # got the lower row after the bump appended {22}, and raised on startup.
    import sqlite3 as _sq
    import sqlite_vec
    seed_path = Path(tempfile.mkdtemp(prefix="v21up_")) / "p.db"
    s = _sq.connect(str(seed_path))
    s.enable_load_extension(True)
    sqlite_vec.load(s)
    s.enable_load_extension(False)
    s.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
    s.execute("INSERT INTO schema_version(version) VALUES (21)")
    s.commit()
    s.close()

    old_path, old_flag = server.DB_PATH, server._db_initialized
    server.DB_PATH = seed_path
    server._db_initialized = False
    try:
        conn = server._init_db()  # pre-fix: RuntimeError "reports 21, constant is 22"
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 22
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 22
        cols = {r[1] for r in conn.execute("PRAGMA table_info(papers)").fetchall()}
        assert {"scite_supporting", "scite_total", "scite_fetched_at"} <= cols
        conn.close()
    finally:
        server.DB_PATH = old_path
        server._db_initialized = old_flag


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-o", "asyncio_mode=auto"]))
