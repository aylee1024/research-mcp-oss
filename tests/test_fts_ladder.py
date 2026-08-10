# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pytest>=8,<10",
#     "mcp[cli]>=2.0.0,<3",
#     "httpx>=0.28,<1",
#     "packaging>=24,<26",
#     "sqlite-vec>=0.1.9,<0.2",
#     "numpy>=2.5,<3",
# ]
# ///
"""Tests for the FTS5 query ladder.

The defect these guard against is silent: a query that is a syntax error, or
that is quoted into one unmatchable phrase, returns an empty list rather than
raising, so the lexical half of hybrid search disappears without a trace. Most
of the tests below therefore assert on *rows returned from a real FTS5 index*
rather than on the generated expression alone.

Run with:

    uv run tests/test_fts_ladder.py
"""

import os
import re
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="fts_ladder_test_"))
os.environ["RESEARCH_MCP_HOME"] = str(_TMP)
os.environ["PAPERS_DB_PATH"] = str(_TMP / "papers.db")
os.environ["JSTOR_DB_PATH"] = str(_TMP / "jstor.db")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import server  # noqa: E402

# A corpus small enough to reason about exactly. Doc 1 contains the query
# phrase verbatim; doc 2 contains every content word but scattered; doc 3
# shares only one uncommon word; doc 4 shares nothing.
DOCS = [
    (1, "Orbital debris mitigation guidelines constrain deployment of large constellations."),
    (2, "Mitigation of debris is discussed, and guidelines for orbital deployment differ."),
    (3, "Constellations of pulsars were catalogued by the survey."),
    (4, "Photosynthesis in deep ocean vents remains unexplained."),
]

SQL = """
    SELECT rowid, body
    FROM docs
    WHERE docs MATCH ?
    ORDER BY rank
    LIMIT ?
"""


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE VIRTUAL TABLE docs USING fts5(body)")
    connection.executemany("INSERT INTO docs (rowid, body) VALUES (?, ?)", DOCS)
    yield connection
    connection.close()


def legacy(conn, sql, query, limit):
    """The behaviour the ladder replaces, kept here to prove the regression."""
    try:
        return conn.execute(sql, (query, limit)).fetchall()
    except sqlite3.OperationalError:
        escaped = '"' + query.replace('"', '""') + '"'
        try:
            return conn.execute(sql, (escaped, limit)).fetchall()
        except sqlite3.OperationalError:
            return []


# --- tokenizer -------------------------------------------------------------


def test_tokenizer_splits_on_every_non_alphanumeric_character():
    assert server._fts5_tokenize("orbital-debris (mitigation), v2.0") == [
        "orbital",
        "debris",
        "mitigation",
        "v2",
        "0",
    ]


def test_tokenizer_keeps_accented_and_mixed_alphanumeric_runs_whole():
    """unicode61 treats letters and digits as token characters regardless of script."""
    assert server._fts5_tokenize("Poincaré CO2 β-decay") == ["Poincaré", "CO2", "β", "decay"]


def test_tokenizer_returns_nothing_for_punctuation_only_input():
    assert server._fts5_tokenize("!!! ... ???") == []


# --- ladder construction ---------------------------------------------------


def test_ladder_orders_rungs_from_precise_to_permissive():
    tiers = [tier for tier, _ in server._fts5_ladder("orbital debris mitigation")]
    assert tiers == ["exact_phrase", "near_10", "near_30", "any_term"]


def test_a_single_term_query_does_not_run_the_same_expression_twice():
    """Several rungs collapse to the same MATCH when the query is one word."""
    expressions = [expression for _, expression in server._fts5_ladder("plasmonics")]
    assert len(expressions) == len(set(expressions))
    assert expressions == ['"plasmonics"']


def test_the_ladder_has_no_conjunction_rung():
    """A conjunction can only reorder the disjunction's hits, and it reorders them worse."""
    for query in ("orbital debris", " ".join(f"term{i}" for i in range(30))):
        assert not any(tier == "all_terms" for tier, _ in server._fts5_ladder(query))


def test_ladder_skips_near_when_the_query_is_too_long_to_fit_a_window():
    over_the_limit = server._FTS_NEAR_MAX_TERMS + 1
    long_query = " ".join(f"term{i}" for i in range(over_the_limit))
    tiers = [tier for tier, _ in server._fts5_ladder(long_query)]
    assert not any(tier.startswith("near_") for tier in tiers)


def test_stopwords_are_dropped_from_wide_rungs_but_kept_in_the_phrase():
    rungs = dict(server._fts5_ladder("the decay of the orbit"))
    assert rungs["exact_phrase"] == '"the decay of the orbit"'
    assert rungs["any_term"] == '"decay" OR "orbit"'


def test_an_all_stopword_query_still_produces_a_searchable_rung():
    """Stripping every term would turn a legitimate query into a syntax error."""
    rungs = dict(server._fts5_ladder("to be or not to be"))
    assert rungs["any_term"] == '"to" OR "be" OR "or" OR "not" OR "to" OR "be"'


def test_wide_rungs_are_capped_so_a_pasted_page_cannot_scan_the_whole_index():
    huge = " ".join(f"term{i}" for i in range(server._FTS_MAX_TERMS + 40))
    rungs = dict(server._fts5_ladder(huge))
    assert rungs["any_term"].count(" OR ") == server._FTS_MAX_TERMS - 1


def test_a_query_with_no_indexable_terms_reports_why_it_returned_nothing(conn):
    rows, diag = server._fts5_search(conn, SQL, "!!! ???", 10)
    assert rows == []
    assert diag.tier == "none"
    assert diag.degraded == "query has no indexable terms"


def test_asking_for_a_rung_that_does_not_exist_is_reported_not_silent(conn):
    """Exact quotation search names its rung; a rename must not make it match nothing."""
    rows, diag = server._fts5_search(conn, SQL, "orbital debris", 10, rungs=("renamed",))
    assert rows == []
    assert "no rung named" in diag.degraded


# --- the caller's string is never an expression ----------------------------


@pytest.mark.parametrize(
    "query",
    [
        "the Court did NOT find the treaty binding",
        "Department of Housing AND Urban Development",
        "the LF HF ratio does NOT measure sympathovagal balance",
        'he called it "mitigation" NEAR the margin',
    ],
)
def test_an_uppercase_word_in_prose_is_never_treated_as_an_operator(conn, query):
    """FTS5 reads bare uppercase AND/OR/NOT/NEAR as operators, so raw strings invert meaning."""
    for _, expression in server._fts5_ladder(query):
        # Whatever survives with the quoted strings removed is the ladder's own
        # syntax; nothing from the caller may appear there.
        skeleton = re.sub(r'"(?:[^"]|"")*"', "", expression)
        assert re.fullmatch(r"[\s,()0-9]*(?:(?:OR|NEAR)[\s,()0-9]*)*", skeleton), expression
    _, diag = server._fts5_search(conn, SQL, query, 10)
    assert "verbatim" not in diag.tiers_tried


@pytest.mark.parametrize("query", ['"orbital debris"', "orbit*", "^orbital", "a nor\'easter"])
def test_operator_characters_are_quoted_away_rather_than_honoured(query):
    """A quotation mark or wildcard in a caller's text is punctuation, not syntax."""
    tiers = [tier for tier, _ in server._fts5_ladder(query)]
    assert "verbatim" not in tiers


def test_a_query_full_of_fts_punctuation_cannot_produce_an_unparseable_rung(conn):
    rows, diag = server._fts5_search(conn, SQL, 'orbital debris "mitigation guidelines', 10)
    assert rows, "the ladder must not be derailed by an unbalanced quotation mark"
    assert diag.errored_rungs == ()


# --- the regression the ladder exists to fix -------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Orbital debris mitigation guidelines (per IADC) constrain deployment.",
        'the "mitigation guidelines" for orbital debris deployment',
        "orbital-debris mitigation: guidelines for constellations",
        "Do mitigation guidelines constrain large constellations?",
    ],
)
def test_natural_language_queries_return_rows_where_the_old_path_returned_none(conn, query):
    assert legacy(conn, SQL, query, 10) == []
    rows, diag = server._fts5_search(conn, SQL, query, 10)
    assert rows, f"ladder found nothing for {query!r} (tried {diag.tiers_tried})"


def test_every_bench_claim_shape_produces_a_valid_expression_on_every_rung(conn):
    """No built rung may ever raise; only the caller's own verbatim rung can."""
    hostile = [
        'He wrote "it is (mostly) fine" — see NOTE*',
        "50% of ‘cases’ (n=12; p<.05) failed",
        "NEAR AND OR NOT",
        "a" * 500,
        "-- ; DROP TABLE docs; --",
        "中文 テスト 한글",
        "",
    ]
    for query in hostile:
        for tier, expression in server._fts5_ladder(query):
            if tier == "verbatim":
                continue
            conn.execute(SQL, (expression, 5)).fetchall()


# --- accumulation ----------------------------------------------------------


def test_a_verbatim_match_outranks_a_document_that_only_shares_terms(conn):
    query = "orbital debris mitigation guidelines"
    rows, diag = server._fts5_search(conn, SQL, query, 10)
    assert diag.tier == "exact_phrase"
    assert rows[0][0] == 1
    assert 2 in [row[0] for row in rows], "looser rungs must still fill the candidate pool"


def test_rows_are_deduplicated_across_rungs(conn):
    rows, _ = server._fts5_search(conn, SQL, "orbital debris mitigation guidelines", 10)
    ids = [row[0] for row in rows]
    assert len(ids) == len(set(ids))


def test_accumulation_stops_once_the_limit_is_reached(conn):
    rows, diag = server._fts5_search(conn, SQL, "orbital debris mitigation", 1)
    assert len(rows) == 1
    assert diag.rows == 1


def test_the_reported_tier_is_the_most_precise_rung_that_contributed(conn):
    """`tier` is the caller's confidence signal, so it must not report the last rung."""
    _, diag = server._fts5_search(conn, SQL, "pulsars photosynthesis", 10)
    assert diag.tier == "any_term"
    _, exact = server._fts5_search(conn, SQL, "deep ocean vents", 10)
    assert exact.tier == "exact_phrase"


def test_a_query_matching_nothing_reports_that_it_tried_every_rung(conn):
    rows, diag = server._fts5_search(conn, SQL, "quasar interferometry", 10)
    assert rows == []
    assert diag.tier == "none"
    assert diag.degraded == "no lexical match at any tier"
    assert "any_term" in diag.tiers_tried


# --- the self-test check ---------------------------------------------------


def _papers_database(path: Path) -> None:
    """A minimal stand-in for the live schema: papers plus its external-content index."""
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            tldr TEXT,
            processed_text TEXT,
            tex_text TEXT,
            has_full_text INTEGER DEFAULT 0
        );
        CREATE VIRTUAL TABLE papers_fts USING fts5(
            paper_id, title, abstract, tldr, processed_text, tex_text,
            content=papers, content_rowid=rowid
        );
    """)
    for index in range(3):
        connection.execute(
            "INSERT INTO papers (paper_id, title, abstract, has_full_text) VALUES (?,?,?,1)",
            (
                f"paper-{index}",
                f"Sustained magnetohydrodynamic confinement in toroidal geometry {index}",
                " ".join(f"abstractword{index}x{word}" for word in range(60)),
            ),
        )
    connection.execute(
        "INSERT INTO papers_fts(rowid, paper_id, title, abstract, tldr, processed_text, tex_text) "
        "SELECT rowid, paper_id, title, abstract, tldr, processed_text, tex_text FROM papers"
    )
    connection.commit()
    connection.close()


def test_lexical_retrieval_check_passes_against_a_healthy_index(tmp_path):
    database = tmp_path / "papers.db"
    _papers_database(database)
    passed, detail = server._check_lexical_retrieval(database)
    assert passed is True, detail
    assert "matched on exact_phrase" in detail


def test_lexical_retrieval_check_catches_a_ladder_collapsed_to_its_widest_rung(tmp_path, monkeypatch):
    """A ladder that answers everything with a disjunction would otherwise look healthy."""
    database = tmp_path / "papers.db"
    _papers_database(database)
    original = server._fts5_ladder
    monkeypatch.setattr(
        server, "_fts5_ladder", lambda q: [r for r in original(q) if r[0] == "any_term"]
    )
    passed, detail = server._check_lexical_retrieval(database)
    assert passed is False
    assert "matched on any_term" in detail


def test_lexical_retrieval_check_fails_on_the_query_handling_it_replaced(tmp_path, monkeypatch):
    """The check only earns its place if the old raw-then-quote path trips it."""
    database = tmp_path / "papers.db"
    _papers_database(database)

    def legacy_search(conn, sql, query, limit):
        return legacy(conn, sql, query, limit), server.FtsDiagnostics("legacy", (), 0, 0, None)

    monkeypatch.setattr(server, "_fts5_search", legacy_search)
    passed, detail = server._check_lexical_retrieval(database)
    assert passed is False
    assert "lexical search is not reaching the index" in detail


def test_lexical_retrieval_check_fails_when_the_schema_hides_its_probes(tmp_path):
    """Full-text papers with no readable title or abstract is a schema fault, not health."""
    database = tmp_path / "papers.db"
    _papers_database(database)
    connection = sqlite3.connect(database)
    connection.execute("UPDATE papers SET abstract = ''")
    connection.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
    connection.commit()
    connection.close()

    passed, detail = server._check_lexical_retrieval(database)
    assert passed is False
    assert "check the schema" in detail


def test_lexical_retrieval_check_skips_an_empty_library(tmp_path):
    database = tmp_path / "papers.db"
    _papers_database(database)
    connection = sqlite3.connect(database)
    connection.execute("DELETE FROM papers")
    connection.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
    connection.commit()
    connection.close()

    passed, detail = server._check_lexical_retrieval(database)
    assert passed is True
    assert "no full-text paper" in detail


def test_lexical_retrieval_check_catches_deleted_proximity_rungs(tmp_path, monkeypatch):
    """The middle probe exists to reach NEAR; it must fail when NEAR is gone."""
    database = tmp_path / "papers.db"
    _papers_database(database)
    original = server._fts5_ladder
    monkeypatch.setattr(
        server,
        "_fts5_ladder",
        lambda q: [r for r in original(q) if not r[0].startswith("near_")],
    )
    passed, detail = server._check_lexical_retrieval(database)
    assert passed is False
    assert "near_10/near_30" in detail


def test_the_chunk_fallback_alarm_fires_only_on_a_broken_chunk_index():
    """The threshold sits above every share a healthy wide search produces."""
    healthy = (0.24, 0.44, 0.55, 0.60, 0.63, 0.65, 0.84)
    assert not any(share > server._CHUNK_FALLBACK_ALARM for share in healthy)
    assert 1.0 > server._CHUNK_FALLBACK_ALARM


def test_lexical_retrieval_check_reports_a_missing_database(tmp_path):
    passed, detail = server._check_lexical_retrieval(tmp_path / "absent.db")
    assert passed is False
    assert "does not exist" in detail


# --- find_quotation --------------------------------------------------------

QUOTATION = (
    "the Committee (acting under Article 12) concluded that such measures "
    "were neither necessary nor proportionate"
)
# The same sentence as the corpus holds it after a split-ligature extraction:
# "sufficient" arrives as three tokens, so no phrase or conjunction can match.
CORRUPTED = QUOTATION.replace("necessary", "suf fi ciently necessary")


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A real papers.db, built by the server's own migrations."""
    monkeypatch.setattr(server, "DB_PATH", tmp_path / "papers.db")
    monkeypatch.setattr(server, "_db_initialized", False)
    connection = server._init_db()
    connection.execute(
        "INSERT INTO papers (paper_id, title, has_full_text, verified) VALUES (?,?,1,1)",
        ("paper-a", "Proportionality in treaty supervision"),
    )
    connection.execute(
        "INSERT INTO paper_chunks (paper_id, chunk_index, chunk_text) VALUES (?,?,?)",
        ("paper-a", 0, f"Background text. {QUOTATION}. Further discussion follows."),
    )
    connection.execute(
        "INSERT INTO papers (paper_id, title, has_full_text, verified) VALUES (?,?,1,1)",
        ("paper-b", "A study of committee practice"),
    )
    connection.execute(
        "INSERT INTO paper_chunks (paper_id, chunk_index, chunk_text) VALUES (?,?,?)",
        ("paper-b", 0, f"Preamble. {CORRUPTED}. Closing remarks."),
    )
    connection.commit()
    connection.close()
    return tmp_path / "papers.db"


def _run(coroutine):
    import asyncio

    return asyncio.run(coroutine)


def test_a_quotation_containing_fts_syntax_is_found_not_reported_missing(library):
    """A parenthesis in a quotation used to raise, and the error read as 'no matches'."""
    result = _run(server.find_quotation(QUOTATION))
    assert "No " not in result.splitlines()[0]
    assert "paper-a" in result


def test_the_old_path_really_did_fail_on_that_quotation(library):
    """Without this the test above could pass for a corpus that never broke."""
    connection = sqlite3.connect(library)
    sql = (
        "SELECT c.chunk_id FROM paper_chunks_fts fts "
        "JOIN paper_chunks c ON fts.rowid = c.chunk_id "
        "WHERE paper_chunks_fts MATCH ? LIMIT ?"
    )
    with pytest.raises(sqlite3.OperationalError):
        connection.execute(sql, (QUOTATION, 20)).fetchall()
    connection.close()


def test_exact_mode_stays_exact_and_does_not_widen_to_term_overlap(library):
    """Widening the phrase rung would turn a pin-cite tool into a topic search."""
    result = _run(server.find_quotation("Committee proportionate measures Article"))
    assert result.startswith("No exact quotation matches")


def test_fuzzy_mode_recovers_a_quotation_broken_by_an_extraction_artifact(library):
    """A split ligature leaves no phrase and no full conjunction to match."""
    scoped = _run(server.find_quotation(QUOTATION, paper_id="paper-b", fuzzy=True))
    assert "paper-b" in scoped
    exact = _run(server.find_quotation(QUOTATION, paper_id="paper-b"))
    assert exact.startswith("No exact quotation matches")


def test_fuzzy_mode_refuses_a_passage_that_only_shares_a_few_words(library):
    """Every result here carries a page number, and a page number becomes a pin cite."""
    result = _run(
        server.find_quotation(
            "the Committee published its supervisory calendar for the following biennium "
            "together with an annex listing rapporteurs",
            fuzzy=True,
        )
    )
    assert result.startswith("No fuzzy quotation matches")
    assert "shared some wording" in result


def test_fuzzy_mode_reports_the_measured_coverage_of_each_match(library):
    result = _run(server.find_quotation(QUOTATION, fuzzy=True))
    assert "quotation_coverage: 100%" in result


def test_fuzzy_coverage_is_measured_against_the_passage_not_the_ranking(library):
    """Ranking by BM25 would put a rarer shared word above a fuller quotation."""
    rows = [
        (1, "nothing here but proportionate"),
        (2, f"prelude {QUOTATION} coda"),
    ]
    kept, coverage = server._filter_by_token_coverage(QUOTATION, rows, 10)
    assert [row[0] for row in kept] == [2]
    assert coverage[2] == 1.0


def test_coverage_requires_the_words_together_not_merely_present():
    """A long passage in the same vocabulary contains most of any quotation's words."""
    content = [t.casefold() for t in server._fts5_tokenize(QUOTATION)]
    content = [t for t in content if t not in server._FTS_STOPWORDS]

    together = f"filler text. {QUOTATION}. more filler."
    scattered = " ".join(
        word + " " + " ".join(f"padding{i}" for i in range(30))
        for i, word in enumerate(QUOTATION.split())
    )
    assert server._quotation_coverage(content, together) == 1.0
    assert server._quotation_coverage(content, scattered) < server._FUZZY_MIN_COVERAGE


def test_coverage_survives_the_extra_tokens_extraction_introduces():
    """A split ligature turns one word into three; the window has to absorb that."""
    content = [t.casefold() for t in server._fts5_tokenize(QUOTATION)]
    content = [t for t in content if t not in server._FTS_STOPWORDS]
    corrupted = QUOTATION.replace("necessary", "suf fi ciently necessary")
    assert server._quotation_coverage(content, corrupted) >= server._FUZZY_MIN_COVERAGE


def test_a_quotation_fully_covers_its_own_text_even_when_it_repeats_words():
    """Counting a repeated word twice in the denominator caps a self-match below 1.0."""
    repetitive = "a wall thickness of 0.05 cm and a wall thickness of 0.1 cm"
    terms = server._quotation_terms(repetitive)
    assert len(terms) == len(set(terms))
    assert server._quotation_coverage(terms, repetitive) == 1.0


def test_coverage_does_not_depend_on_the_caller_deduplicating():
    """The measure has to be right for any caller, not only the one that dedupes."""
    quotation = "wall thickness of the wall was 0.05 cm and the wall thickness varied"
    passage = f"Some preamble here. {quotation}. Trailing discussion follows."
    raw = [t.casefold() for t in server._fts5_tokenize(quotation)]
    raw = [t for t in raw if t not in server._FTS_STOPWORDS]
    assert len(raw) > len(set(raw)), "this quotation must repeat a word to be a test"
    assert server._quotation_coverage(raw, passage) == 1.0
    assert server._quotation_coverage(server._quotation_terms(quotation), passage) == 1.0


def test_a_tie_for_best_coverage_is_reported_not_resolved_silently(library):
    """A short quotation present verbatim in two passages cannot be ranked by coverage."""
    connection = sqlite3.connect(library)
    connection.execute(
        "INSERT INTO paper_chunks (paper_id, chunk_index, chunk_text) VALUES (?,?,?)",
        ("paper-a", 1, f"A different passage that also contains {QUOTATION} word for word."),
    )
    connection.commit()
    connection.close()

    result = _run(server.find_quotation(QUOTATION, fuzzy=True))
    assert "tie for the best coverage" in result
    assert "not a judgement" in result


def test_a_single_best_match_is_not_flagged_as_tied(library):
    """Scoped to one paper there is only one passage, so nothing is being tiebroken."""
    result = _run(server.find_quotation(QUOTATION, paper_id="paper-a", fuzzy=True))
    assert "tie for the best coverage" not in result


def test_coverage_of_a_passage_sharing_nothing_is_zero():
    content = [t.casefold() for t in server._fts5_tokenize(QUOTATION)]
    assert server._quotation_coverage(content, "photosynthesis in deep ocean vents") == 0.0
    assert server._quotation_coverage([], "anything at all") == 0.0


def test_fuzzy_mode_reports_which_rung_matched(library):
    """Verbatim and merely-overlapping hits must not look the same to the caller."""
    verbatim = _run(server.find_quotation(QUOTATION, fuzzy=True))
    assert "[matched on: exact_phrase]" in verbatim
    loose = _run(server.find_quotation(CORRUPTED, paper_id="paper-a", fuzzy=True))
    assert "[matched on: exact_phrase]" not in loose


def test_a_quotation_with_no_searchable_words_says_so(library):
    result = _run(server.find_quotation("(((...)))"))
    assert "no searchable words" in result


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
