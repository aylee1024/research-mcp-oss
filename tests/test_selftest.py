# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pytest>=9.1,<10",
#     "respx>=0.23,<0.24",
#     "httpx>=0.28,<1",
#     "mcp[cli]>=2.0.0,<3",
#     "sqlite-vec>=0.1.9,<0.2",
#     "numpy>=2.5,<3",
# ]
# ///
"""Tests for the server self-test helpers.

Run with:

    uv run tests/test_selftest.py
"""

import inspect
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="selftest_test_"))
os.environ["RESEARCH_MCP_HOME"] = str(_TMP)
os.environ["PAPERS_DB_PATH"] = str(_TMP / "papers.db")
os.environ["JSTOR_DB_PATH"] = str(_TMP / "jstor.db")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import server  # noqa: E402


def _copy_dependency_declarations(tmp_path: Path) -> tuple[Path, Path]:
    script_copy = tmp_path / "server.py"
    project_copy = tmp_path / "pyproject.toml"
    shutil.copy2(PROJECT_ROOT / "server.py", script_copy)
    shutil.copy2(PROJECT_ROOT / "pyproject.toml", project_copy)
    return script_copy, project_copy


def test_dependency_parity_accepts_repo_declarations():
    passed, detail = server._check_dependency_parity(
        PROJECT_ROOT / "server.py",
        PROJECT_ROOT / "pyproject.toml",
    )
    assert passed is True
    assert "9 inline and 17 project requirements are upper-bounded" in detail
    assert "7 core distributions covered" in detail
    assert "9 shared specifier sets agree" in detail


def test_dependency_parity_catches_temp_copy_bound_mismatch(tmp_path):
    script_copy, project_copy = _copy_dependency_declarations(tmp_path)
    project_text = project_copy.read_text(encoding="utf-8")
    changed_text = project_text.replace("httpx>=0.28,<1", "httpx>=0.28,<0.29", 1)
    assert changed_text != project_text
    project_copy.write_text(changed_text, encoding="utf-8")

    passed, detail = server._check_dependency_parity(script_copy, project_copy)
    assert passed is False
    assert "specifier mismatch for httpx" in detail
    assert "<1" in detail
    assert "<0.29" in detail


def test_dependency_parity_catches_temp_copy_unbounded_requirement(tmp_path):
    script_copy, project_copy = _copy_dependency_declarations(tmp_path)
    project_text = project_copy.read_text(encoding="utf-8")
    changed_text = project_text.replace("httpx>=0.28,<1", "httpx>=0.28", 1)
    assert changed_text != project_text
    project_copy.write_text(changed_text, encoding="utf-8")

    passed, detail = server._check_dependency_parity(script_copy, project_copy)
    assert passed is False
    assert "pyproject.toml requirements without upper bounds: httpx>=0.28" in detail


def test_tool_registration_accepts_the_declared_surface():
    passed, detail = server._check_tool_registration(server.mcp, server.EXPECTED_TOOL_NAMES)
    assert passed is True
    assert f"registered={server.EXPECTED_TOOL_COUNT}" in detail


def test_tool_registration_reports_a_missing_tool():
    passed, detail = server._check_tool_registration(
        server.mcp,
        server.EXPECTED_TOOL_NAMES + ("a_tool_that_does_not_exist",),
    )
    assert passed is False
    assert "missing: a_tool_that_does_not_exist" in detail


def test_tool_registration_catches_a_rename_that_preserves_the_count():
    """A renamed tool keeps the count identical, so only names can catch it."""
    renamed = ("library_statistics",) + tuple(
        name for name in server.EXPECTED_TOOL_NAMES if name != "library_stats"
    )
    assert len(renamed) == len(server.EXPECTED_TOOL_NAMES)

    passed, detail = server._check_tool_registration(server.mcp, renamed)
    assert passed is False
    assert "missing: library_statistics" in detail
    assert "unexpected: library_stats" in detail


def test_schema_version_reader_uses_maximum_version(tmp_path):
    database_path = tmp_path / "papers.db"
    conn = sqlite3.connect(database_path)
    try:
        conn.execute("CREATE TABLE papers (paper_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO papers (paper_id) VALUES ('paper-1')")
        conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO schema_version (version) VALUES (?)", [(21,), (22,)])
        conn.commit()
    finally:
        conn.close()

    paper_count, schema_version = server._read_database_stats(database_path)
    source = inspect.getsource(server._read_database_stats)
    assert paper_count == 1
    assert schema_version == 22
    assert "SELECT MAX(version) FROM schema_version" in source
    assert "LIMIT 1" not in source


def test_database_check_accepts_empty_current_schema(tmp_path):
    database_path = tmp_path / "papers.db"
    conn = sqlite3.connect(database_path)
    try:
        conn.execute("CREATE TABLE papers (paper_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (server._SCHEMA_VERSION,))
        conn.commit()
    finally:
        conn.close()

    passed, detail = server._check_database(database_path, server._SCHEMA_VERSION)
    assert passed is True
    assert "papers=0" in detail
    assert f"schema_version={server._SCHEMA_VERSION}" in detail


def test_vector_knn_check_accepts_the_k_values_this_repo_uses():
    passed, detail = server._check_vector_knn_limits(PROJECT_ROOT / "server.py")
    assert passed is True, detail
    assert "accepted" in detail


def test_vector_knn_check_catches_a_k_the_index_will_refuse(tmp_path):
    """Widening k past the index ceiling deletes a search leg instead of widening it."""
    source = tmp_path / "server.py"
    source.write_text(
        "SELECT rowid FROM v WHERE embedding MATCH ? AND k = 5000\n", encoding="utf-8"
    )
    passed, detail = server._check_vector_knn_limits(source)
    assert passed is False
    assert "k=5000" in detail
    assert "silently drops" in detail


def test_vector_knn_check_fails_when_it_stops_matching_any_query(tmp_path):
    """A check that examines nothing would otherwise report success forever."""
    source = tmp_path / "server.py"
    source.write_text("# the SQL was reformatted and k is now bound\n", encoding="utf-8")
    passed, detail = server._check_vector_knn_limits(source)
    assert passed is False
    assert "stopped matching" in detail


@pytest.mark.parametrize(
    "spelling",
    ["AND k=8000", "and  k  =  8000", "AND\n                  k = 8000"],
)
def test_vector_knn_check_survives_reformatting_of_the_query(tmp_path, spelling):
    """The gate must not go blind the first time someone reflows the SQL."""
    source = tmp_path / "server.py"
    source.write_text(f"SELECT rowid FROM v WHERE embedding MATCH ? {spelling}\n", encoding="utf-8")
    passed, detail = server._check_vector_knn_limits(source)
    assert passed is False
    assert "k=8000" in detail


def test_vector_knn_check_resolves_an_interpolated_k(tmp_path):
    source = tmp_path / "server.py"
    source.write_text(
        "f'SELECT rowid FROM v WHERE embedding MATCH ? AND k = {MAX_VEC_KNN_K}'\n",
        encoding="utf-8",
    )
    passed, detail = server._check_vector_knn_limits(source)
    assert passed is True
    assert "1 KNN queries" in detail


def test_vector_knn_check_refuses_a_k_it_cannot_resolve(tmp_path):
    """An unresolvable name means the gate cannot say what k the query asks for."""
    source = tmp_path / "server.py"
    source.write_text(
        "f'SELECT rowid FROM v WHERE embedding MATCH ? AND k = {WIDER_K}'\n", encoding="utf-8"
    )
    passed, detail = server._check_vector_knn_limits(source)
    assert passed is False
    assert "cannot resolve" in detail


def test_referenced_path_check_reports_missing_path(tmp_path):
    present_path = tmp_path / "present.py"
    present_path.touch()

    passed, detail = server._check_referenced_paths(
        tmp_path,
        ("present.py", "missing.py"),
    )
    assert passed is False
    assert detail == "missing: missing.py"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
