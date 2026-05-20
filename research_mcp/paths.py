"""Single source of truth for filesystem paths used by research-mcp.

All paths are configurable via environment variables. Defaults follow the XDG
Base Directory Specification, so on a fresh machine with nothing configured
everything lands under `${XDG_DATA_HOME:-~/.local/share}/research-mcp/`.

Override individual paths to place the SQLite database, PDFs, and inbox on
different filesystems (e.g., an external SSD for papers, a faster local disk
for the DB). Override `RESEARCH_MCP_HOME` to relocate the whole tree at once
while keeping the layout.

Environment variables, in resolution order:
    RESEARCH_MCP_HOME    umbrella directory; defaults to $XDG_DATA_HOME/research-mcp
    PAPERS_DB_PATH       sqlite database; defaults to $RESEARCH_MCP_HOME/papers.db
    JSTOR_DB_PATH        optional jstor metadata sidecar; defaults to $RESEARCH_MCP_HOME/jstor.db
    PAPERS_DIR           canonical PDF library; defaults to $RESEARCH_MCP_HOME/papers
    INBOX_DIR            drop-zone for batch ingestion; defaults to $RESEARCH_MCP_HOME/inbox
    WEB_CAPTURES_DIR     headless-Chrome web captures; defaults to $RESEARCH_MCP_HOME/web-captures
    TEX_DIR              arXiv-extracted TeX source; defaults to $RESEARCH_MCP_HOME/tex

The path values are read once at import time. Tests that need different paths
should set the env vars before importing this module, or monkey-patch the
attributes directly.
"""

from __future__ import annotations

import os
from pathlib import Path


def _xdg_data_home() -> Path:
    """Resolve $XDG_DATA_HOME with the spec's default fallback."""
    raw = os.environ.get("XDG_DATA_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".local" / "share"


def _env_path(name: str, default: Path) -> Path:
    """Read an env-var path with `~` expansion. Falls back to `default`."""
    raw = os.environ.get(name)
    if raw:
        return Path(raw).expanduser()
    return default


RESEARCH_MCP_HOME: Path = _env_path("RESEARCH_MCP_HOME", _xdg_data_home() / "research-mcp")

PAPERS_DB_PATH: Path   = _env_path("PAPERS_DB_PATH",   RESEARCH_MCP_HOME / "papers.db")
JSTOR_DB_PATH: Path    = _env_path("JSTOR_DB_PATH",    RESEARCH_MCP_HOME / "jstor.db")
PAPERS_DIR: Path       = _env_path("PAPERS_DIR",       RESEARCH_MCP_HOME / "papers")
INBOX_DIR: Path        = _env_path("INBOX_DIR",        RESEARCH_MCP_HOME / "inbox")
WEB_CAPTURES_DIR: Path = _env_path("WEB_CAPTURES_DIR", RESEARCH_MCP_HOME / "web-captures")
TEX_DIR: Path          = _env_path("TEX_DIR",          RESEARCH_MCP_HOME / "tex")


def ensure_dirs() -> None:
    """Create every directory in the tree if missing. Safe to call repeatedly.

    Also creates the parent directories of `PAPERS_DB_PATH` and
    `JSTOR_DB_PATH` so that overriding one of those env vars to a path
    outside `RESEARCH_MCP_HOME` (e.g., on a different filesystem) still
    boots without a separate `mkdir`.
    """
    for p in (
        RESEARCH_MCP_HOME,
        PAPERS_DIR,
        INBOX_DIR,
        WEB_CAPTURES_DIR,
        TEX_DIR,
        PAPERS_DB_PATH.parent,
        JSTOR_DB_PATH.parent,
    ):
        p.mkdir(parents=True, exist_ok=True)


__all__ = [
    "RESEARCH_MCP_HOME",
    "PAPERS_DB_PATH",
    "JSTOR_DB_PATH",
    "PAPERS_DIR",
    "INBOX_DIR",
    "WEB_CAPTURES_DIR",
    "TEX_DIR",
    "ensure_dirs",
]
