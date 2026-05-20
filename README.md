# research-mcp 📚

your personal academic library, exposed as MCP tools.

find papers. locate passages. verify claims before you cite them.

---

### the 60-second tour

```bash
git clone https://github.com/aylee1024/research-mcp-oss
cd research-mcp-oss
uv run server.py          # first run installs deps, creates the DB
```

point claude code, codex, or cursor at it and ask for a paper — it'll
find it, fetch the PDF, read it, and answer questions out of the actual
text. citations include a paper id and page range, not a hallucinated
fragment.

```
  📄 pdf  ──→  🔍 docling  ──→  🗄  sqlite
                                ├─ fts5     (keyword)
                                ├─ vec      (embeddings)
                                └─ refs     (citation graph)
                                     │
                                     ▼
                              🛠  mcp tools
                                     │
                                     ▼
                          claude · codex · cursor
```

---

## why this exists 🎯

agents are great at sounding confident about papers they have never
read. this MCP makes them read.

every retrieval tool returns a paper id and a page range. every
verification tool fails closed: if the claim isn't actually supported
by the text, it says "not supported" instead of hedging.

it's local. your library, your laptop, your filesystem. no cloud
component, no shared backend, no telemetry.

---

## what you get 🛠

the server registers around 33 MCP tools. the ones you'll actually use
day-to-day:

**find papers**
- `search_local` — hybrid keyword + vector + citation-graph search over your library
- `search_papers` — semantic scholar search; auto-saves matches
- `search_openalex` — authority-ranked search (citation count, venue, type)
- `match_paper_by_title` — best title match for a fuzzy query

**get the text**
- `download_paper` — fetch an open-access PDF, extract with Docling
- `process_pdf` — bring a PDF you already have into the library
- `process_tex` — same, but for arXiv TeX source (preserves math)
- `get_full_text` — read the extracted text back

**find passages, verify claims**
- `find_quotation` — exact or fuzzy phrase search within a paper
- `search_within_paper` — keyword search scoped to one paper
- `verify_claim` — NLI entailment check; returns supported / refuted / abstain
- `search_passages` — chunk-level search with page ranges

**citations**
- `get_local_references` — walk the citation graph of papers you own
- `get_citations` / `get_references` — fetch incoming/outgoing edges from S2

**housekeeping**
- `library_stats` — coverage metrics
- `check_jstor` — see whether a paywalled paper is in JSTOR
- `set_abstract`, `verify_paper`, `fix_orphan_paper` — curation helpers

every tool returns text suitable for an agent to read and quote. the
structured variants (`structured=True` on the search tools) return JSON
when an agent needs to pipe results through additional logic.

---

## install 🧰

requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and either
macOS or Linux. you'll also want `pandoc` for TeX processing and
optionally `tesseract` for OCR on image-only PDFs.

```bash
# 1. clone
git clone https://github.com/aylee1024/research-mcp-oss
cd research-mcp-oss

# 2. install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. install pandoc (optional but recommended)
brew install pandoc    # macOS
# or: sudo apt install pandoc

# 4. install tesseract (optional, for image-only PDFs)
brew install tesseract # macOS
# or: sudo apt install tesseract-ocr

# 5. boot the server
uv run server.py
```

first run takes ~2 minutes: uv pulls dependencies, the schema migrations
build out a fresh SQLite database under
`$XDG_DATA_HOME/research-mcp/papers.db` (defaults to
`~/.local/share/research-mcp/papers.db`).

embeddings and the NLI verifier are lazy-loaded the first time you call
`search_local` or `verify_claim`, respectively. budget ~1.5 GB of disk
for the embedding model and ~1.2 GB for the NLI model on first download.

---

## configure 🎛

every path is overridable via env var. the defaults follow the XDG
Base Directory spec, so you can ignore this whole section unless you
want PDFs on a different disk or a different DB location.

| variable | default | what it controls |
|---|---|---|
| `RESEARCH_MCP_HOME` | `$XDG_DATA_HOME/research-mcp` | umbrella directory |
| `PAPERS_DB_PATH` | `$RESEARCH_MCP_HOME/papers.db` | the SQLite library |
| `PAPERS_DIR` | `$RESEARCH_MCP_HOME/papers` | canonical PDF location |
| `INBOX_DIR` | `$RESEARCH_MCP_HOME/inbox` | drop-zone for batch ingestion |
| `WEB_CAPTURES_DIR` | `$RESEARCH_MCP_HOME/web-captures` | headless-Chrome captures |
| `TEX_DIR` | `$RESEARCH_MCP_HOME/tex` | extracted arXiv TeX source |
| `JSTOR_DB_PATH` | `$RESEARCH_MCP_HOME/jstor.db` | optional JSTOR sidecar |

optional API keys, all read from the environment:

```bash
export S2_API_KEY=...           # semantic scholar; faster rate limits
export OPENALEX_API_KEY=...     # openalex polite-pool
export OPENALEX_MAILTO=you@x    # openalex polite-pool (alternative)
export UNPAYWALL_EMAIL=you@x    # required by unpaywall TOS for fallback PDF lookup
export PHILPAPERS_API_ID=...    # philpapers categories + search
export PHILPAPERS_API_KEY=...
```

none of these are required. the server works without any of them; the
keyed paths just hit higher rate limits and broader corpora.

---

## ingest your first paper 📥

three ways in:

**1. by identifier** — works for DOIs, arXiv IDs, S2 paper IDs:

```
> download paper 10.1038/s41586-020-2649-2
```

your agent calls `download_paper`, which tries Semantic Scholar's OA
link first, then falls back to Unpaywall. if either works, the PDF
lands in `$PAPERS_DIR` and is extracted into the database in one step.
for batch acquisition with the longer chain (arXiv direct, direct
URLs), use `acquire/acquire_batch.py`.

**2. by file** — for PDFs you already have:

```
> process this PDF: ~/Downloads/some-paper.pdf
```

your agent calls `process_pdf`, which extracts via Docling, identifies
the paper (DOI in metadata or filename, title match against S2), and
files it under canonical `Author_Year_Title.pdf` naming.

**3. by inbox** — for bulk:

```bash
cp ~/Downloads/*.pdf $INBOX_DIR/   # or whatever your INBOX_DIR is
uv run acquire/process_inbox.py
```

walks the inbox, processes each PDF, leaves the originals there until
they're successfully filed.

---

## connect it to an agent 🔌

drop the snippet from
[`examples/claude_code_mcp.json`](examples/claude_code_mcp.json) into
your MCP client config. for Claude Code:

```json
{
  "mcpServers": {
    "research-mcp": {
      "command": "uv",
      "args": ["run", "/absolute/path/to/research-mcp-oss/server.py"]
    }
  }
}
```

restart your client. you should see the tools appear under `research-mcp`.

---

## what's in the box 📦

```
research-mcp-oss/
├── server.py                 the MCP server (33 tools, all in here)
├── process_pdf.py            Docling worker (subprocess; called by server)
├── process_tex.py            pandoc worker (subprocess; called by server)
├── research_mcp/
│   ├── __init__.py           __version__
│   └── paths.py              XDG-default path resolution
├── acquire/
│   ├── acquire_batch.py      batch downloader (OA → arXiv → direct URL)
│   ├── process_inbox.py      drain the INBOX_DIR
│   └── retraction_refresh.py weekly OpenAlex retraction sync
├── maintenance/
│   ├── backfill_embeddings.py    rebuild vector index from scratch
│   ├── backfill_citations.py     populate the citation graph
│   ├── backfill_page_passages.py build page-bounded passages (v18)
│   ├── dedup_papers.py            merge duplicates by DOI/title
│   ├── build_jstor_db.py          optional JSTOR sidecar
│   └── (~10 more)
├── audit/
│   ├── audit_text_match.py   flag title/body mismatches
│   ├── fix_ligatures.py      repair broken " fi " / " fl " ligatures
│   └── redocling_*.py        re-extract papers from source PDFs
├── page_offsets/             pincite calibration (Bluebook-grade page refs)
├── examples/                 MCP config snippets, launchd plist template
└── docs/
    ├── ARCHITECTURE.md       the why behind the schema and pipeline
    ├── DOWNLOAD_BACKENDS.md  how to plug in custom paper sources
    ├── CLIENT_CONVENTIONS.md MCP-client behavior contracts
    └── CONTRIBUTING.md
```

---

## architecture in one paragraph 🏗

PDFs go through Docling and land as `paper_chunks` rows (one row per
~650-token slice, with page boundaries) and a single `papers` row with
the metadata. each chunk gets a 768-dimensional embedding stored in a
sqlite-vec virtual table. queries blend four signals — FTS5 keyword
match, chunk vector similarity, paper-level vector similarity, and a
citation-graph hub score — via reciprocal rank fusion, then a
cross-encoder reranks the top candidates. `verify_claim` runs a
DeBERTa NLI head over the candidate passages and abstains when no
passage entails the claim.

full design notes: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## adding your own download backend 🔌

research-mcp ships with a small set of download sources: open access
via Unpaywall and S2, arXiv preprints, and direct URLs from upstream
metadata (author sites, institutional repositories). It does not
attempt to bypass paywalls or call shadow libraries; for those you
plug in your own backend.

if you want to plug in your institutional library proxy, a private
corpus, or another source, the interface is documented at
[docs/DOWNLOAD_BACKENDS.md](docs/DOWNLOAD_BACKENDS.md). it's a single
async function returning `(result_text, char_count)`. the existing
chain in `acquire/acquire_batch.py` is the reference implementation.

---

## what this isn't 🚫

- **not a hosted service.** it's a local Python process. no cloud,
  no shared backend, no auth, no multi-user model.
- **not battle-tested on Windows.** the code is mostly portable but
  developed on macOS and tested on Linux. Windows likely works but is
  not in the smoke matrix.
- **not yet on PyPI.** v0.1.0 is git-clone-and-run. when the API
  stabilizes a bit more, `uvx research-mcp` will work.
- **not a citation manager.** there's no UI; the agent is the UI.
  if you want a polished bibliography app, use Zotero. if you want
  your agent to read the papers for you, use this.

---

## contributing 🤝

issues and PRs welcome. see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)
for the short version: ruff defaults, conventional-ish commit messages,
single-maintainer review cadence (be patient).

---

## license ⚖

MIT. see [LICENSE](LICENSE).

built with ❤ and a lot of red ink — go read some papers.
