# Client conventions

These are conventions an MCP client (Claude Code, Codex, Cursor, or any
other tool calling the research-mcp server) should follow. The server
cannot enforce them; they are the contract between the server's tool
responses and the caller's behavior.

## 1. `METADATA_NEEDED` on `process_pdf` / `process_tex`

When `process_pdf` or `process_tex` lands a paper with generic metadata
(e.g., `local:Unknown_undated_*`, `local:screenshots`,
`local:1120591183`), its response includes a block starting with:

```
⚠️ METADATA_NEEDED: This paper was stored as `local:XXX` with generic/unknown metadata.
```

followed by a ready-to-use sub-agent prompt.

**Expected client behavior**: on seeing `METADATA_NEEDED` in a tool
response, automatically spawn a small sub-agent (e.g., Claude Haiku)
with the embedded prompt. The sub-agent:

1. Reads the first 3 pages of the PDF (or the main `.tex` file for TeX
   source).
2. Extracts title, first 3 authors, year, DOI / arXiv ID if visible.
3. Calls `match_paper_by_title` to find the canonical S2 record.
4. Calls `fix_orphan_paper`:
   - **Merge mode** if `match_paper_by_title` returns score ≥ 0.85:
     `fix_orphan_paper(orphan_id=<local:...>, canonical_paper_id=<matched S2 ID>)`
   - **In-place update mode** otherwise:
     `fix_orphan_paper(orphan_id=<local:...>, title=..., authors='["..."]', year=..., doi=...)`
5. Returns a one-line summary.

`fix_orphan_paper` handles the mechanical operations: copying
`processed_text`, `tex_text`, `local_pdf_path`, `local_tex_path`, and
chunks to the canonical row; re-embedding from the merged text;
renaming the PDF file to `Author_Year_Title.pdf`; deleting the orphan
entry (merge mode) or updating it in place (in-place mode).

**Do not** skip the sub-agent step because the orphan "seems fine."
The `tex_text` / `processed_text` is searchable but the paper cannot
be cited properly without author and year metadata, and files named
`Unknown_undated_*` are hard to work with downstream.

## 2. TeX source is preferred over PDF for math-heavy content

`get_full_text` returns `tex_text` when available (math-clean from
pandoc LaTeX → plain) and falls back to `processed_text` (Docling PDF
extraction). The response includes a `**Source:**` line indicating
which was used.

If you need authoritative equations, check the source TeX in `$TEX_DIR`
(default `$RESEARCH_MCP_HOME/tex`). The parsed `tex_text` preserves
inline math as `$...$` and display math as `$$...$$`, but the original
TeX has full context (macros, packages, labels).

## 3. Standard library locations

All paths are configurable via env vars (see `research_mcp.paths`):

- PDFs: `$PAPERS_DIR/<Author_Year_Title>.pdf` — flat directory,
  one PDF per paper.
- TeX source: `$TEX_DIR/<paper_folder>/` — one subdirectory per paper,
  containing `main.tex`, `.bib`, figures, `.sty`.
- Web captures: `$WEB_CAPTURES_DIR/` — headless-Chrome PDF archives of
  blog posts, gray literature, supplementary materials.
- Inbox: `$INBOX_DIR/` — drop-zone for `acquire/process_inbox.py`.

Defaults follow XDG Base Directory; if no env vars are set, everything
lives under `$XDG_DATA_HOME/research-mcp/`.

## 4. `ABSTRACT_NEEDED` on `process_pdf` / `process_tex` / `download_paper`

When a paper lands with full text (`processed_text` or `tex_text`) but
an empty `abstract` field, the tool response includes:

```
⚠️ ABSTRACT_NEEDED: Paper `<paper_id>` has full text but no abstract.
```

followed by a ready-to-use sub-agent prompt.

**Expected client behavior**: on seeing `ABSTRACT_NEEDED`, automatically
spawn a sub-agent with the embedded prompt. The sub-agent:

1. Calls `get_full_text` to read the paper's extracted text.
2. Generates a 500–1000 word abstract-style summary covering thesis,
   methods, findings, and significance.
3. Calls `set_abstract` with the generated summary.

`set_abstract` automatically re-embeds the paper from the new abstract,
so paper-level vector search (`vec_papers`) immediately picks up the
improved representation. Books, legal sources, government reports, and
papers that ship without abstracts benefit the most — without a
summary their paper-level embedding is title-only, which is nearly
blind to semantic search.

**Applies to all ingestion paths**: manual `process_pdf` / `process_tex`,
`download_paper` auto-processing, and `acquire_batch` runs.

**Composes with `METADATA_NEEDED`**: a paper can trigger both warnings
in the same response. Handle them in order — fix metadata first (so
the canonical paper is known), then generate the abstract for the
canonical record.

## 5. Citation graph (`paper_references` table)

When `search_local` returns results, the `match_sources` field may
include `hub#N` indicating that the paper received a citation-graph
boost — other candidates in the top 50 for this query either cite it
or are cited by it, and its hub rank is N among seed-set hub scores.

High hub rank (#1-#5) plus strong keyword + semantic + chunk signals
is a strong indicator of a canonical paper for the query's topic.

Papers with no edges in `paper_references` (e.g. freshly-ingested
papers before backfill, or `local:*` papers without S2 / OpenAlex
metadata) receive no hub boost but are otherwise unaffected. The
signal degrades gracefully to zero for sparsely-explored topics.

Edges are populated by three paths:

- **OpenAlex ingestion**: `_store_openalex_work` persists
  `referenced_works` as edges into the table automatically.
- **S2 citation tools**: `get_citations` and `get_references` persist
  edges when called on papers already in the library.
- **One-time backfill**: `maintenance/backfill_citations.py` script
  run against the existing library.

## 6. `verify_claim` abstention

`verify_claim` returns one of three verdicts:

- **`supported`** — at least one passage entails the claim.
- **`refuted`** — at least one passage contradicts the claim.
- **`abstain`** — no passage clears the NLI threshold in either
  direction.

**Treat `abstain` as a real verdict.** If the verifier abstains, your
agent should *not* present the claim as supported. The proper
follow-up is one of: (a) broaden the page range and re-verify;
(b) call `find_quotation` to look for verbatim quote support;
(c) tell the user the claim could not be verified against the cited
paper.

The abstain path exists specifically so that "I don't know" is
available to the upstream agent. Without it, agents end up either
fabricating support or silently skipping the verification step, both
of which are worse than an explicit abstention.

## 7. Structured output

`search_local`, `search_passages`, and several other tools accept
`structured=True` to return JSON instead of human-readable text. Use
the structured form when your client needs to pipe results through
additional programmatic logic; use the default text form when the
result is going directly to a model for reading.
