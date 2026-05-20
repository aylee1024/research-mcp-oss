# Architecture

A reference for how research-mcp is built, in roughly the order a curious
contributor would want to learn it. None of this is required to *use* the
server — see the README for that — but it's the design rationale behind
the choices you'll see in the code.

## Pipeline at a glance

```
identifier or PDF → process_pdf or download_paper
                           │
                           ▼
                   Docling extraction
                           │
                           ▼
              chunking (~650-token slices,
              page-bounded, with page markers)
                           │
                  ┌────────┼────────┐
                  ▼        ▼        ▼
           papers row  chunks  vec embeddings
                           │
                           ▼
              FTS5 + sqlite-vec indices
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
        retrieval tools         verification tools
        (search_local,          (verify_claim, NLI
         search_passages,        entailment with
         find_quotation)         abstain semantics)
```

Everything past Docling lives in a single SQLite file. There is no
separate vector store, no separate search index — sqlite-vec and the FTS5
virtual table both live in `papers.db`. This makes the system a
single-file artifact you can `scp` to a new machine and have working in
under a minute.

## Database schema

Schema version is tracked in the `schema_version` table. The server
runs `_run_schema_migrations()` on every startup; the migrations are
idempotent and always advance the version monotonically. The current
version is `v21`.

The core tables:

### `papers`

One row per paper. Identifiers (`paper_id` PK, `doi`, `arxiv_id`,
`pmid`), bibliographic metadata (`title`, `authors` JSON, `year`,
`venue`), full text (`processed_text` from Docling, `tex_text` from
pandoc), provenance (`local_pdf_path`, `local_tex_path`), and a handful
of pipeline-state flags:

- `has_full_text` — boolean: does this row actually carry text the
  retrieval pipeline can use? Decoupled from `processed_text IS NOT NULL`
  because some sources (web captures, abstract-only S2 records) populate
  text into different columns. The flag is the single source of truth
  for "this is searchable."
- `verified` — boolean: text content has been verified to match the
  paper's identifier (DOI, title, authors). Verified rows are protected
  from pruning.
- `is_retracted` — populated by the weekly OpenAlex retraction sync.
- `pdf_page_offset`, `pages_verified` — pincite calibration (below).

### `paper_chunks`

One row per ~650-token slice. Carries `chunk_text`, `section_header` (if
Docling identified one), and `page_start` / `page_end` (PDF physical
pages). The chunk index together with `paper_id` forms the PK.

Page boundaries are populated by Docling's `<!-- page N -->` markers.
When a chunk spans multiple pages, `page_start` is the earliest and
`page_end` the latest physical page touched by the chunk text.

### `paper_passages` (schema v18+)

Page-bounded sub-windows of the text, smaller than a chunk and never
crossing a page boundary. Built by `maintenance/backfill_page_passages.py`
from existing chunks. The passage layer is what supports Bluebook-grade
pin-cites: when `search_passages` returns a hit, the `page_start` is the
*actual page label of the printed paper* (after `pdf_page_offset` is
applied), not the physical PDF page.

### `paper_references`

Citation graph: edges `(citing_paper_id, cited_paper_id, is_influential,
source)`. Populated by three paths: OpenAlex ingestion (auto),
`get_citations` / `get_references` calls on already-stored papers, and
the one-time `backfill_citations.py` script.

### `vec_papers` and `vec_chunks`

sqlite-vec virtual tables holding 768-dimensional embeddings. The
default embedding model is `nomic-ai/nomic-embed-text-v1.5`, swappable
via `EMBED_VARIANT` env var (qwen3-mlx variants offered for Apple
Silicon).

### `papers_fts` and `paper_chunks_fts`

FTS5 contentless tables. BM25 weights are tuned with title-dominance:
`(0.1, 10.0, 3.0, 3.0, 1.0, 0.0)` for the paper-level index. The weights
are not user-tunable from outside the code — they're derived from
benchmark data and shipped as universals.

### Cascade triggers

Every paper-row deletion cascades to: `paper_chunks`, `paper_passages`,
`paper_references` (both directions), `vec_papers`, `vec_chunks`. This
is enforced by SQLite triggers, not application code, so any path that
deletes a paper — including manual `DELETE FROM papers` from the CLI —
leaves no orphans.

Trigger logic also prevents orphan creation on the *insert* side:
attempts to insert a chunk for a non-existent paper raise an error.

## Hybrid search

`search_local` blends four signals via reciprocal rank fusion (RRF,
`k=60`):

1. **FTS5 keyword match** on the paper-level FTS index, BM25-ranked with
   title-dominant weights.
2. **Paper-level vector similarity** against `vec_papers` (top-`k=5000`
   knn, post-filtered to top 300 candidates).
3. **Chunk aggregation**: for each candidate paper, take the top 3
   chunks by vector similarity from `vec_chunks` and aggregate their
   scores via RRF. This recovers papers whose abstracts don't match but
   whose body text does.
4. **Citation hub score**: count incoming + outgoing edges in the local
   citation graph, weighted by RRF rank against the seed top-50.

After the four-signal RRF fusion, the top-N candidates (configurable;
default ~50, capped via env var) go through a cross-encoder reranker.
Default reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`. Override via
`RERANKER_VARIANT=bge-m3` or `mxbai-large` or set `RERANKER_VARIANT=none`
to skip reranking entirely.

`rerank_on="chunk"` is the default — the reranker scores each top
candidate's best chunk rather than its paper-level metadata. This gives
sharper relevance for letter-level citation use cases where the
*passage* matters more than the *paper*. Override with
`rerank_on="paper"` if you want the older behavior.

`search_passages` is the same pipeline but reports hits at chunk
granularity with page ranges, suitable for direct pin-citing.

## Pincite calibration

A paper's printed page numbers ("page 17 of the article") usually don't
match the physical PDF page numbers ("page 23 of the file"). For
citation-grade pin-cites we need to know the offset.

`pdf_page_offset` on each paper is the integer to add to a PDF page to
get the printed page label. It's populated by a four-stage backfill:

1. **PDF embedded page labels** (`backfill_page_offsets_labels.py`):
   reads PyMuPDF's `/PageLabels` metadata. Zero false-positive rate when
   the PDF carries this metadata; ~6% of corpus has it.
2. **Footer/header heuristic** (`backfill_page_offsets_pdf.py`):
   pdfplumber, looks for a consistent integer sequence in the bottom
   margin. Highest yield (~91% of corpus).
3. **OCR fallback** (`backfill_page_offsets_ocr.py`): Tesseract on the
   footer region for image-only PDFs.
4. **Manual review CSV** (`export_manual_review.py` +
   `apply_manual_offsets.py`): for the long-tail papers the automatic
   passes couldn't classify.

`pages_verified` on a paper row is `1` iff the offset was set by a path
that's confident (auto-detected, OCR'd, or manually applied). Papers
with `pages_verified=0` may have a correct offset by chance but the
pipeline doesn't trust them for citation-grade output.

## NLI verification

`verify_claim(claim, paper_id, page_range)` runs a DeBERTa NLI head
(`cross-encoder/nli-deberta-v3-base` by default) over the candidate
passages in the specified page range. The verdict is one of:

- **supported** — at least one passage entails the claim above the
  confidence threshold.
- **refuted** — at least one passage contradicts the claim above
  threshold.
- **abstain** — no passage clears the threshold in either direction.

A set of gold probes runs the first time the NLI model loads (i.e., on
the first `verify_claim` call, not at server startup): a known-supported
claim, a known-refuted claim, and a known-neutral claim run through the
NLI head to detect label-permutation bugs (model swap, upstream
`id2label` change). The probes are documented at the top of
`_get_nli_model()` in `server.py`.

Abstain is a real verdict, not a fallback. The point of including it as
a first-class outcome is to make "I don't know" available to the
upstream agent. Without it, agents have to either commit to a wrong
answer or skip the verification — both of which are worse than an
explicit abstain.

## Acquisition chain

`acquire/acquire_batch.py` walks a chunk file of paper requests and
tries the following in order, per paper:

1. **OA via Unpaywall / S2 `openAccessPdf`** — URLs the upstream
   metadata explicitly flags as open access.
2. **arXiv direct** — for any paper that has an arXiv ID, even if S2
   didn't flag it as OA.
3. **Direct URL** — author site, institutional repository, bepress.
   The URL comes from the chunk JSON or paper metadata; the chain
   doesn't independently verify a license here, so the caller is
   responsible for not pointing the chain at sources they shouldn't
   download from.

Each step verifies the result (title match in extracted text, author
surname check) and rolls back via in-transaction staging if verification
fails. The chain is shaped this way so that the upstream code only
tries URLs the caller or its metadata source said were OK.

PMC (PubMed Central) is intentionally out of scope for v0.1.0 — bulk
downloads from the public PMC URL are restricted by PMC's terms of
service to the OA Subset only, and that filter is not implemented
here. If you need PMC, write an OA-verified backend per
[`DOWNLOAD_BACKENDS.md`](DOWNLOAD_BACKENDS.md). The same doc covers
integrating institutional library proxies, corporate corpora, and
shadow libraries in jurisdictions where they are legal.

## Operational invariants

A short list of invariants the pipeline maintains, mostly enforced by
triggers and verified by the audit scripts under `audit/`:

- **Every chunk has a paper.** Orphan chunks crash on insert (trigger).
- **Every vec row has a chunk.** Cascades on delete (trigger).
- **Every passage has a chunk.** Cascades on delete (trigger).
- **`has_full_text=1` iff retrieval surface is populated.** Updated
  whenever `processed_text`, `tex_text`, or web-capture text changes.
- **DOI is advisory, not a verification gate.** Verification uses
  title + authors. DOI presence in body text is reported in logs but
  doesn't cause verify-fail (many PDFs lose the DOI during Docling
  extraction).
- **Schema version is monotonic.** Migrations are idempotent and
  one-way; the server refuses to start if the on-disk version is
  newer than the migration head.

## Concurrency

The server runs single-process, single-event-loop. There is no shared
state across processes other than SQLite's WAL mode. Long-running
acquisition workers (`acquire_batch.py` runs separately) coordinate via
a `_acquiring_by` claim column on the `papers` table — a worker claims
a row with a timestamp, refreshes the claim every few minutes during
multi-fallback acquisition, and releases on completion. Stale claims
(>1h) are stealable, so a crashed worker can't deadlock the corpus.

## Embedding determinism

Embeddings are computed in float32 on CPU by default. The
sentence-transformers `encode()` path is deterministic for a given
model checkpoint, batch size, and input. We do *not* use MPS GPU on
Apple Silicon — for the 137M-parameter nomic model, CPU with AMX is
empirically 2x faster than MPS (see PyTorch #77799), and rules out an
entire class of nondeterminism from the GPU pipeline.

The optional `EMBED_VARIANT=qwen3-mlx-{512,1024,2560}` path uses MLX on
Metal for Apple Silicon power users who need bigger embeddings; this
trades the cross-platform determinism for ~3x speedup on M-series chips.

## Where to look in the code

- **Server entry point + tool registry**: `server.py` top to bottom; the
  `@mcp.tool()` decorators are easy to scan.
- **Search**: `search_local`, lines ~6000-6300 in `server.py`. The
  fusion logic is plain Python; you can step through it in your head.
- **Schema and migrations**: `_run_schema_migrations()` in `server.py`,
  around line 1200.
- **Acquisition**: `acquire/acquire_batch.py` — the chain is in `main()`.
- **Pincite calibration**: `page_offsets/` directory.
- **Maintenance scripts**: `maintenance/` — each script has a top
  docstring explaining what it does and when to run it.

## Why these choices

A few decisions that look strange until you've used the system:

- **Single SQLite file instead of dedicated stores.** Operational
  simplicity wins. `scp papers.db user@new-host:` is the whole backup
  story. sqlite-vec is fast enough for a personal-scale corpus
  (low-hundreds-of-thousands of papers).
- **Inline cross-encoder reranking instead of bigger embeddings.**
  Cross-encoders give a stronger signal per unit of compute than
  larger bi-encoders; the runtime cost is paid only on the top-N
  candidates, not the full corpus.
- **NLI verification as a first-class tool.** Retrieval gives candidates;
  the agent decides what to cite. The NLI step turns "the agent thinks
  this is on-topic" into "the agent has evidence it's on-topic," which
  is the difference between a useful citation and a hallucinated one.
- **Strip-and-document for shadow libraries.** See
  [`DOWNLOAD_BACKENDS.md`](DOWNLOAD_BACKENDS.md).
