# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mcp[cli]",
#     "httpx",
#     "sqlite-vec",
#     "sentence-transformers",
#     "numpy",
#     "einops",
#     "torch",
#     "mlx ; platform_machine == 'arm64' and sys_platform == 'darwin'",
#     "mlx-embeddings ; platform_machine == 'arm64' and sys_platform == 'darwin'",
# ]
# ///

import asyncio
import gzip
import hashlib
import json
import os
import re
import shutil
import signal
import sqlite3
import threading
import sys
import tarfile
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
import numpy as np
import sqlite_vec
from mcp.server.fastmcp import FastMCP

# Make the sibling `research_mcp` package importable when this script is
# invoked directly (e.g. `uv run server.py`). The script's own directory is
# already on sys.path; this just makes it explicit and survives chdir.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from research_mcp.paths import (
    PAPERS_DB_PATH,
    JSTOR_DB_PATH,
    PAPERS_DIR,
    TEX_DIR,
    ensure_dirs as _ensure_research_mcp_dirs,
)

# Back-compat alias: the codebase historically uses DB_PATH for the papers DB.
DB_PATH = PAPERS_DB_PATH

# Create the data directories on import so first-run uv-invocations don't
# stumble over a missing parent.
_ensure_research_mcp_dirs()

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_RECOMMEND_BASE = "https://api.semanticscholar.org/recommendations/v1"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
OA_BASE = "https://api.openalex.org"

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

DEFAULT_FIELDS = ",".join([
    "title", "abstract", "tldr", "year", "citationCount",
    "influentialCitationCount", "externalIds", "isOpenAccess",
    "openAccessPdf", "venue", "authors", "publicationTypes",
    "fieldsOfStudy", "publicationDate", "url",
])

CITATION_FIELDS = ",".join([
    "title", "abstract", "tldr", "year", "citationCount",
    "externalIds", "isOpenAccess", "venue", "authors",
    "publicationTypes", "fieldsOfStudy",
])

UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "")
S2_API_KEY = os.environ.get("S2_API_KEY", "")
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "")
OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO", UNPAYWALL_EMAIL)  # polite pool fallback
PHILPAPERS_API_ID = os.environ.get("PHILPAPERS_API_ID", "")
PHILPAPERS_API_KEY = os.environ.get("PHILPAPERS_API_KEY", "")
PROCESS_PDF_SCRIPT = Path(__file__).parent / "process_pdf.py"
PROCESS_TEX_SCRIPT = Path(__file__).parent / "process_tex.py"

_SCHEMA_VERSION = 21


# Phase A.1 (plan v3 §3): page-marker extraction-quality classifier.
#
# Returns 0 (clean), 1 (problematic), or None (unknown — empty body).
# Conservative — flag only the failure modes we can detect WITHOUT
# re-running Docling/OCR:
#   * no `<!-- page N -->` markers found in non-empty processed_text
#     (Docling silently dropped pagination)
#   * any consecutive (b - a) marker diff is < 0 (descending = scanning
#     mishap caused page numbers to go backwards)
#
# Legitimate non-monotone-but-positive transitions (front-matter -16
# -> -15 -> -13 from books that skip even/odd in roman-numeral
# preface, conference proceedings starting at page 200, section breaks
# that legitimately skip a page) are NOT flagged: an audit on the
# 2,889 cite-ready papers in papers.db (maintenance/audit_page_gaps.py)
# found 77 papers (2.7%) with positive gaps and they correlate with
# real book layouts, not extraction errors.
def _classify_page_markers(processed_text: str) -> int | None:
    """Return tri-state: 0=clean, 1=problematic, None=no body.

    See Phase A.1 docs in plan v3 §3.
    """
    if not processed_text:
        return None
    markers = [int(m.group(1)) for m in re.finditer(r"<!--\s*page\s+(-?\d+)\s*-->", processed_text)]
    if not markers:
        return 1  # silent extraction failure
    for a, b in zip(markers, markers[1:]):
        if b - a < 0:
            return 1
    return 0

mcp = FastMCP("semantic-scholar")

class _Throttle:
    """Per-API rate limiter."""
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


# S2 with key: 1 RPS across all endpoints (current key tier)
# S2 without key: 5 RPS (conservative within 5000/5min shared pool)
_s2_detail_throttle = _Throttle(1.0 if S2_API_KEY else 0.2)   # 1 RPS with key, 5 RPS without
_s2_search_throttle = _Throttle(1.0)                           # 1 RPS always
_unpaywall_throttle = _Throttle(1.0)                           # 1 RPS (no documented limit, stay polite)
_openalex_throttle = _Throttle(0.125)                          # 8 RPS (conservative under 10 RPS limit, post Feb 2026 pricing change)
_arxiv_throttle = _Throttle(3.0)                               # 0.33 RPS — arXiv TOS requires >= 3s between requests
_philpapers_throttle = _Throttle(1.0)                          # 1 RPS — no documented limit, conservative behind Cloudflare


# --- Embedding Models (lazy singletons) ---

_embed_model = None
_qwen3_mlx_model = None  # Phase 4R: optional Qwen3-Embedding-4B MLX backend
_qwen3_mlx_tokenizer = None
_cross_encoder_model = None
_nli_model = None
# Tracks which NLI model id has had its label-permutation validated by
# the gold probes. Indexed by model id so swapping models forces a fresh
# validation. Populated lazily by _validate_nli_label_perm on the first
# verify_claim call (probes run when the NLI model is first loaded, not
# at server startup).
_nli_perm_validated: set[str] = set()
_db_initialized = False
_db_init_lock = threading.Lock()

# Phase 4R: embedder selection. EMBED_VARIANT env var controls which
# backend search_local + embedding writers use. Backwards-compatible
# default keeps nomic-embed-text-v1.5 (768d, CPU PyTorch) — no metric
# regression vs prior commits.
#
# Variants:
#   nomic-768d           (default) sentence-transformers/nomic CPU
#   qwen3-mlx-{512|1024|2560} Qwen3-Embedding-4B-4bit-DWQ via mlx-embeddings
#                              with Matryoshka truncation to the named dim
#
# Choosing a non-default forces the call site through the MLX shim and
# routes vec queries to a separate vec_papers_qwen3_<dim> /
# vec_chunks_qwen3_<dim> table (separate-table policy keeps a clean
# rollback story; nomic embeddings stay queryable side-by-side).
EMBED_VARIANT = os.environ.get("EMBED_VARIANT", "nomic-768d").strip().lower()
QWEN3_MLX_MODEL_ID = os.environ.get(
    "QWEN3_MLX_MODEL_ID", "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
)
QWEN3_MLX_NATIVE_DIM = 2560  # native output of Qwen3-Embedding-4B
_QWEN3_MLX_VALID_TARGET_DIMS = (512, 1024, 2560)

# Locks guard lazy model loading so two asyncio.to_thread callers hitting a
# cold server don't both allocate the SentenceTransformer / CrossEncoder and
# blow through RAM. Double-checked locking: read the global unlocked first,
# take the lock only on the cold path, then re-check inside the lock.
_embed_model_lock = threading.Lock()
_qwen3_mlx_lock = threading.Lock()
# Round-3 review HIGH fix: separate lock for MLX inference (vs the
# load lock above). MLX/Metal kernels are not safe under concurrent
# calls on the same model object — silent corruption of embedding
# vectors at concurrency >= 2. The load lock only protects the
# one-time setup; this lock guards every generate() call.
_qwen3_mlx_infer_lock = threading.Lock()
_cross_encoder_lock = threading.Lock()
_nli_model_lock = threading.Lock()


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        with _embed_model_lock:
            if _embed_model is None:
                import torch
                torch.set_num_threads(8)
                os.environ.setdefault("OMP_NUM_THREADS", "8")
                from sentence_transformers import SentenceTransformer
                _embed_model = SentenceTransformer(
                    "nomic-ai/nomic-embed-text-v1.5",
                    trust_remote_code=True,
                    device="cpu",
                )
    return _embed_model


def _parse_qwen3_target_dim(variant: str) -> int:
    """Validate qwen3-mlx-<dim> variant and return <dim>.

    Raises ValueError on unsupported dim. Phase 4R Matryoshka cuts
    are restricted to {512, 1024, 2560} per plan v3 §3 — anything
    else creates corpus-scale storage churn without a tested
    quality data point.
    """
    parts = variant.split("-")
    if len(parts) < 3:
        raise ValueError(
            f"EMBED_VARIANT '{variant}' missing dim suffix; expected "
            f"qwen3-mlx-512 / qwen3-mlx-1024 / qwen3-mlx-2560."
        )
    try:
        dim = int(parts[-1])
    except ValueError as exc:
        raise ValueError(
            f"EMBED_VARIANT '{variant}' dim suffix not an int."
        ) from exc
    if dim not in _QWEN3_MLX_VALID_TARGET_DIMS:
        raise ValueError(
            f"EMBED_VARIANT '{variant}' dim={dim} not in {_QWEN3_MLX_VALID_TARGET_DIMS}."
        )
    return dim


def _get_qwen3_mlx_embed():
    """Lazy-load Qwen3-Embedding-4B MLX. Returns (model, tokenizer).

    Phase 4R: uses mlx_embeddings.load to pull the MLX-converted
    model from QWEN3_MLX_MODEL_ID. ~60s cold-start on first call,
    ~3GB RAM. GPU (Metal) is the default backend.
    """
    global _qwen3_mlx_model, _qwen3_mlx_tokenizer
    if _qwen3_mlx_model is None:
        with _qwen3_mlx_lock:
            if _qwen3_mlx_model is None:
                from mlx_embeddings import load as _mlx_load  # type: ignore[import-not-found]

                model, tokenizer = _mlx_load(QWEN3_MLX_MODEL_ID)
                _qwen3_mlx_model = model
                _qwen3_mlx_tokenizer = tokenizer
    return _qwen3_mlx_model, _qwen3_mlx_tokenizer


def _vec_table_pair() -> tuple[str, str]:
    """Return (vec_papers_table, vec_chunks_table) for the active EMBED_VARIANT.

    Phase 4R: routing for variant-aware sqlite-vec queries. Default
    (nomic-768d) maps to the legacy `vec_papers` / `vec_chunks` tables;
    Qwen3 variants map to `vec_papers_qwen3_<dim>` / `vec_chunks_qwen3_<dim>`.
    """
    if EMBED_VARIANT == "nomic-768d":
        return "vec_papers", "vec_chunks"
    if EMBED_VARIANT.startswith("qwen3-mlx-"):
        dim = _parse_qwen3_target_dim(EMBED_VARIANT)
        return f"vec_papers_qwen3_{dim}", f"vec_chunks_qwen3_{dim}"
    raise ValueError(f"Unknown EMBED_VARIANT '{EMBED_VARIANT}'.")


def _qwen3_mlx_encode(texts: list[str], target_dim: int) -> list[bytes]:
    """Encode texts via Qwen3-MLX, Matryoshka-truncate, L2-normalize, bytes.

    Returns one float32 byte-buffer per text, length 4*target_dim,
    suitable for sqlite-vec virtual-table writes. Truncation is
    correct because Qwen3-Embedding is MRL-trained (any prefix of
    the native 2560d vector is a valid embedding).
    """
    if not texts:
        return []
    from mlx_embeddings import generate as _mlx_generate  # type: ignore[import-not-found]
    import mlx.core as mx  # type: ignore[import-not-found]

    model, tokenizer = _get_qwen3_mlx_embed()
    # Round-3 review HIGH fix: serialize Metal GPU inference. MLX
    # operations on a shared model object aren't thread-safe; concurrent
    # asyncio.to_thread calls (e.g. bench --concurrency 4 with
    # EMBED_VARIANT=qwen3-mlx-1024) would silently corrupt outputs.
    with _qwen3_mlx_infer_lock:
        out = _mlx_generate(model, tokenizer, texts=texts)
    # Quantized MLX variants don't expose a PEP-3118 buffer numpy can
    # reinterpret directly; cast to fp32 in MLX, then go via tolist().
    raw = mx.array(out.text_embeds).astype(mx.float32)
    arr = np.array(raw.tolist(), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != QWEN3_MLX_NATIVE_DIM:
        raise RuntimeError(
            f"Qwen3-MLX returned shape {arr.shape}; expected (n, {QWEN3_MLX_NATIVE_DIM})."
        )
    # Matryoshka truncate
    if target_dim < QWEN3_MLX_NATIVE_DIM:
        arr = arr[:, :target_dim]
    # L2-normalize per row (sentence-transformers default for retrieval)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return [row.astype(np.float32).tobytes() for row in arr]


_RERANKER_VARIANTS: dict = {
    # Phase 6 (Plan v3 §3 revised + post-deliberation 2026-04-25):
    # rerankers library abstraction with license-verified models only.
    # bge-m3 = primary (multilingual, 568M), mxbai-large = scientific
    # (435M, BEIR 57.49). jina-reranker-v3 ruled out per CC-BY-NC-4.0.
    # ms-marco kept as backwards-compat fallback.
    #
    # Each entry: {model_id, model_type (rerankers backend hint), license}.
    # _validate_reranker_license raises if any non-permissive license
    # appears at startup; safety belt against an upstream model-card
    # change re-locking a previously-OK model behind NC.
    "ms-marco": {
        "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "model_type": "cross-encoder",
        "license": "apache-2.0",
    },
    "bge-m3": {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "model_type": "cross-encoder",
        "license": "apache-2.0",
    },
    "mxbai-large": {
        "model_id": "mixedbread-ai/mxbai-rerank-large-v2",
        "model_type": "cross-encoder",
        "license": "apache-2.0",
    },
}
_DEFAULT_RERANKER_VARIANT = os.environ.get("RERANKER_VARIANT", "ms-marco")
_PERMISSIVE_LICENSES = {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause"}


def _validate_reranker_variant(variant: str) -> dict:
    cfg = _RERANKER_VARIANTS.get(variant)
    if cfg is None:
        raise RuntimeError(
            f"Unknown RERANKER_VARIANT={variant!r}. "
            f"Valid: {sorted(_RERANKER_VARIANTS)}"
        )
    lic = cfg.get("license", "").lower()
    if lic not in _PERMISSIVE_LICENSES:
        raise RuntimeError(
            f"Reranker variant {variant!r} declares non-permissive license "
            f"{lic!r}. The default reranker set is restricted to commercially "
            f"redistributable models so downstream users do not inherit a "
            f"viral or non-commercial license. Update _RERANKER_VARIANTS to "
            f"opt in, or pick a different variant."
        )
    return cfg


class _RerankerAdapter:
    """Adapter exposing the .predict(pairs) interface that
    sentence_transformers.CrossEncoder callers expect, on top of the
    rerankers.Reranker.rank() API.

    Backwards-compat goal: Phase 6 swaps the underlying model from
    ms-marco-MiniLM-L-6-v2 (Plan v1-v2) to bge-reranker-v2-m3 +
    mxbai-rerank-large-v2 (Plan v3 §3) without changing the three
    upstream callers (_rerank, search_local chunk-rerank, find_quotation
    pre-filter). All three call .predict(pairs) and consume returned
    scores; this adapter preserves that contract.
    """

    def __init__(self, variant: str) -> None:
        cfg = _validate_reranker_variant(variant)
        self.variant = variant
        self.model_id = cfg["model_id"]
        from rerankers import Reranker
        kwargs: dict = {"verbose": 0}
        mt = cfg.get("model_type")
        if mt:
            kwargs["model_type"] = mt
        self._r = Reranker(self.model_id, **kwargs)

    def predict(self, pairs, show_progress_bar: bool = False, **_) -> list[float]:
        # Callers pass list of (query, doc) tuples, possibly with the
        # same query repeated. Group by query so the underlying ranker
        # sees one query per call (matching its native API).
        if not pairs:
            return []
        query_groups: dict[str, list[tuple[int, str]]] = {}
        for global_idx, (q, d) in enumerate(pairs):
            query_groups.setdefault(q, []).append((global_idx, d))
        scores: list[float] = [0.0] * len(pairs)
        for q, items in query_groups.items():
            docs = [d for _, d in items]
            doc_ids = list(range(len(docs)))
            try:
                ranked = self._r.rank(query=q, docs=docs, doc_ids=doc_ids)
            except TypeError:
                # Older rerankers releases without doc_ids kwarg.
                ranked = self._r.rank(query=q, docs=docs)
            for result in ranked.results:
                local_idx = result.document.doc_id
                if not isinstance(local_idx, int) or local_idx >= len(items):
                    # Defensive: doc_id might be the doc text itself on
                    # some backends. Fall back to text match.
                    local_idx = next(
                        (i for i, (_, d) in enumerate(items)
                         if d == result.document.text),
                        None,
                    )
                    if local_idx is None:
                        continue
                global_idx = items[local_idx][0]
                scores[global_idx] = float(result.score)
        return scores


def _get_cross_encoder():
    """Backwards-compat name. Returns the configured reranker variant
    via the new _RerankerAdapter. The old name is kept so all three
    historical callers (_rerank, search_local chunk-rerank, find_quotation
    pre-filter) keep working without edit. Phase 6 only flips the
    underlying model via RERANKER_VARIANT env var.
    """
    return _get_reranker()


def _get_reranker(variant: str | None = None):
    """Return the configured reranker, defaulting to the env-var variant.

    Phase 6: env var RERANKER_VARIANT selects the model at startup.
    Default 'ms-marco' preserves Plan v1-v2 behavior for runs that
    don't opt in. Set RERANKER_VARIANT=bge-m3 (or mxbai-large) for the
    Plan v3 §3 upgrade. Per-call override via the variant parameter is
    available but currently unused (per-domain routing deferred until
    after a single-variant A/B confirms lift).
    """
    global _cross_encoder_model
    target = variant or _DEFAULT_RERANKER_VARIANT
    if _cross_encoder_model is None or getattr(_cross_encoder_model, "variant", None) != target:
        with _cross_encoder_lock:
            if _cross_encoder_model is None or getattr(_cross_encoder_model, "variant", None) != target:
                _cross_encoder_model = _RerankerAdapter(target)
    return _cross_encoder_model


_NLI_MODEL_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


# Gold probes for the NLI label permutation. Three pairs covering
# (entailment / contradiction / neutral) — independent of any specific
# model card. If _NLI_LABEL_PERM is wrong (model swap, upstream
# id2label change), at least one of these returns the wrong label and
# we raise instead of silently mispermuting every verify_claim verdict.
# The probes run on first NLI-model load (lazy), not at server startup,
# so a fresh boot does not pay the model-load cost until verify_claim
# is actually called.
_NLI_GOLD_PROBES: tuple[tuple[str, str, str], ...] = (
    (
        "The study found a statistically significant reduction in mortality.",
        "The study found a reduction.",
        "support",
    ),
    (
        "The study found no effect.",
        "The study found a significant effect.",
        "refute",
    ),
    (
        "The weather was sunny that day.",
        "The stock market rose.",
        "insufficient",
    ),
)


def _validate_nli_label_perm(model, model_id: str) -> None:
    """Run the gold probes and raise if _NLI_LABEL_PERM is wrong.

    No-op once the model id has been validated in this process. Cached
    in _nli_perm_validated.
    """
    if model_id in _nli_perm_validated:
        return
    fails: list[str] = []
    labels = ("refute", "support", "insufficient")
    for premise, hypothesis, expected in _NLI_GOLD_PROBES:
        raw = model.predict([(premise, hypothesis)])
        if hasattr(raw, "tolist"):
            native = list(raw[0])
        else:
            native = list(raw)
        m = max(native)
        # Stable softmax (matches _nli_predict math).
        exp = [pow(2.71828182845904, s - m) for s in native]
        z = sum(exp) or 1.0
        native_probs = [e / z for e in exp]
        ours = [native_probs[_NLI_LABEL_PERM[i]] for i in range(3)]
        idx = ours.index(max(ours))
        got = labels[idx]
        if got != expected:
            fails.append(
                f"probe '{premise[:50]}...' expected {expected}, got {got} (conf={ours[idx]:.3f})"
            )
    if fails:
        raise RuntimeError(
            "NLI label-permutation probe FAILED — _NLI_LABEL_PERM is mispermuted "
            f"for model '{model_id}'. Mismatches:\n"
            + "\n".join("  " + f for f in fails)
            + "\nFix _NLI_LABEL_PERM in server.py before any verify_claim call. "
            "verify_claim verdicts will be wrong with the current permutation."
        )
    _nli_perm_validated.add(model_id)


def _get_nli_model():
    """Return a sentence-transformers CrossEncoder trained on NLI data.

    Phase 5a.1: swapped cross-encoder/nli-deberta-v3-base (400M) →
    MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (790M).
    The large model reports +4-7 F1 on ANLI-R3 vs base per its model
    card, and its training mix (MNLI + FEVER + ANLI-R1-3 + LingNLI +
    WANLI) is closer to the pincite use-case than the base's
    MNLI-only training.

    LABEL ORDER IS DIFFERENT between the two models:
      base: (contradiction, entailment, neutral)
      large: (entailment, neutral, contradiction)
    The verify_claim mapping to (refute, support, insufficient) is
    permuted in _nli_predict below.

    Phase A.2: on first load, runs three gold probes via
    _validate_nli_label_perm and raises if _NLI_LABEL_PERM is wrong.
    Cached per model id so subsequent calls don't re-probe.

    Lazy-loaded and guarded by a threading.Lock to prevent double-load
    under concurrent callers.
    """
    global _nli_model
    if _nli_model is None:
        with _nli_model_lock:
            if _nli_model is None:
                from sentence_transformers import CrossEncoder
                # Round-4 review CRITICAL fix: previously the global was
                # assigned BEFORE validation. If validation raised, the
                # global retained the unvalidated model object; the
                # outer `if _nli_model is None` check on subsequent
                # calls would skip validation entirely, producing
                # silently mispermuted verdicts. Validate FIRST on a
                # local, then publish only if validation passes.
                candidate = CrossEncoder(_NLI_MODEL_ID, device="cpu")
                _validate_nli_label_perm(candidate, _NLI_MODEL_ID)
                _nli_model = candidate
    return _nli_model


# Label order permutation for the large NLI model.
# Index into the logit vector: position i in _nli_model.predict output.
# MoritzLaurer model card: id2label = {0: entailment, 1: neutral, 2: contradiction}.
# Map to our (refute, support, insufficient) convention:
#   our_idx 0 (refute)       = model_idx 2 (contradiction)
#   our_idx 1 (support)      = model_idx 0 (entailment)
#   our_idx 2 (insufficient) = model_idx 1 (neutral)
_NLI_LABEL_PERM = (2, 0, 1)


def _encode_to_bytes(model, texts: list[str]) -> list[bytes]:
    """Encode texts and return as raw bytes for sqlite-vec. Caller handles prefixing."""
    embeddings = model.encode(texts, show_progress_bar=False)
    return [e.astype(np.float32).tobytes() for e in embeddings]


def _embed_texts(texts: list[str]) -> list[bytes]:
    """Embed document texts for storage. Adds 'search_document: ' prefix
    (nomic) or no prefix (Qwen3-MLX, which has its own template).
    """
    if EMBED_VARIANT == "nomic-768d":
        model = _get_embed_model()
        return _encode_to_bytes(model, [f"search_document: {t}" for t in texts])
    if EMBED_VARIANT.startswith("qwen3-mlx-"):
        target_dim = _parse_qwen3_target_dim(EMBED_VARIANT)
        return _qwen3_mlx_encode(texts, target_dim)
    raise ValueError(f"Unknown EMBED_VARIANT '{EMBED_VARIANT}'.")


_query_embed_cache: dict[str, bytes] = {}
_QUERY_CACHE_MAX = 32
# Round-2 review CRITICAL fix: cache eviction was a non-atomic
# read-modify-write (len check + pop + write) accessed from
# asyncio.to_thread workers. At --concurrency >= 2, two threads could
# both observe `len >= 32`, both pop the same key (one KeyErrors), and
# both write — corrupting the cache. Lock guards eviction + write.
_query_embed_cache_lock = threading.Lock()


def _embed_query_text(text: str) -> bytes:
    """Embed a query text for search. Adds 'search_query: ' prefix
    (nomic) or no prefix (Qwen3-MLX, which has its own query template).
    Cached.
    """
    cache_key = f"{EMBED_VARIANT}:{text}"
    cached = _query_embed_cache.get(cache_key)
    if cached is not None:
        return cached
    if EMBED_VARIANT == "nomic-768d":
        model = _get_embed_model()
        result = _encode_to_bytes(model, [f"search_query: {text}"])[0]
    elif EMBED_VARIANT.startswith("qwen3-mlx-"):
        target_dim = _parse_qwen3_target_dim(EMBED_VARIANT)
        result = _qwen3_mlx_encode([text], target_dim)[0]
    else:
        raise ValueError(f"Unknown EMBED_VARIANT '{EMBED_VARIANT}'.")
    with _query_embed_cache_lock:
        if cache_key in _query_embed_cache:
            # Another thread cached the same key while we encoded; reuse.
            return _query_embed_cache[cache_key]
        if len(_query_embed_cache) >= _QUERY_CACHE_MAX:
            # Pop oldest under the lock so two threads don't both pop
            # the same key (which would KeyError in the second pop).
            try:
                _query_embed_cache.pop(next(iter(_query_embed_cache)))
            except StopIteration:
                pass
        _query_embed_cache[cache_key] = result
    return result


def _paper_embed_text(
    title: str,
    abstract: str | None,
    processed_text: str | None,
    tex_text: str | None = None,
) -> str | None:
    # Prefer tex_text over processed_text for embeddings — math is cleaner from TeX source.
    full_text = tex_text if tex_text else processed_text
    if abstract:
        return f"{title}. {abstract}"
    if full_text:
        words = full_text.split()[:1000]
        return f"{title}. {' '.join(words)}"
    if title:
        return title
    return None


def _fts5_safe_query(conn: sqlite3.Connection, sql: str, query: str, limit: int) -> list:
    """Run an FTS5 MATCH query, falling back to quoted literal on syntax error."""
    try:
        return conn.execute(sql, (query, limit)).fetchall()
    except sqlite3.OperationalError:
        # Query contains FTS5 syntax chars like ( ) " AND OR NOT *
        # Fall back to quoting the entire query as a literal phrase
        escaped = '"' + query.replace('"', '""') + '"'
        try:
            return conn.execute(sql, (escaped, limit)).fetchall()
        except sqlite3.OperationalError:
            return []


# Phase A.4 (plan v3 §3, post-bench tuning): reranker latency + candidate-count guards.
#
# - MAX_RERANK_CANDIDATES caps how many (query, doc) pairs we score per
#   call. Beyond the cap we keep the prefix (which arrives in RRF order
#   from upstream callers). Prevents pathological candidate lists
#   (>1000 pairs from a wide search) from blowing p95 latency.
# - RERANK_SLOW_LATENCY_S logs as a "slow" event so per-bench latency
#   regressions are loud.
# - _rerank_metrics is a process-local counter accessible to bench
#   harness (latency mean/p95, timeout count, slow count).
#
# Cap-level tuning: the v2 baseline corpus produces ~277 candidates per
# search_local rerank call. Plan v3 §3 estimated 1-2pt aggregate gain
# from cap=100; empirically (post-A bench at SHA e5d4134) cap=100 was
# RECALL-regressive: paper@5 -0.73pt, paper@20 -2.21pt vs uncapped
# baseline. The cross-encoder DOES find relevant candidates in the
# 100-300 RRF range often enough that capping at 100 silently drops
# them. Default raised to 500 (effectively unlimited on current corpus)
# while preserving telemetry; configurable via RESEARCH_MCP_RERANK_CAP
# env var for latency-bound use cases.
MAX_RERANK_CANDIDATES = int(os.environ.get("RESEARCH_MCP_RERANK_CAP", "500"))
RERANK_SLOW_LATENCY_S = 5.0
_rerank_metrics: dict[str, float | int] = {
    "calls": 0,
    "candidates_total": 0,
    "candidates_truncated": 0,
    "latency_ms_total": 0.0,
    "latency_ms_max": 0.0,
    "slow_count": 0,
}
# Round-2 review CRITICAL fix: _rerank is called via asyncio.to_thread,
# so multiple worker threads can race the read-modify-write counters
# (especially the latency_ms_max compare-and-swap). Lock guards the
# whole metric update block.
_rerank_metrics_lock = threading.Lock()


def _rerank(query: str, candidates: list[tuple[str, str]], top_k: int = 30) -> list[int]:
    """Rerank candidates using cross-encoder. Returns indices sorted by relevance.

    Phase A.4: caps candidates at MAX_RERANK_CANDIDATES (default 500;
    env-configurable via RESEARCH_MCP_RERANK_CAP)
    and records latency in _rerank_metrics. The cap protects p95
    latency under wide-recall searches that dump 500-2000 candidates
    into the reranker. Returned indices map back to the ORIGINAL
    candidate list (caller-side ordering preserved); only candidates
    BEYOND the cap are dropped from scoring.

    The prior comment claimed "Scores ALL candidates (the prior [:50]
    cap was a bug)" — true that 50 was too small, but the right
    answer is a slightly larger budget bounded by latency, not
    unbounded scoring.
    """
    if not candidates:
        return []
    n_candidates = len(candidates)
    truncated = n_candidates > MAX_RERANK_CANDIDATES
    if truncated:
        candidates = candidates[:MAX_RERANK_CANDIDATES]
    model = _get_cross_encoder()
    pairs = [(query, doc_text) for _, doc_text in candidates]
    t0 = time.perf_counter()
    scores = model.predict(pairs, show_progress_bar=False)
    elapsed_s = time.perf_counter() - t0
    elapsed_ms = elapsed_s * 1000.0

    with _rerank_metrics_lock:
        _rerank_metrics["calls"] += 1
        _rerank_metrics["candidates_total"] += n_candidates
        if truncated:
            _rerank_metrics["candidates_truncated"] += n_candidates - MAX_RERANK_CANDIDATES
        _rerank_metrics["latency_ms_total"] += elapsed_ms
        if elapsed_ms > _rerank_metrics["latency_ms_max"]:
            _rerank_metrics["latency_ms_max"] = elapsed_ms
        if elapsed_s > RERANK_SLOW_LATENCY_S:
            _rerank_metrics["slow_count"] += 1
            print(
                f"[rerank] SLOW: {elapsed_s:.2f}s on {len(candidates)} candidates "
                f"(threshold {RERANK_SLOW_LATENCY_S}s)",
                file=sys.stderr,
            )

    ranked_indices = sorted(range(len(scores)), key=lambda i: -scores[i])
    return ranked_indices[:top_k]


def get_rerank_metrics() -> dict[str, float | int]:
    """Snapshot of module-local rerank counters. Bench can call this
    after a run to capture latency aggregates.

    Returns a fresh dict; mutating it doesn't affect the underlying
    counters. Locked so the snapshot reflects a consistent state.
    """
    with _rerank_metrics_lock:
        snap = dict(_rerank_metrics)
    if snap["calls"] > 0:
        snap["latency_ms_mean"] = snap["latency_ms_total"] / snap["calls"]
    else:
        snap["latency_ms_mean"] = 0.0
    return snap


def reset_rerank_metrics() -> None:
    """Zero the counters. Bench harness uses this between runs."""
    with _rerank_metrics_lock:
        for k in _rerank_metrics:
            _rerank_metrics[k] = 0 if isinstance(_rerank_metrics[k], int) else 0.0


# --- Local Paper Library (SQLite) ---

def _create_fts_and_triggers(conn: sqlite3.Connection) -> None:
    # Schema v13: papers_fts indexes the UNION of processed_text and
    # tex_text for dual-source papers via concatenation into the
    # processed_text column; tex_text column is always empty.
    #
    # Evolution:
    #  v10: indexed BOTH processed_text AND tex_text columns — caused
    #       BM25 double-counting for dual-source papers.
    #  v11: single-body policy (tex_text wins if >500, else processed).
    #       Fixed the double-count but caused RECALL LOSS: 78 dual-source
    #       papers have genuinely different content between the two
    #       sources (different extractions, missing tokens), and
    #       processed-only tokens like "humidity" became unmatchable.
    #  v13: concatenation. Both bodies indexed in ONE column, so every
    #       term is matchable AND term frequency is not doubled.
    #
    # Why concatenation doesn't re-introduce the v10 skew: BM25 computes
    # TF saturation per-column. For a term appearing TF_p times in
    # processed_text AND TF_t times in tex_text, concatenation gives
    # (TF_p + TF_t) in the single column, which BM25 saturates via a
    # log scale. Dual indexing gave each column its own saturated
    # contribution that summed linearly — about 2x. Concatenation
    # recovers single-source-equivalent scoring.
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
            paper_id, title, abstract, tldr, processed_text, tex_text,
            content=papers,
            content_rowid=rowid
        )
    """)
    # Schema v14: index ONLY cite-ready rows (has_full_text=1) in papers_fts.
    # Metadata-only rows (16K+ in live DB) no longer pollute BM25 corpus
    # statistics. Each trigger uses `SELECT ... WHERE has_full_text = 1`
    # so the INSERT is skipped entirely for metadata-only rows.
    conn.executescript("""
        CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
            INSERT INTO papers_fts(rowid, paper_id, title, abstract, tldr, processed_text, tex_text)
            SELECT
                new.rowid, new.paper_id, new.title, new.abstract, new.tldr,
                COALESCE(new.tex_text, '') || ' ' || COALESCE(new.processed_text, ''),
                ''
            WHERE new.has_full_text = 1;
        END;
        CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
            INSERT INTO papers_fts(papers_fts, rowid, paper_id, title, abstract, tldr, processed_text, tex_text)
            SELECT
                'delete', old.rowid, old.paper_id, old.title, old.abstract, old.tldr,
                COALESCE(old.tex_text, '') || ' ' || COALESCE(old.processed_text, ''),
                ''
            WHERE old.has_full_text = 1;
        END;
        CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
            INSERT INTO papers_fts(papers_fts, rowid, paper_id, title, abstract, tldr, processed_text, tex_text)
            SELECT
                'delete', old.rowid, old.paper_id, old.title, old.abstract, old.tldr,
                COALESCE(old.tex_text, '') || ' ' || COALESCE(old.processed_text, ''),
                ''
            WHERE old.has_full_text = 1;
            INSERT INTO papers_fts(rowid, paper_id, title, abstract, tldr, processed_text, tex_text)
            SELECT
                new.rowid, new.paper_id, new.title, new.abstract, new.tldr,
                COALESCE(new.tex_text, '') || ' ' || COALESCE(new.processed_text, ''),
                ''
            WHERE new.has_full_text = 1;
        END;
    """)


def _create_chunks_fts_and_triggers(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            section_header TEXT,
            page_start INTEGER,
            page_end INTEGER
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_paper ON paper_chunks(paper_id)")
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS paper_chunks_fts USING fts5(
            chunk_text,
            content=paper_chunks,
            content_rowid=chunk_id
        )
    """)
    conn.executescript("""
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON paper_chunks BEGIN
            INSERT INTO paper_chunks_fts(rowid, chunk_text)
            VALUES (new.chunk_id, new.chunk_text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON paper_chunks BEGIN
            INSERT INTO paper_chunks_fts(paper_chunks_fts, rowid, chunk_text)
            VALUES ('delete', old.chunk_id, old.chunk_text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON paper_chunks BEGIN
            INSERT INTO paper_chunks_fts(paper_chunks_fts, rowid, chunk_text)
            VALUES ('delete', old.chunk_id, old.chunk_text);
            INSERT INTO paper_chunks_fts(rowid, chunk_text)
            VALUES (new.chunk_id, new.chunk_text);
        END;
    """)


def _all_vec_chunk_tables() -> tuple[str, ...]:
    """All vec-chunk table names that should be cleaned when chunks are deleted.

    Round-2 review fix: previously this list was open-coded at every
    delete site (_chunk_processed_text, _clear_paper_text,
    prune_unverified, dedup_papers.merge_into). Centralizing here so a
    new variant (e.g., qwen3-mlx-2560 backfilled) is covered everywhere
    automatically. The v21 cascade triggers (papers_ad_vec_qwen3_<dim>,
    chunks_ad_vec_qwen3_<dim>) handle this too — explicit pre-delete is
    defense-in-depth per the legacy vec_chunks comment about sqlite-vec
    virtual-table trigger fragility.
    """
    return ("vec_chunks",) + tuple(
        f"vec_chunks_qwen3_{dim}" for dim in _QWEN3_MLX_VALID_TARGET_DIMS
    )


def _all_vec_paper_tables() -> tuple[str, ...]:
    """All vec-paper table names that should be cleaned when a paper is deleted."""
    return ("vec_papers",) + tuple(
        f"vec_papers_qwen3_{dim}" for dim in _QWEN3_MLX_VALID_TARGET_DIMS
    )


def _delete_vec_chunks_for_paper(conn: sqlite3.Connection, paper_id: str) -> None:
    """Delete vec rows for ALL chunk variants belonging to paper_id.

    Round-2 review HIGH fix: replaces open-coded delete loops at the
    four call sites. Tolerates "no such table" / "no such module" so
    fresh DBs and DBs without sqlite-vec loaded don't raise.
    """
    for vec_table in _all_vec_chunk_tables():
        try:
            conn.execute(f"""
                DELETE FROM {vec_table} WHERE rowid IN (
                    SELECT chunk_id FROM paper_chunks WHERE paper_id = ?
                )
            """, (paper_id,))
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if "no such table" not in msg and "no such module" not in msg:
                print(f"  {vec_table} delete error ({paper_id}): {e}", file=sys.stderr)
                raise


def _delete_vec_papers_for_paper(conn: sqlite3.Connection, paper_rowid: int) -> None:
    """Delete vec-paper rows for ALL variants for a given paper rowid."""
    for vec_table in _all_vec_paper_tables():
        try:
            conn.execute(f"DELETE FROM {vec_table} WHERE rowid = ?", (paper_rowid,))
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if "no such table" not in msg and "no such module" not in msg:
                print(f"  {vec_table} paper delete error (rowid={paper_rowid}): {e}", file=sys.stderr)
                raise


def _chunk_processed_text(conn: sqlite3.Connection, paper_id: str, processed_text: str, target_words: int = 500) -> int:
    # Delete old chunk embeddings before deleting chunks (chunk_ids become invalid).
    # Wave-5 review HIGH fix: clears all four vec-chunk variants via
    # _delete_vec_chunks_for_paper. Round-2 fix: extracted into helper
    # so prune_unverified / _clear_paper_text / dedup share the same
    # invariant.
    _delete_vec_chunks_for_paper(conn, paper_id)
    conn.execute("DELETE FROM paper_chunks WHERE paper_id = ?", (paper_id,))

    paragraphs = processed_text.split("\n\n")
    chunks: list[dict] = []
    current_parts: list[str] = []
    current_words = 0
    current_section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    # Phase 2.2: track whether the current chunk has seen its own page
    # marker yet. Previously the emit() path copied page_end → page_start
    # for the NEXT chunk. When that next chunk's first paragraph carried a
    # newer page marker, the `if page_start is None` guard refused to
    # overwrite the stale inherited start and the chunk was stored with
    # (old_end, new_page) — a 1-off at paragraph-span boundaries. Now the
    # first marker in a new chunk ALWAYS takes precedence over the
    # inherited start; the inheritance survives only when the new chunk
    # begins mid-page (no explicit marker).
    chunk_has_marker: bool = False

    def emit():
        nonlocal current_parts, current_words, page_start, chunk_has_marker
        if current_parts:
            chunks.append({
                "text": "\n\n".join(current_parts),
                "section": current_section,
                "page_start": page_start,
                "page_end": page_end,
            })
            current_parts = []
            current_words = 0
            page_start = page_end
            chunk_has_marker = False

    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue

        if stripped.startswith("#"):
            current_section = stripped.lstrip("#").strip()

        page_match = re.search(r"<!-- page (\d+) -->", stripped)
        if page_match:
            pn = int(page_match.group(1))
            if not chunk_has_marker:
                page_start = pn
                chunk_has_marker = True
            page_end = pn

        word_count = len(stripped.split())

        if current_words + word_count > target_words and current_parts:
            emit()

        current_parts.append(para)
        current_words += word_count

    emit()

    for i, chunk in enumerate(chunks):
        conn.execute(
            "INSERT INTO paper_chunks (paper_id, chunk_index, chunk_text, section_header, page_start, page_end) VALUES (?, ?, ?, ?, ?, ?)",
            (paper_id, i, chunk["text"], chunk["section"], chunk["page_start"], chunk["page_end"]),
        )

    # Phase A.1: classify page-marker quality and persist to papers.
    # _classify_page_markers returns None for empty body, 0 for clean,
    # 1 for no markers OR descending markers. NULL stays NULL (caller
    # may have populated processed_text but markers are absent because
    # the body legitimately lacks pagination — e.g., an abstract dump).
    gap_class = _classify_page_markers(processed_text)
    if gap_class is not None:
        conn.execute(
            "UPDATE papers SET has_gapped_extraction = ? WHERE paper_id = ?",
            (gap_class, paper_id),
        )

    return len(chunks)


def _build_paper_passages(
    conn: sqlite3.Connection, paper_id: str, processed_text: str
) -> int:
    """Split processed_text on <!-- page N --> markers and store one
    passage per page in `paper_passages`. Idempotent: deletes existing
    rows for the paper first.

    Used for Bluebook-grade pincites where a multi-page chunk (which the
    500-word paper_chunks table routinely produces) is not a valid
    citation. paper_chunks stays the unit for semantic retrieval.
    """
    conn.execute("DELETE FROM paper_passages WHERE paper_id = ?", (paper_id,))

    if not processed_text:
        return 0

    # Iterate through text, tracking current page and section header.
    # When we hit a <!-- page N --> marker, close the current page
    # buffer and open a new one for page N.
    PAGE_MARKER = re.compile(r"<!--\s*page\s+(-?\d+)\s*-->")

    # Find all markers with their positions
    markers = [(m.start(), m.end(), int(m.group(1))) for m in PAGE_MARKER.finditer(processed_text)]
    if not markers:
        # No page markers at all; store the whole body as page "unknown".
        # Represent as page 0 (no legitimate printed page is 0) and
        # the search layer can label it "unknown pdf_page".
        text = processed_text.strip()
        if text:
            conn.execute(
                "INSERT INTO paper_passages (paper_id, page_num, section_header, passage_text) "
                "VALUES (?, 0, NULL, ?)",
                (paper_id, text),
            )
            return 1
        return 0

    # Walk segments between markers. Segment i spans from markers[i].end
    # to markers[i+1].start and belongs to markers[i].page_num. Anything
    # BEFORE the first marker belongs to the preceding page (unknown),
    # which we label page 0.
    count = 0
    current_section: str | None = None

    def _section_from(text: str) -> str | None:
        # Extract the most-recent # heading in the segment, if any.
        last = None
        for line in text.split("\n"):
            s = line.strip()
            if s.startswith("#"):
                last = s.lstrip("#").strip()
        return last

    # Segment 0: pre-first-marker body.
    first_start = markers[0][0]
    pre = processed_text[:first_start].strip()
    if pre:
        # Strip any remaining page-marker tokens (shouldn't occur but safe).
        pre_clean = PAGE_MARKER.sub("", pre).strip()
        if pre_clean:
            conn.execute(
                "INSERT INTO paper_passages (paper_id, page_num, section_header, passage_text) "
                "VALUES (?, 0, ?, ?)",
                (paper_id, _section_from(pre_clean), pre_clean),
            )
            count += 1
            current_section = _section_from(pre_clean) or current_section

    # Middle + last segments.
    for i, (mstart, mend, pnum) in enumerate(markers):
        seg_end = markers[i + 1][0] if i + 1 < len(markers) else len(processed_text)
        seg = processed_text[mend:seg_end].strip()
        # Strip any inline page markers from the segment body so pasted
        # quotations are clean; the page_num column is the source of truth.
        seg_clean = PAGE_MARKER.sub("", seg).strip()
        if not seg_clean:
            continue
        sec = _section_from(seg_clean) or current_section
        current_section = sec
        conn.execute(
            "INSERT INTO paper_passages (paper_id, page_num, section_header, passage_text) "
            "VALUES (?, ?, ?, ?)",
            (paper_id, pnum, sec, seg_clean),
        )
        count += 1

    return count


def _init_db() -> sqlite3.Connection:
    global _db_initialized
    # check_same_thread=False so async handlers can hand a connection to an
    # asyncio.to_thread worker for the CPU-bound embedding step without
    # crashing on Python's default thread-affinity check. SQLite itself is
    # thread-safe (built with SQLITE_THREADSAFE=1 in the stdlib shipped
    # sqlite) and our usage is always sequential per conn.
    conn = sqlite3.connect(str(DB_PATH), timeout=120, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    if _db_initialized:
        return conn

    # Serialize first-time schema migration across concurrent callers.
    # `asyncio.to_thread` callers can race this flag; threading.Lock is the
    # minimal correct primitive since server + scripts are single-process.
    with _db_init_lock:
        if _db_initialized:
            return conn
        _run_schema_migrations(conn)
        # Phase A.5: lock schema_version against silent drift. After
        # migrations complete, the schema_version row + PRAGMA
        # user_version must match _SCHEMA_VERSION. If they don't, the
        # migration silently failed and downstream queries would hit
        # missing columns/tables. Raising here is far better than
        # discovering it via a verify_claim crash 30 minutes into a
        # bench run.
        row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        observed = row[0] if row else None
        if observed != _SCHEMA_VERSION:
            raise RuntimeError(
                f"Schema version mismatch after migration: schema_version table "
                f"reports {observed!r}, _SCHEMA_VERSION constant is {_SCHEMA_VERSION}. "
                f"Migration silently failed. Inspect the DB at {DB_PATH} and "
                f"_run_schema_migrations in server.py."
            )
        # Mirror to PRAGMA user_version so external tools (sqlite3 CLI,
        # backup scripts) see the same value without needing to know
        # about our schema_version table layout.
        conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        _db_initialized = True
    return conn


def _run_schema_migrations(conn: sqlite3.Connection) -> None:
    """Run schema creation + migration statements. Must be called under _db_init_lock."""

    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY,
            doi TEXT,
            arxiv_id TEXT,
            pmid TEXT,
            title TEXT,
            authors TEXT,
            abstract TEXT,
            tldr TEXT,
            year INTEGER,
            venue TEXT,
            citation_count INTEGER,
            influential_citation_count INTEGER,
            fields_of_study TEXT,
            publication_types TEXT,
            publication_date TEXT,
            is_open_access INTEGER,
            oa_pdf_url TEXT,
            s2_url TEXT,
            local_pdf_path TEXT,
            processed_text TEXT,
            fwci REAL,
            citation_percentile REAL,
            venue_impact_factor REAL,
            venue_h_index INTEGER,
            venue_type TEXT,
            authority_tier INTEGER,
            is_retracted INTEGER DEFAULT 0,
            openalex_id TEXT,
            work_type TEXT,
            first_seen TEXT,
            last_updated TEXT
        )
    """)

    # Migrations: add columns that may be missing in older DBs
    for col, coltype in [
        ("processed_text", "TEXT"),
        ("fwci", "REAL"),
        ("citation_percentile", "REAL"),
        ("venue_impact_factor", "REAL"),
        ("venue_h_index", "INTEGER"),
        ("venue_type", "TEXT"),
        ("authority_tier", "INTEGER"),
        ("is_retracted", "INTEGER DEFAULT 0"),
        ("openalex_id", "TEXT"),
        ("work_type", "TEXT"),
        ("pdf_page_offset", "INTEGER DEFAULT 0"),
        ("pages_verified", "INTEGER DEFAULT 0"),
        ("verified", "INTEGER DEFAULT 0"),
        # TeX source support: arXiv TeX preserves math that Docling garbles in PDF extraction.
        # local_tex_path points to a directory under $TEX_DIR (see research_mcp.paths)
        # containing the full TeX source tree (main.tex, .bib, figures, .sty). tex_text holds the
        # pandoc-extracted plain text, preferred over processed_text for full-text reads
        # and embeddings when available.
        ("local_tex_path", "TEXT"),
        ("tex_text", "TEXT"),
        # Web capture support (schema v7): source_url stores the original URL for
        # web captures; capture_date stores when the page was captured. Both are
        # NULL for academic papers. Web captures use paper_id = "web:{slug}".
        ("source_url", "TEXT"),
        ("capture_date", "TEXT"),
        # Schema v8: has_full_text distinguishes "cite-ready" papers (have
        # processed_text or tex_text) from metadata-only rows. The `verified`
        # column retains its original meaning ("source authority known") and
        # remains 1 for metadata-only papers promoted by OpenAlex/S2 lookups,
        # but search/citation paths should prefer has_full_text for citation
        # readiness.
        ("has_full_text", "INTEGER DEFAULT 0"),
        # Schema v9: worker-advisory claim lock. NULL = free, "pid_uuid" =
        # claimed by that worker. Used by acquire_batch.py so two workers
        # downloading the same paper_id do not wipe each other's text via
        # overlapping store/verify/clear sequences. Stale claims older than
        # 1 hour are reaped at startup.
        ("_acquiring_by", "TEXT"),
        ("_acquiring_at", "TEXT"),
        # Round-3 review CRITICAL fix: schema v19's has_gapped_extraction
        # column was added inside the `current_version < 19` block at the
        # bottom of _run_schema_migrations. But _chunk_processed_text
        # (called by the v3 backfill) writes to that column, so a pre-v19
        # DB with non-empty processed_text rows would boot-loop on
        # `no such column: has_gapped_extraction`. Add it to the bootstrap
        # ALTER TABLE loop so it exists before any backfill block runs.
        # The v19 migration's separate ALTER + index commands remain
        # idempotent (IF NOT EXISTS guards in the column check + index).
        ("has_gapped_extraction", "INTEGER"),
    ]:
        try:
            conn.execute(f"ALTER TABLE papers ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass

    conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi)")
    # Expression index for case-insensitive DOI lookups. Without it,
    # `WHERE LOWER(doi) = LOWER(?)` degenerates into a full-table scan
    # across ~20K rows, which makes check_library_batch O(N*M) for an
    # incoming batch of M source rows.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_doi_lower ON papers(LOWER(doi))")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_title_nocase ON papers(title COLLATE NOCASE)")
    # source_url: schema v12 upgrades this to a partial UNIQUE index so two
    # concurrent store_web_capture calls cannot both insert the same URL.
    # Partial (WHERE source_url IS NOT NULL) so academic-paper rows with
    # NULL source_url don't collide. The non-unique version is dropped in
    # the v12 migration below.
    # Wrapped in try/except so a pre-existing duplicate (rare, but possible
    # if an older DB has them) doesn't abort _init_db. If the unique index
    # can't be created, we log and fall back to a non-unique index so
    # source_url lookups stay fast even without the uniqueness guarantee.
    try:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_source_url_unique "
            "ON papers(source_url) WHERE source_url IS NOT NULL"
        )
    except sqlite3.IntegrityError as e:
        print(
            f"[WARN] idx_papers_source_url_unique creation failed "
            f"(likely pre-existing duplicates): {e}. Falling back to "
            f"non-unique index. Run a dedup sweep on source_url and "
            f"restart to enable the unique guarantee.",
            file=sys.stderr,
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_papers_source_url "
            "ON papers(source_url)"
        )

    # Schema versioning for FTS5 rebuild. PRIMARY KEY added in v9 so two
    # concurrent startup racers can't leave duplicate version rows.
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")
    row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    current_version = row[0] if row else 0

    if current_version < 2:
        conn.execute("DROP TABLE IF EXISTS papers_fts")
        conn.execute("DROP TRIGGER IF EXISTS papers_ai")
        conn.execute("DROP TRIGGER IF EXISTS papers_ad")
        conn.execute("DROP TRIGGER IF EXISTS papers_au")
        _create_fts_and_triggers(conn)
        conn.execute("""
            INSERT INTO papers_fts(rowid, paper_id, title, abstract, tldr, processed_text)
            SELECT rowid, paper_id, title, abstract, tldr, processed_text FROM papers
        """)

    if current_version < 3:
        # Drop triggers so backfill doesn't fire them
        conn.execute("DROP TRIGGER IF EXISTS chunks_ai")
        conn.execute("DROP TRIGGER IF EXISTS chunks_ad")
        conn.execute("DROP TRIGGER IF EXISTS chunks_au")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                section_header TEXT,
                page_start INTEGER,
                page_end INTEGER
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_paper ON paper_chunks(paper_id)")
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS paper_chunks_fts USING fts5(
                chunk_text,
                content=paper_chunks,
                content_rowid=chunk_id
            )
        """)
        rows = conn.execute(
            "SELECT paper_id, processed_text FROM papers WHERE processed_text IS NOT NULL AND processed_text != ''"
        ).fetchall()
        for pid, text in rows:
            _chunk_processed_text(conn, pid, text)
        conn.execute("""
            INSERT INTO paper_chunks_fts(rowid, chunk_text)
            SELECT chunk_id, chunk_text FROM paper_chunks
        """)

    if current_version < 4:
        conn.execute("""
            UPDATE papers SET verified = 1
            WHERE local_pdf_path IS NOT NULL AND local_pdf_path != ''
        """)
        conn.execute("""
            UPDATE papers SET verified = 1
            WHERE processed_text IS NOT NULL AND processed_text != ''
        """)
        conn.execute("""
            UPDATE papers SET verified = 1
            WHERE abstract IS NOT NULL AND abstract != ''
                  AND authority_tier IS NOT NULL
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_verified ON papers(verified)")

    if current_version < 5:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_papers USING vec0(
                embedding float[768] distance_metric=cosine
            )
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                embedding float[768] distance_metric=cosine
            )
        """)

    if current_version < 6:
        # Citation-graph edge table. Populated by get_citations/get_references
        # MCP tools, _store_openalex_work, and the backfill_citations.py script.
        # Used at query time by search_local to compute hub-score RRF signal.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_references (
                citing_paper_id TEXT NOT NULL,
                cited_paper_id  TEXT NOT NULL,
                cited_doi       TEXT,
                source          TEXT NOT NULL,
                is_influential  INTEGER DEFAULT 0,
                first_seen      TEXT,
                PRIMARY KEY (citing_paper_id, cited_paper_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_refs_cited ON paper_references(cited_paper_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_refs_citing ON paper_references(citing_paper_id)")

    if current_version < 8:
        # Backfill has_full_text for existing rows. Cheap one-pass UPDATE.
        conn.execute("""
            UPDATE papers SET has_full_text = 1
            WHERE (LENGTH(COALESCE(processed_text,'')) > 500
                OR LENGTH(COALESCE(tex_text,'')) > 500)
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_has_full_text ON papers(has_full_text)")

    if current_version < 9:
        # Schema v9: give schema_version a PRIMARY KEY so concurrent startup
        # cannot leave duplicate version rows, and reap any stale claim locks
        # from prior runs. The column additions for _acquiring_by/_acquiring_at
        # are handled by the ALTER TABLE try/except block above.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_acquiring_by ON papers(_acquiring_by)")
        # Reap stale claims > 1 hour old, if any predate this migration.
        # Also drop any claim whose _acquiring_at is NOT in SQLite's native
        # 'YYYY-MM-DD HH:MM:SS' format. Earlier builds of _try_claim_paper
        # wrote Python ISO format with a 'T' separator and timezone offset
        # (`2026-04-17T10:00:00+00:00`) which compares lexically greater
        # than any same-date SQLite datetime, so those claims would never
        # expire via the < datetime('now','-1 hour') comparison. Reaping
        # them once at migration time clears the residue.
        conn.execute("""
            UPDATE papers
            SET _acquiring_by = NULL, _acquiring_at = NULL
            WHERE _acquiring_by IS NOT NULL
              AND (_acquiring_at IS NULL
                   OR _acquiring_at LIKE '%T%'
                   OR _acquiring_at < datetime('now', '-1 hour'))
        """)
        # Rebuild schema_version with a PK. Preserves any prior row; dedups
        # duplicates introduced by pre-v9 concurrent starts.
        current_vers = [
            r[0] for r in conn.execute("SELECT DISTINCT version FROM schema_version").fetchall()
        ]
        conn.execute("DROP TABLE schema_version")
        conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
        for v in current_vers:
            conn.execute("INSERT OR IGNORE INTO schema_version(version) VALUES (?)", (v,))

    if current_version < 10:
        # Schema v10: add tex_text column to papers_fts so TeX-only papers
        # surface in paper-level BM25. Prior schema only indexed processed_text;
        # papers with tex_text but no processed_text would miss the FTS leg
        # in search_local even though they had cite-ready body text.
        conn.execute("DROP TRIGGER IF EXISTS papers_ai")
        conn.execute("DROP TRIGGER IF EXISTS papers_ad")
        conn.execute("DROP TRIGGER IF EXISTS papers_au")
        conn.execute("DROP TABLE IF EXISTS papers_fts")
        _create_fts_and_triggers(conn)
        conn.execute("""
            INSERT INTO papers_fts(rowid, paper_id, title, abstract, tldr, processed_text, tex_text)
            SELECT rowid, paper_id, title, abstract, tldr, processed_text, tex_text FROM papers
        """)

    if current_version < 11:
        # Schema v11: rebuild papers_fts with the single-body policy. v10
        # indexed both processed_text AND tex_text for dual-source papers,
        # which double-counted term frequency in BM25 and gave those papers
        # an unfair lift on the FTS leg. v11 indexes exactly one body per
        # row (tex_text when >500 chars, else processed_text). The triggers
        # produced by _create_fts_and_triggers above already implement the
        # CASE WHEN policy; we just need to rebuild the index contents.
        conn.execute("DROP TRIGGER IF EXISTS papers_ai")
        conn.execute("DROP TRIGGER IF EXISTS papers_ad")
        conn.execute("DROP TRIGGER IF EXISTS papers_au")
        conn.execute("DROP TABLE IF EXISTS papers_fts")
        _create_fts_and_triggers(conn)
        conn.execute("""
            INSERT INTO papers_fts(rowid, paper_id, title, abstract, tldr, processed_text, tex_text)
            SELECT
                rowid, paper_id, title, abstract, tldr,
                CASE WHEN LENGTH(COALESCE(tex_text, '')) > 500 THEN '' ELSE COALESCE(processed_text, '') END,
                CASE WHEN LENGTH(COALESCE(tex_text, '')) > 500 THEN tex_text ELSE '' END
            FROM papers
        """)

    if current_version < 12:
        # Schema v12: drop the non-unique idx_papers_source_url so the
        # new partial-unique index is the sole source_url index. The
        # partial-unique index itself was already created above via
        # `CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_source_url_unique`.
        # Dropping a redundant index is idempotent; `IF EXISTS` guards
        # against re-running.
        conn.execute("DROP INDEX IF EXISTS idx_papers_source_url")

    if current_version < 13:
        # Schema v13: rebuild papers_fts with the concatenation policy to
        # restore recall on dual-source papers that v11 accidentally
        # truncated. For each row, index (tex_text || ' ' || processed_text)
        # into the processed_text column; leave the tex_text column empty.
        # All terms from either source are now matchable, and BM25's
        # per-column saturation prevents the double-count that v10
        # originally had.
        conn.execute("DROP TRIGGER IF EXISTS papers_ai")
        conn.execute("DROP TRIGGER IF EXISTS papers_ad")
        conn.execute("DROP TRIGGER IF EXISTS papers_au")
        conn.execute("DROP TABLE IF EXISTS papers_fts")
        _create_fts_and_triggers(conn)
        conn.execute("""
            INSERT INTO papers_fts(rowid, paper_id, title, abstract, tldr, processed_text, tex_text)
            SELECT
                rowid, paper_id, title, abstract, tldr,
                COALESCE(tex_text, '') || ' ' || COALESCE(processed_text, ''),
                ''
            FROM papers
        """)

    if current_version < 14:
        # Schema v14: restrict papers_fts to cite-ready rows only.
        # Previously every paper (including ~16K metadata-only rows with
        # empty body fields) was indexed, which polluted BM25's corpus
        # statistics (average-document-length term, IDF denominators) and
        # distorted ranking for full-text queries. v14 drops+rebuilds
        # papers_fts filtered to has_full_text=1, and the triggers
        # recreated by _create_fts_and_triggers above now gate every
        # INSERT/DELETE/UPDATE on the has_full_text flag via their
        # `WHERE new.has_full_text = 1` / `WHERE old.has_full_text = 1`
        # clauses.
        conn.execute("DROP TRIGGER IF EXISTS papers_ai")
        conn.execute("DROP TRIGGER IF EXISTS papers_ad")
        conn.execute("DROP TRIGGER IF EXISTS papers_au")
        conn.execute("DROP TABLE IF EXISTS papers_fts")
        _create_fts_and_triggers(conn)
        conn.execute("""
            INSERT INTO papers_fts(rowid, paper_id, title, abstract, tldr, processed_text, tex_text)
            SELECT
                rowid, paper_id, title, abstract, tldr,
                COALESCE(tex_text, '') || ' ' || COALESCE(processed_text, ''),
                ''
            FROM papers
            WHERE has_full_text = 1
        """)

    if current_version < 15:
        # Schema v15: orphan-prevention triggers. SQLite virtual tables
        # (vec_papers, vec_chunks) do not support FOREIGN KEY constraints,
        # so referential integrity for the vector tables has been entirely
        # manual. Every call site had to remember to DELETE from vec_papers
        # and vec_chunks before deleting papers/paper_chunks, and omissions
        # left orphan vector rows (62 + 1,549 observed on 2026-04-18).
        #
        # These triggers make the cleanup automatic. `papers_ad_*` fire on
        # AFTER DELETE ON papers; `chunks_ad_vec` fires on the cascade
        # delete into paper_chunks (or any direct delete). A single
        # `DELETE FROM papers WHERE paper_id = ?` now walks the full graph.
        #
        # Note: `papers_ad` (AFTER DELETE ON papers, for papers_fts cleanup)
        # already exists from v14 and is orthogonal; these triggers compose.
        conn.executescript("""
            CREATE TRIGGER IF NOT EXISTS papers_ad_chunks AFTER DELETE ON papers BEGIN
                DELETE FROM paper_chunks WHERE paper_id = old.paper_id;
            END;
            CREATE TRIGGER IF NOT EXISTS papers_ad_refs AFTER DELETE ON papers BEGIN
                DELETE FROM paper_references
                 WHERE citing_paper_id = old.paper_id
                    OR cited_paper_id  = old.paper_id;
            END;
            CREATE TRIGGER IF NOT EXISTS papers_ad_vec AFTER DELETE ON papers BEGIN
                DELETE FROM vec_papers WHERE rowid = old.rowid;
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad_vec AFTER DELETE ON paper_chunks BEGIN
                DELETE FROM vec_chunks WHERE rowid = old.chunk_id;
            END;
        """)

    if current_version < 16:
        # Schema v16: replace papers_ad_refs with a cited-side downgrade
        # so deleting a local paper preserves incoming citations as
        # `__doi:{doi}` placeholders, matching _store_references' policy
        # for not-yet-acquired cited works. v15's original version did a
        # blind DELETE of all edges where the deleted paper appeared on
        # either side, which dropped real citation data. The downgrade
        # path inserts a placeholder row for each existing citing→this
        # edge that has a DOI to downgrade to, then the DELETE sweeps the
        # old real-id edges (including the citing-side ones, which don't
        # round-trip: the deleted paper is the one ceasing to exist as
        # a citation source, not a citation target).
        conn.execute("DROP TRIGGER IF EXISTS papers_ad_refs")
        conn.executescript("""
            CREATE TRIGGER IF NOT EXISTS papers_ad_refs AFTER DELETE ON papers BEGIN
                INSERT OR IGNORE INTO paper_references
                    (citing_paper_id, cited_paper_id, cited_doi, source, is_influential, first_seen)
                SELECT
                    citing_paper_id,
                    '__doi:' || old.doi,
                    COALESCE(cited_doi, old.doi),
                    source,
                    is_influential,
                    first_seen
                FROM paper_references
                WHERE cited_paper_id = old.paper_id
                  AND old.doi IS NOT NULL
                  AND old.doi != '';

                DELETE FROM paper_references
                 WHERE citing_paper_id = old.paper_id
                    OR cited_paper_id  = old.paper_id;
            END;
        """)

    if current_version < 17:
        # Schema v17: papers_ad_refs downgrade uses the BEST DOI available:
        # edge-level `cited_doi` first, then the paper's own `old.doi`. v16
        # required `old.doi` to be present, which meant any paper row with
        # empty `papers.doi` (but whose incoming edges carried `cited_doi`
        # populated by `_store_references`) still dropped the citation data.
        # v17 fires the placeholder insert as long as EITHER source has a
        # DOI to downgrade to.
        conn.execute("DROP TRIGGER IF EXISTS papers_ad_refs")
        conn.executescript("""
            CREATE TRIGGER IF NOT EXISTS papers_ad_refs AFTER DELETE ON papers BEGIN
                INSERT OR IGNORE INTO paper_references
                    (citing_paper_id, cited_paper_id, cited_doi, source, is_influential, first_seen)
                SELECT
                    citing_paper_id,
                    '__doi:' || COALESCE(NULLIF(cited_doi, ''), old.doi),
                    COALESCE(NULLIF(cited_doi, ''), old.doi),
                    source,
                    is_influential,
                    first_seen
                FROM paper_references
                WHERE cited_paper_id = old.paper_id
                  AND (
                        (cited_doi IS NOT NULL AND cited_doi != '')
                     OR (old.doi   IS NOT NULL AND old.doi   != '')
                      );

                DELETE FROM paper_references
                 WHERE citing_paper_id = old.paper_id
                    OR cited_paper_id  = old.paper_id;
            END;
        """)

    if current_version < 18:
        # Schema v18: paper_passages table for page-bounded retrieval.
        # paper_chunks is 500-word-granular and spans page boundaries (64%
        # of current chunks straddle 2+ pages); it remains the unit for
        # semantic retrieval. paper_passages is single-page-granular,
        # derived by splitting processed_text on <!-- page N --> markers
        # inserted by Docling. One row per (paper_id, page_num). Used for
        # Bluebook-grade pincites where a multi-page range is not a valid
        # citation.
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS paper_passages (
                passage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL,
                page_num INTEGER NOT NULL,
                section_header TEXT,
                passage_text TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_passages_paper
                ON paper_passages(paper_id, page_num);
            CREATE INDEX IF NOT EXISTS idx_passages_paper_only
                ON paper_passages(paper_id);
            -- Cascade paper_passages on paper delete (mirror of v15 triggers
            -- for paper_chunks). Virtual tables don't support FOREIGN KEYs.
            CREATE TRIGGER IF NOT EXISTS papers_ad_passages AFTER DELETE ON papers BEGIN
                DELETE FROM paper_passages WHERE paper_id = old.paper_id;
            END;
        """)

    if current_version < 19:
        # Schema v19: has_gapped_extraction tri-state column on papers.
        # NULL = not audited, 0 = clean markers, 1 = no markers OR
        # descending markers (extraction failure signals). Plan v3 §3
        # Phase A.1. Backfilled by maintenance/backfill_has_gapped_extraction.py
        # (one-pass, no DB rewrites). Forward-fill happens at chunk time
        # via _classify_page_markers + _set_has_gapped_extraction.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(papers)").fetchall()}
        if "has_gapped_extraction" not in cols:
            conn.execute("ALTER TABLE papers ADD COLUMN has_gapped_extraction INTEGER")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_papers_has_gapped_extraction "
            "ON papers(has_gapped_extraction) WHERE has_gapped_extraction IS NOT NULL"
        )

    if current_version < 20:
        # Schema v20: Qwen3-Embedding-4B vec virtual tables for the
        # supported Matryoshka dims. Plan v3 §3 Phase 4R. Created
        # empty; populated by maintenance/backfill_embed_qwen3.py.
        # Lives ALONGSIDE vec_papers / vec_chunks (768d nomic) so
        # EMBED_VARIANT can A/B between backends without losing
        # baseline.
        for dim in _QWEN3_MLX_VALID_TARGET_DIMS:
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_papers_qwen3_{dim} "
                f"USING vec0(embedding float[{dim}] distance_metric=cosine)"
            )
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks_qwen3_{dim} "
                f"USING vec0(embedding float[{dim}] distance_metric=cosine)"
            )

    if current_version < 21:
        # Schema v21: cascade triggers for qwen3 vec tables (Wave-5
        # review CRITICAL fix). v15 added papers_ad_vec + chunks_ad_vec
        # for vec_papers/vec_chunks; the v20 qwen3 tables created in
        # the prior block had no cascade, so paper deletions and
        # _chunk_processed_text re-chunks left orphan rows in
        # vec_papers_qwen3_<dim> and vec_chunks_qwen3_<dim>. Adding
        # explicit triggers per dim. Idempotent via IF NOT EXISTS.
        for dim in _QWEN3_MLX_VALID_TARGET_DIMS:
            conn.executescript(
                f"""
                CREATE TRIGGER IF NOT EXISTS papers_ad_vec_qwen3_{dim}
                AFTER DELETE ON papers BEGIN
                    DELETE FROM vec_papers_qwen3_{dim} WHERE rowid = old.rowid;
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_ad_vec_qwen3_{dim}
                AFTER DELETE ON paper_chunks BEGIN
                    DELETE FROM vec_chunks_qwen3_{dim} WHERE rowid = old.chunk_id;
                END;
                """
            )

    if current_version < _SCHEMA_VERSION:
        # Round-2 review HIGH fix: was DELETE + INSERT OR REPLACE which
        # creates an empty-table window on crash. Single INSERT OR
        # REPLACE on the single-PK table is atomic and idempotent.
        conn.execute("INSERT OR REPLACE INTO schema_version(version) VALUES (?)", (_SCHEMA_VERSION,))

    _create_fts_and_triggers(conn)
    _create_chunks_fts_and_triggers(conn)
    conn.commit()


def _store_paper(conn: sqlite3.Connection, p: dict) -> None:
    paper_id = p.get("paperId", "")
    if not paper_id:
        return

    ext_ids = p.get("externalIds") or {}
    tldr = p.get("tldr") or {}
    oa_pdf = p.get("openAccessPdf") or {}
    authors = p.get("authors") or []
    now = datetime.now(timezone.utc).isoformat()

    conn.execute("""
        INSERT INTO papers (
            paper_id, doi, arxiv_id, pmid, title, authors, abstract, tldr,
            year, venue, citation_count, influential_citation_count,
            fields_of_study, publication_types, publication_date,
            is_open_access, oa_pdf_url, s2_url, verified, first_seen, last_updated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            title = COALESCE(NULLIF(title, ''), excluded.title, title),
            abstract = COALESCE(NULLIF(abstract, ''), excluded.abstract, abstract),
            authors = COALESCE(NULLIF(authors, '[]'), excluded.authors, authors),
            tldr = COALESCE(NULLIF(tldr, ''), excluded.tldr, tldr),
            citation_count = excluded.citation_count,
            influential_citation_count = excluded.influential_citation_count,
            is_open_access = excluded.is_open_access,
            oa_pdf_url = excluded.oa_pdf_url,
            last_updated = excluded.last_updated
    """, (
        paper_id,
        ext_ids.get("DOI", ""),
        ext_ids.get("ArXiv", ""),
        ext_ids.get("PubMed", ""),
        p.get("title", ""),
        json.dumps([a.get("name", "") for a in authors]),
        p.get("abstract", ""),
        tldr.get("text", ""),
        p.get("year"),
        p.get("venue", ""),
        p.get("citationCount", 0),
        p.get("influentialCitationCount", 0),
        json.dumps(p.get("fieldsOfStudy") or []),
        json.dumps(p.get("publicationTypes") or []),
        p.get("publicationDate", ""),
        1 if p.get("isOpenAccess") else 0,
        oa_pdf.get("url", ""),
        p.get("url", ""),
        0,
        now,
        now,
    ))


def _store_papers(papers: list[dict]) -> None:
    conn = _init_db()
    try:
        for p in papers:
            _store_paper(conn, p)
        conn.commit()
    finally:
        conn.close()


def _store_references(
    conn: sqlite3.Connection,
    citing_paper_id: str,
    references: list[dict],
    source: str,
) -> int:
    """Persist reference edges (citing_paper -> cited_paper) into paper_references.

    Each entry in `references` is a dict with optional keys:
      - paperId:       S2 paper ID of the cited work
      - doi:           DOI of the cited work
      - openalex_id:   OpenAlex work ID (full URL or short W...) of the cited work
      - is_influential: bool flag (from S2 citation context data)

    Resolves the cited paper against our local library in this order:
    S2 paperId → oa:{W_id} → DOI. If no local match, stores the edge with a
    placeholder cited_paper_id of `__doi:{doi}` so the information isn't lost;
    those edges are invisible to hub-scoring (which only matches on real IDs).

    Returns the number of edges inserted or ignored-as-duplicates.
    """
    if not references:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    for ref in references:
        cited_pid: str | None = None
        cited_doi = (ref.get("doi") or "").strip()
        s2_id = (ref.get("paperId") or "").strip()
        oa_id = (ref.get("openalex_id") or "").strip()

        # Resolve cited paper in our library
        if s2_id:
            row = conn.execute(
                "SELECT paper_id FROM papers WHERE paper_id = ?", (s2_id,)
            ).fetchone()
            if row:
                cited_pid = row[0]
        if not cited_pid and oa_id:
            short = oa_id.split("/")[-1]
            lookup = f"oa:{short}"
            row = conn.execute(
                "SELECT paper_id FROM papers WHERE paper_id = ?", (lookup,)
            ).fetchone()
            if row:
                cited_pid = row[0]
        if not cited_pid and cited_doi:
            row = conn.execute(
                "SELECT paper_id FROM papers WHERE lower(doi) = lower(?)", (cited_doi,)
            ).fetchone()
            if row:
                cited_pid = row[0]

        # Skip if we can't identify the edge at all
        if not cited_pid and not cited_doi:
            continue

        try:
            conn.execute("""
                INSERT OR IGNORE INTO paper_references
                (citing_paper_id, cited_paper_id, cited_doi, source, is_influential, first_seen)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                citing_paper_id,
                cited_pid or f"__doi:{cited_doi}",
                cited_doi or None,
                source,
                1 if ref.get("is_influential") else 0,
                now,
            ))
            count += 1
        except sqlite3.Error as e:
            print(f"  ref-edge insert error ({citing_paper_id} -> {cited_pid}): {e}", file=sys.stderr)
    return count


def _embed_paper_if_verified(
    conn: sqlite3.Connection, paper_id: str, *, raise_on_error: bool = False,
) -> None:
    """Generate and store paper-level embedding if the paper is verified.

    Default raise_on_error=False logs and returns on any failure (keeps
    opportunistic background re-embed paths from raising into their callers).
    Set raise_on_error=True when the caller is inside a savepoint that must
    roll back on embed failure (e.g., dedup_papers.merge_into).
    """
    row = conn.execute(
        "SELECT rowid, title, abstract, processed_text, tex_text, verified, has_full_text "
        "FROM papers WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    if not row:
        return
    rowid, title, abstract, processed_text, tex_text, verified, has_full_text = row
    # Embed when the row is either verified OR already cite-ready
    # (has_full_text = 1 means the text was stored via canonical helpers
    # that ran their own verification). Merges into a verified=0 keeper
    # that end up with has_full_text=1 were previously skipped here,
    # producing merge-survivors whose body text was indexed in FTS but
    # missing from vec_papers.
    if not (verified or has_full_text):
        return
    text = _paper_embed_text(title or "", abstract, processed_text, tex_text)
    if not text:
        return
    # Round-5 Codex review HIGH fix: previously the order was
    # encode → savepoint → delete → insert. If encoding raised AFTER
    # the caller had already committed a text/abstract mutation (e.g.,
    # set_abstract, _store_processed_text), the stale vec rows
    # remained attached to a now-changed source, polluting search.
    # New order: open savepoint → delete all variants → encode
    # (under savepoint) → insert active variant → release. If encode
    # fails, savepoint rolls back the DELETE, restoring the old (but
    # at-least consistent) state. The caller is responsible for not
    # committing the text mutation if downstream embedding fails.
    try:
        conn.execute("SAVEPOINT embed_paper")
    except Exception as e:
        print(f"Embedding savepoint error (paper {paper_id}): {e}", file=sys.stderr)
        if raise_on_error:
            raise
        return
    try:
        # Round-2 codex review CRITICAL fix: previously hardcoded
        # `vec_papers` (nomic) regardless of EMBED_VARIANT. Now route
        # through _vec_table_pair() so writes match the active variant.
        vec_papers_t, _ = _vec_table_pair()
        # Round-3 codex review HIGH fix: delete paper-vec rows from ALL
        # variants for this rowid before inserting the active variant.
        # Round-5 ordering fix: delete BEFORE encode so an encode
        # failure rolls back the delete via the savepoint.
        _delete_vec_papers_for_paper(conn, rowid)
        emb = _embed_texts([text])[0]
        conn.execute(f"INSERT INTO {vec_papers_t}(rowid, embedding) VALUES (?, ?)", (rowid, emb))
        conn.execute("RELEASE embed_paper")
    except Exception as e:
        try:
            conn.execute("ROLLBACK TO embed_paper")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("RELEASE embed_paper")
        except sqlite3.OperationalError:
            pass
        print(f"Embedding storage error (paper {paper_id}): {e}", file=sys.stderr)
        if raise_on_error:
            raise


def _embed_chunks(
    conn: sqlite3.Connection, paper_id: str, *, raise_on_error: bool = False,
) -> None:
    """Generate and store chunk-level embeddings for a paper's chunks.

    See _embed_paper_if_verified for the raise_on_error contract."""
    row = conn.execute(
        "SELECT verified, has_full_text FROM papers WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    if not row:
        return
    verified, has_full_text = row
    # Embed on verified OR has_full_text (see _embed_paper_if_verified for
    # the rationale: merge-keepers can end up has_full_text=1, verified=0
    # and previously had FTS coverage but no vec_chunks).
    if not (verified or has_full_text):
        return
    chunks = conn.execute(
        "SELECT chunk_id, chunk_text FROM paper_chunks WHERE paper_id = ? ORDER BY chunk_index",
        (paper_id,),
    ).fetchall()
    if not chunks:
        return
    chunk_ids = [c[0] for c in chunks]
    texts = [c[1] for c in chunks]
    # Round-5 Codex review HIGH fix: encode-after-delete ordering, same
    # as _embed_paper_if_verified. Delete inside the savepoint, encode
    # under the savepoint; encode failure rolls back the delete.
    try:
        conn.execute("SAVEPOINT embed_chunks")
    except Exception as e:
        print(f"Chunk embedding savepoint error (paper {paper_id}): {e}", file=sys.stderr)
        if raise_on_error:
            raise
        return
    try:
        # Round-2 codex review CRITICAL fix: same vec_chunks routing
        # bug as _embed_paper_if_verified.
        _, vec_chunks_t = _vec_table_pair()
        placeholders = ",".join("?" * len(chunk_ids))
        conn.execute(f"DELETE FROM {vec_chunks_t} WHERE rowid IN ({placeholders})", chunk_ids)
        embeddings = _embed_texts(texts)
        for chunk_id, emb in zip(chunk_ids, embeddings):
            conn.execute(f"INSERT INTO {vec_chunks_t}(rowid, embedding) VALUES (?, ?)", (chunk_id, emb))
        conn.execute("RELEASE embed_chunks")
    except Exception as e:
        try:
            conn.execute("ROLLBACK TO embed_chunks")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("RELEASE embed_chunks")
        except sqlite3.OperationalError:
            pass
        print(f"Chunk embedding storage error (paper {paper_id}): {e}", file=sys.stderr)
        if raise_on_error:
            raise


_STORED_PID_PAT = re.compile(r"^Stored as:\s*(\S.*?)\s*$", re.MULTILINE)


def _parse_stored_pid(process_pdf_result: str) -> str | None:
    """Extract the paper_id that process_pdf actually stored text under.

    process_pdf's success output includes a line `Stored as: {pid}`. Callers need
    this because process_pdf may resolve a different pid than the caller's hint
    (e.g., when paper_id="" is passed, process_pdf's _resolve_paper_id picks one
    based on DOI-from-filename). Verification + cleanup must operate on the
    actual stored pid, not the hint.
    """
    if not process_pdf_result or not process_pdf_result.startswith("Processed:"):
        return None
    m = _STORED_PID_PAT.search(process_pdf_result)
    return m.group(1) if m else None


_CHARS_PAT = re.compile(r"Extracted:\s*([\d,]+)\s*chars")


def _parse_chars(process_pdf_result: str) -> int:
    """Parse the `Extracted: N chars` count from process_pdf's success output."""
    if not process_pdf_result:
        return 0
    m = _CHARS_PAT.search(process_pdf_result)
    return int(m.group(1).replace(",", "")) if m else 0


_VERIFY_STOPWORDS = {
    "about","above","after","again","against","among","around","because","before",
    "being","below","between","both","could","doing","during","each","from","further",
    "having","here","into","itself","more","most","myself","once","only","other",
    "ourselves","over","same","should","some","such","than","that","their","them",
    "themselves","then","there","these","they","this","those","through","under",
    "until","upon","very","were","what","when","where","which","while","with",
    "within","would","your","yours","yourself","also","many","much","like","ones",
    "make","just","know","take","time","year","work","using","used","based","another",
    "towards","across","including","toward",
}


def _distinctive_title_words(title: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z]+", (title or "").lower())
    return [t for t in tokens if len(t) >= 5 and t not in _VERIFY_STOPWORDS]


_VALID_VERIFY_SOURCES = ("processed_text", "tex_text")


import unicodedata as _unicodedata


# Common surname-collision tokens that must NOT count as a verifier hit.
# Round-1 review (opus HIGH): the previous list overreached — Brown / Young /
# Page / May / Best / Long / Short / White / Black / Green are all legit
# academic surnames. Removed those. Kept only:
#   - 2-char Pinyin/Asian surnames (Li / Wu / Yu / An / Bo / Go / Ma / Ng)
#     — overwhelming collision rate in any English text
#   - Filler words (the / and / for) — never real surnames
_AUTHOR_STOPWORDS = {"li", "wu", "yu", "an", "bo", "go", "ma", "ng",
                     "the", "and", "for"}


def _strip_diacritics(s: str) -> str:
    """NFKD normalize and drop combining marks. 'Lehuedé' → 'lehuede'."""
    nkfd = _unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in nkfd if not _unicodedata.combining(c))


def _strip_trailing_unbalanced_paren(s: str) -> str:
    """Remove text from the LAST unbalanced opening paren to the end.

    Round-3 MED fix (codex 1, opus): the previous regex `\\s*\\([^)]*$`
    couldn't cross balanced inner parens. For "John Smith (Dr. (Jr.) Senior"
    that regex saw `(Jr.)` (balanced) and stopped, leaving "Senior" as the
    last token. This helper scans right-to-left counting parens, finds the
    last `(` whose closing `)` is missing, and truncates there.

    Examples:
      "John Smith (MIT" → "John Smith"
      "John Smith (Dr. (Jr.) Senior" → "John Smith"
      "John Smith" → "John Smith"  (no parens)
      "John Smith (MIT)" → "John Smith (MIT)"  (balanced; let other regex strip)
    """
    if not s or "(" not in s:
        return s
    depth = 0
    cut = len(s)
    for i in range(len(s) - 1, -1, -1):
        c = s[i]
        if c == ")":
            depth += 1
        elif c == "(":
            if depth == 0:
                cut = i
                break
            depth -= 1
    return s[:cut].rstrip()


def _author_surnames(authors_field: Any) -> list[str]:
    """Extract distinctive surnames from a paper's authors field.

    Accepts:
      - JSON string (the format papers.authors is stored in: '["A","B"]')
      - list[str] of full names: ["John Smith", "Jane Doe"]
      - list[dict] from S2/OpenAlex shapes: [{"name":"X"}, {"display_name":"Y"},
        {"author":{"display_name":"Z"}}]
      - single str (full name): "John Smith"

    Returns lowercased + diacritic-stripped surnames suitable for whole-word
    matching against a paper's text head (also diacritic-stripped — see
    _check_authors).

    Filters: surnames must be >=3 chars and not in _AUTHOR_STOPWORDS. Strips
    "Jr"/"Sr"/Roman-numeral suffixes (with or without comma) and trailing
    parentheticals like "John Smith (MIT)". Drops empty fragments.
    """
    if not authors_field:
        return []

    def _name_from_dict(d: dict) -> str | None:
        # Round-3 MED fix (codex 1): check nested `author` BEFORE outer
        # generic `name` — a wrapper-shape dict like
        # {"position":1,"author":{"display_name":"X"},"name":"University Y"}
        # should pick "X" not "University Y". Most real shapes have either
        # nested `author` OR top-level `name`, not both.
        a = d.get("author")
        if isinstance(a, dict):
            nested = _name_from_dict(a)
            if nested:
                return nested
        # Round-2 MED fix (codex 5): prefer family-name keys (Crossref's
        # `family`, OpenAlex `family_name`, etc.) over generic `name` —
        # otherwise `{"given_name":"John","family_name":"Smith"}` would
        # extract "john" as the surname.
        for k in ("family", "family_name", "last_name", "surname"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                # Return as a fake "First Last" so downstream surname
                # extraction picks this token.
                return f"_ {v.strip()}"
        for k in ("display_name", "name", "fullName", "full_name"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return None

    raw_list: list[str] = []
    json_parsed_successfully = False  # round-2 MED fix (codex 1)
    if isinstance(authors_field, list):
        for item in authors_field:
            if isinstance(item, str):
                raw_list.append(item)
            elif isinstance(item, dict):
                n = _name_from_dict(item)
                if n:
                    raw_list.append(n)
    elif isinstance(authors_field, dict):
        n = _name_from_dict(authors_field)
        if n:
            raw_list.append(n)
    elif isinstance(authors_field, str):
        s = authors_field.strip()
        # Round-2 MED fix (opus): also handle dict-shaped JSON strings
        # like '{"name": null}' or '{"family":"Smith"}'. Previously these
        # fell through to single-string treatment and produced bogus
        # surnames like "null".
        looks_like_json = (s.startswith("[") and s.endswith("]")) or (
            s.startswith("{") and s.endswith("}")
        )
        if looks_like_json:
            try:
                parsed = json.loads(s)
                json_parsed_successfully = True
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, str):
                            raw_list.append(item)
                        elif isinstance(item, dict):
                            n = _name_from_dict(item)
                            if n:
                                raw_list.append(n)
                elif isinstance(parsed, dict):
                    n = _name_from_dict(parsed)
                    if n:
                        raw_list.append(n)
            except (json.JSONDecodeError, ValueError):
                # Round-2 LOW fix (opus): even if string LOOKS bracket-shaped
                # but isn't valid JSON (e.g. '[abc]'), do NOT fall back to
                # treating as single name — that extracts bogus surnames.
                # Mark as JSON-attempted; the !json_parsed_successfully
                # check below will return [].
                json_parsed_successfully = True
        # Only fall back to single-string interpretation when the input was
        # NOT JSON-shaped at all.
        if not raw_list and not json_parsed_successfully:
            raw_list = [s]

    out: list[str] = []
    for full_name in raw_list:
        name = full_name.strip()
        if not name:
            continue
        # Round-2 MED fix (codex 1, 5): handle interleaved parens + suffixes
        # ("John Smith (MIT) Jr." or "John Smith Jr. (PhD)") by looping
        # both strips until stable.
        # Round-3 MED fix (codex 1): _strip_trailing_unbalanced_paren scans
        # right-to-left counting parens, so it correctly handles nested
        # cases like "John Smith (Dr. (Jr.) Senior" → "John Smith".
        prev_name = None
        while name != prev_name:
            prev_name = name
            # Strip BALANCED trailing parentheticals.
            name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
            # Strip from the LAST UNBALANCED `(` to end (handles nesting).
            name = _strip_trailing_unbalanced_paren(name)
            # Strip Jr/Sr/Roman numeral suffixes (with or without comma).
            name = re.sub(r"[,\s]+(jr\.?|sr\.?|ii|iii|iv|v|vi|esq\.?)$", "", name, flags=re.IGNORECASE).strip()
        if "," in name:
            surname = name.split(",", 1)[0].strip()
        else:
            tokens = name.split()
            surname = tokens[-1] if tokens else ""
        # Diacritic-strip THEN remove non-letters.
        surname = _strip_diacritics(surname)
        surname = re.sub(r"[^A-Za-z]", "", surname).lower()
        if len(surname) < 3:
            continue
        if surname in _AUTHOR_STOPWORDS:
            continue
        out.append(surname)
    return out


def _verify_paper_text_match(
    conn: sqlite3.Connection,
    paper_id: str,
    *,
    target_doi: str | None = None,
    target_title: str | None = None,
    target_arxiv_id: str | None = None,
    target_authors: Any = None,
    text_source: str = "processed_text",
    text_head_chars: int = 50000,
    min_distinctive_matches: int = 2,
    min_distinctive_fraction: float = 0.3,
) -> tuple[bool, str]:
    """Verify that a paper's stored text matches expected identifiers.

    Called after an acquisition path stores text for `paper_id`. If verification
    fails, caller should `_clear_paper_text(paper_id, ...)` with the appropriate
    clear_* kwargs for the source being verified.

    Semantics:
      - DOI is not a gate. Many paper PDFs are extracted without the
        DOI string anywhere in body text (DOI lives in metadata only, or in
        a header that Docling drops). Title + author surname are sufficient
        evidence the right paper was acquired. target_doi is still accepted
        for forward-compat but is logged advisory-only and never causes
        verify-fail.
      - Required gates (when provided): target_arxiv_id, target_title,
        target_authors. ALL provided targets must pass (AND).
      - If NO required-gate targets provided, returns False (unverifiable).
      - text_source selects which column is checked: "processed_text" (default,
        used by PDF acquisition paths) or "tex_text" (used by TeX acquisition
        paths like process_tex / download_arxiv_tex).
      - arXiv id check: arxiv_id must appear verbatim in the first
        `text_head_chars`. Also accepts whitespace-normalized and a loose form
        that tolerates optional `arXiv:` prefix.
      - Title check: at least `min_distinctive_matches` distinctive (>=5 chars,
        non-stopword) title words must appear AND at least
        `min_distinctive_fraction` of all distinctive words must match. If
        title has fewer than 2 distinctive words, fall back to normalized
        exact-phrase containment (whitespace/punctuation squashed on both
        sides so "AlphaGo" matches "Alpha Go").
      - Author check: at least one author's surname must appear as a
        whole-word match in the head. Accepts a JSON string, list[str],
        or single str. See _author_surnames() for parsing/filtering rules.
    """
    if text_source not in _VALID_VERIFY_SOURCES:
        return False, f"bad text_source: {text_source!r}"
    row = conn.execute(
        f"SELECT {text_source} FROM papers WHERE paper_id = ?", (paper_id,)
    ).fetchone()
    if not row or not row[0]:
        return False, f"no {text_source} for {paper_id}"
    head = row[0][:text_head_chars].lower()
    normalized_head = re.sub(r"\s+", "", head)

    def _doi_advisory() -> str:
        """DOI is not a verification gate. Returns a one-line advisory
        describing whether the DOI was found, for log/debug only."""
        doi_lower = target_doi.lower().strip()
        if doi_lower and doi_lower in head:
            return f"doi-advisory: present ({target_doi})"
        normalized_doi = re.sub(r"\s+", "", doi_lower)
        if normalized_doi and normalized_doi in normalized_head:
            return f"doi-advisory: present whitespace-normalized ({target_doi})"
        return f"doi-advisory: absent from first {text_head_chars} chars ({target_doi})"

    def _check_arxiv_id() -> tuple[bool, str]:
        aid = target_arxiv_id.lower().strip()
        # Direct substring (e.g. "2401.12345") or with the "arxiv:" prefix.
        if aid and aid in head:
            return True, f"arxiv-match: {target_arxiv_id}"
        if f"arxiv:{aid}" in head:
            return True, f"arxiv-match (arXiv: prefix): {target_arxiv_id}"
        # Some older arXiv ids have slashes (hep-th/0401234); also try w/o slashes.
        stripped = aid.replace("/", "")
        if stripped and stripped in re.sub(r"[\s/]+", "", head):
            return True, f"arxiv-match (slash-normalized): {target_arxiv_id}"
        return False, f"arxiv-missing: {target_arxiv_id} not in first {text_head_chars} chars"

    def _check_title() -> tuple[bool, str]:
        distinctive = _distinctive_title_words(target_title)
        if len(distinctive) >= 2:
            matched = [w for w in distinctive if w in head]
            frac = len(matched) / len(distinctive)
            if len(matched) >= min_distinctive_matches and frac >= min_distinctive_fraction:
                return True, f"title-match: {len(matched)}/{len(distinctive)} distinctive words ({frac:.0%})"
            title_phrase = target_title.lower().strip()
            if title_phrase and title_phrase in head:
                return True, "title-match: exact phrase"
            return False, f"title-mismatch: {len(matched)}/{len(distinctive)} distinctive words ({frac:.0%})"
        # Fewer than 2 distinctive words: verify via normalized exact-phrase
        # match (whitespace+punctuation squashed so "AlphaGo" matches
        # "Alpha Go" and vice versa).
        title_phrase = target_title.lower().strip()
        if title_phrase and title_phrase in head:
            return True, "title-match: exact phrase"
        normalized_title = re.sub(r"[\s\W_]+", "", title_phrase)
        if normalized_title and normalized_title in re.sub(r"[\s\W_]+", "", head):
            return True, "title-match: normalized phrase"
        return False, f"title-unverifiable: only {len(distinctive)} distinctive words and no phrase match"

    # Diacritic-stripped lowercase head for accent-insensitive author matching
    # (paper text often preserves diacritics; surnames in metadata may be
    # stored in either form). Computed lazily to avoid extra work when no
    # author target is provided.
    _head_no_accents_cache: dict[str, str] = {}
    def _head_no_accents() -> str:
        if "v" not in _head_no_accents_cache:
            _head_no_accents_cache["v"] = _strip_diacritics(head)
        return _head_no_accents_cache["v"]

    def _check_authors(surnames: list[str]) -> tuple[bool, str]:
        # Caller guarantees surnames is non-empty (call site filters).
        h = _head_no_accents()
        matched: list[str] = []
        for s in surnames:
            if re.search(rf"\b{re.escape(s)}\b", h):
                matched.append(s)
        if matched:
            return True, f"author-match: {len(matched)}/{len(surnames)} surname(s): {','.join(matched[:3])}"
        return False, f"author-mismatch: 0/{len(surnames)} surnames in first {text_head_chars} chars (tried: {','.join(surnames[:5])})"

    checks: list[tuple[bool, str]] = []
    advisory_msgs: list[str] = []
    if target_doi:
        # No longer gates. Log advisory only.
        advisory_msgs.append(_doi_advisory())
    if target_arxiv_id:
        checks.append(_check_arxiv_id())
    if target_title:
        checks.append(_check_title())
    # Round-1 review HIGH (codex 1, 3, 5, opus): only ADD the author check
    # to the AND-gate when there's actually something to check. Empty
    # surnames (e.g., target_authors='[]', dict-shape that yields no names,
    # all-filtered) MUST NOT silently pass — that would degrade the gate
    # to title-only without telling the caller.
    if target_authors:
        author_surnames = _author_surnames(target_authors)
        if author_surnames:
            checks.append(_check_authors(author_surnames))
        else:
            advisory_msgs.append(
                f"author-unverifiable: no usable surnames in {target_authors!r}"[:120]
            )
    if not checks:
        return False, "no usable target_arxiv_id, target_title, or target_authors provided (target_doi alone is no longer a gate)"
    fails = [reason for ok, reason in checks if not ok]
    advisory_suffix = f" [{'; '.join(advisory_msgs)}]" if advisory_msgs else ""
    if fails:
        passes = [reason for ok, reason in checks if ok]
        return False, f"verify-fail: {' | '.join(fails)}" + (f" (passed: {'; '.join(passes)})" if passes else "") + advisory_suffix
    return True, "; ".join(reason for _, reason in checks) + advisory_suffix


def _recompute_has_full_text(conn: sqlite3.Connection, paper_id: str) -> int:
    """Recompute has_full_text from the CURRENT state of processed_text + tex_text.

    The flag is 1 iff at least one of processed_text or tex_text exceeds 500 chars.
    Called by every mutation path (store/clear/merge) so the flag never drifts
    out of sync with the actual text fields. Returns the new has_full_text value.

    WHERE clause includes `has_full_text != CASE ...` so no-op recomputes don't
    re-fire the papers_au trigger (papers_fts rebuild) on every store/clear
    call, saving WAL churn on the hot ingestion path.
    """
    conn.execute(
        """
        UPDATE papers
        SET has_full_text = CASE
            WHEN LENGTH(COALESCE(processed_text, '')) > 500
              OR LENGTH(COALESCE(tex_text, '')) > 500
            THEN 1 ELSE 0 END
        WHERE paper_id = ?
          AND has_full_text != CASE
              WHEN LENGTH(COALESCE(processed_text, '')) > 500
                OR LENGTH(COALESCE(tex_text, '')) > 500
              THEN 1 ELSE 0 END
        """,
        (paper_id,),
    )
    row = conn.execute(
        "SELECT has_full_text FROM papers WHERE paper_id = ?", (paper_id,)
    ).fetchone()
    return int(row[0]) if row else 0


def _try_claim_paper(conn: sqlite3.Connection, paper_id: str, worker_tag: str) -> bool:
    """Attempt to claim exclusive acquisition rights on a paper row.

    Returns True iff this worker now owns the claim. Idempotent: re-claiming
    the same paper_id with the same worker_tag returns True. Stale claims
    older than 1 hour are swept so a dead worker cannot permanently block
    other workers.

    Implementation uses a single UPDATE with a WHERE clause that atomically
    checks "currently free OR mine OR stale" — SQLite guarantees row-level
    atomicity for UPDATE within a write transaction, so two concurrent
    _try_claim_paper calls cannot both succeed.

    _acquiring_at is stored via SQLite's `datetime('now')` so it uses the
    same 'YYYY-MM-DD HH:MM:SS' text format as the expiry comparison. A
    previous implementation wrote `datetime.now(timezone.utc).isoformat()`
    which produced `'2026-04-17T10:00:00+00:00'`; that format compares as
    text less-than-or-equal to any `'YYYY-MM-DD HH:MM:SS'` that shares the
    same date prefix, so stale claims failed to expire for hours.
    """
    # Claims are reclaimable when the slot is free (_acquiring_by IS NULL),
    # held by the same worker (idempotent re-claim), or older than 1 hour
    # (dead-worker recovery). Legacy ISO-format claims from pre-fix workers
    # are swept ONCE during the v9 migration — not here, because a still-
    # alive pre-fix worker could otherwise have its legitimate claim
    # stolen immediately by a post-fix worker during upgrade.
    cur = conn.execute(
        """
        UPDATE papers
        SET _acquiring_by = ?, _acquiring_at = datetime('now')
        WHERE paper_id = ?
          AND (_acquiring_by IS NULL
               OR _acquiring_by = ?
               OR _acquiring_at IS NULL
               OR _acquiring_at < datetime('now', '-1 hour'))
        """,
        (worker_tag, paper_id, worker_tag),
    )
    conn.commit()
    return cur.rowcount > 0


def _worker_embed_paper(paper_id: str) -> str | None:
    """Embed a single paper's paper-level vector + chunk vectors, owning
    our own SQLite connection end-to-end.

    Called via asyncio.to_thread from async handlers that have committed
    the row's text/metadata change already. The outer handler closes its
    own conn BEFORE awaiting this, so there is no risk of the outer
    finally racing conn.close() against the worker thread still using
    the same connection.

    Returns None on success; returns a short error string on failure so
    callers can report "embedding deferred" rather than silently claiming
    re-embedded. Vectors remain recoverable via backfill regardless.
    """
    try:
        conn = _init_db()
    except Exception as e:
        print(f"_worker_embed_paper init failed for {paper_id}: {e}", file=sys.stderr)
        return f"db init failed: {type(e).__name__}"
    try:
        try:
            _embed_paper_if_verified(conn, paper_id, raise_on_error=True)
            _embed_chunks(conn, paper_id, raise_on_error=True)
            conn.commit()
            return None
        except Exception as e:
            print(f"_worker_embed_paper failed for {paper_id}: {e}", file=sys.stderr)
            return f"{type(e).__name__}: {e}"
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _release_paper_claim(conn: sqlite3.Connection, paper_id: str, worker_tag: str) -> None:
    """Release an acquisition claim held by `worker_tag` on `paper_id`.

    A mismatch (different worker holds the claim now) is silently ignored so
    cleanup never raises; the stale-sweeper in _try_claim_paper handles the
    rare case where a claim survives a crashed worker.
    """
    conn.execute(
        """
        UPDATE papers
        SET _acquiring_by = NULL, _acquiring_at = NULL
        WHERE paper_id = ?
          AND _acquiring_by = ?
        """,
        (paper_id, worker_tag),
    )
    conn.commit()


def _refresh_paper_claim(conn: sqlite3.Connection, paper_id: str, worker_tag: str) -> bool:
    """Bump _acquiring_at on a claim this worker still holds.

    Returns True iff the refresh updated a row (i.e. the worker still owns
    the claim). Returns False if another worker stole the claim (stale >1h)
    or the row is gone. Called between long-running acquisition fallbacks
    so a single paper taking >1h across OA → arXiv → direct URL
    (each bounded by a 30-min Docling timeout) doesn't let the claim
    lapse mid-acquisition.
    """
    cur = conn.execute(
        """
        UPDATE papers
        SET _acquiring_at = datetime('now')
        WHERE paper_id = ?
          AND _acquiring_by = ?
        """,
        (paper_id, worker_tag),
    )
    conn.commit()
    return cur.rowcount > 0


def _clear_paper_text(
    conn: sqlite3.Connection,
    paper_id: str,
    *,
    clear_processed_text: bool = True,
    clear_tex_text: bool = True,
    clear_pdf_path: bool = False,
    clear_tex_path: bool = False,
) -> None:
    """Wipe a paper's text + chunks + embeddings atomically.

    Single source of truth for "clear a paper's content but keep the metadata row".
    Wraps every delete + update in a savepoint so partial failure rolls back cleanly.

    Parameters
    ----------
    clear_processed_text : default True. When False, leaves processed_text and
        the chunks/embeddings derived from it alone. Used by paths that only
        want to wipe TeX-sourced text (and vice versa).
    clear_tex_text : default True. When False, leaves tex_text alone.
    clear_pdf_path : default False. If True, also nulls local_pdf_path. Only
        set this when the PDF was downloaded by an acquisition path that failed
        verification, because nulling a legitimate PDF path destroys the
        recovery path.
    clear_tex_path : default False. Same for local_tex_path.

    has_full_text is always recomputed from the resulting state of both text
    fields via _recompute_has_full_text, so selective clears leave a consistent
    flag.

    Chunks/paper-vector behavior:
    - Chunks are a single pool regardless of source, so clearing one text
      source requires rebuilding chunks from whatever text SURVIVES (via
      _chunk_processed_text + _embed_*). If no text survives, chunks are
      wiped and the paper becomes metadata-only.
    - When BOTH text sources are preserved (clear_*_text both False), we
      leave the existing chunks/vectors alone — the caller is only nulling
      a path column.
    """
    try:
        conn.execute("SAVEPOINT clear_paper")
    except Exception as e:
        # Propagate: callers rely on this helper to actually clear. Silently
        # returning left wrong text in place while the caller reported
        # "cleared".
        print(f"clear_paper savepoint error ({paper_id}): {e}", file=sys.stderr)
        raise
    try:
        # Decide whether chunks need to be wiped. If neither text source is
        # being cleared, nothing changes about the chunk-source material, so
        # the existing chunks/vectors stay valid. Otherwise we must rebuild
        # from whatever text survives (see re-chunk block below).
        text_changing = clear_processed_text or clear_tex_text
        if text_changing:
            # Round-2 review HIGH fix: previously only cleared vec_chunks
            # (nomic), not the qwen3 variants. Use the centralized helper
            # so all 4 vec-chunk tables are cleared consistently.
            _delete_vec_chunks_for_paper(conn, paper_id)
            conn.execute("DELETE FROM paper_chunks WHERE paper_id = ?", (paper_id,))
            # Also drop the page-bounded passages built from the (now
            # rejected) text. Without this, passage-mode search and
            # verify_claim can still surface evidence from text that was
            # explicitly cleared as wrong. Observed 2026-04-18: 4 live
            # papers had stale paper_passages after failed verification.
            conn.execute("DELETE FROM paper_passages WHERE paper_id = ?", (paper_id,))
            row = conn.execute(
                "SELECT rowid FROM papers WHERE paper_id = ?", (paper_id,)
            ).fetchone()
            if row:
                # Round-2 review HIGH fix: also clear vec_papers_qwen3_*
                # variants; previously only nomic.
                _delete_vec_papers_for_paper(conn, row[0])
        updates: list[str] = []
        if clear_processed_text:
            updates.append("processed_text = NULL")
        if clear_tex_text:
            updates.append("tex_text = NULL")
        if clear_pdf_path:
            updates.append("local_pdf_path = NULL")
        if clear_tex_path:
            updates.append("local_tex_path = NULL")
        if updates:
            conn.execute(
                f"UPDATE papers SET {', '.join(updates)} WHERE paper_id = ?",
                (paper_id,),
            )
        # Recompute has_full_text from the resulting text-column state so the
        # flag stays consistent after selective clears and the chunks/vectors
        # wipe above.
        _recompute_has_full_text(conn, paper_id)
        # If at least one text source was cleared, rebuild chunks + embeddings
        # from the surviving text (tex_text preferred, else processed_text).
        # This avoids the "has_full_text=1 but no chunks" state that made
        # search_passages blind to papers after a selective processed_text
        # clear where tex_text survived.
        # raise_on_error=True so an embed failure propagates up to the
        # outer except block, triggers ROLLBACK TO clear_paper, and the
        # caller sees the failure instead of a half-rebuilt row.
        if text_changing:
            survivor = conn.execute(
                "SELECT processed_text, tex_text FROM papers WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
            if survivor:
                surv_pt, surv_tt = survivor
                rebuild_text = None
                if surv_tt and len(surv_tt) > 500:
                    rebuild_text = surv_tt
                elif surv_pt and len(surv_pt) > 500:
                    rebuild_text = surv_pt
                if rebuild_text is not None:
                    _chunk_processed_text(conn, paper_id, rebuild_text)
                    # Rebuild page-bounded passages from the surviving
                    # text too. Matches the _store_processed_text wiring;
                    # without this the passages would be missing for the
                    # rebuild case (tex-source survivor can't produce
                    # passages so this is a no-op for tex-only rebuilds,
                    # which is correct).
                    _build_paper_passages(conn, paper_id, rebuild_text)
                    _embed_paper_if_verified(conn, paper_id, raise_on_error=True)
                    _embed_chunks(conn, paper_id, raise_on_error=True)
        conn.execute("RELEASE clear_paper")
    except Exception as e:
        try:
            conn.execute("ROLLBACK TO clear_paper")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("RELEASE clear_paper")
        except sqlite3.OperationalError:
            pass
        # Re-raise: a partial or failed clear must not be silently reported as
        # success. Callers currently treat absence of exception as "cleared".
        print(f"clear_paper error ({paper_id}): {e}", file=sys.stderr)
        raise


# --- OpenAlex helpers ---

def _compute_authority_tier(
    fwci: float | None,
    citation_percentile: float | None,
    venue_impact_factor: float | None,
    venue_h_index: int | None,
    work_type: str,
    citation_count: int,
    is_retracted: bool,
) -> int:
    """Compute authority tier (1-4) from OpenAlex metrics.
    Returns 0 if insufficient data to classify."""
    if is_retracted:
        return 4

    score = 0

    # FWCI: field-weighted citation impact (1.0 = average for field/year)
    if fwci is not None:
        if fwci >= 3.0:
            score += 3
        elif fwci >= 1.5:
            score += 2
        elif fwci >= 0.8:
            score += 1

    # Citation percentile (0-100, higher = more cited than peers)
    if citation_percentile is not None:
        if citation_percentile >= 95:
            score += 3
        elif citation_percentile >= 80:
            score += 2
        elif citation_percentile >= 50:
            score += 1

    # Venue quality
    if venue_impact_factor is not None:
        if venue_impact_factor >= 10:
            score += 3
        elif venue_impact_factor >= 3:
            score += 2
        elif venue_impact_factor >= 1:
            score += 1

    if venue_h_index is not None:
        if venue_h_index >= 200:
            score += 2
        elif venue_h_index >= 50:
            score += 1

    # Work type bonus
    if work_type in ("review", "meta-analysis"):
        score += 1

    # Raw citation count as fallback when FWCI/percentile unavailable
    if fwci is None and citation_percentile is None:
        if citation_count >= 100:
            score += 3
        elif citation_count >= 30:
            score += 2
        elif citation_count >= 10:
            score += 1

    # Map score to tier
    if score >= 7:
        return 1
    elif score >= 4:
        return 2
    elif score >= 2:
        return 3
    else:
        return 4


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


def _store_openalex_work(conn: sqlite3.Connection, work: dict) -> str | None:
    """Store an OpenAlex work in the papers table. Returns the paper_id used."""
    oa_id = work.get("id", "")
    doi_raw = work.get("doi", "") or ""
    doi = doi_raw.replace("https://doi.org/", "") if doi_raw else ""

    title = work.get("display_name") or work.get("title", "")
    if not title:
        return None

    # Try to match existing paper by DOI (case-insensitive to match the rest
    # of the edge-resolution and lookup code — DOIs are case-insensitive per
    # the DOI Handbook, and S2/OpenAlex return them in different cases).
    paper_id = None
    if doi:
        row = conn.execute(
            "SELECT paper_id FROM papers WHERE lower(doi) = lower(?) LIMIT 1",
            (doi,),
        ).fetchone()
        if row:
            paper_id = row[0]

    if not paper_id:
        if oa_id:
            paper_id = f"oa:{oa_id.split('/')[-1]}"
        else:
            title_hash = hashlib.md5(title.encode()).hexdigest()[:12]
            paper_id = f"oa:{title_hash}"

    # Extract venue info from primary_location
    primary_loc = work.get("primary_location") or {}
    source = primary_loc.get("source") or {}
    venue_name = source.get("display_name", "")
    venue_type = source.get("type", "")

    # Extract authority metrics
    fwci = work.get("fwci")
    cite_norm = work.get("citation_normalized_percentile") or {}
    citation_percentile_raw = cite_norm.get("value")
    # OpenAlex returns 0-1, tier function expects 0-100
    citation_percentile = citation_percentile_raw * 100 if citation_percentile_raw is not None else None
    cited_by = work.get("cited_by_count", 0)
    is_retracted = work.get("is_retracted", False)
    work_type = work.get("type", "")

    # Get venue quality from source summary_stats if available
    summary = source.get("summary_stats") or {}
    venue_if = summary.get("2yr_mean_citedness")
    venue_h = summary.get("h_index")

    # Compute authority tier
    tier = _compute_authority_tier(
        fwci, citation_percentile, venue_if, venue_h,
        work_type, cited_by, is_retracted,
    )

    # Extract authors
    authorships = work.get("authorships") or []
    authors = [a.get("author", {}).get("display_name", "") for a in authorships[:20]]

    # Extract abstract
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

    # OA info
    oa_info = work.get("open_access") or {}
    is_oa = oa_info.get("is_oa", False)
    oa_url = oa_info.get("oa_url", "")

    # Fields/concepts
    topics = work.get("topics") or []
    fields = [t.get("display_name", "") for t in topics[:5]]

    # External IDs
    ids = work.get("ids") or {}
    pmid = (ids.get("pmid") or "").replace("https://pubmed.ncbi.nlm.nih.gov/", "")

    now = datetime.now(timezone.utc).isoformat()

    conn.execute("""
        INSERT INTO papers (
            paper_id, doi, pmid, title, authors, abstract, year, venue,
            citation_count, fields_of_study, publication_types,
            publication_date, is_open_access, oa_pdf_url, s2_url,
            fwci, citation_percentile, venue_impact_factor, venue_h_index,
            venue_type, authority_tier, is_retracted, openalex_id, work_type,
            verified, first_seen, last_updated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            title = COALESCE(NULLIF(title, ''), excluded.title, title),
            abstract = COALESCE(NULLIF(abstract, ''), excluded.abstract, abstract),
            authors = COALESCE(NULLIF(authors, '[]'), excluded.authors, authors),
            citation_count = excluded.citation_count,
            fwci = COALESCE(excluded.fwci, fwci),
            citation_percentile = COALESCE(excluded.citation_percentile, citation_percentile),
            venue_impact_factor = COALESCE(excluded.venue_impact_factor, venue_impact_factor),
            venue_h_index = COALESCE(excluded.venue_h_index, venue_h_index),
            venue_type = COALESCE(excluded.venue_type, venue_type),
            authority_tier = COALESCE(excluded.authority_tier, authority_tier),
            is_retracted = excluded.is_retracted,
            openalex_id = COALESCE(excluded.openalex_id, openalex_id),
            work_type = COALESCE(excluded.work_type, work_type),
            is_open_access = excluded.is_open_access,
            verified = COALESCE(verified, 0),
            last_updated = excluded.last_updated
    """, (
        paper_id, doi, pmid, title, json.dumps(authors), abstract,
        work.get("publication_year"),
        venue_name, cited_by,
        json.dumps(fields),
        json.dumps([work_type] if work_type else []),
        work.get("publication_date", ""),
        1 if is_oa else 0,
        oa_url,
        f"https://openalex.org/works/{oa_id.split('/')[-1]}" if oa_id else "",
        fwci, citation_percentile, venue_if, venue_h,
        venue_type, tier, 1 if is_retracted else 0,
        oa_id, work_type,
        0,
        now, now,
    ))
    # Embedding deferred: OpenAlex searches store 50+ papers per call.
    # Embedding each synchronously would block the event loop for minutes.
    # Papers get embedded via: backfill script, download_paper, process_pdf, or set_abstract.

    # Persist citation edges from OpenAlex `referenced_works`. Each entry is a
    # full OpenAlex Work URL; _store_references resolves them to local `oa:{W_id}`
    # paper_ids and stores edges that match papers in our library.
    ref_works = work.get("referenced_works") or []
    if ref_works:
        refs_for_store = [{"openalex_id": rw} for rw in ref_works if rw]
        _store_references(conn, paper_id, refs_for_store, source="openalex")

    return paper_id


def _format_openalex_work(work: dict) -> str:
    """Format an OpenAlex work for display with authority metrics."""
    lines = []
    title = work.get("display_name") or work.get("title", "Unknown")
    year = work.get("publication_year", "?")
    cited_by = work.get("cited_by_count", 0)

    authorships = work.get("authorships") or []
    authors = [a.get("author", {}).get("display_name", "") for a in authorships[:5]]
    author_str = ", ".join(a for a in authors if a)
    if len(authorships) > 5:
        author_str += f" (+{len(authorships) - 5} more)"

    lines.append(f"**{title}** ({year})")
    if author_str:
        lines.append(f"Authors: {author_str}")
    lines.append(f"Citations: {cited_by}")

    # Authority metrics
    fwci = work.get("fwci")
    cite_norm = work.get("citation_normalized_percentile") or {}
    percentile_raw = cite_norm.get("value")
    # OpenAlex returns 0-1, scale to 0-100 for display and tier computation
    percentile = percentile_raw * 100 if percentile_raw is not None else None
    is_retracted = work.get("is_retracted", False)
    work_type = work.get("type", "")

    authority_parts = []
    if fwci is not None:
        authority_parts.append(f"FWCI: {fwci:.2f}")
    if percentile is not None:
        authority_parts.append(f"Percentile: {percentile:.1f}")
    if is_retracted:
        authority_parts.append("RETRACTED")
    if work_type:
        authority_parts.append(f"Type: {work_type}")
    if authority_parts:
        lines.append(f"Authority: {' | '.join(authority_parts)}")

    # Venue with quality
    primary_loc = work.get("primary_location") or {}
    source = primary_loc.get("source") or {}
    venue_name = source.get("display_name", "")
    if venue_name:
        venue_parts = [venue_name]
        summary = source.get("summary_stats") or {}
        venue_if = summary.get("2yr_mean_citedness")
        venue_h = summary.get("h_index")
        if venue_if is not None:
            venue_parts.append(f"IF: {venue_if:.1f}")
        if venue_h is not None:
            venue_parts.append(f"h-index: {venue_h}")
        lines.append(f"Venue: {' | '.join(venue_parts)}")

    # Compute tier for display
    tier = _compute_authority_tier(
        fwci, percentile,
        (source.get("summary_stats") or {}).get("2yr_mean_citedness"),
        (source.get("summary_stats") or {}).get("h_index"),
        work_type, cited_by, is_retracted,
    )
    tier_labels = {1: "TIER 1 (highest)", 2: "TIER 2 (strong)", 3: "TIER 3 (moderate)", 4: "TIER 4 (low)"}
    lines.append(f"Authority Tier: {tier_labels.get(tier, f'TIER {tier}')}")

    # DOI and IDs
    doi = work.get("doi", "")
    oa_id = work.get("id", "")
    id_parts = []
    if doi:
        id_parts.append(f"DOI: {doi.replace('https://doi.org/', '')}")
    if oa_id:
        id_parts.append(f"OpenAlex: {oa_id}")
    if id_parts:
        lines.append(" | ".join(id_parts))

    # OA status
    oa = work.get("open_access") or {}
    if oa.get("is_oa"):
        lines.append(f"Open Access: {oa.get('oa_status', 'yes')} ({oa.get('oa_url', 'no URL')})")
    else:
        lines.append("Open Access: No")

    # Topics
    topics = work.get("topics") or []
    if topics:
        topic_names = [t.get("display_name", "") for t in topics[:3]]
        lines.append(f"Topics: {', '.join(t for t in topic_names if t)}")

    # Abstract snippet
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
    if abstract:
        lines.append(f"Abstract: {abstract[:400]}{'...' if len(abstract) > 400 else ''}")

    return "\n".join(lines)


# --- PDF Processing ---


def _canonical_pdf_name(title: str, authors_json: str | None, year: int | None) -> str:
    """Generate a canonical PDF filename: Author_Year_SafeTitle.pdf"""
    # Parse first author's last name
    author = "Unknown"
    if authors_json:
        try:
            authors = json.loads(authors_json)
            if authors and isinstance(authors[0], str) and authors[0]:
                parts = authors[0].split()
                author = parts[-1] if parts else "Unknown"
        except (json.JSONDecodeError, TypeError):
            pass
    author = re.sub(r'[^\w]', '', author)[:30] or "Unknown"

    yr = str(year) if year else "undated"

    safe_title = re.sub(r'[^\w\s]', '', title or "untitled")[:80].strip()
    safe_title = re.sub(r'\s+', '_', safe_title)
    if not safe_title:
        safe_title = "untitled"

    return f"{author}_{yr}_{safe_title}.pdf"


def _is_orphan_paper(paper_id: str, title: str | None, authors_json: str | None, year: int | None) -> bool:
    """Detect when a paper was saved with generic metadata and needs extraction.

    Triggered when ANY of:
    - paper_id starts with 'local:' AND title is missing or empty
    - title is literally 'Unknown' or starts with 'Unknown_' (filename-stem variants)
    - title matches 'Unknown_undated_*' pattern
    - authors is missing AND year is missing AND paper_id is a local: entry
    - title is just the filename stem with underscores replaced (Docling couldn't extract a real title)
    """
    if not paper_id.startswith("local:"):
        return False
    if not title or not title.strip():
        return True
    t = title.strip()
    # Match only literal "Unknown" or "Unknown_..." stem patterns, not legitimate
    # titles like "Unknown Effects of X" (which have a space after "Unknown").
    if t.lower() == "unknown" or t.lower().startswith("unknown_"):
        return True
    # Generic patterns that indicate filename-based auto-title
    if re.match(r"^[\w\- ]*_undated_", t, re.IGNORECASE):
        return True
    # If there's no author and no year, it's an orphan
    has_authors = False
    if authors_json:
        try:
            authors = json.loads(authors_json)
            has_authors = bool(authors) and any(a.strip() for a in authors if isinstance(a, str))
        except (json.JSONDecodeError, TypeError):
            pass
    if not has_authors and not year:
        return True
    # Stem-matched title (paper_id is "local:X" where title looks like X with underscores → spaces)
    stem = paper_id[6:]  # strip "local:"
    stem_as_title = stem.replace("_", " ").replace("-", " ").strip()
    if t.lower() == stem_as_title.lower():
        return True
    return False


def _metadata_needed_message(paper_id: str, pdf_or_tex_path: str, is_tex: bool = False) -> str:
    """Build a METADATA_NEEDED warning for the tool response.

    The warning instructs the caller (Claude) to spawn a haiku subagent that
    reads the file, extracts title/authors/year, looks up the canonical S2
    record, and calls fix_orphan_paper to merge or update metadata.
    """
    source_type = "TeX source" if is_tex else "PDF"
    return f"""
⚠️ METADATA_NEEDED: This paper was stored as `{paper_id}` with generic/unknown metadata.
    File: {pdf_or_tex_path}

    SPAWN a haiku subagent with this prompt to fix it:

    ---
    Read the first 3 pages of {pdf_or_tex_path} (use the Read tool with pages="1-3" for PDFs,
    or Read the main .tex file for TeX). Extract the paper's:
      - Exact title
      - First 3 author last names
      - Publication year
      - DOI or arXiv ID if visible

    Then call `mcp__semantic-scholar__match_paper_by_title` with the extracted title.

    If a match is found with score >= 0.85:
      Call `mcp__semantic-scholar__fix_orphan_paper` with:
        orphan_id="{paper_id}"
        canonical_paper_id=<matched S2 ID>
      This merges the orphan's full text into the canonical record and deletes the orphan.

    If no match is found OR match score < 0.85:
      Call `mcp__semantic-scholar__fix_orphan_paper` with:
        orphan_id="{paper_id}"
        title=<extracted title>
        authors=<JSON array of author names>
        year=<extracted year>
        doi=<extracted DOI if any>
      This updates the local entry in place and renames the file.

    Return a one-line summary of what you did.
    ---"""


def _abstract_needed_message(paper_id: str) -> str:
    """Build an ABSTRACT_NEEDED warning for tool responses.

    Emitted when a paper has full text (processed_text or tex_text) but no
    abstract. The MCP client should spawn a small sub-agent that reads the
    full text, generates a 500-1000 word summary, and calls set_abstract.
    See docs/CLIENT_CONVENTIONS.md section 4 for the full protocol.
    """
    return f"""
⚠️ ABSTRACT_NEEDED: Paper `{paper_id}` has full text but no abstract.
    Paper-level semantic search needs an abstract (or LLM-generated summary)
    to produce strong embeddings. Without one, this paper will be under-
    represented in search_local results.

    SPAWN a haiku subagent with this prompt to generate one:

    ---
    Call `mcp__semantic-scholar__get_full_text` with paper_id="{paper_id}".
    Read the returned full text carefully.
    Generate a 500-1000 word abstract-style summary covering:
      - The paper's thesis or central argument
      - Methods or approach
      - Key findings, claims, or conclusions
      - Significance to its field

    Call `mcp__semantic-scholar__set_abstract` with:
        paper_id="{paper_id}"
        abstract=<your generated summary>

    Return a one-line confirmation.
    ---"""


def _check_and_flag_missing_abstract(conn: sqlite3.Connection, paper_id: str) -> str:
    """Return an ABSTRACT_NEEDED warning string if the paper has full text but
    no abstract. Empty string otherwise. Safe to append to a tool response.
    """
    row = conn.execute(
        "SELECT abstract, processed_text, tex_text FROM papers WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    if not row:
        return ""
    abstract, processed_text, tex_text = row
    if abstract and abstract.strip():
        return ""
    has_text = (processed_text and processed_text.strip()) or (tex_text and tex_text.strip())
    if has_text:
        return _abstract_needed_message(paper_id)
    return ""


def _canonical_pdf_path(conn: sqlite3.Connection, paper_id: str) -> Path:
    """Get the canonical path for a paper's PDF in PAPERS_DIR. Handles collisions."""
    row = conn.execute(
        "SELECT title, authors, year FROM papers WHERE paper_id = ?", (paper_id,)
    ).fetchone()
    if not row:
        return PAPERS_DIR / f"{paper_id}.pdf"

    title, authors_json, year = row
    filename = _canonical_pdf_name(title or "", authors_json, year)
    filepath = PAPERS_DIR / filename

    counter = 1
    while filepath.exists():
        stem = filepath.stem.rsplit('_', 1)
        base = filename[:-4]  # strip .pdf
        filepath = PAPERS_DIR / f"{base}_{counter}.pdf"
        counter += 1

    return filepath


def _kill_pgid_if_alive(proc: "asyncio.subprocess.Process", pgid: int) -> None:
    """Best-effort synchronous kill of a subprocess' process group.

    Called from finally blocks so the Docling/pandoc grandchildren can't
    linger after timeout, cancellation, or any other path out of the
    coroutine. The TOCTOU `proc.returncode is None` check keeps us from
    signalling a reused PGID after the child has already been reaped.
    """
    if proc.returncode is None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


async def _process_pdf(pdf_path: str) -> str:
    """Process a PDF to markdown via the Docling helper script (separate venv).

    Uses a process group (start_new_session=True) so the whole tree — `uv` wrapper
    plus the python3 grandchild that actually runs Docling — is killed when we
    exit this coroutine for ANY reason (timeout, CancelledError, or caller
    exception). Without this, SIGKILL to `uv` leaves the grandchild orphaned
    and still reading pipes, which hangs the caller's communicate() forever.
    """
    proc = await asyncio.create_subprocess_exec(
        "uv", "run", str(PROCESS_PDF_SCRIPT), pdf_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    pgid = proc.pid  # snapshot; PID may be reaped between returncode check and killpg
    try:
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=1800)
        except asyncio.TimeoutError:
            raise RuntimeError(f"PDF processing timed out after 30 minutes: {pdf_path}")
        if proc.returncode != 0:
            raise RuntimeError(f"PDF processing failed: {stderr.decode().strip()}")
        return stdout.decode()
    finally:
        # Kill the process group on every exit path: success (no-op since
        # returncode is set), TimeoutError raised above, CancelledError from
        # the caller, and any other exception.
        _kill_pgid_if_alive(proc, pgid)


async def _process_tex(tex_path: str) -> str:
    """Process a TeX source directory (or .tex file) to plain text via pandoc.

    Unlike process_pdf (which runs Docling), this uses the process_tex.py helper
    which inlines \\input{}/\\include{}, runs pandoc LaTeX→plain, and falls back
    to a regex stripper on failure. The regex fallback preserves LaTeX math syntax
    (\\frac{a}{b}, etc.); the pandoc path renders math as plain Unicode.
    """
    proc = await asyncio.create_subprocess_exec(
        "uv", "run", str(PROCESS_TEX_SCRIPT), tex_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    pgid = proc.pid
    try:
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        except asyncio.TimeoutError:
            raise RuntimeError(f"TeX processing timed out after 5 minutes: {tex_path}")
        if proc.returncode != 0:
            raise RuntimeError(f"TeX processing failed: {stderr.decode().strip()}")
        return stdout.decode()
    finally:
        _kill_pgid_if_alive(proc, pgid)


def _sync_store_pdf_text(
    path: "Path", markdown: str, paper_id_hint: str,
    *,
    verify_doi: str | None = None,
    verify_title: str | None = None,
    verify_authors: Any = None,
    move_to_canonical: bool = True,
) -> tuple[str, str | None, bool, str, bool, str]:
    """Post-Docling store work for process_pdf. Runs in a worker thread via
    asyncio.to_thread.

    Returns (pid, pdf_path_or_None, orphan_flag, abstract_warning, verify_ok, verify_reason).

    When verify_doi or verify_title is provided, the store stages text + chunks +
    embeddings without committing, runs _verify_paper_text_match, and either
    commits on pass or rolls back the entire transaction on fail. This is what
    makes process_pdf's verify gate atomic: there is no window between
    "bad text committed" and "clear issued" that a cancellation or crash can
    leave in an inconsistent state.

    The PDF file is only moved into canonical Papers/ position AFTER verify
    passes. A verify-fail leaves the staging PDF in place — earlier code
    auto-deleted via path-prefix heuristics and wiped user-placed files.
    The caller owns staging cleanup (custom download backends, _auto_process,
    acquire_batch). When path is already in Papers/ we leave it alone
    (caller owns that file).

    move_to_canonical : default True. When False, the file is left in place
        regardless of where it lives. Used by maintenance scripts (e.g.
        redocling_cleared phase 2) that scavenge the user's research tree
        and must NOT move user-owned files into Papers/.
    """
    conn = _init_db()
    try:
        if paper_id_hint:
            row = conn.execute("SELECT paper_id FROM papers WHERE paper_id = ?", (paper_id_hint,)).fetchone()
        else:
            row = conn.execute("SELECT paper_id FROM papers WHERE local_pdf_path = ?", (str(path),)).fetchone()

        if row:
            pid = row[0]
        else:
            resolved = _resolve_paper_id(conn, path, markdown)
            if resolved:
                pid = resolved
            else:
                pid = f"local:{path.stem}"
                now = datetime.now(timezone.utc).isoformat()
                conn.execute("""
                    INSERT INTO papers (paper_id, title, verified, first_seen, last_updated)
                    VALUES (?, ?, 1, ?, ?)
                    ON CONFLICT(paper_id) DO UPDATE SET
                        verified = 1,
                        last_updated = excluded.last_updated
                """, (pid, path.stem.replace("_", " "), now, now))

        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        # "Staged" = PDF lives outside PAPERS_DIR and we'll move it to the
        # canonical Papers/ location on verify-pass. We do NOT try to
        # classify ownership here (earlier attempts to auto-delete on
        # verify-fail using path-prefix heuristics were fragile — a user
        # could hand-drop a file into downloads/ and we'd wipe it). Instead
        # the caller is responsible for cleaning up its own staging file
        # on verify-fail; _sync_store_pdf_text just leaves it in place.
        # When move_to_canonical=False, the caller has explicitly asked us
        # NOT to relocate the file (maintenance scripts scanning the user's
        # research tree). In that case we treat the path as already in its
        # final location regardless of where it is.
        path_resolved = path.resolve()
        path_in_papers = path_resolved.parent == PAPERS_DIR.resolve()
        path_was_staged = (not path_in_papers) and move_to_canonical
        if path_was_staged:
            canonical = _canonical_pdf_path(conn, pid)
            pdf_path_to_store = str(canonical)
        else:
            canonical = path
            pdf_path_to_store = str(path)

        # Stage text + chunks + embeddings WITHOUT committing. If verify is
        # requested and fails, the whole transaction is rolled back atomically.
        _store_processed_text(
            pid, markdown,
            conn=conn, local_pdf_path=pdf_path_to_store, commit=False,
        )

        verify_ok, verify_reason = True, "no target"
        # 2026-04-26 update: verify_doi alone is no longer a gate. Only run
        # the verifier when at least one gating signal (title or authors) is
        # present. verify_doi accompanies as advisory.
        if verify_title or verify_authors:
            try:
                verify_ok, verify_reason = _verify_paper_text_match(
                    conn, pid,
                    target_doi=verify_doi,
                    target_title=verify_title,
                    target_authors=verify_authors,
                )
            except Exception as e:
                verify_ok, verify_reason = False, f"verifier raised {type(e).__name__}: {e}"

        if not verify_ok:
            # Atomic rollback: the staged text, chunks, embeddings, and any
            # stub row INSERT are all undone. No commit happened, so no
            # cancellation-window can leave bad state durably.
            try:
                conn.rollback()
            except Exception:
                pass
            # DO NOT delete the staging file here. Earlier iterations tried
            # to auto-delete based on path prefix, but that incorrectly
            # wiped user-placed files. The caller that staged the file
            # (acquire_batch, redocling_cleared, or a manual invocation)
            # owns cleanup.
            return pid, None, False, "", False, verify_reason

        # Verify passed (or no verify requested). Do ALL remaining DB reads
        # (orphan flag, abstract warning) BEFORE the FS move, so a raise
        # during those queries rolls the DB back with no filesystem change.
        orphan_row = conn.execute(
            "SELECT title, authors, year FROM papers WHERE paper_id = ?", (pid,)
        ).fetchone()
        orphan_flag = False
        if orphan_row:
            otitle, oauthors, oyear = orphan_row
            orphan_flag = _is_orphan_paper(pid, otitle, oauthors, oyear)
        abstract_warning = _check_and_flag_missing_abstract(conn, pid)

        # Now the only operations remaining are the FS move and the commit.
        # Both are wrapped so a failure of either triggers the rename-back
        # path and keeps FS/DB consistent.
        move_done = False
        if path_was_staged:
            try:
                shutil.move(str(path), str(canonical))
                move_done = True
            except Exception as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                return pid, None, False, "", False, f"post-verify move failed: {e}"

        try:
            conn.commit()
        except Exception as commit_err:
            # Commit-after-FS-move failure: DB will roll back on close
            # but the file is already at canonical. Best-effort rename
            # back so DB and FS stay consistent (both at staging). If
            # that also fails, log loudly and leave the file at
            # canonical — the row will be absent from the DB, so a
            # future ingestion can pick up the orphaned file.
            if move_done and path_was_staged:
                try:
                    shutil.move(str(canonical), str(path))
                except Exception as move_back_err:
                    print(
                        f"[CRITICAL] _sync_store_pdf_text: commit failed AND "
                        f"rename-back failed for {pid}: file at {canonical}, "
                        f"DB will revert to pre-commit state. commit={commit_err}; "
                        f"move_back={move_back_err}",
                        file=sys.stderr,
                    )
            raise
        return pid, pdf_path_to_store, orphan_flag, abstract_warning, True, verify_reason
    finally:
        conn.close()


def _store_processed_text(
    paper_id: str,
    text: str,
    *,
    conn: sqlite3.Connection | None = None,
    local_pdf_path: str | None = None,
    commit: bool = True,
) -> None:
    """Store PDF-extracted text (processed_text) + chunks + embeddings.

    Parameters
    ----------
    commit : default True. When False, caller is responsible for committing the
        connection. Used by store-verify-clear flows so verify failure can
        ROLLBACK the store without the text ever being durably visible.
        Only meaningful when `conn` is passed in (owns_conn=False). When
        owns_conn=True the helper always commits on success since the
        connection is closed immediately after.
    """
    owns_conn = conn is None
    if owns_conn:
        conn = _init_db()
    # Round-6 codex review HIGH fix: wrap text + chunk + embed mutations
    # in a SAVEPOINT so encode failure rolls back the text update too.
    # Round-7 schema review CRITICAL fix: SQLite `RELEASE sp` on the
    # OUTERMOST transaction frame commits to disk immediately. Python's
    # sqlite3 doesn't auto-BEGIN on SELECTs, so a caller that passed
    # `commit=False` after only-SELECT setup had no outer transaction;
    # the prior savepoint design then defeated `commit=False` (the
    # store committed before the caller's verify step could decide).
    # Force a BEGIN before the savepoint when there's no outer txn,
    # so RELEASE collapses into the BEGIN (still uncommitted) instead
    # of into autocommit. Track whether we opened the BEGIN so we
    # don't commit a caller's transaction we didn't open.
    sp_name = "store_processed_text"
    opened_outer_begin = False
    try:
        if not conn.in_transaction:
            conn.execute("BEGIN")
            opened_outer_begin = True
        conn.execute(f"SAVEPOINT {sp_name}")
        try:
            updates = ["processed_text = ?", "verified = 1", "last_updated = ?"]
            params: list = [text, datetime.now(timezone.utc).isoformat()]
            if local_pdf_path is not None:
                updates.append("local_pdf_path = ?")
                params.append(local_pdf_path)
            params.append(paper_id)
            conn.execute(f"UPDATE papers SET {', '.join(updates)} WHERE paper_id = ?", params)
            _recompute_has_full_text(conn, paper_id)
            _chunk_processed_text(conn, paper_id, text)
            _build_paper_passages(conn, paper_id, text)
            _embed_paper_if_verified(conn, paper_id, raise_on_error=True)
            _embed_chunks(conn, paper_id, raise_on_error=True)
            conn.execute(f"RELEASE {sp_name}")
        except Exception:
            try:
                conn.execute(f"ROLLBACK TO {sp_name}")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute(f"RELEASE {sp_name}")
            except sqlite3.OperationalError:
                pass
            # If we opened the outer BEGIN, we must rollback it too so
            # the text mutation is fully reverted. If the caller had
            # an outer BEGIN, leave their transaction state alone.
            if opened_outer_begin:
                try:
                    conn.rollback()
                except sqlite3.OperationalError:
                    pass
            raise
        if owns_conn or commit:
            conn.commit()
        # If commit=False AND we opened the BEGIN, defer to caller for
        # final commit/rollback. SQLite leaves the txn open here.
    finally:
        if owns_conn:
            conn.close()


def _store_tex_text(
    paper_id: str,
    text: str,
    *,
    conn: sqlite3.Connection | None = None,
    local_tex_path: str | None = None,
    commit: bool = True,
) -> None:
    """Store TeX-extracted text and source path. Reuses chunk/embedding pipeline.

    Chunking and paper-level embeddings re-run using the TeX text (which is math-clean)
    so the vector search surfaces TeX-sourced papers with proper equation fidelity.
    FTS5's processed_text column is untouched — the PDF-derived text remains the FTS5
    baseline for backwards compatibility.

    See _store_processed_text for the `commit` kwarg contract.
    """
    owns_conn = conn is None
    if owns_conn:
        conn = _init_db()
    # Round-6 codex review HIGH fix + Round-7 schema review CRITICAL
    # fix: same pattern as _store_processed_text. Force a BEGIN if no
    # outer transaction so SAVEPOINT/RELEASE doesn't auto-commit.
    sp_name = "store_tex_text"
    opened_outer_begin = False
    try:
        if not conn.in_transaction:
            conn.execute("BEGIN")
            opened_outer_begin = True
        conn.execute(f"SAVEPOINT {sp_name}")
        try:
            updates = ["tex_text = ?", "verified = 1", "last_updated = ?"]
            params: list = [text, datetime.now(timezone.utc).isoformat()]
            if local_tex_path is not None:
                updates.append("local_tex_path = ?")
                params.append(local_tex_path)
            params.append(paper_id)
            conn.execute(f"UPDATE papers SET {', '.join(updates)} WHERE paper_id = ?", params)
            _recompute_has_full_text(conn, paper_id)
            _chunk_processed_text(conn, paper_id, text)
            _embed_paper_if_verified(conn, paper_id, raise_on_error=True)
            _embed_chunks(conn, paper_id, raise_on_error=True)
            conn.execute(f"RELEASE {sp_name}")
        except Exception:
            try:
                conn.execute(f"ROLLBACK TO {sp_name}")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute(f"RELEASE {sp_name}")
            except sqlite3.OperationalError:
                pass
            if opened_outer_begin:
                try:
                    conn.rollback()
                except sqlite3.OperationalError:
                    pass
            raise
        if owns_conn or commit:
            conn.commit()
    finally:
        if owns_conn:
            conn.close()


# --- HTTP helpers ---

async def _throttled_get(client: httpx.AsyncClient, url: str, *, throttle: _Throttle | None = None, **kwargs) -> httpx.Response:
    if throttle is None:
        throttle = _s2_detail_throttle
    await throttle.wait()

    headers = kwargs.pop("headers", {})
    if S2_API_KEY and "semanticscholar" in url:
        headers["x-api-key"] = S2_API_KEY

    resp = await client.get(url, headers=headers, **kwargs)
    resp.raise_for_status()
    return resp


async def _throttled_post(client: httpx.AsyncClient, url: str, *, throttle: _Throttle | None = None, **kwargs) -> httpx.Response:
    if throttle is None:
        throttle = _s2_detail_throttle
    await throttle.wait()

    headers = kwargs.pop("headers", {})
    if S2_API_KEY and "semanticscholar" in url:
        headers["x-api-key"] = S2_API_KEY

    resp = await client.post(url, headers=headers, **kwargs)
    resp.raise_for_status()
    return resp


def _normalize_id(pid: str) -> str:
    pid = pid.strip()
    if re.match(r"^10\.\d{4,}/", pid):
        pid = f"DOI:{pid}"
    return pid


def _resolve_to_local_paper_id(conn: sqlite3.Connection, lookup: str) -> str | None:
    """Resolve a user-provided/normalized ID to a paper_id in our local papers table.

    Handles S2 SHA1 IDs, DOI: prefixed lookups, ARXIV:, PMID:, and raw DOIs.
    Returns None if no local paper matches.
    """
    lookup = lookup.strip()
    if not lookup:
        return None
    upper = lookup.upper()
    if upper.startswith("DOI:"):
        doi = lookup[4:].strip()
        row = conn.execute(
            "SELECT paper_id FROM papers WHERE lower(doi) = lower(?) LIMIT 1", (doi,)
        ).fetchone()
        return row[0] if row else None
    if upper.startswith("ARXIV:"):
        arxiv_id = lookup.split(":", 1)[1].strip()
        row = conn.execute(
            "SELECT paper_id FROM papers WHERE arxiv_id = ? LIMIT 1", (arxiv_id,)
        ).fetchone()
        return row[0] if row else None
    if upper.startswith("PMID:"):
        pmid = lookup[5:].strip()
        row = conn.execute(
            "SELECT paper_id FROM papers WHERE pmid = ? LIMIT 1", (pmid,)
        ).fetchone()
        return row[0] if row else None
    if re.match(r"^10\.\d{4,}/", lookup):
        row = conn.execute(
            "SELECT paper_id FROM papers WHERE lower(doi) = lower(?) LIMIT 1", (lookup,)
        ).fetchone()
        return row[0] if row else None
    # Raw S2 SHA1 or local:/oa: prefixed IDs — direct match
    row = conn.execute(
        "SELECT paper_id FROM papers WHERE paper_id = ? LIMIT 1", (lookup,)
    ).fetchone()
    return row[0] if row else None


def _format_paper(p: dict) -> str:
    lines = []
    title = p.get("title", "Unknown")
    year = p.get("year", "?")
    citations = p.get("citationCount", 0)
    influential = p.get("influentialCitationCount", 0)

    authors = p.get("authors") or []
    author_str = ", ".join(a.get("name", "") for a in authors[:5])
    if len(authors) > 5:
        author_str += f" (+{len(authors) - 5} more)"

    lines.append(f"**{title}** ({year})")
    if author_str:
        lines.append(f"Authors: {author_str}")
    lines.append(f"Citations: {citations} (influential: {influential})")

    venue = p.get("venue", "")
    if venue:
        lines.append(f"Venue: {venue}")

    pub_types = p.get("publicationTypes") or []
    if pub_types:
        lines.append(f"Type: {', '.join(pub_types)}")

    fields = p.get("fieldsOfStudy") or []
    if fields:
        lines.append(f"Fields: {', '.join(fields)}")

    ext_ids = p.get("externalIds") or {}
    doi = ext_ids.get("DOI", "")
    arxiv = ext_ids.get("ArXiv", "")
    pmid = ext_ids.get("PubMed", "")
    s2_id = p.get("paperId", "")

    id_parts = []
    if doi:
        id_parts.append(f"DOI: {doi}")
    if arxiv:
        id_parts.append(f"ArXiv: {arxiv}")
    if pmid:
        id_parts.append(f"PMID: {pmid}")
    if s2_id:
        id_parts.append(f"S2: {s2_id}")
    if id_parts:
        lines.append(" | ".join(id_parts))

    oa = p.get("isOpenAccess", False)
    oa_pdf = p.get("openAccessPdf") or {}
    if oa and oa_pdf.get("url"):
        lines.append(f"Open Access PDF: {oa_pdf['url']}")
    elif oa:
        lines.append("Open Access: Yes (no direct PDF link)")
    else:
        lines.append("Open Access: No")

    tldr = p.get("tldr") or {}
    if tldr.get("text"):
        lines.append(f"TLDR: {tldr['text']}")

    abstract = p.get("abstract", "")
    if abstract:
        lines.append(f"Abstract: {abstract[:500]}{'...' if len(abstract) > 500 else ''}")

    s2_url = p.get("url", "")
    if s2_url:
        lines.append(f"URL: {s2_url}")

    return "\n".join(lines)


# --- MCP Tools ---

@mcp.tool()
async def search_papers(
    query: str,
    limit: int = 10,
    year_range: str = "",
    publication_types: str = "",
    fields_of_study: str = "",
    open_access_only: bool = False,
    min_citations: int = 0,
) -> str:
    """Search for academic papers by keyword query. Results are auto-saved to the local paper library.

    Args:
        query: Search query string (e.g., "sparse identification nonlinear dynamics")
        limit: Max results to return (1-100, default 10)
        year_range: Filter by year range, e.g. "2020-2024" or "2020-" or "-2024"
        publication_types: Comma-separated types: Review, JournalArticle, Conference, ClinicalTrial, MetaAnalysis, Book, Study
        fields_of_study: Comma-separated fields: Computer Science, Medicine, Biology, Physics, Mathematics, Economics, etc.
        open_access_only: Only return open access papers
        min_citations: Minimum citation count filter
    """
    limit = max(1, min(100, limit))
    params = {
        "query": query,
        "limit": limit,
        "fields": DEFAULT_FIELDS,
    }
    if year_range:
        params["year"] = year_range
    if publication_types:
        params["publicationTypes"] = publication_types
    if fields_of_study:
        params["fieldsOfStudy"] = fields_of_study
    if open_access_only:
        params["openAccessPdf"] = ""
    if min_citations > 0:
        params["minCitationCount"] = min_citations

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await _throttled_get(client, f"{S2_BASE}/paper/search", throttle=_s2_search_throttle, params=params)
        data = resp.json()

    papers = data.get("data", [])
    total = data.get("total", 0)

    _store_papers(papers)

    if not papers:
        return f"No results found for '{query}'"

    parts = [f"Found {total} results (showing {len(papers)}, all saved to local library):\n"]
    for i, p in enumerate(papers, 1):
        parts.append(f"--- Result {i} ---")
        parts.append(_format_paper(p))
        parts.append("")

    return "\n".join(parts)


@mcp.tool()
async def get_paper(paper_id: str) -> str:
    """Get full details for a single paper. Auto-saved to local library.

    Args:
        paper_id: Any supported ID format:
            - DOI: "DOI:10.1073/pnas.1517384113" or just "10.1073/pnas.1517384113"
            - ArXiv: "ARXIV:2106.15928" or "arXiv:2106.15928"
            - PubMed: "PMID:19872477"
            - S2 ID: "649def34f8be52c8b66281af98ae884c09aef38b"
            - URL: "URL:https://arxiv.org/abs/2106.15928v1"
    """
    pid = _normalize_id(paper_id)

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await _throttled_get(
            client,
            f"{S2_BASE}/paper/{pid}",
            params={"fields": DEFAULT_FIELDS},
        )
        data = resp.json()

    _store_papers([data])
    return _format_paper(data)


@mcp.tool()
async def get_papers_batch(paper_ids: list[str]) -> str:
    """Get details for multiple papers in a single API call. Up to 500 IDs per call.

    Far more efficient than calling get_paper repeatedly. Use this when you have
    multiple paper IDs (DOIs, S2 IDs, ArXiv IDs, etc.) to look up. Returns metadata
    including openAccessPdf URLs, so you can download PDFs without a second API call.

    Args:
        paper_ids: List of paper IDs (any supported format: DOI, S2 ID, ArXiv, PMID, URL).
                   Max 500 per call. Example: ["DOI:10.1234/foo", "649def34f8be52c8b66281af98ae884c09aef38b"]
    """
    if not paper_ids:
        return "Error: no paper IDs provided"
    if len(paper_ids) > 500:
        return f"Error: max 500 IDs per batch, got {len(paper_ids)}. Split into multiple calls."

    normalized = [_normalize_id(pid) for pid in paper_ids]

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await _throttled_post(
            client,
            f"{S2_BASE}/paper/batch",
            throttle=_s2_search_throttle,
            params={"fields": DEFAULT_FIELDS},
            json={"ids": normalized},
        )
        results = resp.json()

    papers = [p for p in results if p is not None]
    nulls = sum(1 for p in results if p is None)

    _store_papers(papers)

    parts = [f"Batch lookup: {len(papers)} found, {nulls} not found (all saved to local library).\n"]
    for i, p in enumerate(papers, 1):
        parts.append(f"--- Paper {i} ---")
        parts.append(_format_paper(p))
        parts.append("")

    if nulls > 0:
        not_found_ids = [normalized[i] for i, p in enumerate(results) if p is None]
        parts.append(f"Not found ({nulls}): {', '.join(not_found_ids[:20])}")
        if nulls > 20:
            parts.append(f"  ... and {nulls - 20} more")

    return "\n".join(parts)


@mcp.tool()
async def match_paper_by_title(title: str) -> str:
    """Find the single best-matching paper for a given title. More accurate than search_papers for title resolution.

    Returns the highest-confidence match with a matchScore, or 404 if no match.
    Use this when you have a paper title but no DOI or S2 ID.

    Args:
        title: The paper title to match (e.g., "Attention Is All You Need")
    """
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await _throttled_get(
                client,
                f"{S2_BASE}/paper/search/match",
                throttle=_s2_search_throttle,
                params={"query": title, "fields": DEFAULT_FIELDS},
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return f"No match found for title: {title}"
            raise
        data = resp.json()

    paper = data.get("data", [{}])[0] if data.get("data") else data
    match_score = data.get("matchScore") or paper.get("matchScore", "N/A")

    _store_papers([paper])

    result = _format_paper(paper)
    return f"Match score: {match_score}\n{result}"


@mcp.tool()
async def get_citations(paper_id: str, limit: int = 20) -> str:
    """Get papers that cite a given paper (who cites this work). All results saved to local library.

    Args:
        paper_id: Any supported paper ID format (DOI, ArXiv, PMID, S2 ID, URL)
        limit: Max results (1-1000, default 20)
    """
    pid = _normalize_id(paper_id)
    limit = max(1, min(1000, limit))

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await _throttled_get(
            client,
            f"{S2_BASE}/paper/{pid}/citations",
            params={"fields": CITATION_FIELDS, "limit": limit},
        )
        data = resp.json()

    items = data.get("data", [])
    citing_papers = [item.get("citingPaper", {}) for item in items]
    _store_papers(citing_papers)

    # Persist citation edges. Edges go FROM each citing paper TO the lookup paper.
    # Resolve both sides to local paper_ids before storing.
    conn = _init_db()
    try:
        target_local_id = _resolve_to_local_paper_id(conn, pid)
        if target_local_id:
            now = datetime.now(timezone.utc).isoformat()
            for item in items:
                citing_p = item.get("citingPaper") or {}
                citing_s2_id = (citing_p.get("paperId") or "").strip()
                if not citing_s2_id:
                    continue
                # The citing paper was just stored via _store_papers, so it should
                # exist in papers under its S2 ID.
                citing_local = _resolve_to_local_paper_id(conn, citing_s2_id)
                if not citing_local:
                    continue
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO paper_references
                        (citing_paper_id, cited_paper_id, cited_doi, source, is_influential, first_seen)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        citing_local,
                        target_local_id,
                        None,
                        "s2_api",
                        1 if item.get("isInfluential") else 0,
                        now,
                    ))
                except sqlite3.Error as e:
                    print(f"  citation-edge insert error: {e}", file=sys.stderr)
            conn.commit()
    finally:
        conn.close()

    if not items:
        return "No citations found."

    parts = [f"Found {len(items)} citing papers (all saved to local library):\n"]
    for i, item in enumerate(items, 1):
        citing = item.get("citingPaper", {})
        parts.append(f"--- Citation {i} ---")
        parts.append(_format_paper(citing))
        parts.append("")

    return "\n".join(parts)


@mcp.tool()
async def get_references(paper_id: str, limit: int = 20) -> str:
    """Get papers that a given paper cites (its bibliography/references). All results saved to local library.
    Also persists citation edges in paper_references for graph-boosted retrieval.

    Args:
        paper_id: Any supported paper ID format (DOI, ArXiv, PMID, S2 ID, URL)
        limit: Max results (1-1000, default 20)
    """
    pid = _normalize_id(paper_id)
    limit = max(1, min(1000, limit))

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await _throttled_get(
            client,
            f"{S2_BASE}/paper/{pid}/references",
            params={"fields": CITATION_FIELDS, "limit": limit},
        )
        data = resp.json()

    items = data.get("data", [])
    cited_papers = [item.get("citedPaper", {}) for item in items]
    _store_papers(cited_papers)

    # Persist citation edges. Need to resolve the source paper's local ID first.
    conn = _init_db()
    try:
        citing_local_id = _resolve_to_local_paper_id(conn, pid)
        if citing_local_id and items:
            refs_for_store = [
                {
                    "paperId": (item.get("citedPaper") or {}).get("paperId", ""),
                    "doi": ((item.get("citedPaper") or {}).get("externalIds") or {}).get("DOI", ""),
                    "is_influential": bool(item.get("isInfluential", False)),
                }
                for item in items
            ]
            _store_references(conn, citing_local_id, refs_for_store, source="s2_api")
            conn.commit()
    finally:
        conn.close()

    if not items:
        return "No references found."

    parts = [f"Found {len(items)} references (all saved to local library):\n"]
    for i, item in enumerate(items, 1):
        cited = item.get("citedPaper", {})
        parts.append(f"--- Reference {i} ---")
        parts.append(_format_paper(cited))
        parts.append("")

    return "\n".join(parts)


@mcp.tool()
async def recommend_papers(paper_ids: list[str], limit: int = 10) -> str:
    """Get paper recommendations based on one or more seed papers. All results saved to local library.

    Args:
        paper_ids: List of paper IDs (DOI, ArXiv, PMID, S2 ID) to use as positive examples
        limit: Max recommendations (1-500, default 10)
    """
    cleaned = [_normalize_id(pid) for pid in paper_ids]
    limit = max(1, min(500, limit))

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await _throttled_post(
            client,
            f"{S2_RECOMMEND_BASE}/papers/",
            throttle=_s2_search_throttle,
            json={"positivePaperIds": cleaned, "negativePaperIds": []},
            params={"fields": DEFAULT_FIELDS, "limit": limit},
        )
        data = resp.json()

    papers = data.get("recommendedPapers", [])
    _store_papers(papers)

    if not papers:
        return "No recommendations found."

    parts = [f"Found {len(papers)} recommendations (all saved to local library):\n"]
    for i, p in enumerate(papers, 1):
        parts.append(f"--- Recommendation {i} ---")
        parts.append(_format_paper(p))
        parts.append("")

    return "\n".join(parts)


@mcp.tool()
async def download_paper(paper_id: str, output_dir: str = "") -> str:
    """Download a paper's PDF if openly accessible. Updates local library with PDF path.
    Auto-processes downloaded PDFs to markdown for full-text search.
    Saves to $PAPERS_DIR (configurable via env var; see research_mcp.paths) with
    canonical Author_Year_Title.pdf naming by default.

    Tries Semantic Scholar's OA PDF link first, then Unpaywall as fallback.
    Returns the file path if successful, or instructions for manual retrieval.

    Args:
        paper_id: Any supported paper ID format (DOI, ArXiv, PMID, S2 ID, URL)
        output_dir: Directory to save the PDF. Defaults to $PAPERS_DIR.
    """
    pid = _normalize_id(paper_id)

    out_path = Path(output_dir) if output_dir else PAPERS_DIR
    out_path.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        return f"Error: output directory does not exist: {output_dir}"

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await _throttled_get(
            client,
            f"{S2_BASE}/paper/{pid}",
            params={"fields": DEFAULT_FIELDS},
        )
        paper = resp.json()

        _store_papers([paper])

        title = paper.get("title", "unknown")
        ext_ids = paper.get("externalIds") or {}
        doi = ext_ids.get("DOI", "")
        s2_url = paper.get("url", "")
        s2_id = paper.get("paperId", "")

        authors = paper.get("authors") or []
        authors_json = json.dumps([a.get("name", "") for a in authors])
        year = paper.get("year")
        canonical_name = _canonical_pdf_name(title, authors_json, year)

        def _save_pdf_path(filepath: Path) -> None:
            conn = _init_db()
            try:
                conn.execute(
                    "UPDATE papers SET local_pdf_path = ?, verified = 1, last_updated = ? WHERE paper_id = ?",
                    (str(filepath), datetime.now(timezone.utc).isoformat(), s2_id),
                )
                conn.commit()
            finally:
                conn.close()

        async def _auto_process(filepath: Path) -> str:
            """Try to process downloaded PDF. Returns status suffix.

            Round-8 review writes M1 fix: route the store + verify cycle
            through `_sync_store_pdf_text`, which stages text + chunks +
            embeddings without committing, runs the verify gate, and
            either commits-and-moves on pass or rolls back atomically on
            fail. The previous implementation committed _store_processed_text
            first and only cleared on verify-fail; an interrupt or crash
            in that window left bad text durably committed (the same
            failure mode `_sync_store_pdf_text` was designed to close).
            """
            try:
                markdown = await _process_pdf(str(filepath))
                if not markdown or len(markdown.strip()) < 100:
                    return " | PDF processing deferred (run process_pdf manually)"
                pid_out, pdf_path_out, _orphan_flag, abstract_warning, verify_ok, verify_reason = (
                    await asyncio.to_thread(
                        _sync_store_pdf_text,
                        filepath, markdown, s2_id,
                        verify_doi=doi or None,
                        verify_title=title,
                        verify_authors=authors_json or None,
                        move_to_canonical=False,
                    )
                )
                if not verify_ok:
                    # Atomic rollback already happened inside the helper;
                    # the staged text/chunks/embeddings/pdf_path are
                    # gone. Clear local_pdf_path defensively because
                    # _save_pdf_path() committed it BEFORE _auto_process
                    # ran. Round-9 codex MED fix: use compare-and-clear
                    # so a concurrent download_paper call for the SAME
                    # s2_id that succeeded between our _save_pdf_path
                    # and now isn't penalized — only null the path if
                    # it still points at OUR rejected filepath. SQLite
                    # serializes writers, so the WHERE matches against
                    # whatever the most-recent committed value is.
                    cleanup_conn = _init_db()
                    try:
                        cleanup_conn.execute(
                            "UPDATE papers SET local_pdf_path = NULL, last_updated = ? "
                            "WHERE paper_id = ? AND local_pdf_path = ?",
                            (datetime.now(timezone.utc).isoformat(), s2_id, str(filepath)),
                        )
                        cleanup_conn.commit()
                    finally:
                        cleanup_conn.close()
                    return f" | DOWNLOAD REJECTED: {verify_reason}"
                suffix = f" | Processed: {len(markdown):,} chars extracted, searchable via search_local"
                if abstract_warning:
                    suffix += "\n" + abstract_warning
                return suffix
            except Exception as e:
                print(f"Auto-process error for {filepath}: {e}", file=sys.stderr)
            return " | PDF processing deferred (run process_pdf manually)"

        # Try S2 OA PDF
        oa_pdf = paper.get("openAccessPdf") or {}
        pdf_url = oa_pdf.get("url", "")

        def _is_valid_pdf(content: bytes) -> bool:
            return content[:5] == b"%PDF-"

        if pdf_url:
            try:
                pdf_resp = await client.get(pdf_url, timeout=60)
                if pdf_resp.status_code == 200 and len(pdf_resp.content) > 1000 and _is_valid_pdf(pdf_resp.content):
                    filepath = out_path / canonical_name
                    counter = 1
                    while filepath.exists():
                        filepath = out_path / f"{canonical_name[:-4]}_{counter}.pdf"
                        counter += 1
                    filepath.write_bytes(pdf_resp.content)
                    _save_pdf_path(filepath)
                    process_status = await _auto_process(filepath)
                    return f"Downloaded ({len(pdf_resp.content)} bytes): {filepath}\nTitle: {title}{process_status}"
            except Exception as e:
                print(f"S2 PDF download error for {pdf_url}: {e}", file=sys.stderr)

        # Try Unpaywall if we have a DOI
        if doi and UNPAYWALL_EMAIL:
            try:
                uw_resp = await _throttled_get(
                    client,
                    f"{UNPAYWALL_BASE}/{doi}",
                    throttle=_unpaywall_throttle,
                    params={"email": UNPAYWALL_EMAIL},
                )
                uw_data = uw_resp.json()

                if uw_data.get("is_oa"):
                    best = uw_data.get("best_oa_location") or {}
                    pdf_url = best.get("url_for_pdf") or best.get("url", "")

                    if pdf_url:
                        pdf_resp = await client.get(pdf_url, timeout=60)
                        if pdf_resp.status_code == 200 and len(pdf_resp.content) > 1000 and _is_valid_pdf(pdf_resp.content):
                            filepath = out_path / canonical_name
                            counter = 1
                            while filepath.exists():
                                filepath = out_path / f"{canonical_name[:-4]}_{counter}.pdf"
                                counter += 1
                            filepath.write_bytes(pdf_resp.content)
                            _save_pdf_path(filepath)
                            process_status = await _auto_process(filepath)
                            return f"Downloaded via Unpaywall ({len(pdf_resp.content)} bytes): {filepath}\nTitle: {title}{process_status}"
            except httpx.HTTPStatusError:
                pass
            except Exception as e:
                print(f"Unpaywall download error: {e}", file=sys.stderr)

        lines = [
            f"Paper not available as open access PDF.",
            f"Title: {title}",
        ]
        if doi:
            lines.append(f"DOI: https://doi.org/{doi}")
        if s2_url:
            lines.append(f"Semantic Scholar: {s2_url}")
        lines.append("")
        lines.append("To get this paper manually:")
        if doi:
            lines.append(f"  1. Try: https://doi.org/{doi}")
            lines.append(f"  2. Try your institution's library access or interlibrary loan")
        lines.append(f"  3. Save anywhere, then run process_pdf to extract full text")
        lines.append(f"     (process_pdf will auto-copy to {PAPERS_DIR})")

        return "\n".join(lines)


def _extract_doi_from_filename(stem: str) -> str | None:
    """Try to extract a DOI from a PDF filename.

    Handles: 10_1093_eurheartj_eht151, 10-1093-ije-dyw098, 10.1136 bjsports-2021-104080
    """
    m = re.search(r'10[._\-](\d{4,5})[._\-/\s](.+)', stem)
    if m:
        suffix = m.group(2).rstrip('-_. ')
        return f"10.{m.group(1)}/{suffix}"
    return None


_DOI_LABELED_PAT = re.compile(
    r'(?:doi|DOI|https?://(?:dx\.)?doi\.org/)[:\s]*(10\.\d{4,9}/\S+)'
)
_DOI_BARE_PAT = re.compile(r'\b(10\.\d{4,9}/\S+)')
# Trailing punctuation that's almost certainly sentence noise, not DOI.
# Note: `;` and `,` stay OUT of this set — SICI-style DOIs like
# 10.1130/0091-7613(1999)027<0359:ELOMAI>2.3.CO;2 keep `;` internally.
# Only characters at the end of the DOI string are stripped; internal `>`
# in SICI DOIs is not at the end (`2.3.CO;2` follows), so `>` is safe to
# strip when it's a URL-wrapper like <doi.org/10.1234/foo>.
_DOI_TRAILING_PUNCT = ".\"')>]"


def _clean_doi_suffix(doi: str) -> str:
    """Strip trailing punctuation that was captured by the greedy regex.

    The bare regex uses \\S+ (any non-whitespace) so SICI-style DOIs keep
    their full identifier (parens, angle brackets, `;`, `,`, `:` are all
    legal per the DOI Handbook). Whatever trailing punctuation is pure
    sentence noise (periods, closing quotes, HTML bracket noise) gets
    peeled back here. We also strip a single trailing `.` because a DOI
    never ends in a period.
    """
    return doi.rstrip(_DOI_TRAILING_PUNCT)


def _extract_doi_from_text(text: str, head_chars: int = 50000) -> str | None:
    """Extract the paper's own DOI from extracted markdown.

    Strategy (in order):
      1. Prefer a DOI appearing with an explicit label: "doi:...", "DOI:...",
         "https://doi.org/..." or "dx.doi.org/...". These almost always mark
         the paper's own DOI in the header or copyright block.
      2. If no labeled DOI, scan the first head_chars and return the DOI
         with the highest frequency. The paper's own DOI typically recurs
         in header/footer; citation DOIs appear once each.
      3. Fall back to the LAST unlabeled DOI in the first head_chars
         (paper's DOI usually precedes the references section, where
         citation DOIs appear).

    head_chars defaults to 50K (matches _verify_paper_text_match's window)
    so a paper whose own DOI only appears on page 2 or later of the extracted
    markdown still gets picked up.
    """
    head = text[:head_chars]
    m = _DOI_LABELED_PAT.search(head)
    if m:
        return _clean_doi_suffix(m.group(1))

    unlabeled = _DOI_BARE_PAT.findall(head)
    if not unlabeled:
        return None
    cleaned = [_clean_doi_suffix(d) for d in unlabeled if len(d) > 10]
    if not cleaned:
        return None
    from collections import Counter
    counts = Counter(cleaned)
    most_common_doi, count = counts.most_common(1)[0]
    if count > 1:
        return most_common_doi

    return cleaned[-1]


def _resolve_paper_id(conn: sqlite3.Connection, path: Path, markdown: str) -> str | None:
    """Try to match a PDF to an existing DB entry by DOI before creating a local: entry.

    Checks DOI from filename first (cheap), then DOI from extracted text.
    For each DOI candidate, tries exact match then underscore-to-slash variants
    (filenames encode 10.1093/eurheartj/eht151 as 10_1093_eurheartj_eht151).
    Returns the existing paper_id if found, None otherwise.
    """
    for doi in [_extract_doi_from_filename(path.stem), _extract_doi_from_text(markdown)]:
        if not doi:
            continue
        # Try exact match first
        row = conn.execute(
            "SELECT paper_id FROM papers WHERE lower(doi) = lower(?) LIMIT 1",
            (doi,),
        ).fetchone()
        if row:
            return row[0]
        # Try replacing underscores with slashes in the suffix
        # (filenames use _ where DOIs use /)
        parts = doi.split("/", 1)
        if len(parts) == 2:
            suffix_slashed = parts[1].replace("_", "/")
            if suffix_slashed != parts[1]:
                variant = f"{parts[0]}/{suffix_slashed}"
                row = conn.execute(
                    "SELECT paper_id FROM papers WHERE lower(doi) = lower(?) LIMIT 1",
                    (variant,),
                ).fetchone()
                if row:
                    return row[0]
            # Also try replacing hyphens with slashes (10-1093-ije-dyw098)
            suffix_from_hyphens = parts[1].replace("-", "/")
            if suffix_from_hyphens != parts[1] and suffix_from_hyphens != suffix_slashed:
                variant = f"{parts[0]}/{suffix_from_hyphens}"
                row = conn.execute(
                    "SELECT paper_id FROM papers WHERE lower(doi) = lower(?) LIMIT 1",
                    (variant,),
                ).fetchone()
                if row:
                    return row[0]
    return None


@mcp.tool()
async def process_pdf(
    pdf_path: str,
    paper_id: str = "",
    *,
    verify_doi: str | None = None,
    verify_title: str | None = None,
    verify_authors: Any = None,
    move_to_canonical: bool = True,
) -> str:
    """Process a PDF to markdown using Docling and store full text for search.

    Use this for manually-placed PDFs or to reprocess a previously downloaded paper.
    The extracted text becomes searchable via search_local and search_passages.

    Verification (2026-04-26 update): the gate is now `title + author surnames`.
    DOI is NO LONGER A GATE — verify_doi is accepted but logged advisory-only.
    The verifier runs whenever `verify_title` OR `verify_authors` is provided.
    The store is atomic: text, chunks, embeddings, and local_pdf_path are staged
    uncommitted, then `_verify_paper_text_match` runs, and either the entire
    transaction commits (verify pass) or rolls back (verify fail).

    Args:
        pdf_path: Absolute path to a PDF file
        paper_id: Optional S2 paper ID to associate with. If empty, auto-detects by DOI from filename/text, then by path, or creates a local entry.
        verify_doi: Optional DOI. Logged advisory-only. Does NOT cause
            verify-fail when missing from the extracted text head.
        verify_title: Title whose distinctive words must appear in the stored
            text head. Verify failure rolls back the store atomically (no DB
            trace of the wrong text); the staging PDF on disk is NOT
            auto-deleted — caller owns cleanup.
        verify_authors: Optional authors field (JSON string, list[str], or
            single str). At least one author surname must appear as a
            whole-word match in the stored text head. Same atomic rollback
            semantics as verify_title.
        move_to_canonical: default True. When False, the PDF is left at its
            current location instead of being moved into Papers/. Used by
            maintenance scripts (e.g. redocling_cleared phase 2) that ingest
            from the user's research tree and must NOT relocate user files.
    """
    path = Path(pdf_path)
    if not path.exists():
        return f"Error: file not found: {pdf_path}"
    if path.suffix.lower() != ".pdf":
        return f"Error: not a PDF file: {pdf_path}"

    try:
        markdown = await _process_pdf(str(path))
    except Exception as e:
        return f"Error processing PDF: {e}"

    if not markdown or len(markdown.strip()) < 100:
        return f"Warning: extracted very little text ({len(markdown)} chars). PDF may be image-only or corrupted."

    # Move DB write + chunking + embedding into a worker thread.
    # These are CPU + SQLite-bound and would block the event loop otherwise
    # (embedding alone is 1–60s per paper). `_sync_store_pdf_text` opens its own
    # connection inside the thread so SQLite's thread-affinity is respected.
    pid, pdf_path_to_store, orphan_flag, abstract_warning, verify_ok, verify_reason = await asyncio.to_thread(
        _sync_store_pdf_text, path, markdown, paper_id,
        verify_doi=verify_doi, verify_title=verify_title,
        verify_authors=verify_authors,
        move_to_canonical=move_to_canonical,
    )

    if not verify_ok:
        # Inline-verify atomic rollback. Caller sees VERIFY_REJECTED and the
        # DB has no trace of the wrong text. The staging PDF is NOT deleted
        # here — _sync_store_pdf_text leaves on-disk artifacts intact for
        # the caller to clean up (auto-deletion is fragile when the staging
        # path is user-placed). Round-11 review MED fix: corrected the
        # earlier comment that falsely claimed the file was deleted.
        return f"VERIFY_REJECTED: {verify_reason}"

    char_count = len(markdown)
    word_count = len(markdown.split())
    result = (
        f"Processed: {path.name}\nStored as: {pid}\n"
        f"Extracted: {char_count:,} chars, ~{word_count:,} words\n"
        f"PDF: {pdf_path_to_store}\n"
        f"Full text is now searchable via search_local."
    )
    if orphan_flag:
        result += "\n" + _metadata_needed_message(pid, pdf_path_to_store, is_tex=False)
    if abstract_warning:
        result += "\n" + abstract_warning
    return result


async def _process_pdf_and_verify(
    pdf_path: str,
    paper_id: str,
    *,
    target_doi: str | None = None,
    target_title: str | None = None,
    target_authors: Any = None,
) -> tuple[str, int, bool]:
    """Wrap process_pdf with its in-transaction verify gate.

    Single shared entry point used by BOTH acquire_batch.py and
    process_inbox.py so the verify gate cannot drift across entry points.

    Atomicity: process_pdf + _sync_store_pdf_text now run the verify step
    INSIDE the same transaction as the text store. A failure rolls the store
    back before commit so there is no "bad text durably stored" window that
    a cancellation/crash could leave behind.

    Returns (result_text, chars, verified_ok). On verify failure
    `result_text` starts with "VERIFY_REJECTED:" and chars is 0.
    """
    result_text = await process_pdf(
        pdf_path, paper_id=paper_id,
        verify_doi=target_doi, verify_title=target_title,
        verify_authors=target_authors,
    )
    if result_text.startswith("Processed:"):
        chars = _parse_chars(result_text)
        if chars <= 0:
            # Shouldn't happen — process_pdf's own length check catches <100.
            # Guard anyway so downstream callers never see ok=True with chars=0.
            return "VERIFY_REJECTED: process_pdf returned 0 chars", 0, False
        return result_text, chars, True
    if result_text.startswith("VERIFY_REJECTED"):
        return result_text, 0, False
    # Other errors: Warning:, Error:, etc. Treat as not-downloaded.
    return result_text, 0, False


@mcp.tool()
async def process_tex(tex_path: str, paper_id: str = "") -> str:
    """Process a TeX source directory (or single .tex file) into math-clean full text.

    Runs pandoc (LaTeX→plain with math preserved inline) on the TeX source,
    inlining any \\input{}/\\include{} directives first. Stores the result in the
    `tex_text` column and the source path in `local_tex_path`. Once stored,
    `get_full_text` will return TeX-sourced text in preference to Docling PDF text,
    and new embeddings for the paper are recomputed from the TeX text.

    Use for manually-placed TeX sources (typically under $TEX_DIR)
    or to fix math-garbled PDF extractions. PDF-derived processed_text is left
    untouched as a fallback.

    Args:
        tex_path: Absolute path to a .tex file or a directory containing TeX source.
        paper_id: Optional S2 paper ID, DOI (prefix "DOI:"), or local ID. If empty,
                  falls back to `local:<directory_name>`.
    """
    path = Path(tex_path).expanduser().resolve()
    if not path.exists():
        return f"Error: path not found: {tex_path}"

    try:
        text = await _process_tex(str(path))
    except Exception as e:
        return f"Error processing TeX: {e}"

    if not text or len(text.strip()) < 100:
        return f"Warning: extracted very little text ({len(text)} chars). TeX source may be empty or malformed."

    conn = _init_db()
    try:
        # Resolve paper_id: explicit arg → DOI lookup → ARXIV: resolve → filesystem fallback.
        # row_was_preexisting: True iff the resolved pid pointed at a row that
        # already existed with prior metadata. Only preexisting rows are verified
        # against their stored DOI/arxiv_id/title; rows we just created from the
        # TeX stem have no independent identity to verify against.
        pid: str | None = None
        row_was_preexisting = False
        if paper_id:
            if paper_id.upper().startswith("DOI:"):
                doi = paper_id[4:].strip()
                row = conn.execute(
                    "SELECT paper_id FROM papers WHERE lower(doi) = lower(?)",
                    (doi,),
                ).fetchone()
                if row:
                    pid = row[0]
                    row_was_preexisting = True
            else:
                row = conn.execute(
                    "SELECT paper_id FROM papers WHERE paper_id = ?",
                    (paper_id,),
                ).fetchone()
                if row:
                    pid = row[0]
                    row_was_preexisting = True
                else:
                    # Try resolving ARXIV: prefix or other lookups
                    resolved = _resolve_to_local_paper_id(conn, paper_id)
                    if resolved:
                        pid = resolved
                        row_was_preexisting = True
                    else:
                        # paper_id doesn't match any row. Guard against caller typos
                        # that would silently create orphan rows with invalid IDs.
                        # Accept only recognized prefixes; otherwise reject.
                        allowed_prefix = (
                            paper_id.startswith("local:")
                            or paper_id.startswith("oa:")
                            or paper_id.upper().startswith("ARXIV:")
                            or (len(paper_id) == 40 and all(c in "0123456789abcdef" for c in paper_id))
                        )
                        if not allowed_prefix:
                            return (
                                f"Error: paper_id '{paper_id}' not found in DB and does not match "
                                "a known prefix (local:, oa:, ARXIV:, or 40-hex S2 SHA). "
                                "Use a prefixed ID or omit paper_id to auto-resolve."
                            )
                        pid = paper_id
                        stem = path.name if path.is_dir() else path.stem
                        now = datetime.now(timezone.utc).isoformat()
                        conn.execute("""
                            INSERT INTO papers (paper_id, title, verified, first_seen, last_updated)
                            VALUES (?, ?, 1, ?, ?)
                            ON CONFLICT(paper_id) DO UPDATE SET
                                verified = 1, last_updated = excluded.last_updated
                        """, (pid, stem.replace("_", " "), now, now))

        if pid is None:
            # Try to match by existing local_tex_path
            row = conn.execute(
                "SELECT paper_id FROM papers WHERE local_tex_path = ?",
                (str(path),),
            ).fetchone()
            if row:
                pid = row[0]
                row_was_preexisting = True

        if pid is None:
            # Create a local: entry keyed off the directory/file stem
            stem = path.stem if path.is_file() else path.name
            pid = f"local:{stem}"
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO papers (paper_id, title, verified, first_seen, last_updated)
                VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    verified = 1,
                    last_updated = excluded.last_updated
                """,
                (pid, stem.replace("_", " "), now, now),
            )

        # Stage the TeX text WITHOUT committing so the verify step can reject it
        # atomically. If verify fails we _clear_paper_text and commit the clear;
        # net effect is no durable change to the row's tex_text.
        #
        # This call runs sentence-transformers and is CPU-bound (tens of
        # seconds per paper). It runs SYNCHRONOUSLY in the event loop
        # (not via asyncio.to_thread) specifically because wrapping it in
        # to_thread created a cancellation race: if the outer task was
        # cancelled during the await, the worker thread kept using conn
        # while the outer `finally: conn.close()` fired. process_tex is a
        # manual user tool invoked infrequently, so briefly blocking the
        # MCP event loop is acceptable in exchange for atomicity.
        _store_tex_text(pid, text, conn=conn, local_tex_path=str(path), commit=False)

        # Verify only when the row had prior identity. Rows we just created from
        # the TeX stem have no target DOI/arxiv_id/title to check against.
        if row_was_preexisting:
            meta = conn.execute(
                "SELECT doi, arxiv_id, title, authors FROM papers WHERE paper_id = ?", (pid,)
            ).fetchone()
            target_doi, target_arxiv, target_title, target_authors = (meta or (None, None, None, None))
            # 2026-04-26: DOI is no longer a gate. Run only when title/arxiv/authors present.
            if target_arxiv or target_title or target_authors:
                try:
                    ok, reason = _verify_paper_text_match(
                        conn, pid,
                        target_doi=target_doi,
                        target_arxiv_id=target_arxiv,
                        target_title=target_title,
                        target_authors=target_authors,
                        text_source="tex_text",
                    )
                except Exception as e:
                    ok, reason = False, f"verifier raised {type(e).__name__}: {e}"
                if not ok:
                    try:
                        _clear_paper_text(
                            conn, pid,
                            clear_processed_text=False,
                            clear_tex_text=True,
                            clear_tex_path=True,
                        )
                        conn.commit()
                    except Exception as clear_e:
                        # Double fault: clear itself raised. Roll back the
                        # staged store so wrong tex_text doesn't become
                        # durable. Worst case we lose both the store and the
                        # clear attempt; DB returns to pre-call state.
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        return (
                            f"Error: TeX verify failed for {pid}: {reason}; "
                            f"clear also failed ({type(clear_e).__name__}: {clear_e}); "
                            f"rolled back store"
                        )
                    return f"Error: TeX verify failed for {pid}: {reason}"

        orphan_row = conn.execute(
            "SELECT title, authors, year FROM papers WHERE paper_id = ?", (pid,)
        ).fetchone()
        orphan_flag = False
        if orphan_row:
            otitle, oauthors, oyear = orphan_row
            orphan_flag = _is_orphan_paper(pid, otitle, oauthors, oyear)
        abstract_warning = _check_and_flag_missing_abstract(conn, pid)
        conn.commit()
    finally:
        conn.close()

    char_count = len(text)
    word_count = len(text.split())
    result = (
        f"Processed TeX: {path.name}\n"
        f"Stored as: {pid}\n"
        f"Extracted: {char_count:,} chars, ~{word_count:,} words (math preserved)\n"
        f"TeX source: {path}\n"
        f"get_full_text will now return TeX-sourced text for this paper."
    )
    if orphan_flag:
        result += "\n" + _metadata_needed_message(pid, str(path), is_tex=True)
    if abstract_warning:
        result += "\n" + abstract_warning
    return result


@mcp.tool()
async def fix_orphan_paper(
    orphan_id: str,
    canonical_paper_id: str = "",
    title: str = "",
    authors: str = "",
    year: int = 0,
    doi: str = "",
) -> str:
    """Fix a paper that was stored as a generic `local:Unknown_undated_*` orphan.

    Two modes:

    1. **Merge mode** — pass `canonical_paper_id` (an S2 ID found via match_paper_by_title
       or search_papers). The orphan's full text (processed_text, tex_text), file paths
       (local_pdf_path, local_tex_path), chunks, and embeddings are copied into the
       canonical record, and the orphan entry is deleted. The PDF file is renamed to
       the canonical Author_Year_Title.pdf name in PAPERS_DIR.

    2. **In-place update mode** — pass `title`, `authors`, `year`, and optionally `doi`.
       The orphan entry's metadata is updated with the extracted values, and the file is
       renamed to canonical form. The paper_id stays `local:*` because there's no
       matching S2 record.

    Typical flow: process_pdf returns METADATA_NEEDED → haiku subagent reads PDF first
    few pages → extracts title/authors/year → calls match_paper_by_title → on match,
    calls this tool in merge mode; on no match, in in-place update mode.

    Args:
        orphan_id: Paper ID of the orphan entry (must start with 'local:').
        canonical_paper_id: S2 ID of the matched canonical record (merge mode).
        title: Extracted title for in-place update mode.
        authors: JSON array string of author names, e.g. '["John Smith","Jane Doe"]'.
        year: Publication year for in-place update mode.
        doi: DOI for in-place update mode (optional).
    """
    if not orphan_id.startswith("local:"):
        return f"Error: orphan_id must start with 'local:'. Got: {orphan_id}"

    conn = _init_db()
    try:
        orphan = conn.execute(
            "SELECT title, local_pdf_path, local_tex_path, processed_text, tex_text, authors, year "
            "FROM papers WHERE paper_id = ?",
            (orphan_id,),
        ).fetchone()
        if not orphan:
            return f"Error: orphan paper not found: {orphan_id}"
        o_title, o_pdf, o_tex, o_ptext, o_ttext, _o_authors, _o_year = orphan

        # --- Mode 1: Merge into canonical record ---
        if canonical_paper_id:
            canonical = conn.execute(
                "SELECT paper_id, title, authors, year, local_pdf_path FROM papers WHERE paper_id = ?",
                (canonical_paper_id,),
            ).fetchone()
            if not canonical:
                return f"Error: canonical_paper_id not found: {canonical_paper_id}"
            c_pid, c_title, c_authors, c_year, c_existing_pdf = canonical

            # Copy orphan's full text and paths to canonical (only fields where canonical is empty)
            updates = ["verified = 1", "last_updated = ?"]
            params: list = [datetime.now(timezone.utc).isoformat()]

            if o_pdf:
                updates.append("local_pdf_path = COALESCE(local_pdf_path, ?)")
                params.append(o_pdf)
            if o_tex:
                updates.append("local_tex_path = COALESCE(local_tex_path, ?)")
                params.append(o_tex)
            if o_ptext:
                updates.append("processed_text = COALESCE(NULLIF(processed_text, ''), ?)")
                params.append(o_ptext)
            if o_ttext:
                updates.append("tex_text = COALESCE(NULLIF(tex_text, ''), ?)")
                params.append(o_ttext)

            params.append(c_pid)
            conn.execute(
                f"UPDATE papers SET {', '.join(updates)} WHERE paper_id = ?",
                params,
            )
            # Keep has_full_text in sync after the COALESCE-merge: if orphan
            # contributed text to a previously text-less canonical row, the
            # flag must flip to 1 now. Otherwise search_passages and the
            # chunk-mode eligibility filter miss the merged paper.
            _recompute_has_full_text(conn, c_pid)

            # Move chunks and embeddings from orphan to canonical (if any).
            # First, delete any existing chunks/vec on canonical so we don't MIX
            # two chunk sets under one paper_id (text/chunk drift). The canonical's
            # text is the COALESCE-merged text (canonical preferred), so its old
            # chunks may be stale relative to orphan's text if orphan's text was
            # adopted. Rather than depend on which text won, re-derive chunks from
            # the current papers.processed_text after remap.
            # Round-2 review HIGH fix: clear ALL vec-chunk variants
            # (was nomic-only). Use the centralized helper from earlier
            # in this file.
            _delete_vec_chunks_for_paper(conn, c_pid)
            conn.execute("DELETE FROM paper_chunks WHERE paper_id = ?", (c_pid,))
            _delete_vec_chunks_for_paper(conn, orphan_id)
            conn.execute("DELETE FROM paper_chunks WHERE paper_id = ?", (orphan_id,))

            # PDF handling is deferred until AFTER the DB merge commits so
            # a commit failure (SQLITE_FULL, lock timeout) can't leave the
            # filesystem changed while the DB rolls back. We compute the
            # intended rename target here but perform the actual rename /
            # unlink below the first commit.
            renamed_pdf = None
            pending_rename_target: Path | None = None
            pending_unlink: Path | None = None
            if o_pdf and not c_existing_pdf:
                canonical_name = _canonical_pdf_name(c_title or "", c_authors, c_year)
                target = PAPERS_DIR / canonical_name
                counter = 1
                while target.exists() and str(target) != o_pdf:
                    target = PAPERS_DIR / f"{canonical_name[:-4]}_{counter}.pdf"
                    counter += 1
                if str(target) != o_pdf:
                    pending_rename_target = target
            elif o_pdf and c_existing_pdf:
                # Canonical already has a PDF. Delete orphan's copy to avoid
                # disk clutter with an untracked duplicate file.
                pending_unlink = Path(o_pdf)

            # Rewrite any paper_references rows that reference the orphan so
            # edges survive the merge. Both citing and cited columns may need
            # rewriting. Use a two-step UPDATE via a temp table to avoid
            # primary-key collisions (if an edge both cites canonical and is
            # cited by it, the rewrite would otherwise conflict with itself).
            try:
                # Rewrite citing_paper_id
                conn.execute(
                    """
                    INSERT OR IGNORE INTO paper_references
                    (citing_paper_id, cited_paper_id, cited_doi, source, is_influential, first_seen)
                    SELECT ?, cited_paper_id, cited_doi, source, is_influential, first_seen
                    FROM paper_references
                    WHERE citing_paper_id = ?
                      AND cited_paper_id != ?
                    """,
                    (c_pid, orphan_id, c_pid),
                )
                conn.execute(
                    "DELETE FROM paper_references WHERE citing_paper_id = ?",
                    (orphan_id,),
                )
                # Rewrite cited_paper_id
                conn.execute(
                    """
                    INSERT OR IGNORE INTO paper_references
                    (citing_paper_id, cited_paper_id, cited_doi, source, is_influential, first_seen)
                    SELECT citing_paper_id, ?, cited_doi, source, is_influential, first_seen
                    FROM paper_references
                    WHERE cited_paper_id = ?
                      AND citing_paper_id != ?
                    """,
                    (c_pid, orphan_id, c_pid),
                )
                conn.execute(
                    "DELETE FROM paper_references WHERE cited_paper_id = ?",
                    (orphan_id,),
                )
            except sqlite3.OperationalError:
                # paper_references table missing (pre-v6) — skip silently
                pass

            # Round-2 review HIGH fix: clear paper-level vec for ALL
            # variants before deleting the papers row. Otherwise stale
            # qwen3 vec rows outlive the papers row and could corrupt
            # a future insert that reuses the rowid.
            orphan_row = conn.execute(
                "SELECT rowid FROM papers WHERE paper_id = ?", (orphan_id,)
            ).fetchone()
            if orphan_row:
                _delete_vec_papers_for_paper(conn, orphan_row[0])
            # Delete the orphan entry
            conn.execute("DELETE FROM papers WHERE paper_id = ?", (orphan_id,))

            # Re-derive chunks synchronously (no await, so no cancellation
            # point) then commit the whole merge + FS ops as one durable
            # step BEFORE any CPU-bound await. Previous asyncio.shield
            # approach had a race: on cancellation, the outer finally's
            # conn.close() ran concurrently with the shielded coroutine
            # still using conn. Now: sync commit happens first, then
            # embeds run in to_thread. A cancel during embed leaves DB
            # consistent (merge + FS durable) but embeds pending, which
            # is recoverable via backfill.
            merged_text_row = conn.execute(
                "SELECT processed_text, tex_text FROM papers WHERE paper_id = ?",
                (c_pid,),
            ).fetchone()
            canonical_text = ""
            if merged_text_row:
                canonical_text = merged_text_row[1] or merged_text_row[0] or ""

            if canonical_text and len(canonical_text) > 500:
                _chunk_processed_text(conn, c_pid, canonical_text)
                # Rebuild page-bounded passages from the merged canonical
                # text. Without this, the canonical row would carry chunks
                # derived from the new text but passages from the old body.
                _build_paper_passages(conn, c_pid, canonical_text)

            # Commit the DB-only merge FIRST (no FS mutations yet). If
            # this commit fails, the rename/unlink is still pending and
            # can be discarded. If it succeeds, the FS operation below
            # is paired with committed DB state.
            conn.commit()

            # Now apply the pending FS operation. A rename failure here
            # leaves the DB pointing at o_pdf (which still exists) so
            # the merge remains consistent. A rename success is paired
            # with a second commit that updates local_pdf_path to the
            # new location. If that second commit fails, attempt a
            # rename-back so DB and FS don't diverge.
            if pending_rename_target is not None:
                rename_done = False
                try:
                    Path(o_pdf).rename(pending_rename_target)
                    rename_done = True
                    renamed_pdf = str(pending_rename_target)
                    conn.execute(
                        "UPDATE papers SET local_pdf_path = ? WHERE paper_id = ?",
                        (renamed_pdf, c_pid),
                    )
                    conn.commit()
                except (FileNotFoundError, OSError) as e:
                    print(f"PDF rename failed: {e}", file=sys.stderr)
                except Exception as commit_err:
                    # UPDATE/commit failed AFTER the FS rename succeeded.
                    # Best-effort rename-back so DB (still at o_pdf via
                    # pre-commit state) and FS agree.
                    if rename_done:
                        try:
                            Path(pending_rename_target).rename(o_pdf)
                            renamed_pdf = None
                        except Exception as move_back_err:
                            print(
                                f"[CRITICAL] fix_orphan_paper merge: commit "
                                f"failed AND rename-back failed for {c_pid}: "
                                f"file at {pending_rename_target}, DB points "
                                f"to {o_pdf}. commit={commit_err}; "
                                f"move_back={move_back_err}",
                                file=sys.stderr,
                            )
                    raise
            elif pending_unlink is not None:
                # Unlink is safe post-commit: canonical already has its PDF,
                # we're only removing the orphan's duplicate copy.
                try:
                    pending_unlink.unlink()
                except (FileNotFoundError, OSError) as e:
                    print(f"Orphan PDF cleanup failed: {e}", file=sys.stderr)

            # Close the handler conn so it can't race the worker thread.
            conn.close()
            conn = None

            # CPU-bound embed in a worker thread that owns its own conn.
            # Cancel here does not undo the merge; it only leaves vectors
            # pending (recoverable via backfill).
            embed_err: str | None = None
            try:
                embed_err = await asyncio.to_thread(_worker_embed_paper, c_pid)
            except (asyncio.CancelledError, Exception) as e:
                if isinstance(e, asyncio.CancelledError):
                    raise
                embed_err = f"{type(e).__name__}: {e}"
                print(f"Post-merge embed failed for {c_pid}: {e}", file=sys.stderr)

            tail = (
                "Orphan entry deleted. Canonical record now has full text and embeddings."
                if not embed_err
                else f"Orphan entry deleted. Re-embed deferred ({embed_err}); run backfill_embeddings.py."
            )
            return (
                f"Merged orphan {orphan_id} → {c_pid}\n"
                f"Title: {c_title}\n"
                f"PDF: {renamed_pdf or o_pdf or '(none)'}\n"
                f"{tail}"
            )

        # --- Mode 2: Update orphan in place ---
        if not title and not authors and not year:
            return (
                "Error: provide either canonical_paper_id (merge mode) or "
                "title/authors/year (in-place update mode)."
            )

        # Parse authors JSON if provided
        authors_json: str | None = None
        if authors:
            try:
                json.loads(authors)  # validate
                authors_json = authors
            except json.JSONDecodeError:
                # Accept a plain comma-separated list and convert
                parts = [a.strip() for a in authors.split(",") if a.strip()]
                authors_json = json.dumps(parts)

        updates = ["verified = 1", "last_updated = ?"]
        params = [datetime.now(timezone.utc).isoformat()]
        if title:
            updates.append("title = ?")
            params.append(title)
        if authors_json:
            updates.append("authors = ?")
            params.append(authors_json)
        if year:
            updates.append("year = ?")
            params.append(int(year))
        if doi:
            updates.append("doi = ?")
            params.append(doi)
        params.append(orphan_id)
        conn.execute(
            f"UPDATE papers SET {', '.join(updates)} WHERE paper_id = ?",
            params,
        )

        # Heal any stale has_full_text on this row. In-place mode doesn't
        # write text fields, but if a prior path set the flag inconsistently
        # (e.g. pre-Phase-2 store), _recompute_has_full_text is a no-op-safe
        # heal. The no-op predicate in the helper avoids firing papers_au
        # if nothing actually changes.
        _recompute_has_full_text(conn, orphan_id)

        # Compute the intended canonical rename target, but defer the
        # actual rename until AFTER the first commit so a commit failure
        # can't leave the filesystem changed while the DB rolls back.
        renamed_pdf = None
        pending_rename_target: Path | None = None
        if o_pdf:
            canonical_name = _canonical_pdf_name(
                title or o_title or "", authors_json, int(year) if year else None,
            )
            target = PAPERS_DIR / canonical_name
            counter = 1
            while target.exists() and str(target) != o_pdf:
                target = PAPERS_DIR / f"{canonical_name[:-4]}_{counter}.pdf"
                counter += 1
            if str(target) != o_pdf:
                pending_rename_target = target

        # First commit: row metadata only, no FS mutation yet.
        conn.commit()

        # Now apply the FS rename. A rename failure leaves the DB pointing
        # at o_pdf (which still exists); rename success is paired with a
        # second commit to update local_pdf_path. If that second commit
        # fails, best-effort rename-back so DB and FS stay consistent.
        if pending_rename_target is not None:
            rename_done = False
            try:
                Path(o_pdf).rename(pending_rename_target)
                rename_done = True
                renamed_pdf = str(pending_rename_target)
                conn.execute(
                    "UPDATE papers SET local_pdf_path = ? WHERE paper_id = ?",
                    (renamed_pdf, orphan_id),
                )
                conn.commit()
            except (FileNotFoundError, OSError) as e:
                print(f"PDF rename failed: {e}", file=sys.stderr)
            except Exception as commit_err:
                if rename_done:
                    try:
                        Path(pending_rename_target).rename(o_pdf)
                        renamed_pdf = None
                    except Exception as move_back_err:
                        print(
                            f"[CRITICAL] fix_orphan_paper update: commit "
                            f"failed AND rename-back failed for {orphan_id}: "
                            f"file at {pending_rename_target}, DB points "
                            f"to {o_pdf}. commit={commit_err}; "
                            f"move_back={move_back_err}",
                            file=sys.stderr,
                        )
                raise

        # Close handler conn BEFORE the CPU-bound embed await so the outer
        # finally can't race conn.close() against the worker.
        conn.close()
        conn = None
        embed_err: str | None = None
        try:
            embed_err = await asyncio.to_thread(_worker_embed_paper, orphan_id)
        except (asyncio.CancelledError, Exception) as e:
            if isinstance(e, asyncio.CancelledError):
                raise
            embed_err = f"{type(e).__name__}: {e}"
            print(f"Post-update embed failed for {orphan_id}: {e}", file=sys.stderr)

        embed_line = (
            "" if not embed_err
            else f"\nEmbedding deferred ({embed_err}); run backfill_embeddings.py."
        )
        return (
            f"Updated orphan {orphan_id} in place\n"
            f"Title: {title or o_title}\n"
            f"Authors: {authors_json or '(unchanged)'}\n"
            f"Year: {year or '(unchanged)'}\n"
            f"DOI: {doi or '(none)'}\n"
            f"PDF: {renamed_pdf or o_pdf or '(none)'}"
            + embed_line
        )
    finally:
        # conn may have been closed explicitly BEFORE the embed await
        # (see the `conn.close(); conn = None` pattern above) so a finally
        # close on None or on an already-closed conn should be a no-op.
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


@mcp.tool()
async def search_openalex(
    query: str,
    limit: int = 10,
    min_citations: int = 0,
    work_type: str = "",
    sort_by: str = "relevance",
    year_range: str = "",
) -> str:
    """Search academic papers via OpenAlex with authority metrics (FWCI, venue impact factor, h-index).

    Returns papers with computed authority tiers (1-4) based on real bibliometric data.
    Use this to find high-quality, authoritative sources. Results saved to local library.

    Args:
        query: Search query (e.g., "order book microstructure dynamics")
        limit: Max results (1-50, default 10)
        min_citations: Minimum cited_by_count filter
        work_type: Filter by type: article, review, preprint, book-chapter, dissertation, etc.
        sort_by: Sort order: "relevance" (default), "cited_by_count", "publication_date"
        year_range: Filter by year, e.g. "2020-2024" or "2020-" or "-2024"
    """
    limit = max(1, min(50, limit))
    params: dict = {
        "search": query,
        "per_page": limit,
        "select": ",".join([
            "id", "doi", "display_name", "title", "publication_year", "publication_date",
            "type", "cited_by_count", "fwci", "citation_normalized_percentile",
            "is_retracted", "open_access", "authorships", "primary_location",
            "topics", "abstract_inverted_index", "ids", "referenced_works",
        ]),
    }
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    elif OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO

    # Build filters
    filters = []
    if min_citations > 0:
        filters.append(f"cited_by_count:>{min_citations}")
    if work_type:
        filters.append(f"type:{work_type}")
    if year_range:
        if "-" in year_range:
            parts = year_range.split("-")
            if parts[0] and parts[1]:
                filters.append(f"publication_year:{parts[0]}-{parts[1]}")
            elif parts[0]:
                filters.append(f"from_publication_date:{parts[0]}-01-01")
            elif parts[1]:
                filters.append(f"to_publication_date:{parts[1]}-12-31")
    if filters:
        params["filter"] = ",".join(filters)

    # Sort
    if sort_by == "cited_by_count":
        params["sort"] = "cited_by_count:desc"
    elif sort_by == "publication_date":
        params["sort"] = "publication_date:desc"

    async with httpx.AsyncClient(timeout=30) as client:
        await _openalex_throttle.wait()
        resp = await client.get(f"{OA_BASE}/works", params=params)
        resp.raise_for_status()
        data = resp.json()

        works = data.get("results", [])
        total = data.get("meta", {}).get("count", 0)

        # Enrich works with venue quality data (one batch API call)
        source_ids = set()
        for w in works:
            loc = w.get("primary_location") or {}
            src = loc.get("source") or {}
            sid = src.get("id", "")
            if sid:
                source_ids.add(sid)

        venue_stats: dict = {}
        if source_ids:
            # Extract short IDs (S123456) from full URLs for filter
            short_ids = [sid.split("/")[-1] for sid in source_ids if "/" in sid]
            if not short_ids:
                short_ids = list(source_ids)
            oa_filter = "|".join(short_ids)
            src_params: dict = {
                "filter": f"openalex:{oa_filter}",
                "per_page": len(short_ids),
                "select": "id,summary_stats",
            }
            if OPENALEX_API_KEY:
                src_params["api_key"] = OPENALEX_API_KEY
            elif OPENALEX_MAILTO:
                src_params["mailto"] = OPENALEX_MAILTO
            try:
                await _openalex_throttle.wait()
                src_resp = await client.get(f"{OA_BASE}/sources", params=src_params)
                if src_resp.status_code == 200:
                    for s in src_resp.json().get("results", []):
                        venue_stats[s["id"]] = s.get("summary_stats") or {}
            except Exception:
                pass  # venue enrichment is best-effort

        # Inject venue stats back into work objects for display and storage
        for w in works:
            loc = w.get("primary_location") or {}
            src = loc.get("source") or {}
            sid = src.get("id", "")
            if sid and sid in venue_stats:
                src["summary_stats"] = venue_stats[sid]

    # Store in local library (now with venue stats)
    conn = _init_db()
    try:
        for w in works:
            _store_openalex_work(conn, w)
        conn.commit()
    finally:
        conn.close()

    if not works:
        return f"No OpenAlex results for '{query}'"

    parts = [f"Found {total:,} results (showing {len(works)}, all saved to local library):\n"]
    for i, w in enumerate(works, 1):
        parts.append(f"--- Result {i} ---")
        parts.append(_format_openalex_work(w))
        parts.append("")

    return "\n".join(parts)


@mcp.tool()
async def get_venue_quality(venue_name: str) -> str:
    """Look up a journal/conference/source quality metrics from OpenAlex.

    Returns impact factor (2yr mean citedness), h-index, i10-index, works count,
    cited-by count, and type. Use this to assess whether a venue is top-tier.

    Args:
        venue_name: Name of journal, conference, or source (e.g., "Nature", "NeurIPS", "Physical Review Letters")
    """
    params: dict = {
        "search": venue_name,
        "per_page": 5,
        "select": "id,display_name,type,is_in_doaj,is_oa,is_core,works_count,cited_by_count,summary_stats,host_organization_name,country_code,issns",
    }
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    elif OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO

    async with httpx.AsyncClient(timeout=30) as client:
        await _openalex_throttle.wait()
        resp = await client.get(f"{OA_BASE}/sources", params=params)
        resp.raise_for_status()
        data = resp.json()

    sources = data.get("results", [])
    if not sources:
        return f"No sources found for '{venue_name}'"

    parts = [f"Found {len(sources)} sources matching '{venue_name}':\n"]
    for i, s in enumerate(sources, 1):
        name = s.get("display_name", "Unknown")
        stype = s.get("type", "unknown")
        stats = s.get("summary_stats") or {}
        impact_factor = stats.get("2yr_mean_citedness")
        h_index = stats.get("h_index")
        i10_index = stats.get("i10_index")
        works = s.get("works_count", 0)
        cited = s.get("cited_by_count", 0)
        is_core = s.get("is_core", False)
        host = s.get("host_organization_name", "")

        parts.append(f"--- Source {i} ---")
        parts.append(f"**{name}**")
        parts.append(f"Type: {stype}")
        if host:
            parts.append(f"Publisher: {host}")
        if impact_factor is not None:
            parts.append(f"Impact Factor (2yr mean citedness): {impact_factor:.2f}")
        if h_index is not None:
            parts.append(f"h-index: {h_index}")
        if i10_index is not None:
            parts.append(f"i10-index: {i10_index}")
        parts.append(f"Works: {works:,} | Cited by: {cited:,}")
        if is_core:
            parts.append("CWTS Core Source: Yes (used in Open Leiden Ranking)")
        if s.get("is_in_doaj"):
            parts.append("Listed in DOAJ (Directory of Open Access Journals)")

        # Quality assessment
        quality = []
        if impact_factor is not None:
            if impact_factor >= 10:
                quality.append("Elite venue (IF >= 10)")
            elif impact_factor >= 3:
                quality.append("Strong venue (IF >= 3)")
            elif impact_factor >= 1:
                quality.append("Solid venue (IF >= 1)")
            else:
                quality.append(f"Low impact (IF < 1)")
        if h_index is not None:
            if h_index >= 200:
                quality.append("Top-tier h-index")
            elif h_index >= 50:
                quality.append("Good h-index")
        if quality:
            parts.append(f"Assessment: {' | '.join(quality)}")
        parts.append("")

    return "\n".join(parts)


@mcp.tool()
async def check_jstor(
    title: str = "",
    issn: str = "",
    limit: int = 10,
) -> str:
    """Check if papers are available in JSTOR's corpus (11.8M articles).

    Use this during paper acquisition: when a paper is paywalled and can't be
    downloaded via S2/Unpaywall, check if JSTOR has it. If found, include the
    JSTOR URL in a request list to submit to JSTOR Text Analysis (institutional
    access required for full text).

    Search by title (FTS5 full-text search) and/or ISSN (exact match on journal).
    At least one of title or issn must be provided.

    Args:
        title: Paper title or keywords to search (uses FTS5 full-text matching)
        issn: Journal ISSN to filter by (matches print or online ISSN, digits only)
        limit: Max results (default 10)
    """
    if not title and not issn:
        return "Error: provide at least one of title or issn"

    if not JSTOR_DB_PATH.exists():
        return "JSTOR database not found. Run: uv run build_jstor_db.py <path_to_jsonl>"

    conn = sqlite3.connect(str(JSTOR_DB_PATH))
    conn.row_factory = sqlite3.Row
    def _jstor_fts_safe(sql: str, fts_query: str, extra_params: tuple = ()) -> list:
        """Run JSTOR FTS5 MATCH query with safe fallback on syntax error."""
        try:
            return conn.execute(sql, (fts_query,) + extra_params).fetchall()
        except sqlite3.OperationalError:
            escaped = '"' + fts_query.replace('"', '""') + '"'
            try:
                return conn.execute(sql, (escaped,) + extra_params).fetchall()
            except sqlite3.OperationalError:
                return []

    try:
        results = []
        if title and issn:
            # FTS5 search filtered by ISSN
            clean_issn = re.sub(r"[^0-9Xx]", "", issn)
            results = _jstor_fts_safe("""
                SELECT a.*, rank FROM jstor_fts f
                JOIN jstor_articles a ON a.rowid = f.rowid
                WHERE jstor_fts MATCH ? AND (a.print_issn = ? OR a.online_issn = ?)
                ORDER BY rank LIMIT ?
            """, title, (clean_issn, clean_issn, limit))
        elif title:
            results = _jstor_fts_safe("""
                SELECT a.*, rank FROM jstor_fts f
                JOIN jstor_articles a ON a.rowid = f.rowid
                WHERE jstor_fts MATCH ?
                ORDER BY rank LIMIT ?
            """, title, (limit,))
        elif issn:
            clean_issn = re.sub(r"[^0-9Xx]", "", issn)
            rows = conn.execute("""
                SELECT * FROM jstor_articles
                WHERE print_issn = ? OR online_issn = ?
                ORDER BY published_date DESC LIMIT ?
            """, (clean_issn, clean_issn, limit)).fetchall()
            results = rows

        if not results:
            return f"Not found in JSTOR. Title: '{title}', ISSN: '{issn}'"

        parts = [f"Found {len(results)} match(es) in JSTOR:\n"]
        for i, r in enumerate(results, 1):
            url = r["url"]
            if url and not url.startswith("http"):
                url = f"https://{url}"
            parts.append(f"--- Match {i} ---")
            parts.append(f"**{r['title']}**")
            if r["creators"]:
                parts.append(f"Authors: {r['creators']}")
            if r["journal"]:
                parts.append(f"Journal: {r['journal']}")
            if r["published_date"]:
                parts.append(f"Date: {r['published_date']}")
            if r["disciplines"]:
                try:
                    discs = json.loads(r["disciplines"])
                    if discs:
                        parts.append(f"Disciplines: {', '.join(discs)}")
                except (json.JSONDecodeError, TypeError):
                    pass
            parts.append(f"Subtype: {r['content_subtype']}")
            if url:
                parts.append(f"JSTOR URL: {url}")
            if r["ithaka_doi"]:
                parts.append(f"JSTOR DOI: {r['ithaka_doi']}")
            parts.append("")

        return "\n".join(parts)
    finally:
        conn.close()


# --- PhilPapers API tools ---

def _pp_params() -> dict:
    """Base query params with PhilPapers API auth."""
    p: dict = {}
    if PHILPAPERS_API_ID:
        p["apiId"] = PHILPAPERS_API_ID
    if PHILPAPERS_API_KEY:
        p["apiKey"] = PHILPAPERS_API_KEY
    return p


def _pp_parse_entry(item: dict) -> str:
    """Parse a PhilPapers browse-result item into a readable string."""
    entry_html = item.get("__entry__", "")
    # Extract title
    m = re.search(r"recTitle[^>]*>(.*?)</span>", entry_html)
    title = m.group(1) if m else item.get("id", "?")
    # Extract year
    m = re.search(r'pubYear[^>]*>(\d{4}|forthcoming)</span>', entry_html)
    year = m.group(1) if m else ""
    authors = item.get("authors", [])
    author_str = ", ".join(authors[:4])
    if len(authors) > 4:
        author_str += f" (+{len(authors) - 4})"
    abstract = (item.get("excerpt1") or "").strip()
    if not abstract:
        abstract = (item.get("excerpt2") or "").strip()
    # Extract categories
    cat_names = re.findall(r"catName[^>]*>([^<]+)", item.get("catsHTML", ""))
    # Extract direct link URL
    link_m = re.search(r'href="([^"]+)"', item.get("directLink", ""))
    link = link_m.group(1) if link_m else ""
    pid = item.get("id", "")

    lines = [f"**{title}** ({year})" if year else f"**{title}**"]
    if author_str:
        lines.append(f"Authors: {author_str}")
    lines.append(f"PhilPapers ID: {pid}")
    if cat_names:
        lines.append(f"Categories: {', '.join(cat_names[:5])}")
    if abstract:
        lines.append(f"Abstract: {abstract[:400]}{'...' if len(abstract) > 400 else ''}")
    if link:
        lines.append(f"Link: {link}")
    return "\n".join(lines)


@mcp.tool()
async def search_philpapers(query: str, limit: int = 20) -> str:
    """Search PhilPapers by keyword. Returns matching philosophy papers.

    PhilPapers indexes ~500K philosophy papers with curated category assignments.
    Use for finding philosophy-of-science, philosophy-of-language, phenomenology,
    and other philosophical literature that S2/OpenAlex index poorly.

    Args:
        query: Search keywords (e.g., "Heidegger technology enframing")
        limit: Max results (default 20)
    """
    if not query or len(query.strip()) < 2:
        return "Error: query must be at least 2 characters"
    if not PHILPAPERS_API_ID:
        return "Error: PHILPAPERS_API_ID not set. Export it in your shell environment."

    params = _pp_params()
    params.update({"searchStr": query.strip(), "filterMode": "keywords", "format": "json"})

    await _philpapers_throttle.wait()
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": BROWSER_UA}) as client:
        try:
            resp = await client.get("https://philpapers.org/asearch.pl", params=params)
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            return f"Error: PhilPapers search failed ({e})"

    try:
        data = resp.json()
    except Exception:
        return f"Error: PhilPapers returned non-JSON (status {resp.status_code})"

    items = data.get("data", [])
    if not items:
        return f"No PhilPapers results for '{query}'"

    total = items[0].get("count", "?") if items else "?"
    parts = [f"PhilPapers search: {len(items)} results (of {total} total) for '{query}':\n"]
    for i, item in enumerate(items[:limit], 1):
        name_html = item.get("name", "")
        m = re.search(r"recTitle[^>]*>(.*?)</span>", name_html)
        title = m.group(1) if m else item.get("id", "?")
        pid = item.get("id", "")
        url = f"https://philpapers.org{item.get('url', '')}"
        parts.append(f"--- {i}. {title} ---")
        parts.append(f"PhilPapers ID: {pid}")
        parts.append(f"URL: {url}")
        parts.append("")

    return "\n".join(parts)


@mcp.tool()
async def browse_philpapers(category: str, limit: int = 20, start: int = 0) -> str:
    """Browse a curated PhilPapers category. Returns papers with full metadata.

    PhilPapers organizes philosophy into ~2000 curated categories. Each paper
    is assigned to relevant categories by professional philosophers. This is
    the primary way to find canonical literature in a philosophical subdiscipline.

    Common categories: speech-acts, philosophy-of-technology, phenomenology,
    scientific-models, philosophy-of-economics, philosophy-of-science,
    philosophy-of-language, philosophy-of-mind, metaphysics, epistemology,
    philosophy-of-physics, philosophy-of-action, social-ontology.

    Args:
        category: Category slug (e.g., "speech-acts", "philosophy-of-technology").
                  Use list_philpapers_categories to discover available categories.
        limit: Max results (default 20, max 50)
        start: Offset for pagination (default 0)
    """
    if not category or len(category.strip()) < 2:
        return "Error: category slug required"
    if not PHILPAPERS_API_ID:
        return "Error: PHILPAPERS_API_ID not set. Export it in your shell environment."

    params = _pp_params()
    params.update({"format": "json", "limit": min(limit, 50), "start": start})

    await _philpapers_throttle.wait()
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": BROWSER_UA}) as client:
        try:
            resp = await client.get(
                f"https://philpapers.org/browse/{category.strip()}", params=params
            )
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            return f"Error: PhilPapers browse failed ({e})"

    try:
        data = resp.json()
    except Exception:
        if "Category not found" in resp.text:
            return f"Error: category '{category}' not found. Use list_philpapers_categories to find valid slugs."
        return f"Error: PhilPapers returned non-JSON (status {resp.status_code})"

    total = data.get("found", "?")
    content_blocks = data.get("content", [])
    items = content_blocks[0].get("content", []) if content_blocks else []

    if not items:
        return f"No papers in PhilPapers category '{category}'"

    parts = [f"PhilPapers category '{category}': {len(items)} papers (of {total} total):\n"]
    for i, item in enumerate(items[:limit], 1):
        parts.append(f"--- {i} ---")
        parts.append(_pp_parse_entry(item))
        parts.append("")

    return "\n".join(parts)


@mcp.tool()
async def list_philpapers_categories(parent: str = "") -> str:
    """List PhilPapers subcategories within a parent category.

    Returns available category slugs that can be passed to browse_philpapers.
    Discovers subcategories by browsing the parent and extracting category
    assignments from the returned papers.

    Args:
        parent: Parent category slug (e.g., "philosophy-of-language"). Required.
    """
    if not parent or len(parent.strip()) < 2:
        return "Error: parent category slug required (e.g., 'philosophy-of-language')"
    if not PHILPAPERS_API_ID:
        return "Error: PHILPAPERS_API_ID not set. Export it in your shell environment."

    parent = parent.strip()
    params = _pp_params()
    params.update({"format": "json", "limit": 50})

    await _philpapers_throttle.wait()
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": BROWSER_UA}) as client:
        try:
            resp = await client.get(
                f"https://philpapers.org/browse/{parent}", params=params
            )
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            return f"Error: PhilPapers category listing failed ({e})"

    try:
        data = resp.json()
    except Exception:
        if "Category not found" in resp.text:
            return f"Error: category '{parent}' not found."
        return f"Error: PhilPapers returned non-JSON (status {resp.status_code})"

    total = data.get("found", "?")
    content_blocks = data.get("content", [])
    items = content_blocks[0].get("content", []) if content_blocks else []

    # Extract unique subcategory slugs from papers' catsHTML
    subcats: dict[str, str] = {}  # slug → display name
    for item in items:
        cats_html = item.get("catsHTML", "")
        # Pattern: <a class='catName' href='/browse/{slug}' rel='section'>{Name}</a>
        for slug, name in re.findall(r"href='/browse/([^']+)'[^>]*rel='section'>([^<]+)", cats_html):
            if slug != parent and slug not in subcats:
                subcats[slug] = name

    if not subcats:
        return f"No subcategories found for '{parent}' (it may be a leaf category)."

    parts = [f"PhilPapers category '{parent}' ({total} papers). Subcategories discovered from {len(items)} sample papers:\n"]
    for slug, name in sorted(subcats.items(), key=lambda x: x[1]):
        parts.append(f"  {slug:50s} {name}")

    return "\n".join(parts)


# arXiv ID patterns: new format (2301.12345, with optional version) and old format (hep-th/0401234)
_ARXIV_NEW_PAT = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_ARXIV_OLD_PAT = re.compile(r"^[a-z][\w.-]*/\d{7}(v\d+)?$")


@mcp.tool()
async def download_arxiv_tex(arxiv_id: str, paper_id: str = "") -> str:
    """Download and process an arXiv paper's TeX source for math-clean full text.

    arXiv provides LaTeX source bundles for most papers. The extracted text
    preserves math notation (via pandoc) and produces higher-quality embeddings
    than PDF extraction for math-heavy content. get_full_text automatically
    prefers TeX-sourced text when available.

    Use after download_paper succeeds for an arXiv paper. TeX is a quality
    upgrade, not a replacement for the PDF. If this tool fails, the paper
    still has its PDF-extracted text.

    Returns either:
        "Downloaded arXiv TeX: {arxiv_id}\\n{process_tex result}"
        "Already has TeX: {paper_id}"
        "NO_TEX_SOURCE: {reason}"

    Args:
        arxiv_id: arXiv identifier (e.g., "2301.12345" or "hep-th/0401234").
        paper_id: Optional canonical S2/local paper ID. If empty, process_tex
                  resolves via ARXIV:{arxiv_id} lookup or creates a local:* row.
    """
    arxiv_id = (arxiv_id or "").strip()
    # Strip version suffix for directory naming but keep for download
    base_id = re.sub(r"v\d+$", "", arxiv_id)
    if not arxiv_id or (not _ARXIV_NEW_PAT.match(arxiv_id) and not _ARXIV_OLD_PAT.match(arxiv_id)):
        return "NO_TEX_SOURCE: bad_id"

    # Check if this paper already has TeX text (check both arxiv_id column and paper_id)
    conn = _init_db()
    try:
        row = conn.execute(
            "SELECT paper_id, tex_text FROM papers WHERE arxiv_id = ? OR paper_id = ?",
            (base_id, f"ARXIV:{base_id}"),
        ).fetchone()
        if row and row[1] and len(row[1].strip()) > 100:
            return f"Already has TeX: {row[0]}"
    finally:
        conn.close()

    # Download the e-print source bundle
    await _arxiv_throttle.wait()
    headers = {"User-Agent": BROWSER_UA}
    async with httpx.AsyncClient(timeout=60, follow_redirects=True, headers=headers) as client:
        try:
            resp = await client.get(f"https://arxiv.org/e-print/{arxiv_id}")
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            return f"NO_TEX_SOURCE: transport_error ({type(e).__name__}: {e})"

    if resp.status_code == 404:
        return "NO_TEX_SOURCE: not_found"
    if resp.status_code != 200:
        return f"NO_TEX_SOURCE: http_{resp.status_code}"

    content = resp.content
    content_type = resp.headers.get("content-type", "")

    # Detect PDF-only papers (no TeX source available)
    if "application/pdf" in content_type or content[:5] == b"%PDF-":
        return "NO_TEX_SOURCE: pdf_only"

    # Extract to TEX_DIR/{base_id}/
    # Use base_id (without version) for directory to avoid duplicate dirs per version.
    # Clean any prior partial extraction to avoid stale files mixing with new ones.
    extract_dir = TEX_DIR / base_id.replace("/", "_")
    if extract_dir.exists():
        shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        if content[:2] == b"\x1f\x8b":  # gzip magic
            # Try as tar.gz first (most common)
            try:
                with tarfile.open(fileobj=BytesIO(content), mode="r:gz") as tar:
                    # Security: filter out absolute paths and path traversal
                    members = [m for m in tar.getmembers()
                               if not m.name.startswith("/") and ".." not in m.name]
                    tar.extractall(path=str(extract_dir), members=members)
            except tarfile.TarError:
                # Single compressed .tex file (not a tar)
                tex_content = gzip.decompress(content)
                (extract_dir / "main.tex").write_bytes(tex_content)
        else:
            # Uncompressed single .tex file (rare)
            (extract_dir / "main.tex").write_bytes(content)
    except Exception as e:
        return f"NO_TEX_SOURCE: extraction_failed ({type(e).__name__}: {e})"

    # Hand off to process_tex for pandoc extraction + DB ingest + warnings.
    resolved_pid = paper_id or f"ARXIV:{base_id}"
    try:
        tex_result = await process_tex(str(extract_dir), paper_id=resolved_pid)
    except Exception as e:
        return f"NO_TEX_SOURCE: process_failed ({type(e).__name__}: {e})"

    # process_tex returns error strings (not exceptions) for most failures.
    if tex_result.startswith("Error") or tex_result.startswith("Warning"):
        return f"NO_TEX_SOURCE: process_failed ({tex_result.splitlines()[0]})"

    return f"Downloaded arXiv TeX: {arxiv_id}\n{tex_result}"


@mcp.tool()
async def search_local(query: str, limit: int = 20, rerank_on: str = "chunk", structured: bool = False) -> str:
    """Search the local paper library using hybrid search (keyword + semantic + reranking).

    Combines FTS5 keyword matching with vector similarity search via Reciprocal
    Rank Fusion, then reranks results with a cross-encoder for precision. Finds
    papers even when your search terms differ from the paper's vocabulary.

    Args:
        query: Search terms (e.g., "order book microstructure", "vitamin D absorption")
        limit: Max results (default 20)
        rerank_on: "chunk" (default) feeds the paper's best-matching chunk text to
            the cross-encoder — best for verbatim-citation queries. "abstract"
            feeds title + first 500 chars of abstract (previous behavior) — better
            for purely topical searches where the query matches the paper's
            overall subject rather than a specific passage.
        structured: If True, return JSON with per-candidate raw signals
            (fts_rank, vec_rank, chunk_rank, hub_rank, rerank position, and
            full metadata). Lets frontier-model callers rerank in their own
            context instead of trusting the cross-encoder's final order.
            Default False preserves the existing markdown output for
            backward compatibility.
    """
    conn = _init_db()
    try:
        # Eligibility filter. rerank_on="chunk" requires a paper to have full
        # text (otherwise chunk-based rerank has nothing to score). rerank_on=
        # "abstract" keeps the original `verified=1` filter so metadata-only
        # papers with abstracts still participate.
        # Both modes now admit verified=1 OR has_full_text=1 (wave-3 fix).
        # Previously abstract-mode gated on verified=1 alone, which excluded
        # has_full_text=1 merge-keepers that should be eligible.
        eligibility_sql = (
            "p.has_full_text = 1"
            if rerank_on == "chunk"
            else "(p.verified = 1 OR p.has_full_text = 1)"
        )

        # 1. FTS5 keyword search with column weights.
        # Weights (positional per papers_fts schema: paper_id, title, abstract,
        # tldr, processed_text, tex_text). Title matches dominate because a
        # verbatim phrase in the title is a far stronger signal than body
        # occurrences. Since schema v13 the body (tex_text || processed_text
        # concatenation) is always stored in the processed_text column and the
        # tex_text column is empty, so its weight is 0.
        # An earlier attempt with weights (0.1, 5.0, 3.0, 3.0, 2.0, 0.0) —
        # dropping title from 10→5 and raising body 1→2 — regressed paper@1
        # on every benchmark domain by 12-50 points. The title weight
        # dominates because author names and key terms routinely appear
        # in titles, and the long-tail book chapters that the lower title
        # weight would help are not the modal query shape.
        fts_results = _fts5_safe_query(conn, f"""
            SELECT p.paper_id, p.rowid
            FROM papers_fts fts
            JOIN papers p ON fts.paper_id = p.paper_id
            WHERE papers_fts MATCH ?
              AND {eligibility_sql}
            ORDER BY bm25(papers_fts, 0.1, 10.0, 3.0, 3.0, 1.0, 0.0)
            LIMIT ?
        """, query, 300)
        fts_ranks = {row[0]: i + 1 for i, row in enumerate(fts_results)}

        # 2a. Vector similarity search on paper-level embeddings.
        # KNN + post-filter: vec_papers is populated for all verified rows
        # (~15.9K), but only ~2.9K of those are has_full_text=1 (18% ratio).
        # Phase 3.2: widen k from 2500 → 5000. The post-eligibility filter
        # reduces ~5000 KNN hits to ~900 cite-ready papers (18% ratio); the
        # prior 2500 → ~450, which clipped the expected paper out of the
        # 300-slot candidate pool on 5 of 49 benchmark claims. Cost: ~100ms
        # per query at CPU-embedded KNN scale.
        vec_ranks = {}
        query_emb = None
        try:
            query_emb = await asyncio.to_thread(_embed_query_text, query)
            vec_papers_t, vec_chunks_t = _vec_table_pair()
            vec_results = conn.execute(f"""
                SELECT v.rowid, v.distance
                FROM {vec_papers_t} v
                WHERE v.embedding MATCH ? AND k = 5000
                ORDER BY v.distance
            """, (query_emb,)).fetchall()
            # Batch-map rowids to paper_ids, filter to the same eligibility as FTS.
            if vec_results:
                vec_rowids = [r[0] for r in vec_results]
                placeholders = ",".join("?" for _ in vec_rowids)
                rowid_to_pid = dict(conn.execute(
                    f"SELECT p.rowid, p.paper_id FROM papers p "
                    f"WHERE p.rowid IN ({placeholders}) AND {eligibility_sql}",
                    vec_rowids,
                ).fetchall())
                # Preserve rank order from KNN; drop rowids that fail the
                # eligibility filter. Assign ranks sequentially over surviving
                # rows so RRF sees a dense 1..N ranking rather than one full
                # of gaps that would weaken the signal against FTS/chunk legs.
                seq_rank = 0
                for rowid, _dist in vec_results:
                    pid = rowid_to_pid.get(rowid)
                    if pid is None:
                        continue
                    seq_rank += 1
                    if seq_rank > 300:
                        break
                    vec_ranks[pid] = seq_rank
        except Exception as e:
            print(f"Vector search error: {e}", file=sys.stderr)

        # 2b. Chunk-level vector search aggregated to papers.
        # For each paper, use its best-matching chunk as a signal. Helps books
        # and legal sources whose paper-level embedding is weak because the
        # abstract is thin or missing but specific passages match the query.
        chunk_agg_ranks: dict[str, int] = {}
        if query_emb is not None:
            try:
                # Phase 3.2: chunk-vec k=1500 → 3000. Chunk-agg needs enough
                # candidates to survive paper-level aggregation; with 121K
                # chunks indexed, k=1500 samples ~1.2% which routinely missed
                # the best chunk of a target paper when that paper's chunks
                # were just-past-the-cutoff on vec distance. k=3000 doubles
                # sampling without affecting latency meaningfully.
                chunk_results = conn.execute(f"""
                    SELECT v.rowid, v.distance
                    FROM {vec_chunks_t} v
                    WHERE v.embedding MATCH ? AND k = 3000
                    ORDER BY v.distance
                """, (query_emb,)).fetchall()
                if chunk_results:
                    chunk_rowids = [r[0] for r in chunk_results]
                    placeholders = ",".join("?" for _ in chunk_rowids)
                    cid_to_pid = dict(conn.execute(f"""
                        SELECT c.chunk_id, c.paper_id
                        FROM paper_chunks c
                        JOIN papers p ON c.paper_id = p.paper_id
                        WHERE c.chunk_id IN ({placeholders}) AND {eligibility_sql}
                    """, chunk_rowids).fetchall())
                    # Aggregate: for each paper, take the best (smallest rank_idx) chunk
                    paper_best_rank: dict[str, int] = {}
                    for rank_idx, (cid, _dist) in enumerate(chunk_results):
                        pid = cid_to_pid.get(cid)
                        if pid is None:
                            continue
                        if pid not in paper_best_rank or rank_idx < paper_best_rank[pid]:
                            paper_best_rank[pid] = rank_idx
                    sorted_pids = sorted(paper_best_rank.items(), key=lambda x: x[1])
                    chunk_agg_ranks = {pid: i + 1 for i, (pid, _) in enumerate(sorted_pids)}
            except Exception as e:
                print(f"Chunk aggregation error: {e}", file=sys.stderr)

        # 3. RRF fusion (keyword + paper-vector + chunk-aggregated). Hub signal
        # added after candidate selection below.
        all_pids = set(fts_ranks.keys()) | set(vec_ranks.keys()) | set(chunk_agg_ranks.keys())
        k = 60
        rrf_scores = {}
        for pid in all_pids:
            score = 0.0
            if pid in fts_ranks:
                score += 1.0 / (k + fts_ranks[pid])
            if pid in vec_ranks:
                score += 1.0 / (k + vec_ranks[pid])
            if pid in chunk_agg_ranks:
                score += 1.0 / (k + chunk_agg_ranks[pid])
            rrf_scores[pid] = score
        # Widened from 50 to 300: cross-encoder (the only scorer that reads
        # query+passage text) is the authoritative relevance signal, and it
        # should see enough candidates to surface passages whose vocabulary
        # doesn't mirror the query's abstract wording.
        ranked_pids = sorted(rrf_scores.keys(), key=lambda p: -rrf_scores[p])[:300]

        if not ranked_pids:
            return f"No local results for '{query}'. Try search_papers to search Semantic Scholar."

        # 3b. Citation-graph hub scoring (weighted hub count over the seed set).
        # For each edge in paper_references where both endpoints are in the
        # top-300 seed set (was 50 before commit 853d96b), each endpoint accrues
        # weight proportional to the other's RRF score. Influential edges
        # (S2 isInfluential) weighted 3x. This is NOT true Personalized PageRank
        # (no out-degree normalization); it's a weighted hub-count that
        # correlates well with topic centrality for re-ranking purposes. Hub
        # contribution is additive to RRF via the standard k=60 rank fusion, so
        # it can only re-order within the top 300, never rescue a paper that
        # failed the initial three signals.
        hub_ranks: dict[str, int] = {}
        if len(ranked_pids) > 1:
            try:
                pid_to_rrf = {pid: rrf_scores[pid] for pid in ranked_pids}
                pid_set = set(ranked_pids)
                placeholders = ",".join("?" for _ in ranked_pids)
                edges = conn.execute(f"""
                    SELECT citing_paper_id, cited_paper_id, is_influential
                    FROM paper_references
                    WHERE citing_paper_id IN ({placeholders})
                       AND cited_paper_id IN ({placeholders})
                """, ranked_pids + ranked_pids).fetchall()
                hub_scores: dict[str, float] = {pid: 0.0 for pid in ranked_pids}
                for citing, cited, influential in edges:
                    if citing in pid_set and cited in pid_set:
                        weight = 3.0 if influential else 1.0
                        hub_scores[cited] += pid_to_rrf[citing] * weight
                        hub_scores[citing] += pid_to_rrf[cited] * weight
                sorted_hub = sorted(
                    [(pid, s) for pid, s in hub_scores.items() if s > 0],
                    key=lambda x: -x[1],
                )
                hub_ranks = {pid: i + 1 for i, (pid, _) in enumerate(sorted_hub)}
                if hub_ranks:
                    # Re-score candidates with the hub signal and re-sort.
                    for pid in ranked_pids:
                        if pid in hub_ranks:
                            rrf_scores[pid] += 1.0 / (k + hub_ranks[pid])
                    ranked_pids = sorted(ranked_pids, key=lambda p: -rrf_scores[p])
            except sqlite3.OperationalError as e:
                # "no such table: paper_references" is expected pre-v6 migration.
                # Any other OperationalError (malformed SQL, disk I/O, DB locked)
                # should be reported, not silently swallowed.
                if "no such table" in str(e).lower():
                    pass
                else:
                    print(f"Hub scoring SQL error: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Hub scoring error: {e}", file=sys.stderr)

        # 4. Fetch full details for candidates. has_full_text flag comes from
        # the canonical schema column (1 iff processed_text OR tex_text >500
        # chars) so the formatter stays in sync with the pipeline's eligibility
        # semantics. Length uses MAX of the two text sources so TeX-only papers
        # report their true content length.
        placeholders = ",".join("?" for _ in ranked_pids)
        rows_by_pid = {}
        for row in conn.execute(f"""
            SELECT paper_id, doi, title, authors, abstract, tldr,
                   year, venue, citation_count, influential_citation_count,
                   fields_of_study, publication_types, is_open_access,
                   oa_pdf_url, local_pdf_path, s2_url,
                   has_full_text,
                   MAX(
                       LENGTH(COALESCE(processed_text, '')),
                       LENGTH(COALESCE(tex_text, ''))
                   ),
                   is_retracted
            FROM papers WHERE paper_id IN ({placeholders})
        """, ranked_pids).fetchall():
            rows_by_pid[row[0]] = row

        # 5. Cross-encoder reranking.
        # `rerank_on` controls what text the cross-encoder scores:
        #   - "chunk": pick each paper's best-matching chunk (from the chunk-level
        #       vec search above) and feed (query, chunk_text[:2000]). Best for
        #       verbatim citation search: the reranker sees the actual passage
        #       that matched, not a thin abstract.
        #   - "abstract": fallback to title + abstract[:500] (prior behavior).
        best_chunk_text: dict[str, str] = {}
        if rerank_on == "chunk" and query_emb is not None and ranked_pids:
            # Re-query chunks to pull BEST chunk per paper_id for this query.
            # k=1000: the vec0 KNN returns k GLOBAL nearest chunks, and we then
            # filter by paper_id. With 300 candidate papers (avg ~40 chunks each
            # among full-text papers), k=200 left most papers without a best_chunk
            # and silently fell back to abstract rerank text. 1000 gives most
            # papers at least one chunk match; remaining papers still fall back.
            placeholders_p = ",".join("?" for _ in ranked_pids)
            chunk_rows = conn.execute(f"""
                SELECT c.paper_id, c.chunk_text, v.distance
                FROM {vec_chunks_t} v
                JOIN paper_chunks c ON c.chunk_id = v.rowid
                WHERE v.embedding MATCH ? AND k = 1000
                  AND c.paper_id IN ({placeholders_p})
                ORDER BY v.distance
            """, (query_emb, *ranked_pids)).fetchall()
            for pid, chunk_text, _dist in chunk_rows:
                if pid not in best_chunk_text:
                    best_chunk_text[pid] = chunk_text[:2000]

        candidates = []
        chunk_fallbacks = 0
        for pid in ranked_pids:
            if pid not in rows_by_pid:
                continue
            row = rows_by_pid[pid]
            title = row[2] or ""
            if rerank_on == "chunk" and pid in best_chunk_text:
                rerank_text = f"{title}. {best_chunk_text[pid]}"
            else:
                if rerank_on == "chunk":
                    chunk_fallbacks += 1
                abstract = row[4] or ""
                tldr = row[5] or ""
                rerank_text = f"{title}. {abstract[:500]}" if abstract else (f"{title}. {tldr}" if tldr else title)
            candidates.append((pid, rerank_text))
        # Surface the silent chunk → abstract fallback so it doesn't mask
        # long-tail misses when many candidates lack a chunk match in the
        # k=1000 requery (e.g., papers with orphan or missing vec_chunks rows).
        if rerank_on == "chunk" and chunk_fallbacks > 0:
            print(
                f"[search_local] rerank-chunk fallback to abstract for "
                f"{chunk_fallbacks}/{len(candidates)} candidates "
                f"(no chunk in top-1000)",
                file=sys.stderr,
            )

        try:
            reranked_indices = await asyncio.to_thread(_rerank, query, candidates, limit)
            final_pids = [candidates[i][0] for i in reranked_indices]
        except Exception as e:
            print(f"Rerank error: {e}", file=sys.stderr)
            final_pids = ranked_pids[:limit]

        # Phase 3.1 reverted: the rank-based blended re-injection
        # (0.55*rerank + 0.15*hub + 0.15*chunk_agg + 0.10*vec + 0.05*fts)
        # regressed paper@1 15-50 pts across domains because rank-based
        # RRF with k=60 gives near-flat score differences, so strong
        # secondary signals (especially hub, which is dense on well-cited
        # papers) routinely promoted rerank-5 above rerank-1. The correct
        # blending requires raw cross-encoder scores, not ranks — saved
        # for Phase 4 when we have contextual chunks driving the rerank
        # and the CE scores are more separable.

        # Record final-rerank order for structured-output callers. The
        # cross-encoder rerank is the authoritative final ordering; raw
        # fts/vec/chunk/hub ranks above reflect the RRF input signals.
        rerank_position = {pid: i + 1 for i, pid in enumerate(final_pids)}

    finally:
        conn.close()

    # Structured output path: emit JSON with raw signals for agent-side
    # reranking. Used by Opus 4.7 / GPT 5.4 callers that want to do their
    # own judgment in their own context window.
    if structured:
        out: list[dict] = []
        for pid in final_pids:
            if pid not in rows_by_pid:
                continue
            (paper_id, doi, title, authors_json, abstract, tldr,
             year, venue, citations, influential, fields_json, types_json,
             is_oa, oa_url, local_pdf, s2_url, has_fulltext, fulltext_len,
             is_retracted) = rows_by_pid[pid]
            try:
                authors = json.loads(authors_json) if authors_json else []
            except json.JSONDecodeError:
                authors = []
            try:
                fields = json.loads(fields_json) if fields_json else []
            except json.JSONDecodeError:
                fields = []
            try:
                types = json.loads(types_json) if types_json else []
            except json.JSONDecodeError:
                types = []
            out.append({
                "paper_id": paper_id,
                "title": title,
                "authors": authors,
                "year": year,
                "venue": venue,
                "doi": doi,
                "citations": citations,
                "influential_citations": influential,
                "fields_of_study": fields,
                "work_types": types,
                "is_retracted": bool(is_retracted),
                "is_open_access": bool(is_oa),
                "oa_url": oa_url,
                "s2_url": s2_url,
                "local_pdf_path": local_pdf,
                "has_full_text": bool(has_fulltext),
                "full_text_chars": fulltext_len,
                "abstract": abstract,
                "tldr": tldr,
                "signals": {
                    "fts_rank": fts_ranks.get(pid),
                    "vec_rank": vec_ranks.get(pid),
                    "chunk_agg_rank": chunk_agg_ranks.get(pid),
                    "hub_rank": hub_ranks.get(pid),
                    "rerank_position": rerank_position.get(pid),
                },
                "best_chunk_text": best_chunk_text.get(pid),
            })
        return json.dumps({
            "query": query,
            "rerank_on": rerank_on,
            "count": len(out),
            "results": out,
        }, indent=2)

    # 6. Format results
    parts = [f"Found {len(final_pids)} papers in local library (hybrid search):\n"]
    for i, pid in enumerate(final_pids, 1):
        if pid not in rows_by_pid:
            continue
        (paper_id, doi, title, authors_json, abstract, tldr,
         year, venue, citations, influential, fields_json, types_json,
         is_oa, oa_url, local_pdf, s2_url, has_fulltext, fulltext_len,
         is_retracted) = rows_by_pid[pid]

        match_sources = []
        if pid in fts_ranks:
            match_sources.append(f"keyword#{fts_ranks[pid]}")
        if pid in vec_ranks:
            match_sources.append(f"semantic#{vec_ranks[pid]}")
        if pid in chunk_agg_ranks:
            match_sources.append(f"chunk#{chunk_agg_ranks[pid]}")
        if pid in hub_ranks:
            match_sources.append(f"hub#{hub_ranks[pid]}")

        parts.append(f"--- Local {i} [{', '.join(match_sources)}] ---")
        parts.append(f"**{title}** ({year})")
        if is_retracted:
            parts.append("⚠️ **RETRACTED** — do not cite without verification")

        try:
            authors = json.loads(authors_json) if authors_json else []
            if authors:
                author_str = ", ".join(authors[:5])
                if len(authors) > 5:
                    author_str += f" (+{len(authors) - 5} more)"
                parts.append(f"Authors: {author_str}")
        except json.JSONDecodeError:
            pass

        parts.append(f"Citations: {citations} (influential: {influential})")
        if venue:
            parts.append(f"Venue: {venue}")

        id_parts = []
        if doi:
            id_parts.append(f"DOI: {doi}")
        if paper_id:
            id_parts.append(f"S2: {paper_id}")
        if id_parts:
            parts.append(" | ".join(id_parts))

        if local_pdf:
            parts.append(f"LOCAL PDF: {local_pdf}")
        elif is_oa:
            parts.append("Open Access: Yes")
        else:
            parts.append("Open Access: No")

        if has_fulltext:
            parts.append(f"FULL TEXT: {fulltext_len:,} chars indexed")

        if tldr:
            parts.append(f"TLDR: {tldr}")
        if abstract:
            parts.append(f"Abstract: {abstract[:300]}{'...' if len(abstract) > 300 else ''}")
        if s2_url:
            parts.append(f"URL: {s2_url}")
        parts.append("")

    return "\n".join(parts)


@mcp.tool()
async def get_full_text(paper_id: str) -> str:
    """Retrieve the full processed text of a paper from the local library.

    Returns the best available full text for a paper: TeX-extracted (math-clean) when
    available, otherwise Docling PDF extraction. Identify the paper by S2 paper ID,
    DOI (prefix with "DOI:"), or exact title.

    Use search_local first to find papers, then this tool to read their full text.
    Returns the complete text as stored, no truncation.

    Args:
        paper_id: S2 paper ID, or "DOI:10.xxxx/yyyy", or exact title string
    """
    conn = _init_db()
    try:
        # Try as S2 ID first
        row = conn.execute(
            "SELECT paper_id, title, processed_text, tex_text FROM papers WHERE paper_id = ?",
            (paper_id,)
        ).fetchone()

        # Try as DOI. DOIs are case-insensitive per the DOI Handbook and
        # stored mixed-case in the DB (e.g. SICI DOIs with caps). Use
        # LOWER(doi) = LOWER(?) paired with the idx_papers_doi_lower
        # expression index so the lookup stays indexed and matches
        # stored-mixed-case entries.
        if not row and paper_id.upper().startswith("DOI:"):
            doi = paper_id[4:].strip()
            row = conn.execute(
                "SELECT paper_id, title, processed_text, tex_text FROM papers WHERE LOWER(doi) = LOWER(?)",
                (doi,)
            ).fetchone()
        elif not row and "/" in paper_id and not paper_id.startswith("http"):
            # Bare DOI without prefix
            row = conn.execute(
                "SELECT paper_id, title, processed_text, tex_text FROM papers WHERE LOWER(doi) = LOWER(?)",
                (paper_id,)
            ).fetchone()

        # Try as title
        if not row:
            row = conn.execute(
                "SELECT paper_id, title, processed_text, tex_text FROM papers WHERE title = ? COLLATE NOCASE",
                (paper_id,)
            ).fetchone()

        if not row:
            return f"Paper not found: '{paper_id}'. Use search_local to find papers first."

        pid, title, processed_text, tex_text = row
        # Prefer TeX source over Docling PDF extraction for math fidelity.
        if tex_text and tex_text.strip():
            text = tex_text
            source = "TeX source (math-clean)"
        elif processed_text and processed_text.strip():
            text = processed_text
            source = "PDF (Docling)"
        else:
            return f"Paper found: '{title}' (ID: {pid}), but no full text has been processed. Use download_paper, process_pdf, or process_tex to extract text."

        return f"# {title}\n**Paper ID:** {pid}\n**Source:** {source}\n**Characters:** {len(text):,}\n\n---\n\n{text}"

    finally:
        conn.close()


@mcp.tool()
async def search_passages(query: str, limit: int = 10, structured: bool = False) -> str:
    """Search for specific passages using hybrid search (keyword + semantic + reranking).

    Returns matching text chunks (~500 words each) with paper metadata,
    section headers, and page numbers. Finds passages even when your search
    terms differ from the text's vocabulary.

    Args:
        query: Search terms (e.g., "algorithmic tools international law")
        limit: Max chunks to return (default 10)
        structured: If True, return JSON with per-chunk raw signals
            (fts_rank, vec_rank, rerank_position) and full metadata. Lets
            frontier-model callers rerank in their own context. Default
            False preserves the existing markdown output.
    """
    conn = _init_db()
    try:
        # 1. FTS5 keyword search on chunks. Filter to cite-ready papers
        # (has_full_text=1) — v8 semantics: verified means source-known,
        # has_full_text means the paper has body text suitable for citation.
        # Passage search by definition wants the latter.
        fts_results = _fts5_safe_query(conn, """
            SELECT c.chunk_id
            FROM paper_chunks_fts fts
            JOIN paper_chunks c ON fts.rowid = c.chunk_id
            JOIN papers p ON c.paper_id = p.paper_id
            WHERE paper_chunks_fts MATCH ?
              AND p.has_full_text = 1
            ORDER BY rank
            LIMIT ?
        """, query, 300)
        fts_ranks = {row[0]: i + 1 for i, row in enumerate(fts_results)}

        # 2. Vector similarity search on chunks
        vec_ranks = {}
        try:
            query_emb = await asyncio.to_thread(_embed_query_text, query)
            _, vec_chunks_t = _vec_table_pair()
            # k=1500 before filtering for has_full_text=1 so we still have
            # ~300 eligible chunks in the worst case. Dense-rank reassignment
            # over surviving rows keeps RRF from seeing gaps.
            vec_results = conn.execute(f"""
                SELECT v.rowid, v.distance
                FROM {vec_chunks_t} v
                WHERE v.embedding MATCH ? AND k = 1500
                ORDER BY v.distance
            """, (query_emb,)).fetchall()
            if vec_results:
                vec_rowids = [r[0] for r in vec_results]
                placeholders = ",".join("?" for _ in vec_rowids)
                verified_cids = {r[0] for r in conn.execute(f"""
                    SELECT c.chunk_id FROM paper_chunks c
                    JOIN papers p ON c.paper_id = p.paper_id
                    WHERE c.chunk_id IN ({placeholders}) AND p.has_full_text = 1
                """, vec_rowids).fetchall()}
                seq_rank = 0
                for rowid, _dist in vec_results:
                    if rowid not in verified_cids:
                        continue
                    seq_rank += 1
                    if seq_rank > 300:
                        break
                    vec_ranks[rowid] = seq_rank
        except Exception as e:
            print(f"Vector search error (passages): {e}", file=sys.stderr)

        # 3. RRF fusion
        all_chunk_ids = set(fts_ranks.keys()) | set(vec_ranks.keys())
        k = 60
        rrf_scores = {}
        for cid in all_chunk_ids:
            score = 0.0
            if cid in fts_ranks:
                score += 1.0 / (k + fts_ranks[cid])
            if cid in vec_ranks:
                score += 1.0 / (k + vec_ranks[cid])
            rrf_scores[cid] = score
        # Widened 50 -> 300 so the cross-encoder rerank pass can see passages
        # with low vocabulary overlap to the query (e.g., a historian's concrete
        # prose vs. an abstract query about "constitutional settlement").
        ranked_cids = sorted(rrf_scores.keys(), key=lambda c: -rrf_scores[c])[:300]

        if not ranked_cids:
            return f"No passages found for '{query}'. Try search_local for broader paper-level search, or search_papers for Semantic Scholar."

        # 4. Fetch full chunk details
        placeholders = ",".join("?" for _ in ranked_cids)
        chunks_by_id = {}
        for row in conn.execute(f"""
            SELECT c.chunk_id, c.chunk_text, c.section_header, c.page_start, c.page_end,
                   p.paper_id, p.title, p.authors, p.year, p.venue, p.doi,
                   p.local_pdf_path, p.pdf_page_offset, p.pages_verified,
                   p.is_retracted
            FROM paper_chunks c
            JOIN papers p ON c.paper_id = p.paper_id
            WHERE c.chunk_id IN ({placeholders})
        """, ranked_cids).fetchall():
            chunks_by_id[row[0]] = row

        # 5. Cross-encoder reranking
        candidates = []
        for cid in ranked_cids:
            if cid not in chunks_by_id:
                continue
            row = chunks_by_id[cid]
            # Raise from 500 to 2000 chars: cross-encoder has ~512 tokens of context
            # (~2000 chars), so truncating to 500 wasted 80% of the budget and
            # systematically under-scored longer passages where the match sits past
            # char 500.
            candidates.append((cid, row[1][:2000]))

        try:
            reranked_indices = await asyncio.to_thread(_rerank, query, candidates, limit)
            final_cids = [candidates[i][0] for i in reranked_indices]
        except Exception as e:
            print(f"Rerank error (passages): {e}", file=sys.stderr)
            final_cids = ranked_cids[:limit]

        rerank_position = {cid: i + 1 for i, cid in enumerate(final_cids)}

    finally:
        conn.close()

    # Structured output path for agent-side reranking.
    if structured:
        out: list[dict] = []
        for cid in final_cids:
            if cid not in chunks_by_id:
                continue
            (chunk_id, chunk_text, section, page_start, page_end,
             paper_id, title, authors_json, year, venue, doi,
             local_pdf, pdf_offset, pages_verified, is_retracted) = chunks_by_id[cid]
            try:
                authors = json.loads(authors_json) if authors_json else []
            except json.JSONDecodeError:
                authors = []
            # Label pdf_page vs printed_page honestly based on pages_verified.
            page_label = "printed_page" if pages_verified else "pdf_page"
            out.append({
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "title": title,
                "authors": authors,
                "year": year,
                "venue": venue,
                "doi": doi,
                "is_retracted": bool(is_retracted),
                "section": section,
                "page_label": page_label,
                "page_start": page_start,
                "page_end": page_end,
                "pages_verified": bool(pages_verified),
                "pdf_page_offset": pdf_offset,
                "local_pdf_path": local_pdf,
                "chunk_text": chunk_text,
                "signals": {
                    "fts_rank": fts_ranks.get(cid),
                    "vec_rank": vec_ranks.get(cid),
                    "rerank_position": rerank_position.get(cid),
                },
            })
        return json.dumps({
            "query": query,
            "count": len(out),
            "results": out,
        }, indent=2)

    # 6. Format results
    parts = [f"Found {len(final_cids)} matching passages (hybrid search):\n"]
    for i, cid in enumerate(final_cids, 1):
        if cid not in chunks_by_id:
            continue
        (chunk_id, chunk_text, section, page_start, page_end,
         paper_id, title, authors_json, year, venue, doi,
         local_pdf, pdf_offset, pages_verified, is_retracted) = chunks_by_id[cid]

        match_sources = []
        if cid in fts_ranks:
            match_sources.append(f"keyword#{fts_ranks[cid]}")
        if cid in vec_ranks:
            match_sources.append(f"semantic#{vec_ranks[cid]}")

        parts.append(f"--- Passage {i} [{', '.join(match_sources)}] ---")
        parts.append(f"**{title}** ({year})")
        if is_retracted:
            parts.append("⚠️ **RETRACTED** — do not cite without verification")
        try:
            authors = json.loads(authors_json) if authors_json else []
            if authors:
                parts.append(f"Authors: {', '.join(authors[:5])}")
        except json.JSONDecodeError:
            pass
        if venue:
            parts.append(f"Venue: {venue}")
        parts.append(f"Paper ID: {paper_id}")
        if doi:
            parts.append(f"DOI: {doi}")
        if section:
            parts.append(f"Section: {section}")
        if page_start is not None:
            # Explicit pdf_page vs printed_page labeling so consumers know
            # when a number is Bluebook-citable (pages_verified=1) vs the
            # raw physical index Docling emitted (pages_verified=0).
            label = "printed_page" if pages_verified else "pdf_page"
            if page_end is not None and page_end != page_start:
                parts.append(f"{label}: {page_start}-{page_end}")
            else:
                parts.append(f"{label}: {page_start}")
            if not pages_verified and pdf_offset is not None and pdf_offset != 0:
                parts.append(f"(offset from printed: {pdf_offset})")
        if local_pdf:
            parts.append(f"Local PDF: {local_pdf}")
        parts.append(f"\n{chunk_text}\n")

    return "\n".join(parts)


@mcp.tool()
async def find_quotation(
    text: str,
    paper_id: str | None = None,
    fuzzy: bool = False,
    limit: int = 20,
) -> str:
    """Find an exact or near-exact quotation in the local library.

    Use for verbatim-quote verification (legal pincites, direct quotations in
    academic writing). For semantic search use `search_passages` instead.

    - Exact mode (default, fuzzy=False): FTS5 phrase match on chunk text.
      Returns chunks that contain the quotation as a contiguous phrase.
    - Fuzzy mode (fuzzy=True): tokenizes the quotation and finds chunks
      containing most tokens; useful when a quotation has minor OCR or
      transcription drift.
    - Scoping (paper_id set): restrict to one paper. Accepts an S2 ID,
      local:/web:/oa: id, or "DOI:10.x/...".

    Output lists candidates with paper metadata, page range, section,
    the matched chunk text, and match span indices (char offsets inside
    the chunk). "No room for error" callers should cross-check the
    returned chunk text against the query before citing.

    Args:
        text: The exact or near-exact quotation to find (5+ words recommended).
        paper_id: Optional scope. S2 ID, local:/web:/oa: prefix, or "DOI:..." .
        fuzzy: If True, match on token bag; default False for exact phrase.
        limit: Max results (default 20).
    """
    text = text.strip()
    if not text or len(text) < 3:
        return "Error: quotation too short. Provide at least a few words."

    conn = _init_db()
    try:
        # Resolve paper_id scope if provided. Accept DOI: prefix and title match
        # the same way get_full_text does.
        scope_pid: str | None = None
        if paper_id:
            row = conn.execute(
                "SELECT paper_id FROM papers WHERE paper_id = ?", (paper_id,)
            ).fetchone()
            if not row and paper_id.upper().startswith("DOI:"):
                doi = paper_id[4:].strip()
                row = conn.execute(
                    "SELECT paper_id FROM papers WHERE LOWER(doi) = LOWER(?)", (doi,)
                ).fetchone()
            elif not row and "/" in paper_id and not paper_id.startswith("http"):
                row = conn.execute(
                    "SELECT paper_id FROM papers WHERE LOWER(doi) = LOWER(?)", (paper_id,)
                ).fetchone()
            if not row:
                return f"Paper not found: '{paper_id}'."
            scope_pid = row[0]

        # Build FTS5 query.
        # Exact mode: quote as phrase. Double internal quotes per FTS5 escape rule.
        # Fuzzy mode: tokenize on whitespace, strip punctuation, AND the tokens
        # together (bag-of-words with implicit AND — FTS5 default).
        if fuzzy:
            tokens = re.findall(r"[A-Za-z0-9]+", text)
            # Drop ultra-short tokens (FTS5 tokenizer may strip anyway) and
            # obvious stop words for a cleaner bag.
            tokens = [t for t in tokens if len(t) > 2]
            if not tokens:
                return "Error: no usable tokens in fuzzy query after stripping."
            fts_query = " ".join(f'"{t}"' for t in tokens)
        else:
            fts_query = '"' + text.replace('"', '""') + '"'

        if scope_pid:
            sql = """
                SELECT c.chunk_id, c.chunk_text, c.section_header,
                       c.page_start, c.page_end, c.paper_id,
                       p.title, p.authors, p.year, p.venue, p.doi,
                       p.local_pdf_path, p.pdf_page_offset, p.pages_verified,
                       p.is_retracted
                FROM paper_chunks_fts fts
                JOIN paper_chunks c ON fts.rowid = c.chunk_id
                JOIN papers p ON c.paper_id = p.paper_id
                WHERE paper_chunks_fts MATCH ?
                  AND c.paper_id = ?
                  AND p.has_full_text = 1
                ORDER BY rank
                LIMIT ?
            """
            try:
                rows = conn.execute(sql, (fts_query, scope_pid, limit)).fetchall()
            except sqlite3.OperationalError:
                return f"No matches for that quotation in paper '{scope_pid}'."
        else:
            sql = """
                SELECT c.chunk_id, c.chunk_text, c.section_header,
                       c.page_start, c.page_end, c.paper_id,
                       p.title, p.authors, p.year, p.venue, p.doi,
                       p.local_pdf_path, p.pdf_page_offset, p.pages_verified,
                       p.is_retracted
                FROM paper_chunks_fts fts
                JOIN paper_chunks c ON fts.rowid = c.chunk_id
                JOIN papers p ON c.paper_id = p.paper_id
                WHERE paper_chunks_fts MATCH ?
                  AND p.has_full_text = 1
                ORDER BY rank
                LIMIT ?
            """
            try:
                rows = conn.execute(sql, (fts_query, limit)).fetchall()
            except sqlite3.OperationalError:
                return "No matches for that quotation."

        if not rows:
            mode = "fuzzy" if fuzzy else "exact"
            scope = f" within paper {scope_pid}" if scope_pid else ""
            return f"No {mode} quotation matches{scope}. Try the other mode or use search_passages for semantic search."
    finally:
        conn.close()

    parts = [f"Found {len(rows)} match{'es' if len(rows) != 1 else ''} for {'fuzzy' if fuzzy else 'exact'} quotation:\n"]
    for i, row in enumerate(rows, 1):
        (chunk_id, chunk_text, section, page_start, page_end,
         paper_id_row, title, authors_json, year, venue, doi,
         local_pdf, pdf_offset, pages_verified, is_retracted) = row

        # Locate the exact match span within the chunk for display. In exact
        # mode, this is a substring find; in fuzzy mode, the caller should
        # visually verify. Case-insensitive.
        match_span = ""
        if not fuzzy:
            idx = chunk_text.lower().find(text.lower())
            if idx >= 0:
                match_span = f"match_span: [{idx}, {idx + len(text)})"

        parts.append(f"--- Match {i} ---")
        parts.append(f"**{title}** ({year})")
        if is_retracted:
            parts.append("⚠️ **RETRACTED** — do not cite without verification")
        try:
            authors = json.loads(authors_json) if authors_json else []
            if authors:
                parts.append(f"Authors: {', '.join(authors[:5])}")
        except json.JSONDecodeError:
            pass
        if venue:
            parts.append(f"Venue: {venue}")
        parts.append(f"Paper ID: {paper_id_row}")
        if doi:
            parts.append(f"DOI: {doi}")
        if section:
            parts.append(f"Section: {section}")
        if page_start is not None:
            # Label pdf_page vs printed_page honestly: pages_verified=1 means
            # the printed-page offset has been verified; otherwise the number
            # is the raw PDF page index Docling emitted.
            label = "printed_page" if pages_verified else "pdf_page"
            if page_end is not None and page_end != page_start:
                parts.append(f"{label}: {page_start}-{page_end}")
            else:
                parts.append(f"{label}: {page_start}")
        if match_span:
            parts.append(match_span)
        if local_pdf:
            parts.append(f"Local PDF: {local_pdf}")
        parts.append(f"\n{chunk_text}\n")

    return "\n".join(parts)


# --- Phase 2.1: locate_claim_in_paper ---
# A pincite-oriented locator that takes a known paper_id plus the claim
# text and returns the single most likely page. Three strategies routed by
# claim_type: shingle match for verbatim quotes, embedding KNN + rerank
# for paraphrases, and numeric-token boost for numerical facts. The old
# harness cascade (find_quotation → search_within_paper LIKE) worked only
# when the claim had a near-literal substring in the paper; this tool is
# the semantic fallback that path always needed.

_LIGATURE_MAP = str.maketrans({
    "ﬀ": "ff",  # LATIN SMALL LIGATURE FF
    "ﬁ": "fi",  # LATIN SMALL LIGATURE FI
    "ﬂ": "fl",  # LATIN SMALL LIGATURE FL
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
})


def _normalize_for_match(text: str) -> str:
    """Light normalization for match-time comparison only. NEVER used when
    writing stored text (stored text must remain verbatim to preserve
    Bluebook-accurate pincite spans). Folds ligatures and collapses
    whitespace so FTS5 phrase match tolerates common Docling OCR drift.
    """
    normalized = text.translate(_LIGATURE_MAP)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _make_shingles(text: str, size: int, stride: int = 1) -> list[str]:
    """Generate word-level shingles of the given size with stride 1."""
    tokens = re.findall(r"[A-Za-z0-9'’\-]+", text)
    if len(tokens) < size:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + size]) for i in range(0, len(tokens) - size + 1, stride)]


def _numeric_tokens(text: str) -> list[str]:
    """Return distinct numeric tokens that carry enough information to
    disambiguate a table row. A bare "2" has 0 disambiguating power; a
    "42.5%", "2024", or "60 min" is worth scoring on.
    """
    tokens = []
    for m in re.finditer(r"\b\d+(?:[.,]\d+)?\b", text):
        tok = m.group(0)
        # Drop short unitless integers (1-2 digits) unless they look like
        # years (4 digits) or have a unit suffix within the next ~6 chars.
        digits_only = re.sub(r"\D", "", tok)
        if len(digits_only) >= 3:
            tokens.append(tok)
        else:
            # Look ahead for a unit
            tail = text[m.end() : m.end() + 8]
            if re.match(r"\s*(?:%|percent|bps|min|s\b|ms|years?|hours?|days?|kg|lbs?|cm|mm|m\b|km|hz|k|usd|\\\$|b(?:illion)?|mill?ion)", tail, re.IGNORECASE):
                tokens.append(tok)
    # Preserve order, dedupe
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _fts5_phrase_literal(text: str) -> str:
    """Wrap text as an FTS5 phrase literal with proper escaping."""
    return '"' + text.replace('"', '""') + '"'


@mcp.tool()
async def locate_claim_in_paper(
    paper_id: str,
    claim_text: str,
    claim_type: str = "paraphrase",
) -> str:
    """Locate the single most likely page for a claim inside a known paper.

    Pincite-oriented locator that routes by claim_type:

      - verbatim_quote: overlapping n-gram shingles (default size 6)
        matched as FTS5 phrases against paper_chunks_fts scoped to
        paper_id. Ligature-normalized at match time only. Tolerant of
        OCR drift that breaks single-phrase find_quotation.
      - paraphrase / doctrinal_proposition / methodological_claim:
        embedding KNN over the paper's chunks, top-K cross-encoder
        reranked; winner's page is returned.
      - numerical_fact: embedding KNN + numeric-token boost (3x score
        for passages containing ≥2 distinct numeric tokens from the
        claim, capped at the next-best passage's score × 3).

    Output: markdown in the same page-label format as find_quotation
    (printed_page: N / pdf_page: N), so harnesses that parse page
    markers work without modification.

    Args:
        paper_id: S2 ID, local:/web:/oa: prefix, or "DOI:10.x/..." .
        claim_text: The claim to locate. For verbatim quotes pass only
            the quoted kernel (no surrounding narration).
        claim_type: One of "verbatim_quote", "paraphrase",
            "numerical_fact", "doctrinal_proposition", "methodological_claim".
            Unknown types fall back to the paraphrase path.
    """
    claim_text = (claim_text or "").strip()
    if not claim_text or len(claim_text) < 3:
        return "Error: claim_text too short."

    conn = _init_db()
    try:
        # Resolve paper_id (same pattern as find_quotation / search_within_paper)
        row = conn.execute(
            "SELECT paper_id, title, year, pages_verified, doi, pdf_page_offset, local_pdf_path, has_gapped_extraction "
            "FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        if not row and paper_id.upper().startswith("DOI:"):
            doi = paper_id[4:].strip()
            row = conn.execute(
                "SELECT paper_id, title, year, pages_verified, doi, pdf_page_offset, local_pdf_path, has_gapped_extraction "
                "FROM papers WHERE LOWER(doi) = LOWER(?)", (doi,)
            ).fetchone()
        elif not row and "/" in paper_id and not paper_id.startswith("http"):
            row = conn.execute(
                "SELECT paper_id, title, year, pages_verified, doi, pdf_page_offset, local_pdf_path, has_gapped_extraction "
                "FROM papers WHERE LOWER(doi) = LOWER(?)", (paper_id,)
            ).fetchone()
        if not row:
            return f"Paper not found: '{paper_id}'."
        pid, title, year, pages_verified, doi, pdf_offset, local_pdf, has_gapped = row

        chunk_rows = conn.execute(
            "SELECT chunk_id, chunk_index, chunk_text, section_header, page_start, page_end "
            "FROM paper_chunks WHERE paper_id = ? ORDER BY chunk_index",
            (pid,),
        ).fetchall()
        if not chunk_rows:
            return (
                f"No chunks available for '{title}' ({year}). Paper may be "
                "metadata-only (has_full_text=0) or TeX-only without chunks."
            )
        chunk_ids = [r[0] for r in chunk_rows]
        chunk_meta: dict[int, tuple[int, str, str | None, int | None, int | None]] = {
            cid: (idx, text, section, ps, pe)
            for cid, idx, text, section, ps, pe in chunk_rows
        }

        label = "printed_page" if pages_verified else "pdf_page"
        best_chunk_id: int | None = None
        best_score: float = 0.0
        best_method = claim_type

        # Strategy A: verbatim_quote → shingle-based FTS5 phrase match
        if claim_type == "verbatim_quote":
            normalized_claim = _normalize_for_match(claim_text)
            shingles = _make_shingles(normalized_claim, size=6, stride=1)
            if len(shingles) < 3:
                # Claim too short for shingle scoring; fall back to size-3
                shingles = _make_shingles(normalized_claim, size=3, stride=1)
            shingle_hits: dict[int, int] = {}  # chunk_id -> match count
            placeholders = ",".join("?" for _ in chunk_ids)
            for shingle in shingles[:50]:  # cap per-claim shingle FTS cost
                if len(shingle) < 10:
                    continue
                fts_q = _fts5_phrase_literal(shingle)
                try:
                    rows = conn.execute(
                        f"SELECT c.chunk_id FROM paper_chunks_fts fts "
                        f"JOIN paper_chunks c ON fts.rowid = c.chunk_id "
                        f"WHERE paper_chunks_fts MATCH ? AND c.chunk_id IN ({placeholders})",
                        (fts_q, *chunk_ids),
                    ).fetchall()
                    for (cid,) in rows:
                        shingle_hits[cid] = shingle_hits.get(cid, 0) + 1
                except sqlite3.OperationalError:
                    continue
            if shingle_hits:
                # Pick the chunk with the most matching shingles. Tie-break
                # on smallest chunk_index (earliest in paper) and prefer
                # chunks that have a page_start set (avoid chunks with
                # None page).
                best_chunk_id = max(
                    shingle_hits.keys(),
                    key=lambda cid: (
                        shingle_hits[cid],
                        -chunk_meta[cid][0],
                        chunk_meta[cid][3] is not None,
                    ),
                )
                best_score = float(shingle_hits[best_chunk_id])
                best_method = f"shingle({len(shingles)})"

        # Strategy B: semantic (embedding KNN + rerank) — used for non-
        # verbatim types and as a fallback when shingles miss.
        if best_chunk_id is None:
            try:
                query_emb = await asyncio.to_thread(_embed_query_text, claim_text)
                placeholders = ",".join("?" for _ in chunk_ids)
                # sqlite-vec's MATCH supports rowid IN (...) as a
                # pre-filter, so we can scope KNN to just this paper's
                # chunks without post-filter starvation.
                _, _vc_t = _vec_table_pair()
                vec_rows = conn.execute(
                    f"SELECT v.rowid, v.distance FROM {_vc_t} v "
                    f"WHERE v.embedding MATCH ? "
                    f"  AND v.rowid IN ({placeholders}) "
                    f"  AND k = 50 "
                    f"ORDER BY v.distance",
                    (query_emb, *chunk_ids),
                ).fetchall()
                if vec_rows:
                    # Top-10 by vec distance → cross-encoder rerank
                    top_k_cids = [cid for cid, _ in vec_rows[:10]]
                    pairs = [(claim_text, chunk_meta[cid][1][:2000]) for cid in top_k_cids]
                    scores = await asyncio.to_thread(
                        lambda: _get_cross_encoder().predict(pairs, show_progress_bar=False)
                    )
                    ranked = sorted(range(len(scores)), key=lambda i: -float(scores[i]))

                    # Numerical-fact boost: 3x score multiplier for chunks
                    # containing ≥2 distinct numeric tokens from the claim,
                    # capped so a body-text number match can't override a
                    # strong semantic match by more than 3x.
                    rerank_scores = {top_k_cids[i]: float(scores[i]) for i in range(len(scores))}
                    if claim_type == "numerical_fact":
                        claim_nums = _numeric_tokens(claim_text)
                        if len(claim_nums) >= 1:
                            for cid in top_k_cids:
                                txt = chunk_meta[cid][1]
                                present = sum(1 for n in claim_nums if n in txt)
                                if present >= 2:
                                    rerank_scores[cid] *= 3.0
                                elif present >= 1 and len(claim_nums) == 1:
                                    rerank_scores[cid] *= 1.5
                        # Cap: no chunk's boosted score may exceed 3x the
                        # best un-boosted chunk's score (prevents numeric
                        # noise promoting a table-of-contents row).
                        unboosted_max = max(scores)
                        cap = 3.0 * float(unboosted_max) if unboosted_max > 0 else float("inf")
                        for cid in list(rerank_scores.keys()):
                            if rerank_scores[cid] > cap:
                                rerank_scores[cid] = cap
                        ranked = sorted(rerank_scores.keys(), key=lambda c: -rerank_scores[c])
                        best_chunk_id = ranked[0] if ranked else None
                        best_score = rerank_scores[best_chunk_id] if best_chunk_id else 0.0
                        best_method = "semantic+numeric-boost"
                    else:
                        best_chunk_id = top_k_cids[ranked[0]] if ranked else None
                        best_score = float(scores[ranked[0]]) if ranked else 0.0
                        best_method = "semantic+rerank"
            except Exception as e:
                print(f"locate_claim_in_paper KNN error ({pid}): {e}", file=sys.stderr)

        # Strategy C: fallback — plain LIKE on the original claim's first
        # 120 chars. Last resort; kept so we always return something
        # sensible rather than an error.
        if best_chunk_id is None:
            probe = claim_text[:120]
            like_rows = conn.execute(
                "SELECT chunk_id FROM paper_chunks WHERE paper_id = ? AND chunk_text LIKE ? "
                "ORDER BY chunk_index LIMIT 1",
                (pid, f"%{probe}%"),
            ).fetchall()
            if like_rows:
                best_chunk_id = like_rows[0][0]
                best_method = "like-fallback"

        if best_chunk_id is None:
            return (
                f"No candidate chunk found for claim in '{title}' ({year}). "
                f"Paper may not contain supporting text."
            )

        idx, chunk_text, section, ps, pe = chunk_meta[best_chunk_id]

        # If the chunk has a page range, also look in paper_passages for a
        # single-page passage match (Bluebook wants a single page when
        # possible).
        single_page: int | None = None
        if ps is not None and pe is not None and claim_type != "numerical_fact":
            # Prefer the page whose passage most contains the chunk text
            # prefix. paper_passages is indexed by page_num.
            passage_rows = conn.execute(
                "SELECT page_num, passage_text FROM paper_passages "
                "WHERE paper_id = ? AND page_num BETWEEN ? AND ? ORDER BY page_num",
                (pid, ps, pe),
            ).fetchall()
            if passage_rows and len(passage_rows) > 1:
                # Pick the passage that contains the most normalized query tokens
                claim_tokens = set(re.findall(r"[A-Za-z0-9]{4,}", claim_text.lower()))
                best_pass_page: int | None = None
                best_pass_overlap = 0
                for page_num, passage_text in passage_rows:
                    passage_tokens = set(re.findall(r"[A-Za-z0-9]{4,}", passage_text.lower()))
                    overlap = len(claim_tokens & passage_tokens)
                    if overlap > best_pass_overlap:
                        best_pass_overlap = overlap
                        best_pass_page = page_num
                if best_pass_page is not None and best_pass_overlap >= 2:
                    single_page = best_pass_page

        # Build the output markdown. Format matches find_quotation so
        # downstream PAGE_RE parsers work without modification.
        parts = [
            f"**{title}** ({year})",
            f"Paper ID: {pid}",
        ]
        if doi:
            parts.append(f"DOI: {doi}")
        if section:
            parts.append(f"Section: {section}")
        display_ps = single_page if single_page is not None else ps
        display_pe = single_page if single_page is not None else pe
        if display_ps is not None:
            if display_pe is not None and display_pe != display_ps:
                parts.append(f"{label}: {display_ps}-{display_pe}")
            else:
                parts.append(f"{label}: {display_ps}")
            if not pages_verified and pdf_offset is not None and pdf_offset != 0:
                parts.append(f"(offset from printed: {pdf_offset})")
        parts.append(f"method: {best_method}")
        parts.append(f"score: {best_score:.4f}")
        # Phase A.1: surface has_gapped_extraction so callers can decide
        # whether to trust the page (1 = no markers OR descending markers
        # OR TeX-only ⇒ pincite reliability is degraded).
        if has_gapped == 1:
            parts.append("has_gapped_extraction: 1 (pincite page may be unreliable)")
        if local_pdf:
            parts.append(f"Local PDF: {local_pdf}")
        parts.append(f"\n{chunk_text[:1500]}\n")
        return "\n".join(parts)
    finally:
        conn.close()


# Phase 5c (plan v3 §3): Semantic-Illusion tier-0 screen.
#
# v2 baseline showed NLI false-positive rate of 30/32 = 94% on
# adjacent-topic passages: NLI predicts "support" with high confidence
# on text that's TOPICALLY similar to the claim but doesn't actually
# ground the same entity. The published failure mode (Stacey et al.
# 2024 / Pasaribu 2024) — "Semantic Illusion" — predicts NLI is
# unreliable when (passage_similarity is high) AND (entity overlap is
# low). Tier 0 flags this case so callers can escalate to deep-judge.
#
# Tier-0 thresholds (default): similarity > 0.85, entity overlap < 0.4.
# Both env-configurable via TIER0_SIM_THRESHOLD / TIER0_ENT_THRESHOLD.

# Tier-0 similarity threshold defaults are EMBED_VARIANT-aware: nomic
# raw cosines run ~0.05-0.10 higher than Qwen3 on the same (claim,
# passage) pair (verified by Phase 4R bench at SHA 9c34f12 — Qwen3
# max sim was 0.834, p95=0.807, vs nomic max ~0.878 for similar pairs).
# Use per-variant defaults so the threshold semantics ("near-duplicate
# topical match") stays stable across backends. Manual override via
# TIER0_SIM_THRESHOLD env var still wins.
#
# Wave-5 review CRITICAL fix: previously `TIER0_SIM_THRESHOLD`,
# `TIER0_ENT_THRESHOLD`, and `TIER0_ENABLED` were evaluated once at
# module import; any in-process EMBED_VARIANT switch (test harnesses,
# A/B comparisons) silently used the wrong threshold. Replace with
# per-call accessors that read env vars + EMBED_VARIANT at the moment
# of evaluation. Bench performance is unaffected (env reads are
# microseconds; tier-0 fires once per claim).
def _default_tier0_sim_threshold() -> float:
    if EMBED_VARIANT.startswith("qwen3-mlx-"):
        return 0.78
    return 0.85


def _tier0_sim_threshold() -> float:
    override = os.environ.get("TIER0_SIM_THRESHOLD")
    if override is not None:
        try:
            return float(override)
        except ValueError:
            pass
    return _default_tier0_sim_threshold()


def _tier0_ent_threshold() -> float:
    override = os.environ.get("TIER0_ENT_THRESHOLD")
    if override is not None:
        try:
            return float(override)
        except ValueError:
            pass
    return 0.4


def _tier0_enabled() -> bool:
    return os.environ.get("VERIFY_TIER_0_SCREEN", "1") == "1"


# Backwards-compat aliases for any external callers reading these
# constants. Deprecated: use the function accessors above.
TIER0_SIM_THRESHOLD = _tier0_sim_threshold()
TIER0_ENT_THRESHOLD = _tier0_ent_threshold()
TIER0_ENABLED = _tier0_enabled()

_TIER0_ENTITY_RE = re.compile(
    r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+){0,3}\b"
)
_TIER0_STOP_HEADS = frozenset({
    # Sentence-start / common-word capitalizations that aren't entities.
    "The", "This", "That", "These", "Those", "When", "Where", "While",
    "If", "For", "But", "And", "However", "Because", "Since", "Although",
    "With", "Without", "Under", "Over", "Through", "After", "Before",
    "By", "In", "On", "At", "To", "From", "As", "An", "A", "Given",
    "Whereas", "We", "They", "He", "She", "It", "Our", "Their",
    "Furthermore", "Moreover", "Therefore", "Thus", "Hence",
})


def _tier0_extract_entities(text: str) -> set[str]:
    """Multi-word capitalized entity tokens; lowercase the result for set arithmetic.

    Drops single sentence-start capitalizations via _TIER0_STOP_HEADS.
    """
    if not text:
        return set()
    out: set[str] = set()
    for m in _TIER0_ENTITY_RE.finditer(text):
        head = m.group(0).split()[0]
        if head in _TIER0_STOP_HEADS:
            continue
        out.add(m.group(0).lower())
    return out


def _tier0_cosine(claim: str, passage: str) -> float:
    """Cosine similarity via the active EMBED_VARIANT pipeline.

    Returns 0.0 if either side is empty. nomic SentenceTransformer
    output is NOT pre-normalized; Qwen3-MLX path IS pre-normalized.
    We always L2-normalize here for safety so the threshold is the
    same regardless of variant.
    """
    if not claim or not passage:
        return 0.0
    claim_b = _embed_query_text(claim)
    passage_b = _embed_query_text(passage[:2000])  # cap passage length for cost control
    a = np.frombuffer(claim_b, dtype=np.float32).copy()
    b = np.frombuffer(passage_b, dtype=np.float32).copy()
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _tier0_screen(claim: str, passage: str) -> dict:
    """Return tier-0 metrics + recommended action for one (claim, passage) pair.

    Output keys (always present, even when disabled):
      enabled         (bool)
      similarity      (float | None)
      entity_overlap  (float | None)  # Jaccard
      action          ("no_check" | "ok" | "deep_judge_recommended")
      reason          (str)           # short hint for downstream use

    Plan v3 §3 Phase 5c: caller (verify_claim) writes these into the
    output JSON. A "deep_judge_recommended" action is informational
    today; tier-3 dual-judge wiring (Phase 5c step 3) consumes it
    later to actually invoke the Opus+Codex CLI judges.
    """
    if not _tier0_enabled():
        return {
            "enabled": False,
            "similarity": None,
            "entity_overlap": None,
            "action": "no_check",
            "reason": "tier-0 disabled (VERIFY_TIER_0_SCREEN=0)",
        }
    sim = _tier0_cosine(claim, passage)
    claim_ents = _tier0_extract_entities(claim)
    passage_ents = _tier0_extract_entities(passage)
    if claim_ents:
        overlap_ratio = len(claim_ents & passage_ents) / len(claim_ents)
    else:
        overlap_ratio = 1.0  # no claim entities to ground → can't fail this gate
    sim_thr = _tier0_sim_threshold()
    ent_thr = _tier0_ent_threshold()
    flag = sim > sim_thr and overlap_ratio < ent_thr
    return {
        "enabled": True,
        "similarity": round(sim, 4),
        "entity_overlap": round(overlap_ratio, 4),
        "action": "deep_judge_recommended" if flag else "ok",
        "reason": (
            f"sim={sim:.3f}>{sim_thr} AND "
            f"entity_overlap={overlap_ratio:.3f}<{ent_thr} "
            "(topical match without entity grounding — Semantic Illusion pattern)"
            if flag else
            f"sim={sim:.3f}, entity_overlap={overlap_ratio:.3f}: passes tier-0"
        ),
    }


def _nli_predict(premise: str, hypothesis: str) -> tuple[str, float, list[float]]:
    """Run NLI on (premise, hypothesis) and return (verdict, confidence, probs).

    Returns probs ordered (refute, support, insufficient) — the verify_claim
    convention — regardless of the underlying model's native label order.
    The label permutation is defined by _NLI_LABEL_PERM.
    """
    import numpy as np
    model = _get_nli_model()
    # predict returns an array of shape (N, 3) for a list of pairs.
    scores = model.predict([(premise, hypothesis)])
    if hasattr(scores, "tolist"):
        raw = list(scores[0])
    else:
        raw = list(scores)
    # Softmax over the raw logits, then permute to (refute, support, insufficient).
    m = max(raw)
    exp = [pow(2.71828182845904, s - m) for s in raw]
    z = sum(exp) or 1.0
    native_probs = [e / z for e in exp]
    probs = [native_probs[_NLI_LABEL_PERM[i]] for i in range(3)]
    labels = ["refute", "support", "insufficient"]
    idx = probs.index(max(probs))
    return labels[idx], float(probs[idx]), [float(p) for p in probs]


@mcp.tool()
async def verify_claim(
    claim: str,
    paper_id: str,
    page_range: str | None = None,
    min_confidence: float = 0.7,
) -> str:
    """Check whether evidence in a paper supports, refutes, or fails to
    establish a claim. Use before citing to avoid "the source doesn't
    actually say that" errors.

    Pulls the relevant passage(s) from `paper_passages` (page-bounded) or
    `paper_chunks` (fallback) and runs an NLI model to classify the
    (passage, claim) relation.

    Output:
      - verdict: "support" | "refute" | "insufficient" | "no_evidence"
      - confidence: [0, 1]
      - evidence_span: the passage text that drove the verdict
      - page: the page (or page range) the evidence was on
      - abstain: True when verdict is "insufficient" OR confidence is
        below min_confidence; callers should treat this as "do not cite"

    Args:
        claim: The proposition being verified (e.g., "The plaintiff
            bears the burden of proving negligence by a preponderance
            of the evidence").
        paper_id: S2 ID, local:/web:/oa: prefix, or "DOI:10.x/..." .
        page_range: Optional. Restrict evidence lookup to a page or
            page range. Accepts "42" for a single page, "42-45" for
            a range. If None, searches all passages in the paper.
        min_confidence: Below this NLI probability, the tool reports
            abstain=True regardless of verdict. Default 0.7.
    """
    claim = claim.strip()
    if not claim:
        return json.dumps({"error": "empty claim"})

    conn = _init_db()
    try:
        # Resolve paper_id. Phase 5a.3: also load authors + abstract so the
        # recency/entity gates below can check paper metadata without a
        # second round-trip.
        row = conn.execute(
            "SELECT paper_id, title, year, pages_verified, doi, authors, abstract "
            "FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        if not row and paper_id.upper().startswith("DOI:"):
            doi = paper_id[4:].strip()
            row = conn.execute(
                "SELECT paper_id, title, year, pages_verified, doi, authors, abstract "
                "FROM papers WHERE LOWER(doi) = LOWER(?)", (doi,)
            ).fetchone()
        if not row:
            return json.dumps({"error": f"paper not found: '{paper_id}'"})

        pid, title, year, pages_verified, doi, authors_json, abstract_text = row

        # Phase 5a.3: cheap abstain gates BEFORE pulling passages or
        # running NLI. Catches most negative-control attack patterns
        # (fabricated-study names, future-dated citations) at ~1ms cost.
        #
        # Gate 1 — recency mismatch: if the claim names a year later
        # than the paper's publication year + 1-year grace (for
        # forthcoming/online-first references), the paper cannot
        # plausibly support the claim. Applies only when paper.year is
        # known. The grace year avoids false-triggers on 2024 papers
        # citing 2025 forthcoming work.
        if year is not None:
            claim_years = [int(y) for y in re.findall(r"\b(19[5-9]\d|20[0-4]\d)\b", claim)]
            if claim_years and max(claim_years) > int(year) + 1:
                return json.dumps({
                    "paper_id": pid, "title": title, "year": year,
                    "verdict": "insufficient",
                    "confidence": 0.0,
                    "evidence_span": None,
                    "page": None,
                    "abstain": True,
                    "reason": f"recency-mismatch: claim references year {max(claim_years)}, paper published {year}",
                    # Round-3 review HIGH fix: every verify_claim return path
                    # must carry a tier_0 dict so consumers can read
                    # `verify.tier_0.action` without KeyError.
                    "tier_0": {"enabled": _tier0_enabled(), "similarity": None,
                               "entity_overlap": None, "action": "early_exit",
                               "reason": "recency-gate abstain (no NLI ran)"},
                }, indent=2)

        # Gate 2 — attributional-author mismatch: when the claim
        # attributes a finding to a specific author ("X et al.", "X
        # argues/finds/claims/notes/shows"), that author MUST appear
        # in the paper's metadata. This is the strongest neg-control
        # signal: a claim saying "Bakarji et al. found X" against a
        # paper not by Bakarji is a fabricated-attribution attack even
        # if the topic (SPY, VPIN, etc.) happens to match the paper's
        # subject. Check this BEFORE the general entity gate because
        # topical entities can match real-paper-titles and mask the
        # author-fabrication signal.
        paper_meta_blob_parts = [(title or "").lower()]
        if abstract_text:
            paper_meta_blob_parts.append(abstract_text.lower())
        if authors_json:
            try:
                for a in json.loads(authors_json):
                    if isinstance(a, str):
                        paper_meta_blob_parts.append(a.lower())
            except json.JSONDecodeError:
                pass
        paper_meta_blob = " ".join(paper_meta_blob_parts)

        _STOP_STARTS = {"The", "This", "That", "These", "Those", "When", "Where",
                        "While", "If", "For", "But", "And", "However",
                        "Because", "Since", "Although", "With", "Without",
                        "Under", "Over", "Through", "After", "Before",
                        "By", "In", "On", "At", "To", "From", "As", "Given", "Whereas"}

        # Author-attribution extractor. Matches single-surname + reporting
        # verb OR "<Surname> et al." forms. Surname = capitalized word of
        # length ≥4 (drops short capitalized function words).
        author_patterns = re.findall(
            r"\b([A-Z][a-z]{3,})\s+(?:et\s+al\.?(?:['s]{0,2})?|argued?|analys(?:es|ed|ing)|"
            r"analyz(?:es|ed|ing)|describe[ds]?|describes|noted?|notes|"
            r"claim(?:ed|s)?|found|finds|show(?:ed|s|n)?|demonstrate[ds]?|"
            r"observed?|observes|write[s]?|wrote|suggest(?:ed|s)?|"
            r"contend(?:ed|s)?|argues?)\b",
            claim,
        )
        authors_from_claim = [a for a in author_patterns if a not in _STOP_STARTS]
        if authors_from_claim:
            any_author_match = any(a.lower() in paper_meta_blob for a in authors_from_claim)
            if not any_author_match:
                return json.dumps({
                    "paper_id": pid, "title": title, "year": year,
                    "verdict": "insufficient",
                    "confidence": 0.0,
                    "evidence_span": None,
                    "page": None,
                    "abstain": True,
                    "reason": (
                        "attributional-author-mismatch: claim attributes to "
                        + ", ".join(repr(a) for a in authors_from_claim[:2])
                        + " but that author does not appear in the paper's metadata"
                    ),
                    "tier_0": {"enabled": _tier0_enabled(), "similarity": None,
                               "entity_overlap": None, "action": "early_exit",
                               "reason": "author-attribution-gate abstain (no NLI ran)"},
                }, indent=2)

        # Gate 3 — generic entity-ungrounded: if the claim names a
        # study / cohort / numbered-program (multi-word capitalized term
        # or "<Word> Study/Cohort/Trial"), and NONE of those named
        # entities appear in the paper's title / authors / abstract, the
        # paper cannot ground the claim. Only triggers when the claim has
        # at least one multi-word entity; single-word acronyms are too
        # noisy to require a match on.
        entity_patterns = re.findall(
            r"\b[A-Z][A-Za-z]+(?:[\-\s][A-Z][A-Za-z]+){1,3}\b"
            r"|\b[A-Z][A-Za-z]+\s+(?:Study|Cohort|Trial|Survey|Project)\b",
            claim,
        )
        entities = [e for e in entity_patterns if e.split()[0] not in _STOP_STARTS]
        if entities:
            any_match = any(e.lower() in paper_meta_blob for e in entities)
            if not any_match:
                return json.dumps({
                    "paper_id": pid, "title": title, "year": year,
                    "verdict": "insufficient",
                    "confidence": 0.0,
                    "evidence_span": None,
                    "page": None,
                    "abstain": True,
                    "reason": (
                        "entity-ungrounded: claim references "
                        + ", ".join(repr(e) for e in entities[:3])
                        + " but none appear in the paper's title/authors/abstract"
                    ),
                    "tier_0": {"enabled": _tier0_enabled(), "similarity": None,
                               "entity_overlap": None, "action": "early_exit",
                               "reason": "entity-grounding-gate abstain (no NLI ran)"},
                }, indent=2)

        # Parse page_range
        page_start = page_end = None
        if page_range:
            pr = page_range.strip()
            if "-" in pr:
                a, b = pr.split("-", 1)
                try:
                    page_start = int(a.strip())
                    page_end = int(b.strip())
                except ValueError:
                    return json.dumps({"error": f"invalid page_range '{pr}'"})
            else:
                try:
                    page_start = int(pr)
                    page_end = page_start
                except ValueError:
                    return json.dumps({"error": f"invalid page_range '{pr}'"})

        # Note: Phase 1.4's proposed page_range-required contract was tried
        # and reverted. On the 49-claim bench, the widened abstain rule
        # (Phase 1.1: verdict in {insufficient, refute} OR confidence <
        # min_confidence OR margin<0.30) already catches all 4
        # negative-control false-supports that the contract was designed
        # to prevent (they had verdict=refute at >0.99 confidence).
        # Requiring page_range on all paper_passages-enabled calls only
        # forced false-abstains on real claims where the locator failed,
        # tanking abstain_precision without net gain. Revisit when Phase 2
        # delivers a stricter locate_claim_in_paper primitive.

        # Pull candidate passages. Prefer paper_passages (single-page grain)
        # over paper_chunks. When page_range is given, filter by page.
        passages: list[tuple[int, str, str | None]] = []  # (page_num, text, section)
        if page_range:
            rows = conn.execute(
                "SELECT page_num, passage_text, section_header "
                "FROM paper_passages WHERE paper_id = ? "
                "AND page_num BETWEEN ? AND ? ORDER BY page_num",
                (pid, page_start, page_end),
            ).fetchall()
            passages = [(r[0], r[1], r[2]) for r in rows]
        else:
            rows = conn.execute(
                "SELECT page_num, passage_text, section_header "
                "FROM paper_passages WHERE paper_id = ? "
                "ORDER BY page_num",
                (pid,),
            ).fetchall()
            passages = [(r[0], r[1], r[2]) for r in rows]

        if not passages:
            # Fallback: paper may not have passages (pre-v18, no markers,
            # or TeX-only). Use chunks.
            if page_range:
                rows = conn.execute(
                    "SELECT page_start, chunk_text, section_header "
                    "FROM paper_chunks WHERE paper_id = ? "
                    "AND page_start IS NOT NULL AND page_end IS NOT NULL "
                    "AND NOT (page_end < ? OR page_start > ?) "
                    "ORDER BY chunk_index",
                    (pid, page_start, page_end),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT page_start, chunk_text, section_header "
                    "FROM paper_chunks WHERE paper_id = ? "
                    "ORDER BY chunk_index",
                    (pid,),
                ).fetchall()
            passages = [(r[0] or 0, r[1], r[2]) for r in rows]

        if not passages:
            return json.dumps({
                "paper_id": pid, "title": title, "year": year,
                "verdict": "no_evidence",
                "confidence": 0.0,
                "evidence_span": None,
                "page": None,
                "abstain": True,
                "note": "no passages found in the specified range (or paper has no processed text)",
                "tier_0": {"enabled": _tier0_enabled(), "similarity": None,
                           "entity_overlap": None, "action": "early_exit",
                           "reason": "no passages to score"},
            }, indent=2)

        # Phase 5a.2: token-aware windowing. Instead of cropping to 2000
        # chars (≈500 tokens, at the tokenizer's 512 limit) and silently
        # truncating the tail of multi-page or long chunks, split each
        # passage into ~450-token sliding windows with 50% overlap.
        # Character-based approximation: ~4 chars/token on English prose,
        # so 1800 chars per window. Passages ≤1800 chars become one window
        # each. Longer passages (esp. chunk-fallback for TeX-only papers
        # where passages aren't page-bounded) get multiple windows, each
        # of which gets independently scored — the winning window's
        # passage page is returned. This fixes the "decisive evidence is
        # on page 3 of a 4-page chunk, but the model only saw page 1"
        # failure mode that accounted for ~3 of the 21 high-conf
        # insufficient verdicts on Phase 2's bench.
        WINDOW_CHARS = 1800
        WINDOW_STRIDE = 900  # 50% overlap

        def _windows_of(text: str) -> list[str]:
            if len(text) <= WINDOW_CHARS:
                return [text]
            out: list[str] = []
            i = 0
            while i < len(text):
                out.append(text[i : i + WINDOW_CHARS])
                if i + WINDOW_CHARS >= len(text):
                    break
                i += WINDOW_STRIDE
            return out

        # Flatten passages × windows into candidates. Each candidate is
        # (passage_idx, window_text). Multiple windows per passage are OK.
        candidates: list[tuple[int, str]] = []
        for i, (pn, text, sec) in enumerate(passages):
            for w in _windows_of(text):
                candidates.append((i, w))

        # Cross-encoder pre-filter. With windows, the candidate pool may be
        # 2-3x larger than the passage count, so the top-20 cap still
        # bounds NLI cost at ~2s compute.
        PREFILTER_TOP_K = 20
        if len(candidates) > PREFILTER_TOP_K:
            relevance_scores = await asyncio.to_thread(
                lambda: _get_cross_encoder().predict([(claim, c[1]) for c in candidates])
            )
            ranked = sorted(
                range(len(candidates)),
                key=lambda i: -float(relevance_scores[i]),
            )[:PREFILTER_TOP_K]
            to_check = [candidates[i] for i in ranked]
        else:
            to_check = candidates

        # Note: Phase 5b tried regex-based atomic decomposition (STRIVE-
        # style, split on 'and'/';'/':' then require all atoms to
        # support). It regressed 30.6% → 28.6% because the strict
        # AND-aggregation is wrong for pincite semantics: a compound
        # claim like "Participants had WASO 60 min and reported
        # increased depressive symptoms" is citeable when the paper
        # supports the WASO part even if the exact "depressive
        # symptoms" wording is looser than NLI recognizes. STRIVE's
        # +21.9% HOVER result is on multi-hop fact-checking where all
        # atoms must hold; pincite is more permissive. Revert; richer
        # LLM-based decomposition with softer aggregation is the 5c
        # hook if we revisit.

        # Run NLI on each candidate, keep the highest-confidence non-neutral
        # hit. Neutral (insufficient) is allowed to win if nothing beats it.
        best = {
            "idx": None, "verdict": "insufficient", "confidence": 0.0,
            "probs": [0.0, 0.0, 0.0], "window_text": None,
        }
        for idx, text in to_check:
            verdict, conf, probs = await asyncio.to_thread(
                _nli_predict, text, claim
            )
            better = False
            if verdict in ("support", "refute") and conf >= min_confidence:
                if best["verdict"] not in ("support", "refute") or conf > best["confidence"]:
                    better = True
            elif best["verdict"] == "insufficient" and conf > best["confidence"]:
                better = True
            if better:
                # Round-3 review HIGH fix: store the EXACT window text
                # NLI scored, not just the passage index. Long passages
                # are split into 1800-char windows; previously
                # evidence_span and tier-0 used `passages[best["idx"]]`
                # (the full passage), so they could analyze text NLI
                # never saw (different window of the same passage), or
                # the start of the passage rather than the actual
                # winning span. judge_supports.py audits sees the
                # evidence_span field, so this corrupts the dual-judge
                # signal too.
                best = {
                    "idx": idx, "verdict": verdict, "confidence": conf, "probs": probs,
                    "window_text": text,
                }

        if best["idx"] is None:
            return json.dumps({
                "paper_id": pid, "title": title,
                "verdict": "insufficient",
                "confidence": 0.0,
                "evidence_span": None,
                "page": None,
                "abstain": True,
                "note": "no passage produced a definitive NLI score",
                "tier_0": {
                    "enabled": _tier0_enabled(),
                    "similarity": None,
                    "entity_overlap": None,
                    "action": "no_check",
                    "reason": "no winning passage to score",
                },
            }, indent=2)

        chosen_page, chosen_passage_text, chosen_section = passages[best["idx"]]
        page_label = "printed_page" if pages_verified else "pdf_page"
        # Round-3 review HIGH fix: prefer the winning window over the
        # full passage for tier-0 screening + evidence_span. Falls back
        # to the full passage if window_text is somehow None (shouldn't
        # happen post-fix, but defensive).
        winning_text = best.get("window_text") or chosen_passage_text

        # Phase 5c tier 0 — Semantic-Illusion screen on the WINNING
        # passage window (not the full passage). Plan v3 §3.
        tier_0 = await asyncio.to_thread(_tier0_screen, claim, winning_text)

        # Phase 1.1: widen abstain to (a) include "refute" (a confident
        # contradiction on a caller-supplied target paper is a do-not-cite
        # signal, same as insufficient) and (b) guard against borderline
        # supports by requiring a margin of 0.30 between support and
        # insufficient probabilities. probs order is [refute, support,
        # insufficient].
        # Phase 5c tier 0 (above): also abstain when tier-0 flags a
        # topical-similar/entity-poor match, because NLI's "support"
        # decision is unreliable on that pattern (94% FP rate on v2
        # baseline). Caller can re-enable by setting
        # VERIFY_TIER_0_SCREEN=0 to disable tier-0 entirely.
        support_margin = best["probs"][1] - best["probs"][2]
        tier_0_abstain = (
            tier_0.get("enabled")
            and tier_0.get("action") == "deep_judge_recommended"
            and best["verdict"] == "support"
        )
        abstain = (
            best["verdict"] in ("insufficient", "refute")
            or best["confidence"] < min_confidence
            or (best["verdict"] == "support" and support_margin < 0.30)
            or tier_0_abstain
        )

        return json.dumps({
            "paper_id": pid,
            "title": title,
            "year": year,
            "doi": doi,
            "claim": claim,
            "verdict": best["verdict"],
            "confidence": round(best["confidence"], 4),
            "probs": {
                "refute": round(best["probs"][0], 4),
                "support": round(best["probs"][1], 4),
                "insufficient": round(best["probs"][2], 4),
            },
            "page_label": page_label,
            "page": chosen_page if chosen_page else None,
            "section": chosen_section,
            "evidence_span": winning_text[:1500],
            "abstain": abstain,
            "min_confidence": min_confidence,
            "tier_0": tier_0,
        }, indent=2)
    finally:
        conn.close()


@mcp.tool()
async def search_within_paper(
    paper_id: str,
    query: str,
    mode: str = "passage",
    limit: int = 10,
) -> str:
    """Search for a term/phrase inside a specific paper.

    Use this when you already know which paper contains the claim and want
    to locate the exact page or passage. More precise than corpus-wide
    search_passages when the target paper is known.

    Args:
        paper_id: S2 ID, local:/web:/oa: prefix, or "DOI:10.x/..." .
        query: Plain-text substring to find. Case-insensitive LIKE match
            against passage text. No FTS5 phrase syntax or boolean
            operators; for exact-phrase corpus-wide search use
            `find_quotation` instead. For ranked retrieval across the
            whole library use `search_passages`.
        mode: "passage" (default) searches page-bounded `paper_passages`
            rows (one page per row, Bluebook-grade pincite unit).
            "chunk" searches 500-word `paper_chunks` rows (semantic-
            retrieval unit; pages may span a range).
        limit: Max results (default 10).
    """
    conn = _init_db()
    try:
        # Resolve paper_id (accept DOI prefix / bare DOI like get_full_text).
        row = conn.execute(
            "SELECT paper_id, title, authors, year, venue, doi, local_pdf_path, "
            "pdf_page_offset, pages_verified, is_retracted "
            "FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        if not row and paper_id.upper().startswith("DOI:"):
            doi = paper_id[4:].strip()
            row = conn.execute(
                "SELECT paper_id, title, authors, year, venue, doi, local_pdf_path, "
                "pdf_page_offset, pages_verified, is_retracted "
                "FROM papers WHERE LOWER(doi) = LOWER(?)", (doi,)
            ).fetchone()
        elif not row and "/" in paper_id and not paper_id.startswith("http"):
            row = conn.execute(
                "SELECT paper_id, title, authors, year, venue, doi, local_pdf_path, "
                "pdf_page_offset, pages_verified, is_retracted "
                "FROM papers WHERE LOWER(doi) = LOWER(?)", (paper_id,)
            ).fetchone()
        if not row:
            return f"Paper not found: '{paper_id}'."

        (pid, title, authors_json, year, venue, doi, local_pdf,
         pdf_offset, pages_verified, is_retracted) = row
        try:
            authors = json.loads(authors_json) if authors_json else []
        except json.JSONDecodeError:
            authors = []

        if mode not in ("passage", "chunk"):
            return f"Error: unknown mode '{mode}'. Use 'passage' or 'chunk'."

        q = query.strip()
        if not q:
            return "Error: empty query."

        if mode == "passage":
            # Passage (page-bounded) lookup. paper_passages has no FTS5
            # index yet (small per-paper row count makes LIKE fast
            # enough for in-paper scans); do a direct LIKE scan with
            # fallback-friendly case-insensitive match. This avoids a
            # second FTS5 table and trigger set while keeping correct
            # pincite semantics (one row = one page).
            rows = conn.execute(
                "SELECT passage_id, page_num, section_header, passage_text "
                "FROM paper_passages "
                "WHERE paper_id = ? AND passage_text LIKE ? "
                "ORDER BY page_num LIMIT ?",
                (pid, f"%{q}%", limit),
            ).fetchall()
            if not rows:
                return (
                    f"No page-bounded passages in '{title}' ({year}) contain '{q}'.\n"
                    f"Try mode='chunk' for semantic-unit search, or the paper may "
                    f"not yet have page-bounded passages built (has_full_text=0 or "
                    f"pre-v18 data)."
                )
            parts = [f"Found {len(rows)} passage match{'es' if len(rows) != 1 else ''} in '{title}' ({year}):\n"]
            for i, (passage_id, page_num, section, passage_text) in enumerate(rows, 1):
                label = "printed_page" if pages_verified else "pdf_page"
                parts.append(f"--- Passage {i} ---")
                if section:
                    parts.append(f"Section: {section}")
                if page_num == 0:
                    parts.append(f"{label}: unknown (no page marker)")
                else:
                    parts.append(f"{label}: {page_num}")
                parts.append(f"\n{passage_text}\n")
        else:  # mode == "chunk"
            rows = conn.execute(
                "SELECT chunk_id, chunk_index, section_header, page_start, page_end, chunk_text "
                "FROM paper_chunks "
                "WHERE paper_id = ? AND chunk_text LIKE ? "
                "ORDER BY chunk_index LIMIT ?",
                (pid, f"%{q}%", limit),
            ).fetchall()
            if not rows:
                return f"No chunks in '{title}' ({year}) contain '{q}'."
            parts = [f"Found {len(rows)} chunk match{'es' if len(rows) != 1 else ''} in '{title}' ({year}):\n"]
            for i, (chunk_id, chunk_index, section, ps, pe, chunk_text) in enumerate(rows, 1):
                label = "printed_page" if pages_verified else "pdf_page"
                parts.append(f"--- Chunk {i} ---")
                if section:
                    parts.append(f"Section: {section}")
                if ps is not None:
                    if pe is not None and pe != ps:
                        parts.append(f"{label}: {ps}-{pe}")
                    else:
                        parts.append(f"{label}: {ps}")
                parts.append(f"\n{chunk_text}\n")

        # Header with paper metadata
        header = [f"**{title}** ({year})"]
        if is_retracted:
            header.append("⚠️ **RETRACTED** — do not cite without verification")
        if authors:
            header.append(f"Authors: {', '.join(authors[:5])}")
        if venue:
            header.append(f"Venue: {venue}")
        header.append(f"Paper ID: {pid}")
        if doi:
            header.append(f"DOI: {doi}")
        if local_pdf:
            header.append(f"Local PDF: {local_pdf}")
        header.append("")
        return "\n".join(header + parts)
    finally:
        conn.close()


@mcp.tool()
async def library_stats() -> str:
    """Show statistics about the local paper library."""
    conn = _init_db()
    try:
        total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        with_pdf = conn.execute("SELECT COUNT(*) FROM papers WHERE local_pdf_path IS NOT NULL AND local_pdf_path != ''").fetchone()[0]
        with_fulltext = conn.execute("SELECT COUNT(*) FROM papers WHERE processed_text IS NOT NULL AND processed_text != ''").fetchone()[0]
        open_access = conn.execute("SELECT COUNT(*) FROM papers WHERE is_open_access = 1").fetchone()[0]
        with_abstract = conn.execute("SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL AND abstract != ''").fetchone()[0]
        with_doi = conn.execute("SELECT COUNT(*) FROM papers WHERE doi IS NOT NULL AND doi != ''").fetchone()[0]
        verified_count = conn.execute("SELECT COUNT(*) FROM papers WHERE verified = 1").fetchone()[0]
        unverified_count = conn.execute("SELECT COUNT(*) FROM papers WHERE verified = 0").fetchone()[0]

        year_dist = conn.execute("""
            SELECT year, COUNT(*) as cnt FROM papers
            WHERE year IS NOT NULL
            GROUP BY year ORDER BY year DESC LIMIT 10
        """).fetchall()

        top_fields = conn.execute("""
            SELECT j.value, COUNT(*) as cnt
            FROM papers, json_each(papers.fields_of_study) j
            WHERE fields_of_study IS NOT NULL AND fields_of_study != '[]'
            GROUP BY j.value
            ORDER BY cnt DESC
            LIMIT 10
        """).fetchall()
    finally:
        conn.close()

    lines = [
        f"Local Paper Library: {DB_PATH}",
        f"Total papers: {total}",
        f"With local PDF: {with_pdf}",
        f"With full text (Docling): {with_fulltext}",
        f"Open access: {open_access}",
        f"With abstract: {with_abstract}",
        f"With DOI: {with_doi}",
        f"Verified: {verified_count}",
        f"Unverified: {unverified_count}",
        "",
        "Recent years:",
    ]
    for year, cnt in year_dist:
        lines.append(f"  {year}: {cnt} papers")

    if top_fields:
        lines.append("")
        lines.append("Top fields:")
        for field, cnt in top_fields:
            lines.append(f"  {field}: {cnt}")

    return "\n".join(lines)


@mcp.tool()
async def prune_unverified(dry_run: bool = True) -> str:
    """Delete all unverified papers from the library.

    Unverified papers are citation chain noise: auto-saved metadata without
    PDF, full text, or OpenAlex authority evaluation. They are excluded from
    search_local and search_passages.

    DESTRUCTIVE: permanently removes papers and their chunks.
    Run with dry_run=True first to see what would be deleted.

    Args:
        dry_run: If True (default), only report counts. Set False to execute.
    """
    conn = _init_db()
    try:
        # Prune filter: verified=0 AND has_full_text=0. The has_full_text
        # gate is load-bearing since the wave-2 embed relaxation that
        # admits `verified=0 AND has_full_text=1` merge-keepers as cite-
        # ready rows. Pruning on verified=0 alone would destroy those
        # rows and the chunks, passages, references, and vectors that
        # retrieval now relies on.
        prune_filter = "verified = 0 AND has_full_text = 0"
        to_delete = conn.execute(
            f"SELECT COUNT(*) FROM papers WHERE {prune_filter}"
        ).fetchone()[0]
        to_keep = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE NOT (verified = 0 AND has_full_text = 0)"
        ).fetchone()[0]
        chunks_to_delete = conn.execute(f"""
            SELECT COUNT(*) FROM paper_chunks
            WHERE paper_id IN (SELECT paper_id FROM papers WHERE {prune_filter})
        """).fetchone()[0]

        if dry_run:
            samples = conn.execute(f"""
                SELECT paper_id, title, year FROM papers
                WHERE {prune_filter}
                ORDER BY last_updated DESC
                LIMIT 10
            """).fetchall()

            lines = [
                "DRY RUN - No changes made.",
                f"Papers to DELETE: {to_delete:,}  (verified=0 AND has_full_text=0)",
                f"Chunks to DELETE: {chunks_to_delete:,}",
                f"Papers to KEEP:   {to_keep:,}",
                "",
                "Sample papers that would be deleted:",
            ]
            for pid, title, year in samples:
                t = (title or "")[:80]
                lines.append(f"  - [{year}] {t} ({pid})")
            lines.append("")
            lines.append("Run with dry_run=False to execute.")
            return "\n".join(lines)

        # Delete vector embeddings for papers being pruned.
        # Round-2 review HIGH fix: previously only cleared vec_chunks /
        # vec_papers (nomic). Once the qwen3 backfill landed, prune
        # would leave orphan rows in vec_papers_qwen3_<dim> /
        # vec_chunks_qwen3_<dim>. The v21 cascade triggers fire on
        # DELETE FROM paper_chunks / DELETE FROM papers, but explicit
        # pre-delete is the consistent defense-in-depth pattern.
        for vec_chunks_t in _all_vec_chunk_tables():
            try:
                conn.execute(f"""
                    DELETE FROM {vec_chunks_t} WHERE rowid IN (
                        SELECT c.chunk_id FROM paper_chunks c
                        JOIN papers p ON c.paper_id = p.paper_id
                        WHERE p.{prune_filter}
                    )
                """)
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                if "no such table" not in msg and "no such module" not in msg:
                    raise
        for vec_papers_t in _all_vec_paper_tables():
            try:
                conn.execute(f"""
                    DELETE FROM {vec_papers_t} WHERE rowid IN (
                        SELECT rowid FROM papers WHERE {prune_filter}
                    )
                """)
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                if "no such table" not in msg and "no such module" not in msg:
                    raise
        # Delete chunks, references, and papers (FTS5 triggers handle keyword index cleanup)
        conn.execute(f"""
            DELETE FROM paper_chunks
            WHERE paper_id IN (SELECT paper_id FROM papers WHERE {prune_filter})
        """)
        conn.execute(f"""
            DELETE FROM paper_references
            WHERE citing_paper_id IN (SELECT paper_id FROM papers WHERE {prune_filter})
               OR cited_paper_id  IN (SELECT paper_id FROM papers WHERE {prune_filter})
        """)
        conn.execute(f"DELETE FROM papers WHERE {prune_filter}")
        conn.commit()

        return "\n".join([
            "Prune complete.",
            f"Deleted: {to_delete:,} papers, {chunks_to_delete:,} chunks",
            f"Remaining: {to_keep:,} verified papers",
            f"Run 'sqlite3 {DB_PATH} VACUUM' manually to reclaim disk space.",
        ])
    finally:
        conn.close()


@mcp.tool()
async def verify_paper(paper_id: str) -> str:
    """Manually mark a paper as verified (kept in searches, protected from pruning).

    Args:
        paper_id: The paper ID to verify
    """
    conn = _init_db()
    try:
        row = conn.execute("SELECT title, verified FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        if not row:
            return f"Paper not found: {paper_id}"
        title, already = row
        if already:
            return f"Already verified: {title}"
        conn.execute(
            "UPDATE papers SET verified = 1, last_updated = ? WHERE paper_id = ?",
            (datetime.now(timezone.utc).isoformat(), paper_id),
        )
        # Commit + close the handler's connection BEFORE the CPU-bound
        # to_thread embed so the outer conn can't race conn.close()
        # against the worker thread. The worker owns its own conn.
        conn.commit()
    finally:
        conn.close()
    embed_err: str | None = None
    try:
        embed_err = await asyncio.to_thread(_worker_embed_paper, paper_id)
    except (asyncio.CancelledError, Exception) as e:
        if isinstance(e, asyncio.CancelledError):
            raise
        embed_err = f"{type(e).__name__}: {e}"
        print(f"verify_paper embed failed for {paper_id}: {e}", file=sys.stderr)
    if embed_err:
        return f"Verified: {title} (embedding deferred: {embed_err}; run backfill_embeddings.py)"
    return f"Verified: {title}"


@mcp.tool()
async def set_abstract(paper_id: str, abstract: str) -> str:
    """Set or update the abstract for a paper. Used to store LLM-generated
    summaries for books and papers that lack abstracts.

    Args:
        paper_id: The paper ID
        abstract: The abstract text (500-1000 words recommended)
    """
    conn = _init_db()
    try:
        row = conn.execute("SELECT title, verified FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        if not row:
            return f"Paper not found: {paper_id}"
        title, is_verified = row
        conn.execute(
            "UPDATE papers SET abstract = ?, last_updated = ? WHERE paper_id = ?",
            (abstract, datetime.now(timezone.utc).isoformat(), paper_id),
        )
        conn.commit()
    finally:
        conn.close()
    if is_verified:
        # Handler conn is closed; worker owns its own conn for the embed.
        embed_err: str | None = None
        try:
            embed_err = await asyncio.to_thread(_worker_embed_paper, paper_id)
        except (asyncio.CancelledError, Exception) as e:
            if isinstance(e, asyncio.CancelledError):
                raise
            embed_err = f"{type(e).__name__}: {e}"
            print(f"set_abstract embed failed for {paper_id}: {e}", file=sys.stderr)
        if embed_err:
            return (
                f"Abstract set ({len(abstract)} chars) for: {title} "
                f"(re-embed deferred: {embed_err}; run backfill_embeddings.py)"
            )
        return f"Abstract set ({len(abstract)} chars) and re-embedded for: {title}"
    else:
        return f"Abstract set ({len(abstract)} chars) for: {title} (paper is unverified; embedding skipped until verified)"


def _slugify(text: str, max_len: int = 80) -> str:
    """Generate a URL-safe slug from text. Same logic as acquirer Section 4a."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")[:max_len].rstrip("-")
    return slug or "untitled"


@mcp.tool()
async def store_web_capture(
    url: str,
    title: str,
    text: str,
    author: str = "",
    site_name: str = "",
    authority_tier: int = 4,
    capture_date: str = "",
) -> str:
    """Store a web page capture in papers.db for unified search.

    Web captures are stored alongside academic papers with paper_id prefix
    "web:". They are chunked and embedded for search_local and search_passages.
    Naturally ranked lower than papers (no citations, no hub score).

    Args:
        url: Original URL of the web page
        title: Page title
        text: Extracted article text (markdown or plain text)
        author: Author name (empty if unknown)
        site_name: Site name (e.g., "Brookings Institution")
        authority_tier: Authority tier 1-4 (default 4)
        capture_date: ISO date string (default: today)
    """
    if not url or not title or not text:
        return "Error: url, title, and text are required"
    if len(text.strip()) < 50:
        return f"Error: text too short ({len(text)} chars). Likely a failed extraction."

    slug = _slugify(title)
    paper_id = f"web:{slug}"
    if not capture_date:
        capture_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    conn = _init_db()
    try:
        # Dedup by source_url (non-transactional check; see the try/except
        # around the INSERT below for the race-safe path when two callers
        # hit this concurrently).
        existing = conn.execute(
            "SELECT paper_id FROM papers WHERE source_url = ?", (url,)
        ).fetchone()
        if existing:
            return f"Already stored: {existing[0]} (url: {url})"

        # Dedup by paper_id (slug collision)
        existing = conn.execute(
            "SELECT paper_id FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        if existing:
            # Append hash suffix to avoid collision
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            paper_id = f"web:{slug}-{url_hash}"

        now = datetime.now(timezone.utc).isoformat()
        authors_json = json.dumps([author]) if author else "[]"
        try:
            year = int(capture_date[:4]) if capture_date and len(capture_date) >= 4 else None
        except (ValueError, IndexError):
            year = None

        has_text = 1 if text and len(text) > 500 else 0
        try:
            conn.execute("""
                INSERT INTO papers (
                    paper_id, title, authors, processed_text, source_url,
                    capture_date, work_type, authority_tier, venue, year,
                    verified, has_full_text, first_seen, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, 'web', ?, ?, ?, 1, ?, ?, ?)
            """, (
                paper_id, title, authors_json, text, url,
                capture_date, authority_tier, site_name, year, has_text, now, now,
            ))
        except sqlite3.IntegrityError as e:
            # Concurrent store_web_capture race: another caller already
            # inserted either the same paper_id (PK collision) or the same
            # source_url between our SELECT above and this INSERT. Return
            # the stored pid instead of propagating the IntegrityError.
            conn.rollback()
            row = conn.execute(
                "SELECT paper_id FROM papers WHERE source_url = ? OR paper_id = ?",
                (url, paper_id),
            ).fetchone()
            if row:
                return f"Already stored: {row[0]} (url: {url})"
            raise

        # Chunking is synchronous-cheap (mostly SQLite inserts). Do it before
        # commit so the chunk count reflects the final state.
        n_chunks = _chunk_processed_text(conn, paper_id, text)
        # Rebuild page-bounded passages alongside chunks. Web captures
        # rarely have <!-- page N --> markers from Chrome's PDF engine,
        # so most passages land at page_num=0 (single pseudo-page);
        # that is correct for this content shape.
        _build_paper_passages(conn, paper_id, text)

        # Commit + close BEFORE the CPU-bound embed await so there's no
        # conn-close race against the to_thread worker.
        conn.commit()
    finally:
        conn.close()
    embed_err: str | None = None
    try:
        embed_err = await asyncio.to_thread(_worker_embed_paper, paper_id)
    except (asyncio.CancelledError, Exception) as e:
        if isinstance(e, asyncio.CancelledError):
            raise
        embed_err = f"{type(e).__name__}: {e}"
        print(f"store_web_capture embed failed for {paper_id}: {e}", file=sys.stderr)

    result = (
        f"Stored web capture: {paper_id}\n"
        f"Chars: {len(text):,} | Chunks: {n_chunks}\n"
    )
    if embed_err:
        result += (
            f"Embedding deferred ({embed_err}); run backfill_embeddings.py. "
            f"Keyword search works immediately via FTS."
        )
    else:
        result += "Now searchable via search_local and search_passages."
    return result


@mcp.tool()
async def get_local_references(paper_id: str, direction: str = "both", limit: int = 50) -> str:
    """Traverse the local citation graph. Returns papers in the library
    connected to this paper by citation edges. Instant, no API call.

    The library has ~55K citation edges. This tool finds papers already in your
    library that cite or are cited by the given paper. Use for local citation
    chain following before falling back to get_citations/get_references (S2 API).

    Args:
        paper_id: Paper to find connections for (any format: S2 ID, DOI:, oa:, local:)
        direction: "cites" (papers this one references), "cited_by" (papers citing this), or "both"
        limit: Max results per direction (default 50)
    """
    limit = max(1, min(200, limit))
    conn = _init_db()
    try:
        # Resolve to local paper_id
        resolved = _resolve_to_local_paper_id(conn, paper_id)
        if not resolved:
            resolved = paper_id.strip()
        # Check paper exists
        row = conn.execute("SELECT title FROM papers WHERE paper_id = ?", (resolved,)).fetchone()
        if not row:
            return f"Paper not found: {paper_id}"
        source_title = row[0]

        results = []

        if direction in ("cites", "both"):
            rows = conn.execute("""
                SELECT p.paper_id, p.title, p.year, p.citation_count, p.authority_tier,
                       p.processed_text IS NOT NULL AND p.processed_text != '' AS has_fulltext,
                       pr.is_influential
                FROM paper_references pr
                JOIN papers p ON pr.cited_paper_id = p.paper_id
                WHERE pr.citing_paper_id = ?
                ORDER BY p.citation_count DESC
                LIMIT ?
            """, (resolved, limit)).fetchall()
            for r in rows:
                results.append(("cites", r))

        if direction in ("cited_by", "both"):
            rows = conn.execute("""
                SELECT p.paper_id, p.title, p.year, p.citation_count, p.authority_tier,
                       p.processed_text IS NOT NULL AND p.processed_text != '' AS has_fulltext,
                       pr.is_influential
                FROM paper_references pr
                JOIN papers p ON pr.citing_paper_id = p.paper_id
                WHERE pr.cited_paper_id = ?
                ORDER BY p.citation_count DESC
                LIMIT ?
            """, (resolved, limit)).fetchall()
            for r in rows:
                results.append(("cited_by", r))
    finally:
        conn.close()

    if not results:
        return f"No local citation edges found for: {source_title} ({resolved})"

    parts = [f"Local citations for: {source_title} ({resolved})\n"]
    cites = [(d, r) for d, r in results if d == "cites"]
    cited_by = [(d, r) for d, r in results if d == "cited_by"]

    if cites:
        parts.append(f"**References (this paper cites, {len(cites)} in library):**")
        for _, (pid, title, year, cit, tier, has_ft, influential) in cites:
            yr = str(year) if year else "?"
            ft = " [full text]" if has_ft else ""
            inf = " *influential*" if influential else ""
            t = f"TIER {tier}" if tier else "UNRATED"
            parts.append(f"  - {title} ({yr}) | {cit} cites | {t}{ft}{inf} | {pid}")

    if cited_by:
        parts.append(f"\n**Cited by ({len(cited_by)} in library):**")
        for _, (pid, title, year, cit, tier, has_ft, influential) in cited_by:
            yr = str(year) if year else "?"
            ft = " [full text]" if has_ft else ""
            inf = " *influential*" if influential else ""
            t = f"TIER {tier}" if tier else "UNRATED"
            parts.append(f"  - {title} ({yr}) | {cit} cites | {t}{ft}{inf} | {pid}")

    return "\n".join(parts)


@mcp.tool()
async def backfill_embeddings(dry_run: bool = True) -> str:
    """Show embedding backfill status.

    For bulk backfill, use the standalone script:
      uv run maintenance/backfill_embeddings.py --backfill

    Args:
        dry_run: Ignored (kept for backward compatibility). This tool only reports status.
    """
    conn = _init_db()
    try:
        total_papers = conn.execute("SELECT COUNT(*) FROM papers WHERE verified = 1").fetchone()[0]
        embedded_papers = conn.execute("SELECT COUNT(*) FROM vec_papers").fetchone()[0]
        total_chunks = conn.execute("""
            SELECT COUNT(*) FROM paper_chunks c
            JOIN papers p ON c.paper_id = p.paper_id WHERE p.verified = 1
        """).fetchone()[0]
        embedded_chunks = conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]

        return "\n".join([
            "Embedding status:",
            f"Papers: {embedded_papers:,} / {total_papers:,} embedded",
            f"Chunks: {embedded_chunks:,} / {total_chunks:,} embedded",
            "",
            "To run bulk backfill, use the standalone script:",
            "  uv run maintenance/backfill_embeddings.py --backfill",
        ])
    finally:
        conn.close()


@mcp.tool()
async def check_library_batch(sources_json: str) -> str:
    """Batch check which sources already exist in the local paper library.

    Pure indexed SQL lookups (no FTS5, no API calls). Designed for the /acquire
    skill to dedup hundreds of sources in a single call before spawning agents.

    Args:
        sources_json: JSON array of objects, each with optional keys:
            - s2_id: Semantic Scholar paper ID
            - doi: DOI string
            - title: Paper title
            At least one key must be present per entry.

    Returns:
        JSON array of results, one per input entry:
        [{"index": 0, "matched": true, "paper_id": "abc", "has_fulltext": true, "fulltext_chars": 45000}, ...]
    """
    try:
        sources = json.loads(sources_json)
    except json.JSONDecodeError as e:
        return f"Error: invalid JSON: {e}"

    if not isinstance(sources, list):
        return "Error: sources_json must be a JSON array"

    conn = _init_db()
    try:
        results = []
        for i, src in enumerate(sources):
            s2_id = src.get("s2_id", "")
            doi = src.get("doi", "")
            title = src.get("title", "")

            matched = False
            paper_id = None
            has_fulltext = False
            fulltext_chars = 0

            # Lookup order: paper_id (PK) > doi (indexed, case-insensitive) >
            # title (NOCASE indexed). Full-text length is MAX of processed_text
            # and tex_text so TeX-only papers aren't reported as has_fulltext=0.
            # Order by body length so duplicate-DOI / duplicate-title rows
            # pick the longest-text match (the scalar two-arg MAX returns
            # per-row, so we need an explicit ORDER BY to pick across rows).
            length_expr = "MAX(LENGTH(COALESCE(processed_text, '')), LENGTH(COALESCE(tex_text, '')))"
            row = None
            if s2_id:
                row = conn.execute(
                    f"SELECT paper_id, {length_expr} AS body_len, verified "
                    "FROM papers WHERE paper_id = ? "
                    "ORDER BY body_len DESC LIMIT 1",
                    (s2_id,),
                ).fetchone()
            if not row and doi:
                row = conn.execute(
                    f"SELECT paper_id, {length_expr} AS body_len, verified "
                    "FROM papers WHERE LOWER(doi) = LOWER(?) "
                    "ORDER BY body_len DESC LIMIT 1",
                    (doi,),
                ).fetchone()
            if not row and title:
                row = conn.execute(
                    f"SELECT paper_id, {length_expr} AS body_len, verified "
                    "FROM papers WHERE title = ? COLLATE NOCASE "
                    "ORDER BY body_len DESC LIMIT 1",
                    (title,),
                ).fetchone()

            is_verified = False
            if row:
                matched = True
                paper_id = row[0]
                fulltext_len = row[1] or 0
                has_fulltext = fulltext_len > 500
                fulltext_chars = fulltext_len
                is_verified = bool(row[2])

            results.append({
                "index": i,
                "matched": matched,
                "paper_id": paper_id,
                "has_fulltext": has_fulltext,
                "fulltext_chars": fulltext_chars,
                "verified": is_verified,
            })

        return json.dumps(results)
    finally:
        conn.close()


def main() -> None:
    """Console-script entry point. Runs the FastMCP server over stdio.

    Performs eager schema initialization on startup so a fresh install
    has a usable database and migration head before the first MCP tool
    call lands. Without this the DB is created lazily on the first call,
    which means a quick `library_stats` smoke would otherwise need to
    pay the migration cost at request time.
    """
    conn = _init_db()
    conn.close()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
