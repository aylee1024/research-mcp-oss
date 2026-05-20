# Download backends

## Why this doc exists

research-mcp ships with a small set of download sources whose
redistribution licenses are widely understood:

- **Open Access via Unpaywall and Semantic Scholar's `openAccessPdf`** —
  papers the publisher has explicitly flagged as free in OA metadata.
- **arXiv direct** — preprints whose authors uploaded to the public
  preprint server.
- **Direct URLs** — author personal sites and institutional repositories
  that point to a PDF the user (or the upstream metadata) has reason to
  believe is shareable.

That set covers the easy cases. It does not cover:

- papers behind paywalls that you do have institutional access to,
- private corpora your organization owns,
- the PMC Open Access Subset (which is redistributable, but the
  upstream chain doesn't ship a verifier that checks OA status before
  download, so PMC was excluded from v0.1.0),
- regional archives like J-STAGE or KoreaScience,
- shadow libraries (sci-hub, Anna's Archive, LibGen, Z-Library) in
  jurisdictions where they are legal,
- preprint mirrors outside arXiv (SSRN, ResearchGate, etc.).

This document describes how to add your own source by editing the
chain in `acquire/acquire_batch.py`. The shipped chain is the
reference implementation; new sources follow the same shape.

## What we will not ship

We will not accept PRs that add shadow-library integrations (sci-hub,
Anna's Archive, LibGen, Z-Library, etc.) to the main download chain.
This is a project policy, not a legal opinion. The cleanest posture
for a single-maintainer open-source project is to publish only the
architecture and let researchers in jurisdictions where these sources
are legal plug them in via their own fork.

The U.S. secondary-liability framework for distributing tools that can
be used to infringe is set out in *MGM Studios, Inc. v. Grokster, Ltd.*,
545 U.S. 913 (2005), if you want to read the relevant case law
yourself. This document does not characterize what *Grokster* would
mean for any specific integration; only you and your counsel can do
that.

If you want one of these integrations for your own use, see the
chain shape below. Forks are welcome; upstream merges are not.

This document is not legal advice. Users are responsible for compliance
with copyright law in their jurisdiction.

## The chain shape

A new download source is a new `elif` branch inside the per-paper loop
in `acquire/acquire_batch.py`'s `main()`. Every branch follows the
same skeleton; you can read the OA, arXiv, and direct-URL branches as
reference implementations. The minimum a branch must do:

```python
# ── Try my-source ────────────────────────────────────────────────
if status == "downloaded":
    _record_method(methods_attempted, "mysource", "skipped_after_success",
                   "Downloaded earlier in chain")
elif claim_lost:
    _record_method(methods_attempted, "mysource", "skipped_claim_lost",
                   "Claim lost")
elif <your-preconditions>:
    try:
        async with httpx.AsyncClient(
            timeout=60, follow_redirects=True,
            headers={"User-Agent": BROWSER_UA},
        ) as dl:
            resp = await dl.get(<your-url>)
        if resp.status_code == 200 and resp.content[:5] == b"%PDF-":
            tmp = staging / _staging_name("mysource", paper_id, fallback="mysource")
            tmp.write_bytes(resp.content)
            try:
                result_text, chars_my, ok = await _process_and_verify(
                    str(tmp), paper_id, target_doi=doi, target_title=title,
                    target_authors=_authors_for_verify(paper_id, meta.get("authors")),
                )
                if ok:
                    status = "downloaded"
                    chars = chars_my
                    quality_flag = "mysource"
                    _record_method(methods_attempted, "mysource", "success",
                                   result_text, first_line=True)
                else:
                    _record_method(methods_attempted, "mysource", "failed",
                                   result_text)
            finally:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            _record_method(methods_attempted, "mysource",
                           f"http_{resp.status_code}" if resp.status_code != 200 else "not_pdf",
                           f"HTTP {resp.status_code} or not PDF")
    except Exception as e:
        _record_method(methods_attempted, "mysource", "failed",
                       f"my error: {e}")
else:
    _record_method(methods_attempted, "mysource", "skipped_no_preconditions",
                   "preconditions not met")
```

Insertion point: put your branch in the chain at a position that
matches its success profile. The current order is OA → arXiv → direct
URL; cheaper/faster sources earlier, slower or more-restricted sources
later.

If your source is slow (>10s per request), insert a
`_refresh_paper_claim` call before and after — the existing branches
do this between every step so a single paper that takes more than an
hour doesn't lose its worker claim mid-flight.

## Verification

This is the part most people get wrong. Downloading a PDF for "the
right title" but ending up with the wrong paper is alarmingly common
(shared filenames, redirects, mislabeled mirrors). The shipped chain
runs `_process_and_verify` after every download, which:

1. Extracts the PDF via Docling.
2. Checks the title appears in the first ~10KB of extracted text
   (case-insensitive, whitespace-normalized).
3. Checks at least one author surname appears in the same window.
4. Rolls back the transaction if either check fails, leaving the DB
   row untouched and deleting the staged PDF.

Use this helper. If you write your own verifier, match its rollback
semantics or you'll leave inconsistent state behind on verify-fail.

## Examples to follow

The three shipped branches in `acquire/acquire_batch.py` illustrate the
range:

- **OA path** (`oa_url` block) — the simplest case: single GET, magic-
  byte check, hand off to verifier.
- **arXiv path** — single GET to `https://arxiv.org/pdf/{arxiv_id}`.
- **Direct URL path** — same shape as arXiv, URL comes from chunk
  metadata.

Read those three and your own backend will almost write itself.

## What not to do

- **Don't bypass the verifier.** "I'm sure the URL is right" is
  exactly when the wrong paper sneaks in.
- **Don't add a branch by editing the chain to skip verification** —
  fix your branch instead.
- **Don't put credentials in code.** Use env vars (the shipped chain
  reads `S2_API_KEY`, `OPENALEX_API_KEY`, etc.).
- **Don't hit upstream servers without throttling.** Use
  `_Throttle(rate_seconds)` from `server.py`. Polite throttling is
  on the order of 1 request per second; more aggressive sources
  warrant slower rates.
- **Don't trust file extensions or HTTP `content-type`.** Always check
  for the `%PDF-` magic bytes on the first few bytes of the response.

## Contributing your backend

If your backend is broadly useful and legally clean (institutional
proxies for accredited research, regional archives, additional OA
aggregators), open an issue describing the source and we'll discuss
whether it fits in the upstream chain or belongs in an `extras/`
directory.

For backends that integrate with copyright-gray sources, please keep
them in your own fork. The chain shape is stable; we just won't merge
them upstream.
