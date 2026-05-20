# Contributing

Short version: issues and PRs welcome. This is currently a one-person
project with no commercial backing, so review cadence is real-time when
I'm around and zero when I'm not. Please be patient.

## Before you open a PR

1. **Open an issue first** for anything larger than a bug fix or
   small polish. Coordinating before code is written saves both of us
   time when the design isn't a fit.
2. **Run ruff.** Config is in `pyproject.toml`. `uvx ruff check .`
   should be clean (or install ruff into your environment of choice).
3. **Make sure `uv run server.py` boots** on a fresh `$HOME` after your
   changes. The clean-room test:

   ```bash
   mkdir -p /tmp/rmcp-test
   env -i PATH="$PATH" HOME=/tmp/rmcp-test uv run server.py &
   sleep 8
   sqlite3 /tmp/rmcp-test/.local/share/research-mcp/papers.db \
       "SELECT version FROM schema_version;"
   kill %1
   rm -rf /tmp/rmcp-test
   ```

   If the DB doesn't get created or the schema migration fails, your
   change broke startup. Fix before submitting.

## What's in scope

- new MCP tools that fit the "local academic library" theme
- new download backends (see [DOWNLOAD_BACKENDS.md](DOWNLOAD_BACKENDS.md))
- additional embedding / reranker variants
- maintenance and audit scripts
- documentation
- portability fixes (Windows, less-common Linux distros)

## What's out of scope

- shadow-library integrations in the main download chain (see
  [DOWNLOAD_BACKENDS.md](DOWNLOAD_BACKENDS.md) for the interface to
  fork-and-extend)
- multi-user / hosted-service architecture (this is a single-user
  local tool by design)
- alternative storage backends (Postgres, Elastic, etc.) — sqlite-vec
  is load-bearing
- LLM-backed tools beyond the existing NLI verifier (this is meant to
  be a retrieval primitive, not an agent)

If you're unsure whether your idea fits, open an issue and ask. No
hard feelings either way.

## Commit messages

Conventional-ish. The first line is a short imperative ("add foo",
"fix bar in baz"). If the change is non-trivial, the body explains
the *why*. The diff explains the *what*.

## Schema migrations

If you add a column or table, write a migration in
`_run_schema_migrations()` and bump `_SCHEMA_VERSION`. Migrations
must be idempotent (safe to re-run) and forward-only (no `DROP COLUMN`
without an explicit migration that backs up the data).

## Acquisition pipeline changes

Verify changes against a real OA paper end-to-end before submitting:

```bash
# in a fresh HOME
uv run server.py &
# from another shell with an MCP client connected:
> download paper 10.1038/s41586-020-2649-2
> find_quotation "spike protein" <returned paper_id>
> verify_claim "the spike protein has ..." <returned paper_id>
```

The first call should succeed and produce a non-empty file in
`$PAPERS_DIR`. The verification round-trip should return a real verdict
(not an error).

## License

By contributing, you agree your contributions are licensed under the
project's MIT license (see [LICENSE](../LICENSE)).
