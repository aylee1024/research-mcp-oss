# Security policy

## Reporting a vulnerability

If you find a security issue in research-mcp, please report it
privately. Do not open a public issue.

**Preferred channel**: open a draft security advisory through GitHub's
private vulnerability reporting at
[github.com/aylee1024/research-mcp-oss/security/advisories/new](https://github.com/aylee1024/research-mcp-oss/security/advisories/new).
This routes directly to the maintainer and is invisible to the public
until the advisory is published.

**Fallback**: if private advisories are not available in your context,
open a regular issue describing the *category* of the bug (e.g.,
"input handling in process_pdf") without attaching the proof of
concept. The maintainer will follow up via email.

## Scope

In scope:

- arbitrary file read / write via path traversal in `process_pdf`,
  `process_tex`, or the acquire pipeline
- SQL injection in any of the MCP tool implementations
- shell injection via subprocess workers (`process_pdf.py`, `process_tex.py`)
- secret exposure via tool output (e.g., echoing env-var-sourced API keys)
- denial of service via malformed inputs that crash the long-running
  server

Out of scope:

- denial of service via resource exhaustion (the server is single-user
  by design; DOS your own server all you want)
- third-party-dependency vulnerabilities — please report those upstream
  (Docling, sqlite-vec, transformers, etc.)
- vulnerabilities in shadow-library or other custom download backends
  the user wired in via the documented backend interface — those are
  outside the upstream chain

## Response time

This is a single-maintainer project. I aim to acknowledge reports
within seven days and to land a fix within thirty days for issues
that are exploitable. If the fix takes longer, I will keep the
reporter informed.

## Coordinated disclosure

If you would like credit on the published advisory, please tell me
your preferred name and any URL you would like linked.
