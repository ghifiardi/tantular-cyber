# Privacy and security model

Tantular Cyber is designed to analyze code you are authorized to assess **without sending it
anywhere**.

## What runs and what touches the network

| Component | Network | Notes |
|---|---|---|
| `scan` (default, deterministic) | **none** | Reads files, runs the user-installed Semgrep CLI with `--metrics=off --disable-version-check` and bundled local rules, runs the in-process call-graph scanner. CI runs it inside a Linux network namespace with no external interfaces. |
| `scan --ai` | loopback Ollama only by default | Sends redacted evidence excerpts to `http://127.0.0.1:11434`. Non-loopback URLs and Ollama cloud models (`*:cloud`) are refused unless `--allow-remote-ollama` is given. HTTP proxy variables are ignored. |
| `doctor` | loopback Ollama only, read-only | `/api/version`, `/api/tags`, `/api/show` (never for cloud models, which Ollama would forward). Lists every request in its output. `--no-ollama` skips it. Never downloads or pulls. |
| `sample`, `plugins`, `schema` | none | `sample` copies bundled synthetic files into a directory you name. |
| `benchmark` | none by default | `--llm ollama` uses the local endpoint; `--real-world` reads a local cache only. |
| `realworld fetch` | GitHub (explicit command) | Downloads pointed-to public files into `~/.cache/tantular-cyber/realworld`, verifying hashes. Never part of scans or CI. |
| `compare` | loopback Ollama | Research tool; skips cloud models. |

There is no telemetry, analytics, login, licence server or update check.

## Guarantees in code (tested)

* Scanned code is **never executed or imported**; no `git` commands run inside the target.
* Subprocesses use argument lists without a shell and with timeouts; targets follow `--`.
* Reports contain repo-relative paths and **redacted** snippets; secrets are masked before
  evidence is cut; prompts and source are never logged.
* Scanner findings are authoritative; model output is advisory and cannot change findings,
  their `unverified` status or the report status.
* A scan that is not fully covered (Semgrep missing, scanner failure, unsupported platform,
  requested AI stage unavailable) is reported `degraded` or `failed` — never `complete`.
* Ignore files inside the scanned repository are not honoured (repo files never choose scan
  targets).
* No automatic remediation: suggested code is text in the report; the repository is never
  written to.
* Third-party scanner plugins are **not executed** (no enforceable sandbox yet; see
  [plugin-api.md](plugin-api.md)).

## What stays on disk

Only the report directory you pass with `--output`, and the optional real-world cache
(`~/.cache/tantular-cyber/realworld`, delete with `tantular-cyber realworld clean`). Semgrep
and Ollama keep their own caches and settings under your home directory; consult their
documentation.

## Your responsibilities

* Scan only code you are authorized to assess.
* Treat findings as unverified evidence; no exploitation is attempted or claimed.
* If you enable `--allow-remote-ollama`, redacted excerpts leave your machine.
