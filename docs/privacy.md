# Privacy and security model

Tantular Cyber is designed to analyze code you are authorized to assess **without sending it
anywhere**.

## What runs and what touches the network

| Component | Network | Notes |
|---|---|---|
| `scan` (default, deterministic) | **none** | Reads files, runs the user-installed Semgrep CLI with `--metrics=off --disable-version-check` and bundled local rules, runs the in-process call-graph scanner. CI runs it inside a Linux network namespace with no external interfaces. |
| `scan --ai` | loopback Ollama endpoint by default | Sends redacted evidence excerpts to `http://127.0.0.1:11434`. Non-loopback or credential-bearing URLs are refused, and models Ollama marks as cloud-backed are refused before any chat request, unless `--allow-remote-ollama` is given. HTTP proxy variables are ignored. A loopback endpoint is **not** a guarantee of local inference — see [Residual risks](#residual-risks). |
| `doctor` | loopback Ollama only, read-only | `/api/version`, `/api/tags`, `/api/show` (never for cloud models, which Ollama would forward). Credential-bearing or non-loopback URLs are rejected and not contacted; URLs are shown as scheme://host:port only. Lists every request in its output. `--no-ollama` skips it. Never downloads or pulls, and Tantular Cyber writes no files (the `semgrep --version` subprocess may update Semgrep's own state, e.g. `~/.semgrep`). |
| `sample`, `plugins`, `schema` | none | `sample` copies bundled synthetic files into a directory you name. |
| `benchmark` | none by default | `--llm ollama` verifies the model (including the cloud refusal) before any request; cold-start eviction and the concurrency experiment are skipped if the check fails. `--real-world` reads a local cache only. |
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

## Residual risks

Accepted limitations, documented rather than fixed in POC-006:

* **Loopback ≠ local inference.** Tantular Cyber can only see what Ollama reports. Cloud
  models are detected through the `remote_host` / `remote_model` fields of `/api/tags`; if a
  future Ollama version renames or drops them, detection fails open. A loopback port that is
  actually a proxy or tunnel to another machine (SSH forward, LiteLLM, a different server on
  port 11434) cannot be detected by any client. Only use `--ai` with an Ollama you control.
* **One consent flag.** `--allow-remote-ollama` / `TANTULAR_ALLOW_REMOTE_OLLAMA` accepts both
  a remote endpoint and cloud-backed models; there is no separate cloud-only switch yet.
* **External tool state.** Semgrep (and Ollama) keep their own configuration, logs and
  caches; Tantular Cyber's "no writes" guarantees cover only its own code.
* **Release engineering** (tracked for a later milestone):
  * release smoke jobs install `pip` and Semgrep from PyPI without hash pinning (they run
    with read-only permissions and cannot modify the built artifacts);
  * `SHA256SUMS` travels with the artifacts and detects corruption, not tampering — the
    GitHub build-provenance attestation is the integrity anchor;
  * the older `test` CI job uses tag-pinned actions (the new jobs are SHA-pinned);
  * the reproducibility check rebuilds on the same runner (determinism, not independent
    reproduction);
  * the SBOM omits per-dependency hashes and the license audit rejects compound SPDX
    expressions (fails closed).
* **Local builds.** Wheel data is force-included from `benchmarks/` and `tests/fixtures/`,
  which ignores `.gitignore`; a build from a working copy can pick up stray untracked files.
  Build releases only from a clean checkout (CI and the release workflow always do).

## Your responsibilities

* Scan only code you are authorized to assess.
* Treat findings as unverified evidence; no exploitation is attempted or claimed.
* If you enable `--allow-remote-ollama`, redacted excerpts leave your machine.
