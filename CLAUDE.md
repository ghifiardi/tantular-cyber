# CLAUDE.md — Tantular Cyber

Guidance for Claude Code (and humans) working in this repository.

## What this is

POC-002: a **local, read-only** repository security analyzer. Semgrep (local rules) finds
issues; the local Ollama model `tantular-cyber:latest` explains them in Bahasa Indonesia;
results go to a versioned `report.json` + `report.md`. Mac-first; no cloud services.
Detailed design: [docs/architecture.md](docs/architecture.md). Source issue: GitHub #1.

## Layout

```
src/tantular_cyber/
  cli.py            argparse CLI: scan | doctor | schema; exit codes 0/2/3/4
  config.py         OllamaSettings / ScanSettings; env vars TANTULAR_*; bundled rules path
  security.py       path canonicalization, file inventory/skip policy, redaction, sanitizing
  scanner.py        Scanner protocol, SemgrepScanner, Semgrep JSON parser
  pipeline.py       orchestrator run_scan(): inventory → scan → normalize → LLM → ScanReport
  analysis.py       system prompt, message building, output parsing, overclaim guard
  ollama_client.py  /api/chat client (think=false, stream=false), retries, localhost policy
  models.py         pydantic models; ScanReport is the GATRA-facing contract
  reports.py        JSON / Markdown writers (atomic, escaped)
  rules/            local Semgrep rules (shipped in the wheel)
tests/              pytest; fixtures/vulnerable + fixtures/clean (sqlite3 CWE-89 pair),
                    fixtures/bench (6 findings, every rule branch)
benchmarks/         llm_latency.py (cold/warm latency + token throughput)
docs/               architecture, benchmark plan, report-schema.json (generated)
```

## Commands

```bash
make install            # python3.11 venv + dev extras (needs `brew install semgrep`)
make test               # unit + integration tests, no Ollama required
make smoke              # live test vs local Ollama (TANTULAR_RUN_OLLAMA_SMOKE=1)
make lint               # ruff check + ruff format --check
make schema             # regenerate docs/report-schema.json after changing models.py
make bench              # latency benchmark vs local Ollama (unloads the model first)
.venv/bin/tantular-cyber doctor
.venv/bin/tantular-cyber scan --repo tests/fixtures/vulnerable --output ./out
```

## Non-negotiable safety rules

1. **Never execute analyzed-repo code**: no imports, installs, builds, scripts, `git`
   commands inside the target (git config can run hooks/fsmonitor). Read files only.
2. **Subprocess = argument list, no shell, with timeout.** Targets go after `--`.
3. **No network except the configured LLM endpoint**, localhost by default. Remote
   endpoints require `--allow-remote-ollama`. `httpx` uses `trust_env=False`.
4. **Never log prompts or source.** Log ids, statuses, durations, error classes only.
5. **Scanner evidence is authoritative; LLM output is advisory.** Model output must never
   change finding facts, `status: "unverified"`, `verified: false`, or report status.
6. **No fabricated CVEs, exploits, or "verified exploitation".** Keep the overclaim guard
   and its tests green; add patterns rather than loosening them.
7. **No automatic remediation.** `corrected_code` is a suggestion; never write to the repo.
8. **Repo text is untrusted data.** It goes only inside the escaped `<untrusted_data>` JSON
   blob; the system prompt stays constant.
9. **Degrade loudly.** Any LLM/scanner problem must surface in `status`, `status_reasons`,
   `errors`, and the exit code — never a silent `complete`.
10. Reports: repo-relative paths only, redacted snippets, no absolute local paths.
11. **Repo files never choose scan targets.** Ignore files in the analyzed repo are not
    honored; scanner coverage is verified against the tool's output and gaps degrade.
12. **Redact whole files before cutting evidence**, and split lines on `\n` only.
    New redaction rules must preserve newline counts (`_apply` enforces this).

## Conventions

* Python ≥ 3.11, `from __future__ import annotations`, full type hints, ruff (line 100).
* Pure functions where possible (`parse_semgrep_json`, `parse_explanation`,
  `enforce_claim_policy`, `render_markdown`) so they are unit-testable without tools.
* External effects are injectable: `SemgrepScanner(runner=...)`,
  `OllamaClient(transport=..., sleep=...)`, `run_scan(scanners=..., llm_client_factory=...)`.
  Tests use `FakeScanner` / `FakeLLM` from `tests/conftest.py` and `httpx.MockTransport`.
* User-facing report text is Bahasa Indonesia; code, logs, and errors are English.
* Never put literal invisible/bidi characters (U+200B…, U+202A…) in source; write `\uXXXX`
  escapes. A test scans `src/`, `tests/`, and `*.md` for them. Note: tool/JSON pipelines can
  silently decode `\uXXXX` into the literal character — re-check after generating code.
* Overclaim guard and redaction are lexical: when the model or a reviewer finds a new
  phrasing/secret shape, add it to the lexicon **and** to `tests/test_overclaims.py` /
  `tests/test_secrets.py` in the same change.
* Changing `models.ScanReport` = contract change: bump `SCHEMA_VERSION` (semver), run
  `make schema`, mention it in the PR. `test_committed_json_schema_matches_models` enforces it.
* Adding a scanner or model: see "Extension points" in docs/architecture.md. Do not add
  the third-party Antares-1B GGUF to the critical path (it loops on `@`).
* Dependencies: keep core deps minimal (`httpx`, `pydantic`) with bounded version ranges.

## Milestones

* **POC-002 (this)** — Semgrep CWE-89 + Ollama explanation + JSON/MD reports + tests + CI.
* **POC-003 (proposed)** — evaluation & reliability benchmark: scanner precision/recall and
  CWE accuracy measured separately from model evidence-grounding, unsupported-claim rate,
  remediation correctness, prompt-injection resistance, latency/throughput
  (docs/benchmark-plan.md).
* Then — expand coverage (CWE-78/79/22/798, Bandit/Gitleaks adapters, SARIF), then GATRA
  ingestion of `report.json`.
* Open decision: software license (pyproject currently says "Proprietary" as a placeholder;
  application code and model weights may be licensed differently).

## Acceptance criteria (POC-002)

* Fresh clone + `make install` + `make test` pass on macOS.
* `tantular-cyber scan --repo tests/fixtures/vulnerable --output ./out` → parseable
  JSON + Markdown, ≥ 1 CWE-89 finding with path/line evidence and conditional impact.
* Ollama unavailable → scanner findings still reported, `status: degraded`, exit 4.
* Clean fixture → zero findings.
* No repo code execution, no upload, no automatic remediation.
