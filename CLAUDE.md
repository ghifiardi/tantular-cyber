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
  context.py        bounded AST-aware evidence context + dataflow hints (window fallback)
  callgraph.py      tantular-callgraph: bounded same-repo interprocedural scanner (POC-005)
  analysis.py       system prompt, message building, output parsing, overclaim guard
  ollama_client.py  /api/chat client (think=false, stream=false), retries, localhost policy
  models.py         pydantic models; ScanReport is the GATRA-facing contract
  reports.py        JSON / Markdown writers (atomic, escaped)
  rules/            local Semgrep rules, one file per CWE (89/78/22/79/798; docs/rules.md)
  benchmark/        evaluation: manifest, matching, adversarial, llm_eval, perf, provenance,
                    results, render, runner, context_eval, realworld (pointers), review
                    (human ratings/kappa), compare (models/profiles)
tests/              pytest; fixtures/vulnerable + fixtures/clean (sqlite3 CWE-89 pair),
                    fixtures/bench (6 findings, every rule branch)
benchmarks/corpus/  benchmark corpus: manifest.json (ground truth), cases/ (never run),
                    adversarial/ (overclaim + secret suites)
benchmarks/replays/ recorded raw model outputs (real runs) for deterministic replay
benchmarks/realworld/ pointer-only manifest of upstream vulnerable locations (no code)
docs/               architecture, rules, benchmark guide + results, report-schema.json
```

## Commands

```bash
make install            # python3.11 venv + dev extras (needs `brew install semgrep`)
make test               # unit + integration tests, no Ollama required
make smoke              # live test vs local Ollama (TANTULAR_RUN_OLLAMA_SMOKE=1)
make lint               # ruff check + ruff format --check
make schema             # regenerate docs/report-schema.json after changing models.py
make bench              # offline deterministic benchmark (what CI gates on)
make bench-replay       # LLM-evaluation path with recorded outputs, no model
make bench-live         # opt-in: local Ollama quality + cold/warm perf + concurrency 1,2
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

* **POC-002 (done)** — Semgrep CWE-89 + Ollama explanation + JSON/MD reports + tests + CI.
* **POC-003 (done)** — `tantular-cyber benchmark`: scanner detection (deterministic, gated)
  separate from LLM explanation quality (sample-based, never gated), adversarial suites,
  performance with honest cold/warm labels. Guide: docs/benchmark.md.
* **POC-004 (done)** — CWE-78/22/79/798 rules (scoped, docs/rules.md), CVE-informed
  synthetic fixtures with provenance, per-CWE gates, bounded AST-aware context.
* **POC-005 (this)** — tantular-callgraph, chain patterns, real-world pointers, human-review
  ingestion, controlled model/profile comparison (docs/evaluation.md).
* Then — real-world sanitized evaluation, human-review score ingestion, then GATRA
  ingestion of `report.json` once detection quality is measured.
* Application source code is licensed under Apache-2.0 (see LICENSE). Model weights are separate and retain their own applicable license.

## Benchmark conventions (POC-003)

* Ground truth lives only in `benchmarks/corpus/manifest.json`; case files must not contain
  labels (the model reads them). Never execute or import case files.
* Deterministic results gate CI; LLM and performance numbers are reported, never gated.
* Changing scanner behavior on the corpus = update `expect_detected` in the manifest in the
  same change (an improvement fails the gate until acknowledged).
* Blind spots are data, not secrets: record them with `known_blind_spot: true` + a note.
* Benchmark output schema (`benchmark_schema_version`) is independent of `report.json`.
* New rules need corpus cases first-class: TP, TN, near miss, known FP/FN and a `per_cwe`
  floor (`require_cwe_category_coverage`); document sinks/gaps in docs/rules.md.
* Fixture licensing: independently written only. CVE advisories are references (record
  id, https URL, own-words summary, uncertainty); never copy third-party code, tests,
  payloads or prose unless the license allows it and attribution is recorded
  (`copied_third_party_code`, `third_party_license`, `attribution`). Synthetic secrets only
  (`tc-synthetic-…`), listed as `redaction_canaries`.
* Context: parse the original text, display only redacted lines, keep budgets hard; never
  present dataflow hints as proof of taint.
* Callgraph: resolve conservatively — an unresolved or ambiguous call must never create a
  finding; every finding carries `call_path` and `uncertainty`. Semgrep keeps precedence.
* Real-world data is pointer-only (hashes, no code text); fetching is an explicit command,
  never part of CI or tests (tests use httpx.MockTransport). Mark `development_use` when a
  pointer influenced rules — it is no longer held out.
* Human reviews: pseudonyms only, no imputation, notes never published, never CI gates.
* Model comparisons pin all sampling options and record weights/quantization provenance;
  `tantular-cyber:latest` is a profile of `ghifidanukusumo/tantular` weights.

## Acceptance criteria (POC-002)

* Fresh clone + `make install` + `make test` pass on macOS.
* `tantular-cyber scan --repo tests/fixtures/vulnerable --output ./out` → parseable
  JSON + Markdown, ≥ 1 CWE-89 finding with path/line evidence and conditional impact.
* Ollama unavailable → scanner findings still reported, `status: degraded`, exit 4.
* Clean fixture → zero findings.
* No repo code execution, no upload, no automatic remediation.
