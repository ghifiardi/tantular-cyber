# Changelog

## 0.3.0 — POC-006: downloadable local-first CLI (unreleased)

### Changed (compatibility)
* `scan` is **deterministic by default** (Semgrep + tantular-callgraph). AI explanations are
  opt-in with `--ai`. A new user can scan without Ollama or any model.
* `--no-llm` is **deprecated**: it is accepted (prints a note) because deterministic is now
  the default, and will be removed in a future release. `--ai --no-llm` is rejected.
  Scripts that relied on explanations by default must add `--ai`.
* Report schema stays **1.3.0** (no field changes). The deterministic-only limitation text
  in `limitations` now reads "Deterministic-only scan: AI explanations were not requested".
* `doctor` was rewritten: read-only diagnostics (`--json` for machines), loopback-only Ollama
  checks, Semgrep version range, platform (macOS, Linux, WSL2; native Windows unsupported),
  scan capability `full` / `partial` / `unavailable`, exit codes 0 / 4 / 3. Ollama being
  unavailable no longer makes `doctor` fail: it only affects the optional `--ai`.
* `benchmark`, `compare` and `realworld` default to the corpus/manifest bundled in the wheel.
* Model comparison output schema 1.2.0 (`ModelProvenance.remote`).

### Added
* `tantular-cyber sample --dest DIR` copies a small synthetic vulnerable project to scan.
* `tantular-cyber plugins list|validate`: scanner plugin API 1.0 contract (manifest,
  negotiation, output normalization). **Third-party plugin execution is disabled** until an
  enforceable sandbox exists.
* Scan summary prints `coverage: full|PARTIAL` and whether AI explanations were requested.
* Hash-locked runtime and build requirements, reproducible builds, CycloneDX SBOM, license
  audit, SHA-256 checksums and a tag-triggered draft-release workflow with build provenance
  attestation.
* Docs: install guide with platform matrix, quickstart (English and Bahasa Indonesia),
  privacy model, plugin API, release verification, third-party notices.

### Security
* Ollama **cloud models** (`*:cloud`, entries with `remote_host`) are refused by `scan --ai`
  unless `--allow-remote-ollama` is given: a loopback URL would otherwise still send code
  excerpts off the machine. `doctor` and `compare` never call `/api/show` for such models,
  because Ollama forwards that request to the remote host.
* Native Windows (and other unsupported platforms) can never produce a `complete` report.
* `OllamaClient.chat` verifies the model (presence + cloud refusal) before its first request,
  and `benchmark --llm ollama` checks the model before cold-start eviction and the
  concurrency experiment (previously these could reach a cloud model).
* `doctor` rejects credential-bearing Ollama URLs (never contacted) and displays endpoints as
  scheme://host:port only.
* Residual risks (loopback is not proof of local inference, release-engineering gaps) are
  documented in docs/privacy.md.
