# Changelog

## Unreleased

### Added
* File-level Jinja2 autoescape configuration check in tantular-callgraph, rule
  `tantular.callgraph.xss-jinja2-autoescape-extension` (CWE-79, WARNING / LOW): an
  `Environment(autoescape=select_autoescape(...))` whose literal settings leave HTML-like
  template names in the same file unescaped (e.g. `*.html.j2`). Mirrors Jinja's
  case-insensitive suffix rules incl. `disabled_extensions` and `default`; recognises only
  real `jinja2` imports/aliases; skips non-literal settings (counted). Configuration warning
  without a call path; the cross-Environment false positive is documented. Secret-shaped
  template names are withheld from messages (count only); per-file budgets on names,
  Environments and message size, with limits counted in scanner stats. Corpus: 9 new
  cases, 10 annotations; the FN annotation `xss-jinja-template-known-fn-extension` is
  renamed `xss-jinja-template-extension-mismatch` and is now TP. Benchmark
  45/9/5/25 → 49/10/6/29 (corpus growth; not a quality claim). Report schema unchanged.
* Rule `tantular.python.xss.jinja2-template-autoescape-disabled` (CWE-79, WARNING / LOW):
  `jinja2.Template(source)` constructed directly without `autoescape=True` (it ignores any
  Environment's setting and does not escape). Found as a missed, reviewed likely-real stored
  XSS in a real-repository evaluation; corpus case `xss-jinja-template` (2 TP, 2 TN incl. a
  near miss, 1 known FP for plain-text templates, 1 known FN for `select_autoescape` vs
  `*.html.j2`). Benchmark: TP/FP/FN/TN 43/8/4/23 → 45/9/5/25; the corpus grew, so the
  aggregate change is neither a regression on the unchanged cases nor an overall quality
  claim. Report schema unchanged.
* Real-world manifest 1.1.0: `development_repositories` marks whole repositories as
  development-used (never held out). `ghifiardi/ai-driven-soc` is listed from 2026-11-12.
* docs/rules.md: real-world evaluation backlog separating detection gaps from
  false-positive reductions, with coverage limits.

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
