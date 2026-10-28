# Architecture — Tantular Cyber POC-002

## Pipeline

```
 authorized local repo (read-only)
            │
            ▼
 ┌──────────────────────┐   security.py: canonicalize paths, reject output-inside-repo,
 │ 1. Inventory          │   skip .git / deps / caches / secret-like files / binary /
 └──────────┬───────────┘   oversized / symlinks (escaping symlinks flagged separately)
            ▼
 ┌──────────────────────┐   scanner.py: Scanner protocol; SemgrepScanner runs
 │ 2. Scanners           │   `semgrep scan --json --metrics=off --config <local rules> -- <files>`
 └──────────┬───────────┘   (argument list, no shell, timeout, cwd=repo, SEMGREP_APP* stripped)
            ▼
 ┌──────────────────────┐   pipeline.py: repo-relative paths, evidence read from the file
 │ 3. Normalize          │   (exact lines + line numbers, redacted), CWE, severity,
 └──────────┬───────────┘   deterministic id, status "unverified", dedupe, sort
            ▼
 ┌──────────────────────┐   analysis.py + ollama_client.py: one /api/chat call per finding,
 │ 4. LLM explanation    │   think=false, stream=false, JSON-schema `format`; output validated,
 │    (advisory)         │   sanitized, overclaims relocated, CVEs stripped
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐   reports.py: report.json (versioned schema) + report.md,
 │ 5. Reports            │   optional <tool>-raw.json (paths scrubbed)
 └──────────────────────┘
```

Status semantics (`report.status`, mirrored in the CLI exit code):

| status | exit | meaning |
|---|---|---|
| `complete` | 0 | all scanners succeeded; every finding explained (or `--no-llm`) |
| `degraded` | 4 | scanner findings present, but LLM unreachable / invalid output / limit hit |
| `failed` | 3 | every scanner failed — findings are missing; report still written |
| — | 2 | invalid input paths or rule files; no report written |

## Trust boundaries

| Data | Trust | Handling |
|---|---|---|
| Scanner result (rule, CWE, severity, path, lines) | authoritative | never modified by the LLM |
| Repository file content | untrusted | read-only; redacted before leaving the process; sent to the LLM only as an escaped JSON blob inside `<untrusted_data>` |
| Scanner messages | untrusted | sanitized (control/bidi chars stripped, secrets redacted, length-bounded) |
| LLM output | untrusted, advisory | JSON-schema validated, sanitized, overclaim guard, `verified: false` always |

Prompt-injection resistance does **not** rely on the model obeying the system prompt.
The structural guarantees are: (1) the system prompt is a constant; (2) repo text cannot
close the `<untrusted_data>` block; (3) nothing the model says can change scanner facts,
finding status, or report status; (4) model output is only ever rendered as escaped text.

## Overclaim guard (`analysis.enforce_claim_policy`)

The model is known to overstate impact. For each item in `confirmed_evidence` the guard
checks four claim classes — authentication bypass, data deletion/modification, OS command
execution, verified exploitation. If a claim appears and the evidence/context does not
contain supporting tokens (e.g. `DELETE`/`DROP` for deletion, `os.system`/`subprocess`
for OS execution), the item is moved to `conditional_impact` with a
`[Dipindahkan …]` prefix, confidence is forced to `low`, and a `validation_warnings` entry
is recorded. Certainty words (`terbukti`, `proven`, …) about such claims in `summary` are
flagged. CVE identifiers are removed from all model text. The guard is regex-based and
deliberately conservative; it reduces but does not eliminate overstated explanations.

## Ruleset limits (`src/tantular_cyber/rules/python-sqli.yml`)

One rule, `tantular.python.sqli.dynamic-query-execute`, matching DB-API
`execute` / `executemany` / `executescript` whose query is built with `+`, `%`,
`.format()`, f-strings, or `+=` — inline or via a local variable in the same scope.

* Intra-procedural only: no taint tracking across functions, modules, or files.
* Does not know whether interpolated values are attacker-controlled (hence `unverified`).
* Concatenating constants also matches (possible false positives).
* Misses ORM raw APIs (`text()`, `.raw()`, `.extra()`), query builders, and non-Python code.
* A `.semgrepignore` inside the analyzed repo is honored by Semgrep; a hostile repo could
  use it to hide files from the scan.

## Extension points

* **New scanner** — implement `scanner.Scanner` (`name`, `version()`, `scan(repo, targets)`
  returning `ScannerRun` with `RawFinding`s) and pass it in `run_scan(scanners=[...])` /
  wire it in `pipeline.run_scan`. Normalization, evidence, LLM and reports are reused.
* **New LLM / cybersecurity model** — implement `analysis.LLMClient` (`provider`, `model`,
  `chat(messages, format_schema)`), optionally `health_check()` and `close()`, and pass a
  factory via `run_scan(llm_client_factory=...)`.
* **New rules** — add local YAML under `src/tantular_cyber/rules/` or pass `--rules`.

## GATRA integration contract

`report.json` is the integration surface. It is described by
[`report-schema.json`](report-schema.json) (generated from `models.ScanReport`; a test
fails if it drifts). Consumers should:

* check `schema_version` (semver; breaking changes bump the major version);
* key findings by `findings[].id` (stable across runs for unchanged code);
* treat `status != "complete"` as partial data and read `status_reasons` / `errors`;
* treat every `analysis` block as advisory (`verified` is always `false`).

No GATRA code is modified by this POC.
