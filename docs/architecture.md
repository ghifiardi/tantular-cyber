# Architecture — Tantular Cyber (POC-002 … POC-004)

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
 ┌──────────────────────┐   pipeline.py + context.py: repo-relative paths, whole-file
 │ 3. Normalize          │   redaction, exact evidence lines, bounded AST-aware context +
 └──────────┬───────────┘   dataflow hints, CWE, severity, deterministic id, "unverified"
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
| `degraded` | 4 | reports written, but LLM unreachable / invalid output / limit hit, or a scanner coverage gap / unverified coverage |
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
execution, verified exploitation — using word-family lexicons in Indonesian and English
(stems such as `*hapus*`, `*hancur*`, `mengosong*`; synonyms such as "login tanpa
password", "masuk sebagai admin", "reverse shell", "server compromise", "exploitation is
confirmed"), proximity matching (up to four words between verb and object), and
normalization (NFKC for fullwidth forms, a Latin homoglyph map for common Cyrillic/Greek
lookalikes, zero-width removal, whitespace collapsing). If a claim appears and the evidence/context does not
contain supporting tokens (e.g. `DELETE`/`DROP`/`UPDATE` for modification,
`os.system`/`subprocess` for OS execution), the item is moved to `conditional_impact` with
a `[Dipindahkan …]` prefix, confidence is forced to `low`, and a `validation_warnings`
entry is recorded. Certainty words (`terbukti`, `dipastikan`, `proven`, …) about such
claims in `summary` are flagged. CVE identifiers are removed from all model text.

`tests/test_overclaims.py` holds 36 adversarial paraphrases plus grounded statements that
must not be flagged (e.g. "mengeksekusi perintah SQL"). The guard is still lexical: novel
phrasing outside the lexicons can pass (measured blind spots are listed in
`benchmarks/corpus/adversarial/overclaims.json`). Extend the lexicon and the tests together.

## Rulesets

Five local rulesets (CWE-89, 78, 22, 79, 798), listed in `config.DEFAULT_RULESETS`. Exact
sinks, sanitizer boundaries and known false positives/negatives: [rules.md](rules.md).
All rules are syntactic and same-scope; none tracks taint across functions or files, and
none knows whether a value is attacker-controlled (hence `unverified`). The CWE-79 template
rule runs in Semgrep's `generic` mode on `*.html`/Jinja files, so coverage verification
treats those extensions as in scope.

## Evidence context (`context.py`, report schema 1.2)

Before POC-004 the context was a fixed ±3-line window, which missed values built further
up (e.g. a multi-line query whose tainted part is 5 lines above the sink). Now:

1. The **original** file is parsed with `ast.parse` for structure only (nothing executed).
   All displayed lines come from the **whole-file-redacted** text; redaction preserves the
   line count, and any mismatch, non-Python file or syntax error falls back to the window
   (`context_fallback` records why).
2. The innermost enclosing function is shown whole if it fits; otherwise its header, the
   full (multi-line) finding statement, ±2 lines, and the statements that bind names used
   at the finding (plus one transitive step) — compound statements contribute only their
   header lines. Gaps are rendered as `…`.
3. One level of **same-file helpers** called at the finding or in those assignments is
   included (clipped to `helper_max_lines`).
4. `dataflow_hints` summarise parameters, local assignments and helpers. They are sent to
   the model as `dataflow_hints_unverified` inside the untrusted-data block, and the
   system prompt states they are not proof of taint.
5. Budgets (`ContextBudget`: 60 lines, 4000 chars by default; ≈ chars/4 tokens) are hard:
   lines farthest from the finding are dropped first, then kept lines other than the
   finding line, then a hard character cut. `context_truncated` flags any reduction.

`--context-strategy window` restores the POC-002 behavior. The benchmark measures both
strategies on every detected location (construct coverage, size, budget) — see
[benchmark.md](benchmark.md).

## Scan scope and coverage verification

Repository-controlled files must not be able to shrink the scan silently.

* **Target selection** is done by `security.collect_files`, never by the tools. Repo
  `.semgrepignore` / `.gitignore` files are not consulted; files are passed to Semgrep
  explicitly after `--`. (Verified on Semgrep 1.143: explicit targets are scanned even when
  a `.semgrepignore` matches them; `tests/test_coverage.py` keeps this true.)
* **Coverage check**: from the rules' `languages:` the scanner derives which submitted
  targets are in scope (by extension), then compares them with the tool's own
  `paths.scanned`. Any in-scope file the tool did not scan is listed in
  `scanners[].coverage.not_scanned` (with the tool's skip reason when available), recorded
  as an error, and makes the report `degraded`. If the tool does not report scanned paths,
  coverage is `verified: false` and the report is `degraded` as well.
* **Reported scope**: `target.ignore_files_present` lists repo ignore files found;
  `skipped` lists every file excluded by the inventory policy, with reasons (capped list in
  JSON, first 50 in Markdown).
* Out-of-scope files (no rule language applies) are counted, not flagged. Python files
  without a `.py`/`.pyi` extension (e.g. extensionless scripts) are out of scope for Semgrep
  and therefore for this ruleset.

## Secret redaction (`security.redact`)

Applied to the **whole file** before evidence and context lines are cut, and again to all
scanner messages and model output. Every rule preserves the newline count, so scanner line
numbers stay valid. Layers:

1. Known credential formats: private-key blocks (multiline), AWS, GitHub, Slack, Google,
   `sk-` API keys, JWTs, bearer tokens, credentials in URLs.
2. Encoded: HTTP Basic credentials (base64 `user:pass`); base64 blobs whose decoded content
   matches a known format.
3. Contextual: quoted, triple-quoted (multiline), type-annotated, and unquoted
   (`.env`/YAML style, must contain a digit or symbol) values assigned to credential-like
   names; credentials written in comments (`# admin password is …`, `// api key: …`,
   `# kata sandi: …`).

4. Obfuscated names: texts containing zero-width/bidi characters or Latin lookalikes are
   also matched on a *skeleton* (invisibles removed, lookalikes mapped to Latin, shared
   `CONFUSABLES` table) with a position map; only the value span is replaced in the
   original, so evidence keeps its real characters.

Limits: heuristic. Plain-letter unquoted values (`api_key: abcdefgh`), hex-encoded or
custom-encoded secrets, and secrets in variables with innocuous names are not detected.
Secret-like *files* (`.env`, `*.pem`, …) are never read at all.

Lines are split on `\n` only (like Semgrep). `str.splitlines()` also splits on form feeds
and U+2028, which would let a crafted file shift the displayed evidence line.

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
