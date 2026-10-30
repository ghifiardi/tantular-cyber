# Tantular Cyber — Local Repository Security Analyzer

Read-only security analysis of an **authorized local repository**, entirely on your machine
(macOS, Linux, Windows via WSL2):

```
repo (read-only) → Semgrep + tantular-callgraph → normalized findings → [optional --ai: local Ollama, advisory] → report.json + report.md
```

**Get started:** [Install](docs/install.md) · [Quickstart](docs/quickstart.md) ·
[Panduan cepat (Bahasa Indonesia)](docs/quickstart.id.md) · [Privacy](docs/privacy.md) ·
[Third-party notices](THIRD_PARTY_NOTICES.md). A deterministic scan needs Python 3.11+ and a
user-installed Semgrep — no AI model and no Ollama.

* Scanner evidence is **authoritative**; LLM explanations (Bahasa Indonesia) are **advisory**
  and always marked unverified.
* No repository code is executed, nothing is installed from the repo, nothing is uploaded,
  and no code is modified.
* Scans are deterministic by default; AI explanations are opt-in (`--ai`). If the requested
  AI stage or Semgrep is unavailable, you still get findings, with an explicit **degraded**
  status.

Supported weaknesses (Python; scoped — see [docs/rules.md](docs/rules.md) for exact sinks,
sanitizer boundaries and known gaps):

| CWE | Coverage |
|---|---|
| CWE-89 SQL injection | DB-API `execute*` with string-built queries |
| CWE-78 OS command injection | `subprocess(..., shell=True)`, `os.system`/`popen` with formatted commands |
| CWE-22 Path traversal | `open`/`send_file` with joined/formatted paths; unfiltered `tarfile` extraction |
| CWE-79 Cross-site scripting | Flask/Jinja2 and Django sinks, `\|safe` / `autoescape off` in HTML templates |
| CWE-798 Hard-coded credentials | credential-named literals, `password=` kwargs, Flask secret keys, AWS key ids |

A second, pure-Python scanner — **tantular-callgraph** (POC-005) — follows values through
same-repository function calls (SQL/command/path built in one function, used in another) and
reports the source-to-sink **call path** with explicit uncertainty. Unresolvable calls
(dynamic dispatch, ambiguous imports) never produce findings.

Evidence context is **AST-aware** (POC-004): the enclosing function, the assignments that
build values used at the finding and same-file helpers, within strict line/char budgets,
redacted before slicing; falls back to a fixed window for templates or unparsable files.

Architecture and trust model: [docs/architecture.md](docs/architecture.md).
Rules: [docs/rules.md](docs/rules.md). Benchmark: [docs/benchmark.md](docs/benchmark.md).
Plugin contract: [docs/plugin-api.md](docs/plugin-api.md). Releases:
[docs/release.md](docs/release.md). Developer guide: [CLAUDE.md](CLAUDE.md). License:
Apache-2.0 (application code only; Semgrep, Ollama and model weights keep their own licenses —
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

## Requirements

| Component | Version tested | Required? |
|---|---|---|
| macOS / Linux / WSL2 | macOS 27 (Apple silicon), Ubuntu (CI) | yes — native Windows is unsupported |
| Python | 3.11, 3.12 | yes |
| Semgrep CLI (LGPL-2.1, user-installed, not bundled) | 1.143.0; tested range `>=1.90,<2` | for **full** scans (otherwise PARTIAL) |
| Ollama | 0.40.2 | only for `--ai` |
| Model | Tantular weights (`ghifidanukusumo/tantular`, profile `tantular-cyber:latest`) | only for `--ai`; never downloaded by this tool |

## Setup

Users: install a verified release wheel — see [docs/install.md](docs/install.md).

Developers:

```bash
git clone https://github.com/ghifiardi/tantular-cyber.git
cd tantular-cyber
make install                                # python3.11 venv + dev extras
brew install semgrep                        # if not installed
.venv/bin/tantular-cyber doctor             # read-only: semgrep, platform, optional Ollama, privacy
```

## Running

```bash
.venv/bin/tantular-cyber scan --repo tests/fixtures/vulnerable --output ./out
```

| Option | Default | Purpose |
|---|---|---|
| `--repo PATH` | required | existing directory to analyze (read-only) |
| `--output DIR` | required | report directory; must be **outside** `--repo` unless `--allow-output-inside-repo` |
| `--rules FILE` | bundled rulesets (CWE-89/78/22/79/798) | local Semgrep rule file (repeatable; replaces the defaults) |
| `--no-callgraph` | off | disable the interprocedural scanner |
| `--callgraph-depth N` | 3 | max call levels followed |
| `--context-strategy` | `ast` | `ast` (bounded AST-aware context) or `window` (fixed ±`--context-lines`) |
| `--ai` | off | add local Ollama explanations (optional) |
| `--no-llm` | — | deprecated no-op (deterministic is the default) |
| `--model NAME` | `tantular-cyber:latest` | Ollama model for `--ai`; cloud models are refused |
| `--ollama-url URL` | `http://127.0.0.1:11434` | localhost only… |
| `--allow-remote-ollama` | off | …unless you opt in (code excerpts leave the machine) |
| `--ollama-timeout S` | 300 | per request; first call includes model load |
| `--max-llm-findings N` | 25 | bound LLM time; the rest are marked `skipped` |
| `--keep-raw` | off | also write `semgrep-raw.json` (local paths scrubbed) |
| `-v` | off | progress logging (never logs prompts or source) |

Environment variables (`TANTULAR_OLLAMA_URL`, `TANTULAR_MODEL`, `TANTULAR_OLLAMA_TIMEOUT`,
`TANTULAR_OLLAMA_RETRIES`, `TANTULAR_ALLOW_REMOTE_OLLAMA`, `TANTULAR_SEMGREP_BIN`) are
documented in [.env.example](.env.example). CLI flags take precedence.

**Exit codes:** `0` complete · `2` invalid input · `3` scanner failed · `4` degraded
(reports written; partial coverage such as missing Semgrep, or the requested `--ai` stage was
unavailable or partial). `doctor` returns `0` full / `4` partial / `3` unavailable.

`tantular-cyber schema` prints the JSON schema of `report.json`
(committed as [docs/report-schema.json](docs/report-schema.json) for GATRA).

## Sample output

Deterministic scan (default, no model):

```
$ tantular-cyber scan --repo tests/fixtures/vulnerable --output ./out
status: complete
coverage: full (scanners succeeded: semgrep, tantular-callgraph)
ai explanations: not requested
findings: 2 {'CWE-89': 2}
analyses: {'disabled': 2}
```

With explanations (live run on the Mac baseline; the first run includes model load):

```
$ tantular-cyber scan --repo tests/fixtures/vulnerable --output ./out --ai
status: complete
coverage: full (scanners succeeded: semgrep, tantular-callgraph)
ai explanations: requested
findings: 2 {'CWE-89': 2}
analyses: {'completed': 2}
json: /…/tantular-cyber/out/report.json
markdown: /…/tantular-cyber/out/report.md
```

`report.md` excerpt (finding 1):

````markdown
### 1. [HIGH] CWE-89 — `app.py:14`

- **Finding ID:** `33195d4d98caf1c4`
- **Status:** `unverified` (belum diverifikasi manusia)
- **Tool / rule:** semgrep / `tantular.python.sqli.dynamic-query-execute`

**Bukti (scanner, otoritatif):**

```
14 |     cursor.execute(query)
```

**Bukti terkonfirmasi (menurut model, dari potongan kode):**
- Variabel 'query' dibangun dengan menggabungkan literal string dan nilai variabel 'username' menggunakan operator '+'.
- Query yang telah digabungkan dieksekusi melalui metode cursor.execute() pada baris 14.

**Dampak kondisional (belum terbukti):**
- Jika variabel 'username' berisi input dari pengguna eksternal yang tidak terfilter, penyerang dapat menyuntikkan perintah SQL berbahaya ke dalam query tersebut.
````

`report.json` excerpt:

```json
{
  "schema_version": "1.3.0",
  "report_type": "tantular-cyber.repository-scan",
  "status": "complete",
  "totals": {"findings": 2, "by_severity": {"high": 2}, "by_cwe": {"CWE-89": 2},
             "analyses_by_status": {"completed": 2}},
  "findings": [{
    "id": "33195d4d98caf1c4", "tool": "semgrep",
    "rule_id": "tantular.python.sqli.dynamic-query-execute",
    "cwe": ["CWE-89"], "severity": "high", "path": "app.py",
    "start_line": 14, "end_line": 14, "status": "unverified",
    "evidence": {"snippet": "14 |     cursor.execute(query)", "context": "11 | ...", "...": "..."},
    "analysis": {"status": "completed", "model": "tantular-cyber:latest", "verified": false,
                 "explanation": {"summary": "...", "confirmed_evidence": ["..."],
                                 "conditional_impact": ["..."], "remediation": ["..."],
                                 "corrected_code": "...", "confidence": "high",
                                 "limitations": ["..."]},
                 "validation_warnings": ["..."]}
  }]
}
```

Degraded mode (`--ai` requested, Ollama down) — findings are still reported, exit code 4:

```
$ tantular-cyber scan --repo tests/fixtures/vulnerable --output ./out --ai --ollama-url http://127.0.0.1:9
WARNING: scan DEGRADED — see status_reasons/errors in the report
status: degraded
  - LLM explanation unavailable: Ollama not reachable or model missing; report contains scanner findings only
findings: 2 {'CWE-89': 2}
analyses: {'skipped': 2}
```

Clean fixture: `status: complete`, `findings: 0`, and the LLM is never called.

## Tests

```bash
make test     # .venv/bin/pytest -q   — no Ollama needed; real-semgrep tests auto-skip if missing
make lint
make smoke    # TANTULAR_RUN_OLLAMA_SMOKE=1 pytest -m ollama — live tantular-cyber:latest
make bench    # offline benchmark: scanner precision/recall + adversarial suites (CI gate)
make bench-live  # opt-in: + LLM explanation quality and cold/warm performance (local Ollama)
```

Benchmarking is documented in [docs/benchmark.md](docs/benchmark.md); the opt-in real-world
pointers, human review and model comparison workflows in [docs/evaluation.md](docs/evaluation.md);
fixture provenance and licenses in [docs/provenance-audit.md](docs/provenance-audit.md).

Coverage: Semgrep JSON parsing and command construction, path safety and skip policy
(secrets, binary, oversized, escaping symlinks), scan-coverage verification (including a
hostile `.semgrepignore` against real Semgrep), multiline/encoded/comment secret redaction
end-to-end (evidence, LLM prompt, reports), Ollama payload/retries/localhost policy and
performance metrics (mocked with `httpx.MockTransport`), model-output schema validation and
fallback, adversarial overclaim wording (`tests/test_overclaims.py`), prompt-injection
isolation, report writers, end-to-end CLI with real Semgrep and Ollama down.
CI (`.github/workflows/ci.yml`) runs lint + tests on macOS and Ubuntu, Python 3.11/3.12,
without Ollama.

## Limitations

* **Small, scoped rulesets** for five CWEs, syntactic and same-scope only (no
  cross-function taint), Python plus HTML templates. Each rule's sinks, boundaries and known
  false positives/negatives are listed in docs/rules.md. Benchmark precision/recall come
  from a small synthetic corpus and are regression floors, **not real-world rates**. Zero
  findings ≠ secure.
* AST context and its dataflow hints are explanatory and syntactic; they do not prove that
  untrusted data reaches a sink.
* All findings are `unverified` pattern matches; reachability and attacker control are
  not established.
* The model can overstate impact or invent details. The overclaim guard uses Indonesian
  and English word-family lexicons (36 adversarial paraphrases tested) but is still lexical;
  novel phrasing can pass. Read `validation_warnings`.
* ~31 s per finding (median) on an M4 Pro / 24 GB with the 9.7B Q8 model, ~15 output
  tokens/s, sequential; bounded by `--max-llm-findings`. See
  [docs/benchmark-results.md](docs/benchmark-results.md).
* Skips `.git`, dependency/cache dirs, secret-like files, binaries, files > 1 MB, and all
  symlinks; every skipped file is listed with its reason. Repo `.semgrepignore` /
  `.gitignore` files are **not** used to pick targets, and each scan's coverage is checked
  against Semgrep's own `paths.scanned`; any gap makes the report `degraded`.
* Redaction is heuristic: known token formats, multiline keys and triple-quoted values,
  base64-encoded known tokens, HTTP Basic credentials, credential-named assignments
  (quoted, unquoted with a digit/symbol), and credentials in comments. Plain-letter
  unquoted values, hex/custom encodings, and secrets in innocuously named variables are
  not detected.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `status: failed`, "semgrep binary 'semgrep' not found" | `brew install semgrep` or `--semgrep-bin /path/to/semgrep` |
| `status: degraded`, "Ollama not reachable" | start Ollama (`ollama serve` or the app); check `curl http://127.0.0.1:11434/api/version` |
| "model 'tantular-cyber:latest' is not available" | `ollama list`; recreate the model from its Modelfile |
| `ReadTimeout` on first finding | model loading is slow; raise `--ollama-timeout 600` |
| "generation ended early (done_reason=length)" | output truncated; analysis marked `error`; retry or raise `num_ctx` in config |
| `invalid_output` analyses | model returned non-schema JSON; the scanner finding is unaffected |
| "output directory is inside the analyzed repository" | write reports elsewhere, or `--allow-output-inside-repo` |
| "Ollama host … is not localhost" | intended guard; add `--allow-remote-ollama` only if you accept sending excerpts off-host |
| Semgrep slow on first run | it parses rules once; subsequent runs are faster |

## Security posture

Scope is limited to repositories you are authorized to analyze. The tool performs no
exploit execution, no active testing, no network traffic other than the configured
(local) Ollama endpoint, and never modifies analyzed code. Semgrep runs with
`--metrics=off`, no login, and `SEMGREP_APP*` variables removed from its environment.
