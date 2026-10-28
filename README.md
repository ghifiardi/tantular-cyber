# Tantular Cyber — Local Repository Security Analyzer (POC-002)

Read-only security analysis of an **authorized local repository**, entirely on your Mac:

```
repo (read-only) → Semgrep (local rules) → normalized findings → Ollama tantular-cyber (advisory) → report.json + report.md
```

* Scanner evidence is **authoritative**; LLM explanations (Bahasa Indonesia) are **advisory**
  and always marked unverified.
* No repository code is executed, nothing is installed from the repo, nothing is uploaded,
  and no code is modified.
* If Ollama is unavailable, you still get scanner findings, with an explicit **degraded** status.

Architecture and trust model: [docs/architecture.md](docs/architecture.md).
Developer guide: [CLAUDE.md](CLAUDE.md).

## Requirements

| Component | Version tested | Install |
|---|---|---|
| macOS (Apple silicon) | Darwin 27 | — |
| Python | 3.11, 3.12, 3.14 | `brew install python@3.11` |
| Semgrep CLI | 1.143.0 | `brew install semgrep` (or `pip install '.[semgrep]'`) |
| Ollama | 0.40.2 | <https://ollama.com> |
| Model | `tantular-cyber:latest` (Qwen3.5-9B based) | must already exist in `ollama list` |

The model is optional at runtime (`--no-llm`, or degraded mode when unreachable).

## Setup

```bash
git clone https://github.com/ghifiardi/tantular-cyber.git
cd tantular-cyber
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'          # or: make install
brew install semgrep                        # if not installed
ollama list | grep tantular-cyber           # model must be present
.venv/bin/tantular-cyber doctor             # checks semgrep + Ollama + model
```

## Running

```bash
.venv/bin/tantular-cyber scan --repo tests/fixtures/vulnerable --output ./out
```

| Option | Default | Purpose |
|---|---|---|
| `--repo PATH` | required | existing directory to analyze (read-only) |
| `--output DIR` | required | report directory; must be **outside** `--repo` unless `--allow-output-inside-repo` |
| `--rules FILE` | bundled `python-sqli.yml` | local Semgrep rule file (repeatable) |
| `--no-llm` | off | scanner-only report |
| `--model NAME` | `tantular-cyber:latest` | Ollama model |
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
(reports written, LLM stage unavailable or partial).

`tantular-cyber schema` prints the JSON schema of `report.json`
(committed as [docs/report-schema.json](docs/report-schema.json) for GATRA).

## Sample output

Terminal (live run on the Mac baseline, ~110 s for 2 findings):

```
$ tantular-cyber scan --repo tests/fixtures/vulnerable --output ./out
status: complete
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
  "schema_version": "1.0.0",
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

Degraded mode (Ollama down) — findings are still reported, exit code 4:

```
$ tantular-cyber scan --repo tests/fixtures/vulnerable --output ./out --ollama-url http://127.0.0.1:9
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
```

Coverage: Semgrep JSON parsing and command construction, path safety and skip policy
(secrets, binary, oversized, escaping symlinks), redaction, Ollama payload/retries/localhost
policy (mocked with `httpx.MockTransport`), model-output schema validation and fallback,
overclaim guard (auth bypass / data deletion / OS execution / invented CVEs), prompt
injection isolation, report writers, end-to-end CLI with real Semgrep and Ollama down.
CI (`.github/workflows/ci.yml`) runs lint + tests on macOS and Ubuntu, Python 3.11/3.12,
without Ollama.

## Limitations

* **One rule**: Python CWE-89 via DB-API `execute*` with string-built queries, same scope
  only. No cross-function taint, no ORM raw-SQL APIs, no other languages or CWEs. Zero
  findings ≠ secure. See "Ruleset limits" in docs/architecture.md.
* All findings are `unverified` pattern matches; reachability and attacker control are
  not established.
* The model can overstate impact or invent details. The overclaim guard is regex-based and
  reduces, but does not eliminate, this. Read `validation_warnings`.
* ~50 s per finding on the reference Mac with a 9.7B Q8 model; bounded by
  `--max-llm-findings`.
* Skips `.git`, dependency/cache dirs, secret-like files, binaries, files > 1 MB, and all
  symlinks. A `.semgrepignore` in the target repo is honored by Semgrep.
* Redaction is heuristic (known token formats + `password=`-style assignments).

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
