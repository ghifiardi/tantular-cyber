# Release acceptance checklist — v0.3.0

Checklist version **1.0** · release under test **v0.3.0** (tag commit `056b926`) · issue #14

This checklist validates the **published v0.3.0 release artifacts** as a user receives them.
It is separate from POC-007 (reliability of the optional AI explanations); results from the
two tracks must not be combined.

## How to use this checklist

* This file holds the procedure and the **expected** results only. Never record observed
  results here. Each execution produces a new dated run record in
  [`acceptance-runs/`](acceptance-runs/) (copy [`TEMPLATE.md`](acceptance-runs/TEMPLATE.md)
  to `acceptance-runs/YYYY-MM-DD-v0.3.0.md`).
* Every test in a run record gets **actual result**, **evidence location** and a status:
  * **PASS** — every condition under "Expected" was observed, and the evidence is retained.
  * **FAIL** — the test ran and at least one expected condition was not observed.
  * **BLOCKED** — a prerequisite was not available (for example no Linux host for N3, or no
    authorized repository for section D). Say what is missing. A BLOCKED test is never
    counted as a pass.
  * **N/A** — allowed **only** for optional tests that do not apply to the platform (N5 on
    Linux). Say why.
* Do not assume success: a test passes only on what was observed during that run. Copy
  output exactly; do not paraphrase it into the expected wording.

### Test classes and the acceptance decision

| Class | Tests | Prerequisite | If not PASS |
|---|---|---|---|
| **Mandatory** | A1–A11, B1, B2, B3, B5, B6, B7, B8, B10, B11, N1, C1, C2 | none beyond section A's | FAIL or BLOCKED → **NOT ACCEPTED** |
| **Conditional** | B4 | a cloud model listed by `ollama list`, and the Ollama server request log | BLOCKED → at most **CONDITIONALLY ACCEPTED** |
| **Conditional** | B9, N2 | Ollama running with `ghifidanukusumo/tantular:latest` already pulled | BLOCKED → at most **CONDITIONALLY ACCEPTED** |
| **Conditional** | N3, N4 (N4 also needs the model) | a Linux host with `strace` | BLOCKED → at most **CONDITIONALLY ACCEPTED**; network assurance is **not established** |
| **Optional** | N5 | macOS | N/A or BLOCKED has no effect; FAIL → **NOT ACCEPTED** |
| **Separate evaluation** | D1–D5 | an authorized repository and a reviewer | reported on its own; does not change the release decision |

The run record's overall decision is exactly one of:

* **ACCEPTED** — every mandatory and conditional test is PASS, and no test is FAIL.
* **CONDITIONALLY ACCEPTED** — every mandatory test is PASS, no test is FAIL, and one or
  more conditional tests are BLOCKED. The record lists each blocked test and the assurance
  it leaves unestablished (for example "N3/N4 BLOCKED: no process-tree evidence that
  Semgrep makes no connection attempts").
* **NOT ACCEPTED** — any mandatory test is FAIL or BLOCKED, or any test of any class
  (including optional and section D) is FAIL.

Section D gets its own line in the run record (BLOCKED, or complete with counts). A FAIL in
D (for example the repository was modified, or reports landed inside it) still makes the
run NOT ACCEPTED, because it is a product failure; a D finding classified as a false
positive is a detection-quality result, not a FAIL.
* Run everything from the downloaded release, **not** from a source checkout, unless a test
  says otherwise. Use a fresh working directory, referred to below as `$ACC`.
* Evidence goes in `$ACC/evidence/<test-id>/` (terminal output, report files, logs). Before
  any evidence is committed, redact local absolute paths (replace with `$ACC`), user names,
  hostnames and anything from a scanned repository other than the sample (section D).
* Any FAIL is reported as an issue; release code and v0.3.0 assets are not changed as part
  of this checklist.

### Environment to record in every run

OS and version, CPU architecture, Python version, Semgrep version (`semgrep --version`),
Ollama version (`ollama --version`, or "not installed"), `gh` version, the SHA-256 of every
downloaded asset, the attestation outcome, the tester (pseudonym is fine) and any deviation
from this procedure.

### Common setup

```bash
export ACC=~/tc-acceptance/$(date +%Y-%m-%d)   # fresh, empty directory
mkdir -p "$ACC/release" "$ACC/evidence" && cd "$ACC"
export VERSION=0.3.0
export BASE=https://github.com/ghifiardi/tantular-cyber/releases/download/v$VERSION
export TC="$ACC/venv/bin/tantular-cyber"   # set after A4
```

---

## A. Published-release install and integrity

Prerequisites: Python ≥ 3.11, `gh` (signed in, for attestation), `curl`, Semgrep installed by
the user (`brew install semgrep` or `pipx install semgrep`).

| ID | Procedure | Expected |
|---|---|---|
| A1 | `gh release view v$VERSION --repo ghifiardi/tantular-cyber --json isDraft,isPrerelease,tagName,assets --jq '{isDraft,isPrerelease,tagName,assets:[.assets[].name]}'` | Not a draft, not a prerelease, tag `v0.3.0`. Exactly 7 assets: `SHA256SUMS`, `THIRD_PARTY_NOTICES.md`, `tantular-cyber-0.3.0-licenses.md`, `tantular-cyber-0.3.0-runtime.lock.txt`, `tantular-cyber-0.3.0.cdx.json`, `tantular_cyber-0.3.0-py3-none-any.whl`, `tantular_cyber-0.3.0.tar.gz`. |
| A2 | In `$ACC/release`: download every asset with `curl -fL --connect-timeout 20 --max-time 600 -O "$BASE/<name>"`, then `shasum -a 256 -c SHA256SUMS` | Every line `OK`; exit 0. Record each asset's SHA-256 in the run record. |
| A3 | `gh attestation verify tantular_cyber-$VERSION-py3-none-any.whl --repo ghifiardi/tantular-cyber` and the same for `tantular_cyber-$VERSION.tar.gz` | Both verify; the provenance names workflow `.github/workflows/release.yml` at `refs/tags/v0.3.0` and source commit `056b926…`. Exit 0. |
| A4 | `python3.11 -m venv "$ACC/venv"`; `"$ACC/venv/bin/pip" install --require-hashes -r release/tantular-cyber-$VERSION-runtime.lock.txt`; `"$ACC/venv/bin/pip" install --no-deps release/tantular_cyber-$VERSION-py3-none-any.whl` | Both installs exit 0 with no hash errors. `"$ACC/venv/bin/pip" check` reports no broken requirements. |
| A5 | `$TC --version` and `"$ACC/venv/bin/python" -I -c "import tantular_cyber, sys; print(tantular_cyber.__file__)"` | Prints `tantular-cyber 0.3.0`. The module path is inside `$ACC/venv`, not a source checkout. |
| A6 | `$TC doctor` (Ollama may be running or not) | Lines `[ok  ]` for python, platform, rules (5 bundled rule files) and semgrep. Ends with `scan capability: FULL (Semgrep + tantular-callgraph)`. Exit 0. Header says read-only with no downloads. |
| A7 | `$TC doctor --json > evidence/A7/doctor.json` | Valid JSON (`python3 -m json.tool` parses it) with a `network_requests` field. With `--no-ollama` added, `network_requests` is empty/none. |
| A8 | `$TC sample --dest "$ACC/sample"` | Exit 0; `$ACC/sample` contains the synthetic sample project and the output says it is synthetic, deliberately vulnerable and never executed. |
| A9 | `$TC scan --repo "$ACC/sample" --output "$ACC/reports/a9"` | Summary shows `status: complete`, `coverage: full (scanners succeeded: semgrep, tantular-callgraph)`, `ai explanations: not requested`, `findings: 2 {'CWE-89': 2}`. Exit 0. |
| A10 | Inspect `$ACC/reports/a9/report.json` and `report.md` | Both exist. JSON has `schema_version` `1.3.0`, `status` `complete`, 2 findings, each with `status` `unverified`, a repo-relative path and a line number. No absolute local path appears in either file (`grep -c "$ACC" report.*` → 0). |
| A11 | `ls -la "$ACC/sample"` before and after A9 (or compare `find "$ACC/sample" -type f -exec shasum -a 256 {} +` output) | The scanned directory is byte-for-byte unchanged; no files added. |

## B. Failure handling and trust boundaries

Prerequisites: section A passed. B9 needs Ollama running locally and the model
`ghifidanukusumo/tantular:latest` already pulled (the tester decides whether to pull it;
otherwise BLOCKED). B4 needs a cloud model listed by `ollama list` (for example
`qwen3.5:cloud`) and access to the Ollama server's request log; otherwise BLOCKED. B3, B5
and B6 need no running Ollama.

| ID | Procedure | Expected |
|---|---|---|
| B1 | `$TC scan --repo "$ACC/sample" --output "$ACC/reports/b1" --semgrep-bin /nonexistent` | `WARNING: scan DEGRADED`, `status: degraded`, `coverage: PARTIAL (scanners succeeded: tantular-callgraph)`, a reason listed. Call-graph findings still reported. Exit 4. Never `complete`. |
| B2 | `$TC doctor --semgrep-bin /nonexistent --no-ollama` | `[FAIL] semgrep: '/nonexistent' not found — scans run tantular-callgraph only (PARTIAL)` with install guidance naming LGPL-2.1 and "not bundled"; `scan capability: PARTIAL — reports will be marked degraded`. Exit 4. |
| B3 | Ollama unavailable: `$TC scan --repo "$ACC/sample" --output "$ACC/reports/b3" --ai --ollama-url http://127.0.0.1:9 --model x` (port 9 has no listener). Optionally repeat with the real Ollama stopped. | `status: degraded`, `coverage: full`, `ai explanations: requested`, reason "LLM explanation unavailable…", `findings: 2 {'CWE-89': 2}`, analyses `skipped`. Exit 4. |
| B4 | Cloud model refused (procedure below): `$TC scan --repo "$ACC/sample" --output "$ACC/reports/b4" --ai --model qwen3.5:cloud` under the N1 audit hook, with the Ollama request log captured | Exit 4, `status: degraded`, findings unchanged. `report.json` `errors` contains "runs on a remote host … source excerpts would leave this machine". Audit log: connections only to `127.0.0.1:11434`. **Request log (required HTTP-level evidence):** only `GET "/api/version"` and `GET "/api/tags"` during the scan; no `POST "/api/chat"` and no `POST "/api/show"`. Socket destinations alone cannot show which HTTP requests were sent, so without the request log B4 is BLOCKED, not PASS. |
| B5 | `$TC doctor --ollama-url http://user:pass@127.0.0.1:11434` | Ollama line `[FAIL] … rejected — not contacted (credentials in the Ollama base URL are not supported)`. Neither `user` nor `pass` appears anywhere in the output. `network requests: none`. Exit reflects scan capability only (0 when Semgrep is present). |
| B6 | `$TC scan --repo "$ACC/sample" --output "$ACC/reports/b6" --ai --ollama-url http://user:pass@127.0.0.1:11434`, under the N1 audit hook | `status: degraded`, reason `LLM explanation unavailable: invalid or disallowed Ollama endpoint`, findings unchanged, exit 4. `report.json` `llm.endpoint` shows `[REDACTED:credentials]`; `grep -E 'user:\|pass@' "$ACC/reports/b6"/report.*` finds nothing. The audit log shows no connection attempt. |
| B7 | `$TC scan --repo "$ACC/sample" --output "$ACC/sample/out"` | `error: output directory is inside the analyzed repository…`. Exit 2. `$ACC/sample/out` not created. |
| B8 | `$TC plugins list` | JSON with `plugin_api` `1.0`, `report_schema` `1.3.0`, two built-ins (`semgrep`, `tantular-callgraph`) and `third_party_execution` beginning `disabled`. Exit 0. There is no command that runs a plugin. |
| B9 | AI is advisory: `$TC scan --repo "$ACC/sample" --output "$ACC/reports/b9" --ai --model ghifidanukusumo/tantular:latest`, then compare findings with A9 ignoring `analysis` (snippet below) | Findings identical to A9 apart from `analysis`; every finding still `status: unverified`; report `status` decided by scanners/LLM availability only. Explanations in Bahasa Indonesia. |
| B10 | `$TC scan … --ai --no-llm` | Rejected as contradictory input. Exit 2. |
| B11 | Offline operability: disconnect all networks (Wi-Fi off, cable out), repeat A9 into `$ACC/reports/b11` | Same result as A9, exit 0. **This shows offline operation only; it does not show that no connection was attempted** (that is section N). |

Request log for B4. Ollama writes one `[GIN]` line per HTTP request with method and path:
on macOS to `~/.ollama/logs/server.log`, on Linux to the service journal
(`journalctl -u ollama`) or the terminal running `ollama serve`. The log does not say which
client sent a request, so stop every other Ollama client (editors, other agents, other
terminals) first; if another client may have been active during the window, B4 is BLOCKED.

```bash
mkdir -p "$ACC/evidence/B4"
LOG=~/.ollama/logs/server.log
START=$(wc -l < "$LOG")
"$ACC/venv/bin/python" -I "$ACC/netaudit.py" "$ACC/evidence/B4/net.log" \
    scan --repo "$ACC/sample" --output "$ACC/reports/b4" --ai --model qwen3.5:cloud
tail -n +"$((START + 1))" "$LOG" | grep '\[GIN\]' > "$ACC/evidence/B4/requests.log"
cat "$ACC/evidence/B4/requests.log"
```

(`netaudit.py` is created in section N; create it before running B4.)

Findings comparison for B9 (run with the venv Python, `-I`):

```bash
"$ACC/venv/bin/python" -I - "$ACC/reports/a9/report.json" "$ACC/reports/b9/report.json" <<'EOF'
import json, sys
def facts(p):
    return [{k: v for k, v in f.items() if k != "analysis"} for f in json.load(open(p))["findings"]]
a, b = (facts(p) for p in sys.argv[1:])
print("identical" if a == b else "DIFFERENT")
EOF
```

## N. Network-attempt instrumentation

Purpose: show which connections the tool **attempts**, including failed ones. Offline runs
(B11) and the release workflow's `unshare --net` smoke test prove only that the tool works
without a network.

Pass criteria for every N test:

* default scan (no `--ai`): **no** attempted connection to any IP address or hostname, and
  no DNS lookup;
* `--ai` with the default endpoint: attempts only to `127.0.0.1:11434` (or `::1`), nothing
  else;
* Unix-domain sockets and local IPC (`AF_UNIX`, pipes) are recorded but are **not** network
  destinations; list them separately in the run record with what they appear to be.

| ID | Procedure | Expected |
|---|---|---|
| N1 | Python audit hook, default scan (any OS; procedure below) | Log is empty. Exit 0. |
| N2 | Python audit hook, `--ai --model ghifidanukusumo/tantular:latest` | Only `socket.connect ('127.0.0.1', 11434)` and its `getaddrinfo` for `127.0.0.1`. |
| N3 | **Linux:** `strace -f -e trace=connect,sendto,sendmsg -o "$ACC/evidence/N3/strace.log" $TC scan --repo "$ACC/sample" --output "$ACC/reports/n3"` | No `connect` to `AF_INET`/`AF_INET6` addresses; any `AF_UNIX` entries listed and explained. `-f` follows Semgrep child processes. |
| N4 | **Linux:** as N3 with `--ai --model ghifidanukusumo/tantular:latest` | The only `AF_INET`/`AF_INET6` connects are to `127.0.0.1:11434` (or `::1`). |
| N5 | **macOS (weaker, optional):** `sudo fs_usage -w -f network` or `nettop -p <pid>` while running A9 and B9 | Same destinations as N1/N2. Record that this evidence is observational and may miss short-lived attempts. |

N3/N4 are the strongest evidence because they cover every process, including Semgrep. If
no Linux host is available, mark N3/N4 **BLOCKED** (the run is then at most CONDITIONALLY
ACCEPTED); N1/N2 alone do not satisfy them.

Audit-hook wrapper for N1/N2. It records Python-level socket events, including attempts
that fail, but **does not cover the Semgrep child process** (a separate executable): it
sees neither Semgrep's connection attempts nor its DNS lookups. Socket events also do not
show which HTTP requests were sent (see B4). A clean N1 therefore establishes only that the
Tantular Cyber Python process made no attempt.

```bash
cat > "$ACC/netaudit.py" <<'EOF'
import sys

LOG = open(sys.argv.pop(1), "a")
EVENTS = {"socket.connect", "socket.sendto", "socket.sendmsg", "socket.getaddrinfo"}


def hook(event, args):
    if event in EVENTS:
        detail = args if event == "socket.getaddrinfo" else args[1:]
        LOG.write(f"{event} {detail!r}\n")
        LOG.flush()


sys.addaudithook(hook)
from tantular_cyber.cli import main  # noqa: E402

sys.exit(main())
EOF
mkdir -p "$ACC/evidence/N1" "$ACC/evidence/N2"
"$ACC/venv/bin/python" -I "$ACC/netaudit.py" "$ACC/evidence/N1/net.log" \
    scan --repo "$ACC/sample" --output "$ACC/reports/n1"
"$ACC/venv/bin/python" -I "$ACC/netaudit.py" "$ACC/evidence/N2/net.log" \
    scan --repo "$ACC/sample" --output "$ACC/reports/n2" --ai --model ghifidanukusumo/tantular:latest
```

Keep `netaudit.py` outside `$ACC/sample` and outside any scanned repository.

## C. Synthetic benchmark (regression measurement)

The corpus is synthetic and written for this project. Its precision and recall are
**regression floors, not real-world detection accuracy**; never quote them as such.

| ID | Procedure | Expected |
|---|---|---|
| C1 | `$TC benchmark --output "$ACC/reports/c1"` (uses the corpus bundled in the wheel) | `benchmark: PASS`; every gate `pass`; `scanner: TP=43 FP=8 FN=4 TN=23 precision=0.8431 recall=0.9149`; adversarial overclaims `known_gap=7, passed=50`; adversarial secrets `known_gap=4, passed=23`. Exit 0. |
| C2 | Inspect `benchmark.json` / `benchmark.md` | Both present; per-CWE gates for CWE-22/78/79/798/89 all pass; `llm` is `null` in `benchmark.json` (no model is called unless explicitly requested). |

A different Semgrep version can change these counts. If C1 differs, record the Semgrep
version and the exact differences, mark FAIL, and report it; do not adjust the expected
values in this checklist inside a run.

## D. Authorized real-repository scan

**BLOCKED until an authorized repository is selected** and a reviewer is available to
verify findings independently. Do not pick a repository on the tester's own initiative.

Prerequisites: written confirmation that the tester may assess the repository; a Python
repository; a reviewer who can read its code.

| ID | Procedure | Expected |
|---|---|---|
| D1 | Record repository identity in the run record by pseudonym, plus commit hash and approximate size (files, lines of Python). Do not record its name if the owner has not agreed. | Recorded. |
| D2 | `$TC scan --repo <repo> --output "$ACC/reports/d2"` (reports outside the repository) | Exit 0 or 4. If 4, every reason is listed and understood. Repository unchanged (A11 method). |
| D3 | For each finding, open the cited file and line and classify it as true positive, false positive, or uncertain, with a one-line reason. | Every finding classified. |
| D4 | Known misses: list vulnerabilities in the five supported CWE families that the reviewer knows exist and the tool did not report. | List recorded (may be empty). |
| D5 | Limits: Python only, five CWE families (89, 78, 22, 79, 798), no ground truth for the repository, call graph ≤ 3 levels and no dynamic dispatch, reviewer coverage. | Stated in the run record. |

Rules for D: report counts (TP / FP / uncertain / known misses) only. Do **not** state a
recall or detection rate: without ground truth it cannot be measured. Do not commit code
from the repository; redact snippets and secrets; cite file paths only if the owner agrees.

---

## Change control

* Changing a test or an expected value is a new checklist version (bump the number at the
  top, explain why in the PR). Run records state which checklist version they used.
* Expected values describe documented v0.3.0 behavior. If observed behavior differs, the
  run records FAIL; whether the checklist or the product is wrong is decided in review.
