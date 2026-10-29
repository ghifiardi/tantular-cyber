# Quickstart

[Bahasa Indonesia](quickstart.id.md) · Full installation guide: [install.md](install.md)

Tantular Cyber finds security issues (SQL injection CWE-89, command injection CWE-78, path
traversal CWE-22, XSS CWE-79, hard-coded credentials CWE-798) in **Python** repositories on
your own machine. It only reads files: it never runs, imports or modifies the code it scans
and never uploads it.

## 1. Install (about 5 minutes, no AI model needed)

```bash
brew install semgrep          # Linux/WSL2: pipx install semgrep
pipx install ./tantular_cyber-0.3.0-py3-none-any.whl   # verified release wheel, see install.md
tantular-cyber doctor
```

Look for `scan capability: FULL`. If it says `PARTIAL`, Semgrep is missing; scans still run
but are marked degraded.

## 2. Try it on the bundled sample

```bash
tantular-cyber sample --dest ./tc-demo/sample-app
tantular-cyber scan --repo ./tc-demo/sample-app --output ./tc-demo/reports
```

```
status: complete
coverage: full (scanners succeeded: semgrep, tantular-callgraph)
ai explanations: not requested
findings: 2 {'CWE-89': 2}
```

The sample is synthetic and deliberately vulnerable; it is never executed. Open
`./tc-demo/reports/report.md` (readable report) or `report.json` (machine-readable, schema
1.3.0).

## 3. Scan your own repository

```bash
tantular-cyber scan --repo ~/code/my-project --output ~/tantular-reports/my-project
```

* Scan only repositories you are authorized to assess.
* Reports must be written **outside** the scanned repository.
* Exit codes: `0` complete, `4` degraded (partial coverage or AI stage unavailable),
  `3` failed, `2` invalid input. A degraded scan always says why in `status_reasons`.
* Findings are scanner evidence (`status: unverified`); review them before acting.
  Tantular Cyber never changes your code.

## 4. Optional: AI explanations

With [Ollama](https://ollama.com) running and a local model installed (see
[install.md §4](install.md#4-optional-local-ai-explanations-ollama)):

```bash
tantular-cyber scan --repo ~/code/my-project --output ~/tantular-reports/my-project --ai \
    --model ghifidanukusumo/tantular:latest
```

Explanations are in Bahasa Indonesia and **advisory**: they never change a finding, its
status or the report status. Excerpts are redacted and sent only to the local Ollama server.

## Limitations

* Python only; five CWE families; synthetic-benchmark precision 0.84 / recall 0.91 are
  regression floors, **not** real-world accuracy.
* The call-graph scanner follows calls up to 3 levels within the repository and skips
  dynamic dispatch; some flows will be missed.
* AI explanation quality was measured only with proxy checks on a small synthetic corpus.

## Privacy

No telemetry, no login, no cloud. See [privacy.md](privacy.md).
