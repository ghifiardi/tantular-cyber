# Benchmark plan (POC-003 proposal)

Latency/throughput has a first baseline in [benchmark-results.md](benchmark-results.md);
the detection and quality measurements below are not yet executed.

POC-002 validates the pipeline on two hand-written fixtures. Before relying on it, measure
scanner recall/precision and LLM explanation quality separately.

## 1. Scanner (deterministic)

* **Corpus**: OWASP Benchmark (Python subset), Semgrep's own rule tests for
  `python.lang.security.audit.formatted-sql-query`, and internal repos with known,
  remediated SQLi commits (before/after pairs).
* **Metrics**: TP/FP/FN per CWE, false-positive rate on the "fixed" side of each pair,
  scan time per 10k LOC on an Apple-silicon Mac.
* **Exit criterion**: ≥ 0.8 recall on the intra-procedural cases the rule claims to cover;
  zero findings on parameterized counterparts.

## 2. LLM explanation (advisory)

Per finding, score the structured output on:

| Criterion | Check |
|---|---|
| Schema validity | `analysis.status == "completed"` without fallback parsing |
| Evidence fidelity | every `confirmed_evidence` item is verifiable in the snippet (human rater) |
| Overclaim rate | share of findings where `validation_warnings` contains a relocated claim |
| Conditional framing | `conditional_impact` states a precondition ("jika …") |
| Remediation correctness | parameterized query suggested; `corrected_code` compiles |
| Language | Bahasa Indonesia, technically accurate |
| Latency | seconds/finding on the reference Mac (first call includes model load) |

Run each model (`tantular-cyber:latest`, future cybersecurity models) with the same
findings set, temperature 0.1, 3 seeds; report mean and variance.

## 3. Robustness

* Prompt-injection fixtures (comments/strings instructing the model) — the explanation must
  not declare the finding safe; scanner facts must be unchanged (already unit-tested).
* Secret-bearing fixtures — no secret value appears in `report.json`, `report.md`, or the
  outgoing prompt.
* Ollama down / slow / truncated (`done_reason=length`) — report is `degraded`, never
  silently `complete`.
