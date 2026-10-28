# Benchmark results

Dated results. Method and interpretation: [benchmark.md](benchmark.md). The 2026-10-28
POC-002 baseline below used the earlier standalone latency script (since replaced by
`tantular-cyber benchmark --llm ollama`) on `tests/fixtures/bench`.

## 2026-10-28 — POC-003 first live benchmark

Command: `tantular-cyber -v benchmark --llm ollama --llm-runs 2 --cold --concurrency 1,2
--baseline <offline run> --record-replay ...` (branch `feat/poc-003-benchmark`, corpus manifest 1.0.0).

- Machine: Apple M4 Pro, 24 GB, macOS 27.0.1, Python 3.11.14
- semgrep 1.143.0, Ollama 0.40.2, `tantular-cyber:latest` (qwen35, 9.7B, Q8_0; digest
  `e79106b3b1eb`), think=false, stream=false, temperature 0.1, num_ctx 8192

**Scanner (deterministic):** TP 12, FP 3, FN 3, TN 5 → precision 0.80, recall 0.80, F1 0.80.
All FPs/FNs are the documented known ones; 0 unannotated findings. Gates: all pass.

**Adversarial:** overclaims 50 passed / 7 known gaps; secrets 17 passed / 4 known gaps;
0 unexpected failures.

**LLM explanations (sample-based, n = 16 over 2 runs, 8 locations):** as first scored:
injection_resisted 75%, impact_separated 87.5%, all other checks 100%. Inspection showed
all three failures were **evaluator false alarms** (the model *reported* the injected
"kode aman" claim as a prompt-injection pattern; "bergantung pada" was not recognised as a
hedge). After fixing the checks, the recorded run-1 outputs
(`benchmarks/replays/tantular-cyber-latest-2026-10-28.json`) re-score 100% on every
check (n = 8; Wilson 95% lower bound ≈ 0.68). No overclaim-guard interventions, no canary
leaks, never "high" confidence on the known false positives (4× medium), 11/12 "high" on
true positives. Small n: these are encouraging samples, not a quality guarantee.

**Performance:**

| Run | Kind | Findings | Total s | p50 s | p95 s | Load s | Prompt tok | Output tok | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | cold_after_eviction | 8 | 288.5 | 35.4 | 43.3 | 7.1 | 5128 | 3408 | 13.4 |
| 2 | warm | 8 | 272.1 | 32.5 | 41.6 | 0.0 | 5128 | 3408 | 13.4 |

| Concurrency | Requests | Wall s | Findings/min | p50 s | p95 s |
|---|---|---|---|---|---|
| 1 | 8 | 257.2 | 1.87 | 31.2 | 40.5 |
| 2 | 8 | 257.0 | 1.87 | 64.3 | 67.5 |

- Concurrency 2 gives **no throughput gain** and doubles latency: the server processed
  requests serially in this configuration. Parallelism needs server-side settings
  (e.g. `OLLAMA_NUM_PARALLEL`) and enough memory; not tested.
- Both runs produced identical token counts — generation is effectively deterministic at
  temperature 0.1 on this setup.
- "cold_after_eviction" ≠ cold boot (page cache likely warm).

## 2026-10-28 — POC-002 baseline

- Date: 2026-10-28T13:40:13+00:00
- Machine: Apple M4 Pro (arm64), 24 GB RAM, macOS-27.0.1-arm64-arm-64bit
- Ollama 0.40.2, model `tantular-cyber:latest` (9.7B, Q8_0, 10.7 GB)
- Fixture: `tests/fixtures/bench`, think=false, stream=false, temperature 0.1, num_ctx 8192, sequential calls

| Run | Status | Findings | Total s | Mean s/finding | Min | Max | First-call load s | Prompt tok | Output tok | Output tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (cold) | complete | 6 | 196.7 | 32.3 | 27.7 | 39.3 | 7.8 | 3958 | 2584 | 15.1 |
| 2 (warm) | complete | 6 | 186.3 | 30.5 | 25.0 | 36.9 | 0.0 | 3958 | 2552 | 14.1 |

Per-finding (all runs):

| Run | Finding | Status | Wall s | Load s | Prompt tok | Output tok | tok/s | Guard warnings |
|---|---|---|---|---|---|---|---|---|
| 1 (cold) | `admin.py:7` | completed | 39.3 | 7.75 | 675 | 402 | 14.17 | 0 |
| 1 (cold) | `admin.py:11` | completed | 33.4 | 0.01 | 612 | 477 | 15.29 | 0 |
| 1 (cold) | `orders.py:7` | completed | 28.6 | 0.01 | 662 | 406 | 15.44 | 0 |
| 1 (cold) | `orders.py:12` | completed | 27.7 | 0.01 | 657 | 391 | 15.51 | 0 |
| 1 (cold) | `orders.py:18` | completed | 33.3 | 0.01 | 668 | 474 | 15.26 | 0 |
| 1 (cold) | `orders.py:23` | completed | 31.7 | 0.01 | 684 | 434 | 14.89 | 0 |
| 2 (warm) | `admin.py:7` | completed | 25.0 | 0.00 | 675 | 372 | 15.01 | 0 |
| 2 (warm) | `admin.py:11` | completed | 28.4 | 0.01 | 612 | 396 | 14.1 | 0 |
| 2 (warm) | `orders.py:7` | completed | 30.1 | 0.01 | 662 | 390 | 13.08 | 0 |
| 2 (warm) | `orders.py:12` | completed | 30.7 | 0.01 | 657 | 379 | 12.48 | 0 |
| 2 (warm) | `orders.py:18` | completed | 31.7 | 0.01 | 668 | 481 | 15.34 | 0 |
| 2 (warm) | `orders.py:23` | completed | 36.9 | 0.02 | 684 | 534 | 14.6 | 0 |

Median per-finding wall time across runs: 31.2 s

### Interpretation

* **Generation dominates**: ~400–500 output tokens per explanation at ~14–15 tokens/s
  ≈ 27 s of the ~31 s per finding; prompt evaluation (~650 tokens) and the Semgrep scan
  (~3 s per run) are minor.
* **Model load** on the "cold" run was 7.8 s after `keep_alive: 0` eviction; the weights
  were likely still in the macOS file cache. A first load after reboot was not measured
  and will be slower.
* **Scaling** is linear: calls are sequential, so N findings ≈ N × 31 s
  (default `--max-llm-findings 25` ≈ 13 min). Levers for later: shorter structured output,
  a smaller quantization, parallel requests (`OLLAMA_NUM_PARALLEL`), or only explaining
  high-severity findings.
* All 12 explanations validated against the schema and triggered **0** overclaim-guard
  warnings. `orders.py:23` (`%d` with `int()`) is an intentional scanner false positive;
  whether the model recognises it is a POC-003 quality question.
