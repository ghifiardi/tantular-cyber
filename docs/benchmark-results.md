# Benchmark results

Measured with `make bench` (`benchmarks/llm_latency.py`). Performance only — explanation
quality is the subject of POC-003 (see [benchmark-plan.md](benchmark-plan.md)).

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
