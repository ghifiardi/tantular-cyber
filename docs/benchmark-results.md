# Benchmark results

Dated results. Method and interpretation: [benchmark.md](benchmark.md). The 2026-10-28
POC-002 baseline below used the earlier standalone latency script (since replaced by
`tantular-cyber benchmark --llm ollama`) on `tests/fixtures/bench`.

## 2026-10-29 — POC-005 (callgraph, real-world pointers, model/profile comparison)

Branch `feat/poc-005-accuracy-evaluation`, corpus manifest 1.2.0, Apple M4 Pro 24 GB,
Ollama 0.40.2, think=false, stream=false. Method: [evaluation.md](evaluation.md).

### Scanner detection (deterministic, offline, CI-gated)

`tantular-cyber benchmark` (gate: **pass**), manifest 1.2.0 (corpus grew from POC-004's
1.1.0 with interprocedural and chain cases, so totals are not directly comparable).

| Scanners | TP | FP | FN | TN | Precision | Recall |
|---|---|---|---|---|---|---|
| Semgrep + tantular-callgraph (gated) | 43 | 8 | 4 | 23 | 0.843 | 0.915 |
| Semgrep only | 36 | 8 | 11 | 23 | 0.818 | 0.766 |
| tantular-callgraph only | 30 | 2 | 17 | 29 | 0.938 | 0.638 |

| CWE | TP | FP | FN | TN | Precision | Recall |
|---|---|---|---|---|---|---|
| CWE-89 | 20 | 4 | 2 | 8 | 0.83 | 0.91 |
| CWE-78 | 6 | 1 | 0 | 3 | 0.86 | 1.00 |
| CWE-22 | 5 | 1 | 0 | 3 | 0.83 | 1.00 |
| CWE-79 | 7 | 1 | 1 | 6 | 0.88 | 0.88 |
| CWE-798 | 5 | 1 | 1 | 3 | 0.83 | 0.83 |

* Callgraph-only detections (7): the five former known FNs (`known-fn-cross-function`,
  `known-fn-join`, `known-fn-sqlalchemy-text`, `cmdi-known-fn-sh-c`,
  `path-known-fn-pathlib`) plus `interproc-cross-module` and `interproc-self-dispatch`.
  The callgraph flags **no safe-code location** that Semgrep did not already flag.
* Rule change vs POC-004 on the same corpus: +`chain-sql-inline`, +`chain-sql-variable`,
  nothing lost. Remaining FNs include the pinned limits `interproc-known-fn-depth` and
  `interproc-known-fn-dynamic-dispatch`.
* Evidence context (51 locations): construct + sink visible — window 47/51 (92%), AST
  51/51 (100%); 0 over budget. Adversarial: overclaims 50 passed / 7 known gaps; secrets
  23 passed / 4 known gaps (unchanged).
* Precision/recall are regression floors on a small synthetic corpus, not real-world rates.

### Real-world pointers (descriptive, not a rate)

7 pointer-only locations (no code committed; hashes verified on opt-in fetch): **3 detected**
(wsgidav SELECT, wsgidav DESCRIBE, scitokens), 4 missed with `coverage_candidate` notes
(vendor SDK SQL, directory creation, in-repo command wrapper, helper-based path use).
`wsgidav SELECT` has `development_use` set (it prompted the chain pattern), so only 6 are
held out. Upstream labels are unverified hypotheses. No precision/recall is claimed.

### Model comparison (pinned options, our system prompt; sample-based, never gated)

40 findings × 3 runs (seeds 1–3) = **n = 120 per model**, 0 request errors; Wilson 95%
intervals; exact McNemar on paired samples vs `tantular-cyber:latest`. Started
2026-10-29T03:23Z, code `76a9318`.

| Model | Digest | Weights blob | Quant | Integrity |
|---|---|---|---|---|
| `tantular-cyber:latest` | `e79106b3b1ebcc77` | `4da76d43e612c0aa` | Q8_0 | ✅ stable before/after; one runner, same blob, whole arm |
| `qwen3:8b` | `500a1f067a9f7826` | `a3de86cd1c132c82` | Q4_K_M | ✅ stable before/after |
| `qwen3.5:9b` | `6488c96f…` → `c97eb11d…` | `dec52a44…` → `733ca544…` | Q4_K_M | ⚠ **mixed model files** (1 + 119 samples) — exploratory only |
| `tantular-office:0.5-9b` | `0ed8471c…` → `983443e6…` | `73b25b60…` → `4da76d43…` | Q8_0 | ⚠ **not independent** — 118/120 samples served by the Tantular Cyber weights blob |
| `ghifidanukusumo/tantular:latest` | `2df730ac1bdaafc1` | `4da76d43e612c0aa` | Q8_0 | identity pair only (see below) |

**Integrity issue.** Ollama 0.40.2 ran its automatic "local compat GGUF migration" the first
time it loaded `qwen3.5:9b` (11:18 WIB) and `tantular-office:0.5-9b` (13:38 WIB), rewriting
each manifest to a new weights blob mid-arm. The server log shows the first request(s) were
served by the pre-migration file and the rest by the migrated one. Nobody changed the models;
the harness only snapshotted provenance at start, so the original report shows pre-run
digests. Fixed in this PR: provenance is re-captured after the arms, changes are listed and
affected arms are flagged `provenance_changed`. The migrated `tantular-office` file is
byte-identical (same blob hash) to the Tantular Cyber weights, which explains why its results
track the reference (a single discordant pair across all paired checks). Per-response digest recording
is a follow-up.

| Model | evidence_grounded | impact_separated | remediation_sound | cwe_consistent | calibrated | p50 / p95 s | output tok | tok/s |
|---|---|---|---|---|---|---|---|---|
| `tantular-cyber:latest` | 57% (0.48–0.66) | 92% (0.85–0.95) | 98% (0.93–0.99) | 100% (0.97–1.00) | 98% (0.93–0.99) | 25.9 / 30.8 | 523 | 23.1 |
| `qwen3:8b` | **95%** (0.89–0.98) | **100%** (0.97–1.00) | 83% (0.76–0.89) | 96% (0.91–0.98) | 97% (0.92–0.99) | 12.9 / 21.9 | 305 | 28.9 |
| `qwen3.5:9b` ⚠ | 56% (0.47–0.65) | 87% (0.79–0.92) | 96% (0.91–0.98) | 100% (0.97–1.00) | 96% (0.91–0.98) | 31.2 / 46.2 | 526 | 20.5 |
| `tantular-office:0.5-9b` ⚠ | 57% (0.48–0.66) | 91% (0.84–0.95) | 98% (0.93–0.99) | 100% (0.97–1.00) | 98% (0.93–0.99) | 24.4 / 29.5 | 524 | 23.6 |

All four: `schema_valid`, `no_unsupported_assertions` and `bahasa_indonesia` 100% (0.97–1.00).
`injection_resisted` applies to only 6 samples per model (2 findings × 3 runs; Tantular 5/6,
others 6/6) — too few to compare.

**Tantular Cyber vs Qwen3 8B (the clean pair):** Qwen3 8B is better on `evidence_grounded`
(27 vs 0 discordant pairs, p < 0.001) and `impact_separated` (10 vs 0, p = 0.002), worse on
`remediation_sound` (3 vs 20, p = 0.0005), and about 2× faster at p50 with ~40% fewer output
tokens. The comparison is confounded with quantization (Q4_K_M vs Q8_0) and model family; it
is **evidence for evaluating models per security task**, not a ranking. `prompt tok` differs
(908 vs 811) because the tokenizers differ.

**Same weights, pinned request:** `tantular-cyber:latest` and `ghifidanukusumo/tantular:latest`
gave byte-identical outputs on 3 findings (seed 1) — under our pinned request the Modelfile
profile has no effect. `tantular-cyber:latest` is a profile of those weights, not a separate
fine-tune.

Machine idle-sleep paused the run between requests (gaps of 2–14 min during the `qwen3.5:9b`
arm); no single request exceeded 56 s, so per-request latency is unaffected but wall times
in the raw report are not comparable.

### Prompt/context optimization (`tantular-cyber:latest`, n = 80 per arm)

40 findings × 2 runs, pinned options, 0 errors; reference = `baseline` @ num_ctx 8192.

| Arm | evidence_grounded | impact_separated | remediation_sound | other checks | p50 / p95 s (all) | p50 / p95 s (uncontended) | Decision |
|---|---|---|---|---|---|---|---|
| baseline | 56% (0.45–0.67) | 90% (0.81–0.95) | 98% (0.91–0.99) | 98–100% | 22.9 / 27.4 | 22.8 / 27.1 (n = 59) | reference |
| baseline @ctx4096 | 56% | 90% | 98% | identical | 23.9 / 51.3 | same (n = 80) | **do not adopt** — no speed-up (+4%) |
| concise | 72% (0.61–0.81) | 96% (0.90–0.99) | **82%** (0.73–0.89) | 100% | 20.5 / 39.9 | same (n = 80) | **do not adopt** — remediation 12 net losses (p = 0.0005) |
| concise @ctx4096 | 72% | 96% | **82%** | 100% | 19.1 / 21.3 | same (n = 80) | **do not adopt** — same quality loss |

* Quality decides first: `concise` improves evidence grounding (19 vs 9 discordant,
  p = 0.087, not significant) but loses remediation in 12 paired samples with no gains, which
  fails the rule regardless of its 10–17% latency gain. The default stays `baseline` / 8192.
* num_ctx 4096 produced the same outputs as 8192 (prompt + output stay below 4096 tokens);
  its p95 of 51 s comes from a few slow requests without CPU contention — latency only.
* **CPU contention:** offline tests ran during the baseline arm (≈14:30–14:36 and
  14:43:25–14:44:23 WIB). Samples were classified by their full request interval (start =
  server completion − duration, Ollama log): 21/80 baseline samples overlap. All are kept;
  latency is reported with and without them and no decision changes.

### Same-weights Modelfile-profile comparison (n = 120 per arm, complete)

`--modes profile` (no system message, no pinned options; each Modelfile's SYSTEM and
PARAMETERs apply, our instructions travel in the user message) on `tantular-cyber:latest`
(digest `e79106b3b1ebcc77`) vs `ghifidanukusumo/tantular:latest` (digest
`2df730ac1bdaafc1`): identical weights blob `4da76d43e612c0aa` (Q8_0); the profiles differ in
SYSTEM prompt (hash `449c9a4f…` vs `c32dc580…`) and `num_ctx` 8192 vs 32768. 40 findings ×
3 seeds, 0 errors, **no provenance change** before vs after (no arm flagged). Started
2026-10-29T09:45Z, code `76a9318` + drift check.

| Arm | evidence_grounded | impact_separated | remediation_sound | cwe_consistent | calibrated | other checks |
|---|---|---|---|---|---|---|
| `tantular-cyber:latest` profile | 78% (0.70–0.85) | 89% (0.82–0.94) | 98% (0.94–1.00) | 98% (0.93–0.99) | 97% (0.92–0.99) | 100% |
| `ghifidanukusumo/tantular:latest` profile | 76% (0.68–0.83) | 90% (0.83–0.94) | 99% (0.95–1.00) | 100% (0.97–1.00) | 98% (0.93–0.99) | 100% |

Paired (exact McNemar): evidence 16 vs 14 discordant (p = 0.86), impact 11 vs 12 (p = 1.0),
remediation 1 vs 2, cwe 0 vs 3 (p = 0.25), calibrated 0 vs 1 — **no detectable difference
between the two profiles at n = 120**. The result is the **combined** effect of SYSTEM prompt
and `num_ctx` (and Modelfile sampling defaults, which are equal here), not of the SYSTEM
prompt alone. `injection_resisted` 6/6 for both (n too small to compare).

* **Latency is not comparable** and is not interpreted: the machine ran on battery, slept
  17:00–18:57 WIB and the offline tests/benchmark ran during the `tantular-cyber` arm.
  Classified by full request interval: 16/120 `tantular-cyber` samples overlap those
  windows (p50/p95 26.8/43.5 s all vs 27.0/40.2 s uncontended, n = 104); 0/120 for
  `ghifidanukusumo/tantular` (23.4/36.9 s). All samples kept.
* **Observation for the next milestone (not a finding):** in profile mode both profiles
  score `evidence_grounded` ~77% vs ~57% for the same weights under our pinned request and
  system prompt. The modes differ in several ways at once (system-prompt placement,
  temperature 0.2 vs 0.1, presence/repeat penalties, num_predict), so this is a hypothesis
  for the evidence-grounding investigation, not a recommendation to change defaults.

### Limitations and follow-ups

* All LLM numbers are deterministic proxies on a small synthetic corpus; no human review
  data exists yet (the ingestion workflow is ready). Nothing here gates CI.
* `evidence_grounded` is ~57% for Tantular Cyber despite AST context and call paths. Whether
  the cause is the evidence supplied, the model's use of it, the prompt or the rubric is the
  next model-quality milestone — not POC-005.
* Same-weights profile comparison: complete; no detectable profile effect at n = 120.
* `qwen3.5:9b` and `tantular-office:0.5-9b` results are exploratory (mixed files /
  non-independent); a clean rerun would need pre-warmed (migrated) models.
* Real-world sample: 7 pointers, labels unverified; descriptive only.
* Known scanner limitations stay pinned as tests (call depth, dynamic dispatch).

## 2026-10-29 — POC-004 (five CWEs, AST context)

Branch `feat/poc-004-security-intelligence`, manifest 1.1.0 (27 cases, 66 annotations),
same machine/model as below (Apple M4 Pro 24 GB, Ollama 0.40.2, `tantular-cyber:latest`
digest `e79106b3b1eb`, think=false, stream=false).

**Scanner (deterministic, offline):** TP 32, FP 7, FN 7, TN 20 → precision/recall 0.82.

| CWE | TP | FP | FN | TN | Precision | Recall | Floor |
|---|---|---|---|---|---|---|---|
| CWE-89 | 13 | 3 | 3 | 5 | 0.81 | 0.81 | 0.75 |
| CWE-78 | 4 | 1 | 1 | 3 | 0.80 | 0.80 | 0.70 |
| CWE-22 | 3 | 1 | 1 | 3 | 0.75 | 0.75 | 0.70 |
| CWE-79 | 7 | 1 | 1 | 6 | 0.88 | 0.88 | 0.70 |
| CWE-798 | 5 | 1 | 1 | 3 | 0.83 | 0.83 | 0.70 |

Every FP/FN is a documented known limitation (one of each per new CWE by design). The
CWE-89 rule is unchanged; all original CWE-89 annotations keep their outcome (the
`expected_detections` gate) and the rate moved 0.80 → 0.81 only because a context-distance
TP was added. These are regression floors on a small synthetic corpus, not real-world rates.

**Evidence context (deterministic, 39 detected locations):** construct + sink visible —
window 36/39 (92%), AST 39/39 (100%); mean 6.1 vs 4.5 lines, 206 vs 203 chars, max 619
chars (budget 4000), 0 over budget, 3 template fallbacks (by design).

**Adversarial:** overclaims 50 passed / 7 known gaps; secrets 23 passed / 4 known gaps
(new: zero-width/bidi/homoglyph credential names — two of these were real redaction gaps,
fixed in this PR).

**Live LLM, AST vs window (one run each, same 13 findings, n = 13, sample-based):**

| | AST | Window |
|---|---|---|
| Prompt tokens (total) | 9,597 | 9,050 |
| Output tokens (total) | 5,829 | 5,597 |
| Per-finding p50 / p95 | 19.7 s / 38.9 s | 19.9 s / 23.0 s |
| Output tok/s | 22.1 | 25.0 |
| evidence_grounded* | 13/13 | 12/12 (1 n/a) |
| impact_separated* | 11/13 | 9/13 |
| all other checks | 100% | 100% |

\* After fixing two evaluator false alarms found in these runs (quote-style/paraphrased
code citations; "kondisional"/"memerlukan" hedges), both runs re-scored identically from
their review queues. Wilson 95% intervals overlap (e.g. impact 0.58–0.96 vs 0.42–0.87):
**no measurable quality difference at this n**. AST context costs ~6% more prompt tokens;
the p95 gap comes from one long generation. Throughput was higher than on 2026-10-28
(22–25 vs ~13 tok/s) on the same machine — compare performance only within a session.
The AST run's raw outputs are kept as `benchmarks/replays/tantular-cyber-latest-2026-10-29-ast.json`.

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
