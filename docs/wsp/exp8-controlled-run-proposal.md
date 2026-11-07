# Experiment 8: readiness proposal for one controlled run

Status 2026-11-07: **proposal for review, nothing authorized.** Accepting this proposal does
not authorize any download, model run, account, purchase or expenditure; each needs its own
recorded approval (§7). Prepared from `main` at
`a108489` without downloading any artifact, running any model, contacting any provider or
incurring any charge. Every number below that is not a pin from the existing ledger is an
**estimate or a proposal**, labelled as such. Nothing here is a measurement, a baseline or a
replication result.

It turns the open items of [run-readiness.md](run-readiness.md) (R1–R13, B1–B5, C1–C6) and the
draft [protocol](exp8-replication-protocol.md) into one plan for a first controlled run: a
small **development-set smoke run** on non-MATH-500 items, followed, only if that passes and
the decisions below are made, by the frozen confirmatory MATH-500 run.

## 1. Who decides what

| Kind | Meaning | Examples |
|---|---|---|
| **Resolved here** | Settled from the pinned public metadata already in the repository, or a design proposal Claude Code can implement and test synthetically once you accept it | Checkpoint and tokenizer pins (§2); sequential-load design (§3.1); the analysis method proposal (§4); smoke commands and stop conditions (§6) |
| **Needs measurement** | Can only be settled on the isolated host after an approved fetch; the proposal says exactly what to measure | File hashes and `verify-edit`; KV-cache headroom and throughput; effective stop tokens (§2, §3.2) |
| **Needs your decision** | Roles, money, accounts, licences, downloads, and the statistical choices the independent evaluator must own | R1–R5, R10–R13; hardware provider; vLLM version; the analysis method's adoption; spending ceiling (§5, §7) |

Claude Code proposes and writes code; it does not hold any of the four roles, approve its own
work, operate the isolated host or see held-out results before the evaluator.

## 2. Exact artifacts and unresolved differences

### 2.1 Pinned (resolved; from `ledger/exp8-source-pins-2026-11-03.json`)

| Artifact | Pin | Status |
|---|---|---|
| Base checkpoint | `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`, `model.safetensors-00001-of-00001.safetensors`, LFS SHA-256 `04b1c301…4696` | Pinned; local hash **to measure** after fetch |
| Edited checkpoint (p50c) | `RikiyaT/Qwen3.5-0.8B-weight-space-programming@d1671a178705bcab7644806d23c535517f35723b`, same file name, LFS SHA-256 `fd1e42ac…b553` | Pinned; third-party rebuild; local hash and `verify-edit` **to measure** |
| Edit manifest | `RikiyaT/lm_understand-logs@4d09ba6d…94f:exp8_math/runs/p50c/edit/manifest.json` (git blob `d78897bb…`) | Pinned. Expected edit: `layers.15` and `layers.19` `mlp.down_proj.weight`, `[1024, 3584]` BF16, 65,260 + 64,928 changed entries, four tensor hashes in the ledger |
| Tokenizer | `tokenizer.json` SHA-256 `5f9e4d49…cb42`, identical in both HF repos | Pinned; the authors' export copy is **unknown** |
| Chat template | `chat_template.jinja` git blob `0ef09f21…`, identical in both repos and in the authors' export | Established identical |
| `config.json` | git blob `715f0448…` in both HF repos; language hyperparameters equal to the export's `config.json` apart from `architectures`, a duplicated `partial_rotary_factor` and `transformers_version` | Established |
| Inference settings | thinking on; `temperature 0.6`, `top_p 0.95`, `top_k 20`, `presence_penalty 1.5`, `max_tokens 32768`; `n = 4` per request, request seed `0` for every problem; `max_model_len 34816`, `dtype bfloat16`, `LLM(seed=0)` | Pinned in the draft protocol (all three arms share them; the parser enforces it) |
| Prompt | user message `problem + "\nPlease reason step by step, and put your final answer within \boxed{}."` | Pinned |
| Grader rule | last `\boxed{}`/`\fbox{}` after cutting a stray reasoning block; no box = incorrect; `math-verify 0.9.0` `verify(parse(gold), parse(pred))`, 5 s timeouts on the main thread; `latex2sympy2_extended 1.11.0`, `antlr4-python3-runtime 4.13.2` | Rule pinned; **adapter not written or reviewed**; sympy version **unknown** |
| Dataset | `HuggingFaceH4/MATH-500`, split `test`, ID `unique_id`, 500 items | **Revision and content hash not pinned** (the authors loaded it without a revision) |

### 2.2 Unresolved differences from the published experiment

These stay listed in every report (`required_limitations`) or block the claim (protocol §8).

| Difference | Effect | Proposed handling | Owner |
|---|---|---|---|
| Multimodal release (`Qwen3_5ForConditionalGeneration`) evaluated in place of the authors' unreleased text-only export | Untouched-weight identity with the authors' checkpoint cannot be shown (limitation 1) | State in every result. The `text_only_reconstruction` arm stays unfunded unless you fund it (R2) | You (R2) |
| No `generation_config.json` in the release (the export had one: eos `248044`, `use_cache`) | Engine defaults could differ | Pass every sampling parameter explicitly; record the **effective** stop tokens and sampling parameters per request; choose the vLLM generation-config source (R6). *Measure in the smoke run* that the effective eos/stop set contains `248044` | Research lead (R6) |
| Export `tokenizer.json` unknown | Tokenization identity with the authors is unproven | Record rendered-prompt SHA-256 per request; both arms use the same file | Limitation |
| Edited file is 432 B larger than the base | Unexplained header/metadata difference | `verify-edit` reports it; record the cause before any run | Measure (C5) |
| Engine, library versions, GPU and tensor-parallel layout unknown for the authors' run | Exact output reproduction is impossible; only statistical replication | Pin our own versions and declare the deviation. *Provisional suggestion:* the vLLM version the third-party re-run used (`0.33.0`), so our run is comparable with at least one documented setup; **unverified** until its support for this model is checked | Research lead (R6) |
| MATH-500 revision unpinned by the authors | Item set or answers may differ | Pin a revision and content hash; compare the 500 `unique_id`s and answers with the authors' public `per_problem.json` files. Any mismatch blocks the replication claim (protocol §8) | Research lead (R8) |
| sympy version not stated | Edge-case grading may differ | Pin the version that `math-verify 0.9.0` resolves to in the runtime image; validate on a human-graded sample (R9) | Evaluator (R9) |
| Authors' batch composition (`rows[i::10]`, one `generate` call per shard of 50 prompts × 4 samples) | Batch composition can change logits and therefore outputs | Reproduce the shard batching (§3.1) and declare that bitwise equality is not expected | Research lead (R7) |
| Authors' abort/re-run history not recorded | Unknown whether outcomes were replaced | Limitation; our run records every attempt | Limitation |

## 3. Execution: sequential loading and batching

### 3.1 Design (resolved; to be implemented and synthetically tested on your approval)

The current `run_paired` interleaves arms per item, which assumes all three models are loaded
at once. Three vLLM engines at `gpu_memory_utilization 0.85` cannot share one GPU. Proposed
harness change (C4):

- **Protocol field** `execution_order: "per_arm_sequential"` (the current behaviour becomes
  `"interleaved"`, the default for existing protocols). It is part of the hashed protocol.
- **Order:** `base` → `edited` → `base_reload`, each arm a **separate process and engine
  load** (distinct `load_id`; the GPU is released between arms). The order is recorded, and
  the drift report already expects the reload to be a separate load.
- **Batching:** a batch request type carrying a whole shard. Shards reproduce the authors'
  `rows[i::10]` over the protocol's item order, so each `generate` call holds 50 prompts with
  `n = 4`. The shard index of every record is stored; pairing by (item, sample) is unchanged.
- **Failure handling:** a shard whose call fails produces error rows for every (item, sample)
  in it (never dropped); a crash of the arm process ends the run as `run_status: aborted`,
  which `analyze` and `drift` refuse. No automatic re-run; an operator re-run is a new run
  directory, logged as a deviation with both attempts kept.
- **Where the code lives (needs your decision):** the inference adapter (C1) needs `vllm`
  and `torch`, and the grader adapter (C2) needs `math-verify`. `tools/wsp` is
  standard-library-only by a tested rule. Proposal: a separate `tools/wsp_runtime/` package
  with its own pinned lock file and its own isolation test (no network imports outside the
  fetch step, no `tantular_cyber` import, never imported by `tools/wsp`), runnable only on
  the isolated host. Its tests use stub engines and synthetic strings, never MATH-500
  content in the repository.

### 3.2 Hardware sizing (estimates; needs measurement and your decision)

- **Not the development Mac.** vLLM needs a CUDA GPU, and the laptop is not an isolated
  identity.
- **Weights:** about 0.8 B language parameters in BF16 (≈1.6–2 GB with the vision tower);
  this is not the constraint.
- **KV cache:** each shard holds up to 200 sequences at up to 34,816 tokens. The per-token
  KV size depends on the model's attention layout, which the repository has not recorded;
  **measure** peak memory and achievable concurrency in the smoke run before choosing the
  confirmatory host.
- **Proposed host:** one NVIDIA GPU, `tensor_parallel_size 1`, one engine at a time at
  `gpu_memory_utilization 0.85`. A 24 GB card (L4/A10G class) is the minimum worth trying;
  a 40–80 GB card (A100/H100 class) is preferred so a full shard fits without the scheduler
  queuing most of it. Either way the effective concurrency is recorded, because it changes
  batch composition.
- **Output volume (estimate):** at most 6,000 generations × 32,768 tokens ≈ 197 M output
  tokens. Using the authors' published token-limit rates (93% base, 33% edited) as a rough
  guide, the expected volume is about 150–170 M tokens, dominated by the two base arms.
- **Time (provisional estimate):** at an assumed aggregate 1,000–10,000 output tokens/s,
  the confirmatory run would take about 4–45 GPU-hours. The throughput range is an
  assumption, not a measurement; the smoke run measures the real rate.

## 4. Analysis method for four samples per problem (proposal; needs independent review)

The current tools keep repeated-sample results descriptive: `wsp.py analyze` uses an
item-clustered percentile bootstrap (approximate), and `tools/research` labels every
multi-sample interval `unreviewed_heuristic`. A confirmatory run needs one method the
independent evaluator has reviewed and accepted **before** the run.

**Estimand.** The 500 MATH-500 problems are treated as **fixed** (the replication target is
that benchmark, not a population of problems). For problem *i* and arm *a*, let *p<sub>i,a</sub>*
be the probability that one sample is graded correct. The primary estimand is

θ = (1/500) Σ<sub>i</sub> (p<sub>i,edited</sub> − p<sub>i,base</sub>),

the change in mean per-sample accuracy on this item set, with `count_as_incorrect` for
missing outcomes (matching the authors' `ok / total`).

**Estimator.** *d<sub>i</sub>* = (mean correctness of the edited arm's four samples on *i*) −
(the same for the base arm), so *d<sub>i</sub>* ∈ {−1, −0.75, …, 1}, and θ̂ = mean of *d<sub>i</sub>*.
This equals the difference of the two arms' overall accuracies.

**Proposed interval (non-binding until the evaluator approves it): Hoeffding bound on the
item-level differences.**
- Assumptions: the *d<sub>i</sub>* are independent across problems and bounded in [−1, 1].
  Nothing is assumed about the dependence between a problem's four samples (they come from one
  seeded request and are averaged first), nor that problems are identically distributed:
  Hoeffding's inequality holds for independent, non-identical bounded variables, so it covers
  the fixed-item estimand.
- Two-sided 95% interval, **if** the independence assumption holds: θ̂ ± 2·√(ln(2/0.05) /
  (2·500)) ≈ θ̂ ± **0.122**, a finite-sample bound with no normal or bootstrap approximation.
  It is conservative by design.
- Planning illustration only (not a power calculation): the half-width is fixed at about
  0.12 whatever the data, so an observed difference of about 0.12 or less could not be
  distinguished from zero. If, hypothetically, the observed difference were near the
  authors' reported +0.40, the lower bound would be near +0.28. A power calculation would
  need a stated effect, a model of the per-problem variability and the approved randomness
  design; none of these exists yet.
- Same method for the secondary token-limit-rate difference (also bounded in [−1, 1]),
  reported but not part of the decision rule (no multiplicity adjustment is needed for a
  single primary endpoint; secondaries are labelled descriptive).
- The current item bootstrap and exact per-arm counts stay in the report as descriptive.

**What the evaluator must approve: the randomness design as well as the formula.**
- The width above is valid only under independence across problems, and the authors' design
  does **not** establish that. Every problem uses the same request seed (`0`), all four
  samples come from one seeded request, and problems share a batched `generate` call whose
  composition can change the numerics. Sampling is therefore pseudo-random and possibly
  coupled across problems; what plays the role of "random" in the interval is a modelling
  choice, not a property of the run.
- The evaluator must decide what randomness the interval is meant to describe, and whether
  the replication design supports it, for example: accept the authors' fixed-seed, batched
  design and treat the interval as a model-based approximation; or use a declared
  alternative arm with per-problem seeds (a deviation from the authors' settings, to be
  recorded as such). Until that decision, the Hoeffding interval is reported as
  **proposed, non-binding**.
- A conservative bound loses power for small effects; tighter alternatives (empirical
  Bernstein, a sign-flip randomization test for the null of no effect) need their own review.
- **Agreement with the published result** (R10) is a separate criterion. Options for the
  evaluator: (a) the 95% interval for θ contains the authors' +0.402; (b) θ's lower bound
  exceeds a stated fraction of +0.402; (c) report the comparison descriptively only. It
  must be chosen before confirmatory results.

**What would be built (only after the evaluator accepts the method and the randomness
design):** a `hoeffding_item_mean` method in
`wsp_eval/analysis.py` (and, if `tools/research` is to consume Experiment 8, a matching binding
status for multi-sample items there), with synthetic coverage simulations as tests. Until
then, multi-sample results remain descriptive.

## 5. Isolation, evidence storage, roles and spending

### 5.1 Isolated compute (needs your decision; operator verifies)

- A separate cloud account or project with a research-only identity; no production
  credentials, no production model registry, no deployed GGUF files reachable from it.
- One GPU host, created for the run and destroyed after the evidence is copied out.
- Network: outbound allowed only during the approved fetch step (Hugging Face and the pinned
  GitHub commit); runs execute with `HF_HUB_OFFLINE=1` and egress blocked.
- A pinned runtime image (digest recorded): the chosen vLLM, torch, CUDA, transformers,
  `math-verify 0.9.0`, `latex2sympy2_extended 1.11.0`, `antlr4-python3-runtime 4.13.2` and the
  sympy pin.

### 5.2 Protected evidence storage (needs your decision)

- An object-storage bucket in a **different** account from the compute host, with
  write-once retention (object lock) on run directories and their run-manifest identifiers.
  This is the "externally recorded value" the research workflow needs: the operator can
  write, nobody can overwrite, and the evaluator reads.
- Checkpoints and dataset files are read-only copies there, with their SHA-256 recorded.
- Raw outputs are kept (R13 proposal: yes, needed for grading audit and the human sample),
  readable only by the evaluator and research lead.
- The repository gets only sanitized ledger entries (hashes, settings, counts), never
  weights, raw outputs or held-out content.

### 5.3 Roles (needs your decision; pseudonyms)

| Role | Responsibility | Constraint |
|---|---|---|
| Research lead | Owns the protocol, R6–R8 choices and the budget request | Not the decider on their own run |
| Independent evaluator | Owns the grader review, the analysis method acceptance (§4), R10 bounds, held-out access | Not the operator or research lead |
| Operator | Provisions the isolated host and storage, runs only approved commands, records versions | Not an approver |
| Release owner | Records accept / reject / request-more-evidence | Not the operator or research lead; research acceptance deploys nothing |

### 5.4 Spending ceiling (provisional proposal; needs your decision)

**All figures in this section are provisional.** They rest on the unmeasured throughput
assumption in §3.2 and on an assumed on-demand single-GPU price of roughly US$1–4 per
GPU-hour, which is not a quote. They must be re-checked against the chosen provider's current
pricing, the vLLM version's compatibility with this model, and the throughput the smoke run
measures, before any confirmatory budget is approved.

| Stage | Proposed hard cap | Basis |
|---|---|---|
| Development smoke run (§6) | **US$25** | ≈ 100 generations at full length plus setup; well under 5 GPU-hours |
| Confirmatory run | **US$250** | Upper end of 4–45 GPU-hours, with room for one logged operator re-run |
| Contingency (storage, image builds, a failed attempt) | **US$75** | |
| **Total ceiling (provisional)** | **US$350** | Re-check against provider pricing, engine compatibility and smoke-run throughput |

Cost metering must stop the host at each cap (B5). The `text_only_reconstruction` arm is not
included.

## 6. Development-set smoke run

**Purpose:** prove the pipeline end to end on the isolated host and measure memory,
throughput, truncation cost and failure modes. **No performance claim, no MATH-500 content.**

**Items (needs your decision on the source; download needs approval):** 8 problems from the
**MATH training split** (disjoint from the MATH-500 test subset), pinned by revision and content
hash, labelled `development`. 8 problems × 4 samples × 3 arms = 96 generations, at the real
32,768-token limit so the cost and truncation behaviour are measured, not guessed.

**Smoke protocol:** a copy of the draft protocol with `synthetic: false`, `frozen: false`,
the smoke dataset, and the chosen engine values filled in. Unfrozen, so every verdict is
descriptive by construction.

**Commands** (on the isolated host; those marked *to build* do not exist yet):

| Step | Command | Expected output | Exists |
|---|---|---|---|
| 1. Approved fetch | Operator downloads the two pinned checkpoint revisions, the pinned logs commit's `runs/p50c/edit/manifest.json`, and the smoke items, into research storage | Files present | Operator step |
| 2. File hashes | `sha256sum` on both weight files and both `tokenizer.json` | Exactly the ledger values | Yes (OS) |
| 3. Edit check | `tools/wsp/wsp.py verify-edit BASE EDITED manifest.json` | `status: verified`; `tensors_checked` = the two `down_proj` tensors; `manifest_sha256` recorded. `wsp.py diff BASE EDITED` then shows the header/metadata difference behind the extra 432 B | Yes |
| 4. Dataset check | `tools/wsp/wsp.py dataset smoke-items.jsonl` | 8 items, no duplicates, all `development` | Yes |
| 5. Protocol check | `tools/wsp/wsp.py protocol smoke-protocol.json` | Valid; prints `protocol_sha256` | Yes |
| 6. Run | `tools/wsp_runtime/run.py --protocol smoke-protocol.json --items smoke-items.jsonl --output RUN_DIR --keep-outputs` | `run_status: complete`, 96 records, per-arm load records, effective sampling parameters and stop tokens per request, peak memory, tokens/s | *To build* (C1–C4) |
| 7. Record identifier | `tools/research/improve.py run-manifest-hash RUN_DIR`, then write the printed `run_manifest_sha256` to the protected bucket | The **canonical-JSON** run-manifest identifier, stored write-once outside the run. A plain `sha256sum manifest.json` is a different value (it hashes the file's bytes, not the parsed content) and must not be recorded in its place | Yes |
| 8. Analysis | `tools/wsp/wsp.py analyze --run RUN_DIR --protocol smoke-protocol.json` | Per-arm counts; verdict `descriptive_only_min_effect_not_set` | Yes |
| 9. Drift | `tools/wsp/wsp.py drift --run RUN_DIR --protocol smoke-protocol.json` | Identity checks for `base_reload` vs `base`: all `pass` (artifact digests, distinct `load_id`, rendered-prompt hashes); verdict `descriptive_only_tolerance_not_approved` | Yes |
| 10. Grader spot check | Evaluator hand-grades every smoke sample with a box | Agreement recorded; disagreements block R9 | Manual |

**Stop conditions** (stop, record, do not continue to the confirmatory stage):
1. Any file hash differs from the ledger, or `verify-edit` is not `verified`.
2. Any identity check between `base` and `base_reload` is `fail`, or rendered-prompt hashes
   differ between arms.
3. The effective stop/eos set or sampling parameters differ from the protocol, or differ
   between arms.
4. Any adapter error, abnormal finish (`finish_reason` other than `stop`/`length`), wrong
   sample count, or grader crash that is not explained and fixed in reviewed code.
5. The run aborts, an out-of-memory error occurs, or the engine silently lowers
   concurrency below one shard without that being recorded.
6. Any outbound network connection during steps 6–10.
7. The cost meter reaches the smoke cap.
8. Evaluator hand-grading disagrees with the adapter on any sample.

Stop conditions are operational. None depends on the accuracy observed, so the smoke run
cannot be used to tune the confirmatory rule.

## 7. Decisions needed from you (in gate order)

| # | Decision | Recommendation |
|---|---|---|
| R1 | Assign the four roles (pseudonyms) | — |
| R2 | Spending ceiling | Provisionally US$350 total, with stage caps (§5.4); confirm after pricing, compatibility and smoke-run throughput are checked |
| R3 | Cloud provider, isolated account, GPU type | One 40–80 GB GPU if available, else a 24 GB card for the smoke run first |
| R5 | Human licence review (logs repository parts, Qwen Apache-2.0, MATH and MATH-500 licences) | Before any download |
| R4 | Authorize the fetch of the pinned checkpoints, the manifest and the smoke items | After R5 |
| — | Where the runtime adapters live | `tools/wsp_runtime/` with its own lock file and isolation test (§3.1) |
| R6 | vLLM / transformers versions and generation-config handling | Provisionally vLLM 0.33.0, pending a compatibility check; explicit parameters plus recorded effective defaults |
| R7 | Reproduce the shard batching | Yes (§3.1) |
| R8 | MATH-500 revision and the ID/answer comparison rule | A metadata-only lookup of the current revision needs your permission (no files); any mismatch blocks the replication claim |
| R9 | Grader adapter review, sympy pin, human sample size | Stratified human sample of 50 confirmatory samples (by arm × graded correct/incorrect × boxed/unboxed) |
| §4 | Accept, change or reject the Hoeffding method **and** the randomness design it relies on; choose the agreement criterion | Evaluator decides before any confirmatory result; until then the interval is non-binding |
| R10 | `min_meaningful_effect`, drift bounds | Evaluator, after the smoke run, before freezing |
| R11 | Freeze | After everything above |
| R12 | Smoke items from the MATH training split as `development`; MATH-500 as `heldout` | Yes |
| R13 | Keep raw outputs in protected storage | Yes |

## 8. Follow-ups, prioritized by the first controlled run

| Follow-up | Blocks smoke run? | Blocks confirmatory run? | Note |
|---|---|---|---|
| C1–C4 runtime adapters and sequential/batch execution (§3.1) | **Yes** | Yes | New code, reviewed, synthetically tested |
| C3 dataset loader for the pinned MATH/MATH-500 files | **Yes** | Yes | No network; content hash checked |
| Checkpoint read-then-hash order in `diff_checkpoints` (#26 follow-up) | No (single trusted operator, protected storage) | **Should fix first** | Cheap; the confirmatory `verify-edit` should hash the bytes it compared |
| Strict-identifier fix (separate worktree, unreviewed) | No | **Yes, before freezing** | The frozen protocol and ledger carry digests, pseudonyms and timestamps the current validation accepts with a trailing newline |
| `outputs.jsonl` hash recorded but not verified (run-readiness E) | No | **Yes, before the human grading sample** | The audit reads kept outputs |
| §4 analysis method implemented and reviewed | No (smoke is descriptive) | **Yes** | |
| `verify-edit` traceback on a non-object or deeply nested manifest (#26 follow-up) | No | No | Fails loudly; no report |
| `tools/research` follow-ups X1, X2/N2, N1, N7, F5, F6, J7, controls-register row | No | No, unless Experiment 8 is also passed through `tools/research` | Track; no cleanup cycle needed now |

## 9. Next steps, if you accept this proposal

1. You: R1, R2, R3, R5, then R4 (§7).
2. Claude Code, in parallel and without any artifact: build C1–C4 in `tools/wsp_runtime/`
   with stub-engine tests; the dataset loader; the `execution_order` protocol field; and,
   once the evaluator accepts §4, the analysis method with coverage simulations. Each as its
   own reviewed PR.
3. Operator: provision the isolated host and protected bucket; approved fetch; steps 2–10 of
   §6.
4. Review the smoke evidence; then R6–R10, the strict-identifier and read-then-hash fixes,
   freeze (R11), and only then the confirmatory run.
