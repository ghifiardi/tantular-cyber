# Experiment 8 replication protocol (draft for review)

Status 2026-11-04: **draft, not frozen, not approved, nothing executed.** No weights,
datasets or model outputs have been downloaded and no inference has run. This document makes
no claims about a baseline, a replication, safety, or a model improvement. All numbers quoted
from the post or the model card are *reference claims by their authors*, not Tantular results.
License review: **pending (human)**.

Machine-readable form: [protocols/exp8-replication-draft.json](protocols/exp8-replication-draft.json)
(`wsp.py protocol` validates it; it cannot be frozen while decisions remain open).
Source pins: [ledger/exp8-source-pins-2026-11-03.json](ledger/exp8-source-pins-2026-11-03.json).
Logs cited below are `RikiyaT/lm_understand-logs@4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`,
read as text through the GitHub API. The authors' code was read, never executed.

## 0. Decision log

| ID | Date | Decided by | Decision | Does not authorize |
|---|---|---|---|---|
| D1-scope | 2026-11-04 | research owner | Proceed with the **qualified MATH-500 replication** on the published multimodal checkpoints, with the claim wording below. Other benchmarks get separate protocols. The text-only reconstruction arm is **proposed only**, pending budget and a separate protocol decision | Downloads, model runs, freezing, or completing the license review |

Conditions recorded with D1:
- The engine and versions, hardware, MATH-500 revision and grading implementation are chosen
  **before** the drift bounds and the agreement criterion are approved.
- Those bounds and `min_meaningful_effect` are approved **before** any confirmatory results
  are collected.

The harness enforces part of this ordering:
- An approved tolerance requires a frozen protocol, and a frozen protocol cannot contain the
  open `REPLACE` choices.
- Approval must predate the run start.

The order of human approvals between those two points is a process control.

**Claim wording (fixed before any results):** *Replication of the published p50c
checkpoint's MATH-500 behaviour under the authors' released evaluation settings, using the
multimodal Qwen3.5-0.8B release in place of the authors' unreleased text-only export, on a
different engine version and hardware. This is not a reproduction of the edit's discovery.*

**Required limitations.** These are listed in `analysis.required_limitations` and copied
verbatim into every analysis report:
1. The comparison of the published base and edited checkpoints verifies what changed within
   those files; it cannot establish that the authors' unreleased text-only export had
   identical untouched weights.
2. The multimodal release was evaluated in place of the text-only export. The export's
   `tokenizer.json` and generation-config handling are not reproduced exactly.
3. The edited checkpoint is a third-party rebuild. Its identity rests only on this run's
   tensor checks.
4. The engine, library versions and hardware differ from the authors' run or are unknown
   for it. Only statistical replication is attempted, not output-level replication.
5. This is not a reproduction of the controller's discovery of the edit.

## 1. Claim scope

| Claim | In scope | Note |
|---|---|---|
| Behavioural replication: the published 50-pass edit (`p50c`) changes MATH-500 accuracy and token-limit rate under the authors' evaluation settings | Primary (D1) | Qualified by the checkpoint substitution in §2 and the required limitations in §0 |
| Effect of model class and loading path on identical weights (text-only reconstruction vs multimodal) | Proposed arm only | Pending budget and a separate protocol decision; it narrows the gap but cannot remove limitation 1 |
| Independent reproduction of the controller *discovering* the edit | No | The agent harness (sandbox, forward-pass server) is not released |
| GSM8K / IFEval / other benchmark results | Secondary, separate protocols | Different sample counts. IFEval graders are not released (§7) |
| Any cybersecurity, safety or production claim | No | The package's security gate is separate |

## 2. Evaluated checkpoint vs published checkpoint

| Property | Authors' evaluated checkpoint | Published checkpoints used here | Status |
|---|---|---|---|
| Base weights source | `Qwen/Qwen3.5-0.8B@2fc06364…`, exported text-only | Same repo and revision, full multimodal release | Revision **established** (equal to the authors' pin) |
| Format | `Qwen3_5ForCausalLM`, `model_type: qwen3_5_text`, single `model.safetensors` (profiles.py `qwen35think`) | `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5`, with `vision_config` | **Different**: the export's language tensors are said to keep the names `model.language_model.*` (apply_edit.py docstring) |
| Export recipe and export weights | Not released | n/a | **Missing** |
| Language-model hyperparameters | export `config.json` | release `text_config` | **Established** equal apart from `architectures`, a duplicated top-level `partial_rotary_factor` and `transformers_version` (5.17.0 in the export) |
| `generation_config.json` | Present (eos 248044, `use_cache`) | **Absent** in both HF repos | Difference. Effect on engine defaults must be checked (§4) |
| Chat template | `chat_template.jinja`, git blob `0ef09f21…` | Same blob, and the same text is embedded in the release `tokenizer_config.json` | **Established** identical |
| `tokenizer_config.json` | 1,124 B, Qwen2Tokenizer, same eos/pad | 16,709 B, Qwen2Tokenizer, same eos/pad, `add_bos_token: false`, `added_tokens_decoder` | Serialization differs. Prompt identity must be shown by rendering (§6.1) |
| `tokenizer.json` | Not in the logs | SHA-256 `5f9e4d49…` in both HF repos | Export's copy **unknown** |
| Edited checkpoint | `build_ckpt.py`: submitted float32 `.npy` cast to bfloat16, replacing tensors in the export; safetensors metadata `{"format": "pt"}` | `RikiyaT/…-weight-space-programming@d1671a17…`, a third-party rebuild (akhaliq / ML Intern, copied from `ML-Intern-lab@f4a5dd43…`) of the multimodal release with `apply_edit.py` | Rebuild **claimed** hash-verified by its author |
| Edit content | 2 tensors (`layers.15` and `layers.19` `mlp.down_proj.weight`, `[1024, 3584]` BF16), 65,260 + 64,928 changed entries, base and edited SHA-256 in `runs/p50c/edit/manifest.json` | Model card lists the same two edited hashes | Manifest and card **agree**. Tensors **not yet verified** |
| File size | n/a | Edited file 432 B larger than base | **Unexplained** until diffed |

### What still requires tensor-level verification (after an approved download)

1. The HF base file's SHA-256 equals the LFS value `04b1c301…`, and the HF edited file's
   equals `fd1e42ac…` (recomputed locally, not taken from the Hub).
2. `wsp.py verify-edit BASE EDITED runs/p50c/edit/manifest.json` returns `verified`:
   - exactly the two listed tensors differ;
   - their shapes, dtypes and changed-entry counts match the manifest;
   - their base hashes equal the manifest's `base_sha256`;
   - their edited hashes equal `edited_sha256`;
   - nothing was added, removed or reshaped.

   This also checks every other tensor, including the vision tower and any MTP weights, for
   bitwise identity.
3. The header/metadata difference behind the 432 bytes is identified and recorded.
4. **Cannot be verified from public material** (required limitation 1): that the export's language tensors equal the
   release's. Only the manifest's base hashes for the two edited tensors connect them. Every
   other tensor of the evaluated checkpoint is unpublished.

## 3. The authors' evaluation, as released

From `exp8_math/README.md`, `code/evaluate.py` (protocol `think35`), `code/profiles.py`,
`code/mathqa.py`, `code/merge.py` and `eval/*/parts/*/scores.json`:

- **Engine:** vLLM. `LLM(model=ckpt, dtype="bfloat16", gpu_memory_utilization=0.85, capped
  by a shared-pool file, max_model_len=34816, seed=0)`. Versions are not recorded in the
  released files.
- **Prompt:** chat template with `enable_thinking=True`. User message =
  `problem + "\n" + "Please reason step by step, and put your final answer within \boxed{}."`.
- **Sampling:** `SamplingParams(n=4, temperature=0.6, top_p=0.95, top_k=20,
  presence_penalty=1.5, max_tokens=32768, seed=0)` for MATH-500. This is one request seed for
  all four samples, and the same seed for every problem. GSM8K and IFEval used `n=2`.
- **Batching:** MATH-500 in 10 shards (`rows[i::10]`). Each shard is one `generate` call over
  its 50 prompts.
- **Data:** `load_dataset("HuggingFaceH4/MATH-500", split="test")` with no revision; IDs are
  `unique_id`. GSM8K is `openai/gsm8k` `main` `test`, also unpinned.
- **Grading:** the last `\boxed{}`/`\fbox{}` after cutting a stray reasoning block. No box
  means incorrect. math-verify 0.9.0 (with latex2sympy2_extended 1.11.0 and
  antlr4-python3-runtime 4.13.2) `verify(parse(gold), parse(pred))`, with 5 s timeouts on
  the main thread. The sympy version is not stated.
- **Score:** mean correctness over all samples (`ok / total`). `cut_share` is the share of
  samples with `finish_reason == "length"`. A job that saw any other finish reason raised and
  was **not saved**; how many times jobs were re-run is not recorded.
- **Kept outputs:** `*_first_samples.jsonl.gz` holds only the first 20 problems of each
  shard (`[:20]` in `evaluate.py`). The README describes it as "every problem".
- **Stale documentation:** the docstrings in `evaluate.py`, `mathqa.py` and `build_ckpt.py`
  still describe the earlier non-thinking Qwen3-1.7B setup. The code paths for `think35` are
  what governs.

## 4. This protocol's arms and settings

- **Arms:**
  - `base`: reference, the HF multimodal release at the pinned revision;
  - `edited`: candidate, the HF edited checkpoint at its pinned revision;
  - `base_reload`: a no-edit reload of `base`.

  The three arms share the inference and engine settings above (the protocol parser enforces
  this).
- **Sampling mode:** `n_per_request` with request seed `0` and `n = 4`, matching the authors.
  The harness's per-sample-seed mode would be a deviation.
- **Batching:** the same 10-shard scheme. Batch composition can change sampled tokens on GPU,
  so it is part of the configuration.
- **Generation-config defaults:** the release has no `generation_config.json`, while the
  export had one. The adapter must record the effective sampling parameters for every request
  and pin how engine defaults are sourced (e.g. vLLM's `generation_config` option). Recording
  the parameters is required; choosing the vLLM option is an open decision.
- **Missing outcomes:** `missing_policy: count_as_incorrect`, which matches the authors'
  score. Harness errors are reported per arm and never dropped. An authors-style abort with
  re-run is replaced by recording; any operator re-run is logged as a deviation with both
  attempts kept.
- **Analysis:**
  - paired item-level difference (edited − base) with an item-clustered bootstrap
    (10,000 resamples, proposed);
  - `min_meaningful_effect` is **not set** (`null`), so the harness reports
    `descriptive_only_min_effect_not_set`;
  - a replication-agreement criterion against the published values (7.9% → 48.1%,
    93.0% → 33.1%) is a separate open decision (§8).

## 5. Sources of variability (why exact outputs are not the target)

At temperature 0.6 with a fixed request seed, the sampler is deterministic *given identical
logits*. Logits are not guaranteed bitwise identical across:
- engine and kernel versions;
- GPU type and tensor-parallel layout;
- batch composition (shards, scheduler order, prefix caching);
- reduced-precision accumulation order.

One differing token early in a 32,768-token generation can change the rest of the output. A
reload on the same machine may or may not reproduce bitwise. A different machine generally
will not. The third-party re-run (51.0% / 31.6%) differs from the post (48.1% / 33.1%). That
illustrates the size of between-setup variability, but it is not a Tantular measurement.

## 6. Reload / drift policy

Identity and reproducibility are separate questions. They are reported separately and never
combined into one score. Implemented in `tools/wsp/wsp_eval/drift.py`.

### 6.1 Identity (exact; no tolerance)

Each check is `pass`, `fail` or `not_recorded`. `not_recorded` is not a pass: identity is
only `complete` when everything was recorded.

| Check | Source |
|---|---|
| Declared checkpoint, inference settings and engine settings equal | protocol |
| Loaded provider, model, revision, engine, engine version and **artifact SHA-256s** equal | adapter `describe()` |
| Reload is a separate model load (distinct `load_id`) | adapter `describe()` |
| Rendered prompt SHA-256 equal for every (item, sample), including error rows from abnormal or inconsistent finishes, which keep the hash the adapter reported; rows without a hash (adapter exceptions) are `not_recorded` | adapter, per generation |
| Edit verified against the manifest (`verify-edit`) for the candidate | checkpoint diff, before any run |

An identity `fail` means output differences cannot be attributed to sampling variability.
The verdict is then `identity_failed_reproducibility_not_interpreted`.

### 6.2 Output reproducibility (stochastic; descriptive unless a tolerance is approved)

Reported for each reload arm against the reference, as reload minus reference:
- **Pair categories:** each (item, sample) is classified as identical output, different
  output with the same grade, grade changed, error on one side, both error, or unpaired.
- **Exact output match rate:** reported, **never required**.
- **Accuracy, token-limit rate and error rate differences:** item-level, each with a 95%
  item-clustered bootstrap interval, plus excluded items and items with missing samples.
  As in the paired analysis, a duplicate (arm, item, sample) record raises, and an item whose
  two arms have different sample sets is excluded from these metrics and listed in
  `items_unbalanced` (its unmatched samples count as `unpaired`).
- **Errors:** by arm and class. Adapter exceptions, abnormal finishes (`AbnormalFinish`),
  wrong sample counts (`SampleCountMismatch`), non-`Generation` returns
  (`InvalidGeneration`), token-limit flags that contradict the finish reason
  (`InconsistentFinish`) and grader crashes (`GraderError:<type>`, output hash kept) are
  rows, never dropped. A run the harness itself could not finish is written with
  `run_status: "aborted"` and is refused by `analyze` and `drift`.

### 6.3 Tolerance: not approved

`drift.tolerance.status` is `unapproved`, and the bound fields are null. The parser rejects
an unapproved tolerance that carries numbers or approval fields, so a proposal cannot look
approved. A tolerance is only used when all of the following hold:
1. the protocol is frozen, with `frozen_by` and a `frozen_at` no later than the run's
   start;
2. `status: approved` with all three bounds in [0, 1], `approved_by` (a pseudonym) and
   `approved_at`;
3. `approved_at` is no later than the run's recorded start (the harness checks this).

If any condition fails, the verdict is `descriptive_only_tolerance_not_approved` or
`tolerance_not_applicable`. When a tolerance applies, the rule is equivalence-style:
- `within`: the whole 95% interval lies inside ±bound;
- `exceeds`: the point estimate lies outside ±bound;
- `inconclusive`: otherwise.

`within_tolerance` additionally needs complete identity. The bounds themselves are a
decision for the independent evaluator. They may be informed by a non-confirmatory pilot on
non-MATH-500 items, but never by confirmatory results.

## 7. Missing details and their effect on the claim

| Missing or unverified | Effect | Mitigation in this protocol |
|---|---|---|
| Text-only export weights and recipe | The evaluated checkpoint cannot be run. The claim must say "published multimodal checkpoint substituted for the authors' text-only export" | Tensor verification of the published files (§2). State the substitution in every result |
| `generation_config.json` absent in the release | Engine defaults may differ from the authors' run | Record the effective sampling parameters per request; decide the vLLM option |
| Export `tokenizer.json` | Tokenization identity with the authors' run is unproven | Release files are hash-identical across both HF arms. Rendered-prompt hashes are recorded |
| Dataset revisions (MATH-500, GSM8K) | The item set and content may have changed since the authors' download | Pin a revision and a content hash. Compare the 500 `unique_id`s with the IDs in the authors' `per_problem.json` files (small, public). A mismatch blocks the claim |
| vLLM / transformers / torch / CUDA versions, GPU, tensor parallelism | Exact output reproduction is impossible to target. Only statistical replication is meaningful | Record the actual versions. Treat the version difference as a declared deviation |
| sympy version | Grading of edge cases may differ | Pin it. Re-grade a human-reviewed sample |
| Grader adapter not yet written or reviewed (`mathqa.py` is untrusted until reviewed) | Grading fidelity is unknown | Review, then reimplement or vendor it under MIT with attribution. Validate on the authors' `pred_first` fields where available |
| IFEval checkers and multiple-choice graders (`weightenv…ifevalg`, `mcqa`) not released | IFEval, MMLU-Pro and GPQA results cannot be replicated faithfully | Out of primary scope. Separate protocols only if official checkers are pinned |
| Re-run/abort history of the authors' jobs | Unknown whether any outcomes were replaced | Report as a limitation |
| Agent harness | No reproduction of the discovery | Out of scope (§1) |

## 8. Differences that prevent a faithful replication claim

The result must be reported as **not a faithful replication** if any of these occur:
- `verify-edit` fails, or local file hashes differ from the pins.
- Any reference-vs-reload identity check fails, or the rendered prompts differ between arms.
- Different sampling settings, thinking mode, prompt text, chat template, token limit, sample
  count or request-seed scheme are used without being declared as a separate alternative arm.
- A different base revision, a quantized or converted model (GGUF/Ollama), or a different
  grader rule is used.
- The MATH-500 item set or answers differ from the authors' 500 IDs.
- Errors, truncations or ungraded outputs are dropped instead of scored and reported.

Even with none of these, the strongest available claim is qualified: the published
multimodal checkpoints, under the authors' released settings, on a different engine version
and hardware. The text-only substitution and the unknown engine versions stay stated as
limitations.

## 9. Remaining decisions, in the order D1 requires

Settled: claim and scope (D1). GSM8K and IFEval get their own protocols (D1); IFEval needs
the official checkers.

1. **Resources and approvals:**
   - roles, including who approves the run;
   - the budget, including whether the proposed text-only reconstruction arm is funded;
   - isolated infrastructure;
   - permission to download the two checkpoints and MATH-500;
   - the human license review: logs (CC BY 4.0 / MIT parts), Qwen Apache-2.0, dataset
     licenses.
2. **Run choices (before any bounds are approved):**
   - the vLLM version, generation-config handling, hardware and tensor-parallel layout;
   - the MATH-500 revision and content hash, and the treatment of any mismatch against the
     authors' 500 IDs;
   - the grader adapter: review of `mathqa.py`, the sympy pin and a human grading sample.
3. **Decision rules (before confirmatory results):**
   - `min_meaningful_effect`;
   - `agreement_with_published`;
   - the three drift bounds.

   A pilot on items other than MATH-500 may inform the bounds. Its use and results are
   recorded, and it never sets them automatically.
4. **Review and freeze** (`frozen: true`, `frozen_by`, `frozen_at`; the parser requires
   them, and the analysis and drift rules are binding only for runs that start after
   `frozen_at`), then the approved downloads, then the file, tensor, tokenizer and
   prompt checks (§2, §6.1), before any experiment run. The full list of decisions,
   infrastructure and harness work still needed is in [run-readiness.md](run-readiness.md).
