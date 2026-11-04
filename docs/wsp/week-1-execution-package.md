# Weight-Space Programming: Week 1 execution package

Status: proposed, not approved or executed · Prepared 2026-11-03 · Week 1: 2026-11-09–13

This package starts an isolated research track. It does not authorize production changes,
external model uploads, spending, or an autonomous checkpoint promotion. No experiment,
resource availability, baseline, or effect has been verified in this repository yet.
The project owner assigns the roles and approves resource limits before paid work starts.

## Decision and scope

Week 1 establishes the provenance and evaluation design for Experiment A, plus a
feasibility inventory for a later security defect. Deliverables are a pinned source and
artifact ledger, an environment and cost estimate, a preregistered evaluation protocol,
a small harness smoke test, and a Week 2 readiness decision. No novel candidate edit is
required. Replicating the published checkpoint's behavior is distinct from independently
reproducing the controller's discovery of the edit; report the two separately.

The existing application is a local, read-only repository security analyzer; its optional
Ollama explanations are advisory. `docs/evaluation.md` describes the existing opt-in
comparison workflow, pinned options, paired analysis, provenance limitations and review
rubric. Reuse its findings and methodology where appropriate, but do not equate its
quantized Ollama models with the BF16 research checkpoint. Confirm the actual origin,
license, revision, format and accessibility of the proposed Cyber 9B and Antares-1B
checkpoints before defining Experiments B or C. The existing Tantular Ollama profile is
not evidence that a matching editable checkpoint is available.

## Research reference and replication target

The authors' [post](https://rikiyat.github.io/blog/weight-space-programming/) and
[experiment logs](https://github.com/RikiyaT/lm_understand-logs) are the primary sources;
their [edited checkpoint](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming)
is the primary published artifact. Pin the full Git commit and Hugging Face revisions,
record file SHA-256 hashes and licenses, and check the released instructions and exact
base model revision before downloading or executing anything. Treat released scripts as
untrusted code: review them and run in the isolated environment. Record unresolved
missing dependencies or incompatibilities, not invented replacements.

Experiment 8 reports Qwen3.5-0.8B with thinking enabled, temperature 0.6, top-p 0.95,
top-k 20, presence penalty 1.5 and a 32,768-token limit. MATH-500 used four samples per
problem; GSM8K and IFEval used two. The authors report MATH-500 7.9% to 48.1% and
limit hits 93.0% to 33.1% for the 50-forward-pass edit. Those are reference claims,
not Tantular results. The experiment identifies a loop-related failure; the authors'
Gemma control did not establish a general ability to create new math skills. The
[logs](https://github.com/RikiyaT/lm_understand-logs) govern exact prompts, dataset
versions, grading and inference details when the summary is insufficient.

## Roles and isolation

- **Research lead:** owns protocol, defect definition and budget; cannot approve their own
  production promotion as sole reviewer.
- **Operator:** provisions a separate research workspace, records versions and runs the
  approved commands. Model and dataset downloads are explicit and license-reviewed.
- **Independent evaluator:** controls held-out data, validates scoring and reviews results.
- **Human release owner:** records accept, reject or request-more-evidence decisions;
  research acceptance does not itself deploy a model.

Enforce separation with OS/cloud identities and storage permissions, not just prompts:
production registry and credentials unavailable to the controller and sandbox; baseline
artifacts read-only; candidate outputs in a separate research-only location; no network
or secrets in the edit sandbox unless specifically required and approved. The controller
proposes code or patches; a human reviews them before an operator executes them. Run
held-out evaluation under a separate identity without exposing test content or feedback
to the controller. Never edit deployed GGUF files. A later deployment would require
separate evaluation of the generated quantized copy and an explicit human promotion.
These are design requirements, not controls already implemented by this repository.

## Daily work plan

| Day | Work | Evidence / exit condition |
|---|---|---|
| Mon 9 | Assign roles; inventory checkpoint access, licenses, hardware, disk and isolated identity; set spending ceiling. | Owner, access matrix, budget and blockers recorded; no downloads until approved. |
| Tue 10 | Inspect released Experiment 8 logs, code, patch and evaluator; pin full revisions and enumerate artifacts. | Source ledger includes exact hashes, licenses, acquisition path and known gaps. |
| Wed 11 | Specify dataset splits, inference and scoring; review threats to validity and resource estimate. | Frozen protocol reviewed by independent evaluator; test access restricted. |
| Thu 12 | If resources and controls are approved, provision research environment and run a tiny inference and scoring smoke test on non-held-out examples. | Versioned environment, input/output hashes, logs, costs and failures; no performance claim. |
| Fri 13 | Review smoke results and preregistration; choose Week 2 go, conditional go or stop. | Signed decision, owners, remaining blockers and Week 2 run budget. |

If infrastructure is unavailable, complete the document review and protocol; mark smoke
and readiness blocked. Do not substitute a different model or decoding setup and label it
a reproduction. Dates are indicative and can move with resource availability.

## Evaluation protocol to freeze before full runs

1. **Arms:** unchanged original BF16 checkpoint and authors' published edited checkpoint,
   same exact tokenizer, prompt, model settings, inference engine and dataset. Verify
   tensor-level differences and hashes against the published materials before attributing
   a result to the edit. Independently generated patches are a later, separately logged
   arm and require an explicit development budget. A no-edit reload checks pipeline drift.
2. **Replication:** match the authors' released benchmark and grading procedures, dataset
   versions, thinking mode, token budget, repeated samples and seeds as closely as possible.
   Record all deviations before running. Report accuracy, token-limit hits, errors and
   output length per arm. Validate grading on a human-reviewed sample and record ungraded
   or failed requests; never silently remove them. Token-limit hits are a truncation proxy,
   not proof of a reasoning loop; inspect examples using a separately specified loop rubric.
3. **Alternatives:** compare a development-selected prompt/decoding configuration on the
   unchanged checkpoint and a separately trained LoRA baseline with disclosed data,
   training budget and inference settings. These are separate interventions, not identical
   decoding comparisons. Use the same evaluation examples and account for all tuning and
   operational cost. Their inclusion is required for choosing a preferred method, not for
   claiming faithful replication of the authors' fixed-settings comparison.
4. **Data discipline:** use disjoint development and held-out evaluation sets, with stable
   IDs and content hashes; deduplicate and document known overlap or contamination. Freeze
   the protocol and candidate before independent evaluation. Public benchmarks are useful
   for replication but cannot be assumed unseen by the controller. Security holdouts must
   be separately controlled, derived from authorized material and kept away from it.
5. **Analysis:** define primary endpoint and minimum meaningful effect before collecting
   confirmatory results. Use paired per-problem differences with 95% confidence intervals,
   clustering all repeated samples by problem; report number of problems, samples, missing
   outcomes and seeds. Use a paired bootstrap at the problem level for repeated samples;
   exact McNemar is appropriate for a single binary result per paired item. Predeclare
   multiplicity handling for multiple primary claims. A pilot estimates variance and cost;
   select a sample size for the desired power and effect before unblinding the holdout.
   If precision or power is inadequate, report inconclusive rather than success.
6. **Security and operations:** define severity, adjudication and test coverage before
   confirmation. Include defensive triage, CWE classification, remediation, tool-call
   accuracy, prompt injection and authorization boundaries where applicable. Record
   latency, token consumption, peak memory, compute, API cost and failure rates. An
   untested category is unknown, not a pass; zero observed regressions is not proof of
   zero risk. Offensive cases require an explicitly authorized lab.

Proposed later promotion gate, subject to human preregistration: a positive lower 95% CI
for the prespecified paired improvement and an effect at least as large as the chosen
minimum; no unresolved critical security regression; all required safety categories
executed and adjudicated; provenance and rollback verified; and a documented comparison
with prompt/decoding and LoRA alternatives. Quantitative non-inferiority margins and
sample sizes must be agreed before the confirmatory run, not selected after seeing it.
A reproduced public benchmark effect alone does not meet a cybersecurity release gate.

## Evidence ledger template

Create one immutable record per run in research storage, with a small sanitized manifest
in the repository only when it contains no private data or secrets:

```yaml
run_id: REPLACE
status: planned # planned | complete | failed | invalid
purpose: replication-smoke
operator: REPLACE
source_log_commit: REPLACE_FULL_COMMIT
base_model_revision: REPLACE_FULL_REVISION
edited_model_revision: REPLACE_FULL_REVISION
artifact_sha256: {} # file path -> digest, including tokenizer, config and patch
licenses_reviewed: false
environment: {container_digest: REPLACE, hardware: REPLACE, runtime_versions: REPLACE}
dataset: {name: REPLACE, revision: REPLACE, split: REPLACE, content_hash: REPLACE}
protocol_version: REPLACE
inference: {thinking: true, temperature: 0.6, top_p: 0.95, top_k: 20, presence_penalty: 1.5, max_new_tokens: 32768, seeds: []}
grader_revision: REPLACE
output_location_and_hash: REPLACE
cost_and_compute: REPLACE
deviations_and_errors: []
human_decision: pending
```

Record checkpoint diff metadata: base and candidate hashes, tensor names/shapes/dtypes,
changed-element count, magnitudes, patch code and hash, controller provider/model/version,
prompt and tool history, and sandbox image. Keep sensitive logs and raw responses in
access-controlled storage; never commit model weights, proprietary data, secrets or
unreviewed generated code. A manifest is evidence of intent, not an enforcement boundary.

## Week 1 readiness review

- [ ] Research lead, independent evaluator, operator and release owner assigned.
- [ ] Resource and API spending caps approved; GPU, disk and runtime feasibility measured.
- [ ] Isolation and read-only baseline access verified by operator; production credentials absent.
- [ ] Full source, model and dataset revisions, hashes and licenses recorded.
- [ ] Evaluator confirmed benchmark availability, graders and leakage controls.
- [ ] Protocol, sampling plan, stop criteria and deviations policy reviewed and frozen.
- [ ] Smoke test completed or explicitly blocked, with reproducible evidence and costs.
- [ ] Human decision recorded: go / conditional go with owner and deadline / stop.

Week 2 should implement or adapt a research-only harness only after the evaluation
contract is frozen. Its minimum acceptance is repeatable input and configuration recording,
paired arm execution, explicit failures, reproducible grading and export of per-item data;
exercise it with synthetic fixtures before expensive runs. Do not add a production
promotion path or assume an AI-provider integration is available. If controller research
is later authorized, use a provider-neutral interface and record the actual provider,
model, version and cost; GPT-6 Astra and GLM-5.3 availability remains unverified.
