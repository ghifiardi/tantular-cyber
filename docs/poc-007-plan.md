# POC-007 plan — evidence-grounding reliability and reproducible model evaluation

Status: **planning document, not implementation.** Tracking issue: GitHub #11. No step below
has started. Each step is one pull request, reviewed and merged before the next begins.
Nothing here has a date or a numerical improvement target.

## 1. Objective and boundaries

Make the optional, advisory `--ai` explanations more reliable and their evaluation
reproducible and auditable — without changing deterministic scanner findings, severity,
coverage, report status, privacy defaults, the opt-in nature of `--ai`, or the refusal of
remote/cloud Ollama models.

Out of scope for POC-007: new CWEs, BAS/SOC integrations, fine-tuning or claims about it,
automatic deployment, network access beyond the configured loopback Ollama, automatic
downloads, telemetry. Model output stays advisory (`verified: false`), sanitized, and unable
to change scanner facts or scan status (CLAUDE.md, rule 5).

## 2. Why the current evaluation is not enough

Findings from reviewing the v0.3.0 code and the POC-005 run artifacts kept locally under
`out/` (git-ignored, never published):

1. **Ollama does not say which weights served a response.** `/api/chat` returns no digest.
   `compare` captures model provenance before and after a whole comparison
   (`benchmark/compare.py`, `model_provenance`, `provenance_drift`), not per response.
2. **The recorded code commit can be wrong.** `run_comparison` calls `code_state()` when the
   run *finishes* and discards the dirty flag (`commit, _ = code_state()`); the checkpoint
   header does not include the commit, so a resumed run can span code versions.
   Consequence for the POC-005 runs:

   | Run (`out/…`) | Started (WIB) | Recorded commit | Code that actually ran |
   |---|---|---|---|
   | `models` — the published 57% | 2026-10-29 10:23 | `76a9318` (committed 10:59) | Uncommitted working tree; the run began before the first POC-005 commit (`c8bbcc0`, 10:38). Not identifiable. |
   | `optim` — prompt/context arms, n = 80 per arm | 14:29 | `76a9318` | Most likely `76a9318`; dirty state unknown. |
   | `sameweights` — profile mode, n = 120 per arm | 16:45 | `c889c32` (committed 19:32) | `76a9318` plus the uncommitted drift check (as `docs/benchmark-results.md` states). |

   `76a9318` and `c889c32` exist only on branch `feat/poc-005-accuracy-evaluation`; `main`
   has the squash-merged `e820a98`.
3. **The published 57% cannot be reproduced from retained artifacts.** `out/models` holds
   only the summary (`comparison.json`, `comparison.md`), not per-sample outputs. `optim` and
   `sameweights` retain per-sample validated analyses (`samples.jsonl`) but, by design, no
   prompts.
4. **Human review is not blind.** `review_queue` (`benchmark/llm_eval.py`) exposes the
   ground-truth `label`, every automated check result and `validation_warnings`, and the
   `review_id` encodes the run.
5. **`evidence_grounded` merges distinct failures.** It fails when a cited identifier is not
   in the evidence/context *or* a cited line number is outside the shown range, and is "not
   assessable" when nothing is cited (`check_grounding`). The 57% does not say which.

## 3. Three kinds of result, never one trend

Every POC-007 report labels each number as exactly one of these, and never plots or
tabulates them as a single series:

| Kind | Meaning | Use |
|---|---|---|
| **Historical published measurement** | 57% `evidence_grounded` for `tantular-cyber:latest` (2026-10-29). | Quoted as published, with the statement that it is not reproducible from retained artifacts. |
| **Offline exploratory re-scoring** | Rubric re-applied to retained outputs (Step 0), evidence reconstructed from historical code. | Diagnosis only. Never described as reproducing the published benchmark. |
| **New controlled baseline** | Measured under Steps 1–4 controls on the frozen held-out set. | The only reference for Step 5 comparisons. |

Keeping the `evidence_grounded` definition unchanged (Step 2) gives *metric-definition*
compatibility only; it does not make a new run comparable with the historical 57%.

## 4. Model provenance policy

Per request: snapshot `/api/tags` (manifest digest, quantization) and `/api/ps` (loaded model
digest) immediately before and after the `/api/chat` call. During planning, a read-only
`/api/ps` request to the local Ollama 0.40.2 returned the loaded model's full digest; Step 1
confirms this under test before relying on it.

| Situation | Treatment |
|---|---|
| Both snapshots match and every required provenance field is present | Response is **eligible** for evaluation. |
| Snapshots differ, or any required field is missing | Response is **excluded** from comparable results. It may be kept, labelled, for exploratory analysis. |
| Any response is excluded during a controlled benchmark | The run is marked **incomplete**. Comparable aggregate metrics are not computed for it; the report lists the exclusions. A controlled comparison requires a complete, valid paired sample set. |

Stated limitations: this is a client-side control. It cannot prove which weights served the
request, and equal snapshots cannot detect a model that changed and changed back during a
request. Ollama's own server log remains the only stronger evidence and is not parsed
automatically (it is outside this tool's files and may contain prompts).

## 5. Development and held-out split

Before Step 4, the corpus findings used for LLM evaluation are split into a **development**
set (prompt/context iteration, Step 5 exploration) and a **held-out** set (Step 4 baseline,
Step 5 final comparison only).

- Split by **case file**, not by finding, so near-duplicate code in one file cannot sit on
  both sides; stratify by CWE and by ground-truth label.
- Membership is frozen in a versioned split manifest; the SHA-256 of each side's sorted
  membership and of the manifest are recorded in every run.
- A run that touches held-out findings during development, or whose split hashes differ from
  the frozen ones, is refused.
- The held-out side is small (the v0.3.0 comparison uses 40 findings in total). Intervals
  will be wide; the plan accepts that rather than reusing development samples. Adding cases
  within the five existing CWEs to enlarge it is an open decision (§ 9).

## 6. Statistical controls (apply to Steps 4–6)

- Repeated seeds on the same finding are not independent. Primary analysis aggregates per
  finding (or uses a finding-clustered method); per-sample McNemar is reported only as
  secondary and labelled as such.
- Wilson 95% intervals for rates; exact McNemar on paired, per-finding outcomes.
- **Abstention and "not assessable" counts are reported next to every rate.** `summarize`
  leaves "not assessable" results out of the denominator, so a change that cites less raises
  the `evidence_grounded` rate without improving anything; it is not an improvement.
- Comparison validity is a hard gate: same corpus hash, split hashes, code commit with clean
  tree, model digests, pinned options, Ollama version. Non-comparable runs are flagged and
  excluded from comparisons, not averaged in.
- "No significant difference" is reported as such. No precommitted target.
- Execution conditions are recorded: machine sleep prevented (`caffeinate -i`), power
  source, concurrent load windows, models warmed once before a run so Ollama's GGUF
  migration happens outside the measured arm.

## 7. Steps

### Step 0 — Offline forensic re-scoring (WP 3, exploratory)

Re-apply the rubric to the retained `optim` and `sameweights` outputs and attribute
`evidence_grounded` failures, without any new model run.

- **Method:** check out `76a9318` in a separate worktree; re-scan the bundled corpus there to
  regenerate each finding's evidence and context; confirm the regenerated finding ids equal
  the ids in each `samples.jsonl` header; re-run `check_grounding` on the saved analyses and
  split failures into *identifier not in evidence*, *line reference out of range*, *both*,
  *nothing cited*.
- **Assumptions to state in the output:** the evidence/context produced by `76a9318` equals
  what the model saw (not verifiable: prompts were not saved); `sameweights` ran on
  `76a9318` plus an uncommitted change limited to provenance reporting; dirty state of
  `optim` unknown.
- **Missing artifacts to state:** per-sample outputs for the published `models` run; all
  prompts; per-response model digests.
- **Dependencies:** this plan merged; the historical commits preserved (§ 9).
- **Controls:** read-only use of `out/`; no model calls; findings whose ids do not match are
  excluded and counted.
- **Privacy:** corpus is synthetic; outputs stay local. The published note contains
  aggregate counts only, no model text.
- **Acceptance:** a committed script/test-backed command producing the attribution table; a
  short note in `docs/` labelled *exploratory re-scoring — not a reproduction of the
  published benchmark*, listing assumptions and missing artifacts.
- **Limitations:** n = 80 baseline-arm samples (`optim`) and profile-mode samples
  (`sameweights`); neither is the published run.

### Step 1 — Frozen evaluation corpus and provenance (WP 1)

- **Deliverables:** a versioned evaluation-run schema (`evaluation-run` 1.0.0) recording, per
  run: corpus and manifest hashes, finding ids, evidence/context SHA-256 per finding, prompt
  template and system-prompt hashes, rendered-prompt hash per request, pinned options,
  Ollama version, code commit **and dirty flag captured at start and at end** (a mismatch
  invalidates the run), environment; per response: the § 4 snapshots, response hash,
  timestamps, token and latency metrics, eligibility.
- Checkpoint headers include the code commit, dirty flag and split hashes; resuming across
  any change is refused.
- `compare` and `benchmark --llm` write the new record; existing outputs stay readable
  (additive schema change, versioned).
- **Dependencies:** Step 0 merged (its findings may refine the field list).
- **Controls / fail-closed:** missing or changed provenance → response excluded, run marked
  incomplete per § 4; non-zero exit for a controlled run that is incomplete.
- **Privacy:** rendered prompt *text* is stored only when evaluating the bundled synthetic
  corpus and only with an explicit flag; for any other repository only hashes are stored.
  Nothing is logged (rule 4). No network beyond loopback Ollama; cloud models still refused.
- **Acceptance:** tests for provenance drift mid-run, missing digest, `/api/ps` unavailable,
  dirty-tree start, resume across commits, cloud model in list — all fail closed; a
  documented reproducible local command.
- **Limitations:** § 4 limits; reproducibility is per machine and Ollama version.

### Step 2 — Evidence-failure taxonomy and regression fixtures (WP 3)

- **Deliverables:** sub-checks reported alongside the unchanged composite
  `evidence_grounded`: `identifiers_grounded`, `line_refs_in_range`, `cites_evidence`.
  Deterministic attribution where the data allows it: identifier present in the file but
  not in the shown context → *context selection*; absent from the file → *generation
  (fabrication)*; scanner evidence lines wrong → *scanner evidence*; the remaining classes
  (*prompt/profile*, *rubric mismatch*) are assigned only by human review.
- Replay fixtures (no model) for fabricated citations, wrong line numbers, empty and partial
  evidence, overclaim, prompt injection.
- **Dependencies:** Step 1 merged.
- **Controls:** composite definition byte-for-byte unchanged (test); sub-checks are additive.
- **Privacy:** fixtures synthetic; no repository content.
- **Acceptance:** each attribution class has a passing and a failing fixture; CI runs them;
  metric-compatibility note per § 3.
- **Limitations:** lexical attribution cannot judge whether a grounded citation actually
  supports the conclusion; that stays with human review.

### Step 3 — Blinded human-review protocol (WP 2)

- **Deliverables:** a blinded export: no model or arm identity, no ground-truth label, no
  automated check results or warnings; randomized order; opaque item ids; a separate sealed
  key file mapping ids back, stored outside the reviewer bundle and hashed into the run.
- Rubric v2 per explanation: evidence support **per claim** (supported / partly / not
  supported / cannot judge), remediation soundness (sound / partly / unsound / cannot
  judge), overclaim present (yes / no), clarity in Bahasa Indonesia (1–5), notes.
- Roles: **Reviewer A — TBD, Reviewer B — TBD, Adjudicator — TBD.** Reviewers understand
  Indonesian and application security, rate the same items independently, and see only
  evidence, context and explanation. The adjudicator resolves disagreements only after both
  independent ratings are submitted.
- Agreement: Cohen's kappa for categorical dimensions (quadratic-weighted for the ordinal
  clarity scale), raw agreement and within-±1 agreement alongside; per-dimension n.
- Missing and ambiguous ratings: "cannot judge" is a recorded value, not missing; blank
  fields are missing, excluded per dimension and counted, never imputed; an item with any
  missing primary rating cannot enter consensus for that dimension.
- **Dependencies:** Step 2 merged (rubric dimensions map to the taxonomy).
- **Privacy:** pseudonymous reviewers (existing pattern); no personal data fields; notes not
  published.
- **Acceptance:** tests that no blinded field leaks into the export; summaries clearly state
  **protocol ready, no reviews completed** until real ratings exist.
- **Limitations:** no human validation can be claimed until named reviewers complete it.

### Step 4 — Controlled baseline (WP 4)

- **Deliverables:** the split manifest (§ 5) frozen and hashed; a baseline run of
  `tantular-cyber:latest` (pinned mode) on the held-out set under §§ 4–6, n ≥ 120 responses.
- **Dependencies:** Steps 1–3 merged; split approved.
- **Controls:** all § 6 validity gates; warm-up before the arm; complete paired sample set
  or the run is incomplete.
- **Privacy:** synthetic corpus only.
- **Acceptance:** a complete, valid run reproducible by the documented command on the same
  machine and model digest; report labels it *new controlled baseline* (§ 3).
- **Limitations:** single machine and model build; quantization fixed (Q8_0).

### Step 5 — Paired improvement experiments (WP 4, WP 5)

- **Candidates (development set only while iterating):** evidence-first prompting (quote,
  then explain); line numbers rendered in the context block; a stricter citation format;
  an explicit *insufficient evidence* label (conservative abstention).
- **Final comparison:** each frozen candidate once against the Step 4 baseline on the
  held-out set, paired, under the existing adoption rule (`docs/evaluation.md`), with
  abstention counted per § 6.
- **Dependencies:** Step 4 merged.
- **Controls:** no held-out access during iteration (enforced by split hashes); candidates
  frozen before the held-out run; safety checks (`schema_valid`, `no_unsupported_assertions`,
  `injection_resisted`) may not lose any paired sample net.
- **Privacy / safety:** output remains advisory, sanitized, `verified: false`; overclaim
  guard and CVE stripping unchanged.
- **Acceptance:** adoption only if the rule passes; otherwise the result is reported as
  negative and the default stays.
- **Limitations:** improvements on a small synthetic held-out set may not transfer.

### Step 6 — Research report (WP 6)

- **Deliverables:** a dated evaluation report in `docs/` and `docs/benchmark-results.md`
  covering all three result kinds (§ 3) separately, negative findings, exclusions and
  incomplete runs, limitations, and the regression tests added.
- **Dependencies:** Step 5 merged (or stopped, with the reason).
- **Acceptance:** every number traceable to a run record; human-review status stated
  exactly (protocol ready vs reviews completed).

## 8. Regression guarantees for every step

No change to: deterministic scanner output on the benchmark corpus (`make bench` gates),
privacy defaults, opt-in `--ai`, remote/cloud Ollama refusal, `report.json` schema without a
version bump. CI must pass; `make lint` clean.

## 9. Open decisions

1. **Preserve historical commits.** Proposal: push annotated tags
   `eval/poc-005-76a9318` and `eval/poc-005-c889c32` so Step 0 does not depend on a feature
   branch surviving. Needs approval (pushes refs to the public repository).
2. **Held-out size.** Whether to add cases within the five existing CWEs before Step 4.
3. **Reviewers.** Reviewer A, Reviewer B and Adjudicator remain TBD.
