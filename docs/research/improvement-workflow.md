# Research improvement workflow (`tools/research`)

Status 2026-11-04: **synthetic-only framework.** No model, SIEM, alert stream, dataset,
checkpoint or provider has been accessed. No baseline, triage result or model improvement has
been measured. Every report produced so far is labelled
`synthetic_fixture_not_a_measurement` and says it is not a Tantular Cyber baseline. Every
limit in the fixture study is an invented value, labelled `synthetic_fixture` in the study
and in the report.

This is a small, provider-neutral workflow:
- It registers an unchanged baseline, one target failure and candidate interventions.
- It compares each candidate with the baseline under declared constraints.
- It writes a human-readable report that stays **inconclusive** whenever evidence,
  constraints or preregistration are missing.
- It accepts a separate human decision record.

It has no promotion or deployment path. Like the WSP harness, it is research-only:
- standard library only;
- never imports `tantular_cyber`;
- not in the wheel (the sdist includes `tools/` and `docs/`);
- no network or subprocess calls.

A test enforces these properties. The intent is a trustworthy framework that can later
accept real decisions about the data source, label provenance, calibration split and
operating limits. A real triage adapter waits for those decisions.

## Background: the constraint-aware SOC triage paper

This summary is background only, from the research owner. The paper compares PPO, Isolation
Forest and an autoencoder for SOC alert triage. It emphasizes benign-escalation constraints,
threat coverage, cluster-level evaluation and cross-tenant limitations. Those themes shaped
four parts of this framework:
- the benign-escalation and defer caps a frozen triage study must declare;
- the threat-miss metric;
- the optional cluster unit of analysis;
- the required validity scope.

**The paper's data, code, models, feature mappings, thresholds and results are not
integrated, reproduced or validated here.** The PDF was not read (reads outside the
repository are not authorized). An earlier session looked at a local QRadar app skeleton
before that restriction was noticed. Nothing from it was copied, and it is not a source for
any value here.

## What was reused

| Source | Reused as |
|---|---|
| WSP harness (`tools/wsp`, PR #23) | Canonical hashing, immutable output dirs outside the repo, aborted-run handling, `verify_run`, freeze-before-run preregistration, conservative verdicts. **A verified WSP run is the evidence path for weight-edit candidates.** Its required limitations and preregistration status are copied into the report |
| `docs/evaluation.md` adoption rule | "No safety check loses any paired sample net" becomes the `max_net_paired_losses` constraint |
| `review.py` / WSP ledger rules | Pseudonymous deciders; no self-approval by the study owner or operator |
| POC-007 plan | Development/held-out discipline and reporting negative results. POC-007 puts SOC integrations out of scope, so this stays outside the product |

## Intervention taxonomy (one type per candidate; never mixed)

| Family | Types | Enforced before comparison |
|---|---|---|
| configuration | `prompt_or_threshold`, `feature_change` | Same artifact digest as the baseline; the configuration actually differs; `feature_change` names a feature-contract hash |
| training | `conventional_training`, `lora` | A different artifact digest; training data, training-config hash and compute budget declared; `lora`'s `base_artifact_sha256` is the registered baseline digest |
| soc_triage_model | `soc_triage_model` | A different artifact; model family, feature contract, training data, and a calibration dataset that is **not** the evaluation dataset |
| weight_edit | `weight_edit` | A different artifact; `base_artifact_sha256` is the baseline digest; edit-manifest hash. Status `verified` plus a `report_sha256` is required in a frozen study, **and** the actual `verify-edit` report must be supplied with the WSP evidence, name the registered edit manifest, and come from a held-out WSP run (below). Until `verify-edit` reports carry `manifest_sha256` (a separate WSP change), weight-edit evidence **loaded through `evidence_from_wsp`** cannot pass. A hand-built `Evidence` object can assert an edit-verification result in its `notes` (see [Direct calls](#direct-calls-and-what-is-recomputed-at-the-comparison-boundary)) |

## Metrics: explicit numerators and denominators

All metrics are ratios of per-item sums (`metrics.py`). The report prints every value as
`value (numerator/denominator)`. For triage it also prints a per-arm decision × label
breakdown (`select`/`defer`/`suppress`/`error` × `threat`/`benign`/`unlabelled`), from which
every triage metric can be rebuilt.

| Metric | Numerator | Denominator | Missing labels | Errors | Deferrals | Empty |
|---|---|---|---|---|---|---|
| `benign_escalation_fraction` | selected ∧ benign | selected ∧ labelled | excluded; shown in the breakdown | not selected, so excluded | not selected | undefined (None), so the constraint is `not_evaluated`, never satisfied |
| `threat_miss_rate` | threat ∧ (suppressed ∨ error) | labelled threat | excluded | **counted as a miss** | not a miss (an analyst sees it) | undefined if no threats |
| `defer_rate` | deferred ∨ **error** | all alerts | included | **counted as deferred** (lands on a queue), so errors cannot make an arm look cheaper | counted | — |
| `escalation_rate` | selected | all alerts | included | not selected | not selected | — |
| `error_rate` | errored samples | samples | — | counted | — | — |
| `accuracy` | correct samples (ungraded and errors count as incorrect) | samples | — | incorrect | — | — |

A frozen triage study must declare three separate absolute `<=` caps (scope `all`):
- `benign_escalation_fraction`: the precision of escalations;
- `defer_rate`: analyst-queue load;
- `escalation_rate`: escalation **volume**, a capacity constraint.

Volume is declared on its own, not inferred from precision or deferral. Otherwise an arm that
escalated ten times as many alerts at the same precision would pass. Without all three caps,
missed threats could be "fixed" by escalating or deferring everything.

Bounds and interval ends are rounded **outward** to six decimals (upper bounds up, lower
bounds down), so rounding cannot move a bound across a limit in the candidate's favour by
more than about 1e-12 (a floating-point tolerance in the rounding).

## Intervals and what the report may claim

The synthetic test suite checks that the code does what is described here. It does not
validate any statistical procedure. The method choices below, especially for clustered data,
need independent statistical review.

**Estimands.** Every interval names the quantity it targets (`estimand`), and its point
estimate is computed for that same quantity:

| Estimand | Used for | Definition |
|---|---|---|
| `pooled_ratio` | absolute constraints | Σ numerators / Σ denominators over units in scope |
| `mean_paired_difference_per_item` | target effect, `non_regression` (pairable metrics) | mean over items of the signed per-item difference (positive = better); for single-sample binary items, equal to the difference of pooled ratios |
| `mean_paired_difference_per_cluster` | the same, when `analysis.unit` is `cluster` | per-cluster means, then averaged across clusters (cluster-weighted). It can differ in size **and sign** from the item-weighted value, which is shown beside it |
| `difference_in_pooled_ratios` | `non_regression` on benign escalation (not pairable) | candidate pooled ratio − baseline pooled ratio |

**Method status.** Only two methods are binding: they alone can make a constraint
`satisfied`, or a target meet or regress.
- `exact`: one-sided Clopper–Pearson on whole counts k of n.
  - **Assumes** n independent Bernoulli trials with a common probability: independent alerts
    or items with item units.
- `valid_by_bonferroni`: the paired interval for **binary, single-sample items with
  denominator 1**. Each item's signed difference is −1, 0 or +1.
  - **Assumes** items are independent and identically distributed. Then W (items improved)
    ~ Binomial(n, p₊) and L (items worsened) ~ Binomial(n, p₋), and their exact one-sided
    Clopper–Pearson bounds hold.
  - Taking four such bounds, each at α/4, gives simultaneous coverage ≥ 1 − α by
    Bonferroni's inequality. That holds whatever the dependence between W and L.
  - So [lo(p₊) − hi(p₋), hi(p₊) − lo(p₋)] covers p₊ − p₋, the mean paired difference, with
    probability ≥ 1 − α. It is conservative, and valid only under those assumptions.

Everything else is reported with its status and is **descriptive only**:
- `unreviewed_heuristic`: Clopper–Pearson on rounded effective-n or fractional counts, such
  as multi-sample items (WSP runs with several samples per item). No coverage guarantee has
  been shown, and these are called neither exact nor conservative.
- `approximate`: the percentile bootstrap, which has no finite-sample guarantee. A zero-width
  bootstrap is reported as not estimable.
- `cluster_method_unreviewed`: any cluster-unit interval. A cluster-unit study's candidates
  all end as `descriptive_only_cluster_method_unreviewed`, with the rule outcome shown beside
  it. This stays until a reviewed clustered method and its matching estimand are specified.

Constraint statuses separate what was observed from what is established:

| Status | Meaning |
|---|---|
| `satisfied` | the claim-side one-sided bound is within the limit (binding method) |
| `violated` | the **opposite** one-sided bound is beyond the limit (binding method): a statistically established violation. For `max_net_paired_losses` it means the declared count was exceeded in these records (an observed-count rule, `basis: observed_count_rule`, with `losses: gross` (the default: every new loss counts) or `net` (the `docs/evaluation.md` adoption rule, where fixed items offset new losses)) |
| `point_beyond_limit` | the observed point is beyond the limit, but a violation is not established. Not satisfied; the outcome is at best inconclusive |
| `inconclusive`, `descriptive_only`, `not_declared`, `not_evaluated` | not satisfied |

Only an established (or observed-count) `violated` makes a candidate `constraint_violated`.
`point_beyond_limit` makes it inconclusive, with that reason.

**Confidence levels.** Constraint bounds are one-sided at `analysis.confidence` (e.g. 95%).
The target's lower bound is one end of a two-sided Bonferroni interval, so it is effectively
one-sided at 1 − α/2 (97.5% when the confidence is 95%), which is more conservative than the
constraints.

**Decision rule.** `meets_declared_criteria_pending_human_decision` needs all of:
1. a preregistered, item-unit study;
2. a binding target interval;
3. a **lower confidence bound** above zero **and at least `min_meaningful_effect`**;
4. every constraint `satisfied`.

The minimum effect is a confidence-bound criterion: the data must rule out improvements
smaller than it, and a point estimate above it is not enough. `regression_on_target` needs a
binding interval whose upper bound is below zero.

**Multiplicity.** There is no multiplicity control across candidates or constraints. Each
is evaluated at `analysis.confidence`, so the chance that some candidate meets the criteria by
chance grows with the number of candidates. Every report says so.

## Fail-closed preconditions

When evidence is incomplete, or a precondition cannot be verified, the comparison stops (or
its confirmatory outcome is inconclusive). It never proceeds on an assumption:

| Situation | Result |
|---|---|
| No single verified run pairs baseline and candidate; cluster-unit records lack cluster ids (cluster ids differing between arms are caught at the boundary, below) | `insufficient_evidence` |
| Baseline or candidate loaded digest (for the named file) or configuration differs from the registration | `baseline_changed_not_interpreted` / `candidate_provenance_mismatch` |
| Weight edit without a hash-consistent `verify-edit` report naming the registered manifest; WSP evidence not from a held-out run; triage-model calibration dataset without a content hash, or with the evaluation dataset's content hash or name and revision | `prerequisite_missing` |
| Evaluated items differ from the registered item set (missing from both arms, extra, or substituted) | **`insufficient_evidence`**: the registered metadata cannot be verified, so no metric is reported |
| With the registered item set: any item lacks one of the expected `(arm, sample)` records; unbalanced pairs (different samples per arm) | reported (`items.unbalanced_items`, `items.design_problems`), and the outcome is **`inconclusive`**. There is no exclusion option. The descriptive metrics use the balanced items (same samples in both arms), which may include incomplete ones |
| Accuracy evidence that is not WSP-sourced | `insufficient_evidence` (and `load_evidence` refuses it) |
| At the comparison boundary, on every path including direct calls: a record with an unexpected type; duplicate `(arm, item, sample)` records; a sample outside the registered design (triage: exactly one per alert and arm); an item's tags, cluster or label differing between arms; an unregistered item set or metadata hash; triage metadata or accuracy slice membership not matching the registered hash | `insufficient_evidence` |
| In the loaders: record arms differ from the manifest's arms; records, run, study or dataset hashes differ; triage metadata or WSP slice membership not as registered | evidence refused |

**Registered design.** A frozen study registers:
- `task.dataset.expected_items_sha256`: the canonical hash of the evaluated item-id set,
  `sha256_json(sorted(set(ids)))`;
- `analysis.samples_per_item`: exactly 1 for triage.

The comparison requires exactly those items and, for both the baseline and the candidate,
every `(arm, item, sample)` record for samples `0 … samples_per_item − 1`. A different item set
is `insufficient_evidence`; a missing sample or an unbalanced pair is `inconclusive`.
- A count alone is not used, because it could miss a substituted item.
- Excluding unbalanced items (complete-case analysis) is **not supported**. In simulation it
  let a candidate with a true effect of −0.05 pass, because its failures left no record. It
  could return only with a reviewed sensitivity method such as worst-case imputation.
- Binding methods also require the registered design to be single-sample, never a property
  inferred from the observed values.

**Calibration data.** Stable dataset identity (name, revision) and content hashes are
checked. Hashes can show that two datasets are the same; they **cannot prove that two
differently packaged datasets are disjoint**. That needs an item-level overlap check on the
data itself, which is an operator task (**Op**).

**Triage metadata.** A frozen triage study registers `task.dataset.metadata_sha256`, the
canonical hash of the original per-alert metadata, independent of any arm's outcome:
- **Canonical form:** `sha256_json(sorted([id, sorted(tags), cluster, label] for each alert))`
  (`evidence.triage_metadata_sha256`, `soc.alerts_metadata_sha256`).
- **At load time,** `load_evidence` rebuilds each alert's tags, cluster and label from the
  records, refuses any disagreement between arms, recomputes the hash and refuses a mismatch.
  A study without the hash cannot load triage evidence.
- **What this verifies:** that the records are **consistent with the registered dataset**.
  It does **not** verify that the labels are true; label provenance is an owner decision.
  Features (the evaluators' inputs) are covered by `content_sha256`, which only the bundled
  writer computes, at write time.

**Target slices for accuracy studies.** WSP items carry no tags. A frozen accuracy study binds
`target_failure.slice_membership_sha256` (`evidence.slice_membership_sha256`, over the
evaluated item IDs and their tags). WSP evidence with different membership is refused.
Accuracy evidence is accepted **only** through `evidence_from_wsp`; `load_evidence` refuses
it. The comparison also recomputes the slice-membership hash from the records it compares
(see below). A directly built `Evidence` with re-tagged items is therefore refused, whatever
its `notes` claim. If its item set differs from the registered one, the membership cannot be
verified, and it is refused for that reason instead.

## Trust assumptions of the loader workflow

The supported workflow is: the bundled writer (or the WSP harness) produces a run directory;
the operator records its manifest identifier **outside** that directory; a reader loads the
run through `load_evidence` / `evidence_from_wsp`, passing the recorded identifier; the report
is built from what the loaders return. What each step establishes:

| Step | Establishes | Does not establish |
|---|---|---|
| Records hash (`records_sha256`), record count, run ids, duplicates | `records.jsonl` matches what its manifest says | Anything about the manifest itself, or who wrote either file |
| Study, dataset, metadata and slice hashes | The evidence is consistent with the registered study and dataset | That labels, tags or outcomes are true |
| Manifest identifier (`manifest_sha256`, canonical JSON of the parsed `manifest.json`) | A deterministic integrity and provenance identifier for the run | Anything on its own: a hash kept with the manifest can be recomputed by whoever edits it, so none is stored there |
| Identifier compared with an externally recorded value (`expected_manifest_sha256`; evidence marked `manifest_check: matched_external_record`) | The manifest's start time, arm artifact digests and configuration, `synthetic`, `limitations` and (WSP) `split` are unchanged since the value was recorded | Authentication. The check is only as strong as whatever protects the external record. Unless that record is under independently enforced access controls or signatures (not provided here), someone who can change both can make them agree |
| WSP `verify_run` and the `verify-edit` report linkage | The WSP run matches its protocol and records hashes; the supplied report hashes to the registered value and names the registered files and edit manifest | Who ran WSP or `verify-edit`, or that the tensor comparison was re-run |

**Consequence in the decision rule.** The start time (preregistration), artifact claims and
WSP split all come from the manifest. `meets_declared_criteria_pending_human_decision`
therefore also requires `manifest_check: matched_external_record`. Without it the rule
outcome is `inconclusive`, with a reason saying so, and the report adds a limitation per
unchecked run. In synthetic tests the "external record" is a dictionary in the test process;
that shows the mechanism, not a control.

The report's `evidence_class` is `synthetic_fixture_not_a_measurement`, or, for a
non-synthetic study and evidence, `measured_unauthenticated` (or
`not_interpreted_malformed_evidence` if a malformed object was supplied). Every report states
that its checks are integrity and consistency checks, not authentication.

**Scope of this review cycle.** The checks are designed for the loader workflow above and
its documented assumptions. They validate ordinary input, including hand-built objects with
the wrong shape, but they are not a defence against arbitrary in-process Python code (which
can, for example, mutate a record dict between checks). Real experiments need the separately
controlled infrastructure described in [the WSP readiness plan](../wsp/run-readiness.md):
storage and evidence registers under independent access control, and authenticated approvals.
None of that exists or is implemented here.

## Direct calls and what is recomputed at the comparison boundary

`compare_candidate` and `build_report` can be called with `Evidence` objects built directly
(for example in a notebook), bypassing the loaders. The table separates what the loaders
guarantee from what every call checks, and from what a caller who builds the object can
simply assert.

| | Through the loaders (`load_evidence`, `evidence_from_wsp`) | On every call, including hand-built objects | Caller can assert on a hand-built object |
|---|---|---|---|
| Candidate | Taken from the study | Resolved from the study by id; an unregistered id or a dict that differs from the registration (canonical JSON) is refused | Nothing |
| Shapes and types | Checked, as `EvidenceError` | Checked, as `insufficient_evidence`: record fields (below), `run_id`, `source`, `synthetic`, `started_at` (ISO-8601 with a time zone), `arms`, `notes`, `limitations` | Nothing |
| Item set, design, arm agreement, metadata or slice hash | Checked | Rechecked from the records compared | Nothing |
| Run, records, study and dataset hashes; record count; `run_status` | Checked | Not checked (there is no run to check) | Everything |
| `verify-edit` report linkage, WSP protocol | Checked | Not checked; read from `notes` | `notes["edit_verification"]` |
| Manifest fields: `started_at`, arm artifact digests and configuration, `synthetic`, `limitations`, WSP `split` | Read from `manifest.json`; unchanged since recording **only if** `manifest_check` is `matched_external_record` (not authentication) | Type-checked; `meets` requires `notes["manifest_check"] == "matched_external_record"` | The values, and `notes["manifest_check"]` itself |
| Record outcomes (`correct`, `decision`, `status`, `error_class`) | Read from `records.jsonl` (bound by `records_sha256`) | Type-checked only | The values |
| `source`, `run_id` | Set by the loader | Type-checked only | The values |

So a hand-built object with `source: "wsp"`, consistent metadata, `notes` claiming a held-out
run, a hash-consistent edit verification and a matched manifest check, and asserted outcomes,
can reach
`meets_declared_criteria_pending_human_decision` without any WSP run or `verify-edit`
report. That is the documented limit of this path, not a gap in the checks: **a result from
a direct call is only as trustworthy as the code that built its evidence.** Use the loaders
for anything that will be read as evidence.

**Order on every call** (→ `insufficient_evidence` on any failure, with no metric
reported):
1. the candidate is resolved from the study;
2. every evidence object passed is shape-checked: a record must be an object with string
   `arm`, `item_id` and `status`, an integer `sample` (not a boolean), `tags` a list (or
   tuple) of strings, `cluster`, `label`, `decision`, `error_class` and `run_id` a string or
   null, and `correct` a boolean or null. The object must be exactly an `Evidence` (not a
   subclass), with string `run_id` and `source`, boolean `synthetic`, `started_at` null or
   ISO-8601 with a time zone and representable in UTC, `limitations` a list of strings, and
   `arms` and `notes` plain JSON objects with string keys: each arm's `artifacts` maps file
   names to digest strings, `artifact_sha256s` is a list of strings, `config_sha256` a string
   or null; `notes["edit_verification"]` is an object and `wsp_split`, `manifest_check` and
   `manifest_sha256` strings or null. Types are checked exactly, so ordinary subclasses that
   change hashing, equality or membership are refused (see the scope note above for what
   this does not cover). The `evidence` argument itself must be a list or tuple. One
   malformed object makes every candidate `insufficient_evidence`; `build_report` lists it
   under `evidence_errors`, does not read it further, and labels the report
   `not_interpreted_malformed_evidence` (or `synthetic_fixture_not_a_measurement` for a
   synthetic study);
3. exactly one run pairs the baseline and candidate; accuracy evidence must say
   `source: "wsp"`;
4. no duplicate `(arm, item, sample)` records, and samples inside the registered design
   (triage: exactly one per alert and arm);
5. tags, cluster and label identical for an item in every arm;
6. the registered item set, then the canonical triage metadata hash or the accuracy
   slice-membership hash. The target slice is then taken from these verified tags;
7. for a cluster-unit study, every record has a cluster id.

Then, in this order: baseline and candidate provenance against the registration
(`baseline_changed_not_interpreted`, `candidate_provenance_mismatch`), prerequisites
(`prerequisite_missing`), the registered samples and balance (an incomplete or unbalanced run
makes the rule outcome `inconclusive`), the decision rules, and finally preregistration
timing and the cluster-descriptive rule. When provenance or a prerequisite fails, that gate
is the outcome even if the design check would also fail.

All of these are **consistency checks against the registered study, not authentication**:
caller-supplied evidence remains unauthenticated.

**Open notes (not fixed):**
- *Registered hashes are not cross-checked when the study is parsed.* `expected_items_sha256`
  is not compared with the item ids inside `metadata_sha256` (triage) or
  `slice_membership_sha256` (accuracy). An inconsistent frozen study fails closed, but only
  once evidence is compared.
- *The metadata canonical form is a sorted list, not a set.* Duplicate tags (`["a", "a"]`
  vs `["a"]`) hash differently, which can only cause a false refusal. (The type half of this
  note is closed: the comparison boundary now requires `tags` to be a list of strings.)

## WSP evidence is tied to the actual comparison

`evidence_from_wsp` raises unless:
1. the study candidate is a `weight_edit` that registered this WSP `protocol_sha256`;
2. the WSP `reference` arm maps to the baseline, and a WSP arm with role `candidate` maps to
   that candidate (the no-edit reload can never stand in for the baseline);
3. the WSP run passes `verify_run` (protocol hash, complete, records hash, run IDs, count);
4. the WSP items hash equals the study's dataset `content_sha256`;
5. the WSP protocol uses `missing_policy: count_as_incorrect`, matching this workflow's
   accuracy definition.

The supplied `verify-edit` report must also hash to the registered `report_sha256`, have
status `verified`, name the registered baseline and candidate file digests, and name the
registered edit manifest (`manifest_sha256` equal to `edit_manifest_sha256`).

**`edit_manifest_sha256` format.** 64 lowercase hex characters (validated): the
**canonical-JSON SHA-256 of the parsed manifest** (`wsp_eval.hashing.sha256_json`, sorted
keys, compact separators, UTF-8), exactly as emitted by `wsp.py verify-edit` in
`manifest_sha256` (#26). It is **not** the hash of the file's bytes (`manifest_file_sha256`,
which `sha256sum` would give); registering that hash fails closed. Reports without
`manifest_sha256` (from the current, merged `verify-edit`) are refused, so weight-edit
evidence loaded through `evidence_from_wsp` cannot pass until the separate WSP change that
adds the field is merged. (A hand-built object can assert the result in `notes`; see the
direct-calls section.) The WSP run
must also be from the held-out split. Only then is
the evidence's edit verification recorded as `hash_consistent`. Without such a report the
comparison is `prerequisite_missing`, whatever the study declares.

This is a **hash and provenance consistency check, not authenticated or independent
verification**:
- it does not establish who produced the report;
- it does not re-run the tensor comparison;
- it does not prove that the files compared were the files evaluated, beyond the digests
  each component reports. Loaded digests are checked for the
**named file** (`artifact.file`, required when frozen), so a tokenizer digest cannot stand in
for the model weights.

## Human decisions

`decide` writes a new file, never overwriting one. A decision is `pending`, `reject`,
`request_more_evidence` or `approve_further_evaluation`; there is no deploy value. The record
always carries:
- `deployment_authorized: false`;
- `identity_verified: false`: the pseudonym is typed, not authenticated;
- `enforced_by: "none"`: no system reads it to gate anything.

The decider may not be the study owner or operator. Approval needs a candidate with the
`meets_declared_criteria…` outcome.

The report's `report_sha256` is recomputed when a decision is recorded. This **detects
accidental edits; it is not tamper-evident**:
- it is an unkeyed self-hash, which anyone who edits a report can recompute;
- outcomes are not re-derived from the evidence at that point.

A decision record is a statement of human judgement, kept where the research owner controls
writes. It is not an access-controlled approval.

## Flow

```bash
.venv/bin/python tools/research/improve.py validate tools/research/fixtures/synthetic_study.json
.venv/bin/python tools/research/improve.py demo --output ~/research/runs/demo-$(date +%s)
.venv/bin/python tools/research/improve.py manifest-hash RUN_DIR   # record outside the run
.venv/bin/python tools/research/improve.py report --study STUDY.json --evidence RUN_DIR... \
    --expected-manifest-sha256 SHA... --output DIR
.venv/bin/python tools/research/improve.py decide --study STUDY.json --report DIR/report.json \
    --decision request_more_evidence --decided-by reviewer1 --rationale "..." --output DECISION.json
```

Comparison gates, in order:
1. `insufficient_evidence` (the candidate is unregistered or differs from its registration; an
   evidence object or record is malformed; no single run pairs baseline and candidate;
   accuracy evidence not WSP-sourced; any comparison-boundary check fails; cluster-unit
   records lack cluster ids). All of these run before steps 2–4;
2. `baseline_changed_not_interpreted`;
3. `candidate_provenance_mismatch`;
4. `prerequisite_missing`;
5. the rule outcome (`constraint_violated` > `regression_on_target` >
   `meets_declared_criteria_pending_human_decision` > `inconclusive`);
6. `inconclusive` if, with the registered item set, samples are missing or pairs unbalanced,
   or if a `meets` result comes from evidence whose manifest was not checked against an
   externally recorded identifier;
7. `descriptive_only_not_preregistered`, unless the study was frozen before the evidence run
   started;
8. `descriptive_only_cluster_method_unreviewed` for cluster-unit studies.

The synthetic demo uses the unfrozen fixture, so its outcome is inconclusive. The tests
freeze copies of it to exercise every path.

## Controls register

**Impl + Test** = enforced by code and covered by synthetic tests. **Op** = needs external
infrastructure or people; not implemented or verified here.

| Requirement | Status | Note |
|---|---|---|
| No network, subprocess, ML/data libraries, `eval`/`exec`; no promote/deploy/publish/upload names; not in the wheel | Impl + Test | Property of this code, not a sandbox |
| Explicit denominators; errors, missing labels, deferrals, empty selections handled as tabulated | Impl + Test | Label *quality* (e.g. analyst dispositions as ground truth) is **Op** |
| Only binding methods (exact CP; Bonferroni paired, binary single-sample items) can pass; others descriptive; cluster-unit results always descriptive; estimand named for every interval; multiplicity stated | Impl + Test | The *validity* of the methods is not established by tests: it needs statistical review. A clustered method, its estimand and the cluster definition are **open decisions** |
| Baseline unchanged, checked per named file and configuration | Impl + Test | Relies on evaluators reporting loaded digests truthfully → **Op** |
| One intervention type per candidate; family prerequisites | Impl + Test | Truth of declared training and calibration data is **Op** |
| Weight edits tied to a hash-consistent `verify-edit` report (files and edit manifest), WSP protocol, arms, dataset, slice membership and held-out split | Impl + Test | A consistency check, not authenticated verification. Running `verify-edit` on real files, and trusting who ran it, is **Op**. Manifest linkage needs the separate WSP change. Enforced by the loader only; a hand-built object can assert it |
| Invented limits labelled `synthetic_fixture`; real studies cannot use them | Impl + Test | Who sets real limits, and when, is **Op** |
| Rules fixed before evidence (freeze + name + `frozen_at` ≤ evidence start) | Impl + Test | Timestamps are recorded, not authenticated. The evidence start time comes from the manifest and is bound only by an external manifest record |
| Manifest identifier; `meets` needs a match with an externally recorded value | Impl + Test | Detects post-run edits only if the external record is itself protected (independent access control or signatures) → **Op**; not authentication |
| Evidence bound to study and dataset; tampered, aborted or mixed runs refused | Impl + Test | Protects records against accidents, not against rewriting both records and manifest. The manifest's own fields are bound only by an external record (row below). Loader only; on direct calls only shapes, design and metadata are rechecked |
| Decision recorded; no self-approval; marked unauthenticated and unenforced; report self-hash checked | Impl + Test | The self-hash catches accidental edits only (not tamper-evident). Write access to decision files and any real approval system are **Op** |
| Calibration not the evaluation set | Impl + Test (identity and hash) | Disjointness of differently packaged data is **Op** |
| Escalation volume capped separately from precision and deferral | Impl + Test | Real capacity numbers are an owner decision |
| WSP limitations and human gates preserved | Impl + Test | WSP's own gates (freeze, tolerance approval, license review, downloads) are unchanged |
| Real triage adapter, alert data access, SIEM write-back, shadow mode, drift monitoring | **Op / not built** | `TriageEvaluator` is the extension point; only `SyntheticTriageEvaluator` exists |

## Next decisions (owner: research owner)

1. Data source and access controls for a real triage study; label provenance and how
   unlabelled alerts are handled.
2. Calibration and evaluation splits (disjoint), and the cluster definition (incident,
   campaign, tenant).
3. Operating limits (benign-escalation cap, defer budget, minimum effect) as
   `declared_by_owner` values, frozen before any evidence.
4. Then whether to build a real `TriageEvaluator`, and where it runs. It cannot be under
   `tools/`, because the isolation test forbids ML libraries there.
5. For training or LoRA candidates: data licensing, compute budget and a separate protocol.
   None is authorized.
