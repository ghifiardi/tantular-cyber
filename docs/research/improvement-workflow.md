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
| weight_edit | `weight_edit` | A different artifact; `base_artifact_sha256` is the baseline digest; edit-manifest hash. Status `verified` plus a `report_sha256` is required in a frozen study, **and** the actual `verify-edit` report must be supplied with the WSP evidence, name the registered edit manifest, and come from a held-out WSP run (below). Until `verify-edit` reports carry `manifest_sha256` (a separate WSP change), weight-edit evidence cannot pass |

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
bounds down), so rounding never moves a bound across a limit in the candidate's favour.

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
| `violated` | the **opposite** one-sided bound is beyond the limit (binding method): a statistically established violation. For `max_net_paired_losses` it means the declared count was exceeded in these records (an observed-count rule, `basis: observed_count_rule`, with `losses: net` or `gross`) |
| `point_beyond_limit` | the observed point is beyond the limit, but a violation is not established. Not satisfied; the outcome is at best inconclusive |
| `inconclusive`, `descriptive_only`, `not_declared`, `not_evaluated` | not satisfied |

Only an established (or observed-count) `violated` makes a candidate `constraint_violated`.
`point_beyond_limit` makes it inconclusive, with that reason.

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
| No single verified run pairs baseline and candidate; cluster-unit records lack cluster ids; one item has conflicting cluster ids | `insufficient_evidence` |
| Baseline or candidate loaded digest (for the named file) or configuration differs from the registration | `baseline_changed_not_interpreted` / `candidate_provenance_mismatch` |
| Weight edit without a hash-consistent `verify-edit` report naming the registered manifest; WSP evidence not from a held-out run; triage-model calibration dataset without a content hash, or with the evaluation dataset's content hash or name and revision | `prerequisite_missing` |
| Unbalanced pairs (different samples per arm) | reported, and the confirmatory outcome is `inconclusive` unless `analysis.unbalanced_policy: exclude_listed_items` was prespecified |
| Record arms differ from the manifest's arms; WSP slice membership not the one bound in the study | evidence refused |

**Calibration data.** Stable dataset identity (name, revision) and content hashes are
checked. Hashes can show that two datasets are the same; they **cannot prove that two
differently packaged datasets are disjoint**. That needs an item-level overlap check on the
data itself, which is an operator task (**Op**).

**Target slices.** For triage, slice tags are part of the alert dataset, so they are covered
by its content hash. WSP items carry no tags. A frozen accuracy study therefore binds
`target_failure.slice_membership_sha256` (`evidence.slice_membership_sha256`, over the
evaluated item IDs and their tags), and WSP evidence with different membership is refused.

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
registered edit manifest (`manifest_sha256` equal to `edit_manifest_sha256`). Reports without
`manifest_sha256` (from the current, merged `verify-edit`) are refused, so weight-edit
evidence cannot pass until the separate WSP change that adds the field is merged. The WSP run
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
.venv/bin/python tools/research/improve.py report --study STUDY.json --evidence RUN_DIR... --output DIR
.venv/bin/python tools/research/improve.py decide --study STUDY.json --report DIR/report.json \
    --decision request_more_evidence --decided-by reviewer1 --rationale "..." --output DECISION.json
```

Comparison gates, in order:
1. `insufficient_evidence`;
2. `baseline_changed_not_interpreted`;
3. `candidate_provenance_mismatch`;
4. `prerequisite_missing`;
5. the rule outcome (`constraint_violated` > `regression_on_target` >
   `meets_declared_criteria_pending_human_decision` > `inconclusive`);
6. `descriptive_only_not_preregistered`, unless the study was frozen before the evidence run
   started.

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
| Weight edits tied to a hash-consistent `verify-edit` report (files and edit manifest), WSP protocol, arms, dataset, slice membership and held-out split | Impl + Test | A consistency check, not authenticated verification. Running `verify-edit` on real files, and trusting who ran it, is **Op**. Manifest linkage needs the separate WSP change |
| Invented limits labelled `synthetic_fixture`; real studies cannot use them | Impl + Test | Who sets real limits, and when, is **Op** |
| Rules fixed before evidence (freeze + name + `frozen_at` ≤ evidence start) | Impl + Test | Timestamps are recorded, not authenticated |
| Evidence bound to study and dataset; tampered, aborted or mixed runs refused | Impl + Test | Protects against accidents, not against rewriting both records and manifest |
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
