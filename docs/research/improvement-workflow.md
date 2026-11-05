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
| weight_edit | `weight_edit` | A different artifact; `base_artifact_sha256` is the baseline digest; edit-manifest hash. Status `verified` plus a `report_sha256` is required in a frozen study, **and** the actual `verify-edit` report must be supplied with the WSP evidence and match (below) |

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
| `error_rate` | errored samples | samples | — | counted | — | — |
| `accuracy` | correct samples (ungraded and errors count as incorrect) | samples | — | incorrect | — | — |

A frozen triage study must declare absolute `<=` caps (scope `all`) on both
`benign_escalation_fraction` and `defer_rate`. Without them, missed threats could be "fixed"
by escalating or deferring everything.

## Intervals and what the report may claim

- **Unit of analysis:**
  - `analysis.unit` is `item` (alerts treated as independent) or `cluster`, where each record
    carries an incident or campaign id.
  - Only units with a non-zero denominator count.
  - An item-unit triage report says that correlated alerts make its intervals too narrow.
  - A cluster-unit study whose records lack cluster ids is `insufficient_evidence`.
- **Absolute constraints:**
  - The bound is one-sided at `analysis.confidence`.
  - It is exact Clopper–Pearson for whole counts with item units. Otherwise it is a
    conservative effective-n Clopper–Pearson over units.
  - A point estimate outside the limit means `violated`. Only a bound inside the limit means
    `satisfied`. Anything else is `inconclusive`.
  - Zero observed events still give a non-zero upper bound.
- **Paired improvement** (target effect and `non_regression`, pairable metrics):
  - It uses per-unit signed differences with Bonferroni-combined Clopper–Pearson bounds on
    the gains and the losses.
  - This is exact for single-sample binary items and conservative otherwise.
  - It never collapses to a zero-width interval when many items flip together. Twelve flips
    give a lower bound of about 0.39, not 1.0; as six campaigns, the interval includes zero.
- **Non-pairable comparisons** (benign escalation under `non_regression`): a unit bootstrap.
  A degenerate (zero-width) bootstrap is reported as not estimable.
- **Criteria:**
  - `meets_declared_criteria_pending_human_decision` needs a preregistered study, the whole
    target interval above zero, a point estimate ≥ `min_meaningful_effect`, and every
    constraint `satisfied`.
  - `not_declared` and `not_evaluated` constraints keep the outcome inconclusive.
- **Multiplicity:** nothing is adjusted for several candidates and constraints, and every
  report says so.

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
status `verified`, and name the registered baseline and candidate file digests. Only then is
the evidence's edit verification `matched`. Without a matched report the comparison is
`prerequisite_missing`, whatever the study declares. Loaded digests are checked for the
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
`meets_declared_criteria…` outcome. The report's hash is recomputed, so an edited report is
refused. A decision record is a statement of human judgement, kept where the research owner
controls writes. It is not an access-controlled approval.

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
| Intervals no stronger than the data; cluster unit available; multiplicity stated | Impl + Test | Choosing the right cluster definition is **Op** |
| Baseline unchanged, checked per named file and configuration | Impl + Test | Relies on evaluators reporting loaded digests truthfully → **Op** |
| One intervention type per candidate; family prerequisites | Impl + Test | Truth of declared training and calibration data is **Op** |
| Weight edits tied to a matching `verify-edit` report, WSP protocol, arms and dataset | Impl + Test | Running `verify-edit` on real files is **Op** |
| Invented limits labelled `synthetic_fixture`; real studies cannot use them | Impl + Test | Who sets real limits, and when, is **Op** |
| Rules fixed before evidence (freeze + name + `frozen_at` ≤ evidence start) | Impl + Test | Timestamps are recorded, not authenticated |
| Evidence bound to study and dataset; tampered, aborted or mixed runs refused | Impl + Test | Protects against accidents, not against rewriting both records and manifest |
| Decision recorded; no self-approval; marked unauthenticated and unenforced | Impl + Test | Write access to decision files and any real approval system are **Op** |
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
