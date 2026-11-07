"""Compare each candidate with the unchanged baseline under the study's declared rules.

Gates, in order; the first that fails decides the candidate's outcome:

1. ``insufficient_evidence``: the candidate is not registered in the study, or the dict
   passed differs from its registration (only the registered candidate is compared); an
   evidence object or record has the wrong shape or types (checked before any field is
   read, so malformed input is a gate, not an exception); no single evidence run contains
   both the baseline and the candidate (comparisons are always paired within one run);
   accuracy evidence that is not WSP-sourced; any comparison-boundary consistency check
   fails (duplicate records, a sample outside the registered design, tags/cluster/label
   differing between arms, an evaluated item set that differs from the registered one,
   triage metadata or accuracy slice membership not matching the registered hash); or a
   cluster-level study's records lack cluster ids. These all run before steps 2-4.
2. ``baseline_changed_not_interpreted``: the baseline's loaded artifact (the registered file,
   when named) or configuration differs from the registration, or is not recorded.
3. ``candidate_provenance_mismatch``: the same check for the candidate.
4. ``prerequisite_missing``: an intervention-specific precondition is unmet (a weight edit
   without a ``verify-edit`` report that is hash-consistent with this evidence; a triage model
   calibrated on the evaluation dataset).
5. Metrics are computed. Constraints and the target effect are evaluated; then
   ``constraint_violated`` > ``regression_on_target`` >
   ``meets_declared_criteria_pending_human_decision`` > ``inconclusive``.
6. If the study was not frozen before the evidence run started, the outcome is replaced by
   ``descriptive_only_not_preregistered``. Otherwise, if the study's unit is ``cluster``, it is
   replaced by ``descriptive_only_cluster_method_unreviewed``: no reviewed clustered method
   or matching estimand has been specified. In both cases the rule outcome is kept beside it.

**Decision rule.** The target criterion is a confidence-bound criterion: the *lower* bound of
the improvement interval must be above zero and at least ``min_meaningful_effect``, i.e. the
data must rule out improvements smaller than the declared minimum. A point estimate above the
minimum is not enough. A regression needs the upper bound below zero. ``meets`` also needs
evidence whose manifest matched an externally recorded identifier
(``notes["run_manifest_check"] == "matched_external_record"``): the start time, artifact claims
and WSP split come from the manifest. Otherwise the rule outcome is ``inconclusive``. A match
is an integrity check against that record, not authentication.

**Constraints** distinguish what was observed from what is established:

* ``satisfied``: the claim-side one-sided bound is within the limit (binding method);
* ``violated``: the *opposite* one-sided bound is beyond the limit (binding method), i.e. a
  violation is statistically established. For ``max_net_paired_losses`` it means the declared
  count was exceeded in these records: an observed-count rule, not a statistical claim;
* ``point_beyond_limit``: the observed point is beyond the limit but a violation is not
  established. Not satisfied; the outcome is at best inconclusive;
* ``inconclusive``, ``descriptive_only``, ``not_declared``, ``not_evaluated``: not satisfied.

**Registered design.** The evidence must contain exactly the registered items
(``task.dataset.expected_items_sha256``, the canonical hash of the item-id set) and, for both
the baseline and the candidate, exactly the records ``(arm, item, sample)`` for samples
``0 .. analysis.samples_per_item - 1``. A missing, extra or substituted item is
``insufficient_evidence`` (the registered metadata cannot then be verified, so no metric is
reported). With the registered item set, a missing sample or an unbalanced pair makes the
outcome ``inconclusive``; its descriptive metrics use the balanced items (same samples in both
arms), which may include incomplete ones. There is no exclusion option. Binding methods also
require the registered design to be single-sample.

**Comparison-boundary consistency checks.** Before provenance or prerequisites,
``compare_candidate`` resolves the candidate from the study, shape-checks every evidence
object (``evidence.evidence_problem``: exact types for record fields, ``started_at``,
``arms``, ``notes``, ``limitations``), and recomputes from the records it actually uses: no
duplicate ``(arm, item, sample)`` records; samples inside the
registered design (exactly one per alert and arm for triage); identical tags, cluster and
label for an item in every arm; the registered item set; and the canonical triage metadata
hash (``metadata_sha256``) or accuracy slice-membership hash (``slice_membership_sha256``).
The target slice is then taken from those verified tags. Any failure is
``insufficient_evidence``. These are **consistency checks against the registered study, not
authentication**: caller-supplied evidence remains unauthenticated. Record outcomes
(``correct``, ``decision``, ``status``, ``error_class``), ``source``, ``synthetic``,
``run_id``, ``limitations``, ``notes`` (WSP split, edit verification), ``started_at`` and the
arms' reported digests and configuration remain as asserted by whoever built the object.
Only the loaders also check run, records, study and dataset hashes, the ``verify-edit``
report and (against an external record, when given) the manifest; a hand-built object can
assert all of those in ``notes``. These checks are for ordinary input in the loader
workflow, not a defence against arbitrary in-process code.

**Fail-closed preconditions.** A calibration dataset without a content
hash or with the evaluation dataset's identity or hash, and WSP evidence from a non-held-out
split stop the comparison before any rule is applied.

**Binding methods only.** A bound can make a constraint ``satisfied``, or a target meet or
regress, only if its method status is in ``metrics.BINDING_STATUSES`` (exact Clopper-Pearson
on independent units, or the Bonferroni paired interval for binary single-sample items).
Any other method yields ``descriptive_only``. Nothing is adjusted for testing several
candidates and constraints; the report says so.

``meets_declared_criteria_pending_human_decision`` is the strongest outcome. It promotes or
deploys nothing; a person records a decision (``decision.py``), and any deployment needs its
own evaluation and release process outside this workflow.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from wsp_eval.hashing import sha256_json

from .evidence import (
    Evidence,
    EvidenceError,
    evidence_problem,
    item_set_sha256,
    parse_timestamp,
    records_metadata,
    slice_membership_sha256,
    triage_metadata_sha256,
)
from .metrics import (
    BINDING_STATUSES,
    group,
    item_values,
    net_paired_losses,
    one_sided_bound,
    paired_difference_bootstrap,
    paired_improvement,
    ratio,
    triage_breakdown,
)
from .study import METRICS, Study, StudyError

RULE_OUTCOMES = (
    "constraint_violated",
    "regression_on_target",
    "meets_declared_criteria_pending_human_decision",
    "inconclusive",
)
DESCRIPTIVE_OUTCOMES = (
    "descriptive_only_not_preregistered",
    "descriptive_only_cluster_method_unreviewed",
)
GATE_OUTCOMES = (
    "insufficient_evidence",
    "baseline_changed_not_interpreted",
    "candidate_provenance_mismatch",
    "prerequisite_missing",
)


def preregistration(study: Study, ev: Evidence) -> dict[str, Any]:
    reasons = []
    if not study.frozen:
        reasons.append("study not frozen")
    elif not ev.started_at:
        reasons.append("evidence start time not recorded")
    else:
        try:
            late = parse_timestamp(study.raw["frozen_at"]) > parse_timestamp(ev.started_at)
        except EvidenceError as exc:
            reasons.append(f"evidence start time not usable: {exc}")
        else:
            if late:
                reasons.append("study frozen after the evidence run started")
    return {"preregistered": not reasons, "reasons": reasons}


def _provenance(registered: dict[str, Any], loaded: dict[str, Any] | None) -> list[str]:
    if loaded is None:
        return ["no evidence for this arm"]
    problems = []
    art = registered["artifact"]
    sha, file = art.get("sha256"), art.get("file")
    if sha is None:
        problems.append("registered artifact sha256 not set")
    elif file is not None:
        got = (loaded.get("artifacts") or {}).get(file)
        if got is None:
            problems.append(f"loaded artifacts do not report the registered file {file!r}")
        elif got != sha:
            problems.append(f"loaded {file!r} digest differs from the registration")
    elif sha not in loaded.get("artifact_sha256s", []):
        problems.append("loaded artifacts do not include the registered artifact sha256")
    if loaded.get("config_sha256") != sha256_json(registered["config"]):
        problems.append("loaded configuration differs from the registration")
    return problems


def _prerequisites(study: Study, cand: dict[str, Any], ev: Evidence) -> list[str]:
    iv = cand["intervention"]
    if iv["type"] == "weight_edit":
        if iv["edit_verification"].get("status") != "verified":
            return [f"edit verification is {iv['edit_verification'].get('status')!r}, not verified"]
        check = ev.notes.get("edit_verification", {})
        if check.get("candidate") != cand["id"] or check.get("status") != "hash_consistent":
            return [
                "no verify-edit report that is hash-consistent with this candidate's evidence "
                "(a consistency check, not authenticated verification)"
            ]
        problem = _edit_scope_problem(study, cand, ev)
        if problem:
            return [problem]
    if iv["type"] == "soc_triage_model":
        # Identity and content hashes can show that calibration and evaluation data are the
        # same; they cannot prove that two differently packaged datasets are disjoint (that
        # needs item-level overlap checks on the data itself, an operator task).
        cal = iv["calibration"]["dataset"]
        evd = study.raw["task"]["dataset"]
        if not cal.get("content_sha256"):
            return ["calibration dataset content_sha256 not declared; identity cannot be checked"]
        if cal["content_sha256"] == evd.get("content_sha256"):
            return ["calibration dataset has the evaluation dataset's content hash"]
        if (cal.get("name"), cal.get("revision")) == (evd.get("name"), evd.get("revision")):
            return ["calibration dataset has the evaluation dataset's name and revision"]
    if ev.source == "wsp" and ev.notes.get("wsp_split") != "heldout":
        return [
            f"WSP evidence is from the {ev.notes.get('wsp_split')!r} split; a confirmatory "
            "comparison needs a held-out run"
        ]
    return []


def _edit_scope_problem(study: Study, cand: dict[str, Any], ev: Evidence) -> str | None:
    """A weight edit may change only the registered edited file (W1).

    Both arms must report the same complete set of loaded files, with equal digests for every
    file except the one the intervention registers (``artifact.file``, the file the
    ``verify-edit`` report covers). This version allows exactly one edited file: several edited
    shards or configuration changes need their own registered and reviewed intervention
    design, not an exemption. It relies on the evaluator reporting every file it loaded;
    a file it does not report is invisible here.
    """
    edited = cand["artifact"].get("file")
    if not edited or edited != study.baseline["artifact"].get("file"):
        return (
            "weight edit: the edited file must be registered (artifact.file) and be the same "
            "file as the baseline's registered file"
        )
    base = (ev.arms.get(study.baseline["id"]) or {}).get("artifacts") or {}
    other = (ev.arms.get(cand["id"]) or {}).get("artifacts") or {}
    if not base or not other:
        return (
            "weight edit: both arms must report every loaded file; without that, the "
            "candidate cannot be shown to differ only in the edited file"
        )
    if set(base) != set(other):
        only_base, only_cand = sorted(set(base) - set(other)), sorted(set(other) - set(base))
        return (
            "weight edit: the arms loaded different file sets (only baseline: "
            f"{only_base}; only candidate: {only_cand}); only {edited!r} may differ"
        )
    changed = sorted(f for f in base if f != edited and base[f] != other[f])
    if changed:
        return (
            f"weight edit: files other than the registered edited file {edited!r} differ "
            f"between arms: {changed}"
        )
    return None


def _values(metric: str, records: Sequence[dict[str, Any]], arm: str, items: set[str]):
    return item_values(metric, group(records, arm, items))


def _summary(values: dict, keys: Sequence[str]) -> dict[str, Any]:
    num, den, val = ratio(values, keys)
    return {"value": val, "numerator": num, "denominator": den}


def _improvement(
    study: Study, metric: str, base, cand, keys, clusters, single_sample: bool
) -> dict[str, Any]:
    _, better, pairable = METRICS[metric]
    a = study.analysis
    if pairable:
        return paired_improvement(
            base, cand, keys, better, a["confidence"], clusters, single_sample=single_sample
        )
    return paired_difference_bootstrap(
        base,
        cand,
        keys,
        better,
        a["confidence"],
        a["bootstrap_resamples"],
        a["bootstrap_seed"],
        clusters,
    )


def _absolute(
    out: dict[str, Any], k: dict[str, Any], cand, keys, conf, clusters, single_sample: bool
) -> None:
    op, limit = k["op"], k.get("bound")
    # "Satisfied" needs the bound on the claim side inside the limit; an "established"
    # violation needs the opposite bound beyond it. The observed point alone is not enough.
    claim_side, other_side = ("upper", "lower") if op == "<=" else ("lower", "upper")
    b = one_sided_bound(cand, keys, claim_side, conf, clusters, single_sample=single_sample)
    against = one_sided_bound(cand, keys, other_side, conf, clusters, single_sample=single_sample)
    point = out["candidate"]["value"]
    out.update(op=op, bound=limit, confidence_bound=b, opposite_bound=against)
    beyond = (lambda x: x > limit) if op == "<=" else (lambda x: x < limit)
    within = (lambda x: x <= limit) if op == "<=" else (lambda x: x >= limit)
    binding = b["status"] in BINDING_STATUSES
    if limit is None:
        out["status"] = "not_declared"
    elif point is None or b["bound"] is None:
        out["status"] = "not_evaluated"  # e.g. nothing selected: the fraction is undefined
    elif binding and against["bound"] is not None and beyond(against["bound"]):
        out["status"] = "violated"
    elif beyond(point):
        out["status"] = "point_beyond_limit"
    elif not binding:
        out["status"] = "descriptive_only"
    elif within(b["bound"]):
        out["status"] = "satisfied"
    else:
        out["status"] = "inconclusive"


def _constraint(study, k, records, cand_id, items, clusters, single_sample) -> dict[str, Any]:
    metric, scope = k["metric"], k.get("scope", "all")
    _, better, _ = METRICS[metric]
    keys = sorted(items[scope])
    base = _values(metric, records, study.baseline["id"], items[scope])
    cand = _values(metric, records, cand_id, items[scope])
    out: dict[str, Any] = {
        "id": k["id"],
        "metric": metric,
        "scope": scope,
        "kind": k["kind"],
        "bound_source": k["bound_source"],
        "basis": "observed_count_rule"
        if k["kind"] == "max_net_paired_losses"
        else ("confidence_bound"),
        "candidate": _summary(cand, keys),
        "baseline": _summary(base, keys),
    }
    if k["kind"] == "absolute":
        _absolute(out, k, cand, keys, study.analysis["confidence"], clusters, single_sample)
    elif k["kind"] == "non_regression":
        imp = _improvement(study, metric, base, cand, keys, clusters, single_sample)
        margin = k.get("margin")
        out.update(margin=margin, improvement=imp)
        lo, hi = imp["ci"]
        binding = imp["status"] in BINDING_STATUSES
        if margin is None:
            out["status"] = "not_declared"
        elif imp["point"] is None:
            out["status"] = "not_evaluated"
        elif binding and hi is not None and hi < -margin:
            out["status"] = "violated"
        elif imp["point"] < -margin:
            out["status"] = "point_beyond_limit"
        elif not binding:
            out["status"] = "descriptive_only"
        elif lo is not None and lo >= -margin:
            out["status"] = "satisfied"
        else:
            out["status"] = "inconclusive"
    else:
        # An observed-count rule (docs/evaluation.md adoption rule), not a statistical claim:
        # "violated" means the declared count was exceeded in these records.
        losses = net_paired_losses(base, cand, keys, better)
        limit, mode = k.get("max_losses"), k.get("losses", "gross")
        out.update(max_losses=limit, losses_counted=mode, paired=losses)
        if limit is None:
            out["status"] = "not_declared"
        elif losses is None:
            out["status"] = "not_evaluated"
        else:
            count = losses["net_losses"] if mode == "net" else losses["losses"]
            out["status"] = "satisfied" if count <= limit else "violated"
    return out


Meta = dict[str, tuple[tuple[str, ...], str | None, str | None]]


def _boundary_problems(study: Study, records: Sequence[dict[str, Any]]) -> tuple[list[str], Meta]:
    """Consistency checks recomputed from the records actually compared.

    These check the records against the registered study; they do not authenticate them.
    Returns (problems, canonical per-item metadata). Any problem means insufficient_evidence:
    no metric is reported from records whose metadata could not be verified.
    """
    spi = study.analysis.get("samples_per_item")
    if study.task_kind == "alert_triage":
        allowed = {0}  # one decision per alert and arm
    else:
        allowed = set(range(spi)) if isinstance(spi, int) else None
    seen: set[tuple[str, str, int]] = set()
    for r in records:  # record types were validated with the evidence object
        key = (r["arm"], r["item_id"], r["sample"])
        if key in seen:
            return [f"duplicate record for arm/item/sample {key}"], {}
        seen.add(key)
        if allowed is not None and r["sample"] not in allowed:
            return [
                f"item {r['item_id']}: sample {r['sample']} is outside the registered design"
            ], {}
    try:
        meta = records_metadata(records)
    except EvidenceError as exc:
        return [str(exc)], {}
    ds = study.raw["task"]["dataset"]
    expected = ds.get("expected_items_sha256")
    if expected is None:
        return [
            "task.dataset.expected_items_sha256 not registered; metadata cannot be verified"
        ], {}
    if item_set_sha256(meta) != expected:
        # Missing, extra or substituted items: the registered metadata (which covers the whole
        # registered item set) cannot be verified, so no metric is reported.
        return [
            "evaluated items differ from the registered item set (missing, extra or substituted "
            "items); tags, clusters and labels cannot be verified"
        ], {}
    if study.task_kind == "alert_triage":
        registered = ds.get("metadata_sha256")
        if registered is None:
            return ["task.dataset.metadata_sha256 not registered; metadata cannot be verified"], {}
        if triage_metadata_sha256(meta) != registered:
            return [
                "tags, clusters or labels in the compared records differ from the registered "
                "metadata (consistency check, not authentication)"
            ], {}
    else:
        registered = study.target.get("slice_membership_sha256")
        if registered is None:
            return ["target_failure.slice_membership_sha256 not registered"], {}
        tags = {i: m[0] for i, m in meta.items()}
        if slice_membership_sha256(list(meta), tags) != registered:
            return [
                "target-slice membership in the compared records differs from the registered "
                "slice (consistency check, not authentication)"
            ], {}
    return [], meta


def _registered(study: Study, cand: Any) -> tuple[dict[str, Any] | None, str | None]:
    """The study's own registration of ``cand``, or why it cannot be used.

    A caller may pass any dict; only the registered candidate is compared, and a dict that is
    not exactly the registration (canonical JSON) is refused rather than silently replaced.
    """
    cid = cand.get("id") if type(cand) is dict else None
    if type(cid) is not str:
        return None, "candidate must be an object with a string id"
    try:
        reg = study.candidate(cid)
    except StudyError:
        return None, f"candidate {cid!r} is not registered in the study"
    if cand is not reg:
        try:
            same = sha256_json(cand) == sha256_json(reg)
        except (TypeError, ValueError):
            same = False
        if not same:
            return None, (
                f"candidate {cid!r} differs from the study's registration; only the "
                "registered candidate is compared"
            )
    return reg, None


def compare_candidate(study: Study, cand: dict[str, Any], evidence: Sequence[Evidence]) -> dict:
    base_id = study.baseline["id"]
    result: dict[str, Any] = {
        "candidate": cand.get("id") if type(cand) is dict else None,
        "intervention_type": None,
        "hypothesis": None,
        "reasons": [],
        "constraints": [],
        "target_effect": None,
        "evidence_run": None,
        "rule_outcome": None,
    }
    reg, problem = _registered(study, cand)
    if reg is None:
        result["outcome"] = "insufficient_evidence"
        result["reasons"].append(problem)
        return result
    cand, cid = reg, reg["id"]
    result["intervention_type"] = cand["intervention"]["type"]
    result["hypothesis"] = cand["hypothesis"]
    # Shape checks before any field is read, so malformed input gives a gate, not an exception.
    if type(evidence) not in (list, tuple):
        result["outcome"] = "insufficient_evidence"
        result["reasons"].append("evidence must be a list of Evidence objects")
        return result
    for i, e in enumerate(evidence):
        problem = evidence_problem(e)
        if problem:
            result["outcome"] = "insufficient_evidence"
            result["reasons"].append(f"evidence object {i}: {problem}")
            return result
    runs = [e for e in evidence if base_id in e.arms and cid in e.arms]
    if len(runs) != 1:
        result["outcome"] = "insufficient_evidence"
        result["reasons"].append(
            "no verified evidence run contains baseline and candidate"
            if not runs
            else f"{len(runs)} evidence runs pair this candidate; register one per comparison"
        )
        return result
    (ev,) = runs
    result["evidence_run"] = {"run_id": ev.run_id, "source": ev.source, "synthetic": ev.synthetic}
    if study.task_kind == "accuracy" and ev.source != "wsp":
        result["outcome"] = "insufficient_evidence"
        result["reasons"].append("accuracy evidence must be WSP-sourced (evidence_from_wsp)")
        return result
    records = [r for r in ev.records if r["arm"] in (base_id, cid)]
    boundary, meta = _boundary_problems(study, records)
    if boundary:
        result["outcome"] = "insufficient_evidence"
        result["reasons"] += boundary
        return result
    unit = study.analysis.get("unit", "item")
    clusters = None
    if unit == "cluster":
        if any(m[1] is None for m in meta.values()):
            result["outcome"] = "insufficient_evidence"
            result["reasons"].append("cluster-level analysis declared but records lack cluster ids")
            return result
        # Cluster ids already agree between arms (checked at the boundary).
        clusters = {i: m[1] for i, m in meta.items()}
    for outcome, problems in (
        ("baseline_changed_not_interpreted", _provenance(study.baseline, ev.arms.get(base_id))),
        ("candidate_provenance_mismatch", _provenance(cand, ev.arms.get(cid))),
        ("prerequisite_missing", _prerequisites(study, cand, ev)),
    ):
        if problems:
            result["outcome"] = outcome
            result["reasons"].extend(problems)
            return result

    all_items = {r["item_id"] for r in records}
    samples: dict[tuple[str, str], set[int]] = {}
    for r in records:
        samples.setdefault((r["arm"], r["item_id"]), set()).add(r["sample"])
    unbalanced = sorted(i for i in all_items if samples.get((base_id, i)) != samples.get((cid, i)))
    paired = all_items - set(unbalanced)
    # Registered design: exact item set, and every (arm, item, sample) record present.
    # (The item set itself was verified at the boundary.)
    design_problems: list[str] = []
    spi = study.analysis.get("samples_per_item")
    if spi is None:
        design_problems.append("analysis.samples_per_item not registered")
    else:
        want = set(range(spi))
        incomplete = sorted(
            i
            for i in all_items
            if samples.get((base_id, i), set()) != want or samples.get((cid, i), set()) != want
        )
        if incomplete:
            design_problems.append(
                f"{len(incomplete)} item(s) lack the expected (arm, sample) records "
                f"(e.g. {incomplete[0]})"
            )
    if unbalanced:
        design_problems.append(f"{len(unbalanced)} item(s) have different samples in the two arms")
    single_sample = spi == 1 and not design_problems
    tag = study.target["slice_tag"]
    target_items = {i for i, m in meta.items() if tag in m[0]} & paired  # canonical tags
    items = {"all": paired, "target_slice": target_items}
    result["unit"] = unit
    result["items"] = {
        "paired": len(paired),
        "target_slice": len(target_items),
        "unbalanced_items": unbalanced,
        "design_problems": design_problems,
        "single_sample_design": single_sample,
    }
    if study.task_kind == "alert_triage":
        result["breakdown"] = {
            arm: triage_breakdown(group(records, arm, paired)) for arm in (base_id, cid)
        }

    metric = study.target["metric"]
    keys = sorted(target_items)
    base = _values(metric, records, base_id, target_items)
    candv = _values(metric, records, cid, target_items)
    improvement = _improvement(study, metric, base, candv, keys, clusters, single_sample)
    min_effect = study.target.get("min_meaningful_effect")
    result["target_effect"] = {
        "metric": metric,
        "baseline": _summary(base, keys),
        "candidate": _summary(candv, keys),
        "improvement": improvement,
        "min_meaningful_effect": min_effect,
        "bound_source": study.target["bound_source"],
    }
    result["constraints"] = [
        _constraint(study, k, records, cid, items, clusters, single_sample)
        for k in study.constraints
    ]

    statuses = [c["status"] for c in result["constraints"]]
    lo, hi = improvement["ci"]
    binding = improvement["status"] in BINDING_STATUSES
    if "violated" in statuses:
        rule = "constraint_violated"
        result["reasons"] += [
            f"constraint {c['id']} violated"
            for c in result["constraints"]
            if c["status"] == "violated"
        ]
    elif binding and hi is not None and hi < 0:
        rule = "regression_on_target"
        result["reasons"].append("target metric is worse than baseline (interval below zero)")
    elif (
        binding
        and min_effect is not None
        and lo is not None
        and lo > 0
        and lo >= min_effect
        and all(s == "satisfied" for s in statuses)
    ):
        rule = "meets_declared_criteria_pending_human_decision"
    else:
        rule = "inconclusive"
        if min_effect is None:
            result["reasons"].append("min_meaningful_effect not declared")
        if not binding:
            result["reasons"].append(
                f"target interval method {improvement['method']!r} is "
                f"{improvement['status']}: descriptive only"
            )
        if lo is None:
            result["reasons"].append(
                f"target effect could not be estimated ({improvement['method']})"
            )
        elif not lo > 0:
            result["reasons"].append("target improvement interval includes zero")
        elif min_effect is not None and lo < min_effect:
            result["reasons"].append(
                "lower bound of the target improvement is below min_meaningful_effect"
            )
        for c in result["constraints"]:
            if c["status"] == "point_beyond_limit":
                result["reasons"].append(
                    f"constraint {c['id']}: observed point beyond the limit, violation not "
                    "statistically established"
                )
            elif c["status"] != "satisfied":
                result["reasons"].append(f"constraint {c['id']} {c['status']}")
    if design_problems:
        # An incomplete or unbalanced run is never confirmatory, whatever the rule found.
        rule = "inconclusive"
        result["reasons"] += [f"design: {x}" for x in design_problems]
        result["reasons"].append("incomplete or unbalanced evidence: outcome is inconclusive")
    if rule == "meets_declared_criteria_pending_human_decision" and (
        not study.synthetic and ev.synthetic
    ):
        # W2: synthetic-fixture evidence cannot support a real study's candidate.
        rule = "inconclusive"
        result["reasons"].append(
            "synthetic evidence cannot support a non-synthetic study: outcome is inconclusive"
        )
    if rule == "meets_declared_criteria_pending_human_decision" and (
        ev.notes.get("run_manifest_check") != "matched_external_record"
    ):
        # The start time (preregistration), arm artifact claims and WSP split come from the
        # run's manifest, which no hash in the run directory binds. Without a comparison with
        # an externally recorded run-manifest identifier, post-run edits are undetectable.
        rule = "inconclusive"
        result["reasons"].append(
            "evidence manifest not checked against an externally recorded run_manifest_sha256: "
            "its start time, artifact claims and split are as written in the run directory; "
            "outcome is inconclusive (a match would still not be authentication)"
        )
    prereg = preregistration(study, ev)
    result["preregistration"] = prereg
    result["rule_outcome"] = rule
    if not prereg["preregistered"]:
        result["outcome"] = "descriptive_only_not_preregistered"
        result["reasons"] += prereg["reasons"]
    elif unit == "cluster":
        result["outcome"] = "descriptive_only_cluster_method_unreviewed"
        result["reasons"].append(
            "cluster-unit results are descriptive until a reviewed clustered method and "
            "matching estimand are specified"
        )
    else:
        result["outcome"] = rule
    return result
