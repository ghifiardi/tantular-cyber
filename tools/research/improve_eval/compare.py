"""Compare each candidate with the unchanged baseline under the study's declared rules.

Gates, in order; the first that fails decides the candidate's outcome:

1. ``insufficient_evidence``: no single verified evidence run contains both the baseline and
   the candidate (comparisons are always paired within one run), or a cluster-level study's
   records lack cluster ids.
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
minimum is not enough. A regression needs the upper bound below zero.

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
``0 .. analysis.samples_per_item - 1``. A missing, extra or substituted item, a missing sample,
or an unbalanced pair makes the outcome ``inconclusive`` (the rule outcome is still reported
as ``rule_outcome_before_design_check``). There is no exclusion option. Binding methods also
require the registered design to be single-sample.

**Fail-closed preconditions.** Conflicting cluster ids, a calibration dataset without a content
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
from datetime import UTC, datetime
from typing import Any

from wsp_eval.hashing import sha256_json

from .evidence import Evidence, item_set_sha256
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
from .study import METRICS, Study

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


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)


def preregistration(study: Study, ev: Evidence) -> dict[str, Any]:
    reasons = []
    if not study.frozen:
        reasons.append("study not frozen")
    elif not ev.started_at:
        reasons.append("evidence start time not recorded")
    elif _ts(study.raw["frozen_at"]) > _ts(ev.started_at):
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


def compare_candidate(study: Study, cand: dict[str, Any], evidence: Sequence[Evidence]) -> dict:
    cid, base_id = cand["id"], study.baseline["id"]
    result: dict[str, Any] = {
        "candidate": cid,
        "intervention_type": cand["intervention"]["type"],
        "hypothesis": cand["hypothesis"],
        "reasons": [],
        "constraints": [],
        "target_effect": None,
        "evidence_run": None,
        "rule_outcome": None,
    }
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
    if study.task_kind == "accuracy" and not (
        ev.source == "wsp" and ev.notes.get("slice_membership_verified") is True
    ):
        # Accuracy items carry no registered tags of their own; only evidence_from_wsp checks
        # slice membership against the frozen study.
        result["outcome"] = "insufficient_evidence"
        result["reasons"].append(
            "accuracy evidence must come through evidence_from_wsp with slice membership checked"
        )
        return result
    for outcome, problems in (
        ("baseline_changed_not_interpreted", _provenance(study.baseline, ev.arms.get(base_id))),
        ("candidate_provenance_mismatch", _provenance(cand, ev.arms.get(cid))),
        ("prerequisite_missing", _prerequisites(study, cand, ev)),
    ):
        if problems:
            result["outcome"] = outcome
            result["reasons"].extend(problems)
            return result

    records = [r for r in ev.records if r["arm"] in (base_id, cid)]
    unit = study.analysis.get("unit", "item")
    clusters = None
    if unit == "cluster":
        if any(not r.get("cluster") for r in records):
            result["outcome"] = "insufficient_evidence"
            result["reasons"].append("cluster-level analysis declared but records lack cluster ids")
            return result
        clusters = {}
        for r in records:
            if clusters.setdefault(r["item_id"], r["cluster"]) != r["cluster"]:
                result["outcome"] = "insufficient_evidence"
                result["reasons"].append(f"item {r['item_id']} has conflicting cluster ids")
                return result
    all_items = {r["item_id"] for r in records}
    samples: dict[tuple[str, str], set[int]] = {}
    for r in records:
        samples.setdefault((r["arm"], r["item_id"]), set()).add(r["sample"])
    unbalanced = sorted(i for i in all_items if samples.get((base_id, i)) != samples.get((cid, i)))
    paired = all_items - set(unbalanced)
    # Registered design: exact item set, and every (arm, item, sample) record present.
    design_problems: list[str] = []
    expected_sha = study.raw["task"]["dataset"].get("expected_items_sha256")
    spi = study.analysis.get("samples_per_item")
    if expected_sha is None:
        design_problems.append("task.dataset.expected_items_sha256 not registered")
    elif item_set_sha256(all_items) != expected_sha:
        design_problems.append(
            "evaluated items differ from the registered item set (missing, extra or "
            "substituted items)"
        )
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
    target_items = {r["item_id"] for r in records if tag in r.get("tags", [])} & paired
    items = {"all": paired, "target_slice": target_items}
    result["unit"] = unit
    result["items"] = {
        "paired": len(paired),
        "target_slice": len(target_items),
        "unbalanced_excluded": unbalanced,
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
        result["rule_outcome_before_design_check"] = rule
        rule = "inconclusive"
        result["reasons"] += [f"design: {x}" for x in design_problems]
        result["reasons"].append("incomplete or unbalanced evidence: outcome is inconclusive")
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
