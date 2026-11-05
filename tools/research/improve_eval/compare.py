"""Compare each candidate with the unchanged baseline under the study's declared rules.

Gates, in order; the first that fails decides the candidate's outcome:

1. ``insufficient_evidence``: no single verified evidence run contains both the baseline and
   the candidate (comparisons are always paired within one run), or a cluster-level study's
   records lack cluster ids.
2. ``baseline_changed_not_interpreted``: the baseline's loaded artifact (the registered file,
   when named) or configuration differs from the registration, or is not recorded.
3. ``candidate_provenance_mismatch``: the same check for the candidate.
4. ``prerequisite_missing``: an intervention-specific precondition is unmet (a weight edit
   without a ``verify-edit`` report matched to this evidence; a triage model calibrated on the
   evaluation dataset).
5. Metrics are computed. Constraints and the target effect are evaluated; then
   ``constraint_violated`` > ``regression_on_target`` >
   ``meets_declared_criteria_pending_human_decision`` > ``inconclusive``.
6. If the study was not frozen before the evidence run started, the outcome is replaced by
   ``descriptive_only_not_preregistered`` and the rule outcome is kept beside it.

Claims are only as strong as the intervals (``metrics.py``): the target improvement must have
its whole two-sided interval above zero and a point estimate at least the declared minimum;
constraints are judged on one-sided bounds. Units are items or, if declared, clusters.
Nothing is adjusted for testing several candidates and constraints; the report says so.

``meets_declared_criteria_pending_human_decision`` is the strongest outcome. It promotes or
deploys nothing; a person records a decision (``decision.py``), and any deployment needs its
own evaluation and release process outside this workflow.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from wsp_eval.hashing import sha256_json

from .evidence import Evidence
from .metrics import (
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
        matched = ev.notes.get("edit_verification", {})
        if matched.get("candidate") != cand["id"] or matched.get("status") != "matched":
            return ["no verify-edit report matched to this candidate's evidence"]
    if iv["type"] == "soc_triage_model":
        cal = iv["calibration"].get("dataset")
        if cal == study.raw["task"]["dataset"]:
            return ["operating point calibrated on the evaluation dataset"]
    return []


def _values(metric: str, records: Sequence[dict[str, Any]], arm: str, items: set[str]):
    return item_values(metric, group(records, arm, items))


def _summary(values: dict, keys: Sequence[str]) -> dict[str, Any]:
    num, den, val = ratio(values, keys)
    return {"value": val, "numerator": num, "denominator": den}


def _improvement(study: Study, metric: str, base, cand, keys, clusters) -> dict[str, Any]:
    _, better, pairable = METRICS[metric]
    a = study.analysis
    if pairable:
        return paired_improvement(base, cand, keys, better, a["confidence"], clusters)
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


def _constraint(study, k, records, cand_id, items, clusters) -> dict[str, Any]:
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
        "candidate": _summary(cand, keys),
        "baseline": _summary(base, keys),
    }
    if k["kind"] == "absolute":
        side = "upper" if k["op"] == "<=" else "lower"
        b = one_sided_bound(cand, keys, side, study.analysis["confidence"], clusters)
        point, bound, limit = out["candidate"]["value"], b["bound"], k.get("bound")
        out.update(op=k["op"], bound=limit, confidence_bound=b)
        if limit is None:
            out["status"] = "not_declared"
        elif point is None or bound is None:
            out["status"] = "not_evaluated"  # e.g. nothing selected: the fraction is undefined
        elif (point > limit) if k["op"] == "<=" else (point < limit):
            out["status"] = "violated"
        elif (bound <= limit) if k["op"] == "<=" else (bound >= limit):
            out["status"] = "satisfied"
        else:
            out["status"] = "inconclusive"
    elif k["kind"] == "non_regression":
        imp = _improvement(study, metric, base, cand, keys, clusters)
        margin = k.get("margin")
        out.update(margin=margin, improvement=imp)
        lo = imp["ci"][0]
        if margin is None:
            out["status"] = "not_declared"
        elif imp["point"] is None or lo is None:
            out["status"] = "not_evaluated"
        elif imp["point"] < -margin:
            out["status"] = "violated"
        elif lo >= -margin:
            out["status"] = "satisfied"
        else:
            out["status"] = "inconclusive"
    else:
        losses = net_paired_losses(base, cand, keys, better)
        limit = k.get("max_losses")
        out.update(max_losses=limit, paired=losses)
        if limit is None:
            out["status"] = "not_declared"
        elif losses is None:
            out["status"] = "not_evaluated"
        else:
            out["status"] = "satisfied" if losses["net_losses"] <= limit else "violated"
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
        clusters = {r["item_id"]: r["cluster"] for r in records}
    all_items = {r["item_id"] for r in records}
    samples: dict[tuple[str, str], set[int]] = {}
    for r in records:
        samples.setdefault((r["arm"], r["item_id"]), set()).add(r["sample"])
    unbalanced = sorted(i for i in all_items if samples.get((base_id, i)) != samples.get((cid, i)))
    paired = all_items - set(unbalanced)
    tag = study.target["slice_tag"]
    target_items = {r["item_id"] for r in records if tag in r.get("tags", [])} & paired
    items = {"all": paired, "target_slice": target_items}
    result["unit"] = unit
    result["items"] = {
        "paired": len(paired),
        "target_slice": len(target_items),
        "unbalanced_excluded": unbalanced,
    }
    if study.task_kind == "alert_triage":
        result["breakdown"] = {
            arm: triage_breakdown(group(records, arm, paired)) for arm in (base_id, cid)
        }

    metric = study.target["metric"]
    keys = sorted(target_items)
    base = _values(metric, records, base_id, target_items)
    candv = _values(metric, records, cid, target_items)
    improvement = _improvement(study, metric, base, candv, keys, clusters)
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
        _constraint(study, k, records, cid, items, clusters) for k in study.constraints
    ]

    statuses = [c["status"] for c in result["constraints"]]
    lo, hi = improvement["ci"]
    if "violated" in statuses:
        rule = "constraint_violated"
        result["reasons"] += [
            f"constraint {c['id']} violated"
            for c in result["constraints"]
            if c["status"] == "violated"
        ]
    elif hi is not None and hi < 0:
        rule = "regression_on_target"
        result["reasons"].append("target metric is worse than baseline (interval below zero)")
    elif (
        min_effect is not None
        and lo is not None
        and lo > 0
        and improvement["point"] >= min_effect
        and all(s == "satisfied" for s in statuses)
    ):
        rule = "meets_declared_criteria_pending_human_decision"
    else:
        rule = "inconclusive"
        if min_effect is None:
            result["reasons"].append("min_meaningful_effect not declared")
        if lo is None:
            result["reasons"].append(
                f"target effect could not be estimated ({improvement['method']})"
            )
        elif not lo > 0:
            result["reasons"].append("target improvement interval includes zero")
        elif improvement["point"] < (min_effect or 0):
            result["reasons"].append("target improvement below min_meaningful_effect")
        for c in result["constraints"]:
            if c["status"] != "satisfied":
                result["reasons"].append(f"constraint {c['id']} {c['status']}")
    prereg = preregistration(study, ev)
    result["preregistration"] = prereg
    result["rule_outcome"] = rule
    if prereg["preregistered"]:
        result["outcome"] = rule
    else:
        result["outcome"] = "descriptive_only_not_preregistered"
        result["reasons"] += prereg["reasons"]
    return result
