"""Reload/drift policy for stochastic inference (docs/wsp/exp8-replication-protocol.md, §6).

Two separate questions, never merged into one number:

1. **Identity** (exact, no tolerance): is the reload the *same thing* as the reference? The
   declared checkpoint, inference and engine settings, the artifacts each generator reports
   having loaded, and the rendered prompt per item and sample must be identical. Each check is
   ``pass``, ``fail`` or ``not_recorded``. ``not_recorded`` is not a pass, and identity is
   ``complete`` only when nothing is unrecorded. A ``fail`` means any output difference cannot
   be attributed to sampling.
2. **Reproducibility** (stochastic, descriptive unless approved): how different are the
   outputs? Every (item, sample) pair is classified (identical output, different output with
   the same grade, grade changed, error on one or both sides). Arm-level accuracy, token-limit
   and error-rate differences (reload minus reference) get item-clustered bootstrap intervals.
   Exact output identity is reported but never required: batched GPU sampling is not
   guaranteed bitwise reproducible, even with fixed seeds.

Records are checked, not assumed, as in the paired analysis: a duplicate (arm, item, sample)
record for either arm raises ``ValueError``, and an item whose two arms have different sample
sets is excluded from the metrics and listed in ``items_unbalanced``. Its samples still appear
in the pair categories (the missing side counts as ``unpaired``). Error rows that carry the
adapter-reported prompt hash (abnormal or inconsistent finishes) take part in the
rendered-prompt identity check; rows without one count as ``not_recorded``.

A tolerance is applied only when the protocol was frozen (named, with ``frozen_at``) no later
than the run's start, ``drift.tolerance.status`` is ``approved`` with named approver and
bounds, and the approval timestamp is not later than the run's start. Without an approved
tolerance the verdict is ``descriptive_only_tolerance_not_approved``; with one that fails the
timing or freeze conditions it is ``tolerance_not_applicable``. Bounds are equivalence-style:
a metric is ``within`` only if its whole 95% interval lies inside ±bound, ``exceeds`` if the
point estimate lies outside, and otherwise ``inconclusive``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from statistics import mean
from typing import Any

from .analysis import percentile_bootstrap, preregistration
from .protocol import TOLERANCE_METRICS, Protocol

PAIR_CATEGORIES = (
    "identical_output",
    "different_output_same_grade",
    "different_output_grade_changed",
    "one_side_error",
    "both_error",
    "unpaired",
)
_GENERATOR_KEYS = ("provider", "model", "revision", "engine", "engine_version", "artifacts_sha256")


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)


def _check_unique(records: Sequence[dict[str, Any]], arms: tuple[str, str]) -> None:
    seen: set[tuple[str, str, int]] = set()
    for r in records:
        if r["arm"] not in arms:
            continue
        key = (r["arm"], r["item_id"], r["sample"])
        if key in seen:
            raise ValueError(f"duplicate record for arm/item/sample {key}")
        seen.add(key)


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"check": name, "status": status, "detail": detail}


def identity_report(
    protocol: Protocol,
    manifest: dict[str, Any],
    records: Sequence[dict[str, Any]],
    reference: str,
    reload: str,
) -> dict[str, Any]:
    _check_unique(records, (reference, reload))
    ref, rel = protocol.arm(reference), protocol.arm(reload)
    checks = [
        _check(
            "declared_checkpoint",
            "pass" if ref.checkpoint == rel.checkpoint else "fail",
            "protocol checkpoint declarations",
        ),
        _check(
            "inference_settings",
            "pass" if ref.inference == rel.inference else "fail",
            "protocol inference settings",
        ),
        _check(
            "engine_settings",
            "pass" if ref.engine == rel.engine else "fail",
            "protocol engine settings" if ref.engine else "no engine settings declared",
        ),
    ]
    if not ref.engine:
        checks[-1]["status"] = "not_recorded"

    gens = {a["id"]: a.get("generator", {}) for a in manifest.get("arms", [])}
    ga, gb = gens.get(reference, {}), gens.get(reload, {})
    for key in _GENERATOR_KEYS:
        va, vb = ga.get(key), gb.get(key)
        if va in (None, {}) or vb in (None, {}):
            checks.append(_check(f"loaded_{key}", "not_recorded", "generator did not report it"))
        else:
            checks.append(_check(f"loaded_{key}", "pass" if va == vb else "fail", "describe()"))
    la, lb = ga.get("load_id"), gb.get("load_id")
    if la is None or lb is None:
        checks.append(_check("separate_load", "not_recorded", "no load_id reported"))
    else:
        checks.append(
            _check(
                "separate_load",
                "pass" if la != lb else "fail",
                "reload must be a separate model load" if la == lb else "distinct load ids",
            )
        )

    prompts: dict[str, dict[tuple[str, int], Any]] = {reference: {}, reload: {}}
    for r in records:
        if r["arm"] in prompts:
            prompts[r["arm"]][(r["item_id"], r["sample"])] = r.get("rendered_prompt_sha256")
    keys = sorted(set(prompts[reference]) | set(prompts[reload]))
    unrecorded = [k for k in keys if not prompts[reference].get(k) or not prompts[reload].get(k)]
    differ = [
        k
        for k in keys
        if k not in unrecorded and prompts[reference].get(k) != prompts[reload].get(k)
    ]
    if differ:
        checks.append(_check("rendered_prompts", "fail", f"{len(differ)} of {len(keys)} differ"))
    elif unrecorded:
        checks.append(
            _check("rendered_prompts", "not_recorded", f"{len(unrecorded)} of {len(keys)} missing")
        )
    else:
        checks.append(_check("rendered_prompts", "pass", f"{len(keys)} identical"))

    statuses = {c["status"] for c in checks}
    return {
        "reference_arm": reference,
        "reload_arm": reload,
        "checks": checks,
        "identity": "fail" if "fail" in statuses else "pass",
        "complete": "not_recorded" not in statuses,
    }


def _item_rates(
    rows: list[dict[str, Any]], key: str, missing_policy: str
) -> tuple[float | None, bool]:
    """Item-level mean of a 0/1 outcome; returns (value, has_missing)."""
    if key == "error":
        return mean(r["status"] == "error" for r in rows), False
    if key == "correct":
        missing = any(r["status"] != "graded" for r in rows)
        if missing and missing_policy == "exclude_item":
            return None, True
        return mean(bool(r["correct"]) for r in rows), missing
    vals = [r["hit_token_limit"] for r in rows if r["status"] != "error"]
    vals = [v for v in vals if v is not None]
    return (mean(bool(v) for v in vals) if vals else None), len(vals) != len(rows)


def reproducibility_report(
    protocol: Protocol, records: Sequence[dict[str, Any]], reference: str, reload: str
) -> dict[str, Any]:
    _check_unique(records, (reference, reload))
    analysis = protocol.analysis
    missing_policy = analysis.get("missing_policy", "exclude_item")
    resamples = int(analysis["bootstrap_resamples"])
    seed = int(analysis["bootstrap_seed"])

    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    pairs: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    errors: dict[str, dict[str, int]] = {reference: defaultdict(int), reload: defaultdict(int)}
    for r in records:
        if r["arm"] not in (reference, reload):
            continue
        cells[(r["arm"], r["item_id"])].append(r)
        pairs[(r["item_id"], r["sample"])][r["arm"]] = r
        if r["status"] == "error":
            errors[r["arm"]][r["error_class"]] += 1

    categories: dict[str, int] = dict.fromkeys(PAIR_CATEGORIES, 0)
    for pair in pairs.values():
        a, b = pair.get(reference), pair.get(reload)
        if a is None or b is None:
            categories["unpaired"] += 1
        elif a["status"] == "error" and b["status"] == "error":
            categories["both_error"] += 1
        elif "error" in (a["status"], b["status"]):
            categories["one_side_error"] += 1
        elif a["output_sha256"] == b["output_sha256"]:
            categories["identical_output"] += 1
        elif a["correct"] == b["correct"]:
            categories["different_output_same_grade"] += 1
        else:
            categories["different_output_grade_changed"] += 1
    comparable = sum(
        categories[c]
        for c in (
            "identical_output",
            "different_output_same_grade",
            "different_output_grade_changed",
        )
    )

    items = sorted({i for (_, i) in cells})
    unbalanced = [
        i
        for i in items
        if cells.get((reference, i))
        and cells.get((reload, i))
        and {r["sample"] for r in cells[(reference, i)]}
        != {r["sample"] for r in cells[(reload, i)]}
    ]
    metrics: dict[str, Any] = {}
    for name, key in (
        ("accuracy_diff", "correct"),
        ("token_limit_rate_diff", "hit_token_limit"),
        ("error_rate_diff", "error"),
    ):
        diffs, excluded, with_missing = [], [], 0
        for item in items:
            ra, rb = cells.get((reference, item)), cells.get((reload, item))
            if not ra or not rb or item in unbalanced:
                excluded.append(item)
                continue
            va, ma = _item_rates(ra, key, missing_policy)
            vb, mb = _item_rates(rb, key, missing_policy)
            if va is None or vb is None:
                excluded.append(item)
                continue
            with_missing += ma or mb
            diffs.append(vb - va)
        lo, hi = percentile_bootstrap(diffs, resamples, seed)
        metrics[name] = {
            "point": round(mean(diffs), 6) if diffs else None,
            "ci95_bootstrap": [lo, hi],
            "items_used": len(diffs),
            "items_excluded": excluded,
            "items_with_missing_samples": with_missing,
        }

    return {
        "reference_arm": reference,
        "reload_arm": reload,
        "direction": "reload minus reference",
        "pairs": len(pairs),
        "pair_categories": categories,
        "exact_output_match_rate": round(categories["identical_output"] / comparable, 6)
        if comparable
        else None,
        "errors_by_arm": {arm: dict(sorted(e.items())) for arm, e in errors.items()},
        "items_unbalanced": unbalanced,
        "metrics": metrics,
        "missing_policy": missing_policy,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }


def tolerance_verdict(
    protocol: Protocol,
    manifest: dict[str, Any],
    identity: dict[str, Any],
    repro: dict[str, Any],
) -> dict[str, Any]:
    if identity["identity"] == "fail":
        return {
            "verdict": "identity_failed_reproducibility_not_interpreted",
            "reasons": [c["check"] for c in identity["checks"] if c["status"] == "fail"],
        }
    tol = protocol.drift.get("tolerance") or {}
    if tol.get("status") != "approved":
        return {
            "verdict": "descriptive_only_tolerance_not_approved",
            "reasons": ["no approved tolerance in the protocol; differences are reported only"],
        }
    reasons = list(preregistration(protocol.raw, manifest)["reasons"])
    started = manifest.get("started_at")
    if started and _parse_ts(tol["approved_at"]) > _parse_ts(started):
        reasons.append("tolerance approved after the run started")
    if reasons:
        return {"verdict": "tolerance_not_applicable", "reasons": reasons}

    per_metric = {}
    for bound_key, metric in zip(
        TOLERANCE_METRICS,
        ("accuracy_diff", "token_limit_rate_diff", "error_rate_diff"),
        strict=True,
    ):
        bound = float(tol[bound_key])
        m = repro["metrics"][metric]
        lo, hi = m["ci95_bootstrap"]
        if m["point"] is None or lo is None or hi is None:
            per_metric[metric] = "no_data"
        elif abs(m["point"]) > bound:
            per_metric[metric] = "exceeds"
        elif -bound <= lo and hi <= bound:
            per_metric[metric] = "within"
        else:
            per_metric[metric] = "inconclusive"
    values = set(per_metric.values())
    if "exceeds" in values:
        verdict = "exceeds_tolerance"
    elif values == {"within"}:
        verdict = (
            "within_tolerance" if identity["complete"] else "within_tolerance_identity_incomplete"
        )
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "per_metric": per_metric,
        "approved_by": tol["approved_by"],
        "approved_at": tol["approved_at"],
    }


def drift_report(
    protocol: Protocol, manifest: dict[str, Any], records: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    reference = next(a.id for a in protocol.arms if a.role == "reference")
    out = []
    for arm in protocol.arms:
        if arm.role != "no_edit_reload":
            continue
        identity = identity_report(protocol, manifest, records, reference, arm.id)
        repro = reproducibility_report(protocol, records, reference, arm.id)
        out.append(
            {
                "identity": identity,
                "reproducibility": repro,
                "tolerance": tolerance_verdict(protocol, manifest, identity, repro),
            }
        )
    return out
