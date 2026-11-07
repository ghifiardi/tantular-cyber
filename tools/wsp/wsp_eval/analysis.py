"""Paired, problem-level analysis of run records.

* Per arm: requests, graded / ungraded / error counts, accuracy, token-limit hits and
  output length. Nothing is silently removed.
* Paired: per-item score = mean correctness over that item's samples; the candidate-minus-
  reference difference is averaged over items, and a percentile bootstrap that resamples
  **items** (clustering all repeated samples of an item) gives the 95% interval. With one
  sample per item an exact McNemar test on discordant items is reported as well.
* Missing outcomes follow the protocol's ``missing_policy``: ``exclude_item`` drops an item
  from the paired estimate if any of its samples in either arm is not graded (and lists it);
  ``count_as_incorrect`` scores ungraded/error samples as 0. Both are reported counts.

Pairing is checked, not assumed: duplicate (arm, item, sample) records raise, and an item
whose two arms do not have the same sample indices is excluded and listed as unbalanced.

The verdict is descriptive. Without a ``min_meaningful_effect`` (a draft protocol) it is
``descriptive_only_min_effect_not_set``. With one, the decision rule is applied only if it was
preregistered (see :func:`preregistration`: protocol frozen, named, and frozen no later than
the run start); otherwise the verdict is ``descriptive_only_not_preregistered`` and the rule's
outcome is shown separately as ``unregistered_rule_outcome``, never as the verdict.
``criteria_met_pending_human_review`` means the preregistered statistical criterion was met
on these records; it does not promote or deploy anything and says nothing about security
regressions, which are adjudicated separately.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from statistics import mean, median
from typing import Any

from .protocol import missing_policy


def mcnemar_exact(b: int, c: int) -> float | None:
    """Two-sided exact McNemar p-value from discordant counts (same formula as the
    production comparison module; reimplemented so this harness stays independent)."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return round(min(1.0, 2 * tail), 4)


def percentile_bootstrap(
    diffs: Sequence[float], resamples: int, seed: int, level: float = 0.95
) -> tuple[float | None, float | None]:
    if not diffs or resamples < 1:
        return None, None
    rng = random.Random(seed)  # noqa: S311 - seeded statistical resampling, not security
    n = len(diffs)
    stats = sorted(mean(rng.choices(diffs, k=n)) for _ in range(resamples))
    alpha = (1 - level) / 2
    lo = stats[max(0, math.floor(alpha * (resamples - 1)))]
    hi = stats[min(resamples - 1, math.ceil((1 - alpha) * (resamples - 1)))]
    return round(lo, 6), round(hi, 6)


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)


def preregistration(protocol_raw: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Was the analysis plan fixed before this run started? Purely from recorded fields."""
    reasons = []
    if not protocol_raw.get("frozen"):
        reasons.append("protocol not frozen")
    if not manifest.get("protocol_frozen"):
        reasons.append("run manifest does not record a frozen protocol")
    frozen_at, started = protocol_raw.get("frozen_at"), manifest.get("started_at")
    if not frozen_at:
        reasons.append("frozen_at not recorded")
    if not started:
        reasons.append("run start time not recorded")
    if frozen_at and started and _parse_ts(frozen_at) > _parse_ts(started):
        reasons.append("protocol frozen after the run started")
    return {
        "preregistered": not reasons,
        "frozen_by": protocol_raw.get("frozen_by"),
        "frozen_at": frozen_at,
        "run_started_at": started,
        "reasons": reasons,
    }


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [r for r in rows if r["status"] == "graded"]
    responded = [r for r in rows if r["status"] != "error"]
    limit = [r["hit_token_limit"] for r in responded if r["hit_token_limit"] is not None]
    tokens = [r["output_tokens"] for r in responded if r["output_tokens"] is not None]
    errors: dict[str, int] = defaultdict(int)
    for r in rows:
        if r["status"] == "error":
            errors[r["error_class"]] += 1
    return {
        "requests": len(rows),
        "graded": len(graded),
        "ungraded": sum(r["status"] == "ungraded" for r in rows),
        "errors": len(rows) - len(responded),
        "error_classes": dict(sorted(errors.items())),
        "accuracy_of_graded": round(mean(r["correct"] for r in graded), 6) if graded else None,
        "accuracy_of_requests": round(sum(bool(r["correct"]) for r in rows) / len(rows), 6)
        if rows
        else None,
        "token_limit_hits": sum(limit),
        "token_limit_assessed": len(limit),
        "token_limit_rate": round(sum(limit) / len(limit), 6) if limit else None,
        "output_tokens_mean": round(mean(tokens), 2) if tokens else None,
        "output_tokens_median": median(tokens) if tokens else None,
    }


def analyze(
    records: Sequence[dict[str, Any]],
    protocol_analysis: dict[str, Any],
    *,
    synthetic: bool,
    prereg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """``prereg`` is :func:`preregistration` output; without it the rule is not binding."""
    ref = protocol_analysis["reference_arm"]
    cand = protocol_analysis["candidate_arm"]
    policy = missing_policy(protocol_analysis)
    raw_effect = protocol_analysis["min_meaningful_effect"]
    min_effect = None if raw_effect is None else float(raw_effect)

    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, int]] = set()
    for r in records:
        key = (r["arm"], r["item_id"], r["sample"])
        if key in seen:
            raise ValueError(f"duplicate record for arm/item/sample {key}")
        seen.add(key)
        by_arm[r["arm"]].append(r)
        cells[(r["arm"], r["item_id"])].append(r)
    if ref not in by_arm or cand not in by_arm:
        raise ValueError("records do not contain both analysis arms")

    items = sorted({r["item_id"] for r in records})
    samples = {len(cells[(a, i)]) for a in (ref, cand) for i in items}
    diffs: list[float] = []
    excluded: list[str] = []
    unbalanced: list[str] = []
    per_item: list[dict[str, Any]] = []
    for item in items:
        a, b = cells.get((ref, item), []), cells.get((cand, item), [])
        if not a or not b:
            excluded.append(item)
            continue
        if {r["sample"] for r in a} != {r["sample"] for r in b}:
            excluded.append(item)
            unbalanced.append(item)
            continue
        if policy == "exclude_item" and any(r["status"] != "graded" for r in a + b):
            excluded.append(item)
            continue
        sa = mean(bool(r["correct"]) for r in a)
        sb = mean(bool(r["correct"]) for r in b)
        diffs.append(sb - sa)
        per_item.append(
            {"item_id": item, "reference": sa, "candidate": sb, "diff": round(sb - sa, 6)}
        )

    lo, hi = percentile_bootstrap(
        diffs,
        int(protocol_analysis["bootstrap_resamples"]),
        int(protocol_analysis["bootstrap_seed"]),
    )
    point = round(mean(diffs), 6) if diffs else None
    mcnemar = None
    if samples == {1} and per_item:
        b_only = sum(1 for p in per_item if p["reference"] and not p["candidate"])
        c_only = sum(1 for p in per_item if p["candidate"] and not p["reference"])
        mcnemar = {
            "reference_only": b_only,
            "candidate_only": c_only,
            "p_value": mcnemar_exact(b_only, c_only),
        }

    rule_outcome = None
    if point is None or lo is None or hi is None:
        verdict = "no_data"
    elif min_effect is None:
        verdict = "descriptive_only_min_effect_not_set"
    else:
        if lo > 0 and point >= min_effect:
            rule_outcome = "criteria_met_pending_human_review"
        elif hi < 0:
            rule_outcome = "candidate_worse"
        else:
            rule_outcome = "inconclusive"
        verdict = rule_outcome
        if not (prereg and prereg.get("preregistered")):
            verdict = "descriptive_only_not_preregistered"
    prereg_out = dict(prereg) if prereg else {"preregistered": False, "reasons": ["not assessed"]}

    return {
        "evidence_class": "synthetic_fixture_not_a_measurement" if synthetic else "measured",
        "reference_arm": ref,
        "candidate_arm": cand,
        "primary_endpoint": protocol_analysis["primary_endpoint"],
        "missing_policy": policy,
        "arms": {arm: _arm_summary(rows) for arm, rows in sorted(by_arm.items())},
        "paired": {
            "items_total": len(items),
            "items_paired": len(diffs),
            "items_excluded": excluded,
            "items_unbalanced": unbalanced,
            "samples_per_item": sorted(samples),
            "mean_difference": point,
            "ci95_bootstrap": [lo, hi],
            "bootstrap_resamples": int(protocol_analysis["bootstrap_resamples"]),
            "bootstrap_seed": int(protocol_analysis["bootstrap_seed"]),
            "min_meaningful_effect": min_effect,
            "mcnemar_exact": mcnemar,
        },
        "per_item": per_item,
        "verdict": verdict,
        "preregistration": prereg_out,
        # The decision rule's outcome when it is not binding; None when it is the verdict.
        "unregistered_rule_outcome": rule_outcome if verdict != rule_outcome else None,
        # Limitations the protocol requires every report to carry (e.g. a checkpoint
        # substitution); copied verbatim, never summarized.
        "limitations": list(protocol_analysis.get("required_limitations", [])),
        "decision_note": (
            "Descriptive only. No checkpoint is promoted by this harness; a human release "
            "owner records any decision in the run ledger after security adjudication."
        ),
    }
