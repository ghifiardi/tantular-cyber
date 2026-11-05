"""Task metrics as per-item (numerator, denominator) pairs, with confidence bounds whose
method status is always reported.

Every metric is a ratio of sums over items, so it can be computed per arm, bounded and
compared item-paired between arms. Definitions (``item_values``):

* ``accuracy`` (accuracy tasks): per item, the mean correctness over its samples. Ungraded and
  error samples count as incorrect (they are also reported separately). Denominator 1.
* ``error_rate``: per item, the fraction of samples that errored. Denominator 1.
* ``benign_escalation_fraction`` (triage): numerator = selected alerts labelled benign;
  denominator = selected alerts with a label. Unlabelled selections and errors are outside
  the denominator and are counted in :func:`triage_breakdown`. If an arm selects nothing
  labelled, the value is undefined (None), never 0. The denominator depends on the arm, so
  this metric is not pairable.
* ``threat_miss_rate`` (triage): numerator = alerts labelled threat that were suppressed or
  errored (nobody looks at them); denominator = alerts labelled threat. Deferred threats
  reach an analyst and are not misses.
* ``defer_rate`` (triage): numerator = alerts deferred **or errored** (both land on an analyst
  queue, so errors cannot make an arm look cheaper); denominator = all alerts.
* ``escalation_rate`` (triage): numerator = alerts selected (escalated); denominator = all
  alerts. This is escalation *volume*, which the benign-escalation fraction (a precision
  among selections) and the defer rate do not measure; a frozen triage study must cap it as
  a separately declared capacity constraint.

Bounds and interval ends are rounded *outward* to six decimals (upper bounds up, lower bounds
down), so rounding never moves a bound across a limit in the candidate's favour.

**Estimands.** Every bound or interval names the quantity it targets (``estimand``), and its
point estimate is computed for that same quantity:

* ``pooled_ratio``: sum of numerators / sum of denominators over the units in scope;
* ``mean_paired_difference_per_item``: the mean over items of (candidate - baseline) per-item
  value, signed so that positive means improvement. For items with denominator 1 this equals
  the difference of the pooled ratios;
* ``mean_paired_difference_per_cluster``: the same, averaged first within each cluster and then
  across clusters (cluster-weighted). It can differ in size and sign from the per-item mean,
  which is reported beside it as ``item_weighted_point``;
* ``difference_in_pooled_ratios``: for non-pairable metrics.

**Method status.** Only two methods can support a passing result (``BINDING_STATUSES``). Both
also require ``single_sample=True``, which the caller sets from the *design* (exactly one sample
per arm and item), never from the observed values: multi-sample items whose means happen to be
0 or 1 are not single-sample Bernoulli outcomes.

* ``exact``: one-sided Clopper-Pearson on whole counts (k of n). This assumes the n units are
  independent Bernoulli trials with a common probability, i.e. independent alerts or items.
* ``valid_by_bonferroni``: the paired interval for binary, single-sample items with
  denominator 1. Each item's signed difference is -1, 0 or +1. W (items improved) and L
  (items worsened) are each Binomial(n, p+) and Binomial(n, p-) under independent,
  identically distributed items, so one-sided Clopper-Pearson bounds on p+ and p- are exact.
  Combining four such bounds, each at alpha/4, gives simultaneous coverage of at least
  1 - alpha by Bonferroni's inequality, which holds whatever the dependence between W and
  L. Hence [lo(p+) - hi(p-), hi(p+) - lo(p-)] covers p+ - p- with probability at least
  1 - alpha. It is valid, and conservative, under those assumptions only.

Everything else is non-binding and is reported as descriptive:

* ``unreviewed_heuristic``: Clopper-Pearson applied to rounded fractional or effective-n counts
  (clusters, multi-sample items). No coverage guarantee has been shown for these, and they are
  called neither exact nor conservative;
* ``approximate``: the percentile bootstrap (no finite-sample guarantee);
* ``cluster_method_unreviewed``: any interval whose unit is a cluster. A reviewed clustered
  method and its estimand have not been specified.

A degenerate (zero-width) bootstrap is reported as not estimable.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import mean
from typing import Any

Values = dict[str, tuple[float, float]]  # item id -> (numerator, denominator)


def item_values(metric: str, rows_by_item: Mapping[str, Sequence[dict[str, Any]]]) -> Values:
    out: Values = {}
    for item, rows in rows_by_item.items():
        if metric == "accuracy":
            out[item] = (mean(r["status"] == "graded" and r["correct"] is True for r in rows), 1.0)
        elif metric == "error_rate":
            out[item] = (mean(r["status"] == "error" for r in rows), 1.0)
        else:
            (r,) = rows  # triage: one decision per alert and arm
            label, dec, err = r.get("label"), r.get("decision"), r["status"] == "error"
            if metric == "benign_escalation_fraction":
                den = float(dec == "select" and label in ("threat", "benign"))
                out[item] = (float(den and label == "benign"), den)
            elif metric == "threat_miss_rate":
                den = float(label == "threat")
                out[item] = (float(den and (err or dec == "suppress")), den)
            elif metric == "defer_rate":
                out[item] = (float(err or dec == "defer"), 1.0)
            elif metric == "escalation_rate":
                out[item] = (float(dec == "select"), 1.0)
            else:
                raise ValueError(f"unknown metric {metric!r}")
    return out


def triage_breakdown(rows_by_item: Mapping[str, Sequence[dict[str, Any]]]) -> dict[str, int]:
    """Counts of decision x label for one arm: every triage metric can be rebuilt from these."""
    c: Counter[str] = Counter()
    for rows in rows_by_item.values():
        for r in rows:
            dec = "error" if r["status"] == "error" else r.get("decision")
            c[f"{dec}/{r.get('label') or 'unlabelled'}"] += 1
    keys = [
        f"{d}/{lab}"
        for d in ("select", "defer", "suppress", "error")
        for lab in ("threat", "benign", "unlabelled")
    ]
    return {k: c.get(k, 0) for k in keys}


def group(records: Iterable[dict[str, Any]], arm: str, items: set[str] | None = None) -> dict:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r["arm"] == arm and (items is None or r["item_id"] in items):
            by[r["item_id"]].append(r)
    return dict(by)


def ratio(values: Values, items: Iterable[str] | None = None) -> tuple[float, float, float | None]:
    keys = list(values) if items is None else [i for i in items if i in values]
    num = sum(values[i][0] for i in keys)
    den = sum(values[i][1] for i in keys)
    return num, den, (num / den if den else None)


def units_of(keys: Iterable[str], clusters: Mapping[str, str] | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        out[clusters[k] if clusters is not None else k].append(k)
    return dict(out)


def _binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(min(k, n) + 1):
        total += math.exp(
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
        )
    return min(1.0, total)


def clopper_pearson(k: int, n: int, confidence: float, side: str) -> float:
    """One-sided exact bound: ``side`` 'upper' or 'lower'."""
    if n <= 0:
        return 1.0 if side == "upper" else 0.0
    alpha = 1 - confidence
    if side == "upper":
        if k >= n:
            return 1.0
        lo, hi = k / n, 1.0
        for _ in range(100):  # largest p with P(X <= k) >= alpha
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if _binom_cdf(k, n, mid) > alpha else (lo, mid)
        return hi
    if k <= 0:
        return 0.0
    lo, hi = 0.0, k / n
    for _ in range(100):  # smallest p with P(X >= k) >= alpha
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if 1 - _binom_cdf(k - 1, n, mid) > alpha else (mid, hi)
    return lo


def outward(x: float, side: str) -> float:
    """Round to six decimals away from the claim: upper bounds up, lower bounds down."""
    scaled = x * 1e6
    r = math.ceil(scaled - 1e-6) if side == "upper" else math.floor(scaled + 1e-6)
    return r / 1e6


def _whole(x: float) -> bool:
    return float(x).is_integer()


BINDING_STATUSES = ("exact", "valid_by_bonferroni")


def one_sided_bound(
    values: Values,
    items: Sequence[str],
    side: str,
    confidence: float,
    clusters: Mapping[str, str] | None = None,
    *,
    single_sample: bool = False,
) -> dict[str, Any]:
    """One-sided bound on the pooled ratio of the units in scope."""
    keys = [i for i in items if i in values and values[i][1] > 0]
    base = {"estimand": "pooled_ratio"}
    if not keys:
        return {**base, "bound": None, "method": "no_data", "status": "no_data", "units": 0}
    num = sum(values[i][0] for i in keys)
    den = sum(values[i][1] for i in keys)
    whole = all(_whole(values[i][0]) and _whole(values[i][1]) for i in keys)
    if single_sample and clusters is None and whole:
        b = clopper_pearson(int(num), int(den), confidence, side)
        return {
            **base,
            "bound": outward(b, side),
            "method": "clopper_pearson",
            "status": "exact",
            "units": int(den),
        }
    # Rounded effective-n Clopper-Pearson: descriptive only (no coverage guarantee shown).
    n = len(units_of(keys, clusters))
    k = num / den * n
    k_int = math.ceil(k - 1e-9) if side == "upper" else math.floor(k + 1e-9)
    b = clopper_pearson(k_int, n, confidence, side)
    return {
        **base,
        "bound": outward(b, side),
        "method": "rounded_effective_n_clopper_pearson",
        "status": "cluster_method_unreviewed" if clusters is not None else "unreviewed_heuristic",
        "units": n,
    }


def paired_improvement(
    base: Values,
    cand: Values,
    items: Sequence[str],
    better: int,
    confidence: float,
    clusters: Mapping[str, str] | None = None,
    *,
    single_sample: bool = False,
) -> dict[str, Any]:
    """Mean paired improvement (positive = better) over units, with its interval.

    The point estimate and the interval target the same estimand: the mean over units of the
    per-unit mean signed difference (per item, or per cluster when ``clusters`` is given).
    """
    keys = [i for i in items if i in base and i in cand and base[i][1] > 0]
    for i in keys:
        if base[i][1] != cand[i][1]:
            raise ValueError(f"item {i}: denominators differ; metric is not pairable")
    estimand = (
        "mean_paired_difference_per_cluster"
        if clusters is not None
        else "mean_paired_difference_per_item"
    )
    if not keys:
        return {
            "estimand": estimand,
            "point": None,
            "ci": [None, None],
            "units": 0,
            "method": "no_data",
            "status": "no_data",
        }
    per_unit = []
    for unit_keys in units_of(keys, clusters).values():
        d = [(cand[i][0] - base[i][0]) / base[i][1] * better for i in unit_keys]
        per_unit.append(mean(d))
    n = len(per_unit)
    point = round(mean(per_unit), 6)
    w = sum(max(d, 0.0) for d in per_unit)
    lo_s = sum(max(-d, 0.0) for d in per_unit)
    tail = 1 - (1 - confidence) / 4  # each of four one-sided bounds at alpha/4
    lo = clopper_pearson(math.floor(w + 1e-9), n, tail, "lower") - clopper_pearson(
        math.ceil(lo_s - 1e-9), n, tail, "upper"
    )
    hi = clopper_pearson(math.ceil(w - 1e-9), n, tail, "upper") - clopper_pearson(
        math.floor(lo_s + 1e-9), n, tail, "lower"
    )
    binary = (
        single_sample
        and clusters is None
        and all(
            base[i][1] == 1.0 and base[i][0] in (0.0, 1.0) and cand[i][0] in (0.0, 1.0)
            for i in keys
        )
    )
    if binary:
        method, status = "bonferroni_clopper_pearson", "valid_by_bonferroni"
    elif clusters is not None:
        method, status = "rounded_bonferroni_clopper_pearson", "cluster_method_unreviewed"
    else:
        method, status = "rounded_bonferroni_clopper_pearson", "unreviewed_heuristic"
    out = {
        "estimand": estimand,
        "point": point,
        "ci": [outward(lo, "lower"), outward(hi, "upper")],
        "units": n,
        "method": method,
        "status": status,
    }
    if clusters is not None:
        out["item_weighted_point"] = round(
            mean((cand[i][0] - base[i][0]) / base[i][1] * better for i in keys), 6
        )
    return out


def paired_difference_bootstrap(
    base: Values,
    cand: Values,
    items: Sequence[str],
    better: int,
    confidence: float,
    resamples: int,
    seed: int,
    clusters: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Difference in pooled ratios (positive = better) for arm-dependent denominators."""
    keys = [i for i in items if i in base and i in cand]
    bval, cval = ratio(base, keys)[2], ratio(cand, keys)[2]
    status = "cluster_method_unreviewed" if clusters is not None else "approximate"
    if bval is None or cval is None:
        return {
            "estimand": "difference_in_pooled_ratios",
            "point": None,
            "ci": [None, None],
            "units": 0,
            "method": "no_data",
            "status": "no_data",
        }
    groups = list(units_of(keys, clusters).values())
    rng = random.Random(seed)  # noqa: S311 - seeded statistical resampling, not security
    stats, skipped = [], 0
    for _ in range(resamples):
        pick = [i for g in rng.choices(groups, k=len(groups)) for i in g]
        b = sum(base[i][1] for i in pick)
        c = sum(cand[i][1] for i in pick)
        if not b or not c:
            skipped += 1
            continue
        stats.append(
            (sum(cand[i][0] for i in pick) / c - sum(base[i][0] for i in pick) / b) * better
        )
    stats.sort()
    alpha = (1 - confidence) / 2
    lo = hi = None
    method = "percentile_bootstrap"
    if stats and skipped <= resamples * 0.05:
        lo = outward(stats[max(0, math.floor(alpha * (len(stats) - 1)))], "lower")
        hi = outward(stats[min(len(stats) - 1, math.ceil((1 - alpha) * (len(stats) - 1)))], "upper")
        if lo == hi:
            lo = hi = None  # a zero-width interval is not evidence of certainty
            method = "percentile_bootstrap_degenerate"
    return {
        "estimand": "difference_in_pooled_ratios",
        "point": round((cval - bval) * better, 6),
        "ci": [lo, hi],
        "units": len(groups),
        "method": method,
        "status": status,
        "skipped_resamples": skipped,
    }


def net_paired_losses(
    base: Values, cand: Values, items: Sequence[str], better: int
) -> dict[str, int] | None:
    """Per-item wins/losses for a pairable metric (denominators equal per item)."""
    wins = losses = 0
    used = 0
    for i in items:
        if i not in base or i not in cand or base[i][1] == 0:
            continue
        if base[i][1] != cand[i][1]:
            raise ValueError(f"item {i}: denominators differ; metric is not pairable")
        used += 1
        d = (cand[i][0] - base[i][0]) * better
        wins += d > 0
        losses += d < 0
    if not used:
        return None
    return {"items": used, "wins": wins, "losses": losses, "net_losses": losses - wins}
