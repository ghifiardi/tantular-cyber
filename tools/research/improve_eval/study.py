"""Study registry: one unchanged baseline, one target failure, candidate interventions and
declared constraints, as a JSON document with a canonical hash.

A study is the preregistration. ``study_sha256`` is recorded in every evidence manifest, so
evidence collected under a different study text cannot be attached to this one. A frozen study
names ``frozen_by`` and ``frozen_at``, has no ``REPLACE`` placeholders and no undeclared bounds;
its decision rules are binding only for evidence whose run started no earlier than
``frozen_at``. An unfrozen (draft) study may leave bounds null; every result is then
descriptive and the report outcome is inconclusive.

Every declared limit (each constraint and the target's minimum effect) names its
``bound_source``: ``synthetic_fixture`` (an invented value, required in synthetic studies
and forbidden in real ones) or ``declared_by_owner``. The report prints the source next to
each value. Triage studies also state a ``validity_scope`` (e.g. one tenant and one data
source; results are not assumed to transfer). A frozen triage study must cap both the benign
escalation fraction and the defer rate, since a target such as missed threats can otherwise be
"fixed" by escalating or deferring everything.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wsp_eval.hashing import SHA256_RE, contains_placeholder, sha256_json

from .interventions import EDIT_VERIFICATION_STATUSES, TYPES

PSEUDONYM_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?Z$")
TASK_KINDS = ("accuracy", "alert_triage")
# metric -> (task kind, better direction: +1 higher is better, -1 lower is better, pairable)
# A metric is "pairable" when its per-item denominator does not depend on the arm, so per-item
# wins and losses are defined (needed for max_net_paired_losses).
METRICS: dict[str, tuple[str, int, bool]] = {
    "accuracy": ("accuracy", +1, True),
    "error_rate": ("*", -1, True),
    "benign_escalation_fraction": ("alert_triage", -1, False),
    "threat_miss_rate": ("alert_triage", -1, True),
    "defer_rate": ("alert_triage", -1, True),
    "escalation_rate": ("alert_triage", -1, True),
}
CONSTRAINT_KINDS = ("absolute", "non_regression", "max_net_paired_losses")
SCOPES = ("all", "target_slice")
UNITS = ("item", "cluster")
BOUND_SOURCES = ("synthetic_fixture", "declared_by_owner")
# Precision of escalations, analyst-queue load and escalation volume (capacity) are capped
# separately; none is inferred from another.
REQUIRED_TRIAGE_CAPS = ("benign_escalation_fraction", "defer_rate", "escalation_rate")
UNBALANCED_POLICIES = ("inconclusive", "exclude_listed_items")
LOSS_COUNTS = ("net", "gross")


class StudyError(ValueError):
    pass


@dataclass(frozen=True)
class Study:
    raw: dict[str, Any]
    sha256: str
    frozen: bool
    synthetic: bool
    task_kind: str
    baseline: dict[str, Any]
    target: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]
    constraints: tuple[dict[str, Any], ...]
    analysis: dict[str, Any]

    def candidate(self, cid: str) -> dict[str, Any]:
        for c in self.candidates:
            if c["id"] == cid:
                return c
        raise StudyError(f"unknown candidate {cid!r}")

    @property
    def arm_ids(self) -> tuple[str, ...]:
        return (self.baseline["id"], *(c["id"] for c in self.candidates))


def _need(d: Any, key: str, kind: type | tuple[type, ...], where: str) -> Any:
    if not isinstance(d, dict) or key not in d:
        raise StudyError(f"{where}: missing field {key!r}")
    v = d[key]
    kinds = kind if isinstance(kind, tuple) else (kind,)
    if not isinstance(v, kinds) or (isinstance(v, bool) and bool not in kinds):
        raise StudyError(f"{where}: field {key!r} has wrong type")
    return v


def _artifact(a: Any, where: str, frozen: bool) -> None:
    _need({"artifact": a}, "artifact", dict, where)
    _need(a, "name", str, f"{where}.artifact")
    if frozen and not (isinstance(a.get("file"), str) and a["file"]):
        # The digest must be checked against a named file, not against any loaded file.
        raise StudyError(f"{where}.artifact: a frozen study names the artifact file")
    sha = a.get("sha256")
    if sha is None:
        if frozen:
            raise StudyError(f"{where}.artifact: a frozen study needs the artifact sha256")
    elif not (isinstance(sha, str) and SHA256_RE.match(sha)):
        raise StudyError(f"{where}.artifact.sha256 must be 64 lowercase hex")


def _number_or_null(d: dict[str, Any], key: str, where: str, frozen: bool) -> None:
    v = d.get(key)
    if v is None:
        if frozen:
            raise StudyError(f"{where}: a frozen study needs a numeric {key}")
        return
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise StudyError(f"{where}: {key} must be a number or null")


def parse_study(raw: dict[str, Any]) -> Study:
    if not isinstance(raw, dict):
        raise StudyError("study must be a JSON object")
    _need(raw, "study_version", str, "study")
    frozen = _need(raw, "frozen", bool, "study")
    synthetic = _need(raw, "synthetic", bool, "study")
    for role in ("owner", "operator"):
        v = _need(raw, role, str, "study")
        if not PSEUDONYM_RE.match(v) and not (not frozen and v == "REPLACE"):
            raise StudyError(f"study.{role} must be a pseudonym")
    task = _need(raw, "task", dict, "study")
    kind = _need(task, "kind", str, "task")
    if kind not in TASK_KINDS:
        raise StudyError(f"task.kind must be one of {TASK_KINDS}")
    _need(task, "dataset", dict, "task")
    if kind == "alert_triage":
        _need(task, "validity_scope", str, "task")

    baseline = _need(raw, "baseline", dict, "study")
    _need(baseline, "id", str, "baseline")
    if baseline.get("status") != "unchanged":
        raise StudyError("baseline.status must be 'unchanged': the baseline is never modified")
    _artifact(baseline.get("artifact"), "baseline", frozen)
    _need(baseline, "config", dict, "baseline")

    target = _need(raw, "target_failure", dict, "study")
    for key in ("id", "description", "slice_tag", "metric"):
        _need(target, key, str, "target_failure")
    _check_metric(target["metric"], kind, "target_failure")
    _number_or_null(target, "min_meaningful_effect", "target_failure", frozen)
    sm = target.get("slice_membership_sha256")
    if sm is not None and not (isinstance(sm, str) and SHA256_RE.match(sm)):
        raise StudyError("target_failure.slice_membership_sha256 must be 64 hex")
    if frozen and kind == "accuracy" and sm is None:
        # Accuracy items (WSP) carry no tags of their own, so slice membership is supplied
        # separately and must be fixed before evidence. Triage tags are in the dataset hash.
        raise StudyError("a frozen accuracy study binds target_failure.slice_membership_sha256")
    _bound_source(target, "target_failure", synthetic)

    candidates = _need(raw, "candidates", list, "study")
    if not candidates:
        raise StudyError("at least one candidate is required")
    ids = [baseline["id"]]
    for n, c in enumerate(candidates):
        where = f"candidates[{n}]"
        ids.append(_need(c, "id", str, where))
        _check_candidate(c, baseline, where, frozen)
    if len(set(ids)) != len(ids):
        raise StudyError("baseline and candidate ids must be unique")

    constraints = _need(raw, "constraints", list, "study")
    cids = []
    for n, k in enumerate(constraints):
        where = f"constraints[{n}]"
        cids.append(_need(k, "id", str, where))
        _check_constraint(k, kind, where, frozen)
        _bound_source(k, where, synthetic)
    if len(set(cids)) != len(cids):
        raise StudyError("constraint ids must be unique")
    if kind == "alert_triage" and frozen:
        capped = {
            k["metric"]
            for k in constraints
            if k["kind"] == "absolute" and k["op"] == "<=" and k.get("scope", "all") == "all"
        }
        missing = [m for m in REQUIRED_TRIAGE_CAPS if m not in capped]
        if missing:
            raise StudyError(
                f"a frozen triage study needs absolute '<=' caps (scope all) on {missing}"
            )

    analysis = _need(raw, "analysis", dict, "study")
    for key in ("bootstrap_resamples", "bootstrap_seed"):
        if _need(analysis, key, int, "analysis") < 0:
            raise StudyError(f"analysis.{key} must be non-negative")
    if analysis.get("unbalanced_policy", "inconclusive") not in UNBALANCED_POLICIES:
        raise StudyError(f"analysis.unbalanced_policy must be one of {UNBALANCED_POLICIES}")
    if analysis.get("unit", "item") not in UNITS:
        raise StudyError(f"analysis.unit must be one of {UNITS}")
    conf = _need(analysis, "confidence", float, "analysis")
    if not 0.5 < conf < 1:
        raise StudyError("analysis.confidence must be in (0.5, 1)")
    if not isinstance(raw.get("limitations", []), list):
        raise StudyError("study.limitations must be a list")

    if frozen:
        holes = contains_placeholder(raw)
        if holes:
            raise StudyError(f"frozen study still has placeholders at: {holes}")
        if not (isinstance(raw.get("frozen_by"), str) and PSEUDONYM_RE.match(raw["frozen_by"])):
            raise StudyError("a frozen study needs frozen_by (pseudonym)")
        if not (isinstance(raw.get("frozen_at"), str) and TIMESTAMP_RE.match(raw["frozen_at"])):
            raise StudyError("a frozen study needs frozen_at (UTC, e.g. 2026-11-13T10:00Z)")
    elif raw.get("frozen_by") or raw.get("frozen_at"):
        raise StudyError("an unfrozen study must not carry frozen_by / frozen_at")

    return Study(
        raw=raw,
        sha256=sha256_json(raw),
        frozen=frozen,
        synthetic=synthetic,
        task_kind=kind,
        baseline=baseline,
        target=target,
        candidates=tuple(candidates),
        constraints=tuple(constraints),
        analysis=analysis,
    )


def _check_metric(metric: str, kind: str, where: str) -> tuple[str, int, bool]:
    spec = METRICS.get(metric)
    if spec is None or spec[0] not in ("*", kind):
        valid = sorted(m for m, s in METRICS.items() if s[0] in ("*", kind))
        raise StudyError(f"{where}: metric {metric!r} is not defined for {kind}; use {valid}")
    return spec


def _same_artifact(a: dict[str, Any], b: dict[str, Any]) -> bool:
    # Identity is the digest when both are pinned; a different name over the same bytes is the
    # same artifact. Unpinned drafts fall back to name and revision.
    if a.get("sha256") and b.get("sha256"):
        return a["sha256"] == b["sha256"]
    return (a.get("name"), a.get("revision")) == (b.get("name"), b.get("revision"))


def _check_candidate(c: dict[str, Any], baseline: dict[str, Any], where: str, frozen: bool) -> None:
    _need(c, "hypothesis", str, where)
    iv = _need(c, "intervention", dict, where)
    itype = iv.get("type")
    if isinstance(itype, list):
        raise StudyError(f"{where}: one intervention type per candidate; register parts separately")
    spec = TYPES.get(itype) if isinstance(itype, str) else None
    if spec is None:
        raise StudyError(f"{where}: intervention.type must be one of {sorted(TYPES)}")
    if iv.get("family", spec.family) != spec.family:
        raise StudyError(f"{where}: {itype} belongs to family {spec.family!r}")
    for key in spec.required:
        if iv.get(key) in (None, "", {}, []):
            raise StudyError(f"{where}: {itype} needs intervention.{key}")
    _artifact(c.get("artifact"), where, frozen)
    _need(c, "config", dict, where)
    same = _same_artifact(c["artifact"], baseline["artifact"])
    if not spec.changes_weights and not same:
        raise StudyError(
            f"{where}: {itype} does not change weights; its artifact must equal the baseline's"
        )
    if spec.changes_weights and same:
        raise StudyError(f"{where}: {itype} changes weights; its artifact must differ")
    if not spec.changes_weights and c["config"] == baseline["config"]:
        raise StudyError(f"{where}: {itype} must change the configuration it names")
    base_sha = iv.get("base_artifact_sha256")
    if base_sha is not None and base_sha != baseline["artifact"].get("sha256"):
        raise StudyError(f"{where}: base_artifact_sha256 must be the registered baseline digest")
    if itype == "weight_edit":
        ver = iv["edit_verification"]
        if not isinstance(ver, dict) or ver.get("status") not in EDIT_VERIFICATION_STATUSES:
            raise StudyError(
                f"{where}: edit_verification.status must be one of {EDIT_VERIFICATION_STATUSES}"
            )
        rsha = ver.get("report_sha256")
        if rsha is not None and not (isinstance(rsha, str) and SHA256_RE.match(rsha)):
            raise StudyError(f"{where}: edit_verification.report_sha256 must be 64 hex")
        if frozen and ver["status"] == "verified" and rsha is None:
            raise StudyError(f"{where}: a verified edit names its verify-edit report_sha256")
    if itype == "soc_triage_model":
        cal = iv["calibration"]
        ds = cal.get("dataset") if isinstance(cal, dict) else None
        if not isinstance(ds, dict) or not ds.get("name") or not ds.get("revision"):
            raise StudyError(f"{where}: calibration.dataset needs a name and revision")
        csha = ds.get("content_sha256")
        if csha is not None and not (isinstance(csha, str) and SHA256_RE.match(csha)):
            raise StudyError(f"{where}: calibration.dataset.content_sha256 must be 64 hex")
        if frozen and csha is None:
            raise StudyError(f"{where}: a frozen study names the calibration content_sha256")


def _bound_source(d: dict[str, Any], where: str, synthetic: bool) -> None:
    src = d.get("bound_source")
    if src not in BOUND_SOURCES:
        raise StudyError(f"{where}: bound_source must be one of {BOUND_SOURCES}")
    if synthetic and src != "synthetic_fixture":
        raise StudyError(f"{where}: a synthetic study's limits are synthetic_fixture values")
    if not synthetic and src == "synthetic_fixture":
        raise StudyError(f"{where}: a real study cannot use synthetic_fixture limits")


def _check_constraint(k: dict[str, Any], kind: str, where: str, frozen: bool) -> None:
    metric = _need(k, "metric", str, where)
    _, _, pairable = _check_metric(metric, kind, where)
    ckind = _need(k, "kind", str, where)
    if ckind not in CONSTRAINT_KINDS:
        raise StudyError(f"{where}: kind must be one of {CONSTRAINT_KINDS}")
    if k.get("scope", "all") not in SCOPES:
        raise StudyError(f"{where}: scope must be one of {SCOPES}")
    if ckind == "absolute":
        if k.get("op") not in ("<=", ">="):
            raise StudyError(f"{where}: absolute constraints need op '<=' or '>='")
        _number_or_null(k, "bound", where, frozen)
    elif ckind == "non_regression":
        _number_or_null(k, "margin", where, frozen)
        if isinstance(k.get("margin"), int | float) and k["margin"] < 0:
            raise StudyError(f"{where}: margin must be >= 0")
    else:
        if not pairable:
            raise StudyError(f"{where}: {metric} has arm-dependent denominators; not pairable")
        if k.get("losses", "net") not in LOSS_COUNTS:
            raise StudyError(f"{where}: losses must be one of {LOSS_COUNTS}")
        v = k.get("max_losses")
        if v is None and frozen:
            raise StudyError(f"{where}: a frozen study needs max_losses")
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
            raise StudyError(f"{where}: max_losses must be a non-negative integer")


def load_study(path: Path) -> Study:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StudyError(f"{path}: invalid JSON ({exc.msg})") from exc
    return parse_study(raw)
