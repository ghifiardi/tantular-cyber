"""Extension point for SOC alert-triage evaluation (provider-neutral; synthetic only).

A real triage system (a SIEM-side scorer, an anomaly model with an operating point, an analyst
queue) would be an adapter implementing :class:`TriageEvaluator`. **No real adapter exists in
this repository**, and nothing here contacts a SIEM, reads production alerts or writes
decisions anywhere. :class:`SyntheticTriageEvaluator` is a scripted threshold over a synthetic
score in fixture alerts; it is not a model and its results are not measurements.

Evaluators never see labels: :func:`run_triage_evaluation` passes :class:`Alert` objects (id,
tags, features) and joins the separately held ground-truth labels only when writing records.
Decisions are ``select`` (escalate), ``defer`` (to an analyst) or ``suppress``.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from wsp_eval.hashing import sha256_file, sha256_json
from wsp_eval.runner import RunError
from wsp_eval.runner import _prepare_out_dir as prepare_out_dir

from . import WORKFLOW_VERSION
from .evidence import triage_metadata_sha256
from .study import Study

DECISIONS = ("select", "defer", "suppress")
LABELS = ("threat", "benign")


@dataclass(frozen=True)
class Alert:
    id: str
    tags: tuple[str, ...]
    features: dict[str, Any] = field(default_factory=dict)
    cluster: str | None = None  # incident/campaign id; correlated alerts share one


@dataclass(frozen=True)
class TriageDecision:
    decision: str
    score: float | None = None


class TriageEvaluator(Protocol):
    def describe(self) -> dict[str, Any]:
        """Provider, model, revision, ``artifacts_sha256`` (file -> digest actually loaded),
        ``config`` (operating point, thresholds, feature contract) and ``synthetic``."""
        ...

    def decide(self, alert: Alert) -> TriageDecision: ...


class SyntheticTriageEvaluator:
    """Scripted thresholds over ``features["synthetic_score"]``; ``fail_ids`` raise errors."""

    def __init__(
        self,
        name: str,
        *,
        select_at: float,
        defer_at: float,
        artifact_sha256: str,
        score_feature: str = "synthetic_score",
        fail_ids: Sequence[str] = (),
    ):
        self.name = name
        self.config = {"select_at": select_at, "defer_at": defer_at, "score_feature": score_feature}
        self.artifact_sha256 = artifact_sha256
        self.fail_ids = set(fail_ids)

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "synthetic",
            "model": self.name,
            "revision": None,
            "artifacts_sha256": {"synthetic-scorer": self.artifact_sha256},
            "config": dict(self.config),
            "synthetic": True,
        }

    def decide(self, alert: Alert) -> TriageDecision:
        if alert.id in self.fail_ids:
            raise TimeoutError("synthetic scorer timeout")
        score = float(alert.features[self.config["score_feature"]])
        if score >= self.config["select_at"]:
            return TriageDecision("select", score)
        if score >= self.config["defer_at"]:
            return TriageDecision("defer", score)
        return TriageDecision("suppress", score)


def load_alerts(path: Path) -> tuple[list[Alert], dict[str, str | None]]:
    """Fixture rows ``{id, tags, features, label[, cluster]}``; returns alerts and labels."""
    alerts, labels = [], {}
    for n, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        allowed = {"id", "tags", "features", "label", "cluster"}
        if not {"id", "tags", "features", "label"} <= set(row) <= allowed:
            raise ValueError(f"{path.name}:{n}: expected id, tags, features, label[, cluster]")
        if row["label"] is not None and row["label"] not in LABELS:
            raise ValueError(f"{path.name}:{n}: label must be one of {LABELS} or null")
        if row["id"] in labels:
            raise ValueError(f"{path.name}:{n}: duplicate alert id {row['id']!r}")
        alerts.append(
            Alert(row["id"], tuple(row["tags"]), dict(row["features"]), row.get("cluster"))
        )
        labels[row["id"]] = row["label"]
    return alerts, labels


def alerts_sha256(alerts: Sequence[Alert], labels: Mapping[str, str | None]) -> str:
    """Content hash of the evaluation set (alerts and their ground-truth labels)."""
    return sha256_json(
        sorted([a.id, list(a.tags), a.features, a.cluster, labels[a.id]] for a in alerts)
    )


def alerts_metadata_sha256(alerts: Sequence[Alert], labels: Mapping[str, str | None]) -> str:
    """Registered form of the triage metadata (``task.dataset.metadata_sha256``)."""
    return triage_metadata_sha256({a.id: (a.tags, a.cluster, labels[a.id]) for a in alerts})


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_triage_evaluation(
    study: Study,
    alerts: Sequence[Alert],
    labels: Mapping[str, str | None],
    evaluators: Mapping[str, TriageEvaluator],
    out_dir: Path,
    *,
    operator: str,
    allow_in_repo: bool = False,
    run_id: str | None = None,
    now: Callable[[], str] = _now,
) -> Path:
    """Evaluate every registered arm on every alert; one record per (alert, arm), always."""
    if study.task_kind != "alert_triage":
        raise RunError("run_triage_evaluation needs an alert_triage study")
    if sorted(evaluators) != sorted(study.arm_ids):
        raise RunError(f"evaluators must match study arms {sorted(study.arm_ids)}")
    if not study.synthetic and any(e.describe().get("synthetic") for e in evaluators.values()):
        raise RunError("synthetic evaluators may only be used with a synthetic study")
    if set(labels) != {a.id for a in alerts}:
        raise RunError("labels must cover exactly the alerts")
    content = alerts_sha256(alerts, labels)
    declared = study.raw["task"]["dataset"].get("content_sha256")
    if declared is not None and declared != content:
        raise RunError("alerts do not match the study's dataset content_sha256")
    out = prepare_out_dir(out_dir, allow_in_repo)
    run_id = run_id or f"imp-{uuid.uuid4().hex[:12]}"
    started = now()
    records: list[dict[str, Any]] = []
    abort: BaseException | None = None
    try:
        for alert in sorted(alerts, key=lambda a: a.id):
            for arm in study.arm_ids:
                rec: dict[str, Any] = {
                    "run_id": run_id,
                    "arm": arm,
                    "item_id": alert.id,
                    "tags": list(alert.tags),
                    "cluster": alert.cluster,
                    "sample": 0,
                    "label": labels[alert.id],
                }
                try:
                    out_dec = evaluators[arm].decide(alert)
                except Exception as exc:  # recorded, never dropped
                    rec.update(
                        status="error", decision=None, score=None, error_class=type(exc).__name__
                    )
                else:
                    if not isinstance(out_dec, TriageDecision) or out_dec.decision not in DECISIONS:
                        rec.update(
                            status="error", decision=None, score=None, error_class="InvalidDecision"
                        )
                    else:
                        rec.update(
                            status="decided",
                            decision=out_dec.decision,
                            score=out_dec.score,
                            error_class=None,
                        )
                records.append(rec)
    except BaseException as exc:  # harness failure: write what exists, then re-raise
        abort = exc

    with (out / "records.jsonl").open("x", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    described = {arm: evaluators[arm].describe() for arm in study.arm_ids}
    manifest = {
        "run_id": run_id,
        "source": "synthetic_triage",
        "workflow_version": WORKFLOW_VERSION,
        "study_sha256": study.sha256,
        "study_frozen": study.frozen,
        "synthetic": study.synthetic,
        "task_kind": study.task_kind,
        "dataset_sha256": content,
        "operator": operator,
        "started_at": started,
        "finished_at": now(),
        "run_status": "aborted" if abort else "complete",
        "abort_reason": type(abort).__name__ if abort else None,
        "expected_records": len(alerts) * len(study.arm_ids),
        "counts": dict(sorted(Counter(r["status"] for r in records).items())),
        "records_sha256": sha256_file(out / "records.jsonl"),
        "arms": {
            arm: {
                "artifacts": dict(d.get("artifacts_sha256") or {}),
                "artifact_sha256s": sorted(set((d.get("artifacts_sha256") or {}).values())),
                "config_sha256": sha256_json(d.get("config", {})),
                "evaluator": d,
            }
            for arm, d in described.items()
        },
        "labels_seen_by_evaluators": False,
        "human_decision": "pending",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if abort is not None:
        raise abort
    return out
