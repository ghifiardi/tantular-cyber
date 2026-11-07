"""Scripted evaluators for the synthetic study fixture. Not models; outputs mean nothing.

They exercise every outcome path: a threshold candidate that addresses the target failure, a
"triage model" candidate that errors on two alerts (a declared constraint is violated), and a
weight-edit candidate whose tensor-level verification has not run (prerequisite missing).
"""

from __future__ import annotations

from typing import Any

from .soc import SyntheticTriageEvaluator
from .study import Study


def synthetic_evaluators(study: Study, **overrides: Any) -> dict[str, SyntheticTriageEvaluator]:
    def make(arm: dict[str, Any], **kw: Any) -> SyntheticTriageEvaluator:
        cfg = arm["config"]
        return SyntheticTriageEvaluator(
            arm["artifact"]["name"],
            select_at=cfg["select_at"],
            defer_at=cfg["defer_at"],
            score_feature=cfg["score_feature"],
            artifact_sha256=arm["artifact"]["sha256"],
            **kw,
        )

    out = {study.baseline["id"]: make(study.baseline)}
    for c in study.candidates:
        kw = {"fail_ids": ("syn-alert-031", "syn-alert-032")} if c["id"] == "soc-model-alt" else {}
        out[c["id"]] = make(c, **kw)
    out.update(overrides)
    return out
