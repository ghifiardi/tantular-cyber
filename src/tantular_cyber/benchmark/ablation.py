"""Scanner ablation: attribute detection to Semgrep rules vs tantular-callgraph.

The corpus is scanned three times — Semgrep only, tantular-callgraph only, and both
(the default; Semgrep has precedence on duplicates) — and each run is matched to ground
truth with the same rules. Unique true positives per scanner show which tool is
responsible for which detection; the combined run is the one that gates.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ..callgraph import CallgraphScanner
from ..config import ScanSettings
from ..pipeline import run_scan
from ..scanner import SemgrepScanner
from .manifest import Manifest
from .matching import Metrics, match_findings


class AblationArm(BaseModel):
    scanners: str
    metrics: Metrics
    by_cwe: dict[str, Metrics]
    detected: list[str]


class Ablation(BaseModel):
    arms: list[AblationArm]
    semgrep_unique: list[str]
    callgraph_unique: list[str]
    both: list[str]
    callgraph_new_fp: list[str]


def run_ablation(
    manifest: Manifest, cases_root: Path, tmp: Path, semgrep_bin: str, context_strategy: str
) -> Ablation:
    settings = ScanSettings(
        repo=cases_root,
        output=tmp / "ablation",
        llm_enabled=False,
        semgrep_bin=semgrep_bin,
        context_strategy=context_strategy,
    )
    semgrep = SemgrepScanner(
        settings.rules, binary=semgrep_bin, max_target_bytes=settings.max_file_bytes
    )
    arms: list[AblationArm] = []
    detected: dict[str, set[str]] = {}
    labels: dict[str, str] = {}
    for name, scanners in (
        ("semgrep", [semgrep]),
        ("callgraph", [CallgraphScanner()]),
        ("semgrep+callgraph", [semgrep, CallgraphScanner()]),
    ):
        report = run_scan(settings, scanners=scanners).report
        match = match_findings(manifest, report.findings)
        hits = {r.annotation_id for r in match.records if r.detected}
        labels.update({r.annotation_id: r.label for r in match.records})
        detected[name] = hits
        arms.append(
            AblationArm(
                scanners=name, metrics=match.metrics, by_cwe=match.by_cwe, detected=sorted(hits)
            )
        )
    sg, cg = detected["semgrep"], detected["callgraph"]
    tp = {a for a, label in labels.items() if label in ("true_positive", "known_false_negative")}
    return Ablation(
        arms=arms,
        semgrep_unique=sorted((sg - cg) & tp),
        callgraph_unique=sorted((cg - sg) & tp),
        both=sorted(sg & cg & tp),
        callgraph_new_fp=sorted((cg - sg) - tp),
    )
