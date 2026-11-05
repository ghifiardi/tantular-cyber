"""Study report: machine-readable JSON and a human-readable Markdown rendering.

The overall outcome is ``inconclusive`` unless at least one candidate meets every declared
criterion under a preregistered study; then it is
``candidates_meet_declared_criteria_pending_human_decision``. The report never recommends or
performs a promotion. ``human_decision`` is always ``pending`` in a report; decisions are
separate, unauthenticated records (``decision.py``).

Every value is shown with its numerator and denominator, every declared limit with its
``bound_source`` (synthetic fixture values are labelled as such), and every interval with its
method and number of independent units.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from wsp_eval.hashing import sha256_json

from . import WORKFLOW_VERSION
from .compare import compare_candidate
from .evidence import Evidence
from .study import Study

EVIDENCE_LIMIT = (
    "Every result here comes from synthetic fixtures. It is not a measurement of any model, "
    "SOC deployment or dataset, and it is not a Tantular Cyber baseline."
)
DECISION_NOTE = (
    "No artifact is promoted or deployed by this workflow, and nothing here enforces a "
    "decision. A decision is a separate record made under a pseudonym; the identity is "
    "recorded, not authenticated. Any deployment needs its own evaluation and release process."
)


def _multiplicity(study: Study) -> str:
    n_c, n_k = len(study.candidates), len(study.constraints)
    return (
        f"{n_c} candidate(s) and {n_k} constraint(s) are evaluated, each at "
        f"{study.analysis['confidence']:.0%} confidence, with no adjustment for multiple "
        "comparisons. The chance that at least one candidate meets the criteria by chance is "
        "higher than for a single test."
    )


def build_report(
    study: Study, evidence: Sequence[Evidence], evidence_errors: Sequence[str] = ()
) -> dict[str, Any]:
    results = [compare_candidate(study, c, evidence) for c in study.candidates]
    meets = [r["candidate"] for r in results if r["outcome"].startswith("meets_declared")]
    synthetic = study.synthetic or any(e.synthetic for e in evidence) or not evidence
    limitations = list(study.raw.get("limitations", []))
    for e in evidence:
        limitations += [x for x in e.limitations if x not in limitations]
    limitations.append(_multiplicity(study))
    unit = study.analysis.get("unit", "item")
    if study.task_kind == "alert_triage" and unit == "item":
        limitations.append(
            "Intervals treat alerts as independent (unit: item). Correlated alerts from one "
            "incident or campaign make these intervals too narrow; use unit: cluster when "
            "cluster ids exist."
        )
    if synthetic:
        limitations.insert(0, EVIDENCE_LIMIT)
    report = {
        "workflow_version": WORKFLOW_VERSION,
        "study_version": study.raw["study_version"],
        "study_sha256": study.sha256,
        "study_frozen": study.frozen,
        "evidence_class": "synthetic_fixture_not_a_measurement" if synthetic else "measured",
        "task_kind": study.task_kind,
        "validity_scope": study.raw["task"].get("validity_scope"),
        "unit": unit,
        "baseline": {
            "id": study.baseline["id"],
            "artifact": study.baseline["artifact"],
            "status": "unchanged",
        },
        "target_failure": study.target,
        "evidence_runs": [
            {"run_id": e.run_id, "source": e.source, "synthetic": e.synthetic, **e.notes}
            for e in evidence
        ],
        "evidence_errors": list(evidence_errors),
        "candidates": results,
        "outcome": (
            "candidates_meet_declared_criteria_pending_human_decision" if meets else "inconclusive"
        ),
        "candidates_meeting_criteria": meets,
        "limitations": limitations,
        "human_decision": "pending",
        "decision_note": DECISION_NOTE,
    }
    report["report_sha256"] = sha256_json(report)
    return report


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _ci(ci: Sequence[Any]) -> str:
    return f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"


def _count(x: Any) -> str:
    return str(int(x)) if isinstance(x, float) and x.is_integer() else _fmt(x)


def _frac(s: dict[str, Any]) -> str:
    return f"{_fmt(s['value'])} ({_count(s['numerator'])}/{_count(s['denominator'])})"


def _src(source: str) -> str:
    return (
        " *(synthetic fixture value)*" if source == "synthetic_fixture" else " *(owner-declared)*"
    )


def _constraint_value(c: dict[str, Any]) -> str:
    cand, base = _frac(c["candidate"]), _frac(c["baseline"])
    if c["kind"] == "absolute":
        b = c["confidence_bound"]
        return (
            f"candidate {cand}; one-sided bound {_fmt(b['bound'])} ({b['method']}, "
            f"{b['units']} units) {c['op']} {_fmt(c.get('bound'))}{_src(c['bound_source'])}; "
            f"baseline {base}"
        )
    if c["kind"] == "non_regression":
        imp = c["improvement"]
        return (
            f"improvement {_fmt(imp['point'])} {_ci(imp['ci'])} ({imp['method']}, "
            f"{imp['units']} units) ≥ −{_fmt(c.get('margin'))}{_src(c['bound_source'])}; "
            f"candidate {cand}, baseline {base}"
        )
    p = c.get("paired") or {}
    return (
        f"net paired losses {_fmt(p.get('net_losses'))} ({_fmt(p.get('losses'))} lost, "
        f"{_fmt(p.get('wins'))} won of {_fmt(p.get('items'))}) ≤ {_fmt(c.get('max_losses'))}"
        f"{_src(c['bound_source'])}"
    )


def _breakdown(r: dict[str, Any]) -> list[str]:
    arms = list(r["breakdown"])
    lines = [
        "",
        "| Decision / label | " + " | ".join(f"`{a}`" for a in arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    for key in r["breakdown"][arms[0]]:
        counts = " | ".join(str(r["breakdown"][a][key]) for a in arms)
        lines.append(f"| {key} | {counts} |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    t = report["target_failure"]
    lines = [
        f"# Improvement study report: {report['study_version']}",
        "",
        f"**Outcome: `{report['outcome']}`** · evidence class "
        f"`{report['evidence_class']}` · human decision: **{report['human_decision']}**",
        "",
        f"Study `{report['study_sha256'][:16]}…` (frozen: {report['study_frozen']}), report "
        f"`{report['report_sha256'][:16]}…`, workflow {report['workflow_version']}. "
        f"Unit of analysis: `{report['unit']}`.",
    ]
    if report.get("validity_scope"):
        lines += ["", f"Validity scope: {report['validity_scope']}"]
    lines += [
        "",
        "## Limitations",
        "",
        *[f"- {x}" for x in report["limitations"]],
        "",
        "## Baseline and target failure",
        "",
        f"- Baseline `{report['baseline']['id']}`: `{report['baseline']['artifact']['name']}`, "
        "registered as unchanged.",
        f"- Target failure `{t['id']}`: {t['description']} (metric `{t['metric']}` on slice "
        f"`{t['slice_tag']}`, minimum meaningful improvement "
        f"{_fmt(t.get('min_meaningful_effect'))}{_src(t['bound_source'])}).",
        "",
        "## Candidates",
        "",
        "| Candidate | Intervention | Outcome | Target improvement [CI] (method, units) |",
        "|---|---|---|---|",
    ]
    none = {"point": None, "ci": [None, None], "method": "—", "units": 0}
    for r in report["candidates"]:
        imp = (r.get("target_effect") or {}).get("improvement") or none
        lines.append(
            f"| `{r['candidate']}` | {r['intervention_type']} | `{r['outcome']}` | "
            f"{_fmt(imp['point'])} {_ci(imp['ci'])} ({imp['method']}, {imp['units']}) |"
        )
    for r in report["candidates"]:
        lines += ["", f"### `{r['candidate']}` ({r['intervention_type']})", ""]
        lines.append(f"Hypothesis: {r['hypothesis']}")
        if r.get("rule_outcome") and r["rule_outcome"] != r["outcome"]:
            lines.append(f"Rule outcome if it were binding: `{r['rule_outcome']}`.")
        eff = r.get("target_effect")
        if eff:
            lines.append(
                f"Target `{eff['metric']}`: baseline {_frac(eff['baseline'])}, candidate "
                f"{_frac(eff['candidate'])}."
            )
        if r["reasons"]:
            lines += ["", "Reasons:", *[f"- {x}" for x in r["reasons"]]]
        if r.get("breakdown"):
            lines += _breakdown(r)
        if r["constraints"]:
            lines += [
                "",
                "| Constraint | Metric (scope) | Kind | Value vs declared limit | Status |",
                "|---|---|---|---|---|",
            ]
            for c in r["constraints"]:
                lines.append(
                    f"| `{c['id']}` | {c['metric']} ({c['scope']}) | {c['kind']} | "
                    f"{_constraint_value(c)} | `{c['status']}` |"
                )
    if report["evidence_errors"]:
        lines += ["", "## Evidence not used", "", *[f"- {x}" for x in report["evidence_errors"]]]
    lines += ["", "## Decision", "", report["decision_note"], ""]
    return "\n".join(lines)
