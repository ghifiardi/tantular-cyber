"""POC-007 Step 0: offline forensic re-scoring of retained POC-005 comparison outputs.

Exploratory only (docs/poc-007-plan.md, sections 3 and 7). For each retained run this:

1. imports the *historical* package from a checkout of a preservation tag
   (eval/poc-005-<sha>) and refuses to continue if any other copy is imported;
2. rebuilds the corpus findings by scanning the bundled corpus at that commit, exactly as the
   historical harness did when resuming from a checkpoint (no LLM call, no network);
3. re-applies that commit's rubric (``evaluate_finding``) to each saved analysis;
4. splits ``evidence_grounded`` outcomes into grounded, missing identifiers, line references
   out of range, both, and nothing cited.

It is not a reproduction of the published 57%: that run's per-sample outputs were not kept.
Output is aggregate counts only, never model text, prompts or source. The exit code is 0 only
when ``validate`` finds nothing: every recorded finding rebuilt, the selection and corpus equal
the run header, every recorded arm present and matching its recorded result, and no sample
excluded.

Usage (historical checkout created with ``git worktree add --detach DIR eval/poc-005-76a9318``):

    python tools/eval/step0_rescore.py --historical DIR --run out/optim --output out/step0/optim
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

CATEGORIES = ("grounded", "missing_identifiers", "line_refs_out_of_range", "both", "nothing_cited")


def classify(passed: bool | None, detail: str) -> str:
    """One evidence_grounded result -> category, from the historical check's own detail text."""
    if passed is None:
        return "nothing_cited"
    if passed:
        return "grounded"
    missing = "not in evidence:" in detail
    lines = "line refs outside" in detail
    if missing and lines:
        return "both"
    if missing:
        return "missing_identifiers"
    if lines:
        return "line_refs_out_of_range"
    raise ValueError(f"unrecognised evidence_grounded failure detail: {detail!r}")


def wilson(passed: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = passed / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(centre - half, 4), round(centre + half, 4)


def import_historical(checkout: Path) -> SimpleNamespace:
    """Import tantular_cyber from ``checkout/src`` and prove that is the copy in use."""
    src = (checkout / "src").resolve()
    if not (src / "tantular_cyber" / "__init__.py").is_file():
        raise SystemExit(f"not a tantular-cyber checkout: {checkout}")
    if any(m == "tantular_cyber" or m.startswith("tantular_cyber.") for m in sys.modules):
        raise SystemExit("tantular_cyber already imported; run this script in a fresh process")
    sys.path.insert(0, str(src))
    pkg = importlib.import_module("tantular_cyber")
    if not Path(pkg.__file__).resolve().is_relative_to(src):
        raise SystemExit(f"imported {pkg.__file__}, not the historical checkout {src}")
    compare = importlib.import_module("tantular_cyber.benchmark.compare")
    manifest = importlib.import_module("tantular_cyber.benchmark.manifest")
    llm_eval = importlib.import_module("tantular_cyber.benchmark.llm_eval")
    models = importlib.import_module("tantular_cyber.models")
    git = shutil.which("git")
    if git is None:
        raise SystemExit("git not found; it is needed to record the historical commit")
    commit = subprocess.run(
        [git, "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.strip()
    return SimpleNamespace(
        commit=commit,
        select_findings=compare.select_findings,
        load_manifest=manifest.load_manifest,
        evaluate_finding=llm_eval.evaluate_finding,
        FindingAnalysis=models.FindingAnalysis,
    )


def semgrep_version() -> str | None:
    semgrep = shutil.which("semgrep")
    if semgrep is None:
        return None
    try:
        out = subprocess.run(
            [semgrep, "--version"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def rescore(hist: SimpleNamespace, checkout: Path, run_dir: Path) -> dict[str, object]:
    lines = (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])["header"]
    manifest, cases_root = hist.load_manifest(checkout / "benchmarks" / "corpus")
    corpus_ok = (
        manifest.corpus_id == header["corpus"]
        and manifest.manifest_version == header["manifest_version"]
    )
    with tempfile.TemporaryDirectory() as tmp:
        # Same selection the harness used, without the cap, so every recorded id can be found.
        found, lookup = hist.select_findings(manifest, cases_root, 10_000, Path(tmp))
    by_id = {f.id: f for f in found}
    expected = list(header["findings"])
    missing_ids = [i for i in expected if i not in by_id]
    first_n = sorted(by_id)[: len(expected)]

    tallies: dict[str, Counter[str]] = defaultdict(Counter)
    excluded: dict[str, Counter[str]] = defaultdict(Counter)
    for raw in lines[1:]:
        row = json.loads(raw)
        arm = row["arm"]
        if row.get("error"):
            excluded[arm]["request_error"] += 1
            continue
        base = by_id.get(row["finding_id"])
        if base is None:
            excluded[arm]["finding_not_reconstructed"] += 1
            continue
        copy = base.model_copy(deep=True)
        try:
            copy.analysis = hist.FindingAnalysis.model_validate(row["analysis"])
        except ValueError:
            excluded[arm]["analysis_invalid"] += 1
            continue
        ev = hist.evaluate_finding(
            copy, lookup["cases"][copy.path], lookup["records"].get(copy.id), row["run"]
        )
        check = next((c for c in ev.checks if c.id == "evidence_grounded"), None)
        if check is None:
            # No validated explanation: the historical rubric never assessed grounding here.
            excluded[arm]["no_explanation"] += 1
            continue
        tallies[arm][classify(check.passed, check.detail)] += 1

    # The run's own summary defines which arms must be present and what they recorded.
    recorded = {}
    comparison = run_dir / "comparison.json"
    if comparison.is_file():
        for a in json.loads(comparison.read_text(encoding="utf-8")).get("arms", []):
            eg = a.get("checks", {}).get("evidence_grounded", {})
            recorded[a["arm"]] = {"passed": eg.get("passed"), "assessed": eg.get("assessed")}

    arms = {}
    for arm in sorted(set(tallies) | set(excluded) | set(recorded)):
        t = tallies[arm]
        assessed = sum(t[c] for c in CATEGORIES if c != "nothing_cited")
        lo, hi = wilson(t["grounded"], assessed)
        rec = recorded.get(arm)
        arms[arm] = {
            "categories": {c: t[c] for c in CATEGORIES},
            "scored": sum(t.values()),
            "excluded": dict(excluded[arm]),
            "evidence_grounded": {
                "passed": t["grounded"],
                "assessed": assessed,
                "rate": round(t["grounded"] / assessed, 4) if assessed else None,
                "ci95": [lo, hi],
            },
            "recorded_in_run": rec,
            "matches_recorded": (
                rec is not None and rec == {"passed": t["grounded"], "assessed": assessed}
            ),
        }

    return {
        "result_type": "tantular-cyber.poc007-step0-rescoring",
        "schema_version": "1.0.0",
        "label": "exploratory re-scoring; not a reproduction of the published benchmark",
        "run": run_dir.name,
        "historical_commit": hist.commit,
        "semgrep_version_now": semgrep_version(),
        "corpus": {"id": header["corpus"], "manifest_version": header["manifest_version"]},
        "corpus_matches_header": corpus_ok,
        "reconstruction": {
            "expected_findings": len(expected),
            "reconstructed": len(expected) - len(missing_ids),
            "missing_ids": missing_ids,
            "selection_matches_header": first_n == sorted(expected),
        },
        "recorded_summary_found": comparison.is_file(),
        "expected_arms": sorted(recorded),
        "arms": arms,
    }


def validate(r: dict[str, object]) -> list[str]:
    """Every reason the re-scoring cannot be trusted as a reconstruction; empty means valid.

    Fails closed: a run without a recorded summary has no expected arms, and that is a
    failure, not a vacuous pass.
    """
    failures: list[str] = []
    rec = r["reconstruction"]
    if rec["missing_ids"]:
        failures.append(f"{len(rec['missing_ids'])} recorded finding(s) not reconstructed")
    if rec["selection_matches_header"] is not True:
        failures.append("reconstructed finding selection differs from the run header")
    if r["corpus_matches_header"] is not True:
        failures.append("corpus id or manifest version differs from the run header")
    expected = r["expected_arms"]
    if not r["recorded_summary_found"] or not expected:
        failures.append("no recorded arm results (comparison.json) to check against")
    arms = r["arms"]
    for arm in expected:
        if arm not in arms or arms[arm]["scored"] + sum(arms[arm]["excluded"].values()) == 0:
            failures.append(f"expected arm missing from samples: {arm}")
    for arm in sorted(set(arms) - set(expected)):
        failures.append(f"arm in samples but not in the recorded results: {arm}")
    for arm in expected:
        if arm in arms and arms[arm]["matches_recorded"] is not True:
            failures.append(f"re-scored result differs from the recorded result: {arm}")
    excluded = sum(n for a in arms.values() for n in a["excluded"].values())
    if excluded:
        failures.append(f"{excluded} sample(s) excluded")
    return failures


def render(r: dict[str, object]) -> str:
    rec = r["reconstruction"]
    out = [
        f"# Step 0 re-scoring — `{r['run']}` (exploratory)",
        "",
        "Not a reproduction of the published benchmark. Rubric and evidence from historical "
        f"commit `{r['historical_commit'][:7]}`; Semgrep used now: {r['semgrep_version_now']}.",
        "",
        f"Findings reconstructed: {rec['reconstructed']}/{rec['expected_findings']}; "
        f"selection matches run header: {rec['selection_matches_header']}; "
        f"corpus matches header: {r['corpus_matches_header']}.",
        "",
        "Validation: "
        + (
            "passed."
            if r["validation"]["passed"]
            else "**FAILED** — " + "; ".join(r["validation"]["failures"]) + "."
        ),
        "",
        "| Arm | Grounded | Missing ids | Lines out of range | Both | Nothing cited "
        "| Rate (Wilson 95%) | Recorded in run | Match | Excluded |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, a in r["arms"].items():
        c, eg, rec_ = a["categories"], a["evidence_grounded"], a["recorded_in_run"]
        rate = "–" if eg["rate"] is None else f"{eg['passed']}/{eg['assessed']} = {eg['rate']:.3f}"
        ci = "" if eg["ci95"][0] is None else f" ({eg['ci95'][0]:.2f}–{eg['ci95'][1]:.2f})"
        recs = "–" if not rec_ else f"{rec_['passed']}/{rec_['assessed']}"
        excl = ", ".join(f"{k} {v}" for k, v in a["excluded"].items()) or "0"
        out.append(
            f"| `{arm}` | {c['grounded']} | {c['missing_identifiers']} | "
            f"{c['line_refs_out_of_range']} | {c['both']} | {c['nothing_cited']} | {rate}{ci} | "
            f"{recs} | {'yes' if a['matches_recorded'] else 'NO'} | {excl} |"
        )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--historical", type=Path, required=True, help="checkout of an eval/ tag")
    p.add_argument("--run", type=Path, required=True, help="dir with samples.jsonl")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    hist = import_historical(args.historical)
    result = rescore(hist, args.historical, args.run)
    failures = validate(result)
    result["validation"] = {"passed": not failures, "failures": failures}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rescoring.json").write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    (args.output / "rescoring.md").write_text(render(result), "utf-8")
    print(render(result))
    return 0 if result["validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
