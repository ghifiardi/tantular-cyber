"""Evidence runs: per-item records plus a manifest, verified before anything is compared.

An evidence run directory holds ``manifest.json`` and ``records.jsonl`` (one record per
item, sample and arm). :func:`load_evidence` refuses a run that belongs to another study, did
not complete, does not match its records hash, mixes run ids, has the wrong record count or
contains duplicate (arm, item, sample) records. Nothing is repaired or imputed.

Two sources exist:

* ``synthetic_triage``: written by :func:`improve_eval.soc.run_triage_evaluation` from scripted
  synthetic evaluators (the only triage evaluators in this repository);
* ``wsp``: a run of the WSP harness (``tools/wsp``), verified with ``wsp_eval.runner.verify_run``
  against its protocol and translated into accuracy records for one weight-edit comparison.
  It is tied to that comparison: the candidate registered this WSP protocol hash; the WSP items
  hash is the study's dataset hash; the WSP ``reference`` arm is the baseline and a
  ``candidate``-role arm is the candidate; the WSP protocol counts missing outcomes as
  incorrect (as this workflow does). A supplied ``verify-edit`` report must hash to the
  registered ``report_sha256``, be ``verified``, and name the registered base and candidate file
  digests and the registered edit manifest (``manifest_sha256``; reports from a ``verify-edit``
  without that field are refused); only then is it recorded as ``hash_consistent``. Target-slice
  membership must hash to the study's ``slice_membership_sha256``. That is a hash and provenance
  consistency check of a report someone supplied: it does not authenticate who produced the
  report and does not re-run the tensor comparison. The WSP protocol's required limitations
  and preregistration status travel into the report.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wsp_eval.analysis import preregistration as wsp_preregistration
from wsp_eval.hashing import sha256_file, sha256_json
from wsp_eval.protocol import Protocol
from wsp_eval.runner import RunError, verify_run

from .study import Study

SOURCES = ("synthetic_triage", "wsp")


class EvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class Evidence:
    run_id: str
    source: str
    synthetic: bool
    started_at: str | None
    # arm id -> {"artifact_sha256s": [...], "config_sha256": str | None, "evaluator": {...}}
    arms: dict[str, dict[str, Any]]
    records: tuple[dict[str, Any], ...]
    limitations: tuple[str, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)


def _check_records(records: Sequence[dict[str, Any]], run_id: str) -> None:
    seen: set[tuple[str, str, int]] = set()
    for r in records:
        if r.get("run_id") != run_id:
            raise EvidenceError("records.jsonl contains records from another run")
        key = (r["arm"], r["item_id"], r["sample"])
        if key in seen:
            raise EvidenceError(f"duplicate record for arm/item/sample {key}")
        seen.add(key)


def load_evidence(run_dir: Path, study: Study) -> Evidence:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source") != "synthetic_triage":
        raise EvidenceError("use evidence_from_wsp() for WSP runs; unknown evidence source")
    if manifest.get("study_sha256") != study.sha256:
        raise EvidenceError("evidence was collected under a different study (study_sha256)")
    if manifest.get("run_status") != "complete":
        raise EvidenceError(f"run_status is {manifest.get('run_status')!r}; not used")
    path = run_dir / "records.jsonl"
    if sha256_file(path) != manifest.get("records_sha256"):
        raise EvidenceError("records.jsonl does not match records_sha256 in the manifest")
    records = [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]
    if len(records) != manifest.get("expected_records"):
        raise EvidenceError("record count differs from expected_records in the manifest")
    _check_records(records, manifest["run_id"])
    if {r["arm"] for r in records} != set(manifest["arms"]):
        raise EvidenceError("record arms differ from the arms the manifest declares")
    unknown = sorted(set(manifest["arms"]) - set(study.arm_ids))
    if unknown:
        raise EvidenceError(f"evidence arms not registered in the study: {unknown}")
    if manifest.get("task_kind") != study.task_kind:
        raise EvidenceError("evidence task kind differs from the study")
    declared = study.raw["task"]["dataset"].get("content_sha256")
    if declared is None or manifest.get("dataset_sha256") != declared:
        raise EvidenceError("evidence dataset hash differs from (or is missing in) the study")
    return Evidence(
        run_id=manifest["run_id"],
        source="synthetic_triage",
        synthetic=bool(manifest.get("synthetic", True)),
        started_at=manifest.get("started_at"),
        arms=manifest["arms"],
        records=tuple(records),
        limitations=tuple(manifest.get("limitations", [])),
    )


def slice_membership_sha256(item_ids: Sequence[str], item_tags: Mapping[str, Sequence[str]]) -> str:
    """Hash of which items carry which tags, over exactly ``item_ids``. Register it in the
    study (``target_failure.slice_membership_sha256``) before collecting evidence."""
    return sha256_json(sorted([i, sorted(item_tags.get(i, []))] for i in set(item_ids)))


def evidence_from_wsp(
    run_dir: Path,
    protocol: Protocol,
    study: Study,
    arm_map: Mapping[str, str],
    item_tags: Mapping[str, Sequence[str]],
    edit_report: Mapping[str, Any] | None = None,
) -> Evidence:
    """Translate a verified WSP run into accuracy evidence for one weight-edit comparison.

    ``arm_map`` maps exactly two WSP arms to study arms: the WSP ``reference`` arm to the
    baseline and a WSP ``candidate``-role arm to one weight-edit candidate. ``item_tags``
    supplies slice membership, which WSP items do not carry; untagged items belong only to the
    ``all`` scope. ``edit_report`` is the ``wsp.py verify-edit`` output for that candidate;
    without a matching report the comparison stops at ``prerequisite_missing``.
    """
    if study.task_kind != "accuracy":
        raise EvidenceError("WSP evidence is accuracy evidence; the study task differs")
    unknown = sorted(set(arm_map.values()) - set(study.arm_ids))
    if unknown:
        raise EvidenceError(f"arm_map targets not registered in the study: {unknown}")
    roles = {a.id: a.role for a in protocol.arms}
    base_arms = [w for w, s in arm_map.items() if s == study.baseline["id"]]
    if len(base_arms) != 1 or roles.get(base_arms[0]) != "reference":
        raise EvidenceError("exactly the WSP reference arm must map to the study baseline")
    cand_arms = [w for w, s in arm_map.items() if s != study.baseline["id"]]
    if len(cand_arms) != 1:
        raise EvidenceError("map exactly one WSP arm to one study candidate")
    if roles.get(cand_arms[0]) != "candidate":
        raise EvidenceError("the study candidate must map to a WSP arm with role 'candidate'")
    cand = study.candidate(arm_map[cand_arms[0]])
    if cand["intervention"]["type"] != "weight_edit":
        raise EvidenceError("WSP evidence is the weight-edit path; the candidate is not one")
    if cand["intervention"].get("evaluation", {}).get("protocol_sha256") != protocol.sha256:
        raise EvidenceError("the weight-edit candidate must register this WSP protocol_sha256")
    if protocol.analysis.get("missing_policy") != "count_as_incorrect":
        raise EvidenceError("the WSP protocol must count missing outcomes as incorrect, as here")
    try:
        manifest, wsp_records = verify_run(run_dir, protocol)
    except RunError as exc:
        raise EvidenceError(f"WSP run failed verification: {exc}") from exc
    declared_data = study.raw["task"]["dataset"].get("content_sha256")
    if declared_data is None or manifest["dataset"].get("items_file_sha256") != declared_data:
        raise EvidenceError("WSP items hash differs from (or is missing in) the study dataset")
    # Slice membership decides the target effect, so it must be the one fixed in the study.
    declared_slice = study.target.get("slice_membership_sha256")
    got_slice = slice_membership_sha256([r["item_id"] for r in wsp_records], item_tags)
    if declared_slice is None or got_slice != declared_slice:
        raise EvidenceError(
            "target slice membership is not the one bound in the study "
            "(target_failure.slice_membership_sha256 missing or different)"
        )
    arms: dict[str, dict[str, Any]] = {}
    for a in manifest["arms"]:
        if a["id"] not in arm_map:
            continue
        loaded = a.get("generator", {}).get("artifacts_sha256") or {}
        arms[arm_map[a["id"]]] = {
            "artifacts": dict(loaded),
            "artifact_sha256s": sorted(set(loaded.values())),
            "config_sha256": sha256_json({"inference": a["inference"], "engine": a["engine"]}),
            "evaluator": a.get("generator", {}),
        }
    records = tuple(
        {
            "run_id": r["run_id"],
            "arm": arm_map[r["arm"]],
            "item_id": r["item_id"],
            "tags": list(item_tags.get(r["item_id"], [])),
            "cluster": None,
            "sample": r["sample"],
            "status": r["status"],
            "correct": r["correct"],
            "error_class": r["error_class"],
        }
        for r in wsp_records
        if r["arm"] in arm_map
    )
    verification = {"candidate": cand["id"], "status": "no_report_supplied"}
    if edit_report is not None:
        verification = _match_edit_report(study, cand, edit_report)
    prereg = wsp_preregistration(protocol.raw, manifest)
    return Evidence(
        run_id=manifest["run_id"],
        source="wsp",
        synthetic=bool(manifest.get("synthetic", True)),
        started_at=manifest.get("started_at"),
        arms=arms,
        records=records,
        limitations=tuple(protocol.analysis.get("required_limitations", [])),
        notes={
            "wsp_protocol_sha256": protocol.sha256,
            "wsp_protocol_frozen": protocol.frozen,
            "wsp_protocol_preregistered": prereg["preregistered"],
            "wsp_split": manifest.get("split"),
            "edit_verification": verification,
        },
    )


def _match_edit_report(
    study: Study, cand: dict[str, Any], report: Mapping[str, Any]
) -> dict[str, Any]:
    declared = cand["intervention"]["edit_verification"].get("report_sha256")
    if declared is None or sha256_json(dict(report)) != declared:
        raise EvidenceError("verify-edit report does not hash to the registered report_sha256")
    # The report must name the edit manifest it was checked against (wsp verify-edit
    # manifest_sha256), and that must be the manifest this candidate registered. Reports
    # without that field cannot be linked and are refused (fail closed).
    if "manifest_sha256" not in report:
        raise EvidenceError(
            "verify-edit report has no manifest_sha256; its manifest linkage cannot be checked"
        )
    if report["manifest_sha256"] != cand["intervention"]["edit_manifest_sha256"]:
        raise EvidenceError("verify-edit report was checked against a different edit manifest")
    if report.get("status") != "verified":
        raise EvidenceError(f"verify-edit report status is {report.get('status')!r}")
    if report.get("base_file_sha256") != study.baseline["artifact"].get("sha256"):
        raise EvidenceError("verify-edit base file is not the registered baseline artifact")
    if report.get("candidate_file_sha256") != cand["artifact"].get("sha256"):
        raise EvidenceError("verify-edit candidate file is not the registered candidate artifact")
    return {
        "candidate": cand["id"],
        "status": "hash_consistent",
        "report_sha256": declared,
        "meaning": (
            "the supplied verify-edit report hashes to the registered value, says verified, "
            "and names the registered base and candidate files and edit manifest; not "
            "authenticated and not an independent re-verification"
        ),
    }
