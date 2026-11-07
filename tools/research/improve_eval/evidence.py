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

**Run-manifest identifier and what it does not prove.** (Not the edit-manifest hash above.)
``records_sha256`` covers only the records. Fields read from ``manifest.json`` itself
(``started_at``, each arm's artifact digests and configuration, ``synthetic``, for triage
``limitations``, and for WSP the ``split``) are not covered by it. Both loaders therefore
compute :func:`run_manifest_digest`, a deterministic canonical-JSON hash of the parsed
manifest, as an integrity and provenance *identifier* of the run. A hash written next to the
manifest would protect nothing (whoever edits the manifest can recompute it), so none is stored
there. To detect post-run edits, pass ``expected_run_manifest_sha256``: a value recorded
*outside* the run directory when the run finished (for example in an evidence register). The
loader refuses a mismatch, and the evidence is marked ``run_manifest_check:
matched_external_record``; otherwise ``not_checked``. Even a match is **not authentication**:
it shows only that the manifest equals what that external record says, and is as trustworthy as
the access controls or signatures protecting the record, which this workflow does not provide.

Malformed input of any shape makes the loaders raise :class:`EvidenceError`; no other
exception type escapes them.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wsp_eval.analysis import preregistration as wsp_preregistration
from wsp_eval.hashing import sha256_file, sha256_json
from wsp_eval.protocol import Protocol, ProtocolError, parse_protocol
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


def record_problem(r: Any) -> str | None:
    """Ordinary-input validation of one record, before any field is used (not authentication).

    Types are checked exactly (``type(x) is str``), so subclasses that change hashing,
    equality or membership are refused. The values themselves remain as asserted.
    """
    if type(r) is not dict:
        return "a record is not an object"
    if not (type(r.get("arm")) is str and type(r.get("item_id")) is str):
        return "record arm and item_id must be strings"
    item = r["item_id"]
    if type(r.get("sample")) is not int:
        return f"item {item}: sample must be an integer"
    if type(r.get("status")) is not str:
        return f"item {item}: status must be a string"
    tags = r.get("tags", [])
    if type(tags) not in (list, tuple) or not all(type(t) is str for t in tags):
        return f"item {item}: tags must be a list of strings"
    for key in ("cluster", "label", "decision", "error_class", "run_id"):
        v = r.get(key)
        if v is not None and type(v) is not str:
            return f"item {item}: {key} must be a string or null"
    if r.get("correct") is not None and type(r["correct"]) is not bool:
        return f"item {item}: correct must be a boolean or null"
    return None


def parse_timestamp(s: Any) -> datetime:
    """Parse an ISO-8601 timestamp to UTC; EvidenceError if it is not one."""
    if type(s) is not str:
        raise EvidenceError("timestamp must be an ISO-8601 string")
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceError(f"timestamp {s!r} is not ISO-8601") from exc
    if t.tzinfo is None:
        raise EvidenceError(f"timestamp {s!r} has no time zone")
    try:
        return t.astimezone(UTC)
    except OverflowError as exc:  # e.g. 0001-01-01T00:00+01:00 is before datetime.min in UTC
        raise EvidenceError(f"timestamp {s!r} is out of range in UTC") from exc


def _json_problem(value: Any) -> str | None:
    """Whether a value is plain JSON data with string keys (what the report serializes)."""
    stack = [value]
    while stack:
        v = stack.pop()
        if type(v) is dict:
            if not all(type(k) is str for k in v):
                return "not plain JSON data with string keys"
            stack.extend(v.values())
        elif type(v) in (list, tuple):
            stack.extend(v)
        elif v is not None and type(v) not in (str, int, float, bool):
            return "not plain JSON data with string keys"
    try:
        sha256_json(value)
    except (TypeError, ValueError, RecursionError):
        return "not plain JSON data with string keys"
    return None


def _arm_problem(arm: str, v: Any) -> str | None:
    if type(v) is not dict:
        return f"evidence arm {arm!r} is not an object"
    arts = v.get("artifacts", {})
    if arts is not None and (
        type(arts) is not dict
        or not all(type(k) is str and type(x) is str for k, x in arts.items())
    ):
        return f"evidence arm {arm!r}: artifacts must map file names to digests"
    shas = v.get("artifact_sha256s", [])
    if type(shas) not in (list, tuple) or not all(type(x) is str for x in shas):
        return f"evidence arm {arm!r}: artifact_sha256s must be a list of strings"
    cfg = v.get("config_sha256")
    if cfg is not None and type(cfg) is not str:
        return f"evidence arm {arm!r}: config_sha256 must be a string or null"
    problem = _json_problem(v)
    return f"evidence arm {arm!r}: {problem}" if problem else None


def evidence_problem(ev: Any) -> str | None:
    """Ordinary-input validation of a (possibly hand-built) Evidence object and its records.

    This lets a direct caller get a gate outcome instead of an exception. It checks shapes
    only: the values are still caller-asserted, and only the loaders bind them to run,
    records, study and dataset hashes.
    """
    if type(ev) is not Evidence:  # exact: a subclass could change what later reads return
        return "evidence is not an Evidence object"
    if type(ev.run_id) is not str or type(ev.source) is not str:
        return "evidence run_id and source must be strings"
    if type(ev.synthetic) is not bool:
        return "evidence synthetic must be a boolean"
    if ev.started_at is not None:
        try:
            parse_timestamp(ev.started_at)
        except EvidenceError as exc:
            return f"evidence started_at: {exc}"
    if type(ev.arms) is not dict or not all(type(k) is str for k in ev.arms):
        return "evidence arms must map arm ids to objects"
    for k, v in ev.arms.items():
        problem = _arm_problem(k, v)
        if problem:
            return problem
    if type(ev.notes) is not dict:
        return "evidence notes must be an object"
    problem = _json_problem(ev.notes)
    if problem:
        return f"evidence notes: {problem}"
    if "edit_verification" in ev.notes and type(ev.notes["edit_verification"]) is not dict:
        return "evidence notes edit_verification must be an object"
    for key in ("wsp_split", "run_manifest_check", "run_manifest_sha256"):
        if ev.notes.get(key) is not None and type(ev.notes[key]) is not str:
            return f"evidence notes {key} must be a string or null"
    if type(ev.limitations) not in (list, tuple) or not all(type(x) is str for x in ev.limitations):
        return "evidence limitations must be a list of strings"
    if type(ev.records) not in (list, tuple):
        return "evidence records must be a list"
    for r in ev.records:
        problem = record_problem(r)
        if problem:
            return problem
    return None


def _check_records(records: Sequence[Any], run_id: str) -> None:
    seen: set[tuple[str, str, int]] = set()
    for r in records:
        problem = record_problem(r)
        if problem:
            raise EvidenceError(problem)
        if r.get("run_id") != run_id:
            raise EvidenceError("records.jsonl contains records from another run")
        key = (r["arm"], r["item_id"], r["sample"])
        if key in seen:
            raise EvidenceError(f"duplicate record for arm/item/sample {key}")
        seen.add(key)


def item_set_sha256(item_ids: Iterable[str]) -> str:
    """Canonical hash of a set of item ids (sorted, de-duplicated, canonical JSON)."""
    return sha256_json(sorted(set(item_ids)))


def triage_metadata_sha256(meta: Mapping[str, tuple[Sequence[str], str | None, str | None]]) -> str:
    """Canonical hash of the original per-alert metadata, independent of arm outcomes.

    ``meta`` maps alert id -> (tags, cluster, label). The canonical form is the sorted list of
    ``[id, sorted(tags), cluster, label]``, encoded as canonical JSON. Features (the
    evaluator's inputs) are covered by the dataset ``content_sha256`` at write time; this hash
    covers what the comparison uses: slice tags, cluster ids and ground-truth labels.
    """
    return sha256_json(
        sorted([i, sorted(tags), cluster, label] for i, (tags, cluster, label) in meta.items())
    )


def records_metadata(
    records: Iterable[dict[str, Any]],
) -> dict[str, tuple[tuple[str, ...], str | None, str | None]]:
    """Per-alert (tags, cluster, label) from records; every arm must agree, or EvidenceError."""
    meta: dict[str, tuple[tuple[str, ...], str | None, str | None]] = {}
    for r in records:
        m = (tuple(sorted(r.get("tags", []))), r.get("cluster"), r.get("label"))
        if meta.setdefault(r["item_id"], m) != m:
            raise EvidenceError(
                f"item {r['item_id']}: tags, cluster or label differ between records"
            )
    return meta


SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_MALFORMED = (
    KeyError,
    IndexError,
    TypeError,
    AttributeError,
    ValueError,
    OverflowError,
    OSError,
    RecursionError,  # e.g. JSON nested thousands of levels deep
)


def run_manifest_digest(manifest: Any) -> str:
    """Deterministic identifier of a parsed run manifest (canonical JSON, ``sha256_json``).

    An integrity and provenance identifier, not authentication: compare it with a value
    recorded outside the run directory to detect later edits (see the module docstring).
    """
    try:
        return sha256_json(manifest)
    except (TypeError, ValueError, RecursionError) as exc:
        raise EvidenceError("manifest.json is not plain JSON data") from exc


def run_manifest_sha256(run_dir: Path) -> str:
    """:func:`run_manifest_digest` of ``run_dir/manifest.json``, to record outside the run."""
    return run_manifest_digest(_read_json(Path(run_dir) / "manifest.json"))


def _run_manifest_check(manifest: Any, expected: str | None) -> dict[str, str]:
    got = run_manifest_digest(manifest)
    if expected is None:
        return {"run_manifest_sha256": got, "run_manifest_check": "not_checked"}
    if type(expected) is not str or not SHA256_RE.fullmatch(expected):
        raise EvidenceError("expected_run_manifest_sha256 must be 64 lowercase hex characters")
    if got != expected:
        raise EvidenceError(
            "manifest.json differs from the externally recorded run_manifest_sha256 (edited after "
            "the run, or a different run)"
        )
    return {"run_manifest_sha256": got, "run_manifest_check": "matched_external_record"}


def _normalized(fn):
    """Run a loader so that malformed input of any shape surfaces as EvidenceError."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Evidence:
        try:
            return fn(*args, **kwargs)
        except EvidenceError:
            raise
        except _MALFORMED as exc:
            raise EvidenceError(f"malformed evidence input ({type(exc).__name__}: {exc})") from exc

    return wrapper


def _parse_json(text: str, where: str) -> Any:
    try:
        return json.loads(text)
    except ValueError as exc:
        raise EvidenceError(f"{where} is not valid JSON") from exc
    except RecursionError as exc:
        raise EvidenceError(f"{where} is nested too deeply to parse") from exc


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceError(f"cannot read {path.name} ({type(exc).__name__})") from exc


def _read_json(path: Path) -> Any:
    return _parse_json(_read_text(path), path.name)


@_normalized
def load_evidence(
    run_dir: Path, study: Study, expected_run_manifest_sha256: str | None = None
) -> Evidence:
    """Load and verify synthetic-triage evidence. Accuracy evidence must come through
    :func:`evidence_from_wsp`, which checks slice membership and the held-out split.

    ``expected_run_manifest_sha256`` is a value recorded outside the run directory; see the
    module docstring for what a match does and does not show."""
    if study.task_kind != "alert_triage":
        raise EvidenceError(
            "load_evidence accepts alert_triage studies only; accuracy evidence must come "
            "through evidence_from_wsp (slice membership and held-out checks)"
        )
    manifest = _read_json(run_dir / "manifest.json")
    if type(manifest) is not dict:
        raise EvidenceError("manifest.json is not an object")
    if manifest.get("source") != "synthetic_triage" or manifest.get("task_kind") != "alert_triage":
        raise EvidenceError("load_evidence accepts synthetic_triage alert_triage evidence only")
    if manifest.get("study_sha256") != study.sha256:
        raise EvidenceError("evidence was collected under a different study (study_sha256)")
    if manifest.get("run_status") != "complete":
        raise EvidenceError(f"run_status is {manifest.get('run_status')!r}; not used")
    check = _run_manifest_check(manifest, expected_run_manifest_sha256)
    path = run_dir / "records.jsonl"
    if not path.is_file():
        raise EvidenceError("records.jsonl is missing")
    if sha256_file(path) != manifest.get("records_sha256"):
        raise EvidenceError("records.jsonl does not match records_sha256 in the manifest")
    records = [_parse_json(x, "records.jsonl") for x in _read_text(path).split("\n") if x.strip()]
    if len(records) != manifest.get("expected_records"):
        raise EvidenceError("record count differs from expected_records in the manifest")
    if type(manifest.get("run_id")) is not str:
        raise EvidenceError("manifest run_id must be a string")
    arms = manifest.get("arms")
    if type(arms) is not dict or not all(
        type(k) is str and type(v) is dict for k, v in arms.items()
    ):
        raise EvidenceError("manifest arms must map arm ids to objects")
    lims = manifest.get("limitations", [])
    if type(lims) is not list or not all(type(x) is str for x in lims):
        raise EvidenceError("manifest limitations must be a list of strings")
    if type(manifest.get("synthetic", True)) is not bool:
        raise EvidenceError("manifest synthetic must be a boolean")
    if not study.synthetic and manifest.get("synthetic", True):
        raise EvidenceError("a non-synthetic study cannot use synthetic evidence")
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
    # Recompute what the comparison relies on from the records themselves: the per-alert
    # metadata must agree across arms and match the registered canonical hash. This checks
    # consistency with the registered dataset; it says nothing about whether labels are true.
    declared_meta = study.raw["task"]["dataset"].get("metadata_sha256")
    if declared_meta is None:
        raise EvidenceError("task.dataset.metadata_sha256 not registered; metadata unverifiable")
    if triage_metadata_sha256(records_metadata(records)) != declared_meta:
        raise EvidenceError("alert tags, clusters or labels in the records differ from the study")
    ev = Evidence(
        run_id=manifest["run_id"],
        source="synthetic_triage",
        synthetic=manifest.get("synthetic", True),
        started_at=manifest.get("started_at"),
        arms=manifest["arms"],
        records=tuple(records),
        limitations=tuple(lims),
        notes=check,
    )
    problem = evidence_problem(ev)
    if problem:
        raise EvidenceError(problem)
    return ev


def slice_membership_sha256(item_ids: Sequence[str], item_tags: Mapping[str, Sequence[str]]) -> str:
    """Hash of which items carry which tags, over exactly ``item_ids``. Register it in the
    study (``target_failure.slice_membership_sha256``) before collecting evidence."""
    return sha256_json(sorted([i, sorted(item_tags.get(i, []))] for i in set(item_ids)))


@_normalized
def evidence_from_wsp(
    run_dir: Path,
    protocol: Protocol,
    study: Study,
    arm_map: Mapping[str, str],
    item_tags: Mapping[str, Sequence[str]],
    edit_report: Mapping[str, Any] | None = None,
    expected_run_manifest_sha256: str | None = None,
) -> Evidence:
    """Translate a verified WSP run into accuracy evidence for one weight-edit comparison.

    ``arm_map`` maps exactly two WSP arms to study arms: the WSP ``reference`` arm to the
    baseline and a WSP ``candidate``-role arm to one weight-edit candidate. ``item_tags``
    supplies slice membership, which WSP items do not carry; untagged items belong only to the
    ``all`` scope. ``edit_report`` is the ``wsp.py verify-edit`` output for that candidate;
    without a matching report the comparison stops at ``prerequisite_missing``.
    ``expected_run_manifest_sha256`` is an externally recorded identifier of the WSP run's
    ``manifest.json``. The split, start time and artifact claims are read from that manifest
    and are bound only by such a record (module docstring).
    """
    if not isinstance(item_tags, Mapping) or not isinstance(arm_map, Mapping):
        raise EvidenceError("arm_map and item_tags must be mappings")
    if not all(type(k) is str and type(v) is str for k, v in arm_map.items()):
        raise EvidenceError("arm_map must map WSP arm ids to study arm ids (strings)")
    if edit_report is not None and (
        not isinstance(edit_report, Mapping) or _json_problem(dict(edit_report))
    ):
        raise EvidenceError("edit_report must be a plain JSON object")
    if study.task_kind != "accuracy":
        raise EvidenceError("WSP evidence is accuracy evidence; the study task differs")
    # W4: use only the protocol as parsed from its own raw text, and require that parse to be
    # exactly the object passed in and to hash to the protocol_sha256 it carries. Roles,
    # samples_per_item, missing_policy, arm settings and the synthetic flag are then read from
    # text the registered protocol_sha256 binds, not from attributes set in memory.
    if not isinstance(protocol, Protocol):
        raise EvidenceError("protocol must be a parsed WSP Protocol")
    try:
        reparsed = parse_protocol(protocol.raw)
    except ProtocolError as exc:
        raise EvidenceError(f"WSP protocol does not parse from its raw text: {exc}") from exc
    if reparsed.sha256 != protocol.sha256 or reparsed != protocol:
        raise EvidenceError(
            "WSP protocol object differs from the parse of its own raw text (protocol_sha256 "
            "would not describe it)"
        )
    protocol = reparsed
    # W2: synthetic-fixture evidence cannot support a real (non-synthetic) study.
    if not study.synthetic and protocol.synthetic:
        raise EvidenceError("a non-synthetic study cannot use evidence from a synthetic protocol")
    if not all(
        type(i) is str and type(t) in (list, tuple) and all(type(x) is str for x in t)
        for i, t in item_tags.items()
    ):
        raise EvidenceError("item_tags must map item ids to lists of strings")
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
    if study.analysis.get("samples_per_item") != protocol.samples_per_item:
        raise EvidenceError(
            f"study analysis.samples_per_item ({study.analysis.get('samples_per_item')}) differs "
            f"from the WSP protocol's samples_per_item ({protocol.samples_per_item})"
        )
    if protocol.analysis.get("missing_policy") != "count_as_incorrect":
        raise EvidenceError("the WSP protocol must count missing outcomes as incorrect, as here")
    try:
        manifest, wsp_records = verify_run(run_dir, protocol)
    except RunError as exc:
        raise EvidenceError(f"WSP run failed verification: {exc}") from exc
    if type(manifest) is not dict:
        raise EvidenceError("WSP manifest is not an object")
    if type(manifest.get("synthetic", True)) is not bool:
        raise EvidenceError("WSP manifest synthetic must be a boolean")
    if not study.synthetic and manifest.get("synthetic", True):
        raise EvidenceError("a non-synthetic study cannot use a synthetic WSP run")
    check = _run_manifest_check(
        _read_json(Path(run_dir) / "manifest.json"), expected_run_manifest_sha256
    )
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
    specs = {a.id: a for a in protocol.arms}
    for a in manifest["arms"]:
        if a["id"] not in arm_map:
            continue
        # W6: the recorded inference and engine settings must be the protocol's registered
        # settings for that arm (the WSP runner writes them from the protocol). This binds the
        # configuration hash to the registered protocol, not only to the run's manifest.
        spec = specs.get(a["id"])
        if spec is None or sha256_json(a["inference"]) != sha256_json(spec.inference):
            raise EvidenceError(
                f"WSP arm {a['id']!r}: recorded inference settings differ from the protocol's"
            )
        if sha256_json(a["engine"]) != sha256_json(spec.engine):
            raise EvidenceError(
                f"WSP arm {a['id']!r}: recorded engine settings differ from the protocol's"
            )
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
    try:
        _check_records(records, manifest["run_id"])
    except EvidenceError as exc:
        raise EvidenceError(f"WSP records: {exc}") from exc
    verification = {"candidate": cand["id"], "status": "no_report_supplied"}
    if edit_report is not None:
        verification = _match_edit_report(study, cand, edit_report)
    prereg = wsp_preregistration(protocol.raw, manifest)
    ev = Evidence(
        run_id=manifest["run_id"],
        source="wsp",
        synthetic=manifest.get("synthetic", True),
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
            **check,
        },
    )
    problem = evidence_problem(ev)
    if problem:
        raise EvidenceError(f"WSP evidence: {problem}")
    return ev


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
