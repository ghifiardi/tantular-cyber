"""Evaluation protocol: arms, inference settings, sampling plan and analysis plan.

A protocol is a JSON document. ``protocol_sha256`` is the canonical hash of the whole
document and is recorded in every run manifest, so any change after freezing is visible.
A frozen protocol may not contain placeholders. Replication arms must share identical
inference and engine settings; alternative interventions (prompt/decoding, LoRA) are declared
as such and are excluded from that check. A no-edit reload arm must declare the reference
arm's checkpoint exactly.

Sampling modes:

* ``per_sample_seed``: one request per sample, ``seeds[k]`` for sample ``k``;
* ``n_per_request``: one request per item returning ``samples_per_item`` samples under a single
  request seed (``seeds == [seed]``), as vLLM ``SamplingParams(n=..., seed=...)`` does.

A frozen protocol also names who froze it and when (``frozen_by`` pseudonym, ``frozen_at`` UTC
timestamp). The analysis treats its decision rule as preregistered only when ``frozen_at`` is not
later than the run's start; the analysis arms must be the reference arm and a candidate (or
declared alternative) arm, never the no-edit reload.

The optional ``drift`` section holds the reload/drift policy (docs/wsp/exp8-replication-
protocol.md). Its tolerance is ``unapproved`` until a named person approves numeric bounds in a
frozen protocol; an unfrozen protocol cannot carry an approved tolerance.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .hashing import contains_placeholder, sha256_json

ARM_ROLES = ("reference", "candidate", "no_edit_reload", "alternative")
REPLICATION_ROLES = ("reference", "candidate", "no_edit_reload")
SPLITS = ("development", "heldout")
SAMPLING_MODES = ("per_sample_seed", "n_per_request")
MISSING_POLICIES = ("exclude_item", "count_as_incorrect")
TOLERANCE_METRICS = (
    "max_abs_accuracy_diff",
    "max_abs_token_limit_rate_diff",
    "max_abs_error_rate_diff",
)
TOLERANCE_KEYS = ("status", *TOLERANCE_METRICS, "approved_by", "approved_at", "note")
_PSEUDONYM = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?Z$")


class ProtocolError(ValueError):
    pass


def missing_policy(analysis: dict[str, Any]) -> str:
    """The protocol's missing-outcome policy, or ProtocolError (a ValueError) if it is missing
    or unknown. parse_protocol, analyze() and the drift report all go through this check, so
    a plan that cannot be executed is rejected the same way everywhere."""
    policy = analysis.get("missing_policy") if isinstance(analysis, dict) else None
    if policy not in MISSING_POLICIES:
        raise ProtocolError(f"analysis.missing_policy must be one of {MISSING_POLICIES}")
    return policy


@dataclass(frozen=True)
class ArmSpec:
    id: str
    role: str
    checkpoint: dict[str, Any]
    inference: dict[str, Any]
    engine: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Protocol:
    raw: dict[str, Any]
    protocol_version: str
    frozen: bool
    synthetic: bool
    arms: tuple[ArmSpec, ...]
    samples_per_item: int
    seeds: tuple[int, ...]
    dataset: dict[str, Any]
    grader: dict[str, Any]
    analysis: dict[str, Any]
    sampling_mode: str = "per_sample_seed"
    drift: dict[str, Any] = field(default_factory=dict)
    sha256: str = field(default="")

    def arm(self, arm_id: str) -> ArmSpec:
        for a in self.arms:
            if a.id == arm_id:
                return a
        raise ProtocolError(f"unknown arm {arm_id!r}")


def _require(d: dict[str, Any], key: str, kind: type | tuple[type, ...]) -> Any:
    if key not in d:
        raise ProtocolError(f"missing field {key!r}")
    kinds = kind if isinstance(kind, tuple) else (kind,)
    # bool is a subclass of int: reject True/False where a number is expected.
    if not isinstance(d[key], kind) or (isinstance(d[key], bool) and bool not in kinds):
        raise ProtocolError(f"field {key!r} has wrong type")
    return d[key]


def parse_protocol(raw: dict[str, Any]) -> Protocol:
    if not isinstance(raw, dict):
        raise ProtocolError("protocol must be a JSON object")
    version = _require(raw, "protocol_version", str)
    frozen = _require(raw, "frozen", bool)
    synthetic = _require(raw, "synthetic", bool)
    samples = _require(raw, "samples_per_item", int)
    seeds = tuple(_require(raw, "seeds", list))
    dataset = _require(raw, "dataset", dict)
    grader = _require(raw, "grader", dict)
    analysis = _require(raw, "analysis", dict)
    arms_raw = _require(raw, "arms", list)

    if samples < 1:
        raise ProtocolError("samples_per_item must be >= 1")
    mode = raw.get("sampling_mode", "per_sample_seed")
    if mode not in SAMPLING_MODES:
        raise ProtocolError(f"sampling_mode must be one of {SAMPLING_MODES}")
    if not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds):
        raise ProtocolError("seeds must be integers")
    if mode == "per_sample_seed" and len(seeds) != samples:
        raise ProtocolError("seeds must list exactly one integer per sample")
    if mode == "n_per_request" and len(seeds) != 1:
        raise ProtocolError("n_per_request uses exactly one request seed")
    if len(set(seeds)) != len(seeds):
        raise ProtocolError("seeds must be distinct")

    arms: list[ArmSpec] = []
    for a in arms_raw:
        if not isinstance(a, dict):
            raise ProtocolError("each arm must be an object")
        arm = ArmSpec(
            id=_require(a, "id", str),
            role=_require(a, "role", str),
            checkpoint=_require(a, "checkpoint", dict),
            inference=_require(a, "inference", dict),
            engine=a.get("engine", {}),
        )
        if not isinstance(arm.engine, dict):
            raise ProtocolError(f"arm {arm.id!r}: engine must be an object")
        if arm.role not in ARM_ROLES:
            raise ProtocolError(f"arm {arm.id!r}: role must be one of {ARM_ROLES}")
        arms.append(arm)
    ids = [a.id for a in arms]
    if len(set(ids)) != len(ids):
        raise ProtocolError("arm ids must be unique")
    if sum(a.role == "reference" for a in arms) != 1:
        raise ProtocolError("exactly one reference arm is required")

    replication = [a for a in arms if a.role in REPLICATION_ROLES]
    first = replication[0].inference
    for a in replication[1:]:
        if a.inference != first or a.engine != replication[0].engine:
            raise ProtocolError(
                f"replication arm {a.id!r} has different inference or engine settings than "
                f"{replication[0].id!r}; declare it role 'alternative' instead"
            )
    reference = next(a for a in arms if a.role == "reference")
    for a in arms:
        if a.role == "no_edit_reload" and a.checkpoint != reference.checkpoint:
            raise ProtocolError(
                f"no-edit reload arm {a.id!r} must declare the reference checkpoint"
            )

    drift = raw.get("drift", {})
    _validate_drift(drift, frozen, [a.id for a in arms if a.role == "no_edit_reload"])

    for key in ("reference_arm", "candidate_arm", "primary_endpoint"):
        _require(analysis, key, str)
    # Validated here, before any run: a run under a protocol whose analysis plan cannot be
    # executed would be unanalyzable, and fixing the plan afterwards changes the protocol hash.
    missing_policy(analysis)
    for key in ("bootstrap_resamples", "bootstrap_seed"):
        _require(analysis, key, int)
    if "min_meaningful_effect" not in analysis:
        raise ProtocolError("missing field 'min_meaningful_effect'")
    if analysis["min_meaningful_effect"] is None:
        # Undecided is allowed only in a draft; it is never filled in by the harness.
        if frozen:
            raise ProtocolError("a frozen protocol needs a numeric min_meaningful_effect")
    else:
        _require(analysis, "min_meaningful_effect", (int, float))
    if analysis["reference_arm"] not in ids or analysis["candidate_arm"] not in ids:
        raise ProtocolError("analysis arms must name declared arms")
    roles = {a.id: a.role for a in arms}
    if roles[analysis["reference_arm"]] != "reference":
        raise ProtocolError("analysis.reference_arm must be the arm with role 'reference'")
    if roles[analysis["candidate_arm"]] not in ("candidate", "alternative"):
        raise ProtocolError("analysis.candidate_arm must have role 'candidate' or 'alternative'")
    if analysis["primary_endpoint"] not in ("accuracy",):
        raise ProtocolError("only the 'accuracy' primary endpoint is implemented")

    if frozen:
        holes = contains_placeholder(raw)
        if holes:
            raise ProtocolError(f"frozen protocol still has placeholders at: {holes}")
        if not (isinstance(raw.get("frozen_by"), str) and _PSEUDONYM.match(raw["frozen_by"])):
            raise ProtocolError("a frozen protocol needs frozen_by (pseudonym)")
        if not (isinstance(raw.get("frozen_at"), str) and _TIMESTAMP.match(raw["frozen_at"])):
            raise ProtocolError("a frozen protocol needs frozen_at (UTC, e.g. 2026-11-13T10:00Z)")
    elif raw.get("frozen_by") or raw.get("frozen_at"):
        raise ProtocolError("an unfrozen protocol must not carry frozen_by / frozen_at")

    proto = Protocol(
        raw=raw,
        protocol_version=version,
        frozen=frozen,
        synthetic=synthetic,
        arms=tuple(arms),
        samples_per_item=samples,
        seeds=seeds,
        dataset=dataset,
        grader=grader,
        analysis=analysis,
        sampling_mode=mode,
        drift=drift,
    )
    object.__setattr__(proto, "sha256", sha256_json(raw))
    return proto


def _validate_drift(drift: Any, frozen: bool, reload_arms: list[str]) -> None:
    if not isinstance(drift, dict):
        raise ProtocolError("drift must be an object")
    if not drift:
        return
    if not reload_arms:
        raise ProtocolError("a drift policy needs at least one no_edit_reload arm")
    tol = drift.get("tolerance")
    if not isinstance(tol, dict) or tol.get("status") not in ("unapproved", "approved"):
        raise ProtocolError("drift.tolerance.status must be 'unapproved' or 'approved'")
    unknown = sorted(set(tol) - set(TOLERANCE_KEYS))
    if unknown:
        # e.g. a "proposed_accuracy_bound" next to an unapproved status would read as a bound.
        raise ProtocolError(f"drift.tolerance has unknown keys: {unknown}")
    if tol["status"] == "unapproved":
        set_values = [m for m in TOLERANCE_METRICS if tol.get(m) is not None]
        if set_values or tol.get("approved_by") or tol.get("approved_at"):
            # Proposed numbers belong in the protocol text, not in fields that look approved.
            raise ProtocolError("an unapproved tolerance must not carry bounds or approval fields")
        return
    if not frozen:
        raise ProtocolError("an approved drift tolerance requires a frozen protocol")
    for m in TOLERANCE_METRICS:
        v = tol.get(m)
        if isinstance(v, bool) or not isinstance(v, int | float) or not 0 <= v <= 1:
            raise ProtocolError(f"approved tolerance needs {m} in [0, 1]")
    if not (isinstance(tol.get("approved_by"), str) and _PSEUDONYM.match(tol["approved_by"])):
        raise ProtocolError("approved tolerance needs approved_by (pseudonym)")
    if not (isinstance(tol.get("approved_at"), str) and _TIMESTAMP.match(tol["approved_at"])):
        raise ProtocolError("approved tolerance needs approved_at (UTC timestamp)")


def load_protocol(path: Path) -> Protocol:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"{path}: invalid JSON ({exc.msg})") from exc
    return parse_protocol(raw)
