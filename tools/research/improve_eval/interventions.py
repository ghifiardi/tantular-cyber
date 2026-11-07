"""Intervention taxonomy. Each candidate is exactly one intervention type; types are never mixed.

The four families are kept apart because they change different things and need different
evidence before any comparison means anything:

* ``configuration``: ``prompt_or_threshold`` and ``feature_change``. Same model artifact as
  the baseline; only settings, thresholds, prompts or the feature contract change.
* ``training``: ``conventional_training`` and ``lora``. New weights, or an adapter on the
  baseline weights, produced by training on declared data under a declared configuration.
* ``soc_triage_model``: a separate alert-triage model (e.g. an anomaly scorer with an operating
  point). Its operating point must be calibrated on data disjoint from the evaluation set.
* ``weight_edit``: a direct edit of baseline weights. The edit must be verified at tensor
  level (the WSP harness ``verify-edit``) before its evaluation is interpreted.

A combined intervention (say, a LoRA plus a new threshold) is not one candidate; register the
parts separately or define a new study.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InterventionType:
    name: str
    family: str
    changes_weights: bool
    required: tuple[str, ...]
    note: str


TYPES: dict[str, InterventionType] = {
    t.name: t
    for t in (
        InterventionType(
            "prompt_or_threshold",
            "configuration",
            False,
            ("change",),
            "settings/prompt/threshold change on the baseline artifact",
        ),
        InterventionType(
            "feature_change",
            "configuration",
            False,
            ("change", "feature_contract_sha256"),
            "feature contract or pipeline change on the baseline artifact",
        ),
        InterventionType(
            "conventional_training",
            "training",
            True,
            ("training_data", "training_config_sha256", "compute_budget"),
            "new weights from training; needs its own data and budget approval",
        ),
        InterventionType(
            "lora",
            "training",
            True,
            ("base_artifact_sha256", "training_data", "training_config_sha256", "compute_budget"),
            "adapter trained on the baseline weights",
        ),
        InterventionType(
            "soc_triage_model",
            "soc_triage_model",
            True,
            ("model_family", "feature_contract_sha256", "training_data", "calibration"),
            "separate triage model with an operating point calibrated on disjoint data",
        ),
        InterventionType(
            "weight_edit",
            "weight_edit",
            True,
            ("base_artifact_sha256", "edit_manifest_sha256", "edit_verification"),
            "direct weight edit; tensor-level verification (WSP verify-edit) required",
        ),
    )
}
FAMILIES = tuple(sorted({t.family for t in TYPES.values()}))
EDIT_VERIFICATION_STATUSES = ("verified", "failed", "not_run")
