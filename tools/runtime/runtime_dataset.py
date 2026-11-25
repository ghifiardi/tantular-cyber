"""Offline runtime-explanation dataset tooling (separate from the D1 source dataset).

No model, network, download, training or fixture execution. Subcommands:

    validate  EXAMPLES.json --manifest NEW.json      gates + split/leakage checks
    export    MANIFEST.json --out NEW.jsonl --dry-run  reviewed train records -> chat JSONL
    run-manifest --out NEW.json                       training run manifest + prerequisites

Every example stores the imported ``RuntimeEvidenceReport`` and the retrieved chunk ids; the
request is rebuilt with the serving ``build_runtime_messages`` so ``input_sha256`` proves the
example matches what production would send. Unreviewed drafts never reach an export. The
held-out/transfer runtime suite and the scanner benchmark corpus never enter training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tantular_cyber.runtime.analysis import (
    RUNTIME_EXPLANATION_JSON_SCHEMA,
    RuntimeExplanation,
    build_runtime_messages,
    enforce_runtime_claim_policy,
    prompt_version,
    select_records,
)
from tantular_cyber.runtime.evaluation import load_suite
from tantular_cyber.runtime.models import RuntimeEvidenceReport
from tantular_cyber.runtime.retrieval import chunk_for_prompt, load_corpus
from tantular_cyber.security import redact

RUNTIME_DATASET_VERSION = "1.0"
PSEUDONYM = r"^[a-zA-Z][a-zA-Z0-9_-]{1,63}$"
_INVISIBLE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u2060\ufeff\u202a-\u202e\u2066-\u2069]"
)


def canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RuntimeReview(_Strict):
    reviewer: str = Field(pattern=PSEUDONYM)
    evidence_fidelity: Literal["yes", "partial", "no"]
    attempt_outcome_distinction: Literal["yes", "no"]
    unsupported_claim_present: Literal["yes", "no"]
    indonesian_clarity: int = Field(ge=1, le=5)


class RuntimeExample(_Strict):
    id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,95}$")
    split: Literal["train", "dev", "test"]
    family: str = Field(min_length=1, max_length=96)
    scenario: Literal["benign", "suspicious", "ambiguous", "coverage_gap", "adversarial"]
    synthetic: bool
    evidence: dict[str, Any]
    chunk_ids: list[str] = Field(default_factory=list)
    corpus_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target: dict[str, Any]
    target_origin: Literal["human_written", "ai_draft_human_edited"]
    author: str = Field(pattern=PSEUDONYM)
    reviews: list[RuntimeReview] = Field(default_factory=list)
    review_state: Literal["draft", "reviewed", "rejected"]
    license: str = Field(min_length=1)
    license_review_by: str | None = Field(default=None, pattern=PSEUDONYM)
    copied_third_party: bool = False
    attribution: str | None = None
    generator_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    prompt_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    redaction_canaries: list[str] = Field(default_factory=list)
    injection_canaries: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent(self) -> RuntimeExample:
        reviewers = [r.reviewer for r in self.reviews]
        if self.author in reviewers or len(set(reviewers)) != len(reviewers):
            raise ValueError("reviewers must be distinct and independent of the author")
        if self.review_state == "reviewed" and not self.reviews:
            raise ValueError("reviewed examples need at least one review")
        if self.review_state == "reviewed" and not self.license_review_by:
            raise ValueError("reviewed examples need a licence review pseudonym")
        if self.copied_third_party and not self.attribution:
            raise ValueError("copied third-party material needs attribution")
        if any(not c.startswith("tc-synthetic-") for c in self.redaction_canaries):
            raise ValueError("redaction canaries must be synthetic")
        if bool(self.chunk_ids) != bool(self.corpus_sha256):
            raise ValueError("chunk_ids and corpus_sha256 go together")
        return self


def strict_target(target: dict[str, Any]) -> RuntimeExplanation:
    if set(target) != set(RUNTIME_EXPLANATION_JSON_SCHEMA["required"]):
        raise ValueError("target fields must exactly match the runtime explanation schema")
    parsed = RuntimeExplanation.model_validate(target)
    if parsed.model_dump(mode="json") != target:
        raise ValueError("target must not require normalization")
    return parsed


def _chunks(example: RuntimeExample, corpus: dict[str, Any] | None) -> list[dict[str, str]]:
    if not example.chunk_ids:
        return []
    if corpus is None or corpus["corpus_sha256"] != example.corpus_sha256:
        raise ValueError("example corpus does not match the loaded corpus")
    by_id = {c["chunk_id"]: c for c in corpus["chunks"]}
    missing = [c for c in example.chunk_ids if c not in by_id]
    if missing:
        raise ValueError("example cites chunks absent from the corpus")
    return [chunk_for_prompt(by_id[c]) for c in example.chunk_ids]


def messages(example: RuntimeExample, corpus: dict[str, Any] | None) -> list[dict[str, str]]:
    evidence = RuntimeEvidenceReport.model_validate(example.evidence)
    return build_runtime_messages(evidence, _chunks(example, corpus))


def gate_results(example: RuntimeExample, corpus: dict[str, Any] | None) -> dict[str, bool]:
    target = strict_target(example.target)
    evidence = RuntimeEvidenceReport.model_validate(example.evidence)
    msgs = messages(example, corpus)
    _, warnings, bad_refs, bad_cites = enforce_runtime_claim_policy(
        target,
        valid_refs={r.ref for r in select_records(evidence)},
        valid_chunks=set(example.chunk_ids),
    )
    answer = canonical(example.target)
    request = "\n".join(m["content"] for m in msgs)
    stored = [answer, canonical(example.evidence)]
    return {
        "input_matches": example.input_sha256 == digest(msgs),
        "prompt_matches": example.prompt_version == prompt_version(),
        "claims_clean": not warnings,
        "refs_valid": not bad_refs,
        "citations_valid": not bad_cites,
        "evidence_unverified": evidence.verification.verified is False,
        "redacted": all(redact(t)[1] == 0 for t in _strings(example.target)),
        "canary_free": all(
            c.casefold() not in t.casefold()
            for c in example.redaction_canaries
            for t in [request, *stored]
        ),
        "injection_not_followed": all(
            c.casefold() not in answer.casefold() for c in example.injection_canaries
        ),
        "invisible_free": not any(_INVISIBLE.search(t) for t in [answer, request]),
        "human_review": example.review_state == "reviewed"
        and all(
            r.evidence_fidelity == "yes"
            and r.attempt_outcome_distinction == "yes"
            and r.unsupported_claim_present == "no"
            and r.indonesian_clarity >= 4
            for r in example.reviews
        ),
        "licence_ok": not example.copied_third_party or bool(example.attribution),
    }


def protected_inputs() -> tuple[set[str], set[str]]:
    """Input SHA-256s and family names that must never be trained on."""
    root, manifest = load_suite()
    shas, families = set(), set()
    for case in manifest["cases"]:
        if case["split"] in ("heldout", "transfer"):
            shas.add(hashlib.sha256((root / case["path"]).read_bytes()).hexdigest())
            families.add(case["family"])
    return shas, families


def _scanner_corpus_ids() -> set[str]:
    from tantular_cyber.config import default_corpus

    data = json.loads((default_corpus() / "manifest.json").read_text(encoding="utf-8"))
    return {c["id"] for c in data.get("cases", []) if isinstance(c, dict) and "id" in c}


def _norm_target(target: dict[str, Any]) -> str:
    return " ".join(canonical(target).lower().split())


def build_manifest(examples: list[RuntimeExample], corpus: dict[str, Any] | None) -> dict[str, Any]:
    if not examples:
        raise ValueError("dataset is empty")
    held_shas, held_families = protected_inputs()
    scanner_ids = _scanner_corpus_ids()
    ids: set[str] = set()
    families: dict[str, str] = {}
    targets: dict[str, str] = {}
    inputs: dict[str, str] = {}
    records = []
    for ex in sorted(examples, key=lambda e: e.id):
        if ex.id in ids:
            raise ValueError("duplicate example id")
        ids.add(ex.id)
        if families.setdefault(ex.family, ex.split) != ex.split:
            raise ValueError("scenario family crosses splits")
        if ex.family in held_families or ex.id in scanner_ids:
            raise ValueError(f"{ex.id}: leaks a held-out runtime family or scanner case")
        evidence_input = ex.evidence.get("input", {}).get("sha256")
        if evidence_input in held_shas:
            raise ValueError(f"{ex.id}: evidence comes from the held-out runtime suite")
        norm = _norm_target(ex.target)
        if norm in targets:
            raise ValueError(f"{ex.id}: duplicate target of {targets[norm]}")
        targets[norm] = ex.id
        if inputs.setdefault(ex.input_sha256, ex.split) != ex.split:
            raise ValueError(f"{ex.id}: identical input appears in two splits")
        gates = gate_results(ex, corpus)
        failed = sorted(k for k, v in gates.items() if v is not True and k != "human_review")
        if failed:
            raise ValueError(f"{ex.id}: failed gates: {', '.join(failed)}")
        record = ex.model_dump(mode="json")
        records.append(
            {
                "record": record,
                "record_sha256": digest(record),
                "target_sha256": digest(ex.target),
                "gates": gates,
                "trainable": ex.split == "train" and gates["human_review"],
            }
        )
    return {
        "manifest_version": RUNTIME_DATASET_VERSION,
        "dataset": "tantular-runtime-explanations",
        "training_authorized": False,
        "prompt_version": prompt_version(),
        "corpus_sha256": corpus["corpus_sha256"] if corpus else None,
        "leakage_audit": {
            "heldout_runtime_suite": "checked: families + input hashes excluded",
            "scanner_benchmark": "checked: case ids excluded",
            "external": "pending",
        },
        "records": records,
        "records_sha256": digest(records),
    }


def export_dry_run(manifest: dict[str, Any], corpus: dict[str, Any] | None) -> list[str]:
    """Chat-format JSONL lines for reviewed train records only. Nothing is trained."""
    lines = []
    for item in manifest["records"]:
        if not item["trainable"]:
            continue
        ex = RuntimeExample.model_validate(item["record"])
        msgs = messages(ex, corpus)
        lines.append(
            canonical(
                {
                    "id": ex.id,
                    "messages": [*msgs, {"role": "assistant", "content": canonical(ex.target)}],
                    "record_sha256": item["record_sha256"],
                }
            )
        )
    return lines


def run_manifest() -> dict[str, Any]:
    return {
        "run_manifest_version": "1.0",
        "status": "not_runnable: prerequisites unresolved",
        "training_authorized": False,
        "objective": "LoRA/adapter on reviewed runtime explanations, only after measured "
        "insufficiency of prompt/retrieval interventions (rung-3 plan)",
        "base_checkpoint": {
            "id": "UNRESOLVED",
            "requirements": [
                "editable safetensors weights (an Ollama profile or GGUF is not trainable here)",
                "verified lineage to the served tantular-cyber weights",
                "compatible tokenizer + chat template matching the serving prompt",
                "licence permitting fine-tuning and redistribution of adapters",
            ],
            "not_allowed": [
                "Office Qwen checkpoints (different product)",
                "Antares-1B GGUF (loops on '@')",
            ],
        },
        "backends": {
            "dry_run_export": "exercised (this tool)",
            "transformers+peft": "unvalidated (not installed; no checkpoint chosen)",
            "mlx-lm": "unvalidated",
            "ollama": "serving only; cannot train",
        },
        "data": {
            "source": "runtime_dataset.py validate manifest (reviewed train split only)",
            "excluded": ["benchmarks/runtime heldout+transfer", "benchmarks/corpus (scanner)"],
        },
        "evaluation_entry_point": "tantular-cyber runtime eval --llm ollama --model <candidate> "
        "--runs 3 --splits heldout,transfer",
        "candidate_provenance_required": [
            "base checkpoint id + sha256",
            "adapter sha256",
            "training data manifest sha256",
            "hyperparameters + seed",
            "hardware",
            "quantization of the served candidate",
            "Ollama digest of the served candidate",
            "prompt + corpus sha256",
        ],
        "prerequisites_pending": [
            "compute budget",
            "chosen editable checkpoint + licence review",
            "~150 reviewed runtime examples with independent reviewers",
            "external-data egress decision (none assumed)",
            "baseline vs retrieval comparison showing insufficiency",
        ],
        "never": [
            "paid teacher generation",
            "remote training",
            "model download",
            "production model replacement",
        ],
    }


def _write_new(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("examples", type=Path)
    v.add_argument("--manifest", type=Path, required=True)
    v.add_argument("--corpus", type=Path, default=None, help="default: bundled corpus")
    e = sub.add_parser("export")
    e.add_argument("manifest", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--corpus", type=Path, default=None)
    e.add_argument("--dry-run", action="store_true", required=True)
    r = sub.add_parser("run-manifest")
    r.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "run-manifest":
            _write_new(args.out, json.dumps(run_manifest(), indent=2) + "\n")
            return 0
        corpus = load_corpus(args.corpus)
        if args.cmd == "validate":
            raw = json.loads(args.examples.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("examples must be a JSON array")
            manifest = build_manifest([RuntimeExample.model_validate(x) for x in raw], corpus)
            _write_new(args.manifest, canonical(manifest) + "\n")
            n = sum(1 for x in manifest["records"] if x["trainable"])
            print(f"validated {len(manifest['records'])} example(s); trainable={n}")
            return 0
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        lines = export_dry_run(manifest, corpus)
        _write_new(args.out, "".join(line + "\n" for line in lines))
        print(f"dry-run export: {len(lines)} reviewed train record(s); nothing was trained")
        return 0
    except (ValueError, ValidationError, OSError) as exc:
        if isinstance(exc, ValidationError) or not isinstance(exc, ValueError):
            detail = type(exc).__name__
        else:
            detail = str(exc).split("\n")[0]
        print(f"rejected: {detail[:200]} (no source content logged)", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
