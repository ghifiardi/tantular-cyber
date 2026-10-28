"""Benchmark corpus manifest: cases, ground-truth annotations, thresholds, and validation.

Ground truth lives only in ``manifest.json`` — never inside case files, which the LLM reads.
Every annotation carries an ``anchor`` (exact text on its last line) so edits to a case
file that move lines are caught by ``validate_manifest``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

SUPPORTED_MANIFEST_MAJOR = 1

Truth = Literal["vulnerable", "safe"]


class Annotation(BaseModel):
    id: str
    truth: Truth = Field(description="Whether the code at this location is actually vulnerable.")
    cwe: str
    line: int = Field(ge=1, description="First line of the vulnerable construct.")
    end_line: int = Field(ge=1, description="Last line (the sink); must contain ``anchor``.")
    anchor: str = Field(min_length=1, description="Exact text expected on ``end_line``.")
    rule_id: str | None = None
    expect_detected: bool = Field(
        description="Regression baseline: whether the current ruleset is expected to flag "
        "this location. truth=safe + expect_detected=true is a known false positive; "
        "truth=vulnerable + expect_detected=false is a known false negative."
    )
    preconditions: list[str] = Field(default_factory=list)
    note: str = ""

    @property
    def label(self) -> str:
        return {
            ("vulnerable", True): "true_positive",
            ("vulnerable", False): "known_false_negative",
            ("safe", True): "known_false_positive",
            ("safe", False): "true_negative",
        }[(self.truth, self.expect_detected)]


class Case(BaseModel):
    id: str
    path: str = Field(description="POSIX path relative to the cases root.")
    description: str
    tags: list[str] = Field(default_factory=list)
    llm_eval: bool = Field(default=False, description="Include in LLM explanation evaluation.")
    injection_canary: str | None = Field(
        default=None, description="Word an embedded instruction asks the model to output."
    )
    redaction_canaries: list[str] = Field(
        default_factory=list, description="Values that must never appear in any output."
    )
    annotations: list[Annotation]


class Thresholds(BaseModel):
    min_precision: float = Field(ge=0, le=1)
    min_recall: float = Field(ge=0, le=1)
    require_expected_detections: bool = True
    max_unannotated_findings: int = Field(default=0, ge=0)


class Manifest(BaseModel):
    manifest_version: str
    corpus_id: str
    description: str
    language: str
    cases_root: str = "cases"
    thresholds: Thresholds
    cases: list[Case]

    def annotations(self) -> list[tuple[Case, Annotation]]:
        return [(c, a) for c in self.cases for a in c.annotations]


class ManifestError(ValueError):
    pass


def load_manifest(corpus_dir: Path) -> tuple[Manifest, Path]:
    path = corpus_dir / "manifest.json"
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")
    try:
        manifest = Manifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ManifestError(f"invalid manifest: {exc}") from exc
    major = manifest.manifest_version.split(".", 1)[0]
    if not major.isdigit() or int(major) != SUPPORTED_MANIFEST_MAJOR:
        raise ManifestError(f"unsupported manifest_version {manifest.manifest_version}")
    return manifest, (corpus_dir / manifest.cases_root).resolve()


def _split_lines(text: str) -> list[str]:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def validate_manifest(manifest: Manifest, cases_root: Path) -> list[str]:
    """Return a list of problems (empty when the manifest is consistent with the corpus)."""
    problems: list[str] = []
    case_ids = [c.id for c in manifest.cases]
    ann_ids = [a.id for _, a in manifest.annotations()]
    for kind, ids in (("case", case_ids), ("annotation", ann_ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            problems.append(f"duplicate {kind} ids: {', '.join(dupes)}")

    listed: set[str] = set()
    for case in manifest.cases:
        target = (cases_root / case.path).resolve()
        if cases_root not in target.parents:
            problems.append(f"{case.id}: path escapes the cases root")
            continue
        if not target.is_file():
            problems.append(f"{case.id}: file not found: {case.path}")
            continue
        listed.add(target.relative_to(cases_root).as_posix())
        if case.llm_eval and not any(a.expect_detected for a in case.annotations):
            problems.append(f"{case.id}: llm_eval requires at least one expected detection")
        lines = _split_lines(target.read_text(encoding="utf-8"))
        for a in case.annotations:
            if a.end_line < a.line:
                problems.append(f"{a.id}: end_line < line")
            elif a.end_line > len(lines):
                problems.append(f"{a.id}: end_line {a.end_line} beyond end of {case.path}")
            elif a.anchor not in lines[a.end_line - 1]:
                problems.append(f"{a.id}: anchor not found on {case.path}:{a.end_line}")
            if a.truth == "vulnerable" and not a.preconditions:
                problems.append(f"{a.id}: vulnerable annotation needs preconditions")
        text = target.read_text(encoding="utf-8")
        if case.injection_canary and case.injection_canary not in text:
            problems.append(f"{case.id}: injection canary not present in case file")
        for canary in case.redaction_canaries:
            if canary not in text:
                problems.append(f"{case.id}: redaction canary not present in case file")

    on_disk = {
        p.relative_to(cases_root).as_posix()
        for p in cases_root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    for extra in sorted(on_disk - listed):
        problems.append(f"unlisted corpus file: {extra}")
    return problems


def corpus_digest(corpus_dir: Path) -> str:
    """sha256 over (relative path, file sha256) of every corpus file, for provenance."""
    h = hashlib.sha256()
    for p in sorted(corpus_dir.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(p.relative_to(corpus_dir).as_posix().encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()
