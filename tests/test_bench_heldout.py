"""The held-out evaluation set: valid, label-free, and with no detected overlap with the
development corpus (shared ids, canaries, files, code lines). Overlap checks cannot establish
statistical independence; see docs/benchmark.md."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tantular_cyber.benchmark.manifest import load_manifest, validate_manifest
from tantular_cyber.benchmark.runner import adversarial_suite_dir
from tantular_cyber.config import default_corpus

ROOT = Path(__file__).parents[1] / "benchmarks"
HELDOUT = ROOT / "heldout"
CORPUS = ROOT / "corpus"


def _normalized(path: Path) -> str:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return hashlib.sha256("\n".join(ln for ln in lines if ln).encode()).hexdigest()


def _code_lines(root: Path) -> set[str]:
    out: set[str] = set()
    for f in root.rglob("*.py"):
        for ln in f.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if len(s) >= 30 and not s.startswith(("#", '"""', "import ", "from ")):
                out.add(s)
    return out


def test_heldout_manifest_is_consistent() -> None:
    manifest, cases_root = load_manifest(HELDOUT)
    assert validate_manifest(manifest, cases_root) == []
    assert len(manifest.cases) == 30
    assert sum(c.llm_eval for c in manifest.cases) >= 25


def test_heldout_covers_every_cwe_and_case_kind() -> None:
    manifest, _ = load_manifest(HELDOUT)
    cwes = {a.cwe for c in manifest.cases for a in c.annotations}
    assert cwes == {"CWE-89", "CWE-78", "CWE-22", "CWE-79", "CWE-798"}
    tags = {t for c in manifest.cases for t in c.tags}
    assert {"insufficient_evidence", "prompt_injection", "known_false_positive"} <= tags
    assert sum(1 for c in manifest.cases if c.injection_canary) == 3
    assert all(v.startswith("tc-synthetic-") for c in manifest.cases for v in c.redaction_canaries)


def test_heldout_has_no_detected_overlap_with_the_development_corpus() -> None:
    dev, _ = load_manifest(CORPUS)
    held, _ = load_manifest(HELDOUT)
    assert not {c.id for c in dev.cases} & {c.id for c in held.cases}
    dev_canaries = {c.injection_canary for c in dev.cases} | {
        v for c in dev.cases for v in c.redaction_canaries
    }
    held_canaries = {c.injection_canary for c in held.cases} | {
        v for c in held.cases for v in c.redaction_canaries
    }
    assert not (dev_canaries - {None}) & (held_canaries - {None})
    dev_files = {_normalized(f) for f in (CORPUS / "cases").rglob("*.py")}
    held_files = {_normalized(f) for f in (HELDOUT / "cases").rglob("*.py")}
    assert not dev_files & held_files
    shared = _code_lines(CORPUS / "cases") & _code_lines(HELDOUT / "cases")
    assert not shared, f"code lines copied from the development corpus: {sorted(shared)[:5]}"


def test_heldout_case_files_carry_no_labels() -> None:
    for f in (HELDOUT / "cases").rglob("*.py"):
        text = f.read_text(encoding="utf-8").lower()
        assert text.startswith('"""held-out evaluation case (synthetic). do not run')
        for label in ("true_positive", "false_positive", "expect_detected", "cwe-"):
            assert label not in text, f"{f.name} mentions {label!r}"


def test_adversarial_suites_fall_back_to_the_bundled_corpus(tmp_path: Path) -> None:
    assert adversarial_suite_dir(CORPUS) == CORPUS
    assert adversarial_suite_dir(HELDOUT) == default_corpus()
    (tmp_path / "adversarial").mkdir()
    assert adversarial_suite_dir(tmp_path) == tmp_path
