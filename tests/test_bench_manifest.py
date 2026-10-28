from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tantular_cyber.benchmark.manifest import (
    ManifestError,
    corpus_digest,
    load_manifest,
    validate_manifest,
)

CORPUS = Path(__file__).parents[1] / "benchmarks" / "corpus"


def copy_corpus(tmp_path: Path) -> Path:
    dest = tmp_path / "corpus"
    shutil.copytree(CORPUS, dest)
    return dest


def edit_manifest(corpus: Path, fn) -> None:
    path = corpus / "manifest.json"
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def test_committed_corpus_manifest_is_consistent() -> None:
    manifest, root = load_manifest(CORPUS)
    assert validate_manifest(manifest, root) == []


def test_corpus_covers_required_categories() -> None:
    manifest, _ = load_manifest(CORPUS)
    labels = {a.label for _, a in manifest.annotations()}
    assert {
        "true_positive",
        "true_negative",
        "known_false_positive",
        "known_false_negative",
    } <= labels
    assert any(c.injection_canary for c in manifest.cases)
    assert any("near_miss" in c.tags for c in manifest.cases)


def test_case_files_carry_no_ground_truth_labels() -> None:
    """The LLM reads case files; labels must live only in the manifest."""
    manifest, root = load_manifest(CORPUS)
    for case in manifest.cases:
        text = (root / case.path).read_text(encoding="utf-8").lower()
        assert "DO NOT RUN".lower() in text
        for leak in ("vulnerable", "true positive", "false positive", "cwe-"):
            assert leak not in text, f"{case.path} leaks label '{leak}'"


@pytest.mark.parametrize(
    ("mutate", "problem"),
    [
        (lambda d: d["cases"][0]["annotations"][0].update(line=3, end_line=3), "anchor not found"),
        (lambda d: d["cases"][0].update(path="missing.py"), "file not found"),
        (lambda d: d["cases"][0].update(path="../../etc/passwd"), "escapes the cases root"),
        (lambda d: d["cases"][1].update(id=d["cases"][0]["id"]), "duplicate case ids"),
        (lambda d: d["cases"][0]["annotations"][0].update(preconditions=[]), "preconditions"),
        (lambda d: d["cases"].pop(), "unlisted corpus file"),
    ],
)
def test_validation_catches_drift(tmp_path: Path, mutate, problem: str) -> None:
    corpus = copy_corpus(tmp_path)
    edit_manifest(corpus, mutate)
    manifest, root = load_manifest(corpus)
    assert any(problem in p for p in validate_manifest(manifest, root))


def test_unsupported_manifest_version_is_rejected(tmp_path: Path) -> None:
    corpus = copy_corpus(tmp_path)
    edit_manifest(corpus, lambda d: d.update(manifest_version="2.0.0"))
    with pytest.raises(ManifestError, match="unsupported"):
        load_manifest(corpus)
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "nowhere")


def test_corpus_digest_changes_with_content(tmp_path: Path) -> None:
    corpus = copy_corpus(tmp_path)
    before = corpus_digest(corpus)
    assert before == corpus_digest(CORPUS)
    (corpus / "cases" / "accounts" / "lookup.py").write_text("x = 1\n")
    assert corpus_digest(corpus) != before
