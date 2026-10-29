"""Pointer-only real-world evaluation: manifest hygiene, verified fetch, offline evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest
from conftest import requires_semgrep

from tantular_cyber.benchmark.realworld import (
    RealWorldManifest,
    cache_path,
    clean_cache,
    evaluate_pointers,
    fetch_pointers,
    git_blob_sha1,
    load_realworld,
    validate_realworld,
)

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "benchmarks" / "realworld" / "manifest.json"
# Synthetic stand-in for an upstream file (tests never download anything).
UPSTREAM = b"import sqlite3\n\ndef q(cur, t):\n    cur.execute('SELECT * FROM x.' + t + ' y')\n"
LINE = "cur.execute('SELECT * FROM x.' + t + ' y')"


def synthetic_manifest(**overrides) -> RealWorldManifest:
    entry = {
        "id": "rw-test",
        "cwe": "CWE-89",
        "advisory": {"id": "TEST-1", "url": "https://example.invalid/advisory"},
        "repository": "https://github.com/example/project",
        "commit": "a" * 40,
        "path": "pkg/db.py",
        "line": 4,
        "end_line": 4,
        "line_sha256": hashlib.sha256(LINE.encode()).hexdigest(),
        "git_blob_sha1": git_blob_sha1(UPSTREAM),
        "upstream_license": "MIT",
        "license_source": "test",
        "description": "test",
        "verification": {"status": "upstream_label"},
    }
    entry.update(overrides)
    return RealWorldManifest(
        manifest_version="1.0.0", kind="pointer_only", policy="test", entries=[entry]
    )


def test_committed_manifest_is_valid_and_pointer_only() -> None:
    manifest = load_realworld(MANIFEST)
    assert validate_realworld(manifest) == []
    raw = MANIFEST.read_text()
    for forbidden in ("execute(", "def ", "import ", "subprocess", "os.path.join("):
        assert forbidden not in raw, f"manifest must not contain upstream code ({forbidden})"
    assert {e.cwe for e in manifest.entries} >= {"CWE-89", "CWE-22", "CWE-78"}
    assert all(not e.verification.independently_verified for e in manifest.entries)
    tuned = [e.id for e in manifest.entries if e.development_use]
    assert tuned == ["rw-wsgidav-mysql-select"]  # disclosed: not held out


@pytest.mark.parametrize(
    ("override", "problem"),
    [
        ({"commit": "main"}, "40-hex"),
        ({"path": "../etc/passwd"}, "inside the repository"),
        ({"repository": "https://gitlab.com/a/b"}, "github.com"),
        (
            {"verification": {"status": "independently_verified", "independently_verified": True}},
            "pseudonym",
        ),
        ({"verification": {"status": "upstream_label", "verified_by": "a@b.c"}}, "e-mail"),
    ],
)
def test_manifest_validation(override: dict, problem: str) -> None:
    assert any(problem in p for p in validate_realworld(synthetic_manifest(**override)))


def test_fetch_verifies_blob_hash_and_never_stores_mismatches(tmp_path: Path) -> None:
    seen: list[str] = []

    def ok(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=UPSTREAM)

    manifest = synthetic_manifest()
    assert fetch_pointers(manifest, tmp_path, transport=httpx.MockTransport(ok)) == [
        ("rw-test", "fetched")
    ]
    assert seen == [f"https://raw.githubusercontent.com/example/project/{'a' * 40}/pkg/db.py"]
    assert fetch_pointers(manifest, tmp_path, transport=httpx.MockTransport(ok))[0][1] == "cached"

    other = tmp_path / "other"
    tampered = httpx.MockTransport(lambda r: httpx.Response(200, content=UPSTREAM + b"#"))
    (outcome,) = fetch_pointers(manifest, other, transport=tampered)
    assert (
        outcome[1].startswith("integrity_failed")
        and not cache_path(other, manifest.entries[0]).exists()
    )


def test_evaluation_is_offline_and_reports_states(tmp_path: Path) -> None:
    manifest = synthetic_manifest()
    result = evaluate_pointers(manifest, tmp_path)
    assert result.results[0].state == "not_fetched" and result.evaluated == 0
    target = cache_path(tmp_path, manifest.entries[0])
    target.parent.mkdir(parents=True)
    target.write_bytes(UPSTREAM.replace(b"y'", b"z'"))
    assert evaluate_pointers(manifest, tmp_path).results[0].state == "integrity_failed"


@requires_semgrep
@pytest.mark.semgrep
def test_evaluation_detects_from_cache_and_cleans(tmp_path: Path) -> None:
    manifest = synthetic_manifest()
    target = cache_path(tmp_path, manifest.entries[0])
    target.parent.mkdir(parents=True)
    target.write_bytes(UPSTREAM)
    result = evaluate_pointers(manifest, tmp_path)
    assert result.results[0].state == "detected" and result.detected == 1
    assert result.results[0].held_out
    moved = synthetic_manifest(line=3, end_line=3)
    assert evaluate_pointers(moved, tmp_path).results[0].state == "anchor_moved"
    assert clean_cache(tmp_path) and not tmp_path.exists()
    assert json.loads(result.model_dump_json())["caveat"].startswith("Pointer-only")


def test_missed_sinks_are_recorded_as_coverage_candidates() -> None:
    manifest = load_realworld(MANIFEST)
    candidates = {e.id for e in manifest.entries if e.coverage_candidate}
    assert candidates == {
        "rw-mem0-databricks-delete",
        "rw-djust-fixture-path",
        "rw-setuptools-namespace-dir",
        "rw-glances-action-command",
    }
