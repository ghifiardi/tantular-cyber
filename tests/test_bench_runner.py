"""End-to-end benchmark runs (real Semgrep, no Ollama)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from conftest import requires_semgrep

from tantular_cyber.benchmark.manifest import corpus_digest
from tantular_cyber.benchmark.runner import BenchmarkOptions, run_benchmark
from tantular_cyber.cli import EXIT_GATES, EXIT_INPUT, EXIT_OK, main

ROOT = Path(__file__).parents[1]
CORPUS = ROOT / "benchmarks" / "corpus"
REPLAY = ROOT / "tests" / "fixtures" / "bench_replay.json"

pytestmark = [requires_semgrep, pytest.mark.semgrep]


@pytest.fixture(scope="module")
def offline(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("bench")
    before = corpus_digest(CORPUS)
    result = run_benchmark(BenchmarkOptions(corpus=CORPUS, output=out))
    assert corpus_digest(CORPUS) == before, "benchmark must not modify the corpus"
    return result, out


@pytest.fixture(scope="module")
def replay(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("bench-replay")
    result = run_benchmark(
        BenchmarkOptions(
            corpus=CORPUS,
            output=out,
            llm="replay",
            replay_file=REPLAY,
            record_replay=out / "recorded.json",
        )
    )
    return result, out


def test_offline_benchmark_passes_with_expected_metrics(offline) -> None:
    result, out = offline
    assert result.status == "pass", [g for g in result.gates if not g.passed]
    m = result.scanner.metrics
    assert (m.tp, m.fp, m.fn, m.tn) == (12, 3, 3, 5)
    assert m.precision == 0.8 and m.recall == 0.8
    labels = {r.label for r in result.scanner.records if r.outcome in ("TP", "TN", "FP", "FN")}
    assert {
        "true_positive",
        "true_negative",
        "known_false_positive",
        "known_false_negative",
    } <= labels
    assert result.llm is None and result.performance is None
    data = json.loads((out / "benchmark.json").read_text())
    assert data["benchmark_schema_version"] == "1.0.0"
    assert data["result_type"] == "tantular-cyber.benchmark"
    assert data["provenance"]["corpus_sha256"] == corpus_digest(CORPUS)
    assert json.loads((out / "scan-report.json").read_text())["schema_version"] == "1.1.0"


def test_outputs_contain_no_local_paths_or_secrets(offline, replay) -> None:
    for _, out in (offline, replay):
        for path in out.glob("*"):
            text = path.read_text(encoding="utf-8")
            assert str(CORPUS.resolve()) not in text and str(Path.home()) not in text, path
            for canary in ("AKIAIOSFODNN7EXAMPLE", "corpus-example-password-1"):
                assert canary not in text, path


def test_replay_mode_exercises_every_check(replay) -> None:
    result, out = replay
    assert result.status == "pass"  # LLM quality never gates
    llm = result.llm
    assert llm is not None and llm.mode == "replay" and llm.summary.n_explanations == 8
    by_case: dict[str, dict[str, bool | None]] = {}
    for ev in llm.evaluations:
        by_case.setdefault(ev.case_id, {}).update({c.id: c.passed for c in ev.checks})
    assert by_case["injection-en"]["injection_resisted"] is False
    assert by_case["injection-en"]["bahasa_indonesia"] is False
    assert by_case["injection-id"]["injection_resisted"] is True
    assert by_case["auth-login"]["no_unsupported_assertions"] is False
    assert by_case["auth-login"]["cwe_consistent"] is False
    evals = {(e.case_id, e.finding_id): e for e in llm.evaluations}
    checks = [
        {c.id: c.passed for c in e.checks}
        for e in evals.values()
        if e.case_id == "known-fp-constants"
    ]
    assert any(c["calibrated"] is False for c in checks)
    assert any(c["remediation_sound"] is False for c in checks)
    assert any(
        c.get("evidence_grounded") is False
        for e in llm.evaluations
        for c in [{x.id: x.passed for x in e.checks}]
        if e.case_id == "admin-maintenance"
    )
    assert all(llm.injection_cases_detected.values())
    queue = json.loads((out / "human_review.json").read_text())
    assert len(queue) == 8 and all(set(q["reviewer"]) >= {"clarity_id", "notes"} for q in queue)
    recorded = json.loads((out / "recorded.json").read_text())
    assert set(recorded["responses"]) == set(json.loads(REPLAY.read_text())["responses"])


def test_regressions_fail_the_gates(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus)
    manifest = json.loads((corpus / "manifest.json").read_text())
    for case in manifest["cases"]:
        for a in case["annotations"]:
            if a["id"] == "known-fn-join":
                a["expect_detected"] = True  # pretend the rule used to catch it
    (corpus / "manifest.json").write_text(json.dumps(manifest))
    suite = json.loads((corpus / "adversarial" / "secrets.json").read_text())
    suite["items"][0]["expect"] = "kept"  # behavior no longer matches expectation
    (corpus / "adversarial" / "secrets.json").write_text(json.dumps(suite))

    code = main(["benchmark", "--corpus", str(corpus), "--output", str(tmp_path / "out")])
    assert code == EXIT_GATES
    data = json.loads((tmp_path / "out" / "benchmark.json").read_text())
    failed = {g["id"] for g in data["gates"] if not g["passed"]}
    assert failed == {"expected_detections", "adversarial_secrets"}


def test_baseline_comparison(offline, tmp_path: Path) -> None:
    _, out = offline
    result = run_benchmark(
        BenchmarkOptions(corpus=CORPUS, output=tmp_path, baseline=out / "benchmark.json")
    )
    cmp = result.comparison
    assert cmp is not None and cmp.same_corpus
    assert not any(d.worse for d in cmp.deltas)
    assert "Comparison with baseline" in (tmp_path / "benchmark.md").read_text()


def test_cli_input_errors(tmp_path: Path) -> None:
    out = str(tmp_path / "o")
    assert main(["benchmark", "--corpus", str(tmp_path), "--output", out]) == EXIT_INPUT
    assert main(["benchmark", "--output", out, "--concurrency", "9"]) == EXIT_INPUT
    assert main(["benchmark", "--output", out, "--llm", "replay"]) == EXIT_INPUT
    assert main(["benchmark", "--output", out, "--baseline", str(tmp_path / "x")]) == EXIT_INPUT


def test_cli_offline_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["benchmark", "--corpus", str(CORPUS), "--output", str(tmp_path)]) == EXIT_OK
    assert "benchmark: PASS" in capsys.readouterr().out


def test_recorded_real_model_outputs_rescore_deterministically(tmp_path: Path) -> None:
    """Raw tantular-cyber:latest outputs from the 2026-10-28 live run, replayed offline.

    Guards the evaluator against false alarms on real model wording (two were found and
    fixed in that run: reported injection claims and 'bergantung pada' hedges).
    """
    replay_file = ROOT / "benchmarks" / "replays" / "tantular-cyber-latest-2026-10-28.json"
    result = run_benchmark(
        BenchmarkOptions(corpus=CORPUS, output=tmp_path, llm="replay", replay_file=replay_file)
    )
    llm = result.llm
    assert llm is not None and llm.summary.n_explanations == 8
    assert llm.summary.checks["schema_valid"].rate == 1.0
    assert llm.summary.checks["injection_resisted"].rate == 1.0
    assert llm.summary.checks["impact_separated"].rate == 1.0
