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
    assert (m.tp, m.fp, m.fn, m.tn) == (43, 8, 4, 23)
    assert m.precision == 0.8431 and m.recall == 0.9149
    assert "tantular.callgraph.sqli-interprocedural" in result.scanner.by_rule
    ctx = {s.strategy: s for s in result.context.strategies}
    assert ctx["ast"].construct_covered == ctx["ast"].n
    assert ctx["window"].construct_covered < ctx["ast"].construct_covered
    assert not result.context.lost_coverage
    for cwe in ("CWE-89", "CWE-78", "CWE-22", "CWE-79", "CWE-798"):
        assert cwe in result.scanner.by_cwe
    labels = {r.label for r in result.scanner.records if r.outcome in ("TP", "TN", "FP", "FN")}
    assert {
        "true_positive",
        "true_negative",
        "known_false_positive",
        "known_false_negative",
    } <= labels
    assert result.llm is None and result.performance is None
    data = json.loads((out / "benchmark.json").read_text())
    assert data["benchmark_schema_version"] == "1.2.0"
    assert data["result_type"] == "tantular-cyber.benchmark"
    assert data["provenance"]["corpus_sha256"] == corpus_digest(CORPUS)
    assert json.loads((out / "scan-report.json").read_text())["schema_version"] == "1.3.0"


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
    assert llm is not None and llm.mode == "replay" and llm.summary.n_explanations == 13
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
    assert by_case["cmdi-launcher"]["remediation_sound"] is True
    assert by_case["cmdi-launcher"]["cwe_consistent"] is True
    assert by_case["path-archives"]["cwe_consistent"] is True
    assert by_case["creds-flask"]["remediation_sound"] is True
    django = [
        {c.id: c.passed for c in e.checks} for e in llm.evaluations if e.case_id == "xss-django"
    ]
    assert sorted(c["remediation_sound"] for c in django) == [False, True]
    queue = json.loads((out / "human_review.json").read_text())
    assert len(queue) == 13 and all(set(q["reviewer"]) >= {"clarity_id", "notes"} for q in queue)
    recorded = json.loads((out / "recorded.json").read_text())
    assert set(recorded["responses"]) == set(json.loads(REPLAY.read_text())["responses"])


def test_regressions_fail_the_gates(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus)
    manifest = json.loads((corpus / "manifest.json").read_text())
    for case in manifest["cases"]:
        for a in case["annotations"]:
            if a["id"] == "interproc-known-fn-depth":
                a["expect_detected"] = True  # pretend the scanner used to catch it
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
    assert any("covers 6 of 12" in n for n in llm.notes)
    assert llm.summary.checks["schema_valid"].rate == 1.0
    assert llm.summary.checks["injection_resisted"].rate == 1.0
    assert llm.summary.checks["impact_separated"].rate == 1.0


def test_recorded_poc004_outputs_cover_new_cwes(tmp_path: Path) -> None:
    """Raw outputs from the 2026-10-29 live run (AST context), incl. CWE-78/22/79/798."""
    replay_file = ROOT / "benchmarks" / "replays" / "tantular-cyber-latest-2026-10-29-ast.json"
    result = run_benchmark(
        BenchmarkOptions(corpus=CORPUS, output=tmp_path, llm="replay", replay_file=replay_file)
    )
    llm = result.llm
    assert llm is not None and llm.summary.n_explanations == 13
    assert any("covers 10 of 12" in n for n in llm.notes)  # recorded before POC-005 cases
    checks = llm.summary.checks
    for cid in (
        "schema_valid",
        "cwe_consistent",
        "evidence_grounded",
        "remediation_sound",
        "injection_resisted",
        "no_unsupported_assertions",
    ):
        assert checks[cid].rate == 1.0, cid
    assert checks["impact_separated"].passed == 11  # two genuinely unhedged statements


def test_comparison_across_different_corpora_does_not_judge_counts(offline) -> None:
    from tantular_cyber.benchmark.results import compare

    result, _ = offline
    baseline = json.loads(result.model_dump_json())
    baseline["provenance"]["corpus_sha256"] = "different"
    baseline["scanner"]["metrics"].update(fp=0, fn=0, precision=0.99)
    cmp = compare(result, baseline)
    assert not cmp.same_corpus
    deltas = {d.metric: d for d in cmp.deltas}
    assert not deltas["scanner.fp"].worse and not deltas["scanner.fn"].worse
    assert deltas["scanner.precision"].worse  # rates are still judged
    assert "scanner.cwe.CWE-89.precision" in deltas


def test_ablation_attributes_detections_and_pins_known_limits(offline) -> None:
    result, _ = offline
    ab = result.ablation
    assert ab is not None and [a.scanners for a in ab.arms] == [
        "semgrep",
        "callgraph",
        "semgrep+callgraph",
    ]
    former_gaps = {
        "known-fn-cross-function",
        "known-fn-sqlalchemy-text",
        "known-fn-join",
        "cmdi-known-fn-sh-c",
        "path-known-fn-pathlib",
    }
    assert former_gaps <= set(ab.callgraph_unique)
    assert ab.callgraph_new_fp == []  # no false positive that Semgrep did not already raise
    combined = {r.annotation_id: r for r in result.scanner.records}
    # Documented limitations stay undetected at the default depth budget (regression pin).
    for limit in ("interproc-known-fn-depth", "interproc-known-fn-dynamic-dispatch"):
        assert not combined[limit].detected and combined[limit].expectation_met


def test_cloud_model_blocks_every_benchmark_ollama_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (PR #10 review #1): cold start and concurrency must not bypass the
    cloud-model refusal. Only read-only metadata requests may reach Ollama."""
    import httpx

    from tantular_cyber.benchmark import runner
    from tantular_cyber.config import OllamaSettings
    from tantular_cyber.ollama_client import OllamaClient

    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.2"})
        if request.url.path == "/api/tags":
            cloud = {"name": "x:cloud", "remote_host": "https://ollama.com:443"}
            return httpx.Response(200, json={"models": [cloud]})
        return httpx.Response(200, json={"message": {"content": "{}"}, "done_reason": "stop"})

    class Client(OllamaClient):
        def __init__(self, settings, **kw):
            super().__init__(settings, transport=httpx.MockTransport(handler))

    called: list[str] = []
    monkeypatch.setattr(runner, "OllamaClient", Client)
    monkeypatch.setattr(runner, "evict_model", lambda *a, **k: called.append("evict"))
    monkeypatch.setattr(
        runner, "concurrency_experiment", lambda *a, **k: called.append("concurrency") or []
    )
    monkeypatch.setattr(runner, "model_state", lambda *a, **k: {})
    run_benchmark(
        BenchmarkOptions(
            corpus=CORPUS,
            output=tmp_path,
            llm="ollama",
            cold=True,
            concurrency=(1, 2),
            ollama=OllamaSettings(model="x:cloud", max_retries=0),
        )
    )
    assert called == []
    assert set(paths) <= {"/api/version", "/api/tags"}, paths
    notes = json.loads((tmp_path / "benchmark.json").read_text())["performance"]["notes"]
    assert any("model check failed" in n and "remote host" in n for n in notes)
