"""Runtime evaluation, review ingestion, lab adapter (fakes only), sandbox policy drafts, CLI."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from tantular_cyber.cli import main
from tantular_cyber.runtime import lab
from tantular_cyber.runtime.evaluation import (
    ReplayClient,
    RuntimeReviewFile,
    load_suite,
    review_queue,
    review_template,
    run_runtime_eval,
    summarize_runtime_reviews,
)
from tantular_cyber.runtime.importer import import_export
from tantular_cyber.runtime.retrieval import load_corpus
from tantular_cyber.runtime.sandbox_policy import (
    explain_policy,
    propose_from_evidence,
    validate_policy,
)

ROOT = Path(__file__).parents[1]
REPLAY = ROOT / "benchmarks" / "runtime" / "replays" / "handwritten-replies.json"
CASES = ROOT / "benchmarks" / "runtime" / "cases"


def test_eval_offline_import_only() -> None:
    result = run_runtime_eval(llm_mode="off", corpus=load_corpus())
    assert result.import_passed and result.synthetic is True
    assert result.quality_items == [] and result.protocol["frozen"] is True
    assert {r.split for r in result.import_results} == {"dev", "heldout", "transfer"}


def test_eval_replay_both_arms_detects_defects() -> None:
    replies = json.loads(REPLAY.read_text())
    corpus = load_corpus()
    result = run_runtime_eval(llm_mode="replay", client=ReplayClient(replies), corpus=corpus)
    arms = {a.arm: a for a in result.arms}
    assert set(arms) == {"retrieval_off", "retrieval_on"}
    assert arms["retrieval_off"].n == arms["retrieval_on"].n == len(result.import_results)
    on = [i for i in result.quality_items if i.arm == "retrieval_on"]
    assert all(i.retrieval.corpus_sha256 == corpus["corpus_sha256"] for i in on)
    assert all(i.retrieval.chunk_ids for i in on)
    hostile = next(i for i in on if i.case_id == "xfer-reference-injection")
    assert "synthetic-hostile:reference#0" in hostile.retrieval.chunk_ids
    by = {(i.case_id, i.arm): i.checks for i in result.quality_items}
    assert by[("dev-metadata-connect", "retrieval_off")]["no_unsupported_claims"] is False
    assert by[("dev-metadata-connect", "retrieval_off")]["attempt_outcome_distinct"] is False
    assert by[("dev-injection-fields", "retrieval_off")]["injection_resisted"] is False
    assert by[("held-unsupported-version", "retrieval_on")]["citations_valid"] is False
    assert by[("held-unsupported-version", "retrieval_on")]["evidence_refs_valid"] is False
    assert by[("dev-benign-exec", "retrieval_on")]["citations_valid"] is True
    assert by[("xfer-empty-interval", "retrieval_off")]["abstention_consistent"] is True
    assert all(by[(c, a)]["privacy"] for c, a in by)
    # abstention is tracked so silence cannot masquerade as quality
    assert arms["retrieval_off"].abstained >= 1
    assert arms["retrieval_off"].assessable < arms["retrieval_off"].n


def test_invalid_refs_do_not_count_as_unsupported_claims() -> None:
    """Regression: a bad evidence ref must fail evidence_refs_valid only, not the claim check."""
    replies = json.loads(REPLAY.read_text())
    key = "dev-benign-exec/retrieval_off"
    reply = json.loads(replies[key])
    for obs in reply["observations"]:
        obs["evidence_refs"] = ["exec#999"]
    replies[key] = json.dumps(reply)
    result = run_runtime_eval(
        llm_mode="replay",
        client=ReplayClient(replies),
        corpus=load_corpus(),
        splits=("dev",),
        arms=("retrieval_off",),
    )
    checks = next(i.checks for i in result.quality_items if i.case_id == "dev-benign-exec")
    assert checks["evidence_refs_valid"] is False
    assert checks["no_unsupported_claims"] is True


def test_record_replies_roundtrip_and_guards(tmp_path: Path) -> None:
    replies = json.loads(REPLAY.read_text())
    captured: dict[str, str] = {}
    run_runtime_eval(
        llm_mode="replay",
        client=ReplayClient(replies),
        corpus=load_corpus(),
        splits=("dev",),
        replies_out=captured,
    )
    assert captured and all(captured[k] == replies[k] for k in captured)
    out = tmp_path / "r.json"
    out.write_text("{}")
    base = ["runtime", "eval", "--output", str(tmp_path / "o"), "--record-replies", str(out)]
    assert main([*base, "--llm", "ollama", "--ollama-url", "http://127.0.0.1:9"]) == 2  # exists
    assert main([*base, "--llm", "off"]) == 2  # requires ollama


def test_review_queue_blank_and_summary_never_imputes(tmp_path: Path) -> None:
    replies = json.loads(REPLAY.read_text())
    result = run_runtime_eval(
        llm_mode="replay", client=ReplayClient(replies), corpus=load_corpus(), splits=("dev",)
    )
    queue = review_queue(result)
    assert all(v is None for q in queue for v in q["dimensions"].values())
    form = review_template(queue, "rev_a", "q" * 64)
    assert all(v is None for i in form["items"] for k, v in i.items() if k != "review_id")
    a = RuntimeReviewFile.model_validate(
        {
            **form,
            "items": [
                {"review_id": queue[0]["review_id"], "indonesian_clarity": 4, "privacy_ok": "yes"}
            ],
        }
    )
    b = RuntimeReviewFile.model_validate(
        {
            **review_template(queue, "rev_b", "q" * 64),
            "items": [
                {"review_id": queue[0]["review_id"], "indonesian_clarity": 5},
                {"review_id": queue[1]["review_id"], "privacy_ok": "no"},
            ],
        }
    )
    s = summarize_runtime_reviews("q" * 64, [a, b])
    assert s["dimensions"]["indonesian_clarity"]["counts"] == {"4": 1, "5": 1}
    assert s["dimensions"]["privacy_ok"]["pairwise"] == []  # no overlap, nothing imputed
    with pytest.raises(ValueError):
        review_template(queue, "real name@example.com", "q" * 64)


# --- lab adapter ------------------------------------------------------------------------


class FakeRunner:
    def __init__(self, export: bytes) -> None:
        self.calls: list[list[str]] = []
        self.export = export

    def __call__(self, argv: Sequence[str], timeout: float) -> tuple[int, bytes, bytes]:
        self.calls.append(list(argv))
        assert timeout > 0
        if "map" in argv:
            return 0, self.export, b""
        return 0, b"Linux lab 7.4.0-rc1-bpf-next+\n", b""


def test_lab_refuses_unconfirmed_or_production_like_targets(tmp_path: Path) -> None:
    for target in (
        lab.LabTarget(name="lab-vm1", ssh_host="10.0.2.15", disposable_confirmed=False),
        lab.LabTarget(name="prod-web-1", ssh_host="10.0.2.15", disposable_confirmed=True),
        lab.LabTarget(name="lab-vm1", ssh_host="host;rm -rf /", disposable_confirmed=True),
    ):
        with pytest.raises(lab.LabRefused):
            lab.collect(target, tmp_path, runner=FakeRunner(b""))


@pytest.mark.parametrize(
    "remote",
    [
        ["shirudo-runtime", "upatch", "load", "x.upatch"],
        ["shirudo-runtime", "upatch", "unload", "x"],
        ["shirudo", "genkey"],
        ["sh", "-c", "id"],
        ["shirudo-runtime", "upatch", "map", "kernel-obs", "exec", "--otel"],
        ["shirudo-runtime", "upatch", "map", "kernel-obs", "exec;id"],
    ],
)
def test_lab_allowlist_blocks_enforcement_and_shell(remote: list[str]) -> None:
    with pytest.raises(lab.LabRefused):
        lab.check_remote(remote)


def test_lab_collect_with_fake_runner_then_import(tmp_path: Path) -> None:
    export = (CASES / "synthetic-dev-benign-exec.ndjson").read_bytes().strip()
    runner = FakeRunner(export)
    target = lab.LabTarget(name="lab-shirudo-1", ssh_host="10.0.2.15", disposable_confirmed=True)
    run = lab.collect(target, tmp_path, runner=runner, maps=("exec",))
    assert all(c[0] == "ssh" and "--" in c for c in runner.calls)
    map_calls = [c for c in runner.calls if "map" in c]
    assert map_calls[0][-1] == "--peek"  # do not steal the exporter's interval by default
    assert run.export_files == ["kernel-obs-export.ndjson"]
    report = import_export((tmp_path / "kernel-obs-export.ndjson").read_bytes())
    assert report.status == "complete" and report.coverage.maps["exec"].records == 2
    meta = json.loads((tmp_path / "lab-run.json").read_text())
    assert meta["live_validation"] == "unreviewed"


def test_lab_manifest_records_requirements() -> None:
    m = lab.compatibility_manifest()
    assert m["upstream"]["revision"].startswith("44926eb7")
    assert "tpm" in m["target"] and "container" in m["target"]["not_sufficient"]
    assert m["live_validation"].startswith("pending")


# --- sandbox policy drafts ------------------------------------------------------------------

GOOD_POLICY = """
[networks]
localnet = ["127.0.0.0/8", "::1/128"]

[sandbox.apache]
exec = []
network = ["@localnet", "192.0.2.10/32"]

[sandbox.apache.deny]
create = ["*.php"]
write = ["*.php", "config.inc"]
"""


def test_policy_valid_and_explained_as_draft_with_gaps() -> None:
    check = validate_policy(GOOD_POLICY, {"apache"})
    assert check.valid, check.errors
    text = explain_policy(check)
    assert "DRAF" in text and "belum dimuat" in text
    assert "descriptor" in text and "patch" in text and "bukan sandbox agen" in text
    assert "t9114" in text  # upstream outcomes are not claimed


@pytest.mark.parametrize(
    ("policy", "error"),
    [
        ("[sandbox.nginx]\nexec = []\n", "declares no policy nginx"),
        ('[sandbox.apache]\nexec = ["helpers"]\n', "no policy helpers"),
        ('[sandbox.apache]\nnetwork = ["10.0.0.1/8"]\n', "not a CIDR"),
        ('[sandbox.apache]\nnetwork = ["@missing"]\n', "no [networks] entry"),
        ('[sandbox.apache.deny]\nwrite = ["a*b*c"]\n', "at most one"),
        ('[sandbox.apache.deny]\nwrite = ["dir/x"]\n', "at most one"),
        ('[sandbox.apache.deny]\nwrite = ["' + "a" * 33 + '*"]\n', "32 bytes"),
        ('[sandbox.apache.deny]\nexecute = ["x"]\n', "unknown key"),
        ("[other]\n", "unknown table"),
        ("not toml [", "invalid TOML"),
    ],
)
def test_policy_rejections(policy: str, error: str) -> None:
    check = validate_policy(policy, {"apache"})
    assert not check.valid and any(error in e for e in check.errors), check.errors


def test_policy_capacity_limits() -> None:
    nets = ", ".join(f'"10.0.{i}.0/24"' for i in range(129))
    check = validate_policy(f"[sandbox.apache]\nnetwork = [{nets}]\n", {"apache"})
    assert any("networks; the object holds 128" in e for e in check.errors)


def test_propose_uses_identity_not_name() -> None:
    report = import_export(
        (CASES / "synthetic-dev-metadata-connect.ndjson").read_bytes(), path_mode="hash_only"
    )
    draft = propose_from_evidence(report, policy_id="curl", executable_path="/usr/bin/curl")
    assert '"169.254.169.254/32"' in draft and draft.startswith("# DRAFT ONLY")
    assert validate_policy(draft.split("\n", 3)[3], {"curl"}).valid
    none = propose_from_evidence(report, policy_id="curl", executable_path="/bin/curl")
    assert "network = []" in none


# --- CLI ---------------------------------------------------------------------------------------


def test_cli_import_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    assert (
        main(
            [
                "runtime",
                "import",
                "--input",
                str(CASES / "synthetic-dev-benign-exec.ndjson"),
                "--output",
                str(out),
            ]
        )
        == 0
    )
    assert (out / "runtime-evidence.json").is_file()
    assert (
        main(
            [
                "runtime",
                "import",
                "--input",
                str(CASES / "synthetic-dev-loss-overflow.ndjson"),
                "--output",
                str(out),
            ]
        )
        == 4
    )
    empty = tmp_path / "empty.ndjson"
    empty.write_text("")
    assert main(["runtime", "import", "--input", str(empty), "--output", str(out)]) == 3
    assert main(["runtime", "import", "--input", str(tmp_path), "--output", str(out)]) == 2
    f = tmp_path / "file"
    f.write_text("x")
    assert main(["runtime", "import", "--input", str(empty), "--output", str(f)]) == 2


def test_cli_analyze_replay_and_unavailable(tmp_path: Path) -> None:
    out = tmp_path / "ev"
    main(
        [
            "runtime",
            "import",
            "--input",
            str(CASES / "synthetic-dev-benign-exec.ndjson"),
            "--output",
            str(out),
        ]
    )
    replies = tmp_path / "r.json"
    good = json.loads(REPLAY.read_text())["dev-benign-exec/retrieval_off"]
    replies.write_text(json.dumps({"analyze": good}))
    code = main(
        [
            "runtime",
            "analyze",
            "--evidence",
            str(out / "runtime-evidence.json"),
            "--output",
            str(tmp_path / "an"),
            "--replay-file",
            str(replies),
        ]
    )
    data = json.loads((tmp_path / "an" / "runtime-analysis.json").read_text())
    assert code in (0, 4) and data["explanation"] is not None
    assert data["evidence_verified"] is False
    code = main(
        [
            "runtime",
            "analyze",
            "--evidence",
            str(out / "runtime-evidence.json"),
            "--output",
            str(tmp_path / "an2"),
            "--ollama-url",
            "http://203.0.113.1:1",
        ]
    )
    assert code == 4  # remote endpoint refused -> degraded, never a silent complete
    data = json.loads((tmp_path / "an2" / "runtime-analysis.json").read_text())
    assert data["status"] == "degraded" and "RemoteEndpointNotAllowed" in data["error"]


def test_cli_eval_lab_policy_schema(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["runtime", "eval", "--output", str(tmp_path / "e")]) == 0
    assert (
        main(
            [
                "runtime",
                "eval",
                "--output",
                str(tmp_path / "r"),
                "--llm",
                "replay",
                "--replay-file",
                str(REPLAY),
            ]
        )
        == 0
    )
    assert (tmp_path / "r" / "runtime-review-queue.json").is_file()
    assert main(["runtime", "lab", "manifest"]) == 0
    assert main(["runtime", "lab", "collect", "--name", "lab-a", "--host", "10.0.2.15"]) == 2
    capsys.readouterr()
    assert (
        main(
            [
                "runtime",
                "lab",
                "collect",
                "--name",
                "lab-a",
                "--host",
                "10.0.2.15",
                "--i-confirm-disposable-lab-vm",
            ]
        )
        == 0
    )  # dry run without --execute
    assert "dry run" in capsys.readouterr().out
    pol = tmp_path / "p.toml"
    pol.write_text(GOOD_POLICY)
    assert (
        main(["runtime", "policy", "validate", "--policy", str(pol), "--declared", "apache"]) == 0
    )
    assert main(["runtime", "policy", "validate", "--policy", str(pol)]) == 2
    assert main(["runtime", "schema", "evidence"]) == 0
    assert main(["runtime", "corpus", "info"]) == 0


def test_scan_contract_unchanged() -> None:
    from tantular_cyber.models import SCHEMA_VERSION

    assert SCHEMA_VERSION == "1.3.0"


def test_suite_split_families_disjoint() -> None:
    _, manifest = load_suite()
    seen: dict[str, str] = {}
    for case in manifest["cases"]:
        assert seen.setdefault(case["family"], case["split"]) == case["split"]
