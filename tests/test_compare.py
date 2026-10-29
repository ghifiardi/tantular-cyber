"""Model/profile comparison harness, offline (Ollama replaced by a fake client)."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import GOOD_EXPLANATION, requires_semgrep

from tantular_cyber.benchmark import compare
from tantular_cyber.benchmark.compare import (
    PINNED_OPTIONS,
    Arm,
    ArmSummary,
    ModelProvenance,
    PairedResult,
    adoption_decision,
    mcnemar_exact,
    parse_arms,
    render_comparison,
    run_comparison,
)

CORPUS = Path(__file__).parents[1] / "benchmarks" / "corpus"


def test_mcnemar_exact() -> None:
    assert mcnemar_exact(0, 0) is None
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(8, 0) == round(2 / 2**8, 4)


def test_parse_arms() -> None:
    arms = parse_arms("m1,m2", "baseline,concise", "8192,4096")
    assert len(arms) == 8 and arms[0] == Arm("m1", "baseline", 8192)
    assert Arm("m", "concise", 4096).label == "m|concise@ctx4096"
    with pytest.raises(ValueError):
        parse_arms("m", "nonexistent", "8192")


def summary(arm: str, p50: float, errors: int = 0) -> ArmSummary:
    return ArmSummary(
        arm=arm,
        model="m",
        profile="p",
        num_ctx=8192,
        n=10,
        status={},
        checks={},
        latency={"p50_s": p50},
        prompt_tokens_mean=None,
        prompt_cached_tokens_mean=None,
        output_tokens_mean=None,
        output_tokens_per_s_mean=None,
        errors=errors,
    )


def pair(check: str, ref_only: int, arm_only: int) -> PairedResult:
    return PairedResult(
        arm="b",
        reference="a",
        check=check,
        pairs=40,
        reference_only=ref_only,
        arm_only=arm_only,
        mcnemar_p=None,
    )


def test_adoption_rule() -> None:
    ref = summary("a", 20.0)
    assert adoption_decision(summary("b", 15.0), ref, [pair("calibrated", 1, 0)]).adopt
    assert not adoption_decision(summary("b", 15.0), ref, [pair("calibrated", 3, 0)]).adopt
    safety = adoption_decision(summary("b", 10.0), ref, [pair("injection_resisted", 1, 0)])
    assert not safety.adopt and any("safety" in r for r in safety.reasons)
    assert not adoption_decision(summary("b", 19.5), ref, []).adopt  # < 10% faster
    assert not adoption_decision(summary("b", 10.0, errors=2), ref, []).adopt


@requires_semgrep
@pytest.mark.semgrep
def test_run_comparison_offline_with_pinned_options(monkeypatch) -> None:
    sent: list[dict] = []

    class FakeClient:
        provider = "ollama"

        def __init__(self, settings) -> None:
            self.model = settings.model
            self.settings = settings
            self.last_metrics = None

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            pass

        def chat(self, messages, format_schema=None) -> str:
            s = self.settings
            sent.append(
                {
                    "model": s.model,
                    "seed": s.seed,
                    "top_p": s.top_p,
                    "top_k": s.top_k,
                    "repeat_penalty": s.repeat_penalty,
                    "num_predict": s.num_predict,
                    "system": messages[0]["content"],
                }
            )
            return GOOD_EXPLANATION

    def fake_provenance(base_url, names):
        return [
            ModelProvenance(
                name=n, installed=n != "missing:1", weights_blob="w" * 16, quantization="Q8_0"
            )
            for n in names
        ]

    monkeypatch.setattr(compare, "OllamaClient", FakeClient)
    monkeypatch.setattr(compare, "model_provenance", fake_provenance)
    arms = [*parse_arms("m:a,m:b,missing:1", "baseline", "8192"), Arm("m:a", "concise")]
    result = run_comparison(CORPUS, arms, runs=2, max_findings=5)
    assert result.findings == 5 and [a.arm for a in result.arms] == [
        "m:a|baseline",
        "m:b|baseline",
        "m:a|concise",
    ]
    assert any("missing:1 not installed" in n for n in result.notes)
    assert {s["seed"] for s in sent} == {1, 2}
    assert all(
        s["top_p"] == PINNED_OPTIONS["top_p"]
        and s["top_k"] == PINNED_OPTIONS["top_k"]
        and s["num_predict"] == PINNED_OPTIONS["num_predict"]
        for s in sent
    )
    assert any("Batasan panjang" in s["system"] for s in sent)  # concise profile applied
    assert all(a.n == 10 for a in result.arms)
    assert [a.arm for a in result.adoption] == ["m:a|concise"]  # same model only
    assert "Paired comparison" in render_comparison(result)


def test_modelfile_profile_mode_sends_no_system_role_or_pinned_options() -> None:
    import httpx
    from test_analysis import make_finding

    from tantular_cyber.analysis import build_messages
    from tantular_cyber.benchmark.compare import _settings
    from tantular_cyber.config import OllamaSettings
    from tantular_cyber.ollama_client import OllamaClient

    arms = parse_arms("a,b", "baseline", "8192,4096", "pinned,profile")
    assert [x.label for x in arms if x.mode == "profile"] == [
        "a|baseline~modelfile-profile",
        "b|baseline~modelfile-profile",
    ]
    msgs = build_messages(make_finding(), system_role=False)
    assert [m["role"] for m in msgs] == ["user"] and "untrusted_data" in msgs[0]["content"]
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(__import__("json").loads(request.content))
        return httpx.Response(200, json={"message": {"content": "{}"}, "done_reason": "stop"})

    settings = _settings(arms[-1], OllamaSettings(), run=2)
    OllamaClient(settings, transport=httpx.MockTransport(handler)).chat(msgs)
    assert sent[0]["options"] == {"seed": 2}  # Modelfile temperature/num_ctx/etc. apply
    assert sent[0]["think"] is False and sent[0]["stream"] is False


@requires_semgrep
@pytest.mark.semgrep
def test_checkpoint_resumes_without_repeating_calls(monkeypatch, tmp_path: Path) -> None:
    calls = {"n": 0}

    class Client:
        provider = "ollama"

        def __init__(self, settings) -> None:
            self.model = settings.model
            self.last_metrics = None

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            pass

        def chat(self, messages, format_schema=None) -> str:
            calls["n"] += 1
            if calls["n"] > limit["n"]:
                raise compare.OllamaError("interrupted")
            return GOOD_EXPLANATION

    limit = {"n": 10**6}
    monkeypatch.setattr(compare, "OllamaClient", Client)
    monkeypatch.setattr(
        compare,
        "model_provenance",
        lambda url, names: [ModelProvenance(name=n, installed=True) for n in names],
    )
    ckpt = tmp_path / "samples.jsonl"
    arms = parse_arms("m:a", "baseline", "8192")
    first = run_comparison(CORPUS, arms, runs=2, max_findings=3, checkpoint_path=ckpt)
    assert calls["n"] == 6 and first.arms[0].n == 6
    calls["n"], limit["n"] = 0, 0  # any new call would now fail
    again = run_comparison(CORPUS, arms, runs=2, max_findings=3, checkpoint_path=ckpt)
    assert calls["n"] == 0 and again.arms[0].n == 6
    assert any("resumed from checkpoint: 6" in n for n in again.notes)
    with pytest.raises(ValueError, match="different settings"):
        run_comparison(CORPUS, arms, runs=3, max_findings=3, checkpoint_path=ckpt)
    limit["n"] = 10**6
    fresh = run_comparison(CORPUS, arms, runs=1, max_findings=3, checkpoint_path=ckpt, fresh=True)
    assert fresh.arms[0].n == 3
