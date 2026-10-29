"""Controlled, opt-in comparison of local models and prompt profiles ("arms").

Every arm sees the same findings, the same evidence (same scan, same context), the same
rubric and **pinned** decoding options (temperature, top_p, top_k, repeat/presence
penalty, num_ctx, num_predict, per-run seed), so Modelfile defaults cannot differ between
models. Results are sample-based: per-check rates with Wilson intervals, paired
wins/losses against a reference arm with an exact McNemar test, and latency/token use.

Model provenance (digest, weight blob, quantization, Modelfile SYSTEM/PARAMETER presence,
shared weights) is recorded so a "model" difference can be attributed to weights or to a
prompt/parameter profile. Never used in CI.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from ..analysis import PROMPT_PROFILES, FindingAnalyzer
from ..config import OllamaSettings, ScanSettings
from ..models import Finding, FindingAnalysis
from ..ollama_client import OllamaClient, OllamaError, validate_base_url
from ..pipeline import run_scan
from .llm_eval import RUBRIC, ExplanationEval, evaluate_finding, wilson
from .manifest import Manifest, load_manifest
from .matching import match_findings
from .perf import latency_stats
from .provenance import code_state, machine, now

SAFETY_CHECKS = ("no_unsupported_assertions", "injection_resisted", "schema_valid")
_FROM = re.compile(r"^FROM\s+(\S+)", re.MULTILINE)

PINNED_OPTIONS = {
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 20,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "num_ctx": 8192,
    "num_predict": 2048,
}


@dataclass(frozen=True)
class Arm:
    model: str
    profile: str = "baseline"
    num_ctx: int = 8192
    # "pinned": our system prompt + pinned options. "profile": the model's own Modelfile
    # SYSTEM prompt and PARAMETERs apply (only the seed is sent; our instructions travel in
    # the user message) — isolates a Modelfile profile on identical weights.
    mode: str = "pinned"

    @property
    def label(self) -> str:
        if self.mode == "profile":
            return f"{self.model}|{self.profile}~modelfile-profile"
        extra = "" if self.num_ctx == 8192 else f"@ctx{self.num_ctx}"
        return f"{self.model}|{self.profile}{extra}"


class ModelProvenance(BaseModel):
    name: str
    installed: bool
    digest: str | None = None
    weights_blob: str | None = None
    extra_blobs: list[str] = Field(default_factory=list)
    family: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    modelfile_has_system: bool | None = None
    system_sha256: str | None = Field(
        default=None, description="Hash of the Modelfile SYSTEM prompt (text not stored)."
    )
    modelfile_parameters: dict[str, str] = Field(default_factory=dict)
    shares_weights_with: list[str] = Field(default_factory=list)


class ArmSummary(BaseModel):
    arm: str
    model: str
    profile: str
    num_ctx: int
    n: int
    status: dict[str, int]
    checks: dict[str, dict[str, float | int | None]]
    latency: dict[str, float | int | None]
    prompt_tokens_mean: float | None
    prompt_cached_tokens_mean: float | None
    output_tokens_mean: float | None
    output_tokens_per_s_mean: float | None
    errors: int
    provenance_changed: bool = Field(
        default=False,
        description="The model's provenance differed before vs after the run: samples may "
        "come from more than one model file and must not be read as a single-version result.",
    )


class PairedResult(BaseModel):
    arm: str
    reference: str
    check: str
    pairs: int
    reference_only: int  # reference passed, arm failed
    arm_only: int  # arm passed, reference failed
    mcnemar_p: float | None


class Adoption(BaseModel):
    arm: str
    adopt: bool
    reasons: list[str]


class ComparisonResult(BaseModel):
    result_type: str = "tantular-cyber.model-comparison"
    schema_version: str = "1.1.0"
    started_at: str
    finished_at: str | None = None
    sample_based: bool = True
    machine: dict[str, object]
    code_commit: str | None
    corpus_id: str
    findings: int
    runs: int
    pinned_options: dict[str, float | int]
    reference: str
    models: list[ModelProvenance]
    arms: list[ArmSummary]
    paired: list[PairedResult]
    adoption: list[Adoption]
    profile_identity: dict[str, object] | None = None
    provenance_changes: list[str] = Field(
        default_factory=list,
        description="Differences between the provenance captured before and after the arms "
        "(e.g. Ollama migrated or re-pulled a model mid-run).",
    )
    notes: list[str]
    caveats: list[str] = [
        "Sample-based: rates come from deterministic proxy checks on a small synthetic corpus "
        "and carry wide intervals; they are not a quality guarantee.",
        "Quantization differs between models (e.g. Q8_0 vs Q4_K_M); weight comparisons are "
        "confounded with quantization unless both arms share it.",
        "Our request supplies its own system message; Ollama then does not apply the "
        "Modelfile SYSTEM prompt, and all sampling options are pinned per request.",
    ]


def mcnemar_exact(b: int, c: int) -> float | None:
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return round(min(1.0, 2 * tail), 4)


def model_provenance(base_url: str, names: Sequence[str]) -> list[ModelProvenance]:
    out: list[ModelProvenance] = []
    with httpx.Client(base_url=base_url, timeout=30, trust_env=False) as http:
        tags = {m["name"]: m for m in http.get("/api/tags").json().get("models", [])}
        for name in names:
            tag = tags.get(name) or tags.get(f"{name}:latest")
            if tag is None:
                out.append(ModelProvenance(name=name, installed=False))
                continue
            show = http.post("/api/show", json={"model": name}).json()
            modelfile = show.get("modelfile", "")
            blobs = [
                b.rsplit("sha256-", 1)[-1][:16] for b in _FROM.findall(modelfile) if "sha256-" in b
            ]
            params = {}
            for line in modelfile.splitlines():
                if line.startswith("PARAMETER "):
                    _, key, *value = line.split()
                    params[key] = " ".join(value)
            details = tag.get("details") or {}
            out.append(
                ModelProvenance(
                    name=name,
                    installed=True,
                    digest=str(tag.get("digest", ""))[:16],
                    weights_blob=blobs[0] if blobs else None,
                    extra_blobs=blobs[1:],
                    family=details.get("family"),
                    parameter_size=details.get("parameter_size"),
                    quantization=details.get("quantization_level"),
                    modelfile_has_system="\nSYSTEM " in f"\n{modelfile}",
                    system_sha256=(
                        hashlib.sha256(str(show.get("system", "")).encode()).hexdigest()[:16]
                        if show.get("system")
                        else None
                    ),
                    modelfile_parameters=params,
                )
            )
    for a in out:
        a.shares_weights_with = sorted(
            b.name
            for b in out
            if b is not a and a.weights_blob and a.weights_blob == b.weights_blob
        )
    return out


def provenance_drift(
    before: Sequence[ModelProvenance], after: Sequence[ModelProvenance]
) -> dict[str, list[str]]:
    """Per model, the fields that changed between two provenance snapshots."""
    later = {m.name: m for m in after}
    changes: dict[str, list[str]] = {}
    for m in before:
        b = later.get(m.name)
        if b is None:
            changes[m.name] = [f"{m.name}: missing from the post-run snapshot"]
            continue
        diffs = [
            f"{m.name}: {key} {getattr(m, key)} -> {getattr(b, key)}"
            for key in ("installed", "digest", "weights_blob", "quantization", "system_sha256")
            if getattr(m, key) != getattr(b, key)
        ]
        if diffs:
            changes[m.name] = diffs
    return changes


@dataclass
class _Sample:
    run: int
    finding: Finding
    evaluation: ExplanationEval | None
    error: str | None = None
    metrics: dict[str, float | int | None] = field(default_factory=dict)


def select_findings(
    manifest: Manifest, cases_root: Path, max_findings: int, tmp: Path
) -> tuple[list[Finding], dict[str, object]]:
    report = run_scan(ScanSettings(repo=cases_root, output=tmp / "scan", llm_enabled=False)).report
    records = match_findings(manifest, report.findings).records
    by_finding = {r.finding_id: r for r in records if r.finding_id}
    py = [f for f in report.findings if f.id in by_finding and f.path.endswith(".py")]
    chosen = sorted(py, key=lambda f: f.id)[:max_findings]
    return chosen, {"records": by_finding, "cases": {c.path: c for c in manifest.cases}}


class Checkpoint:
    """Append-only JSONL of completed calls so long comparisons can resume.

    Each line holds the arm label, seed, finding id and the validated analysis (no prompts).
    A header line records the run parameters; a checkpoint written with different
    parameters is refused rather than silently mixed.
    """

    def __init__(self, path: Path | None, header: dict[str, object], fresh: bool = False):
        self.path = path
        self.done: dict[tuple[str, int, str], dict[str, object]] = {}
        if path is None:
            return
        if fresh and path.exists():
            path.unlink()
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            if lines and json.loads(lines[0]).get("header") != header:
                raise ValueError(
                    f"checkpoint {path} was written with different settings; "
                    "use --fresh or another --output"
                )
            for line in lines[1:]:
                row = json.loads(line)
                self.done[(row["arm"], row["run"], row["finding_id"])] = row
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"header": header}) + "\n", encoding="utf-8")

    def get(self, arm: Arm, run: int, finding_id: str) -> dict[str, object] | None:
        return self.done.get((arm.label, run, finding_id))

    def add(self, arm: Arm, run: int, finding: Finding, error: str | None) -> None:
        if self.path is None:
            return
        row = {
            "arm": arm.label,
            "run": run,
            "finding_id": finding.id,
            "error": error,
            "analysis": None if error else finding.analysis.model_dump(mode="json"),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_arm(
    arm: Arm,
    findings: Sequence[Finding],
    lookup: dict[str, object],
    runs: int,
    base: OllamaSettings,
    on_progress: Callable[[str], None] | None = None,
    checkpoint: Checkpoint | None = None,
) -> list[_Sample]:
    samples: list[_Sample] = []
    records = lookup["records"]
    cases = lookup["cases"]
    checkpoint = checkpoint or Checkpoint(None, {})
    for run in range(1, runs + 1):
        pending = []
        for f in findings:
            saved = checkpoint.get(arm, run, f.id)
            if saved is None:
                pending.append(f)
                continue
            copy = f.model_copy(deep=True)
            if saved["error"]:
                samples.append(_Sample(run, copy, None, error=str(saved["error"])))
                continue
            copy.analysis = FindingAnalysis.model_validate(saved["analysis"])
            m = copy.analysis.metrics
            ev = evaluate_finding(copy, cases[copy.path], records.get(copy.id), run)
            samples.append(_Sample(run, copy, ev, metrics=m.model_dump() if m else {}))
        if not pending:
            continue
        settings = (
            _settings(arm, base, run)
            if arm.mode == "profile"
            else OllamaSettings(
                base_url=base.base_url,
                model=arm.model,
                timeout_s=base.timeout_s,
                max_retries=base.max_retries,
                allow_remote=base.allow_remote,
                temperature=float(PINNED_OPTIONS["temperature"]),
                num_ctx=arm.num_ctx,
                top_p=float(PINNED_OPTIONS["top_p"]),
                top_k=int(PINNED_OPTIONS["top_k"]),
                repeat_penalty=float(PINNED_OPTIONS["repeat_penalty"]),
                presence_penalty=float(PINNED_OPTIONS["presence_penalty"]),
                num_predict=int(PINNED_OPTIONS["num_predict"]),
                seed=run,
            )
        )
        with OllamaClient(settings) as client:
            analyzer = FindingAnalyzer(
                client, profile=arm.profile, system_role=arm.mode != "profile"
            )
            for f in pending:
                copy = f.model_copy(deep=True)
                try:
                    copy.analysis = analyzer.analyze(copy)
                except OllamaError as exc:
                    samples.append(_Sample(run, copy, None, error=type(exc).__name__))
                    checkpoint.add(arm, run, copy, type(exc).__name__)
                    continue
                checkpoint.add(arm, run, copy, None)
                m = copy.analysis.metrics
                ev = evaluate_finding(copy, cases[copy.path], records.get(copy.id), run)
                samples.append(_Sample(run, copy, ev, metrics=m.model_dump() if m else {}))
                if on_progress:
                    on_progress(f"{arm.label} run {run}: {len(samples)} done")
    return samples


def _settings(arm: Arm, base: OllamaSettings, run: int) -> OllamaSettings:
    return OllamaSettings(
        base_url=base.base_url,
        model=arm.model,
        timeout_s=base.timeout_s,
        max_retries=base.max_retries,
        allow_remote=base.allow_remote,
        seed=run,
        pin_core_options=False,
    )


def summarize_arm(arm: Arm, samples: Sequence[_Sample]) -> ArmSummary:
    evals = [s.evaluation for s in samples if s.evaluation]
    tallies: dict[str, list[bool]] = defaultdict(list)
    for ev in evals:
        for c in ev.checks:
            if c.passed is not None:
                tallies[c.id].append(c.passed)
    checks = {}
    for cid in RUBRIC:
        vals = tallies.get(cid, [])
        lo, hi = wilson(sum(vals), len(vals))
        checks[cid] = {
            "passed": sum(vals),
            "assessed": len(vals),
            "rate": round(sum(vals) / len(vals), 4) if vals else None,
            "ci95_low": lo,
            "ci95_high": hi,
        }

    def mean(key: str) -> float | None:
        vals = [s.metrics.get(key) for s in samples if s.metrics.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None  # type: ignore[arg-type]

    lat = latency_stats(
        [
            float(s.metrics["wall_s"])
            for s in samples  # type: ignore[arg-type]
            if s.metrics.get("wall_s") is not None
        ]
    )
    return ArmSummary(
        arm=arm.label,
        model=arm.model,
        profile=arm.profile,
        num_ctx=arm.num_ctx,
        n=len(evals),
        status=dict(Counter(e.analysis_status for e in evals)),
        checks=checks,
        latency=lat.model_dump(),
        prompt_tokens_mean=mean("prompt_tokens"),
        prompt_cached_tokens_mean=mean("prompt_cached_tokens"),
        output_tokens_mean=mean("output_tokens"),
        output_tokens_per_s_mean=mean("output_tokens_per_s"),
        errors=sum(1 for s in samples if s.error),
    )


def paired(
    reference: Arm, ref: Sequence[_Sample], arm: Arm, other: Sequence[_Sample]
) -> list[PairedResult]:
    def index(samples: Sequence[_Sample]) -> dict[tuple[int, str], dict[str, bool | None]]:
        return {
            (s.run, s.finding.id): {c.id: c.passed for c in s.evaluation.checks}
            for s in samples
            if s.evaluation
        }

    a, b = index(ref), index(other)
    out = []
    for cid in RUBRIC:
        pairs = [(a[k].get(cid), b[k].get(cid)) for k in sorted(set(a) & set(b))]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        ref_only = sum(1 for x, y in pairs if x and not y)
        arm_only = sum(1 for x, y in pairs if y and not x)
        out.append(
            PairedResult(
                arm=arm.label,
                reference=reference.label,
                check=cid,
                pairs=len(pairs),
                reference_only=ref_only,
                arm_only=arm_only,
                mcnemar_p=mcnemar_exact(ref_only, arm_only),
            )
        )
    return out


def adoption_decision(
    arm: ArmSummary, ref: ArmSummary, pairs: Sequence[PairedResult], min_speedup: float = 0.10
) -> Adoption:
    """Optimization rule: adopt only if no paired check degrades by more than one sample,
    no safety check degrades at all, and median latency improves by >= ``min_speedup``."""
    reasons = []
    ok = True
    for p in pairs:
        worse = p.reference_only - p.arm_only
        if p.check in SAFETY_CHECKS and worse > 0:
            ok = False
            reasons.append(f"safety check {p.check} degraded ({worse} net)")
        elif worse > 1:
            ok = False
            reasons.append(f"{p.check} degraded ({worse} net paired losses)")
    r50, a50 = ref.latency.get("p50_s"), arm.latency.get("p50_s")
    if r50 and a50:
        gain = (float(r50) - float(a50)) / float(r50)
        reasons.append(f"p50 latency change {-gain:+.0%}")
        if gain < min_speedup:
            ok = False
            reasons.append(f"speed-up below {min_speedup:.0%}")
    if arm.errors > ref.errors:
        ok = False
        reasons.append("more request errors than reference")
    return Adoption(arm=arm.arm, adopt=ok, reasons=reasons)


def profile_identity_check(
    models: Sequence[str], findings: Sequence[Finding], base: OllamaSettings, n: int = 3
) -> dict[str, object]:
    """Same seed + pinned options + our system prompt: do two models that share weights
    produce identical raw outputs? Identical output means the Modelfile profile has no
    effect in this pipeline (any difference then comes from weights, not prompting)."""
    outputs: dict[str, list[str]] = {}
    for model in models:
        settings = OllamaSettings(
            base_url=base.base_url,
            model=model,
            timeout_s=base.timeout_s,
            temperature=0.1,
            num_ctx=8192,
            top_p=0.9,
            top_k=20,
            repeat_penalty=1.0,
            presence_penalty=0.0,
            num_predict=2048,
            seed=1,
        )
        with OllamaClient(settings) as client:
            from ..analysis import EXPLANATION_JSON_SCHEMA, build_messages

            outputs[model] = [
                client.chat(build_messages(f), EXPLANATION_JSON_SCHEMA) for f in findings[:n]
            ]
    first, *rest = models
    identical = all(outputs[m] == outputs[first] for m in rest)
    return {"models": list(models), "samples": n, "identical_outputs": identical}


def run_comparison(
    corpus: Path,
    arms: Sequence[Arm],
    *,
    runs: int = 3,
    max_findings: int = 40,
    ollama: OllamaSettings | None = None,
    identity_pair: Sequence[str] = (),
    on_progress: Callable[[str], None] | None = None,
    checkpoint_path: Path | None = None,
    fresh: bool = False,
) -> ComparisonResult:
    ollama = ollama or OllamaSettings()
    validate_base_url(ollama.base_url, allow_remote=ollama.allow_remote)
    started = now()
    manifest, cases_root = load_manifest(corpus)
    names = sorted({a.model for a in arms} | set(identity_pair))
    provenance = model_provenance(ollama.base_url, names)
    installed = {p.name for p in provenance if p.installed}
    notes = [f"{p.name} not installed; skipped" for p in provenance if not p.installed]
    arms = [a for a in arms if a.model in installed]
    if not arms:
        raise ValueError("none of the requested models is installed")
    with tempfile.TemporaryDirectory(prefix="tantular-cmp-") as tmp:
        findings, lookup = select_findings(manifest, cases_root, max_findings, Path(tmp))
    header = {
        "corpus": manifest.corpus_id,
        "manifest_version": manifest.manifest_version,
        "findings": sorted(f.id for f in findings),
        "runs": runs,
        "pinned_options": PINNED_OPTIONS,
    }
    checkpoint = Checkpoint(checkpoint_path, header, fresh=fresh)
    if checkpoint.done:
        notes.append(f"resumed from checkpoint: {len(checkpoint.done)} completed call(s) reused")
    samples: dict[str, list[_Sample]] = {}
    for arm in arms:
        t0 = time.monotonic()
        samples[arm.label] = run_arm(arm, findings, lookup, runs, ollama, on_progress, checkpoint)
        notes.append(f"{arm.label}: {time.monotonic() - t0:.0f} s wall")
    summaries = [summarize_arm(a, samples[a.label]) for a in arms]
    reference = arms[0]
    pairs = [
        p
        for a in arms[1:]
        for p in paired(reference, samples[reference.label], a, samples[a.label])
    ]
    adoption = [
        adoption_decision(s, summaries[0], [p for p in pairs if p.arm == s.arm])
        for s in summaries[1:]
        if s.model == reference.model
    ]
    identity = None
    if len(identity_pair) == 2 and all(m in installed for m in identity_pair):
        identity = profile_identity_check(identity_pair, findings, ollama)
    # Ollama can rewrite a manifest during a run (e.g. its GGUF compat migration on first
    # load), so the pre-run snapshot alone may not describe the weights that were served.
    drift = provenance_drift(provenance, model_provenance(ollama.base_url, names))
    for s in summaries:
        s.provenance_changed = s.model in drift
    commit, _ = code_state()
    return ComparisonResult(
        started_at=started,
        finished_at=now(),
        machine=machine(),
        code_commit=commit,
        corpus_id=manifest.corpus_id,
        findings=len(findings),
        runs=runs,
        pinned_options=PINNED_OPTIONS,
        reference=reference.label,
        models=provenance,
        arms=summaries,
        paired=pairs,
        adoption=adoption,
        profile_identity=identity,
        provenance_changes=[c for diffs in drift.values() for c in diffs],
        notes=notes,
    )


def render_comparison(r: ComparisonResult) -> str:
    out = [
        "# Local model / prompt-profile comparison (sample-based)",
        "",
        f"Findings {r.findings} × runs {r.runs} per arm; reference arm `{r.reference}`; "
        f"pinned options `{r.pinned_options}`; started {r.started_at}.",
        "",
    ]
    out += [f"> {c}" for c in r.caveats] + [
        "",
        "## Models",
        "",
        "| Model | Installed | Digest | Weights blob | Quant | Params | Modelfile SYSTEM "
        "| Shares weights with |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in r.models:
        out.append(
            f"| `{m.name}` | {m.installed} | `{m.digest or '-'}` | `{m.weights_blob or '-'}` | "
            f"{m.quantization or '-'} | {m.parameter_size or '-'} | "
            f"{m.modelfile_has_system} | {', '.join(m.shares_weights_with) or '-'} |"
        )
    same = [
        (a, b)
        for i, a in enumerate(r.models)
        for b in r.models[i + 1 :]
        if a.installed and a.weights_blob and a.weights_blob == b.weights_blob
    ]
    for a, b in same:
        diffs = [
            f"SYSTEM prompt (hash `{a.system_sha256}` vs `{b.system_sha256}`)"
            if a.system_sha256 != b.system_sha256
            else ""
        ]
        for key in sorted(set(a.modelfile_parameters) | set(b.modelfile_parameters)):
            va, vb = a.modelfile_parameters.get(key), b.modelfile_parameters.get(key)
            if va != vb:
                diffs.append(f"`{key}` {va} vs {vb}")
        diffs = [d for d in diffs if d]
        out += [
            "",
            f"**Same weights:** `{a.name}` and `{b.name}` share weights blob "
            f"`{a.weights_blob}`. Their Modelfile profiles differ in: "
            + ("; ".join(diffs) or "nothing detected")
            + ". In `~modelfile-profile` arms any difference is the **combined** profile "
            "effect of all of these, not of the SYSTEM prompt alone.",
        ]
    if r.provenance_changes:
        out += [
            "",
            "**Provenance changed during the run** (the table shows the pre-run snapshot; "
            "affected arms may mix weights — check the Ollama server log):",
            "",
            *[f"- {c}" for c in r.provenance_changes],
        ]
    if r.profile_identity:
        pi = r.profile_identity
        out += [
            "",
            f"Profile identity check ({', '.join(pi['models'])}, {pi['samples']} "  # type: ignore[arg-type]
            f"samples, seed 1): identical outputs = **{pi['identical_outputs']}**.",
        ]
    out += [
        "",
        "## Arms",
        "",
        "| Arm | n | " + " | ".join(RUBRIC) + " | p50 s | p95 s | "
        "prompt tok | cached tok | output tok | tok/s | errors |",
        "|---|---|" + "---|" * len(RUBRIC) + "---|---|---|---|---|---|---|",
    ]
    for a in r.arms:
        cells = []
        for cid in RUBRIC:
            c = a.checks[cid]
            cells.append(
                "n/a"
                if c["rate"] is None
                else f"{c['rate']:.0%} ({c['ci95_low']:.2f}–{c['ci95_high']:.2f})"
            )
        out.append(
            f"| `{a.arm}`{' ⚠ provenance changed' if a.provenance_changed else ''} | {a.n} | "
            + " | ".join(cells)
            + f" | {a.latency.get('p50_s')} | {a.latency.get('p95_s')} | "
            f"{a.prompt_tokens_mean} | {a.prompt_cached_tokens_mean} | "
            f"{a.output_tokens_mean} | {a.output_tokens_per_s_mean} | {a.errors} |"
        )
    out += [
        "",
        "## Paired comparison vs reference (discordant pairs, exact McNemar)",
        "",
        "| Arm | Check | Pairs | Ref only | Arm only | p |",
        "|---|---|---|---|---|---|",
    ]
    out += [
        f"| `{p.arm}` | {p.check} | {p.pairs} | {p.reference_only} | {p.arm_only} | "
        f"{p.mcnemar_p if p.mcnemar_p is not None else '-'} |"
        for p in r.paired
        if p.reference_only or p.arm_only
    ]
    if r.adoption:
        out += ["", "## Optimization adoption (same model, different profile)", ""]
        out += [
            f"- `{a.arm}`: **{'adopt' if a.adopt else 'do not adopt'}** — " + "; ".join(a.reasons)
            for a in r.adoption
        ]
    if r.notes:
        out += ["", "## Notes", ""] + [f"- {n}" for n in r.notes]
    return "\n".join(out) + "\n"


def parse_arms(models: str, profiles: str, contexts: str, modes: str = "pinned") -> list[Arm]:
    ms = [m.strip() for m in models.split(",") if m.strip()]
    ps = [p.strip() for p in profiles.split(",") if p.strip()] or ["baseline"]
    cs = [int(c) for c in contexts.split(",") if c.strip()] or [8192]
    md = [m.strip() for m in modes.split(",") if m.strip()] or ["pinned"]
    for m in md:
        if m not in ("pinned", "profile"):
            raise ValueError(f"unknown mode {m}; choose pinned or profile")
    for p in ps:
        if p not in PROMPT_PROFILES:
            raise ValueError(f"unknown profile {p}; choose from {sorted(PROMPT_PROFILES)}")
    arms: list[Arm] = []
    for mode in md:
        for m in ms:
            for p in ps:
                for c in cs if mode == "pinned" else [8192]:
                    arm = Arm(m, p, c, mode)
                    if arm not in arms:
                        arms.append(arm)
    return arms
