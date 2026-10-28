"""LLM latency/throughput benchmark for the Tantular Cyber pipeline (local Ollama).

Runs the full scan pipeline N times on a multi-finding fixture. Run 1 starts from a cold
model (unloaded with ``keep_alive: 0``) unless ``--warm-only``; later runs are warm.

    .venv/bin/python benchmarks/llm_latency.py --runs 2 --output out/bench

Writes ``latency.json`` and ``latency.md`` to ``--output``. Requires semgrep + Ollama.
Measures performance only; explanation quality is out of scope (see docs/benchmark-plan.md).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from tantular_cyber.config import OllamaSettings, ScanSettings
from tantular_cyber.ollama_client import validate_base_url
from tantular_cyber.pipeline import run_scan

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT / "tests" / "fixtures" / "bench"


def machine_info() -> dict[str, Any]:
    def sysctl(key: str) -> str | None:
        try:
            out = subprocess.run(
                ["/usr/sbin/sysctl", "-n", key], capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() or None

    mem = sysctl("hw.memsize")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": sysctl("machdep.cpu.brand_string"),
        "memory_gb": round(int(mem) / 2**30) if mem and mem.isdigit() else None,
        "python": platform.python_version(),
    }


def model_info(http: httpx.Client, model: str) -> dict[str, Any]:
    tags = http.get("/api/tags").json().get("models", [])
    entry = next((m for m in tags if m.get("name") == model), {})
    details = entry.get("details", {})
    return {
        "name": model,
        "parameter_size": details.get("parameter_size"),
        "quantization": details.get("quantization_level"),
        "size_gb": round(entry.get("size", 0) / 1e9, 1) if entry.get("size") else None,
        "server_version": http.get("/api/version").json().get("version"),
    }


def unload(http: httpx.Client, model: str, timeout_s: float = 60.0) -> None:
    """Ask Ollama to evict the model so the next call includes a cold load."""
    http.post("/api/generate", json={"model": model, "keep_alive": 0})
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        loaded = [m.get("name") for m in http.get("/api/ps").json().get("models", [])]
        if model not in loaded:
            return
        time.sleep(1)


def run_once(repo: Path, ollama: OllamaSettings, label: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        report = run_scan(ScanSettings(repo=repo, output=Path(tmp) / "out", ollama=ollama)).report
    per_finding = [
        {
            "finding": f"{f.path}:{f.start_line}",
            "status": f.analysis.status,
            "warnings": len(f.analysis.validation_warnings),
            **(f.analysis.metrics.model_dump() if f.analysis.metrics else {}),
        }
        for f in report.findings
    ]
    perf = report.llm.performance
    return {
        "label": label,
        "status": report.status,
        "findings": report.totals.findings,
        "analyses": report.totals.analyses_by_status,
        "scan_duration_s": report.duration_s,
        "scanner_duration_s": sum(s.duration_s for s in report.scanners),
        "performance": perf.model_dump() if perf else None,
        "per_finding": per_finding,
    }


def render_markdown(result: dict[str, Any]) -> str:
    m, mdl = result["machine"], result["model"]
    lines = [
        "# LLM latency benchmark",
        "",
        f"- Date: {result['date']}",
        f"- Machine: {m['cpu']} ({m['machine']}), {m['memory_gb']} GB RAM, {m['platform']}",
        f"- Ollama {mdl['server_version']}, model `{mdl['name']}` "
        f"({mdl['parameter_size']}, {mdl['quantization']}, {mdl['size_gb']} GB)",
        f"- Fixture: `{result['fixture']}`, think=false, stream=false, temperature 0.1, "
        "num_ctx 8192, sequential calls",
        "",
        "| Run | Status | Findings | Total s | Mean s/finding | Min | Max | First-call load s "
        "| Prompt tok | Output tok | Output tok/s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in result["runs"]:
        p = run["performance"] or {}
        lines.append(
            f"| {run['label']} | {run['status']} | {run['findings']} | "
            f"{run['scan_duration_s']:.1f} | {p.get('mean_wall_s', 0):.1f} | "
            f"{p.get('min_wall_s', 0):.1f} | {p.get('max_wall_s', 0):.1f} | "
            f"{p.get('first_call_load_s') or 0:.1f} | {p.get('prompt_tokens', 0)} | "
            f"{p.get('output_tokens', 0)} | {p.get('mean_output_tokens_per_s') or 0:.1f} |"
        )
    lines += [
        "",
        "Per-finding (all runs):",
        "",
        "| Run | Finding | Status | Wall s | Load s | Prompt tok | Output tok | tok/s "
        "| Guard warnings |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run in result["runs"]:
        for f in run["per_finding"]:
            lines.append(
                f"| {run['label']} | `{f['finding']}` | {f['status']} | {f.get('wall_s', 0):.1f} "
                f"| {f.get('load_s') or 0:.2f} | {f.get('prompt_tokens')} | "
                f"{f.get('output_tokens')} | {f.get('output_tokens_per_s')} | {f['warnings']} |"
            )
    walls = [f["wall_s"] for r in result["runs"] for f in r["per_finding"] if "wall_s" in f]
    if walls:
        lines += [
            "",
            f"Median per-finding wall time across runs: {statistics.median(walls):.1f} s",
            "",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warm-only", action="store_true", help="do not unload the model first")
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "bench")
    args = parser.parse_args(argv)

    ollama = OllamaSettings.from_env()
    base_url = validate_base_url(ollama.base_url, allow_remote=ollama.allow_remote)
    with httpx.Client(base_url=base_url, timeout=120, trust_env=False) as http:
        info = model_info(http, ollama.model)
        runs = []
        for i in range(1, args.runs + 1):
            cold = i == 1 and not args.warm_only
            if cold:
                unload(http, ollama.model)
            label = f"{i} ({'cold' if cold else 'warm'})"
            print(f"run {label} ...", file=sys.stderr, flush=True)
            runs.append(run_once(args.repo.resolve(), ollama, label))

    try:
        fixture = args.repo.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        fixture = args.repo.name
    result = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": machine_info(),
        "model": info,
        "fixture": fixture,
        "runs": runs,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "latency.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "latency.md").write_text(render_markdown(result))
    print(render_markdown(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
