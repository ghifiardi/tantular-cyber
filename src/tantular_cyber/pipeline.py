"""Orchestrator: inventory -> scanners -> normalized findings -> LLM explanations -> report.

The pipeline never raises for scanner or LLM failures; they are recorded in the report and
reflected in its ``status`` (``complete`` / ``degraded`` / ``failed``). Only invalid input
paths raise ``PathSafetyError`` before any work is done.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .analysis import FindingAnalyzer, LLMClient
from .config import ScanSettings
from .models import (
    SEVERITY_ORDER,
    Evidence,
    EvidenceLine,
    Finding,
    FindingAnalysis,
    Generator,
    LLMInfo,
    ReportError,
    ScannerInfo,
    ScanReport,
    SkippedEntry,
    SkippedSummary,
    TargetInfo,
    Totals,
)
from .ollama_client import OllamaClient, OllamaError, OllamaUnavailable
from .scanner import RawFinding, Scanner, ScannerRun, SemgrepScanner
from .security import FileInventory, collect_files, redact, resolve_output, resolve_repo

log = logging.getLogger(__name__)

MAX_EVIDENCE_LINES = 20
MAX_SKIPPED_ENTRIES = 200

BASE_LIMITATIONS = [
    "Static analysis only: findings are pattern matches and remain 'unverified' until a human "
    "confirms reachability and exploitability.",
    "The bundled ruleset is intentionally small (Python CWE-89 via DB-API execute with string "
    "building); absence of findings is not evidence of absence of vulnerabilities.",
    "Rules are intra-procedural and do not track taint across functions or files.",
    "LLM explanations are advisory, may be incorrect or overstated, and are never treated as "
    "verification. Scanner evidence is authoritative.",
    "Excluded from scanning: VCS metadata, dependency/cache directories, likely secret files, "
    "binary and oversized files, and symlinks.",
    "No repository code was executed, no dependencies were installed, and no code was modified.",
]


@dataclass
class ScanResult:
    report: ScanReport
    repo: Path
    output: Path
    raw_scanner_outputs: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def build_evidence(lines: list[str], start: int, end: int) -> Evidence:
    start = max(1, start)
    end = max(start, min(end, len(lines)))
    shown_end = min(end, start + MAX_EVIDENCE_LINES - 1)
    width = len(str(shown_end))
    ev_lines: list[EvidenceLine] = []
    redactions = 0
    for n in range(start, shown_end + 1):
        text, count = redact(lines[n - 1] if n - 1 < len(lines) else "")
        redactions += count
        ev_lines.append(EvidenceLine(line=n, text=text))
    snippet = "\n".join(f"{ln.line:>{width}} | {ln.text}" for ln in ev_lines)
    return Evidence(
        start_line=start,
        end_line=end,
        lines=ev_lines,
        snippet=snippet,
        truncated=shown_end < end,
        redactions=redactions,
    )


def build_context(lines: list[str], start: int, end: int, radius: int) -> str:
    lo, hi = max(1, start - radius), min(len(lines), end + radius)
    width = len(str(hi))
    out = [f"{n:>{width}} | {lines[n - 1]}" for n in range(lo, hi + 1)]
    return redact("\n".join(out))[0]


def finding_id(raw: RawFinding, evidence: Evidence) -> str:
    material = "|".join(
        [
            raw.tool,
            raw.rule_id,
            raw.path,
            str(raw.start_line),
            str(raw.end_line),
            "\n".join(ln.text.strip() for ln in evidence.lines),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def normalize_findings(
    repo: Path, raws: Sequence[RawFinding], eligible: set[str], context_lines: int
) -> tuple[list[Finding], list[ReportError]]:
    findings: dict[str, Finding] = {}
    errors: list[ReportError] = []
    cache: dict[str, list[str]] = {}
    for raw in raws:
        if raw.path not in eligible:
            errors.append(
                ReportError(
                    stage="normalize", message="finding for a non-target file was discarded"
                )
            )
            continue
        if raw.path not in cache:
            cache[raw.path] = read_lines(repo / raw.path)
        lines = cache[raw.path]
        evidence = build_evidence(lines, raw.start_line, raw.end_line)
        fid = finding_id(raw, evidence)
        if fid in findings:
            continue
        evidence.context = build_context(
            lines, evidence.start_line, evidence.end_line, context_lines
        )
        findings[fid] = Finding(
            id=fid,
            tool=raw.tool,
            rule_id=raw.rule_id,
            cwe=list(raw.cwe),
            severity=raw.severity,
            path=raw.path,
            start_line=evidence.start_line,
            end_line=evidence.end_line,
            evidence=evidence,
            message=raw.message,
            confidence=raw.confidence,
        )
    ordered = sorted(
        findings.values(),
        key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.path, f.start_line, f.rule_id),
    )
    return ordered, errors


def _skipped_summary(inventory: FileInventory) -> SkippedSummary:
    entries = [
        SkippedEntry(path=s.path, reason=s.reason) for s in inventory.skipped[:MAX_SKIPPED_ENTRIES]
    ]
    return SkippedSummary(
        total=len(inventory.skipped),
        by_reason=dict(sorted(Counter(s.reason for s in inventory.skipped).items())),
        entries=entries,
        entries_truncated=len(inventory.skipped) > MAX_SKIPPED_ENTRIES,
    )


def _analyze(
    findings: list[Finding],
    settings: ScanSettings,
    client_factory: Callable[[], LLMClient],
    errors: list[ReportError],
    reasons: list[str],
) -> LLMInfo:
    ollama = settings.ollama
    info = LLMInfo(
        enabled=settings.llm_enabled,
        provider="ollama",
        model=ollama.model,
        endpoint=redact(ollama.base_url)[0],
    )
    if not settings.llm_enabled:
        for f in findings:
            f.analysis = FindingAnalysis(status="disabled")
        return info

    def mark_all(status: str, error: str | None) -> None:
        for f in findings:
            if f.analysis.status in ("disabled", "skipped") and f.analysis.error is None:
                f.analysis = FindingAnalysis(
                    status=status,
                    error=error,  # type: ignore[arg-type]
                    provider="ollama",
                    model=ollama.model,
                )

    try:
        client = client_factory()
    except OllamaError as exc:
        info.available, info.error = False, str(exc)
        errors.append(ReportError(stage="llm", message=str(exc)))
        reasons.append("LLM explanation unavailable: invalid or disallowed Ollama endpoint")
        mark_all("skipped", "LLM unavailable")
        return info

    info.provider = client.provider
    info.model = client.model
    info.endpoint = getattr(client, "base_url", None)
    try:
        health_check = getattr(client, "health_check", None)
        if findings:
            info.server_version = health_check() if callable(health_check) else None
            info.available = True
    except OllamaError as exc:
        info.available, info.error = False, str(exc)
        errors.append(ReportError(stage="llm", message=str(exc)))
        reasons.append(
            "LLM explanation unavailable: Ollama not reachable or model missing; "
            "report contains scanner findings only"
        )
        mark_all("skipped", "LLM unavailable")
        _close(client)
        return info

    analyzer = FindingAnalyzer(client)
    limit = max(0, settings.max_llm_findings)
    try:
        for index, finding in enumerate(findings):
            if index >= limit:
                finding.analysis = FindingAnalysis(
                    status="skipped",
                    provider=info.provider,
                    model=info.model,
                    error=f"exceeds --max-llm-findings={limit}",
                )
                continue
            try:
                finding.analysis = analyzer.analyze(finding)
                log.info("analyzed finding %s: %s", finding.id, finding.analysis.status)
            except OllamaUnavailable as exc:
                info.available, info.error = False, str(exc)
                errors.append(ReportError(stage="llm", message=str(exc)))
                finding.analysis = FindingAnalysis(
                    status="error", error=str(exc), provider=info.provider, model=info.model
                )
                for rest in findings[index + 1 :]:
                    rest.analysis = FindingAnalysis(
                        status="skipped",
                        error="LLM became unavailable",
                        provider=info.provider,
                        model=info.model,
                    )
                break
            except OllamaError as exc:
                errors.append(ReportError(stage="llm", message=f"{finding.id}: {exc}"))
                finding.analysis = FindingAnalysis(
                    status="error", error=str(exc), provider=info.provider, model=info.model
                )
    finally:
        _close(client)

    statuses = Counter(f.analysis.status for f in findings)
    if statuses.get("error") or statuses.get("invalid_output"):
        reasons.append(
            f"{statuses.get('error', 0)} LLM error(s) and "
            f"{statuses.get('invalid_output', 0)} invalid model output(s)"
        )
    if statuses.get("skipped"):
        reasons.append(f"{statuses['skipped']} finding(s) not explained by the LLM")
    return info


def _close(client: object) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def run_scan(
    settings: ScanSettings,
    *,
    scanners: Sequence[Scanner] | None = None,
    llm_client_factory: Callable[[], LLMClient] | None = None,
) -> ScanResult:
    started_at, t0 = _now(), time.monotonic()
    repo = resolve_repo(settings.repo)
    output = resolve_output(
        settings.output, repo, allow_inside_repo=settings.allow_output_inside_repo
    )

    errors: list[ReportError] = []
    reasons: list[str] = []
    inventory = collect_files(repo, max_file_bytes=settings.max_file_bytes)
    log.info("inventory: %d target(s), %d skipped", len(inventory.files), len(inventory.skipped))

    if scanners is None:
        scanners = [
            SemgrepScanner(
                settings.rules,
                binary=settings.semgrep_bin,
                timeout_s=settings.semgrep_timeout_s,
                max_target_bytes=settings.max_file_bytes,
            )
        ]

    runs: list[ScannerRun] = []
    for scanner in scanners:
        run = scanner.scan(repo, inventory.files)
        runs.append(run)
        errors.extend(ReportError(stage="scanner", message=e) for e in run.errors)

    raws = [f for run in runs for f in run.findings]
    findings, norm_errors = normalize_findings(
        repo, raws, set(inventory.files), settings.context_lines
    )
    errors.extend(norm_errors)

    scanner_failed = any(not r.succeeded for r in runs)
    if scanner_failed:
        reasons.append("one or more scanners failed; findings may be incomplete or missing")

    factory = llm_client_factory or (lambda: OllamaClient(settings.ollama))
    llm_info = _analyze(findings, settings, factory, errors, reasons)

    if scanner_failed and not any(r.succeeded for r in runs):
        status = "failed"
    elif reasons:
        status = "degraded"
    else:
        status = "complete"

    limitations = list(BASE_LIMITATIONS)
    if not settings.llm_enabled:
        limitations.append("LLM explanations were disabled for this run (--no-llm).")

    report = ScanReport(
        generator=Generator(version=__version__),
        scan_id=str(uuid.uuid4()),
        status=status,
        status_reasons=reasons,
        started_at=started_at,
        finished_at=_now(),
        duration_s=round(time.monotonic() - t0, 3),
        target=TargetInfo(
            name=repo.name, files_scanned=len(inventory.files), files_skipped=len(inventory.skipped)
        ),
        scanners=[
            ScannerInfo(
                name=r.tool,
                version=r.version,
                exit_code=r.exit_code,
                duration_s=r.duration_s,
                rules=r.rules,
                targets=r.targets,
                succeeded=r.succeeded,
            )
            for r in runs
        ],
        llm=llm_info,
        totals=Totals(
            findings=len(findings),
            by_severity=dict(Counter(f.severity for f in findings)),
            by_cwe=dict(Counter(c for f in findings for c in (f.cwe or ["unmapped"]))),
            analyses_by_status=dict(Counter(f.analysis.status for f in findings)),
        ),
        findings=findings,
        skipped=_skipped_summary(inventory),
        errors=errors,
        limitations=limitations,
    )
    return ScanResult(
        report=report,
        raw_scanner_outputs={r.tool: r.raw_outputs for r in runs},
        repo=repo,
        output=output,
    )
