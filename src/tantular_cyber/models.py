"""Pydantic models: normalized findings, LLM explanation contract, and the versioned report.

``ScanReport`` is the stable machine-readable contract (``report.json``) intended for
downstream consumers such as GATRA. Bump ``SCHEMA_VERSION`` on any breaking change and
regenerate ``docs/report-schema.json`` (``tantular-cyber schema > docs/report-schema.json``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.2.0"
REPORT_TYPE = "tantular-cyber.repository-scan"

Severity = Literal["critical", "high", "medium", "low", "info"]
Confidence = Literal["low", "medium", "high"]
ReportStatus = Literal["complete", "degraded", "failed"]
AnalysisStatus = Literal["completed", "invalid_output", "error", "skipped", "disabled"]

SEVERITY_ORDER: dict[str, int] = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

ADVISORY_DISCLAIMER = (
    "Penjelasan ini dihasilkan oleh model bahasa lokal dan bersifat saran (advisory). "
    "Semua klaim model belum diverifikasi dan harus diperiksa secara independen. "
    "Bukti dari scanner adalah sumber otoritatif."
)

_MAX_ITEMS = 10
_MAX_ITEM_LEN = 1200


class EvidenceLine(BaseModel):
    line: int
    text: str


class Evidence(BaseModel):
    start_line: int
    end_line: int
    lines: list[EvidenceLine]
    snippet: str = Field(description="Exact source lines, prefixed with line numbers, redacted.")
    truncated: bool = False
    redactions: int = 0
    context: str = Field(
        default="",
        description="Surrounding lines (+/- context_lines) with line numbers, redacted. "
        "Reader aid only; the match is defined by start_line..end_line.",
    )
    context_kind: Literal["ast_function", "ast_module", "window"] = Field(
        default="window",
        description="How context was selected (schema 1.2): enclosing function via AST, "
        "module-level statement via AST, or a fixed line window.",
    )
    context_lines: list[int] = Field(
        default_factory=list, description="Source line numbers included in ``context``."
    )
    context_truncated: bool = False
    context_fallback: str | None = Field(
        default=None, description="Why AST context was not used (e.g. syntax error)."
    )
    dataflow_hints: list[str] = Field(
        default_factory=list,
        description="Syntactic hints (parameters, local assignments, same-file helpers). "
        "Explanatory only; not proof that untrusted data reaches the sink.",
    )


def _coerce_str_list(value: object) -> object:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v).strip()][:_MAX_ITEMS]
    return value


class LLMExplanation(BaseModel):
    """Contract the model must satisfy (validated; extra keys ignored)."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(min_length=1, max_length=4000)
    confirmed_evidence: list[str] = Field(
        description="Only facts directly visible in the scanner evidence."
    )
    conditional_impact: list[str] = Field(
        description="Impact that MAY occur, with the conditions required."
    )
    remediation: list[str]
    corrected_code: str | None = Field(
        default=None, description="Only when supported by the evidence; never applied."
    )
    confidence: Confidence
    limitations: list[str] = Field(default_factory=list)

    @field_validator(
        "confirmed_evidence", "conditional_impact", "remediation", "limitations", mode="before"
    )
    @classmethod
    def _lists(cls, value: object) -> object:
        return _coerce_str_list(value)

    @field_validator("confirmed_evidence", "conditional_impact", "remediation", "limitations")
    @classmethod
    def _bound_items(cls, value: list[str]) -> list[str]:
        return [v[:_MAX_ITEM_LEN] for v in value]

    @field_validator("confidence", mode="before")
    @classmethod
    def _lower_confidence(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("corrected_code", mode="before")
    @classmethod
    def _empty_code(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class LLMMetrics(BaseModel):
    """Per-call timing/token counters reported by the LLM server (Ollama ``/api/chat``)."""

    wall_s: float = Field(description="Client-measured request time.")
    total_s: float | None = None
    load_s: float | None = Field(default=None, description="Model load time (cold start).")
    prompt_tokens: int | None = None
    prompt_eval_s: float | None = None
    output_tokens: int | None = None
    eval_s: float | None = None
    output_tokens_per_s: float | None = None


class LLMPerformance(BaseModel):
    """Aggregate of ``LLMMetrics`` over all completed LLM calls in a run."""

    calls: int
    total_wall_s: float
    mean_wall_s: float
    min_wall_s: float
    max_wall_s: float
    first_call_load_s: float | None
    total_load_s: float
    prompt_tokens: int
    output_tokens: int
    mean_output_tokens_per_s: float | None


class FindingAnalysis(BaseModel):
    status: AnalysisStatus
    provider: str | None = None
    model: str | None = None
    language: str = "id"
    explanation: LLMExplanation | None = None
    verified: Literal[False] = False
    disclaimer: str = ADVISORY_DISCLAIMER
    validation_warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    duration_s: float | None = None
    metrics: LLMMetrics | None = None


class Finding(BaseModel):
    id: str = Field(description="Deterministic: sha256(tool|rule|path|start|end|evidence)[:16].")
    tool: str
    rule_id: str
    cwe: list[str]
    severity: Severity
    path: str = Field(description="Repository-relative POSIX path.")
    start_line: int
    end_line: int
    evidence: Evidence
    message: str
    confidence: str | None = Field(default=None, description="Confidence declared by the rule.")
    confidence_source: str = "scanner_rule_metadata"
    status: Literal["unverified"] = "unverified"
    analysis: FindingAnalysis = Field(default_factory=lambda: FindingAnalysis(status="disabled"))


class Generator(BaseModel):
    name: str = "tantular-cyber"
    version: str


class SkippedEntry(BaseModel):
    path: str
    reason: str


class TargetInfo(BaseModel):
    name: str = Field(description="Repository directory name only (no absolute path).")
    files_scanned: int = Field(description="Eligible files submitted to the scanners.")
    files_skipped: int = Field(description="Files excluded by the inventory safety policy.")
    ignore_files_present: list[str] = Field(
        default_factory=list,
        description="Repository-controlled ignore files found (e.g. .semgrepignore). They are "
        "NOT used to select targets; coverage is verified per scanner instead.",
    )


class ScanCoverage(BaseModel):
    """What a scanner actually analyzed, verified against the tool's own output."""

    languages: list[str] = Field(description="Languages declared by the rules.")
    targets_submitted: int
    in_scope: int = Field(description="Submitted files whose language the rules cover.")
    scanned: int = Field(description="In-scope files the tool reported as scanned.")
    not_scanned: list[SkippedEntry] = Field(
        description="In-scope files the tool did not scan (coverage gap)."
    )
    out_of_scope: int = Field(description="Submitted files no rule language applies to.")
    verified: bool = Field(description="Whether the tool reported its scanned paths.")


class ScannerInfo(BaseModel):
    name: str
    version: str | None
    exit_code: int | None
    duration_s: float
    rules: list[str]
    targets: int
    succeeded: bool
    coverage: ScanCoverage | None = None


class LLMInfo(BaseModel):
    enabled: bool
    provider: str | None = None
    endpoint: str | None = None
    model: str | None = None
    server_version: str | None = None
    available: bool | None = None
    error: str | None = None
    performance: LLMPerformance | None = None


class SkippedSummary(BaseModel):
    total: int
    by_reason: dict[str, int]
    entries: list[SkippedEntry] = Field(description="Capped sample of skipped paths.")
    entries_truncated: bool


class Totals(BaseModel):
    findings: int
    by_severity: dict[str, int]
    by_cwe: dict[str, int]
    analyses_by_status: dict[str, int]


class ReportError(BaseModel):
    stage: Literal["input", "scanner", "normalize", "llm", "report"]
    message: str


class ScanReport(BaseModel):
    schema_version: str = SCHEMA_VERSION
    report_type: str = REPORT_TYPE
    generator: Generator
    scan_id: str
    status: ReportStatus
    status_reasons: list[str]
    started_at: str
    finished_at: str
    duration_s: float
    target: TargetInfo
    scanners: list[ScannerInfo]
    llm: LLMInfo
    totals: Totals
    findings: list[Finding]
    skipped: SkippedSummary
    errors: list[ReportError]
    limitations: list[str]
