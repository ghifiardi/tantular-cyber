"""Strict contracts for the application; never reuse the scanner explanation prompt."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["analyst", "dataset", "validator", "reporter"]
ToolName = Literal[
    "read_file",
    "list_files",
    "search_files",
    "run_check",
    "scan",
    "validate_dataset",
    "write_draft",
    "artifact_index",
]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Limits(Strict):
    max_tasks: int = Field(default=8, ge=1, le=20)
    max_steps: int = Field(default=32, ge=1, le=200)
    max_model_calls: int = Field(default=24, ge=1, le=200)
    max_retries: int = Field(default=1, ge=0, le=3)
    max_output_bytes: int = Field(default=24000, ge=512, le=200000)
    request_timeout: float = Field(default=60.0, ge=0.05, le=600)
    max_duration: float = Field(default=300.0, ge=0.05, le=7200)
    process_timeout: float = Field(default=30.0, ge=0.05, le=600)
    max_file_bytes: int = Field(default=100000, ge=100, le=1000000)
    max_files: int = Field(default=2000, ge=1, le=20000)
    num_predict: int = Field(default=1600, ge=128, le=8192)
    num_ctx: int = Field(default=0, ge=0, le=131072)

    @model_validator(mode="after")
    def valid_context(self) -> Limits:
        if self.num_ctx != 0 and self.num_ctx < 512:
            raise ValueError("num_ctx must be 0 (auto) or at least 512")
        return self


class Assignment(Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    role: Role
    instruction: str = Field(min_length=1, max_length=2000)
    depends_on: list[str] = Field(default_factory=list, max_length=20)


class Plan(Strict):
    language: Literal["id", "en"]
    summary: str = Field(min_length=1, max_length=600)
    tasks: list[Assignment] = Field(max_length=20)
    question: str | None = Field(default=None, max_length=600)

    @model_validator(mode="after")
    def dependencies(self) -> Plan:
        ids = {task.id for task in self.tasks}
        if len(ids) != len(self.tasks):
            raise ValueError("duplicate task id")
        if bool(self.tasks) == bool(self.question):
            raise ValueError("provide tasks or one clarification question")
        resolved: set[str] = set()
        while len(resolved) < len(ids):
            ready = {
                t.id for t in self.tasks if t.id not in resolved and set(t.depends_on) <= resolved
            }
            if not ready:
                raise ValueError("cyclic or missing dependency")
            resolved.update(ready)
        return self


class ToolRequest(Strict):
    name: ToolName
    arguments: dict[str, object]


class WorkerResult(Strict):
    status: Literal["completed", "failed", "blocked"]
    summary: str = Field(min_length=1, max_length=1500)
    evidence: list[str] = Field(max_length=32)
    artifact_paths: list[str] = Field(max_length=32)
    limitations: list[str] = Field(max_length=16)
    blocker: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def honest_status(self) -> WorkerResult:
        if self.status != "completed" and not self.blocker:
            raise ValueError("failed or blocked result requires a blocker")
        return self


class Decision(Strict):
    action: Literal["tool", "result"]
    tool: ToolRequest | None
    result: WorkerResult | None

    @model_validator(mode="after")
    def exclusive(self) -> Decision:
        if self.action == "tool" and (self.tool is None or self.result is not None):
            raise ValueError("tool action requires only tool")
        if self.action == "result" and (self.result is None or self.tool is not None):
            raise ValueError("result action requires only result")
        return self


class MasterReport(Strict):
    summary: str = Field(min_length=1, max_length=2000)
    limitations: list[str] = Field(max_length=16)
    decisions_needed: list[str] = Field(max_length=16)
