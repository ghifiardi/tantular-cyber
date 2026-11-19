"""Closed registry: read-only targets, fixed trusted executables, draft-only writes."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field

from tantular_cyber.security import EXCLUDED_DIRS, is_secret_filename, redact

from .schemas import Limits, Strict, ToolRequest

PERMISSIONS = {
    "analyst": {"read_file", "list_files", "search_files"},
    "dataset": {"read_file", "list_files", "search_files", "write_draft", "artifact_index"},
    "validator": {"read_file", "list_files", "run_check", "scan", "validate_dataset"},
    "reporter": {"read_file", "artifact_index", "write_draft"},
}


class FileArgs(Strict):
    root: int = Field(default=0, ge=0)
    path: str = Field(min_length=1, max_length=300)


class ReadArgs(FileArgs):
    offset: int = Field(default=0, ge=0, description="Use next_offset from the previous page.")


class ListArgs(Strict):
    root: int = Field(default=0, ge=0)
    path: str = Field(default=".", max_length=300)


class SearchArgs(ListArgs):
    query: str = Field(min_length=1, max_length=100)


class CheckArgs(Strict):
    preset: Literal["core", "agent-master"]


class DraftArgs(Strict):
    title: str = Field(min_length=1, max_length=200)
    draft: str = Field(min_length=1, max_length=12000)
    sources: list[str] = Field(default_factory=list, max_length=20)


class EmptyArgs(Strict):
    pass


ARGUMENTS = {
    "read_file": ReadArgs,
    "list_files": ListArgs,
    "search_files": SearchArgs,
    "run_check": CheckArgs,
    "scan": ListArgs,
    "validate_dataset": FileArgs,
    "write_draft": DraftArgs,
    "artifact_index": EmptyArgs,
}


class ToolDenied(Exception):
    """No effect performed; caller receives a generic reason only."""


class Registry:
    def __init__(
        self, roots: list[Path], output: Path, limits: Limits, trusted: Path | None = None
    ) -> None:
        self.roots = [p.resolve(strict=True) for p in roots]
        if not self.roots or any(not p.is_dir() for p in self.roots):
            raise ValueError("existing workspace directories required")
        if output.is_symlink():
            raise ValueError("symlink output directory refused")
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.output = output.resolve(strict=True)
        self.limits = limits
        self.trusted = trusted or Path(__file__).resolve().parents[3]

    def policy(self) -> dict:
        return {
            "roots": [str(r) for r in self.roots],
            "output": str(self.output),
            "trusted": str(self.trusted.resolve()),
            "version": 1,
        }

    def validator_script(self) -> Path:
        return self.trusted / "tools" / "rung3" / "dataset.py"

    def available(self, role: str) -> set[str]:
        tools = set(PERMISSIONS[role])
        script = self.validator_script()
        # Offer dataset validation only when the trusted validator ships with this checkout.
        if not script.is_file() or script.is_symlink():
            tools.discard("validate_dataset")
        return tools

    def catalog(self, role: str) -> dict:
        return {name: ARGUMENTS[name].model_json_schema() for name in sorted(self.available(role))}

    def _parts(self, raw: str) -> tuple[str, ...]:
        path = PurePosixPath(raw)
        if path.is_absolute() or ".." in path.parts or "\\" in raw or "\x00" in raw:
            raise ToolDenied("path outside authorized workspace")
        for part in path.parts:
            if part.startswith(".") or part in EXCLUDED_DIRS or is_secret_filename(part):
                raise ToolDenied("private or excluded path")
        return path.parts

    def path(self, root: int, raw: str, *, directory: bool = False) -> Path:
        if root >= len(self.roots):
            raise ToolDenied("unknown workspace root")
        parts = self._parts(raw)
        p = self.roots[root]
        for part in parts:
            p = p / part
            if p.is_symlink():
                raise ToolDenied("symlink paths refused")
        if not p.resolve(strict=True).is_relative_to(self.roots[root]):
            raise ToolDenied("path escape")
        if directory and not p.is_dir():
            raise ToolDenied("directory required")
        return p

    def read(self, root: int, raw: str) -> str:
        self.path(root, raw)
        # Walk with no-follow dir fds so a concurrent symlink swap cannot escape scope.
        parts = self._parts(raw)
        if not parts:
            raise ToolDenied("file required")
        fd = os.open(self.roots[root], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = child
            file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
            with os.fdopen(file_fd, "rb") as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > self.limits.max_file_bytes:
                    raise ToolDenied("oversized or non-regular file")
                raw_bytes = handle.read(self.limits.max_file_bytes + 1)
            if len(raw_bytes) > self.limits.max_file_bytes or b"\0" in raw_bytes:
                raise ToolDenied("binary or oversized file")
            # Always redact the complete file before taking an evidence excerpt.
            return redact(raw_bytes.decode("utf-8"))[0]
        finally:
            os.close(fd)

    def inventory(self, root: int, raw: str) -> list[str]:
        start = self.path(root, raw, directory=True)
        files: list[str] = []
        visited = 0
        for folder, dirs, names in os.walk(start, followlinks=False):
            dirs[:] = sorted(
                d
                for d in dirs
                if not d.startswith(".")
                and d not in EXCLUDED_DIRS
                and not (Path(folder) / d).is_symlink()
                and (Path(folder) / d).resolve() != self.output
            )
            visited += len(dirs) + len(names)
            if visited > self.limits.max_files:
                raise ToolDenied("inventory traversal budget exceeded; choose a subdirectory")
            for name in sorted(names):
                if name.startswith(".") or is_secret_filename(name):
                    continue
                p = Path(folder) / name
                if p.is_symlink() or not p.is_file():
                    continue
                if p.resolve().is_relative_to(self.output):
                    continue
                files.append(p.relative_to(self.roots[root]).as_posix())
        return files

    async def process(self, argv: list[str]) -> int:
        env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "TMPDIR", "SYSTEMROOT"}}
        env.update(
            {
                "PYTHONPATH": str(self.trusted / "src"),
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.trusted,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            async with asyncio.timeout(self.limits.process_timeout):
                return await proc.wait()
        finally:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()

    async def dispatch(self, role: str, request: ToolRequest, effect: str) -> dict:
        if request.name not in PERMISSIONS[role]:
            raise ToolDenied("tool not allowed for this role")
        if request.name not in self.available(role):
            raise ToolDenied("tool not available in this checkout")
        args = ARGUMENTS[request.name].model_validate(request.arguments)
        name = request.name
        artifacts: list[str] = []
        ok = True
        data: dict = {}
        if name == "read_file":
            try:
                text = self.read(args.root, args.path)
            except FileNotFoundError:
                ok = False
                data = {"path": args.path, "error": "file_not_found"}
            else:
                end = args.offset + 8000
                next_offset = end if end < len(text) else None
                data = {
                    "root": args.root,
                    "path": args.path,
                    "offset": args.offset,
                    "text": text[args.offset : end],
                    "truncated": next_offset is not None,
                    "next_offset": next_offset,
                }
        elif name == "list_files":
            files = self.inventory(args.root, args.path)
            data = {"files": files[:100], "total": len(files), "truncated": len(files) > 100}
        elif name == "search_files":
            hits = []
            for path in self.inventory(args.root, args.path):
                try:
                    text = self.read(args.root, path)
                except (ToolDenied, UnicodeError, OSError):
                    continue
                for line, value in enumerate(text.split("\n"), 1):
                    if args.query.casefold() in value.casefold():
                        hits.append({"path": path, "line": line, "excerpt": value[:200]})
                    if len(hits) >= 40:
                        break
                if len(hits) >= 40:
                    break
            data = {"matches": hits, "possibly_truncated": len(hits) >= 40}
        elif name == "artifact_index":
            data = {
                "files": sorted(
                    p.name for p in self.output.iterdir() if p.is_file() and not p.is_symlink()
                )[:100]
            }
        elif name == "write_draft":
            filename = f"{effect}.json"
            # Model data cannot set authorship, review verdicts, or training gates.
            draft = {
                "schema": "tantular/agent-draft@1",
                "authorship": "ai_generated",
                "human_reviewer": "",
                "human_review_verdict": "",
                "license_gate": "pending",
                "leakage_gate": "pending",
                "training_gate": "pending",
                "training_authorized": False,
                "title": args.title,
                "draft": args.draft,
                "sources": args.sources,
            }
            blob = json.dumps(draft, ensure_ascii=True, indent=2).encode()
            if len(blob) > self.limits.max_output_bytes:
                raise ToolDenied("draft output limit exceeded")
            directory_fd = os.open(self.output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                fd = os.open(
                    filename,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory_fd,
                )
                with os.fdopen(fd, "wb") as handle:
                    handle.write(blob)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                os.close(directory_fd)
            artifacts = [filename]
            data = {"artifact": filename, "authorship": "ai_generated", "gates": "pending"}
        elif name == "run_check":
            presets = {
                "core": ["test_analysis.py", "test_ollama_client.py"],
                "agent-master": ["test_agent_master_local.py"],
            }
            paths = [str(self.trusted / "tests" / p) for p in presets[args.preset]]
            if any(not Path(p).is_file() or Path(p).is_symlink() for p in paths):
                raise ToolDenied("registered checks require the trusted development checkout")
            exit_code = await self.process(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-c",
                    "/dev/null",
                    "--confcutdir",
                    str(self.trusted / "tests"),
                    "-q",
                    "--",
                    *paths,
                ]
            )
            ok = exit_code == 0
            data = {"preset": args.preset, "exit_code": exit_code, "passed": ok}
        elif name in {"scan", "validate_dataset"}:
            target = self.path(args.root, args.path, directory=name == "scan")
            output = self.output / effect
            output.mkdir(mode=0o700)  # no overwrite; intent remains uncertain after a crash
            if name == "scan":
                command = [
                    sys.executable,
                    "-m",
                    "tantular_cyber.cli",
                    "scan",
                    "--repo",
                    str(target),
                    "--output",
                    str(output),
                    "--no-llm",
                    "--allow-output-inside-repo",
                ]
            else:
                # Validate a supplied record file; draft creation never invokes training.
                self.read(args.root, args.path)  # enforce size, regular-file and secret policy
                command = [
                    sys.executable,
                    str(self.validator_script()),
                    "--manifest",
                    str(output / "manifest.json"),
                    "--",
                    str(target),
                ]
            exit_code = await self.process(command)
            ok = exit_code == 0
            artifacts = [
                f"{effect}/{p.name}" for p in output.iterdir() if p.is_file() and not p.is_symlink()
            ]
            data = {"exit_code": exit_code, "ai_enabled": False, "passed": ok}
            if name == "scan" and (output / "report.json").is_file():
                report = json.loads((output / "report.json").read_text())
                data.update(
                    {
                        "scanner_status": report.get("status"),
                        "findings": len(report.get("findings", [])),
                        "authority": "deterministic scanner; model advisory only",
                    }
                )
        else:
            raise ToolDenied("unsupported tool")
        # Artifact paths are output-relative, never arbitrary model-provided paths.
        return {
            "evidence_id": effect,
            "ok": ok,
            "tool": name,
            "artifact_paths": artifacts,
            "data": data,
        }
