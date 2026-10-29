"""tantular-callgraph: bounded, same-repository interprocedural scanner for Python.

Purpose: catch flows the syntactic Semgrep rules miss — a query/command/path *built* in
one function and used at a sink in another, or wrapped (``text()``, ``" ".join``,
``["sh", "-c", ...]``, ``Path(base) / name``).

Design (conservative by construction):

* Parsing only (``ast.parse``); nothing in the analyzed repository is executed or imported.
* Taint sources: function parameters (callers unknown → low confidence) and well-known
  request inputs (``request.args`` / ``form`` / ``GET`` ..., ``sys.argv``, ``input()``).
* A finding requires a value that was *built* (formatting, concatenation, join, path
  composition) from a source — or raw request input — reaching a sink. Passing a
  parameter straight through to a sink only creates a summary for callers.
* Sanitizers clear taint: ``int``/``float``/``bool``/``len``, ``shlex.quote``,
  ``os.path.basename``, ``secure_filename``, ``html.escape``/``markupsafe.escape``.
  Module-level UPPER_CASE constants are never tainted.
* Calls are followed only when they resolve to a function in this repository (same
  module, explicit import, or ``self.method`` within the defining class — the latter is
  flagged as uncertain). Dynamic dispatch, ambiguous definitions, star imports and
  external calls are **unresolved**: they never produce findings, only counts.
* Budgets: call depth (summary rounds), files, functions, and trace length.

Every finding carries the source-to-sink call path and explicit uncertainty notes, and is
``unverified`` like all scanner findings.
"""

from __future__ import annotations

import ast
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import __version__
from .scanner import RawFinding, ScannerRun

TOOL = "tantular-callgraph"
RULES = {
    "CWE-89": "tantular.callgraph.sqli-interprocedural",
    "CWE-78": "tantular.callgraph.cmdi-interprocedural",
    "CWE-22": "tantular.callgraph.path-interprocedural",
}
_MESSAGES = {
    "CWE-89": "A SQL statement built from {origin} reaches {sink} through the call path shown. "
    "If the value is attacker-controlled this may allow SQL injection; use parameters.",
    "CWE-78": "A command built from {origin} is executed by a shell via {sink} through the "
    "call path shown. If the value is attacker-controlled this may allow OS command "
    "injection; pass an argument list without a shell.",
    "CWE-22": "A file path built from {origin} reaches {sink} through the call path shown. If "
    "the value is attacker-controlled it may escape the intended directory.",
}
_SEVERITY = {"CWE-89": "high", "CWE-78": "high", "CWE-22": "medium"}

_SANITIZERS = {
    "int",
    "float",
    "bool",
    "len",
    "shlex.quote",
    "quote",
    "os.path.basename",
    "basename",
    "secure_filename",
    "werkzeug.utils.secure_filename",
    "html.escape",
    "markupsafe.escape",
    "escape",
}
_STR_PASSTHROUGH = {"str", "text", "sqlalchemy.text", "sqlalchemy.sql.text", "literal_column"}
_PATH_PASSTHROUGH = {
    "Path",
    "pathlib.Path",
    "PurePath",
    "pathlib.PurePath",
    "os.path.join",
    "os.path.abspath",
    "os.path.normpath",
    "os.path.realpath",
    "os.path.expanduser",
}
_METHOD_PASSTHROUGH = {
    "strip",
    "lstrip",
    "rstrip",
    "lower",
    "upper",
    "replace",
    "encode",
    "decode",
    "resolve",
    "absolute",
    "joinpath",
    "with_suffix",
}
_SOURCES = re.compile(
    r"^(flask\.)?request\.(args|form|values|json|data|files|cookies|headers|GET|POST|body)"
    r"|^sys\.argv"
)
_SQL_METHODS = {"execute", "executemany", "executescript"}
_DB_NAME = re.compile(r"^(conn|connection|cur|cursor|db|database|session|engine|tx)\w*$", re.I)
_DB_TYPE = re.compile(r"(Connection|Cursor|Session|Engine)")
_SUBPROCESS = {
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
}
_SHELLS = {"sh", "bash", "zsh", "dash", "/bin/sh", "/bin/bash", "/usr/bin/env"}
_PATH_SINKS = {"open", "io.open", "flask.send_file", "send_file"}
_PATH_METHODS = {"read_text", "read_bytes", "write_text", "write_bytes", "open"}
_CONSTANT = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass(frozen=True)
class CallgraphBudget:
    max_depth: int = 3
    max_files: int = 2000
    max_functions: int = 20000
    max_trace_steps: int = 8


@dataclass(frozen=True)
class Step:
    path: str
    line: int
    function: str
    kind: str  # source | build | call | return | wrapper | sink
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "line": self.line,
            "function": self.function,
            "kind": self.kind,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Taint:
    origin: str  # "param:<name>" or "source:<expr>"
    built: bool
    kind: str  # "str" | "path"
    steps: tuple[Step, ...]
    notes: tuple[str, ...] = ()

    def with_step(self, step: Step, **changes: object) -> Taint:
        return replace(self, steps=(*self.steps, step), **changes)  # type: ignore[arg-type]


@dataclass(frozen=True)
class SinkTemplate:
    cwe: str
    path: str
    line: int
    end_line: int
    sink: str
    steps: tuple[Step, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hit:
    cwe: str
    path: str
    line: int
    end_line: int
    sink: str
    taint: Taint
    tail: tuple[Step, ...]
    notes: tuple[str, ...]


@dataclass
class Function:
    qualname: str
    module: str
    path: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    params: list[str]
    cls: str | None
    annotations: dict[str, str]


@dataclass
class Summary:
    returns: dict[str, list[Taint]] = field(default_factory=dict)  # param name -> taints
    sinks: dict[str, list[SinkTemplate]] = field(default_factory=dict)


@dataclass
class Stats:
    files: int = 0
    functions: int = 0
    resolved_calls: int = 0
    unresolved_calls: int = 0
    self_dispatch: int = 0
    depth_limited: bool = False


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    if isinstance(node, ast.Call):  # e.g. Path(x).read_text -> "Path().read_text"
        inner = _dotted(node.func)
        if inner:
            parts.append(inner + "()")
            return ".".join(reversed(parts))
    return None


def module_name(rel: str) -> str:
    parts = rel[:-3].split("/") if rel.endswith(".py") else rel.split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class _Index:
    """Functions, imports and module constants for the analyzed repository."""

    def __init__(self) -> None:
        self.functions: dict[str, Function] = {}
        self.imports: dict[str, dict[str, str]] = {}  # module -> local name -> target
        self.local_defs: dict[str, dict[str, list[str]]] = {}  # module -> name -> qualnames
        self.star_imports: dict[str, bool] = {}
        self.modules: set[str] = set()

    def add_module(self, rel: str, tree: ast.Module, budget: CallgraphBudget) -> None:
        mod = module_name(rel)
        self.modules.add(mod)
        imports: dict[str, str] = {}
        defs: dict[str, list[str]] = {}
        package = mod if rel.endswith("__init__.py") else mod.rpartition(".")[0]
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        imports[alias.asname] = alias.name
                    else:
                        imports[alias.name.split(".")[0]] = alias.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    pkg_parts = package.split(".") if package else []
                    keep = (
                        pkg_parts[: len(pkg_parts) - (node.level - 1)]
                        if node.level > 1
                        else pkg_parts
                    )
                    base = ".".join([*keep, base] if base else keep)
                for alias in node.names:
                    if alias.name == "*":
                        self.star_imports[mod] = True
                        continue
                    imports[alias.asname or alias.name] = (
                        f"{base}.{alias.name}" if base else alias.name
                    )
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                self._add_function(rel, mod, node, None, defs, budget)
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                        self._add_function(rel, mod, item, node.name, defs, budget)
        self.imports[mod] = imports
        self.local_defs[mod] = defs

    def _add_function(
        self,
        rel: str,
        mod: str,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        cls: str | None,
        defs: dict[str, list[str]],
        budget: CallgraphBudget,
    ) -> None:
        if len(self.functions) >= budget.max_functions:
            return
        name = f"{cls}.{node.name}" if cls else node.name
        qual = f"{mod}.{name}"
        args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        params = [a.arg for a in args]
        annotations = {a.arg: ast.unparse(a.annotation) for a in args if a.annotation}
        if qual in self.functions:  # conditional redefinition: ambiguous
            defs.setdefault(name, []).append(qual)
            return
        self.functions[qual] = Function(qual, mod, rel, node, params, cls, annotations)
        defs.setdefault(name, []).append(qual)

    def resolve(self, fn: Function, callee: str, stats: Stats) -> tuple[Function | None, str]:
        """Resolve a dotted callee name to a repository function, conservatively."""
        mod = fn.module
        defs = self.local_defs.get(mod, {})
        head, _, rest = callee.partition(".")
        if head == "self" and fn.cls and rest and "." not in rest:
            target = defs.get(f"{fn.cls}.{rest}", [])
            if len(target) == 1:
                stats.self_dispatch += 1
                return self.functions[target[0]], (
                    f"`self.{rest}` resolved to `{fn.cls}.{rest}`; subclasses that override it "
                    "were not considered"
                )
            return None, ""
        if not rest:
            local = defs.get(callee, [])
            if len(local) == 1:
                return self.functions.get(local[0]), ""
            if len(local) > 1:
                return None, ""  # ambiguous redefinition
        target = self.imports.get(mod, {}).get(head)
        if target:
            full = f"{target}.{rest}" if rest else target
            if full in self.functions:
                return self.functions[full], ""
        return None, ""


def _scope_statements(node: ast.AST) -> list[ast.stmt]:
    """Statements of a function body in source order, excluding nested defs/classes."""
    out: list[ast.stmt] = []

    def visit(body: list[ast.stmt]) -> None:
        for stmt in body:
            if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            out.append(stmt)
            for name in ("body", "orelse", "finalbody"):
                block = getattr(stmt, name, None)
                if isinstance(block, list):
                    visit(block)
            for handler in getattr(stmt, "handlers", None) or []:
                visit(handler.body)

    visit(getattr(node, "body", []))
    return out


def _own_nodes(stmt: ast.stmt) -> list[ast.AST]:
    """Nodes of a statement excluding nested statement blocks (bodies are visited apart)."""
    nodes: list[ast.AST] = []
    stack: list[ast.AST] = [stmt]
    while stack:
        node = stack.pop()
        nodes.append(node)
        for name, value in ast.iter_fields(node):
            if node is stmt and name in ("body", "orelse", "finalbody", "handlers"):
                continue
            if isinstance(value, list):
                stack.extend(
                    v for v in value if isinstance(v, ast.AST) and not isinstance(v, ast.stmt)
                )
            elif isinstance(value, ast.AST) and not isinstance(value, ast.stmt):
                stack.append(value)
    return nodes


class _FunctionAnalysis:
    def __init__(
        self,
        fn: Function,
        index: _Index,
        summaries: dict[str, Summary],
        budget: CallgraphBudget,
        stats: Stats,
    ) -> None:
        self.fn = fn
        self.index = index
        self.summaries = summaries
        self.budget = budget
        self.stats = stats
        self.env: dict[str, list[Taint]] = {}
        self.db_vars: set[str] = set()
        self.summary = Summary()
        self.hits: list[Hit] = []

    def step(self, node: ast.AST, kind: str, detail: str) -> Step:
        return Step(
            self.fn.path,
            getattr(node, "lineno", self.fn.node.lineno),
            self.fn.qualname.rsplit(".", 2)[-1]
            if self.fn.cls
            else self.fn.qualname.rsplit(".", 1)[-1],
            kind,
            detail,
        )

    def _trim(self, steps: tuple[Step, ...]) -> tuple[Step, ...]:
        limit = self.budget.max_trace_steps
        return steps if len(steps) <= limit else (*steps[: limit - 1], steps[-1])

    # -- expression evaluation ---------------------------------------------------------
    def eval(self, node: ast.AST | None) -> list[Taint]:
        if node is None:
            return []
        if isinstance(node, ast.Name):
            if _CONSTANT.match(node.id):
                return []
            return list(self.env.get(node.id, []))
        if isinstance(node, ast.Constant):
            return []
        if isinstance(node, ast.JoinedStr):
            parts = [
                t
                for v in node.values
                if isinstance(v, ast.FormattedValue)
                for t in self.eval(v.value)
            ]
            return [t.with_step(self.step(node, "build", "f-string"), built=True) for t in parts]
        if isinstance(node, ast.BinOp):
            left, right = self.eval(node.left), self.eval(node.right)
            if isinstance(node.op, ast.Div):
                pathish = any(t.kind == "path" for t in left) or _is_path_ctor(node.left)
                if pathish:
                    return [
                        t.with_step(
                            self.step(node, "build", "path composition with `/`"),
                            built=True,
                            kind="path",
                        )
                        for t in left + right
                    ]
                return left + right
            if isinstance(node.op, ast.Add | ast.Mod):
                label = "concatenation" if isinstance(node.op, ast.Add) else "%-formatting"
                return [
                    t.with_step(self.step(node, "build", label), built=True) for t in left + right
                ]
            return left + right
        if isinstance(node, ast.Call):
            return self.eval_call(node)
        if isinstance(node, ast.Attribute):
            dotted = _dotted(node)
            if dotted and _SOURCES.match(dotted):
                return [self.source(node, dotted)]
            return self.eval(node.value)
        if isinstance(node, ast.Subscript):
            dotted = _dotted(node.value)
            if dotted and _SOURCES.match(dotted):
                return [self.source(node, dotted)]
            return self.eval(node.value)
        if isinstance(node, ast.List | ast.Tuple | ast.Set):
            return [t for e in node.elts for t in self.eval(e)]
        if isinstance(node, ast.IfExp):
            return self.eval(node.body) + self.eval(node.orelse)
        if isinstance(node, ast.BoolOp):
            return [t for v in node.values for t in self.eval(v)]
        if isinstance(node, ast.Starred):
            return self.eval(node.value)
        return []

    def source(self, node: ast.AST, dotted: str) -> Taint:
        return Taint(
            origin=f"source:{dotted}",
            built=False,
            kind="str",
            steps=(self.step(node, "source", f"`{dotted}`"),),
        )

    def eval_call(self, node: ast.Call) -> list[Taint]:
        name = _dotted(node.func) or ""
        last = _method_name(node)
        args = [*node.args, *(k.value for k in node.keywords)]
        if name in _SANITIZERS or (
            last in ("basename", "secure_filename", "quote", "escape")
            and name not in _STR_PASSTHROUGH
        ):
            return []
        if name == "input" or (name and _SOURCES.match(name)):
            return [self.source(node, name)]
        if isinstance(node.func, ast.Attribute):
            receiver = node.func.value
            dotted_recv = _dotted(receiver)
            if dotted_recv and _SOURCES.match(dotted_recv) and last in ("get", "getlist"):
                return [self.source(node, f"{dotted_recv}.{last}()")]
            if last == "format":
                taints = self.eval(receiver) + [t for a in args for t in self.eval(a)]
                return [
                    t.with_step(self.step(node, "build", ".format()"), built=True) for t in taints
                ]
            if last == "join" and not name.startswith("os.path"):
                taints = [t for a in args for t in self.eval(a)]
                return [
                    t.with_step(self.step(node, "build", "str.join()"), built=True) for t in taints
                ]
            if last in _METHOD_PASSTHROUGH:
                return self.eval(receiver) + (
                    [t for a in args for t in self.eval(a)] if last == "joinpath" else []
                )
        if name in _STR_PASSTHROUGH or last == "text":
            return [
                t.with_step(self.step(node, "wrapper", f"`{name}()` wrapper"))
                for a in args
                for t in self.eval(a)
            ]
        if name in _PATH_PASSTHROUGH or last in ("Path", "PurePath"):
            taints = [t for a in args for t in self.eval(a)]
            built = name == "os.path.join" or len(node.args) > 1
            return [
                t.with_step(
                    self.step(node, "build" if built else "wrapper", f"`{name}()`"),
                    kind="path",
                    built=t.built or built,
                )
                for t in taints
            ]
        callee, note = self.index.resolve(self.fn, name, self.stats) if name else (None, "")
        if callee is None:
            if name and name not in _SUBPROCESS and last not in _SQL_METHODS:
                self.stats.unresolved_calls += 1
            return []
        self.stats.resolved_calls += 1
        return self.apply_summary(node, callee, note)

    def bind_args(self, node: ast.Call, callee: Function) -> dict[str, list[Taint]]:
        params = list(callee.params)
        if callee.cls and params and params[0] in ("self", "cls"):
            params = params[1:]
        bound: dict[str, list[Taint]] = {}
        for i, arg in enumerate(node.args):
            if isinstance(arg, ast.Starred):
                break
            if i < len(params):
                bound[params[i]] = self.eval(arg)
        for kw in node.keywords:
            if kw.arg in callee.params:
                bound[kw.arg] = self.eval(kw.value)
        return bound

    def apply_summary(self, node: ast.Call, callee: Function, note: str) -> list[Taint]:
        summary = self.summaries.get(callee.qualname)
        bound = self.bind_args(node, callee)
        call_step = self.step(node, "call", f"calls `{callee.qualname.rsplit('.', 1)[-1]}()`")
        notes = (note,) if note else ()
        result: list[Taint] = []
        if summary is None:
            return result
        for param, templates in summary.returns.items():
            for arg_taint in bound.get(param, []):
                for tmpl in templates:
                    result.append(
                        replace(
                            arg_taint,
                            built=arg_taint.built or tmpl.built,
                            kind="path" if "path" in (arg_taint.kind, tmpl.kind) else "str",
                            steps=self._trim((*arg_taint.steps, call_step, *tmpl.steps[1:])),
                            notes=(*arg_taint.notes, *tmpl.notes, *notes),
                        )
                    )
        for tmpl in summary.returns.get("__source__", []):
            result.append(
                replace(
                    tmpl, steps=self._trim((call_step, *tmpl.steps)), notes=(*tmpl.notes, *notes)
                )
            )
        for param, sinks in summary.sinks.items():
            for arg_taint in bound.get(param, []):
                for sink in sinks:
                    self.record_sink(
                        sink.cwe,
                        None,
                        arg_taint,
                        sink.sink,
                        call_step,
                        template=sink,
                        extra_notes=notes,
                    )
        return result

    # -- sinks -------------------------------------------------------------------------
    def record_sink(
        self,
        cwe: str,
        node: ast.AST | None,
        taint: Taint,
        sink: str,
        via: Step | None = None,
        template: SinkTemplate | None = None,
        extra_notes: tuple[str, ...] = (),
    ) -> None:
        if template is not None:
            path, line, end = template.path, template.line, template.end_line
            tail = (*((via,) if via else ()), *template.steps)
            notes = (*template.notes, *extra_notes)
        elif node is not None:
            path, line = self.fn.path, node.lineno  # type: ignore[attr-defined]
            end = getattr(node, "end_lineno", line) or line
            tail = (self.step(node, "sink", sink),)
            notes = extra_notes
        else:
            return
        dangerous = taint.built or taint.origin.startswith("source:")
        if dangerous:
            self.hits.append(Hit(cwe, path, line, end, sink, taint, tail, (*taint.notes, *notes)))
        elif taint.origin.startswith("param:"):
            param = taint.origin.split(":", 1)[1]
            self.summary.sinks.setdefault(param, []).append(
                SinkTemplate(
                    cwe,
                    path,
                    line,
                    end,
                    sink,
                    self._trim((*taint.steps, *tail)),
                    (*taint.notes, *notes),
                )
            )

    def is_db_receiver(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            if node.id in self.db_vars or _DB_NAME.match(node.id):
                return True
            return bool(_DB_TYPE.search(self.fn.annotations.get(node.id, "")))
        if isinstance(node, ast.Call):  # conn.cursor().execute(...)
            return isinstance(node.func, ast.Attribute) and node.func.attr in ("cursor", "connect")
        if isinstance(node, ast.Attribute):
            return bool(_DB_NAME.match(node.attr))
        return False

    def check_sinks(self, call: ast.Call) -> None:
        name = _dotted(call.func) or ""
        last = _method_name(call)
        if isinstance(call.func, ast.Attribute) and last in _SQL_METHODS and call.args:
            if self.is_db_receiver(call.func.value):
                for t in self.eval(call.args[0]):
                    self.record_sink("CWE-89", call, t, f"`{name}()`")
            return
        if name in _SUBPROCESS and call.args:
            first = call.args[0]
            shell = any(
                k.arg == "shell" and isinstance(k.value, ast.Constant) and k.value.value
                for k in call.keywords
            )
            if shell:
                for t in self.eval(first):
                    self.record_sink("CWE-78", call, t, f"`{name}(..., shell=True)`")
            elif isinstance(first, ast.List | ast.Tuple):
                consts = [e.value if isinstance(e, ast.Constant) else None for e in first.elts]
                if any(c in _SHELLS for c in consts) and "-c" in consts:
                    after = first.elts[consts.index("-c") + 1 :]
                    for elt in after:
                        for t in self.eval(elt):
                            self.record_sink("CWE-78", call, t, f"`{name}([shell, '-c', ...])`")
            return
        if name in ("os.system", "os.popen") and call.args:
            for t in self.eval(call.args[0]):
                self.record_sink("CWE-78", call, t, f"`{name}()`")
            return
        if name in _PATH_SINKS and call.args:
            for t in self.eval(call.args[0]):
                self.record_sink("CWE-22", call, t, f"`{name}()`")
            return
        if isinstance(call.func, ast.Attribute) and last in _PATH_METHODS:
            receiver = self.eval(call.func.value)
            for t in receiver:
                if t.kind == "path":
                    self.record_sink("CWE-22", call, t, f"`Path.{last}()`")

    # -- driver ------------------------------------------------------------------------
    def run(self) -> tuple[Summary, list[Hit]]:
        fn = self.fn
        for p in fn.params:
            if p in ("self", "cls"):
                continue
            self.env[p] = [
                Taint(
                    origin=f"param:{p}",
                    built=False,
                    kind="str",
                    steps=(
                        Step(
                            fn.path,
                            fn.node.lineno,
                            fn.qualname.rsplit(".", 1)[-1],
                            "source",
                            f"parameter `{p}` of `{fn.node.name}`",
                        ),
                    ),
                )
            ]
            if _DB_TYPE.search(fn.annotations.get(p, "")):
                self.db_vars.add(p)
        for stmt in _scope_statements(fn.node):
            calls = sorted(
                (n for n in _own_nodes(stmt) if isinstance(n, ast.Call)),
                key=lambda n: (n.lineno, n.col_offset),
            )
            for call in calls:
                self.check_sinks(call)
            self.update_env(stmt)
        return self.summary, self.hits

    def update_env(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            value = self.eval(stmt.value)
            for target in stmt.targets:
                self.assign(target, value, stmt.value)
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            self.assign(stmt.target, self.eval(stmt.value), stmt.value)
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            old = self.env.get(stmt.target.id, [])
            new = self.eval(stmt.value)
            label = "augmented assignment"
            self.env[stmt.target.id] = [
                t.with_step(self.step(stmt, "build", label), built=True) for t in old + new
            ]
        elif isinstance(stmt, ast.With | ast.AsyncWith):
            for item in stmt.items:
                if item.optional_vars is not None:
                    self.assign(item.optional_vars, self.eval(item.context_expr), item.context_expr)
        elif isinstance(stmt, ast.For | ast.AsyncFor):
            self.assign(stmt.target, self.eval(stmt.iter), stmt.iter)
        elif isinstance(stmt, ast.Expr):
            self.eval(stmt.value)  # bare calls: apply callee sink summaries
        elif isinstance(stmt, ast.If | ast.While | ast.Assert):
            self.eval(stmt.test)
        elif isinstance(stmt, ast.Return) and stmt.value is not None:
            for t in self.eval(stmt.value):
                ret = t.with_step(self.step(stmt, "return", "returned to caller"))
                key = t.origin.split(":", 1)[1] if t.origin.startswith("param:") else "__source__"
                self.summary.returns.setdefault(key, []).append(ret)

    def assign(self, target: ast.AST, value: list[Taint], value_node: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = value
            if isinstance(value_node, ast.Call):
                callee = _dotted(value_node.func) or ""
                if callee.rsplit(".", 1)[-1] in ("connect", "cursor", "Session", "session"):
                    self.db_vars.add(target.id)
        elif isinstance(target, ast.Tuple | ast.List):
            for elt in target.elts:
                self.assign(elt, value, value_node)


def _method_name(call: ast.Call) -> str:
    """Final name of the callee, also for receivers that are not plain names
    (``" ".join(...)``, ``"...".format(...)``, ``(Path(a) / b).read_text()``)."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ""


def _is_path_ctor(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and (_dotted(node.func) or "").rsplit(".", 1)[-1] in (
        "Path",
        "PurePath",
    )


def _dedupe(hits: list[Hit]) -> list[Hit]:
    """One hit per (sink location, CWE); keep the shortest, then lexicographically first."""
    best: dict[tuple[str, int, str], tuple[Hit, int]] = {}
    for hit in hits:
        key = (hit.path, hit.line, hit.cwe)
        steps = len(hit.taint.steps) + len(hit.tail)
        rank = (steps, [(s.path, s.line) for s in hit.taint.steps])
        if key not in best:
            best[key] = (hit, 1)
            continue
        current, count = best[key]
        cur_rank = (
            len(current.taint.steps) + len(current.tail),
            [(s.path, s.line) for s in current.taint.steps],
        )
        best[key] = (hit if rank < cur_rank else current, count + 1)
    out = []
    for hit, count in best.values():
        if count > 1:
            hit = replace(
                hit,
                notes=(
                    *hit.notes,
                    f"{count - 1} other call path(s) reach this sink; the shortest is shown",
                ),
            )
        out.append(hit)
    return sorted(out, key=lambda h: (h.path, h.line, h.cwe))


def analyze(
    sources: dict[str, str], budget: CallgraphBudget | None = None
) -> tuple[list[Hit], Stats, dict[str, str]]:
    """Analyze ``{relative_path: source_text}``. Returns hits, stats and skipped files."""
    budget = budget or CallgraphBudget()
    index = _Index()
    stats = Stats()
    skipped: dict[str, str] = {}
    for rel in sorted(sources):
        if stats.files >= budget.max_files:
            skipped[rel] = "callgraph_file_budget_exceeded"
            continue
        try:
            tree = ast.parse(sources[rel])
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            skipped[rel] = "python_syntax_error"
            continue
        stats.files += 1
        index.add_module(rel, tree, budget)
    stats.functions = len(index.functions)
    functions = [index.functions[q] for q in sorted(index.functions)]

    summaries: dict[str, Summary] = {}
    for _ in range(max(1, budget.max_depth)):
        round_stats = Stats()
        new: dict[str, Summary] = {}
        for fn in functions:
            summary, _ = _FunctionAnalysis(fn, index, summaries, budget, round_stats).run()
            new[fn.qualname] = summary
        if _signature(new) == _signature(summaries):
            break
        summaries = new
    else:
        stats.depth_limited = True

    hits: list[Hit] = []
    final_stats = Stats()
    for fn in functions:
        _, fn_hits = _FunctionAnalysis(fn, index, summaries, budget, final_stats).run()
        hits.extend(fn_hits)
    stats.resolved_calls = final_stats.resolved_calls
    stats.unresolved_calls = final_stats.unresolved_calls
    stats.self_dispatch = final_stats.self_dispatch
    return _dedupe(hits), stats, skipped


def _signature(summaries: dict[str, Summary]) -> list[tuple[str, str, int, int]]:
    return sorted(
        (q, k, len(s.returns.get(k, [])), len(s.sinks.get(k, [])))
        for q, s in summaries.items()
        for k in {*s.returns, *s.sinks}
    )


def _origin_text(taint: Taint) -> str:
    kind, _, value = taint.origin.partition(":")
    return f"parameter `{value}`" if kind == "param" else f"request input {value}"


def hit_to_raw(hit: Hit) -> RawFinding:
    steps = (*hit.taint.steps, *hit.tail)
    notes = list(dict.fromkeys(hit.notes))
    if hit.taint.origin.startswith("param:"):
        notes.insert(
            0, "origin is a function parameter; callers were not proven to pass untrusted data"
        )
    confidence = "medium" if hit.taint.origin.startswith("source:") else "low"
    interprocedural = any(s.kind in ("call", "return") for s in steps)
    if not interprocedural:
        notes.append(
            "flow is within one function (wrapper/composition not matched by the syntactic rules)"
        )
    return RawFinding(
        tool=TOOL,
        rule_id=RULES[hit.cwe],
        path=hit.path,
        start_line=hit.line,
        end_line=hit.end_line,
        message=_MESSAGES[hit.cwe].format(origin=_origin_text(hit.taint), sink=hit.sink),
        severity=_SEVERITY[hit.cwe],
        cwe=(hit.cwe,),
        confidence=confidence,
        trace=tuple(s.as_dict() for s in steps),
        uncertainty=tuple(notes),
    )


class CallgraphScanner:
    """Scanner plugin wrapping :func:`analyze` (pure Python, no subprocess)."""

    name = TOOL

    def __init__(self, budget: CallgraphBudget | None = None, max_file_bytes: int = 1_000_000):
        self.budget = budget or CallgraphBudget()
        self.max_file_bytes = max_file_bytes

    def version(self) -> str | None:
        return __version__

    def scan(self, repo: Path, targets: Sequence[str]) -> ScannerRun:
        started = time.monotonic()
        py = [t for t in targets if t.endswith((".py", ".pyi"))]
        run = ScannerRun(
            tool=self.name,
            version=self.version(),
            rules=sorted(RULES.values()),
            targets=len(targets),
            languages=["python"],
            in_scope=py,
        )
        run.exit_code = 0
        sources: dict[str, str] = {}
        for rel in py:
            sources[rel] = (repo / rel).read_text(encoding="utf-8", errors="replace")
        hits, stats, skipped = analyze(sources, self.budget)
        run.findings = [hit_to_raw(h) for h in hits]
        run.scanned = {rel for rel in py if rel not in skipped}
        run.scanned_reported = True
        run.tool_skipped = skipped
        run.stats = {
            "files": stats.files,
            "functions": stats.functions,
            "resolved_calls": stats.resolved_calls,
            "unresolved_calls": stats.unresolved_calls,
            "self_dispatch": stats.self_dispatch,
            "depth_limited": stats.depth_limited,
            "max_depth": self.budget.max_depth,
        }
        run.duration_s = round(time.monotonic() - started, 3)
        return run
