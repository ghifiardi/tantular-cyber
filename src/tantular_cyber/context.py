"""Bounded, AST-aware evidence context for findings.

Structure comes from ``ast.parse`` of the original file (parsing only — nothing is
executed or imported); every displayed line comes from the *redacted* text, so secrets
never reach a prompt or report. Redaction preserves line counts; if the counts ever
differ, or the file is not Python, or parsing fails, extraction falls back to the fixed
line window used before POC-004.

The context is explanatory: it shows where values used at the finding are defined, but
the "dataflow hints" are syntactic and do not prove that untrusted data reaches the sink.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Literal

Strategy = Literal["ast", "window"]
ContextKind = Literal["ast_function", "ast_module", "window"]
GAP = "…"


@dataclass(frozen=True)
class ContextBudget:
    max_lines: int = 60
    max_chars: int = 4000
    window: int = 3
    helper_max_lines: int = 15


@dataclass
class ExtractedContext:
    text: str
    kind: ContextKind
    line_numbers: list[int]
    hints: list[str] = field(default_factory=list)
    truncated: bool = False
    fallback_reason: str | None = None

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def approx_tokens(self) -> int:
        """Rough size estimate (chars / 4); budgets are enforced on chars."""
        return (len(self.text) + 3) // 4


def _render(lines: list[str], numbers: list[int]) -> str:
    if not numbers:
        return ""
    width = len(str(max(numbers)))
    out: list[str] = []
    prev = None
    for n in numbers:
        if prev is not None and n != prev + 1:
            out.append(f"{' ' * width}   {GAP}")
        out.append(f"{n:>{width}} | {lines[n - 1]}")
        prev = n
    return "\n".join(out)


def window_context(
    lines: list[str], start: int, end: int, budget: ContextBudget, reason: str | None = None
) -> ExtractedContext:
    lo, hi = max(1, start - budget.window), min(len(lines), end + budget.window)
    numbers = list(range(lo, hi + 1))
    ctx = ExtractedContext(text="", kind="window", line_numbers=numbers, fallback_reason=reason)
    return _fit(ctx, lines, numbers, keep=set(range(start, end + 1)), budget=budget)


def _fit(
    ctx: ExtractedContext,
    lines: list[str],
    numbers: list[int],
    keep: set[int],
    budget: ContextBudget,
    center: int | None = None,
) -> ExtractedContext:
    """Drop the lines farthest from the finding until line and char budgets hold."""
    numbers = sorted(set(numbers))
    center = center if center is not None else (min(keep) if keep else numbers[0])
    removable = sorted((n for n in numbers if n not in keep), key=lambda n: -abs(n - center))
    text = _render(lines, numbers)
    while (len(numbers) > budget.max_lines or len(text) > budget.max_chars) and removable:
        numbers.remove(removable.pop(0))
        ctx.truncated = True
        text = _render(lines, numbers)
    # Still over budget: drop kept lines farthest from the finding (never the finding line).
    extra = sorted((n for n in numbers if n != center), key=lambda n: -abs(n - center))
    while (len(numbers) > budget.max_lines or len(text) > budget.max_chars) and extra:
        numbers.remove(extra.pop(0))
        ctx.truncated = True
        text = _render(lines, numbers)
    if len(text) > budget.max_chars:  # the finding itself is huge: hard cut
        text = text[: budget.max_chars - 1] + GAP
        ctx.truncated = True
    ctx.text, ctx.line_numbers = text, numbers
    return ctx


def _span(node: ast.AST) -> tuple[int, int]:
    first = node.lineno  # type: ignore[attr-defined]
    for dec in getattr(node, "decorator_list", []):
        first = min(first, dec.lineno)
    return first, node.end_lineno or node.lineno  # type: ignore[attr-defined]


_FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)


def _enclosing_function(tree: ast.Module, start: int, end: int) -> ast.AST | None:
    best = None
    for node in ast.walk(tree):
        if isinstance(node, _FUNC):
            lo, hi = _span(node)
            if lo <= start and end <= hi and (best is None or lo >= _span(best)[0]):
                best = node
    return best


def _child_blocks(stmt: ast.stmt) -> list[ast.stmt]:
    blocks: list[ast.stmt] = []
    for name in ("body", "orelse", "finalbody"):
        value = getattr(stmt, name, None)
        if isinstance(value, list):
            blocks += [s for s in value if isinstance(s, ast.stmt)]
    for handler in getattr(stmt, "handlers", None) or []:
        blocks += handler.body
    for case in getattr(stmt, "cases", None) or []:
        blocks += case.body
    return blocks


def _innermost_statements(root: ast.AST, start: int, end: int) -> list[tuple[ast.stmt, int, int]]:
    """Deepest statements overlapping [start, end], with the line range to show.

    For a compound statement whose children do not overlap the finding (e.g. a finding on
    a ``with``/``if`` header), only the header lines are used.
    """
    found: list[tuple[ast.stmt, int, int]] = []

    def overlaps(s: ast.stmt) -> bool:
        return s.lineno <= end and (s.end_lineno or s.lineno) >= start

    def visit(body: list[ast.stmt]) -> None:
        for stmt in body:
            if not overlaps(stmt):
                continue
            children = _child_blocks(stmt)
            inner = [c for c in children if overlaps(c)]
            if inner:
                visit(children)
            elif children:
                found.append((stmt, stmt.lineno, max(stmt.lineno, children[0].lineno - 1)))
            else:
                found.append((stmt, stmt.lineno, stmt.end_lineno or stmt.lineno))

    visit(getattr(root, "body", []))
    return found


def _loaded_names(nodes: list[ast.AST]) -> set[str]:
    return {
        n.id
        for node in nodes
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _bound_names(stmt: ast.stmt) -> set[str]:
    targets: list[ast.AST] = []
    if isinstance(stmt, ast.Assign):
        targets = list(stmt.targets)
    elif isinstance(stmt, ast.AugAssign | ast.AnnAssign | ast.For | ast.AsyncFor):
        targets = [stmt.target]
    elif isinstance(stmt, ast.With | ast.AsyncWith):
        targets = [i.optional_vars for i in stmt.items if i.optional_vars is not None]
    return {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}


def _shown_range(stmt: ast.stmt) -> range:
    """Full range for simple statements; header lines only for compound ones."""
    children = _child_blocks(stmt)
    if children:
        return range(stmt.lineno, max(stmt.lineno, children[0].lineno - 1) + 1)
    return range(stmt.lineno, (stmt.end_lineno or stmt.lineno) + 1)


def _all_statements(func: ast.AST) -> list[ast.stmt]:
    return [n for n in ast.walk(func) if isinstance(n, ast.stmt) and n is not func]


def extract_context(
    original_text: str,
    redacted_lines: list[str],
    start: int,
    end: int,
    *,
    path: str,
    strategy: Strategy = "ast",
    budget: ContextBudget | None = None,
) -> ExtractedContext:
    budget = budget or ContextBudget()
    start, end = max(1, start), max(start, min(end, len(redacted_lines)))
    if strategy == "window":
        return window_context(redacted_lines, start, end, budget)
    if not path.endswith((".py", ".pyi")):
        return window_context(redacted_lines, start, end, budget, "not a Python file")
    original_lines = original_text.split("\n")
    if original_lines and original_lines[-1] == "":
        original_lines.pop()
    if len(original_lines) != len(redacted_lines):
        return window_context(redacted_lines, start, end, budget, "line count mismatch")
    try:
        tree = ast.parse(original_text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return window_context(redacted_lines, start, end, budget, "syntax error")

    func = _enclosing_function(tree, start, end)
    scope: ast.AST = func if func is not None else tree
    found = _innermost_statements(scope, start, end)
    stmts = [stmt for stmt, _, _ in found]
    keep: set[int] = set(range(start, end + 1))
    for _, lo_s, hi_s in found:
        keep.update(range(lo_s, hi_s + 1))
    names = _loaded_names(list(stmts))
    hints: list[str] = []
    def_stmts: list[ast.stmt] = []
    numbers: set[int] = set(keep)

    imported = {
        (alias.asname or alias.name).split(".")[0]
        for n in tree.body
        if isinstance(n, ast.Import | ast.ImportFrom)
        for alias in n.names
    }
    module_funcs = {n.name: n for n in tree.body if isinstance(n, _FUNC)}
    if func is not None:
        lo, hi = _span(func)
        header_end = func.body[0].lineno - 1 if func.body else func.lineno
        header = set(range(lo, max(lo, header_end) + 1))
        keep |= header
        params = {a.arg for a in ast.walk(func.args) if isinstance(a, ast.arg)}
        defs: dict[str, list[ast.stmt]] = {}
        for stmt in _all_statements(func):
            if stmt.lineno < start or stmt in stmts:
                for name in _bound_names(stmt):
                    defs.setdefault(name, []).append(stmt)
        relevant = set(names)
        for name in list(names):  # one transitive step: names used to build those values
            for stmt in defs.get(name, []):
                relevant |= _loaded_names([stmt.value]) if hasattr(stmt, "value") else set()
        for name in sorted(relevant):
            if name in defs:
                for stmt in defs[name]:
                    keep.update(_shown_range(stmt))
                    if stmt not in stmts:
                        def_stmts.append(stmt)
                hints.append(
                    f"`{name}` assigned at line(s) {', '.join(str(s.lineno) for s in defs[name])}"
                )
            elif name in params:
                hints.append(f"`{name}` is a parameter of `{func.name}` (line {func.lineno})")
        if hi - lo + 1 <= budget.max_lines:
            numbers = set(range(lo, hi + 1))
        else:
            numbers = set(keep)
            for n in range(start - 2, end + 3):
                if lo <= n <= hi:
                    numbers.add(n)
        kind: ContextKind = "ast_function"
        center = start
    else:
        numbers = set(keep)
        for n in range(start - budget.window, end + budget.window + 1):
            if 1 <= n <= len(redacted_lines):
                numbers.add(n)
        kind = "ast_module"
        center = start

    # One level of same-module helpers called at the finding (e.g. a query builder).
    called = {
        n.func.id
        for stmt in [*stmts, *def_stmts]
        for n in ast.walk(stmt)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.lineno in _shown_range(stmt)
    }
    for name in sorted(called):
        helper = module_funcs.get(name)
        if helper is None or helper is func:
            continue
        h_lo, h_hi = _span(helper)
        shown = list(range(h_lo, min(h_hi, h_lo + budget.helper_max_lines - 1) + 1))
        numbers.update(shown)
        clipped = " (clipped)" if h_hi > shown[-1] else ""
        hints.append(f"`{name}()` is defined in this file at lines {h_lo}-{h_hi}{clipped}")

    for name in sorted(names - {h.split("`")[1] for h in hints if h.startswith("`")}):
        if (
            func is not None
            and name not in module_funcs
            and name not in imported
            and not name[:1].isupper()
        ):
            hints.append(f"`{name}` is not defined in this function (module/global or builtin)")
    hints = hints[:12]

    ctx = ExtractedContext(text="", kind=kind, line_numbers=[], hints=hints)
    return _fit(
        ctx,
        redacted_lines,
        sorted(numbers),
        keep=keep & numbers or set(range(start, end + 1)),
        budget=budget,
        center=center,
    )
