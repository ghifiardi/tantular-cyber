"""Deterministic evaluation of evidence-context strategies (window vs AST).

For every annotation the scanner detected, a strategy "covers" the construct when its
context includes both the annotation's first line (where the value is built) and its
last line (the sink). Sizes are measured in lines and characters; the budget gate checks
that no context exceeds the configured limits.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ..context import ContextBudget, Strategy, extract_context
from ..pipeline import read_source
from .manifest import Manifest
from .matching import MatchRecord


class StrategyStats(BaseModel):
    strategy: str
    n: int
    construct_covered: int
    coverage_rate: float | None
    mean_lines: float | None
    mean_chars: float | None
    max_chars: int
    max_lines: int
    truncated: int
    fallbacks: dict[str, int]
    over_budget: int


class ContextEval(BaseModel):
    budget: dict[str, int]
    strategies: list[StrategyStats]
    newly_covered: list[str]
    lost_coverage: list[str]


def evaluate_context(
    manifest: Manifest,
    cases_root: Path,
    records: list[MatchRecord],
    budget: ContextBudget | None = None,
) -> ContextEval:
    budget = budget or ContextBudget()
    detected = [r for r in records if r.detected]
    covered_by: dict[str, set[str]] = {}
    stats = []
    sources: dict[str, tuple[str, list[str]]] = {}
    for strategy in ("window", "ast"):
        s: Strategy = strategy  # type: ignore[assignment]
        lines_n, chars_n, covered, truncated, over = [], [], 0, 0, 0
        fallbacks: dict[str, int] = {}
        covered_by[strategy] = set()
        for r in detected:
            if r.path not in sources:
                sources[r.path] = read_source(cases_root / r.path)
            original, lines = sources[r.path]
            # Contexts are cut around the sink line, as the scanner reports it.
            ctx = extract_context(
                original, lines, r.end_line, r.end_line, path=r.path, strategy=s, budget=budget
            )
            shown = set(ctx.line_numbers)
            if r.line in shown and r.end_line in shown:
                covered += 1
                covered_by[strategy].add(r.annotation_id)
            lines_n.append(len(ctx.line_numbers))
            chars_n.append(ctx.chars)
            truncated += ctx.truncated
            over += ctx.chars > budget.max_chars or len(ctx.line_numbers) > budget.max_lines
            if ctx.fallback_reason:
                fallbacks[ctx.fallback_reason] = fallbacks.get(ctx.fallback_reason, 0) + 1
        n = len(detected)
        stats.append(
            StrategyStats(
                strategy=strategy,
                n=n,
                construct_covered=covered,
                coverage_rate=round(covered / n, 4) if n else None,
                mean_lines=round(sum(lines_n) / n, 1) if n else None,
                mean_chars=round(sum(chars_n) / n, 1) if n else None,
                max_chars=max(chars_n, default=0),
                max_lines=max(lines_n, default=0),
                truncated=truncated,
                fallbacks=fallbacks,
                over_budget=over,
            )
        )
    return ContextEval(
        budget={
            "max_lines": budget.max_lines,
            "max_chars": budget.max_chars,
            "window": budget.window,
            "helper_max_lines": budget.helper_max_lines,
        },
        strategies=stats,
        newly_covered=sorted(covered_by["ast"] - covered_by["window"]),
        lost_coverage=sorted(covered_by["window"] - covered_by["ast"]),
    )
