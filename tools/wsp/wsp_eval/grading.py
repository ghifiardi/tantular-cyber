"""Graders. A grader returns correct True/False, or None when the output cannot be graded.

``None`` is an explicit "ungraded" outcome counted in every summary; it is never treated as
wrong or dropped. The real benchmark graders must come from the authors' released evaluator
(docs/wsp/week-1-execution-package.md); :class:`FinalAnswerGrader` is a simple synthetic
grader for harness tests and does not reproduce any published grading procedure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .dataset import Item


@dataclass(frozen=True)
class Grade:
    correct: bool | None
    detail: str


class Grader(Protocol):
    id: str
    revision: str

    def grade(self, item: Item, text: str) -> Grade: ...


_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER = re.compile(r"(?im)^\s*answer\s*:\s*(.+?)\s*$")


def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", s).rstrip(".").lower()


class FinalAnswerGrader:
    """Last ``\\boxed{...}`` or ``Answer: ...`` line, whitespace/case-insensitive match."""

    id = "wsp-synthetic-final-answer"
    revision = "1"

    def grade(self, item: Item, text: str) -> Grade:
        candidates = _BOXED.findall(text) or _ANSWER.findall(text)
        if not candidates:
            return Grade(None, "no final answer found")
        got = candidates[-1]
        ok = _normalize(got) == _normalize(item.answer)
        return Grade(ok, "match" if ok else "mismatch")
