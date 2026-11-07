"""C2: the pinned Experiment 8 grading rule, as a harness ``Grader``.

Rule (protocol ``grader.rule``): take the response after cutting a stray reasoning block; use
the **last** ``\\boxed{...}`` or ``\\fbox{...}``; no box means **incorrect** (not ungraded); then
``math_verify.verify(parse(gold), parse(pred))`` with 5 s timeouts on the main thread.

This is a reimplementation from that written rule, **not yet reviewed against the authors'
``exp8_math/code/mathqa.py``** (git blob ``808a04776906b1cf4630d397cef54f03d8795477``). Three
points are interpretations that the evaluator must confirm against that file (R9) before the
grader's revision is entered in a protocol; each is isolated in one function:

1. :func:`strip_reasoning`: "cutting a stray reasoning block" is read as keeping only the text
   after the last ``</think>``; a response with an unclosed ``<think>`` is searched whole.
2. :func:`last_boxed`: braces are matched (so ``\\boxed{\\frac{1}{2}}`` yields ``\\frac{1}{2}``);
   ``\\boxed 5`` without braces is not recognised.
3. :class:`MathVerifyBackend`: gold and prediction are wrapped as ``$...$`` before
   ``parse``, and the timeouts are passed as math-verify's ``parsing_timeout`` /
   ``timeout_seconds``. Both must be checked against mathqa.py and the installed version.

``math_verify`` is imported only by :meth:`MathVerifyBackend.load`. Tests use a stub backend
and synthetic strings only, never MATH-500 content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from wsp_eval.dataset import Item
from wsp_eval.grading import Grade

GRADER_ID = "exp8-mathqa-math-verify"
# Entered in the protocol's grader.revision; "unreviewed" until R9 is complete.
GRADER_REVISION = "wsp-runtime-0.1 (unreviewed adapter of mathqa.py blob 808a0477)"
TIMEOUT_S = 5

_BOX_COMMANDS = ("\\boxed", "\\fbox")


def strip_reasoning(text: str) -> str:
    """Keep only the text after the last ``</think>`` (interpretation 1 above)."""
    marker = "</think>"
    return text.rsplit(marker, 1)[1] if marker in text else text


def last_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` / ``\\fbox{...}``, with balanced braces, or None."""
    best: tuple[int, str] | None = None
    for command in _BOX_COMMANDS:
        start = 0
        while (i := text.find(command, start)) != -1:
            start = i + len(command)
            j = start
            while j < len(text) and text[j] in " \t":
                j += 1
            if j >= len(text) or text[j] != "{":
                continue
            depth, k = 0, j
            while k < len(text):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            if depth != 0:
                continue  # unbalanced: not a complete box
            if best is None or i > best[0]:
                best = (i, text[j + 1 : k])
    return None if best is None else best[1]


class VerifyBackend(Protocol):
    def equivalent(self, gold: str, pred: str) -> bool: ...


@dataclass
class MathVerifyBackend:
    """math-verify, imported lazily (interpretation 3 above)."""

    module: Any
    timeout_s: int = TIMEOUT_S

    @classmethod
    def load(cls) -> MathVerifyBackend:
        import math_verify

        return cls(module=math_verify)

    def equivalent(self, gold: str, pred: str) -> bool:
        parse, verify = self.module.parse, self.module.verify
        g = parse(f"${gold}$", parsing_timeout=self.timeout_s)
        p = parse(f"${pred}$", parsing_timeout=self.timeout_s)
        return bool(verify(g, p, timeout_seconds=self.timeout_s))


class MathQAGrader:
    """Harness ``Grader``: ``id`` and ``revision`` must match the protocol's ``grader`` block."""

    id = GRADER_ID
    revision = GRADER_REVISION

    def __init__(self, backend: VerifyBackend):
        self.backend = backend

    def grade(self, item: Item, text: str) -> Grade:
        pred = last_boxed(strip_reasoning(text))
        if pred is None:
            # The authors score an unboxed response as wrong, so it is graded (False), not
            # ungraded: it counts in the accuracy denominator.
            return Grade(False, "no boxed answer")
        # Exceptions (including timeouts surfaced as errors) propagate: the runner records them
        # as GraderError rows rather than guessing a grade.
        ok = self.backend.equivalent(item.answer, pred)
        return Grade(bool(ok), "math-verify equivalent" if ok else "math-verify not equivalent")
