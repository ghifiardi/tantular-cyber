"""C2: the pinned Experiment 8 grading rule, as a harness ``Grader``.

Rule (protocol ``grader.rule``): take the response after cutting a stray reasoning block; use
the **last** ``\\boxed{...}`` or ``\\fbox{...}``; no box means **incorrect** (not ungraded); then
``math_verify.verify(parse(gold), parse(pred))`` with 5 s timeouts on the main thread.

This is a reimplementation from that written rule, **unreviewed**: it has not been compared
with the authors' released ``exp8_math/code/mathqa.py`` (git blob
``808a04776906b1cf4630d397cef54f03d8795477``), and it stays unreviewed until it has been and the
independent evaluator has accepted it (R9). Until then :data:`REVIEWED` is False and the
runtime CLI refuses held-out execution. Each point below is an interpretation the evaluator
must confirm or change:

1. :func:`strip_reasoning`: keep only the text after the last ``</think>``; a response with an
   unclosed ``<think>`` is searched whole.
2. ``</think>`` is assumed to survive detokenisation. If it is a special token and vLLM skips
   special tokens (its default), nothing is cut and boxes inside the reasoning count.
3. :func:`last_boxed`: only braced boxes count; ``\\boxed 5`` without braces is not recognised.
4. ``\\boxed`` and ``\\fbox`` are treated alike, and whichever occurs last wins (some reference
   implementations prefer ``\\boxed`` and use ``\\fbox`` only when there is no ``\\boxed``).
5. If the last box command is not a complete braced group (unbalanced, or unbraced), the
   response has no answer; there is no fallback to an earlier box.
6. Whitespace between the command and its brace is accepted (``\\boxed {4}``).
7. Braces are matched literally; ``\\{`` and ``\\}`` count as braces (balanced escaped braces
   therefore work).
8. An empty box (``\\boxed{}``) is incorrect without calling math-verify.
9. :class:`MathVerifyBackend`: gold and prediction are wrapped as ``$...$`` before ``parse``,
   with ``parsing_timeout`` / ``timeout_seconds`` of 5 s.
10. Timeouts and math-verify errors are **grader errors** (``raise_on_error=True``; the
    harness records a ``GraderError`` row), never ordinary incorrect answers. math-verify's
    own default is to return an empty parse or ``False`` on a timeout, which would count as
    incorrect; adopting that needs an explicit review decision. Unverified: stub exceptions
    only show the propagation; whether 0.9.0 raises on a timeout with ``raise_on_error=True``
    must be checked against the real library.
11. A gold answer that does not parse is a grader error; a prediction that parses to nothing
    is incorrect.
12. math-verify's timeouts use signals, so grading must run on the main thread; the backend
    refuses any other thread.

The keyword names (``parsing_timeout``, ``timeout_seconds``, ``raise_on_error``) are those of the
math-verify API as documented, not checked against 0.9.0 (no library is installed here). If
0.9.0 does not accept one of them, every grade fails loudly as a grader error.

``math_verify`` is imported only by :meth:`MathVerifyBackend.load`. Tests use a stub backend
and synthetic strings only, never MATH-500 content.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Protocol

from wsp_eval.dataset import Item
from wsp_eval.grading import Grade

GRADER_ID = "exp8-mathqa-math-verify"
# Entered in the protocol's grader.revision; "unreviewed" until R9 is complete.
GRADER_REVISION = "wsp-runtime-0.2 (unreviewed adapter of mathqa.py blob 808a0477)"
# Set to True only after the comparison with mathqa.py and the independent evaluator's
# acceptance (R9). The runtime CLI refuses held-out execution while this is False.
REVIEWED = False
TIMEOUT_S = 5

_BOX_COMMANDS = ("\\boxed", "\\fbox")


class GraderInputError(ValueError):
    """The gold answer cannot be graded (for example it does not parse)."""


def strip_reasoning(text: str) -> str:
    """Keep only the text after the last ``</think>`` (interpretation 1)."""
    marker = "</think>"
    return text.rsplit(marker, 1)[1] if marker in text else text


def last_boxed(text: str) -> str | None:
    """Content of the last box command if it is a complete braced group, else None
    (interpretations 3-7)."""
    starts = [(text.rfind(c), c) for c in _BOX_COMMANDS]
    i, command = max(starts)
    if i == -1:
        return None
    j = i + len(command)
    while j < len(text) and text[j] in " \t":
        j += 1
    if j >= len(text) or text[j] != "{":
        return None  # the last box is unbraced: no answer, no fallback to an earlier box
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1 : k]
    return None  # unbalanced: no answer, no fallback


class VerifyBackend(Protocol):
    def equivalent(self, gold: str, pred: str) -> bool: ...


@dataclass
class MathVerifyBackend:
    """math-verify, imported lazily (interpretations 9-12)."""

    module: Any
    timeout_s: int = TIMEOUT_S

    @classmethod
    def load(cls) -> MathVerifyBackend:
        import math_verify

        return cls(module=math_verify)

    def equivalent(self, gold: str, pred: str) -> bool:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("math-verify timeouts need the main thread; grade there")
        parse, verify = self.module.parse, self.module.verify
        t = self.timeout_s
        g = parse(f"${gold}$", parsing_timeout=t, raise_on_error=True)
        if not g:
            raise GraderInputError("the gold answer did not parse")
        p = parse(f"${pred}$", parsing_timeout=t, raise_on_error=True)
        if not p:
            return False
        return bool(verify(g, p, timeout_seconds=t, raise_on_error=True))


def check_dependencies(
    pinned: Mapping[str, Any] | None, lookup: Callable[[str], str] = metadata.version
) -> dict[str, dict[str, Any]]:
    """Installed versions of the protocol's grader dependencies. A pinned version that differs
    (or a pinned package that is not installed) is refused; a ``REPLACE`` pin is recorded as
    unpinned."""
    out: dict[str, dict[str, Any]] = {}
    for name, want in sorted((pinned or {}).items()):
        try:
            have: str | None = lookup(name)
        except metadata.PackageNotFoundError:
            have = None
        unpinned = not isinstance(want, str) or want.startswith("REPLACE")
        if not unpinned and have != want:
            raise ValueError(f"grader dependency {name} is {have!r}, the protocol pins {want!r}")
        out[name] = {"pinned": None if unpinned else want, "installed": have}
    return out


class MathQAGrader:
    """Harness ``Grader``: ``id`` and ``revision`` must match the protocol's ``grader`` block."""

    id = GRADER_ID
    revision = GRADER_REVISION
    reviewed = REVIEWED

    def __init__(self, backend: VerifyBackend):
        self.backend = backend

    def grade(self, item: Item, text: str) -> Grade:
        pred = last_boxed(strip_reasoning(text))
        if pred is None:
            # The authors score an unboxed response as wrong, so it is graded (False), not
            # ungraded: it counts in the accuracy denominator.
            return Grade(False, "no boxed answer")
        if not pred.strip():
            return Grade(False, "empty boxed answer")
        # Exceptions (timeouts, math-verify errors, an unparseable gold answer) propagate: the
        # runner records them as GraderError rows rather than guessing a grade.
        ok = self.backend.equivalent(item.answer, pred)
        return Grade(bool(ok), "math-verify equivalent" if ok else "math-verify not equivalent")
