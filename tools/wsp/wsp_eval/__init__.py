"""Research-only evaluation harness for the Weight-Space Programming (WSP) track.

Scope (docs/wsp/week-1-execution-package.md, docs/wsp/harness.md):

* Independent of the production scanner: standard library only, never imports
  ``tantular_cyber``, never shipped in the wheel.
* No network, no subprocess, no model loading. Model access goes through the
  provider-neutral :class:`wsp_eval.generation.Generator` protocol; the only implementation
  in this repository is a deterministic synthetic generator for tests.
* No promotion path. Analysis reports whether preregistered criteria are met; a checkpoint
  decision is a human record (:mod:`wsp_eval.ledger`), never an action taken by this code.
"""

from __future__ import annotations

HARNESS_VERSION = "0.1.0"
