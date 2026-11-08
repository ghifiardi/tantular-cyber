"""Experiment 8 runtime adapters (run-readiness C1–C3). Research-only; not part of the product.

This package holds the parts of an evaluation run that need third-party runtime libraries:

* :mod:`wsp_runtime.vllm_adapter` (C1): an inference engine implementing the harness's
  ``BatchGenerator`` / ``ArmLoader`` interfaces on top of vLLM;
* :mod:`wsp_runtime.grader` (C2): the pinned MATH grading rule on top of ``math-verify``;
* :mod:`wsp_runtime.datasets` (C3): local MATH / MATH-500 JSONL files to harness ``Item`` rows.

It depends on ``tools/wsp`` (the standard-library harness); ``tools/wsp`` never imports it.
vLLM, torch and math-verify are imported lazily, only inside the functions that need them, so
the package can be imported and tested (with stub engines and synthetic strings) where those
libraries are not installed. Nothing here downloads a model or dataset: the adapters accept
local paths only and force the Hugging Face libraries into offline mode.

Status 2026-11-08: tested with stand-in modules only. No model has been loaded and no inference
has run. See ``docs/wsp/runtime.md`` for what still needs a real GPU.
"""

RUNTIME_VERSION = "0.4.0"
