# WSP runtime adapters (`tools/wsp_runtime`): C1–C3

Status 2026-11-08: **implemented and tested with stub engines, synthetic grading strings and
synthetic dataset files only.** No model, dataset or runtime library has been downloaded or
installed, no GPU has been used and no inference has run. Nothing here is a measurement. The
analysis method for repeated samples is unchanged and still awaits the evaluator.

This package holds the parts of an Experiment 8 run that need third-party libraries (vLLM,
torch, math-verify). It depends on the standard-library harness in `tools/wsp`; `tools/wsp`
never imports it (a test checks both directions). The heavy libraries are imported only inside
the functions that need them, so the package imports and its tests run without them (also
tested). `requirements.in` lists the runtime pins; it is installed only in the pinned image on
the isolated host, and no lock exists yet (resolving one needs network access and decisions R6
and R9).

| Item | Module | What it does |
|---|---|---|
| C1 inference adapter | `wsp_runtime/vllm_adapter.py` | `VllmEngine` implements the harness's `BatchGenerator` for one arm; `arm_loader()` is its `ArmLoader` |
| C2 grader adapter | `wsp_runtime/grader.py` | `MathQAGrader`: the pinned grading rule on a `math-verify` backend |
| C3 dataset loader | `wsp_runtime/datasets.py` | Local MATH-500 / MATH-train JSONL to harness `Item` rows with stable IDs, hashes and split checks |
| CLI | `run.py` | Runs a `per_arm_sequential` protocol with one `--checkpoint ARM=DIR` per arm |

## C1: what the engine guarantees (tested with a stub backend)

- **Provenance before loading.** Every file in the checkpoint directory is hashed (symbolic
  links are refused). The registered weights file must match the protocol's
  `weights_sha256_lfs`, and `chat_template.jinja` must match the git blob id in the protocol's
  `chat_template`; otherwise nothing is loaded. Documentation files (`README*`, `LICENSE*`,
  `.gitattributes`) are hashed but reported in `other_files_sha256`, apart from the loaded
  `artifacts_sha256`, so a README difference between repositories is visible but does not look
  like a model difference to the harness's edit-scope check.
- **Pinned engine settings.** The protocol's engine name, version, `dtype`, `max_model_len`,
  `llm_seed`, `gpu_memory_utilization` and `tensor_parallel_size` must be set (a `REPLACE`
  placeholder or a different installed vLLM version is refused). Unknown inference keys are
  refused rather than ignored.
- **Prompt.** The protocol records the prompt in a notation (`<problem>\nPlease reason …`),
  where the two characters `\n` stand for the authors' newline in
  `problem + "\n" + instruction`. Only that exact notation is accepted, and the engine builds
  `problem + newline + instruction`, then applies the checkpoint's chat template with
  `add_generation_prompt=True` and `enable_thinking` from the protocol. *The research lead
  should confirm this reading of the notation (R6/R7).*
- **Load identity and settings.** Each load gets a fresh `load_id`. `describe()` reports the
  hashed files, the engine version, the engine keyword arguments, the sampling settings sent
  with every request, the engine's own default sampling parameters (if the vLLM version exposes
  them), the tokenizer's eos id, the environment (vLLM, torch, CUDA, GPU name) and
  `runtime_code_sha256` (this package's code, which the harness's own code hash does not cover).
- **Request echoes.** Each shard is one `LLM.generate` call. Every vLLM output's prompt must be
  the prompt the engine rendered for the request at that position, or the whole batch fails
  (`BatchMismatch`); outputs are then wrapped in `BatchItem(request, …)`. A request whose arm or
  inference settings differ from the engine's fails only itself. These checks detect adapter
  or engine inconsistencies; they still rely on vLLM attaching outputs to the right prompts.
- **Close.** `close()` (idempotent) asks the backend to shut the engine down, drops the
  engine's references to the vLLM objects, re-hashes the checkpoint directory (a file changed
  during the run fails the close), and, if `--max-gpu-memory-after-close` is given, fails when
  the backend still reports more GPU memory allocated. A failed close aborts the run before the
  next arm loads. Each close report (shutdown steps, memory after close, files unchanged) is
  printed with the run summary.

## C2: grading rule and the points to review (R9)

Rule: cut a stray reasoning block, take the **last** `\boxed{}` or `\fbox{}` (braces matched),
no box = **incorrect** (graded, in the denominator), then `math-verify`. The adapter was written
from the protocol's description of the rule, **not reviewed against the authors'
`exp8_math/code/mathqa.py`** (git blob `808a0477…`). Its revision string says "unreviewed". The
evaluator must check three interpretations against that file before R9:
1. "Cutting a stray reasoning block" = keep the text after the last `</think>`; an unclosed
   `<think>` is searched whole.
2. `\boxed 5` without braces is not recognised.
3. Gold and prediction are wrapped as `$…$` before `parse`, and the 5 s timeouts are passed as
   `parsing_timeout` / `timeout_seconds`.

Grader exceptions (for example a timeout raised as an error) propagate, and the harness records
them as `GraderError` rows.

## C3: dataset loader

- **MATH-500** (`test.jsonl` of `HuggingFaceH4/MATH-500`): IDs are `unique_id`; the split must be
  `heldout`; an expected count, source-file SHA-256 or ID-set hash that differs is an error.
- **MATH training split** (smoke run): the answer is the last `\boxed{}` of the solution, and the
  ID is `math-train-` plus 16 hex digits of the problem's SHA-256 (independent of file order);
  the split must be `development`; `limit` keeps the first rows in file order.
- Every load reports the source file's SHA-256, the ID-set hash and the harness dataset hash.
  `combine()` re-runs the harness's duplicate and development/held-out leakage checks across
  groups. `write_items()` writes the harness JSONL format and never overwrites.
- JSONL only (standard library). A source in another format is converted on the isolated host
  as a recorded operator step.

## What still needs a real GPU (or the real libraries)

None of these can be shown with stub engines. Each must be checked on the isolated host,
recorded, and reviewed before the confirmatory run.

| Check | Why it is open |
|---|---|
| **Engine shutdown and GPU-memory release** | `close()` calls whatever shutdown hooks the installed vLLM exposes (`llm_engine.shutdown`, `parallel_state.destroy_*`, `torch.cuda.empty_cache`), drops references and collects garbage. Whether that frees the GPU before the next arm loads depends on the vLLM version and must be measured (for example `torch.cuda.memory_allocated` and `nvidia-smi` after each close). The stub test shows only that Python references are dropped |
| Which files vLLM actually opens | The adapter hashes every file in the directory it points vLLM at; a file vLLM reads from elsewhere (a cache, a remote repository) would be invisible. Offline mode is set, but this needs checking |
| Effective defaults and stop tokens | The release has no `generation_config.json`; record what `get_default_sampling_params()` and the effective stop/eos set are on the chosen vLLM version, and confirm they include eos `248044` |
| vLLM API details | `LLM(...)` keyword names, `generate(prompts, params_list)` returning outputs in input order with `.prompt`, `finish_reason` values and `token_ids` all follow the vLLM API as documented, and must be confirmed on the pinned version |
| Chat template rendering | Confirm that the real tokenizer's `apply_chat_template(..., enable_thinking=True)` renders the prompt the authors used |
| math-verify 0.9.0 | Confirm the `parse` / `verify` keyword names and timeout behaviour on the installed version, and the C2 interpretations above against `mathqa.py` |
| Throughput, KV-cache headroom, peak memory | Measured in the development smoke run, not estimated here |

## Running (isolated host only; not authorized yet)

```bash
python tools/wsp_runtime/run.py --protocol P.json --items ITEMS.jsonl \
    --checkpoint base=/research/ckpt/base --checkpoint edited=/research/ckpt/edited \
    --checkpoint base_reload=/research/ckpt/base --split development \
    --operator op1 --purpose smoke --output /research/runs/RUN \
    [--keep-outputs] [--max-gpu-memory-after-close BYTES]
```

This needs the runtime image, local checkpoint and dataset copies from an approved fetch, a
`per_arm_sequential` protocol with pinned engine values, and the decisions listed in
[run-readiness.md](run-readiness.md) and in the draft readiness proposal (PR #27): roles,
budget, isolated infrastructure and the licence review. None of these exists yet.
