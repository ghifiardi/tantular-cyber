# WSP runtime adapters (`tools/wsp_runtime`): C1–C3

Status 2026-11-08: **implemented and tested with stand-in vLLM and torch modules, synthetic
grading strings and synthetic dataset files only.** No model, dataset or runtime library has
been downloaded or installed, no GPU has been used and no inference has run. Nothing here is a
measurement. The analysis method for repeated samples is unchanged and still awaits the
evaluator.

The test backend is a subclass of the real `VllmBackend`, so its shutdown steps, settings
read-back and memory queries run the real implementation against stand-in modules. **These
tests validate the adapters' control flow.** The stand-ins follow the adapters' own reading of
the vLLM and math-verify APIs, so the tests cannot show compatibility with the real libraries,
and the storage tests present the test directory's parents as root-owned, so they cannot show
storage isolation on a real host.

This package holds the parts of an Experiment 8 run that need third-party libraries (vLLM,
transformers, torch, math-verify). It depends on the standard-library harness in `tools/wsp`;
`tools/wsp` never imports it (a test checks both directions). The heavy libraries are imported
only inside the functions that need them, so the package imports and its tests run without them
(also tested). `requirements.in` lists the runtime pins; it is installed only in the pinned image
on the isolated host.

| Item | Module | What it does |
|---|---|---|
| C1 inference adapter | `wsp_runtime/vllm_adapter.py` | `VllmEngine` implements the harness's `BatchGenerator` for one arm; `arm_loader()` is its `ArmLoader`; `preflight()` runs every check that needs no GPU or loaded engine |
| C2 grader adapter | `wsp_runtime/grader.py` | `MathQAGrader`: the pinned grading rule on a `math-verify` backend (**unreviewed**) |
| C3 dataset loader | `wsp_runtime/datasets.py` | Local MATH-500 / MATH-train JSONL to harness `Item` rows, bound to the protocol's `dataset.runtime_sources` registration, with split and leakage checks |
| Worker processes | `wsp_runtime/process_worker.py` | One `spawn` worker process per arm, each in its own verified process group (`WorkerEngine`, a `BatchGenerator` proxy), and a short-lived probe worker; the parent never imports vLLM, transformers or torch (it does load math-verify for grading) |
| CLI | `run.py` | Checks the environment, dataset, leakage reference, versions, sampling read-back and every arm before anything runs, then runs a `per_arm_sequential` protocol with one `--checkpoint ARM=DIR` per arm |

## Real-run blockers

None of these can be resolved in this repository. Each must be done, recorded and reviewed
before a real run:

0. **Per-arm worker processes: implemented with stub workers only.** As the readiness proposal
   (PR #27, §"Sequential loading") specifies, each arm now runs in its own worker process (see
   "One worker process per arm" below). It has been exercised only with stub engines and stub
   workers. Still open for a real run: the real engine in a worker on the GPU host, and the
   fact that **a worker that exited, with an empty process group, does not by itself show that
   every GPU resource was released**: a descendant that leaves the group is not seen, and GPU
   release stays unverified (items 4 and the GPU-host table).
1. **No runtime lock.** `requirements.in` has no lock file; vLLM and transformers are
   `REPLACE` (R6) and sympy is `REPLACE` (R9). Resolving a lock needs network access and those
   decisions.
2. **Pinned-library compatibility.** Every vLLM, transformers and math-verify call is written
   to the documented API, not checked on the pinned versions (table below). This includes two
   version-dependent assumptions: the telemetry opt-out variables and the engine shutdown hook
   names.
3. **Effective engine settings.** The engine arguments the adapter does not set
   (`describe()["requested"]["engine_args_left_at_vllm_defaults"]`), the effective model
   configuration and the stop set applied at generation time are vLLM's choice; their effective
   values must be read back and recorded on the GPU host.
4. **Worker-process exit and device-level GPU memory.** The adapter cannot show that vLLM's
   workers exited or that the GPU was released (see C1, close). The timing of the engine
   process's exit after shutdown is version-dependent and must be recorded too.
5. **Storage verified by the operator.** The local storage checks do not establish everything
   (see C1, storage); the operator must verify and record the points they cannot, including
   privileged remounts, overlays and writable aliases.
6. **Grader review (R9).** The grader stays unreviewed until it has been compared with the
   released `mathqa.py` and the independent evaluator has accepted it. Until then held-out
   execution is refused. Its timeout behaviour on the real library is unverified.
7. **Protocol registration.** The Experiment 8 draft has no `execution` block,
   `dataset.runtime_sources`, `checkpoint.loadable_files_sha256`, `engine.generation_config` or
   pinned `engine.transformers_version`; it is refused until they are recorded (after the
   approved fetch and R6), and is intentionally unchanged until then.
8. The roles, budget, isolated compute account and licence review ([run-readiness.md](run-readiness.md)).

## One worker process per arm (`process_worker.py`, tested with stub workers)

The command line always runs each arm in its own **worker process**, started with the `spawn`
start method (a fresh interpreter), in protocol order. The parent (the CLI and the harness
runner) never imports vLLM, transformers or torch; it talks to the worker through
`WorkerEngine`, which implements the harness's `BatchGenerator`. The parent **does** still load
math-verify (the grader's backend, `MathVerifyBackend.load()`), because grading runs in the
parent. (`run()` also has an `isolation="in-process"` seam that builds the engines in the parent;
it exists for the adapter tests only, and no command-line option selects it.)

- **Start and process group.** Every worker (and the probe) first makes itself the leader of a
  new process group and reports that it has started. The parent checks, while the worker is
  alive and waiting, that the worker's process group is the worker's own and is **not the
  parent's**, and only then lets it continue; a worker it cannot verify is refused and stopped
  as a single process. No signal is ever sent to the parent's own group (`_killpg` refuses it).
- **Load.** The worker re-parses the protocol (its SHA-256 must equal the parent's), creates the
  arm's engine (the real factory builds `VllmEngine` with `VllmBackend`, inside the worker) and
  replies with the engine's `describe()`, which must carry a `load_id`. The parent adds
  `isolation` (mode, start method, worker pid and process group). A failed load is reported,
  the worker exits, and the run aborts as before. `--load-timeout` covers the start, the
  process-group check and the load.
- **Before the run**, a short-lived **probe worker** imports vLLM and transformers, reports their
  `__version__` (reconciled with the pins as before) and builds and reads back a
  `SamplingParams` object for every (n, seed) of every arm; it must exit with code 0. The parent
  checks that there is one read-back for every (n, seed) of every protocol arm and that each
  holds the requested values (`preflight(..., sampling_observed=...)`, recorded as
  `sampling_built_in: probe_worker`). This is how preflight keeps those imports out of the
  parent. A failing, interrupted or malformed probe stops the run before any output is written;
  an interrupted probe is stopped at once, without waiting `--kill-grace` first.
- **Messages.** Each message is an 8-byte length followed by UTF-8 JSON, read and written by
  `process_worker` itself over a `multiprocessing` pipe; the parent never unpickles worker
  output. The parent's end is non-blocking and **every timeout covers the whole message**: a
  worker that sends part of a message (even part of the length) and stalls times out exactly
  like one that sends nothing, and a request the worker does not read times out too. A message
  announcing more than 256 MiB is refused before it is read. JSON is strict: `NaN`, `Infinity`
  and numbers that overflow to infinity are refused, and so is JSON too deeply nested to parse.
- **Batches.** One message per shard, in the runner's order; the worker answers in order.
  Every reply is validated strictly: exact keys and types (no booleans for integers), one entry
  per request, each echoing its request with the same JSON values and types (`true` is not `1`,
  `1.0` is not `1`). The parent returns `BatchItem`s carrying its own request objects. A
  request the engine failed comes back as that request's error (`WorkerRequestError`); a batch
  the engine failed as a whole fails that batch (`WorkerBatchError`) and the worker stays usable.
- **Crashes, timeouts, malformed replies.** A worker that exits, does not send its whole reply
  within `--batch-timeout`, or sends anything malformed is marked dead (and its process group
  terminated, then killed, if still running). So is a worker after any other unexpected error
  once the request was sent (for example `MemoryError` while decoding). That batch fails, every
  later batch of the arm fails at once (`WorkerDied`), and the close fails, so the runner stops
  before the next arm and writes its aborted manifest with every record kept.
- **Close.** The worker closes its engine, replies with the engine's close report, and exits.
  `close()` returns only once the worker has **exited with code 0 and been reaped** and **no
  process is left in its process group**; a worker that fails to close, exits with another
  code, or does not exit within `--close-timeout` is terminated (after `--kill-grace`, killed)
  and the close fails. Processes left in the group after the worker exits get `--kill-grace` to
  exit by themselves; whatever is still there is killed and the close fails. The next arm is
  therefore never started while the previous worker, or anything left in its group, is alive.
  The close report records the worker's pid, process group, exit code, how it ended, what was
  found in its group (`none`, `exited`, `killed`, `still_running`, `unknown`), and the
  engine's own close report. A worker the parent stopped earlier (after a timeout or a
  malformed reply) is recorded as `terminated` or `killed`, not as `exited`. If the parent goes
  away, the worker releases its engine and exits.
- **Signals the parent may not send.** If the group (or the worker) cannot be signalled
  (`PermissionError`, for example a setuid program left in the group), the group is recorded as
  `unknown` and the close fails; its processes are **not** reported as stopped, and may still
  be running. The close report is still recorded and the pipe still closed.
- **Interrupts.** Workers run in their own process groups, so a terminal Ctrl-C reaches only
  the parent; they also ignore SIGINT. The parent decides: an interrupt while a batch is in
  flight terminates the worker during the runner's close, and the close report records that.
  While the parent stops a worker, closes its pipe and records its close report, a **further
  Ctrl-C is held back until that has finished**, then raised. The hold is bounded by the
  cleanup: at most about **5 × `--kill-grace`** (150 s at the default 30 s) when a failed load
  or a probe is given `--kill-grace` to exit first, and about 4 × `--kill-grace` for a close.
  Starting a worker is covered as well: an interrupt while the worker process starts is held
  until the parent can stop it. Only the main thread receives SIGINT. A Ctrl-C arriving before
  the runner reaches a worker's close (see the follow-ups) is not covered here.
- **What process groups do not cover.** A descendant that leaves the worker's process group
  (`setsid`, `setpgid`, a daemonising double fork) is neither seen nor stopped, and neither is
  anything another service starts on the engine's behalf. A worker that is unkillable (for
  example stuck in the driver) is reported `still_running`, and the interpreter then waits for
  it at exit.
- **What process exit does not show.** A worker that exited, with an empty process group, does
  **not** by itself show that every GPU resource was released: an escaped descendant may still
  hold the device, and the driver releases device memory asynchronously. GPU release is
  **unverified**; device-level checks (NVML / `nvidia-smi` and the process list after each arm)
  remain required on the GPU host.
- **Timeouts** (`--load-timeout` 3600 s, `--batch-timeout` 14400 s, `--close-timeout` 600 s,
  `--kill-grace` 30 s) are provisional defaults to set for the host and the run. Each must be a
  finite number of seconds in (0, 7 days]; `--kill-grace` at least 1 s. Anything else is a usage
  error before any work starts.

## C1: what the engine guarantees (control flow, tested with stand-in modules)

- **Complete, typed settings.** All inference keys the engine uses must be present
  (`chat_template`, `enable_thinking`, `user_message`, `temperature`, `top_p`, `top_k`,
  `presence_penalty`, `max_tokens`), typed (`enable_thinking` a real boolean; numbers not
  booleans; `top_k` an integer) and in range; `max_tokens` must be below `max_model_len`. Any
  other inference key is refused. Engine keys must all be known: `name`, `version`,
  `transformers_version`, `dtype`, `max_model_len`, `llm_seed`, `gpu_memory_utilization`,
  `tensor_parallel_size` and `generation_config` are required and pinned, and `batching`,
  `sharding`, `hardware` are optional pinned strings, and so are `vllm_module_version` /
  `transformers_module_version` (see versions below). `REPLACE` (any case, leading spaces
  ignored) is refused everywhere. Any other engine key (for example `quantization` or
  `trust_remote_code`) is refused, not dropped. `dtype` must be `bfloat16`, `float16` or
  `float32` (`auto` would let vLLM choose). `generation_config` must be `auto` (read from the
  hashed checkpoint) or `vllm` (vLLM's defaults); a path is refused because vLLM would read a
  file outside the checkpoint. The engine always passes `trust_remote_code=False` and
  `load_format="safetensors"`.
- **Checked before any arm runs or any output is written** (`run.py`, in this order): the
  offline and no-telemetry variables are forced (a conflicting value is refused); the dataset
  and leakage reference are loaded against the protocol (C3); the grader's dependency versions
  are checked and both version sources of vLLM and transformers are reconciled (below); a
  `SamplingParams` object is built and read back for every (n, seed) the run requests; and
  every arm's settings, storage and loaded-file inventory are checked (`preflight()`). Each load
  repeats the settings, storage and inventory checks before the backend is created, and checks
  the imported modules' versions against the protocol again. What needs a loaded engine (the
  tokenizer, observed engine settings) is checked or recorded at load time; the effective
  engine configuration stays a GPU-host check.
- **Versions, compared exactly.** The protocol's `version` / `transformers_version` are the
  **package-metadata** versions (`importlib.metadata`); preflight requires the installed
  metadata to equal them. The modules' `__version__` strings, read in preflight (an import,
  assumed to need no GPU; see the GPU-host table) and again at load, must equal the same pins,
  or an explicitly registered `vllm_module_version` / `transformers_module_version` when the
  module reports the pin plus a local-version suffix (`<pin>+<local>`, for example `+cu124`).
  Only that form can be registered: a different release under the module name is a mixed
  install and is refused. Nothing is stripped or normalised: a suffix is refused until it is
  registered.
- **Storage (`verify_storage`).** Required, or the load is refused:
  - the path, made absolute, contains no `..` component and no symbolic-link component (it is
    loaded by its resolved path), and the checkpoint contains no symbolic link;
  - **every entry**, files as well as directories, reports a **read-only mount** (`statvfs`
    `ST_RDONLY`), so a single file bind-mounted from writable storage is refused; the set of
    device ids is recorded;
  - no entry is writable by this process (`access(W_OK)`, which also applies ACLs for this
    process's identity);
  - every parent directory up to `/` is **owned by root**, has no group or other write bit and
    is not writable by this process, so no ordinary user can rename the checkpoint away and put
    another directory in its place. **There is no exemption for sticky directories**: a
    checkpoint below `/tmp` is refused. A process running as root is also caught here, because
    every parent on a writable filesystem is writable by root; run as an unprivileged user.
  - the process does not run with **effective uid 0**: refused explicitly, whatever the
    filesystem reports (on a fully read-only filesystem root's writes fail, so the parent check
    alone would not catch it). Writability is checked with the process's **effective** ids
    (`os.access(..., effective_ids=True)`).

  Recorded, not required: entries whose mode bits would allow writing on a writable mount (the
  mount, not the mode bits, is relied on). **Not established by any local check**, so the
  operator must verify and record them (`STORAGE_OPERATOR_CHECKS`, also in every load's
  `observed.storage`):
  - that the read-only mount is enforced by the host (mount options or an immutable image), not
    only reported by `statvfs` inside a container;
  - that no writable alias exists: no other mount, bind mount or overlay exposes the same
    storage writably, to this or any other user or process;
  - that no privileged process can mount over, remount read-write or replace the checkpoint or
    any parent **at any time during the run**, including after load (files may be opened lazily,
    and the close re-hash reads them again); the local checks trust root;
  - that no ACL, group membership or privileged process outside this one can write the
    checkpoint or its parents;
  - that the files on the mount are the ones fetched and registered (re-hashed after mounting).

  The before/after hashes only show that the files were the same when hashed at load and at
  close (**snapshot consistency**); they do not show what vLLM read in between, which is why
  protected storage is required for the whole run.
- **Registered loadable artifacts.** Every file in the checkpoint directory is hashed; symbolic
  links, FIFOs, sockets and devices are refused. The protocol's
  `checkpoint.loadable_files_sha256` must map **every** loadable file to its SHA-256, and
  include the weights file (equal to `weights_sha256_lfs`) and `chat_template.jinja`. A file in
  the directory that is not registered (an extra weight shard, a `generation_config.json`, a
  `.cache/` entry) is refused, as are missing or changed files. Only `README*`, `LICENSE*`,
  `NOTICE*` (`.md`/`.txt`/`.rst`) and `.gitattributes` **at the root** are treated as
  documentation (hashed, reported in `other_files_sha256`), unless the protocol registers one as
  loadable. `chat_template.jinja` must match the protocol's git blob id, checked on the same
  bytes that are hashed and rendered.
- **Prompt.** The protocol records the prompt in a notation (`<problem>\nPlease reason …`),
  where the two characters `\n` stand for the authors' newline in
  `problem + "\n" + instruction`. Only that exact notation is accepted. The engine applies the
  **checked bytes** of `chat_template.jinja`, passed explicitly as `chat_template`, with
  `add_generation_prompt=True` and `enable_thinking` from the protocol. Whether the tokenizer's
  own built-in template matches is recorded, not used. *The research lead should confirm the
  notation reading, and whether the authors rendered with this file (R6/R7).*
- **Requested and observed settings, recorded separately.** `describe()["requested"]` holds the
  engine keyword arguments, the list of engine arguments left at vLLM's defaults, the sampling
  settings and the template arguments. `describe()["observed"]` holds what the backend reported:
  the values a probe `SamplingParams` holds, vLLM's default sampling parameters, a best-effort
  read-back of the engine's model configuration (`dtype`, `max_model_len`, `seed`; attribute
  names are version-dependent), the tokenizer's eos id, whether its built-in template matches,
  library versions, the storage check, and the process's `VLLM_*`/`HF_*`/`TORCH*`/`CUDA_*`/
  `NCCL_*`/`DO_NOT_TRACK` variables (credential-like names redacted). Every per-request
  `SamplingParams` object is read back too (`n`, `seed` and the sampling settings); a value it
  does not hold fails the batch. The stop set applied at generation time is not observable here.
- **Offline and no telemetry.** `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`,
  `VLLM_NO_USAGE_STATS` and `DO_NOT_TRACK` are set to `1` (by `run.py` first, and again by
  `VllmBackend.load`); a variable already set to anything else is refused rather than
  overridden. That vLLM honours `VLLM_NO_USAGE_STATS` / `DO_NOT_TRACK` is a **version-dependent
  assumption** to check on the pinned release (for example with egress blocked and logged).
- **Load identity.** Each load gets a fresh `load_id`.
- **Request echoes.** Each shard is one `LLM.generate` call. Every vLLM output's prompt must be
  the prompt rendered for the request at that position, or the whole batch fails
  (`BatchMismatch`). This cannot tell apart two requests that render the same prompt (one item
  under several seeds, or duplicated problems), and it compares text, not token ids. A request
  whose arm or settings differ from the engine's fails only itself.
- **Failed loads are cleaned up.** If anything fails after vLLM was created (the tokenizer, the
  settings read-back, `describe`), the engine is torn down the same way as in a close before the
  error propagates, since the runner never receives it; if vLLM's constructor itself fails, only
  the release steps run. The cleanup is recorded as a `load_failed` report.
  - **Frame clearing is best-effort.** A frame inside the engine's own methods can keep the
    engine alive through the release, in the propagating error's traceback or in that of an
    error it was raised from (`__cause__`) or during (`__context__`). The local variables of the
    frames listed in those tracebacks are cleared (cycles are detected). The errors themselves,
    their types, messages, chaining and traceback lines are kept. **This reduces post-mortem
    debugging information:** a debugger can no longer inspect those frames' local variables. On
    Python 3.11, clearing the frame of a suspended generator or coroutine finalises it, so its
    `finally` code runs during the cleanup. It does **not** find every reference: an error
    attribute such as `AttributeError.obj`, exception-group members, frames reachable only
    through `f_back`, and library objects can still hold the engine. No attempt is made to find
    them; the observation (C1, close) records whether the engine object actually went away. An
    interrupt during the clearing does not skip the teardown.
  - **Ordinary cleanup failures** (a failed or missing shutdown step) are attached to the
    original error as a note (`cleanup after this failed load also failed: …`) and recorded in
    the report; the original error is re-raised. If the error cannot take a note (a library
    replaced `__notes__`), the report records `note_not_attached` and the original error is
    still the one raised.
  - **An interrupt during the cleanup** propagates instead of the original error; the original
    failure stays reachable as the interrupt's `__context__`, and the report records
    `interrupted: true`.
- **Engine-object observation (Python object lifetime only).** At load the engine takes a weak
  reference to the vLLM object. In the release, after garbage collection and before the CUDA
  cache is cleared, it records `engine_object_after_release`: `released` (the object is gone),
  `alive` (something still references it), `not_observable` (it cannot be weak-referenced, or
  the observation did not run) or `no_engine` (vLLM's constructor failed). At a **normal close,
  anything but `released` fails the close**, so the next arm is not loaded: a still-referenced
  object, and an object whose release cannot be observed, are not treated as released. After a
  failed load it is recorded only, since the run aborts anyway. This establishes **Python
  object lifetime only**: that the adapter's own vLLM object was freed in this process. It says
  nothing about worker processes, CUDA contexts or device memory, which stay separate real-run
  checks.
- **Close, in this order.** (1) The engine's own shutdown, a required step:
  `llm_engine.shutdown`, or `llm_engine.engine_core.shutdown` on the V1 engine (these names are
  **version-dependent assumptions**; a version with neither fails every close). (2) Every
  reference the engine holds to vLLM objects is dropped. (3) The release steps:
  `destroy_model_parallel`, `destroy_distributed_environment`, `gc.collect`, the observation
  above, then `torch.cuda.empty_cache`, so memory freed by the collection is returned to the
  device as well. Each step is
  recorded with its status (`ok`, `failed`, `unavailable`); a failed step does not stop the
  later ones. Then this process's allocated **and reserved** PyTorch memory and its live
  `multiprocessing` children are recorded, and the checkpoint is re-hashed. A close report is
  always recorded, with every error. Any error fails the close and stops the run before the next
  arm; a later `close()` raises again, so a failed close is never reported as a success.
- **Interrupts during close.** An interrupt (`KeyboardInterrupt`, `SystemExit`) raised inside
  the engine's shutdown hook ends that step (recorded as failed); the references are still
  dropped and the release steps still run, but the memory, child-process and re-hash checks do
  not, the report records `interrupted: true`, and the interrupt then propagates. An interrupt
  during the release steps ends them at that point. An interrupt during the memory,
  child-process or re-hash checks marks the report `interrupted: true` with what ran so far.
  If the runner closes the engine while another failure is propagating, it records only
  `closed: false` in the manifest; why is in the close report. In particular, when Ctrl-C
  arrives during generation, the propagating interrupt's traceback still holds the engine's
  frame, so the close made on the way out observes the object `alive` and reports that error.

**What survives an interruption.** The close reports are kept in memory and written only into
`runtime_report.json`. After an interrupt **during the run** (once the runner's abort handling
is active), the runner writes its aborted manifest and records, and `run.py` then writes
`runtime_report.json` best-effort, marked `interrupted: true`, with every close report recorded
so far; the same interrupt object then propagates and no summary is printed. The report is
written **atomically** (a temporary file, flushed, then hard-linked into place, never
overwriting), so it is either complete or absent. A **second interrupt** during that write is
dropped in favour of the first, and leaves no report. An interrupt **before the run starts**
(during the checks) writes nothing, and one that arrives inside the runner after it created
the output directory but before its abort handling starts leaves the directory without a
manifest, so the report is skipped. After an ordinary failure the report and the printed
summary both carry the close reports.

**What the close does not show.** `torch.cuda.memory_allocated` and `memory_reserved` count only
the PyTorch allocator of the process that holds the adapter: the arm's worker process (in the
default process isolation) or the parent (with the in-process test seam). vLLM's own engine
processes, CUDA contexts and NCCL buffers are not counted, so zeros here do not show that vLLM
released the GPU. `--max-parent-torch-memory-after-close` (compared with both figures; the name
predates the worker processes) can only catch memory held in that process. The `multiprocessing` children list misses processes started any other way, and
right after shutdown it may still list an engine process that is exiting, or no longer list one
whose memory is not yet released. Worker exit and device-level memory (NVML / `nvidia-smi` after
each close) are real-run blockers.

## C2: grading rule and the interpretations to review (R9)

Rule: cut a stray reasoning block, take the **last** `\boxed{}` or `\fbox{}`, no box =
**incorrect** (graded, in the denominator), then `math-verify`. The adapter was written from the
protocol's description of the rule, **not compared with the authors' released
`exp8_math/code/mathqa.py`** (git blob `808a0477…`). Its revision string says "unreviewed" and
`grader.REVIEWED` is `False`; both stay so until the comparison is done **and** the independent
evaluator has accepted the grader. While it is unreviewed the CLI refuses `--split heldout`.
The interpretations to confirm or change:

1. Reasoning cut: keep the text after the last `</think>`; an unclosed `<think>` is searched
   whole.
2. `</think>` is assumed to survive detokenisation. If it is a special token and vLLM skips
   special tokens (its default), nothing is cut and boxes inside the reasoning count.
3. Only braced boxes count; `\boxed 5` is not recognised.
4. `\boxed` and `\fbox` are treated alike and the later one wins (some reference
   implementations prefer `\boxed`).
5. If the last box command is not a complete braced group (unbalanced or unbraced), the
   response has no answer; there is no fallback to an earlier box.
6. Whitespace between the command and its brace is accepted (`\boxed {4}`).
7. Braces are matched literally, so balanced `\{ \}` work and unbalanced ones make the box
   incomplete.
8. An empty box is incorrect without calling math-verify.
9. Gold and prediction are wrapped as `$…$` before `parse`, with `parsing_timeout` and
   `timeout_seconds` of 5 s.
10. **Timeouts and math-verify errors are grader errors, not incorrect answers.** The backend
    passes `raise_on_error=True`, so the harness records a `GraderError` row (status `error`,
    `correct: null`) and the CLI exits non-zero. math-verify's own default is to return an empty
    parse or `False` on a timeout, which would count as incorrect; adopting that needs an
    explicit, recorded review decision. **Unverified:** the tests only show that an exception
    raised by a stand-in backend becomes a grader error. Whether math-verify 0.9.0 accepts
    `raise_on_error` and actually raises on a timeout, rather than returning an empty parse or
    `False`, must be checked against the real library on the isolated host (for example with a
    deliberately slow synthetic expression) before any run relies on this.
11. A gold answer that does not parse is a grader error; a prediction that parses to nothing is
    incorrect.
12. math-verify's timeouts use signals, so grading must run on the main thread; the backend
    refuses any other thread.

The keyword names in 9–10 are from the documented math-verify API and unchecked on 0.9.0; if it
does not accept one, every grade fails loudly as a grader error. The installed versions of the
protocol's `grader.dependencies` are checked before anything runs (a pinned version that
differs is refused; a `REPLACE` pin is recorded as unpinned) and written to
`runtime_report.json`.

## C3: dataset loader, protocol registration and leakage

- **MATH-500** (`test.jsonl` of `HuggingFaceH4/MATH-500`): IDs are the source's `unique_id`; the
  split must be `heldout`.
- **MATH training split** (smoke run): the answer is the last complete `\boxed{}` of the
  solution. The ID is the source's `unique_id` when the file has one (all rows or none),
  otherwise `math-train-` plus 16 hex digits of the SHA-256 of the **normalised** problem text
  (independent of file order and of whitespace or case differences). Two rows with the same
  normalised problem then share an ID and the file is refused; removing duplicates is a recorded
  operator step. The split must be `development`.
- **One byte snapshot.** The file is read once; its SHA-256 and the parsed rows come from the
  same bytes. An empty file is refused. Lines are split on `\n` only.
- **Protocol registration.** The source of each split is registered in the protocol, under
  `dataset.runtime_sources`, and the CLI takes no expected values of its own:

  ```json
  "runtime_sources": {
    "hash_definitions": "wsp-runtime-dataset-1",
    "development": {"shape": "math_train", "source_file_sha256": "…", "source_rows": 7500,
                    "source_ids_sha256": "…",
                    "selection": {"rule": "first_n_in_file_order", "n": 20},
                    "items_sha256": "…"},
    "heldout": {"shape": "math500", "source_file_sha256": "…", "source_rows": 500,
                "source_ids_sha256": "…", "selection": {"rule": "all"}, "items_sha256": "…"}
  }
  ```

  Hash definitions (`wsp-runtime-dataset-1`, also written into `runtime_report.json`):
  `source_file_sha256` is the SHA-256 of the **raw source file bytes** as fetched, before any
  conversion; `source_rows` is the number of items converted from the whole file;
  `source_ids_sha256` is the SHA-256 of the canonical JSON of the sorted item IDs of the whole
  file; `items_sha256` is the harness's `dataset_sha256` of the **converted items after the
  selection rule** (IDs, split and prompt/answer content hashes) — a hash of converted items,
  not of the source file, and the same value the run manifest records as `items_file_sha256`.
  Selection rules: `all`, or `first_n_in_file_order` with a positive `n` not above
  `source_rows`. Held-out entries must use `all` and register `items_sha256`, and their
  `source_rows` must equal `dataset.n_items` when present (which must then be an integer).
  Unknown keys, in a split entry or at the `runtime_sources` level, and placeholders are
  refused.
- **Leakage.** `check_disjoint` refuses any item whose normalised problem text (NFKC, case
  folded, whitespace collapsed) or ID also occurs in the reference set, **whatever the answers
  say**, since answer formatting differs between sources. `combine()` runs it between splits as
  well as the harness's exact-content check. Normalisation catches copies that differ only in
  such details; it **cannot prove** that two sets are disjoint, because paraphrased,
  reformatted or translated problems are not detected.
- **Wired into the CLI.** `run.py` loads the whole `--source` (the run's split) and the whole
  `--leakage-reference` (the other split), each against its registration, refuses any overlap
  between the **whole** files, and only then applies each file's registered selection rule and
  checks its registered converted-items hash, **for both files**: a wrong registration of the
  other split is found in a development run, not at that split's first run.

## `runtime_report.json` (supplementary evidence)

After a run (or an abort that left a manifest), `run.py` writes `runtime_report.json` next to
the manifest: the run ID, the dataset and reference reports, the hash definitions, the leakage
result, the forced environment variables, the installed versions, the grader and its dependency
versions, every arm's preflight result and close report, the manifest's SHA-256 and whether the
manifest's `items_file_sha256` matches the selected items.

- **Only into this invocation's run.** It is written only if the manifest there carries the run
  ID this invocation passed to the runner; a directory from another run (for example a rerun
  into a used `--output`) is never touched, and the summary says it was skipped. The file is
  created exclusively and never overwritten.
- **The original failure is kept.** If the report cannot be written, the error is recorded in
  the summary as `runtime_report_error` and the run's own failure (if any) is still the one
  reported; the exit code is non-zero.
- **Interrupts.** See "What survives an interruption" above: an interrupt during the run still
  gets a best-effort report marked `interrupted: true`; the interrupt itself propagates and the
  exit status is the interpreter's, not one of the documented codes.
- **Its SHA-256 is printed** in the summary (`runtime_report_sha256`) so it can be recorded
  externally. **Printing the hash does not protect the report**, and naming the manifest's hash
  does not either: anyone who can write the run directory can change the file. Any decision that
  relies on its dataset, leakage or cleanup checks must first verify the report against a
  separately protected record of that hash (for example in access-controlled storage or a signed
  record), as for the run manifest. `verify_run` does not read it.

## What still needs a real GPU (or the real libraries)

None of these can be shown with stand-in modules. Each must be checked on the isolated host,
recorded, and reviewed before the confirmatory run.

| Check | Why it is open |
|---|---|
| **Runtime lock** | No lock exists; the versions the checks below run against are unknown until it is resolved (R6, R9) |
| **The real engine in a worker process** | `vllm_engine` and `runtime_probe` (in the worker) have not run with the real libraries; confirm that vLLM starts, serves and shuts down inside a `spawn` worker that is not a daemon, and how long a load and a close take, to set the timeouts |
| **Engine shutdown, worker exit and GPU-memory release** | `close()` calls the shutdown hooks it finds (names version-dependent). Whether vLLM's worker processes exit and the device memory is freed before the next arm loads must be measured at device level (NVML / `nvidia-smi` and the process list after each close, allowing for the engine process's exit time). This process's PyTorch figures do not show it |
| **Effective engine settings** | Read back and record the effective values of the engine arguments left at vLLM's defaults (batch limits, prefix caching, eager/compilation, KV-cache dtype, attention backend, scheduler) and the effective model configuration |
| Storage | The operator checks listed under C1, storage |
| Which files vLLM actually opens | Every file in the directory is registered and hashed, offline mode is forced and `generation_config` cannot be a path, but a file read from elsewhere (vLLM's caches, a compile, Triton or FlashInfer cache, environment-selected paths) would be invisible. Check with file-access tracing on the host |
| Importing vLLM and transformers in preflight | Preflight imports both (for `__version__` and `SamplingParams`), assumed to need no GPU. On the real libraries the import may run platform or CUDA detection, plugin discovery or logging setup; record what it does on the host |
| Weak-referenceability | Confirm that the pinned vLLM's `LLM` object can be weak-referenced (otherwise every close fails as `not_observable`) |
| Version strings | Record the package-metadata and module `__version__` strings of vLLM and transformers in the image; register a module version (`<pin>+<local>`) explicitly if they differ by a local suffix |
| Telemetry | Confirm that `VLLM_NO_USAGE_STATS` / `DO_NOT_TRACK` disable usage reporting on the pinned vLLM (with egress blocked and logged) |
| Effective stop tokens | The stop set vLLM applies at generation time is not observable from the adapter; record it on the chosen version and confirm it includes eos `248044`. Check whether `</think>` survives `skip_special_tokens` |
| vLLM API details | `LLM(...)` keyword names (`generation_config`, `load_format`, `trust_remote_code`), `generate(prompts, params_list)` returning outputs in input order with `.prompt`, `finish_reason` and `token_ids`, `SamplingParams` attributes, `get_default_sampling_params`, `llm_engine.model_config`, the shutdown hooks, `torch.cuda.memory_reserved`, and how child samples of an `n > 1` request are seeded |
| Chat template rendering | Confirm that `apply_chat_template(..., chat_template=<file>, enable_thinking=True)` on the pinned transformers renders the prompt the authors used |
| math-verify 0.9.0 | Confirm `parse` / `verify` keyword names (`parsing_timeout`, `timeout_seconds`, `raise_on_error`); that a timeout raises with `raise_on_error=True` instead of returning an empty parse or `False` (stand-in exceptions cannot show this); error behaviour; and the C2 interpretations against `mathqa.py` |
| Throughput, KV-cache headroom, peak memory | Measured in the development smoke run, not estimated here |

## Follow-ups

- **Repeated hashing (performance).** Each arm's checkpoint is hashed up to four times (CLI
  preflight, the loader's preflight, the engine's inspection, the close re-hash), and `base` and
  `base_reload` share a directory. That is correct but slow on large weights; reusing a snapshot
  between the checks that run back to back is a possible optimisation, kept out of this change.
- The repository's invisible-character test does not flag U+2028.
- **Message size cap (R30-9).** The 256 MiB per-message cap is a fixed bound, not one derived
  from the protocol. Sizing it from `max_tokens`, `n` and the shard size, and recording the
  largest reply seen, remain to do. (An oversized message is now refused as such, as a protocol
  error, rather than reported as a worker that died.)
- **A batch while one is in flight (R30-13).** `WorkerEngine.generate_batch` does not itself
  refuse a new batch after an interrupted one. The runner never does this (an interrupt ends
  the run), and a stale reply would fail the echo check unless the requests were identical;
  refusing it in `generate_batch` is a small hardening for direct callers.
- **Runner-side interrupt window (re-check note 2).** In `tools/wsp` (`runner.py`,
  `_close_quietly`), a second Ctrl-C that lands after a batch was interrupted but before
  `WorkerEngine.close()` enters its cleanup is swallowed, and the busy worker can be left
  running (and the interpreter then waits for it at exit). The window is a few bytecodes; the
  fix belongs in the harness runner (outside this runtime) or in a last-resort stop of live
  workers at exit.
- **Close wait restarts its timeout (re-check note 5).** After a successful `closed` reply,
  `close()` waits up to a fresh `--close-timeout` for the worker to exit instead of the time
  left. A close is still bounded (about 2 × `--close-timeout` + 4 × `--kill-grace`); sharing
  one deadline is a tidy-up.
- **Check `--checkpoint` before the probe (R30-14).** The probe (which imports vLLM) runs
  before `--checkpoint` is parsed and before storage preflight, so a mistyped arm path is
  reported only after the probe. Parsing the checkpoints first would report it sooner.

## Running (isolated host only; not authorized yet)

```bash
python tools/wsp_runtime/run.py --protocol P.json \
    --source /research/data/math_train.jsonl \
    --leakage-reference /research/data/math500_test.jsonl \
    --checkpoint base=/research/ckpt/base --checkpoint edited=/research/ckpt/edited \
    --checkpoint base_reload=/research/ckpt/base --split development \
    --operator op1 --purpose smoke --output /research/runs/RUN [--keep-outputs]
```

Run as an unprivileged user. This needs the runtime image and lock, checkpoints on a verified
read-only mount below root-owned parents, local dataset copies from an approved fetch registered
in the protocol, a `per_arm_sequential` protocol with complete, pinned settings and registered
loadable files, and the decisions listed in [run-readiness.md](run-readiness.md) and in the
draft readiness proposal (PR #27): roles, budget, isolated infrastructure and the licence
review. None of these exists yet. The exit code is non-zero if the run aborts, any record is an
error, or the report could not be written.
