# Local Agent Master

A separate terminal application using **one installed local Tantular model** for the
Master and all workers. No cloud fallback, model downloads, training, recursive delegation,
or changes to the production scanner explanation prompt. No extra dependencies.

## Start

From this checkout, with the existing development virtual environment:

```sh
ollama list
PYTHONPATH=src .venv/bin/python -m tantular_cyber.agent_master \
  --model tantular-cyber:latest --root .
```

The tag above was found on the development machine; choose an actually installed Tantular
tag from your own `ollama list`. `--model` (or `TANTULAR_MASTER_MODEL`) is required. The
application checks `/api/tags` and `/api/show`, refuses remote/cloud models, and uses
loopback Ollama only. Native tool-calling support is not required. Generation uses
`think=false`, `stream=false`, a JSON schema, pinned sampling settings, and no native tools.
The installed entry point is `tantular-agent-master` after reinstalling the wheel/editable
package. The module command above works immediately in a development checkout.

Example messages:

- `Tolong baca README.md dan ringkas tujuan repositori. Jangan jalankan kode.`
- `List the files in src/tantular_cyber and report the module responsibilities.`
- `Prepare an AI-authored draft review packet; keep human verdicts blank.`
- `Ask the validator to run the registered core checks and report the exit status.`

Only the Master addresses the terminal. Progress shows the plan, assignments, task outcomes,
and a final report. It does not print internal agent conversations or chain of thought.
Enter your request at `you>`. The chat keeps pasted lines together and ignores Agent Master
launch commands and recognizable terminal transcripts. Use `/quit` to return to the shell
before launching another session.
Final reports list recorded tool actions and their outcomes. When one successful file read
completes the only task, the Master presents the worker's summary directly. Other reports
use model synthesis informed by the original request and tool receipts.
Language detection uses a small Indonesian keyword heuristic; use `--language id` or
`--language en` when an ambiguous or mixed-language message needs an explicit choice.

## Inference context (`--num-ctx`)

Ollama treats a different `num_ctx` as a different runner configuration. If the same
weights are already loaded with another context (for example by a long-running job), a
request with a mismatched context can wait behind that work for a reload. On a shared
runtime this was measured as a planning timeout (see [agent-master-smoke.md](agent-master-smoke.md)).

`--num-ctx 0` (the default) resolves the context automatically after the model is verified
as local GGUF:

1. `/api/ps`: if the selected tag, or any alias with the **same weights digest**, is
   already loaded, reuse its `context_length` (startup prints `(loaded model)`).
2. Otherwise the model's declared `num_ctx` parameter from `/api/show` (`(model default)`).
3. Otherwise 8192 (`(default)`). Older Ollama versions without `/api/ps` (404) fall through.

An explicit `--num-ctx N` (512..131072) is always respected; startup warns when it differs
from a loaded runner and suggests `--num-ctx 0`. The application never unloads models,
stops other workloads, or changes the daemon. The resolved context is persisted in the run
limits, and `/resume` refuses to continue a run whose inference context would change.
While awaiting a generation the Master prints metadata-only progress every 10 seconds
(`Waiting for local Ollama: 20s; context=16384.`); on timeout or cancellation the pending
request is cancelled and awaited.

## Queue, dependencies, restart and cancellation

- `/status`: recent run IDs, status and consumed budgets.
- `/status RUN_ID`: task states and the saved final report.
- `/cancel`: cancel the active request, close its HTTP operation or kill its process group.
- `/resume RUN_ID`: continue an interrupted run's **queued** tasks with its original model,
  workspace policy, call/step counters, limits (budgets and tool file/traversal/output/
  subprocess bounds) and absolute deadline.
- `/quit`, EOF or Ctrl+C: stop active work and preserve task status.

SQLite state and drafts default to `~/.local/state/tantular-agent-master/`; use
`--state-dir /private/path` to isolate sessions. State directories are private and the
SQLite file is mode 0600. A process lock prevents two Masters recovering the same live queue.
Do not share a state directory between users. Task descriptions, structured results,
metadata-only effect receipts and final reports persist. Raw prompts, conversations, source
observations, and subprocess stdout/stderr are not saved to ordinary logs or SQLite.
Treat private state and deliberate draft/scan artifacts as potentially sensitive.

Completed actions are never automatically repeated. A task interrupted in-flight becomes
blocked; its committed intent and any artifact are retained for inspection. Its dependent
tasks are blocked too. Independent queued tasks can resume. If planning was interrupted,
submit a fresh request. Expired runs do not acquire a fresh deadline on restart. There is
no automatic retry of uncertain writes or subprocess effects. SQLite plus create-exclusive
writes provides conservative recovery, not transactional exactly-once subprocess execution.

Cancellation closes the application's model request; the Ollama server may take additional
time to notice a disconnect and stop inference. The application dispatches no further tool
requests after cancellation. No explicit Ollama model unload or daemon changes are made.

## Roles and registered tools

Workers run sequentially, each in a separate bounded context. They share model weights and
endpoint, not conversation history. Dependency results are passed as structured summaries.

| Role | Tools |
| --- | --- |
| analyst | `read_file`, `list_files`, `search_files` |
| dataset | read/list/search, `write_draft`, `artifact_index` |
| validator | read/list, `run_check`, `scan`, `validate_dataset` |
| reporter | read, `artifact_index`, `write_draft` |

Roots are explicitly authorized with repeatable `--root` options. Tool paths are relative
to a numeric root (default 0). Parent traversal, absolute paths, symlinks, secret filenames,
hidden metadata, oversized files and binary file reads are refused. Search is literal, not
a model-supplied regular expression. Complete files are redacted before evidence is cut.
Inventories and excerpts are bounded and report truncation. Concurrent modification of an
authorized directory by another local process is not an OS sandbox; use a stable checkout.
When `read_file` returns `next_offset`, a worker can pass it back as `offset` to read the
next excerpt. A missing file returns a receipt that lets the worker correct the path.
A reporter can cite successful evidence from completed dependencies to answer in chat.

The only process tools are fixed argument lists in the trusted tool checkout:

- `run_check` presets: `core` (analysis and Ollama client tests), `agent-master`
  (`tests/test_agent_master_local.py`).
- `scan`: the existing scanner CLI with `--no-llm`; deterministic findings remain
  authoritative, with scanner status and exit code recorded. An incomplete scan cannot
  count as a successful check. Target code is never imported, installed, built, or executed.
- `validate_dataset`: the trusted checkout's `tools/rung3/dataset.py` with a new output
  manifest. It is offered to the Master and validator only when that script exists as a
  regular file; this branch and the wheel do not ship it, so the tool is normally absent.

Registered tests/validators are trusted application tooling explicitly authorized by this
application. Files from arbitrary analyzed workspaces are never selected as executable tests
or plugins. Test/validator presets require a development checkout. Model-supplied shell,
command names, arbitrary subprocess flags, permission changes, and delegation are rejected.
Process stdout/stderr are discarded; deterministic reports go only to the artifact directory.

`write_draft` creates an output-relative JSON artifact with AI authorship, blank human review
fields, pending licence/leakage/training gates and `training_authorized=false`. Model fields
cannot override that envelope. It never exports a training dataset, certifies human review,
or modifies production weights. These AI drafts deliberately do not satisfy D1's current
human-written dataset policy. Human review and policy decisions remain necessary.

## Limits and tests

Use `--help` to inspect every configurable limit. Defaults: 8 tasks, 32 worker steps,
24 generation calls including schema retries, one malformed-output retry, 24,000 response
bytes, 60-second request timeout, 300-second total task duration, 30-second subprocess timeout,
100,000-byte file limit, 2,000-entry inventory traversal, 1,600 output tokens per call and
automatic context (`--num-ctx 0`).
Metadata discovery has its own bounded request timeout before a run begins. Calls and steps
are charged before dispatch. Limit failures are explicit; a failed final Master call produces
a deterministic status report marked `needs_input`. No model is substituted.

```sh
PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_master_local.py -q
.venv/bin/ruff check src/tantular_cyber/agent_master tests/test_agent_master_local.py
```

A bounded real-model smoke request takes three generation calls on the successful path.
Tantular generates the plan, the read request, and the worker result.

```sh
PYTHONPATH=src .venv/bin/python -m tantular_cyber.agent_master \
  --model tantular-cyber:latest --root . --state-dir /tmp/tantular-master-my-smoke \
  --max-tasks 1 --max-steps 2 --max-model-calls 3 --max-retries 0 \
  --request-timeout 60 --max-duration 240 --num-predict 768 \
  --once 'Read README.md through one analyst using read_file, then report its purpose. No writes.'
```

Exit 0 means every assigned task and the final report completed with cited successful tool
evidence. Exit 3 means the request failed or needs input; exit 2 is configuration/startup
failure, and exit 130 indicates Ctrl+C. Inspect persisted tasks and effect receipts when a
model cannot reliably conform to the schema. One successful smoke is not a broad reliability
benchmark. Final natural-language conclusions remain model-generated and advisory; the Master
checks tool evidence, statuses, dependencies and actual artifact paths deterministically.
