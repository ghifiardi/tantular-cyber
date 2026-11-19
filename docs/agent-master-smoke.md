# Agent Master bounded local smoke results

Implementation: `src/tantular_cyber/agent_master/`. Usage: [agent-master.md](agent-master.md).

Actual model discovered by `ollama list` / `/api/ps`: **tantular-cyber:latest**, local GGUF,
9.7B Q8_0, Ollama 0.40.16.
Weights digest: `e79106b3b1ebcc77e88b3e3a4bf0894089e97ab39dc216c804961b3e5f9ff9ce`.
Every attempt used this same local model for the Master and all workers. Loopback tag
discovery and local-GGUF verification succeeded. No alternate model, cloud API, downloads
or native tool calls were used. No model was unloaded and the daemon was not reconfigured.

## Current run (2026-11-19)

The smoke repeated the English README request and budgets from `5e398bba568a`, including the
four-call limit. The Master displayed the analyst's summary and recorded the successful read.
The summary accurately described the repository as a local, read-only security analyzer.

| Check | Observed |
| --- | --- |
| Run `85336b8a1470` | Completed, exit 0 |
| Resolved context | 8192 tokens (model default) |
| Generation calls | 3 (plan, read request, analyst result) |
| Worker steps | 2, for one analyst task |
| Tool evidence | One successful `read_file` receipt, cited by the analyst |
| Draft artifacts, scans, and checks | 0 |

### Deterministic verification

- Agent Master suite (`tests/test_agent_master_local.py`) passed all 54 tests with mocked model
  responses. One regression test replaced the two README rewrite cases. It checks that the
  stored and displayed Indonesian summary preserves both access modes.
- Full repository regression passed 630 tests and skipped the opt-in Ollama scanner smoke.
- Ruff lint and format checks passed for `src/tantular_cyber/agent_master` and
  `tests/test_agent_master_local.py`.
- The Agent Master tests moved from `tests/test_agent_master.py` to
  `tests/test_agent_master_local.py`. The former name is now used by the separate
  `tantular_cyber.agents` work.

## Context diagnosis and earlier passing runs (2026-11-18)

Diagnosis: the original adapter hard-coded `num_ctx=8192`, while a concurrent 500-agent job
had the same weights loaded with context 16384. Controlled metadata-only planning probes
against that runtime (same tag, same Plan schema):

| Context | Result |
| --- | --- |
| 16384 (matching the loaded runner) | valid one-task plan, 13.4 s, `done_reason=stop`, 117 output tokens |
| 8192 (mismatched) | timeout at 25 s |

This implicates context-mismatch reload queueing behind active inference. Ollama scheduler
internals were not inspected; only the controlled behavior above was measured. The fix is
automatic context resolution (`--num-ctx 0`, see the usage guide), not larger timeouts.

Each earlier run used one analyst task, two worker steps, four generation calls, zero retries,
60 s per request, 240 s total, and `num_predict=768` for a single `read_file` of `README.md`.

| Run | Code | Runtime state at start | Context chosen | Result |
| --- | --- | --- | --- | --- |
| `fadef0cb2ba9` | staged fix (pre-merge copy) | 16384 runner loaded, shared with the 500-agent job | 16384 (loaded model) | completed, exit 0 |
| `5e398bba568a` | target repository with a separate Master report call | no Tantular runner loaded | 8192 (model default) | completed, exit 0 |

| Required live check | Observed (both runs) |
| --- | --- |
| Master creates a valid assignment | yes: one analyst assignment |
| Worker performs a permitted read | yes: exactly one `read_file` effect, status `done` |
| Worker returns structured result to Master | yes: `completed`, citing the exact successful effect ID |
| Master generates understandable final report | yes: concise repository-purpose summary |

Both runs: status `completed`, calls 4, steps 2, zero writes/scans/tests, zero draft artifacts.

Limitations of this evidence:

- Three single-task workflows passed across the runs above. The sample is too small to
  estimate reliability.
- The target-repository run did **not** exercise the loaded-context reuse path: the 16K runner
  was no longer loaded, so it resolved the model's declared 8192 and loaded that runner
  itself. Reuse under shared load is shown only by the staged run and by mocked tests
  (name match, digest-matched alias, explicit override, 404 fallback).
- If another workload later requests 16384 on the same weights, it may now wait for a
  reload itself; the application cannot coordinate contexts across independent clients.
- Final natural-language summaries remain model-generated and advisory.

Persisted metadata of the target-repository run (no prompts, source, or conversations):

```json
{
  "run_id": "5e398bba568a",
  "model": "tantular-cyber:latest",
  "status": "completed",
  "calls": 4,
  "steps": 2,
  "num_ctx": 8192,
  "context_source": "model default",
  "tasks": [{"id": "task_1", "role": "analyst", "status": "completed"}],
  "effects": [{"tool": "read_file", "status": "done"}],
  "evidence_is_effect_id": true,
  "draft_artifacts": 0
}
```

## Historical attempts (before the context fix): NOT VERIFIED

The initial Master planning request timed out at 90 seconds. One further bounded attempt
from the installed code timed out at 180 seconds. Each attempt was restricted to one analyst,
two worker steps, four generation calls and zero retries. Only the first generation call
was dispatched in each run: **two calls total**. No task, tool action or draft was created.
The application reported the failure, persisted status, closed the request, and exited 3.

| Required live check | Observed |
| --- | --- |
| Master creates a valid assignment | Not reached: planning timed out |
| Worker performs a permitted read | Not reached; no tool ran |
| Worker returns structured result to Master | Not reached |
| Master generates understandable final report | Not reached; application failure report displayed |

An existing 500-agent Tantular workflow was using the same Ollama endpoint when these tests
started. Contention may have contributed, but was not isolated. These measurements establish
that the tested local runtime did not complete planning within the configured limits;
they do **not** establish whether schema conformance works reliably on an idle runtime.
No success or broad reliability claim is made. A fresh smoke command is in the usage guide.

### Deterministic verification at that time

- Final Agent Master suite: **42 passed** with mocked model responses.
- Complete installed-repository regression run: **557 passed, 1 skipped** (the existing
  opt-in Ollama scanner smoke). Two additional adapter/terminal tests were added after
  that full run; the final 42-test Agent Master suite and Ruff checks passed afterwards.
- Ruff check and format check: pass for all new Python files and the new tests.
- Existing production explanation prompt and unrelated dirty scanner/research files were
  preserved; only the separate entry point was added to `pyproject.toml`.

Persisted smoke metadata (no prompts, source, or raw model conversations):

```json
[
  {
    "id": "af92ad260468",
    "model": "tantular-cyber:latest",
    "status": "failed",
    "calls": 1,
    "steps": 0,
    "request_timeout_s": 90.0,
    "generation_call_limit": 4,
    "duration_limit_s": 240.0,
    "tasks": 0,
    "effects": 0
  },
  {
    "id": "3267d332802b",
    "model": "tantular-cyber:latest",
    "status": "failed",
    "calls": 1,
    "steps": 0,
    "request_timeout_s": 180.0,
    "generation_call_limit": 4,
    "duration_limit_s": 480.0,
    "tasks": 0,
    "effects": 0
  }
]
```
