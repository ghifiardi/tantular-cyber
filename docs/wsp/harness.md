# WSP research evaluation harness (`tools/wsp`)

Status 2026-11-04: **synthetic-only scaffold**. No model, dataset, checkpoint or provider has
been accessed. No baseline, replication, safety result or model improvement has been
measured. Every number this harness has produced so far comes from scripted synthetic arms
and is labelled `synthetic_fixture_not_a_measurement`.

It supports the Week 1 package ([week-1-execution-package.md](week-1-execution-package.md)).
It is research-only and separate from the product. It uses only the standard library, does
not import `tantular_cyber`, is not in the wheel (it is in the sdist, which includes `tools/`
and `docs/`), and makes no network or subprocess calls. A
test enforces those properties. The package states that the evaluation contract must be
frozen before Week 2 harness work. The components here do not depend on that contract
(recording, paired execution, explicit failures, grading hooks, per-item export,
analysis), so they can be reviewed now. The contract-dependent parts are still open: the
real graders, prompts, dataset adapters and inference adapters.

## Layout

| Path | Purpose |
|---|---|
| `tools/wsp/wsp.py` | CLI: `protocol`, `dataset`, `ledger`, `diff`, `verify-edit`, `smoke`, `analyze`, `drift` |
| `wsp_eval/protocol.py` | Arms, inference settings, seeds, grader, analysis plan; canonical `protocol_sha256`; frozen ⇒ no `REPLACE`; replication arms must share inference settings |
| `wsp_eval/dataset.py` | Items with stable IDs and content hashes; duplicate-ID, cross-split leakage and within-split duplicate checks |
| `wsp_eval/generation.py` | Provider-neutral `Generator` protocol (`describe()` reports the real provider/model/revision/engine); `SyntheticGenerator` only |
| `wsp_eval/grading.py` | `Grader` protocol; `None` = explicit "ungraded"; a synthetic final-answer grader, **not** the authors' evaluator |
| `wsp_eval/runner.py` | Paired, interleaved arm execution: one record per (item, sample, arm), adapter/grader/finish failures recorded as error rows, output hashed, raw text opt-in, immutable output dir outside the repo; `records_sha256`, `run_status` (aborted runs still write what exists) and `verify_run` |
| `wsp_eval/analysis.py` | Per-arm counts/accuracy/limit hits/length; item-level paired difference (duplicates raise, unbalanced items excluded), seeded item-cluster bootstrap 95% CI, exact McNemar for single samples; decision rule binding only when preregistered (`preregistration`) |
| `wsp_eval/drift.py` | Reload/drift policy: exact identity checks (pass/fail/not_recorded) separate from stochastic reproducibility (pair categories, item-clustered CIs for accuracy, token-limit and error-rate differences); a tolerance applies only if approved in a frozen protocol before the run |
| `wsp_eval/ledger.py` | Validates evidence-ledger records ([ledger-template.json](ledger-template.json), JSON form of the package's YAML) |
| `wsp_eval/checkpoint_diff.py` | Read-only tensor-level diff of two `.safetensors` files (names, shapes, dtypes, per-tensor SHA-256, changed-element counts, BF16/F16/F32/F64 magnitudes) |
| `wsp_eval/synthetic.py`, `fixtures/` | Scripted synthetic arms and items used by the smoke run and tests |
| `docs/wsp/exp8-replication-protocol.md`, `docs/wsp/protocols/exp8-replication-draft.json` | Draft Experiment 8 replication protocol (unfrozen; drift tolerance and minimum effect unset) |
| `docs/wsp/ledger/` | Ledger records. `exp8-source-pins-2026-11-03.json`: public-metadata pins for Experiment A (status `planned`) |
| `docs/wsp/run-readiness.md` | Decisions, infrastructure and harness work required before a real run |
| `tests/test_wsp_harness.py`, `tests/test_wsp_drift.py`, `tests/test_wsp_review.py` | 78 tests, synthetic fixtures plus validation of the recorded ledger entries and the draft protocol |

```bash
.venv/bin/python tools/wsp/wsp.py smoke --output ~/wsp-research/runs/smoke-$(date +%s)
.venv/bin/python tools/wsp/wsp.py analyze --run <run dir> --protocol tools/wsp/fixtures/synthetic_protocol.json
.venv/bin/python tools/wsp/wsp.py ledger docs/wsp/ledger-template.json
.venv/bin/pytest -q tests/test_wsp_harness.py
```

Exit codes: 0 ok, 1 validation or integrity problem, 2 usage error.

## What was reused from the existing evaluation work

| Existing (`src/tantular_cyber/benchmark`, `docs/evaluation.md`) | Use here |
|---|---|
| `compare.mcnemar_exact` | Reimplemented to keep the harness independent; a test checks it is identical over a 12×12 grid |
| Paired wins/losses, pinned options, provenance before/after (`compare.py`) | Methodology adopted: identical inputs and seeds per arm, settings recorded per arm, provenance in every manifest. Ollama-specific code (`model_provenance`, `/api/show`) not applicable to BF16 safetensors |
| `llm_eval.wilson` | Not used: the package asks for item-clustered intervals with repeated samples, so a paired item-level bootstrap is used instead |
| Pseudonym rule from `review.py` | Same `[A-Za-z0-9_-]{1,32}` pattern for operator, research lead and decider in the ledger |
| `provenance.code_state` (git) | Not reused; the harness avoids subprocesses. It records `harness_code_sha256` instead, and the operator records the commit in the ledger |
| Step 0 lesson (per-sample outputs not kept) | Every request leaves a per-item record with an output hash; raw text can be kept (`keep_outputs`) in research storage |

The Ollama comparison results (quantized Q8_0/Q4_K_M models) are **not** evidence about the BF16 research checkpoint and are not imported.

## Controls register

Legend: **Doc** = written requirement only · **Impl** = enforced by harness code ·
**Test** = covered by a synthetic test · **Op** = needs operator/infrastructure verification
(not done).

| Requirement | Status | Note |
|---|---|---|
| No network, subprocess, pickle/torch, `eval`/`exec` in harness code | Impl + Test | AST import scan. This is a property of *this code*, not a sandbox |
| Harness independent of production package; not shipped in the wheel | Impl + Test | Included in the sdist (`tools/`, `docs/`, `tests/` are listed there); it is not installed |
| No promotion/deploy path | Impl + Test | No such function exists (name scan); verdict text says "pending human review"; manifest `human_decision: pending` |
| Human decision; no self-approval; `accept` only for complete records | Impl + Test | Ledger validation of a *record*; it cannot stop someone editing a file |
| Every request recorded; errors and ungraded outputs never dropped | Impl + Test | Includes grader crashes, invalid adapter returns and contradictory finish flags |
| Run integrity: records hashed, aborted or tampered runs not analyzed | Impl + Test | `verify_run` (CLI `analyze`/`drift`) checks protocol hash, `run_status`, `records_sha256`, run IDs and record count; `outputs_sha256` is recorded but not verified. Protects against accidents, not against someone who rewrites both files |
| Decision rules not chosen after results | Impl + Test | Frozen protocols need `frozen_by`/`frozen_at`; the analysis verdict and drift tolerance are binding only if both predate the run start; otherwise the rule outcome is shown as `unregistered_rule_outcome`. Timestamps are recorded, not authenticated |
| Paired comparison really paired | Impl + Test | Analysis arms must have reference/candidate roles; duplicate records raise; items with different sample sets per arm are excluded and listed |
| Runs immutable; research output outside the repo by default | Impl + Test | Only `allow_in_repo` overrides this, and it exists for throwaway synthetic runs |
| Held-out split only with frozen protocol and named evaluator | Impl + Test | The name is recorded but not authenticated. Restricting who can *read* held-out data is **Op** |
| Raw model output not written unless requested | Impl + Test | |
| Replication arms share exact inference settings | Impl + Test | Tokenizer/engine identity is the adapter's job → **Op** |
| Tensor-level diff before attributing an effect to an edit | Impl + Test | `verify-edit` checks a diff against a published manifest. Tested on synthetic files only; never run on a real checkpoint |
| Drift tolerance never chosen after results | Impl + Test | Unapproved tolerances cannot carry bounds; approval needs a frozen protocol, a named approver and a timestamp no later than the run start. Who approves, and when, is still a human process |
| Reload is a separate load with identical artifacts and prompts | Impl + Test | Depends on a real adapter reporting `load_id`, artifact hashes and rendered-prompt hashes truthfully → **Op** |
| Production credentials and model registry unavailable to controller and sandbox | **Doc** | Requires OS/cloud identities and storage permissions → **Op** |
| Baseline artifacts read-only; candidates in research-only storage | **Doc** | Diff opens files read-only, but storage ACLs are **Op** |
| No network/secrets in the edit sandbox | **Doc** | No sandbox exists yet → **Op** |
| Held-out content and feedback hidden from the controller | **Doc** | No controller integration exists → **Op** |
| Never edit deployed GGUF; separate evaluation before any deployment | **Doc** | No GGUF code in the harness; process control |
| Provider-neutral controller interface, actual provider/model/version/cost recorded | Partly Impl | `Generator.describe()` + ledger `controller` field; no controller adapter exists. GPT-6 Astra and GLM-5.3 access is unverified |

## Review log

2026-11-03 review fixes: pinned revisions and digests are validated at every ledger status
(previously only on `complete`); ledger fields for sources, git blob SHA-1s, licenses, open
questions and verification limits; malformed safetensors shapes/offsets raise
`SafetensorsError`; booleans are rejected where the protocol expects numbers. Known limit:
the no-edit-reload drift check compares output hashes, so GPU sampling non-determinism under
vLLM will show up as drift. The protocol must say whether that is tolerated (and how it is
measured) before a real run.

2026-11-04: `n_per_request` sampling mode (the authors' `SamplingParams(n=4, seed=0)`);
per-arm `engine` settings that replication arms must share; no-edit reload must declare the
reference checkpoint; generations carry `finish_reason` (abnormal finishes and short batches
become error rows instead of aborting) and `rendered_prompt_sha256`; `drift.py` replaces the
earlier hash-only drift check; `verify_edit`; `min_meaningful_effect: null` allowed only in
unfrozen protocols, with a descriptive-only verdict.

2026-11-04 review (synthetic fixtures only; nothing measured):
- *Failure handling.* A grader exception used to abort the whole run with nothing written;
  it is now a `GraderError:<type>` row with the output hash kept. Non-`Generation` returns
  (`InvalidGeneration`) and `hit_token_limit` contradicting `finish_reason`
  (`InconsistentFinish`) are error rows. A harness failure mid-run (bug, interrupt) writes
  the partial records and a manifest with `run_status: "aborted"` before re-raising.
- *Provenance.* The manifest records `records_sha256`, `outputs_sha256`, `run_status` and
  `expected_records`. `analyze`/`drift` go through `verify_run`, which refuses mismatched,
  aborted, truncated or mixed runs. `harness_code_sha256` now covers `wsp.py` too.
  Safetensors headers with duplicate tensor names are rejected (JSON would keep the last one
  silently).
- *Paired evaluation.* Analysis arms must be the reference arm and a candidate/alternative
  arm (not the reload). Duplicate (arm, item, sample) records raise. Items whose arms have
  different sample sets are excluded and listed in `items_unbalanced`.
- *Preregistration.* Previously an **unfrozen** protocol with a `min_meaningful_effect`
  produced a binding verdict (the synthetic smoke run said
  `criteria_met_pending_human_review`). A frozen protocol now requires `frozen_by` and
  `frozen_at`. The verdict is binding only when the protocol was frozen no later than the
  run start; otherwise it is `descriptive_only_not_preregistered`, with the rule outcome in
  `unregistered_rule_outcome`. The drift tolerance uses the same check.
- *Protocol strictness.* `drift.tolerance` rejects unknown keys, so a "proposed" bound
  cannot sit next to an unapproved status.
- *Drift consistency* (second pass, findings 1–3): `identity_report`,
  `reproducibility_report` and `drift_report` reject duplicate records when called directly.
  Items whose reference and reload arms have different sample sets are excluded from the
  drift metrics and listed in `items_unbalanced`, as in the paired analysis. Error rows for
  `AbnormalFinish` and `InconsistentFinish` keep the adapter-reported `finish_reason` and
  `rendered_prompt_sha256`, so the identity check compares them: equal → pass, different →
  fail, absent → `not_recorded`. Their output fields stay null.
- *Still open, not blocking:* `verify_run` does not check `outputs_sha256`; an item missing
  from one arm makes the analysis report `samples_per_item` 0 and skip McNemar (only possible
  through direct calls).
- *Known gaps for real runs* (documented, not fixed): the runner assumes every arm is loaded
  at once and sends one request per (item, arm). The authors batched shards and a GPU cannot
  hold three vLLM instances at 0.85 memory utilization. See
  [run-readiness.md](run-readiness.md) C4.

2026-11-04, post-merge (PR #23 Codex review, P2): `parse_protocol` now requires
`analysis.missing_policy` to be `exclude_item` or `count_as_incorrect`, frozen or not.
Previously a protocol with a missing or misspelled policy was accepted, the run completed,
and only `analyze()` failed, leaving a frozen run unanalyzable. One validator
(`protocol.missing_policy`) is now used by `parse_protocol`, `analyze()` and the drift report.
Each raises `ProtocolError` for a missing or unknown policy, even if the analysis plan was
altered after parsing. Previously the drift report silently defaulted a missing policy to
`exclude_item` (the opposite of the draft protocol's `count_as_incorrect`), and computed an
unknown one as `count_as_incorrect` while reporting the unknown name.

## Readiness (unchanged from the package)

None of the Week 1 readiness items is complete. Roles, budgets, isolation, source and model
pins, dataset access, the frozen protocol, the real smoke test and the go/stop decision are
still open. The synthetic smoke run tests the harness, not the Thursday smoke test, which
needs a real checkpoint on approved infrastructure.

## Blockers

1. **Roles and budget:** research lead, independent evaluator, operator and release owner are
   not assigned. No spending ceiling is set.
2. **Source pins (partly resolved 2026-11-03):** the logs commit, base and edited HF
   revisions, published file hashes and declared licenses are recorded in
   [ledger/exp8-source-pins-2026-11-03.json](ledger/exp8-source-pins-2026-11-03.json)
   from public metadata only. They have not been verified against downloaded files. The
   open questions are listed in that record: third-party rebuild of the multimodal
   checkpoint vs the authors' text-only export, config differences, unpinned dataset
   revisions, unknown original engine versions and seeds, and a pending human license
   review.
3. **Evaluator:** the released grading code, prompts and dataset versions for MATH-500,
   GSM8K and IFEval have not been reviewed. `FinalAnswerGrader` is not a substitute.
4. **Infrastructure:** no isolated research environment, identity or storage exists. The
   inference engine and hardware are undecided.
5. **Cyber 9B / Antares-1B:** origin, license, revision and an editable (non-GGUF) format
   are unconfirmed, so Experiments B and C stay undefined.

## Next concrete steps

1. Owner assigns roles and the spending ceiling (Mon).
2. Done 2026-11-03 (metadata only): see the ledger record above. Next, the research lead
   resolves its open questions, especially which checkpoint and config files define each
   arm, and a human completes the license review.
3. Review the authors' evaluator and write the frozen protocol JSON: real dataset revision,
   prompts, the evaluator as a `Grader` adapter, `missing_policy`, minimum effect and
   bootstrap settings. Then `wsp.py protocol` it with `frozen: true`, `frozen_by` and
   `frozen_at`.
4. On approved isolated infrastructure: write a reviewed inference adapter implementing
   `Generator` and the per-load execution mode ([run-readiness.md](run-readiness.md) C1, C4).
   Run `wsp.py diff` on base and edited safetensors, then a tiny
   development-split run (base, edited and no-edit reload) with the drift check. Record the
   costs and failures in the ledger.
5. Friday: the human go / conditional go / stop decision goes in the ledger, decided by
   someone other than the operator and research lead.
