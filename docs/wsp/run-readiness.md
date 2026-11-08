# Real-run readiness: Experiment 8 MATH-500 replication

Status 2026-11-04: **not ready.** Nothing in this document authorizes downloads, model runs,
spending or production changes. It lists what has to be decided, built and verified before the
first real (non-synthetic) run of [exp8-replication-draft.json](protocols/exp8-replication-draft.json),
in the order D1 requires ([exp8-replication-protocol.md](exp8-replication-protocol.md) §0, §9).
Nothing has been measured; the harness has only processed synthetic fixtures.

## A. Decisions (human; none made except D1)

| # | Decision | Owner | Where it is recorded | Must precede |
|---|---|---|---|---|
| R1 | Roles: research lead, independent evaluator, operator, release owner, and who approves the run. All are pseudonyms, and the decider must not be the operator or research lead | research owner | ledger `operator`, `research_lead`, `decided_by`; protocol `frozen_by`; tolerance `approved_by` | everything below |
| R2 | Spending ceiling, and whether the proposed `text_only_reconstruction` arm is funded (it needs its own protocol decision) | research owner | ledger `cost_and_compute`; protocol `decisions` | R3 |
| R3 | Isolated infrastructure: provider, account and identity (section B) | research owner, operator | ledger `environment` | R4 |
| R4 | Permission to download the two pinned checkpoints and MATH-500 at the R8 revision into research storage | research owner | protocol `status_notes.downloads_authorized`; ledger | any download |
| R5 | Human license review: the logs repository's custom license (CC BY 4.0 / MIT / GPL-3.0 parts), Qwen Apache-2.0, and the dataset license | named human | ledger `licenses_reviewed`, `licenses` | R4 |
| R6 | Engine: vLLM and transformers versions, and how the release's missing `generation_config.json` is handled (explicit sampling parameters only, or engine defaults) | research lead | protocol `arms[].engine` | R10 |
| R7 | Hardware, tensor-parallel size, `gpu_memory_utilization`, and whether the authors' 10-shard batching (`rows[i::10]`, one generate call per shard) is reproduced (see C4) | research lead | protocol `arms[].engine` | R10 |
| R8 | MATH-500 revision, content hash, and what happens if the IDs differ from the authors' 500 | research lead | protocol `dataset` | R10 |
| R9 | Grader: a reviewed adapter of `exp8_math/code/mathqa.py`, the sympy pin, and the size of a human grading sample | evaluator | protocol `grader` | R10 |
| R10 | `min_meaningful_effect`, `agreement_with_published` and the three drift bounds, chosen after R6–R9 and before any confirmatory result | evaluator | protocol `analysis`, `drift.tolerance` (`approved_by`, `approved_at`) | R11 |
| R11 | Freeze: `frozen: true` with `frozen_by` and `frozen_at`; the protocol SHA-256 is copied into the ledger | evaluator + research lead | protocol; ledger `protocol_sha256` | confirmatory run |
| R12 | Split labelling. *Proposed, not decided:* MATH-500 items are `heldout` for the confirmatory run (this requires a frozen protocol and a named evaluator). Any pilot uses non-MATH-500 `development` items, and its use is recorded | evaluator | dataset file; run `heldout_authorized_by` | R10 pilot |
| R13 | Whether raw outputs are kept (`keep_outputs`, needed to audit grading and draw the human sample) and where they are stored | research owner | run manifest `outputs_kept`; ledger `output_location_and_hash` | first run |

For sizing only (this is not a cost estimate): 3 arms × 500 items × 4 samples = 6,000
generations. At the 32,768-token cap that is at most about 197 million output tokens.
Real lengths are unknown until measured.

## B. Infrastructure (operator; none exists, nothing verified)

1. **An isolated compute account and host** with a research-only identity. Production
   credentials, the production model registry and deployed GGUF models must be unreachable
   from it. This is checked by the operator, not by the harness.
2. **Research storage outside the repository** for checkpoints, the dataset file and run
   directories. Run directories are write-once (the harness refuses a non-empty directory).
   Access to held-out items and raw outputs is restricted to the named roles.
3. **Network:** downloads only in an approved fetch step, from the pinned revisions, with the
   SHA-256 checked against the ledger. Runs execute with the engine in offline mode and no
   outbound network.
4. **A pinned runtime image** (digest recorded in the ledger `environment`) containing the
   R6 engine stack and the R9 grading stack (math-verify 0.9.0, latex2sympy2_extended 1.11.0,
   antlr4-python3-runtime 4.13.2, and the R9 sympy pin).
5. **Cost metering** tied to the R2 ceiling, with a hard stop.

## C. Harness work still needed (code; not written)

Each item needs review, and each new adapter needs synthetic tests before it touches a real
artifact.

1. **Inference adapter.** A `Generator` that serves `n_per_request` requests and returns
   exactly `n` `Generation`s, each with `finish_reason`, output token count and
   `rendered_prompt_sha256` (hashed from the chat-templated prompt). `describe()` must report
   artifact SHA-256s computed from the files actually loaded, plus a distinct `load_id` per
   load.
   *Decision needed:* the isolation test forbids `torch`, network and subprocess imports
   under `tools/wsp`. The adapter therefore has to live in a separate, reviewed location that
   runs only on the isolated host, unless that control is deliberately changed.
   *Status 2026-11-08:* written in `tools/wsp_runtime` (separate from `tools/wsp`, heavy
   imports lazy) and tested with a stub engine only; see [runtime.md](runtime.md) for what
   still needs a real GPU, especially engine shutdown, worker-process exit and device-level
   GPU-memory release, and the effective engine settings. It requires complete, typed and
   pinned settings, a `checkpoint.loadable_files_sha256` registration for every loadable file
   (which the Experiment 8 draft protocol does not have yet; recorded after the approved
   fetch), and checkpoints on a read-only mount below root-owned, non-writable parents; what
   the local storage checks cannot establish (privileged remounts, overlays, writable aliases)
   stays an explicit operator verification. Running with effective uid 0 is refused, and both
   version sources (package metadata and module `__version__`) are reconciled in preflight.
   *Per-arm processes:* each arm now runs in its own `spawn` worker process, as the readiness
   proposal specifies (`wsp_runtime/process_worker.py`), tested with stub workers only. A
   worker that exited does not by itself show that every GPU resource was released;
   device-level checks remain required on the GPU host.
2. **Grader adapter** for math-verify, following `mathqa.py`: last `\boxed{}`/`\fbox{}`, no
   box means incorrect, 5 s timeouts. Its `id` and `revision` must match the protocol. Test it
   on synthetic strings only, never on MATH-500 content in the repo.
   *Status 2026-11-08:* written (`wsp_runtime/grader.py`), tested on synthetic strings with a
   stub math-verify. **Unreviewed:** twelve interpretations are listed in
   [runtime.md](runtime.md) for comparison with the released `mathqa.py`, and it stays
   unreviewed until that comparison is done and the independent evaluator has accepted it
   (R9). Until then the runtime CLI refuses held-out execution.
3. **Dataset loader:** a local MATH-500 file at the R8 revision converted to `Item` rows with
   `unique_id` as the ID, then a content-hash and ID check. No network.
   *Status 2026-11-08:* written (`wsp_runtime/datasets.py`) for MATH-500 and the MATH training
   split (smoke items), tested on synthetic files only, and wired into the runtime CLI. The
   source and the leakage reference are bound to the protocol's `dataset.runtime_sources`
   (raw file hash, row count, ID-set hash, selection rule, converted-items hash; the
   Experiment 8 draft does not have it yet); the whole source is checked against the whole
   reference before the selection rule (normalised problem text, answers ignored; this cannot
   prove disjointness).
4. **Execution order and batching.** `run_paired` currently interleaves arms per item and
   sends one request per (item, arm). That assumes every arm's model is loaded at the same
   time. Three vLLM instances at the authors' `gpu_memory_utilization` 0.85 cannot share one
   GPU. The authors also batched a whole shard into one generate call; batch composition can
   change floating-point results, and therefore outputs, even with per-request seeds. Whether
   that matters here has not been measured.
   *Needed:* a sequential-per-arm (or per-load) mode, with a batch request type if R7 keeps
   the shard batching. Pairing by (item, sample) is unaffected; the drift identity checks
   already expect separate loads.
   *Status 2026-11-07:* the harness side is implemented and tested with stub engines only
   (`execution: per_arm_sequential` with `shards`, `ArmLoader`, `BatchGenerator`; see
   [harness.md](harness.md)). The real inference adapter (C1) that implements these
   interfaces is not written, and no protocol in this repository declares the mode yet
   (adding it to the Experiment 8 draft is part of R7).
5. **Checkpoint checks** on the downloaded files: `wsp.py diff` / `verify-edit` against
   `exp8_math/runs/p50c/edit/manifest.json`, file SHA-256 against the ledger, and a tokenizer
   and chat-template comparison (protocol §2). These have run on synthetic files only.
6. **Ledger run record** for the confirmatory run: `output_location_and_hash` carries the
   manifest's `records_sha256`, and `protocol_sha256` and `harness_code_sha256` are copied
   from the manifest.

## D. Gate order

1. R1 → R2 → R3 → R5 → R4 (licenses are reviewed before downloads).
2. Build and review C1–C4 with synthetic tests.
3. R6–R9, then an approved fetch and the C5 checks. Any identity failure stops here.
4. An optional, recorded non-MATH-500 pilot (R12), then R10 and R11.
5. Confirmatory run: `wsp.py analyze` and `wsp.py drift`, both of which verify the protocol
   hash, `run_status: complete`, `records_sha256`, run IDs and the record count.
6. Ledger record. The human decision is made by someone other than the operator and the
   research lead.

## E. What the harness checks vs what it relies on

| Checked by code (synthetic tests) | Relied on (operator or adapter honesty; not verifiable here) |
|---|---|
| A frozen protocol has no placeholders and names `frozen_by` / `frozen_at`; analysis and drift rules are binding only if those predate the run start | That the timestamps and pseudonyms are truthful (they are recorded, not authenticated) |
| Records are hashed and complete; `wsp.py analyze`/`drift` (`verify_run`) refuse aborted, truncated, mixed or protocol-mismatched runs. Calling `analyze()` / `drift_report()` directly skips these checks, and the `outputs.jsonl` hash is recorded but not yet verified | That the run directory sits in access-controlled storage, and that no one rewrites both the records and the manifest |
| Every request yields a row; adapter, grader, abnormal-finish, sample-count and inconsistent-finish failures are error rows | That the adapter reports finish reasons, token counts and artifact hashes truthfully |
| Pairing by (item, sample) in both the analysis and the drift report; duplicates raise; unbalanced items are excluded and listed. Abnormal or inconsistent finishes keep the reported prompt hash for the identity check | That the engine received the prompt whose hash the adapter reports |
| Edit verification against a manifest; duplicate safetensors header keys are rejected | That the downloaded files are the pinned ones (checked once the C5 step runs) |
