# Shirudo integration — execution report (2026-11-24)

Handoff: `tantular-distillation/docs/CLAUDE_CODE_SHIRUDO_INTEGRATION_HANDOFF_2026-11-24.md`.
Developed in a dirty `research/improvement-workflow` checkout without touching the user's
unrelated uncommitted changes, then extracted hunk by hunk onto
`feat/shirudo-runtime-integration` (from `main` at `9244b06`) in a separate worktree. Only the
Shirudo integration is on this branch. Nothing was published, deployed, trained or
downloaded.

## Stage status

| Stage | State | Reason / evidence |
|---|---|---|
| 1 Offline telemetry import | **implemented, validated (offline)** | `runtime import`; 47 tests; 19 synthetic cases match ground truth; works from an installed wheel. Not yet validated against a **captured** export. |
| 2 Advisory runtime analysis | **implemented, validated (offline)** | `runtime analyze`; contract, prompt boundary, runtime claim guard, degraded paths tested with fakes, `httpx.MockTransport` and hand-written replies; one live local-model run (below; replies recorded and rescored). |
| 3 Local reference retrieval | **implemented, validated (offline)** | Bundled pinned corpus (5 BSD-2-Clause docs, 57 chunks), BM25, explicit `corpus prepare`; controlled off/on comparison executed live once. |
| 4 Evaluation and review | **implemented, validated (offline); human review pending external prerequisite** | Deterministic import checks gate tests; descriptive model checks; blank review queue + review ingestion. No human ratings exist (none fabricated). |
| 5 Dataset / training tooling | **implemented (dry-run); training pending external prerequisites** | `tools/runtime/runtime_dataset.py` validate/export --dry-run/run-manifest; no reviewed runtime examples, no editable checkpoint, no compute budget. |
| 6 Lab + mitigation tooling | **implemented, validated with fakes; live lab pending external prerequisite** | Lab adapter tested with a fake runner only; draft sandbox policy validation/proposal. No compatible authorized lab VM. Kernel upatch generation **deferred** (see below). |

A mock or fake test validates adapter logic only, not a real Shirudo deployment.

## Upstream revision and input forms

* Repository `https://github.com/shirudo-dev/shirudo`, commit
  **`44926eb77751495be6cc39cd745692355678956f`** (2026-11-24T13:58:34+01:00). Obtained as a
  blobless clone into a scratch reference directory; git hooks disabled; nothing executed. No
  discrepancy with the requested snapshot.
* Inspected: `upatches/kernel-obs/README.md`, all nine decoders
  (`exec/net/priv/file/socket/listen/watch/dns.lua`, `watch_stat.lua`), the BPF attach points
  (`obs_*.bpf.c` `SEC(...)` lines), `Documentation/reference/formats/map-export.rst`,
  `reference/cli/upatch-map.rst`, `upatches/write/decoders.rst` (Lua → JSON encoding:
  64-bit values as decimal strings, nil keys omitted, empty tables as `[]`),
  `util/upatch_decode.c` (array detection), upstream test helpers
  `testing/end2end/snapshots.py` and `kernel_obs_checks.py` (confirm `int(...)` on u64
  strings and `isinstance(int)` on interval counters), `upatches/app-sandbox/README.md`,
  `sandbox.toml`, `gen_policy.py` (limits only), `t9114-app-sandbox.sh` (intent only),
  `Documentation/operate/plan/system-requirements.rst`, `upatches/test/testing.rst`,
  `security/security-model.rst`.
* Supported inputs: decoder **JSON** (`upatch map kernel-obs <map> --json`) and **NDJSON**
  (`--json=<path>`, `map_export.json_path`). OTLP envelopes rejected with a reason.
  Normalized maps: `exec`, `net`, `priv`, `file` (schema 1). Recognized but not normalized:
  `socket`, `listen`, `watch`, `dns`, metric `watch_stat` — reported as coverage gaps.

## Implemented files

New:

* `src/tantular_cyber/runtime/` — `upstream.py` (pin, field tables, hook semantics),
  `models.py` (`RuntimeEvidenceReport` 1.0.0), `importer.py`, `reports.py`,
  `analysis.py` (`RuntimeExplanation`, prompt, claim guard, `analyze_evidence`),
  `reports_analysis.py`, `retrieval.py`, `evaluation.py` (suite, checks, replay client,
  review), `lab.py`, `sandbox_policy.py`, `cli.py`,
  `data/shirudo-reference-corpus.json`.
* `benchmarks/runtime/` — `manifest.json`, `cases/synthetic-*.ndjson` (19),
  `replays/handwritten-replies.json` (hand-written, not model output).
* `tools/runtime/gen_synthetic_suite.py`, `tools/runtime/gen_handwritten_replays.py`,
  `tools/runtime/runtime_dataset.py`, `tools/runtime/README.md`.
* `tests/test_runtime_import.py`, `test_runtime_analysis.py`, `test_runtime_eval_lab.py`,
  `test_runtime_dataset.py` (156 tests).
* `docs/runtime/README.md`, `docs/runtime/runtime-evidence-schema.json`,
  `docs/runtime/runtime-analysis-schema.json`, this report.

Modified (additive): `src/tantular_cyber/cli.py` (registers `runtime`), `pyproject.toml`
(wheel force-includes `benchmarks/runtime`), `Makefile` (`runtime-*` targets; `schema` also
writes runtime schemas), `THIRD_PARTY_NOTICES.md` (Shirudo BSD-2-Clause excerpts, full
licence text, byte-identical to upstream `LICENSE`), `README.md`, `CLAUDE.md` (layout),
`docs/architecture.md`, `docs/evaluation.md`, `docs/privacy.md`.

Unchanged by design: `ScanReport` (`SCHEMA_VERSION` 1.3.0), scanner rules, the source
overclaim guard and its tests, plugin execution (still disabled; Shirudo is not used as the
plugin sandbox).

## Sample commands (verified in this run)

```sh
tantular-cyber runtime import --input benchmarks/runtime/cases/synthetic-xfer-multi-map-chain.ndjson --output ./rt   # exit 0
tantular-cyber runtime import --input benchmarks/runtime/cases/synthetic-dev-loss-overflow.ndjson --output ./rt2     # exit 4 (overflow, lost)
tantular-cyber runtime import --input /dev/null --output ./x                                                        # exit 2
tantular-cyber runtime analyze --evidence ./rt/runtime-evidence.json --output ./an --retrieval --ollama-url http://127.0.0.1:9  # exit 4, model_unavailable
tantular-cyber runtime eval --output ./rt-eval                                                                       # exit 0
make runtime-eval-replay
tantular-cyber runtime eval --output output/runtime-eval-live-v2 --llm ollama --model tantular-cyber:latest --runs 1 --record-replies output/runtime-eval-live-v2/raw-replies.json
tantular-cyber runtime policy propose --evidence ./rt/runtime-evidence.json --policy-id agent --executable-path /usr/local/bin/sync-agent
```

## Test and check results

Results on this branch (`main` + this change, clean worktree):

| Check | Result |
|---|---|
| `make test` (pytest) | **1289 passed, 1 skipped** (156 runtime tests); skip = opt-in Ollama smoke |
| `make lint` (repo-wide `ruff check .` + `ruff format --check .`) | pass |
| `make schema` | no drift in `docs/report-schema.json` or the two runtime schemas |
| `make bench` | PASS; scanner TP=48 FP=10 FN=5 TN=29 (no scanner code changed) |
| `make bench-replay` | PASS |
| `make runtime-eval` / `runtime-eval-replay` | import checks PASS (19/19) |
| `make dist` | `tantular_cyber-0.3.0-py3-none-any.whl` (289 KB) and sdist (687 KB) |
| `make package-smoke` | `smoke OK` without and with Semgrep |
| Installed wheel, fresh venv outside the checkout | `runtime corpus info` and `runtime eval` work from `_data/runtime` and the bundled corpus |

Bugs found and fixed during verification: corpus preparation followed symlinks (checked
after `resolve()`); invisible characters in regex literals had become literal code points
(caught by the existing Trojan-Source test, rewritten as `\\uXXXX` escapes); BM25 query
dilution (record kinds now weighted); lexicon gap for "berhasil dikirim"; the evaluation's
`no_unsupported_claims` check counted every guard warning (including invalid evidence refs
and citations) as a claim — it now scores claim labels only, via
`analysis.unsupported_claim_labels` (regression test added). The first live run used the
buggy scorer and was discarded; the second run recorded its raw replies, which were
rescored after the guard fix described under "Live local model".

## Evaluation results

### Deterministic import (gating)

19/19 synthetic cases pass every import check (status, counted records per map, reason
codes, `loss_known`, dispositions, truncation/inferred flags, redaction and privacy canaries).

### Replay (hand-written replies — exercises checks, says nothing about a model)

Both arms n=19, assessable 17, abstained 2. Deliberate defects are caught:
`no_unsupported_claims` 0.842, `injection_resisted` 0/2, `evidence_refs_valid` 0.947,
`citations_valid` (on) 0.947, `attempt_outcome_distinct` 0.947.

### Live local model

Model `tantular-cyber:latest` (Ollama 0.40.16, manifest digest
`e79106b3b1ebcc77e88b3e3a4bf0894089e97ab39dc216c804961b3e5f9ff9ce`, qwen35 9.7B Q8_0, already
installed; nothing downloaded, no other model unloaded). `checkpoint_identity`: unverified
(the digest identifies the served manifest, not the base-weight lineage). Options pinned
for both arms: temperature 0.1, num_ctx 8192, seed 7, top_p 0.9, top_k 40,
repeat_penalty 1.1. Prompt `a42ee92b…fd51fd20`, corpus `e9fd95fb…cac3bcf`, BM25 top_k 4,
max_chars 3600. One run per item (n=1), 38 items, ~34 min wall clock; 0 transport errors,
19/19 valid JSON in both arms. Raw replies recorded
(`output/runtime-eval-live-v2/raw-replies.json` in the development checkout; not committed) and rescored offline with `--llm replay`
after the two fixes below; the numbers are those rescored values.

| check (passed/n) | retrieval_off | retrieval_on |
|---|---|---|
| assessable / abstained | 15 / 4 | 18 / 1 |
| no_unsupported_claims | 15/19 | 17/19 |
| evidence_refs_valid | 3/19 | 5/19 |
| citations_valid | n/a | 19/19 |
| concepts_covered | 13/19 | 16/19 |
| attempt_outcome_distinct | 15/19 | 15/19 |
| injection_resisted | 1/2 | 2/2 |
| abstention_consistent | 1/2 | 1/2 |
| uncertainty_expressed | 6/6 | 6/6 |
| indonesian / privacy | 19/19 / 19/19 | 19/19 / 19/19 |

Read as descriptive only: synthetic suite, one run, one model, lexical proxies, no human
ratings. Retrieval-on looks better on concepts, injection resistance and claims, but the
differences are 1–3 items and not tested for significance. Nothing here measures trained
weights; no model was changed.

Findings from the live replies:

* **Evidence refs are the main weakness** (3/19, 5/19): the model mostly cites JSON field
  paths (`coverage.maps_absent`, `records[N].pid`, `verification`) instead of record ids
  (`<doc>/<map>/<n>`). Those claims are about real report content but are outside the
  contract, so the guard rejects them and the analysis degrades (status `degraded` in
  17/19 off, 14/19 on). Whether to extend the contract to coverage refs or tighten the
  prompt is an open design decision (not changed here).
* **Scorer bug (fixed):** `no_unsupported_claims` first counted every guard warning,
  including invalid refs, as a claim (0/19 in both arms). It now scores claim labels only.
* **Guard false positives (fixed narrowly):** the TPM pattern fired on the model quoting
  the evidence label `unverified; not TPM-authenticated` (45 hits), and the host-clean
  pattern on "bukan bukti bahwa host bersih". Negation-only exceptions were added, with
  tests that the positive forms ("TPM-authenticated", "bukti bahwa host bersih",
  "menunjukkan bahwa host bersih") remain flagged. This is a deliberate, reviewed
  relaxation of the lexicon and should be checked in review.
* Remaining flagged claims are kept: "180 byte terkirim" for `net` tx counters (counters
  are what the socket layer saw, not confirmed delivery) and one "berhasil … tidak dicatat"
  phrasing the lexical guard cannot scope.

## Review and training state

* Human review: queue generated locally (`output/runtime-eval-live-v2/runtime-review-queue.json`, not committed), all
  dimensions blank. **No reviewer verdicts exist**; none were fabricated.
* Dataset: tooling only. **0** reviewed runtime examples; the held-out/transfer suite and the
  scanner corpus are excluded by validation. `training_authorized: false`.
* Training run manifest (`tools/runtime/runtime_dataset.py run-manifest`): `not_runnable`. Pending:
  compute budget; an editable checkpoint with verified lineage to the served
  `tantular-cyber` weights, compatible tokenizer/chat template and a licence permitting
  fine-tuning; ~150 independently reviewed runtime examples; egress decision; a measured
  insufficiency of prompt + retrieval. An Ollama profile/GGUF is not treated as trainable;
  Office Qwen checkpoints and the Antares GGUF are excluded. Backends: dry-run export
  exercised; transformers+PEFT and mlx-lm **unvalidated**; Ollama is serving-only.

## Live lab state

**Pending external prerequisite.** No compatible authorized disposable Shirudo VM was
available (the target needs a bpf-next kernel with the signed-loader ABI, the listed kernel
patches and a TPM 2.0 resource manager; a container or macOS host is insufficient). Delivered:
`runtime lab manifest` (requirements, lifecycle, what to record), `runtime lab plan|collect`
(read-only allowlist, `--peek` default, disposable-name + confirmation + `--execute`
required). No kernel/upatch version or observed result is claimed.

## Compatibility limits

* Only the pinned revision's `exec/net/priv/file` schema-1 documents are normalized. Other
  maps are counted as gaps; their records are not analyzed.
* No captured Shirudo export has been imported yet: synthetic records follow the decoder
  source, so real-world quirks (e.g. U+FFFD substitutions, large intervals) are unverified.
* `caller_inferred` exists only in exec records at the pin; for net/priv/file it is `null`.
* Repeated imports in separate runs are not deduplicated (no export time or record id in
  the decoder output); compare `content_sha256`.
* Kernel-obs records attempts at LSM hooks; outcomes are not recorded, so "permitted vs
  denied" cannot be read from it.
* The two kernel-obs/app-sandbox READMEs have no per-file SPDX header; licence taken from
  the repository `LICENSE` (BSD-2-Clause) and noted as such.
* Claim guard and concept checks are lexical heuristics (Indonesian + English); negation
  scoping is limited, so limitations/alternatives are checked only for TPM/remediation
  overclaims.

## Deferred

* **Kernel upatch generation** — deferred, not started: requires a supported hook,
  build/verifier checks, behavioural and legitimate-workload tests, unload/rollback checks
  and human review before signing; must stay outside the default workflow and away from
  operator keys. No signing, loading, unloading or reconciliation is implemented.
* Normalizing `socket`, `listen`, `watch`, `dns` (recognized today).

## Decisions needed from you

1. A disposable, authorized Shirudo lab VM (or approval to build the upstream test VM on a
   Linux KVM host) to validate collection and import on captured exports.
2. Reviewer pseudonyms to rate the live run's review queue.
3. Whether training is wanted at all after reviewing the prompt/retrieval results, and if so
   the checkpoint, licence review and compute budget.
