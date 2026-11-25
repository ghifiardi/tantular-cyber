# Runtime telemetry (Shirudo kernel-obs) — offline import and advisory analysis

Tantular Cyber can import host runtime telemetry exported by
[Shirudo](https://github.com/shirudo-dev/shirudo)'s `kernel-obs` upatch, turn it into a
versioned **runtime-evidence** report, and — only when you ask — have a local model explain
it in Bahasa Indonesia. This is separate from the repository scanner: runtime evidence has its
own contract and never changes `report.json` (`ScanReport` 1.3.0 is unchanged), and telemetry
does not add CWE findings or expand scanner coverage.

| Property | Behaviour |
|---|---|
| Needs | nothing beyond the wheel: no Shirudo, root, VM, network, Semgrep or model |
| Never | loads BPF, drains maps, runs Lua decoders, runs plugins, opens a host path from a record |
| Evidence | `verified: false`, `status: unverified`, `tpm_authenticated: false` — always |
| Model | opt-in `runtime analyze`; output is advisory and stored in a separate file |

## Compatibility target

Pinned upstream revision: **`44926eb77751495be6cc39cd745692355678956f`** (2026-11-24),
recorded in `src/tantular_cyber/runtime/upstream.py`. The importer was written from the
decoders' output (`upatches/kernel-obs/decoders/*.lua`) and the export docs
(`Documentation/reference/formats/map-export.rst`, `reference/cli/upatch-map.rst`) at that
commit. Upstream code was read, not executed and not copied.

| Map | At the pin | Tantular |
|---|---|---|
| `exec`, `net`, `priv`, `file` | log decoders, `schema = 1` | **normalized** |
| `socket`, `listen`, `watch`, `dns` | log decoders, `schema = 1` | recognized → `recognized_unsupported_map` (counted, degrades) |
| `watch_stat` | metric decoder (list of OTLP metrics) | recognized → `metric_document` |
| other upatch / other schema / unknown map | — | `unsupported_upatch` / `unsupported_schema` / `unknown_map` |

Changing the target needs a new pinned revision, updated `upstream.py` field tables, new
fixtures, and a row in the table below in the same change.

| Revision | Date | Status |
|---|---|---|
| `44926eb7…` | 2026-11-24 | supported (offline fixtures only; **no captured trace validated yet**) |

### Input forms

* **`json`** — one decoder document: `shirudo upatch map kernel-obs exec --json [--peek]`.
* **`ndjson`** — one decoder document per line: `--json=<path>`, or `map_export.json_path`
  written by `shirudo-map-export.timer` (`upatch map --all`).

The decoder document is `{"schema": 1, "upatch": "kernel-obs", "map", "interval":
{since_unix_nano, used, overflow, lost, peek}, "records": [...]}`. 64-bit values are
**decimal strings** (Lua cannot hold them exactly) and stay strings in Tantular — timestamps
are never converted to floats. Keys whose value is nil are absent (`interval` when there were
no pages, timestamps of 0, `caller` when unknown, `address` for other families).

OTLP log records (`--otel`) and S3 snapshot references are **rejected with a reason**
(`otlp_envelope_unsupported_export_with_--json`); export with `--json` instead.

## Commands

```sh
# Stage 1 — offline import (macOS/Linux, no Shirudo needed)
tantular-cyber runtime import --input kernel-obs.ndjson --output ./rt [--path-mode masked_user]
#   -> rt/runtime-evidence.json + rt/runtime-evidence.md

# Stage 2/3 — opt-in advisory interpretation by a LOCAL model, optionally with retrieval
tantular-cyber runtime analyze --evidence rt/runtime-evidence.json --output ./rt \
    [--retrieval] [--model tantular-cyber:latest]
#   -> rt/runtime-analysis.json + rt/runtime-analysis.md  (evidence file untouched)

# Reference corpus: inspect the bundled one, or prepare one from a LOCAL checkout
tantular-cyber runtime corpus info
tantular-cyber runtime corpus prepare --checkout ~/src/shirudo --output corpus.json \
    [--revision 44926eb77751495be6cc39cd745692355678956f]

# Stage 4 — evaluation (synthetic suite) and human review
tantular-cyber runtime eval --output ./rt-eval                       # deterministic, gates
tantular-cyber runtime eval --output ./rt-eval-replay --llm replay \
    --replay-file benchmarks/runtime/replays/handwritten-replies.json
tantular-cyber runtime eval --output ./rt-eval-live --llm ollama --runs 3   # opt-in
# keep raw replies (runs=1) so the same run can be rescored later with --llm replay
tantular-cyber runtime eval --output ./rt-eval-live --llm ollama --runs 1 \
    --record-replies ./rt-eval-live/raw-replies.json
tantular-cyber runtime review template --queue rt-eval-live/runtime-review-queue.json \
    --reviewer rev_a --output rev_a.json
tantular-cyber runtime review summarize --queue rt-eval-live/runtime-review-queue.json \
    --reviews rev_a.json rev_b.json --output review-summary.json

# Stage 6 — lab adapter and draft sandbox policies
tantular-cyber runtime lab manifest
tantular-cyber runtime lab collect --name lab-shirudo-1 --host 10.0.2.15 \
    --i-confirm-disposable-lab-vm [--execute --output ./lab-run]
tantular-cyber runtime policy propose --evidence rt/runtime-evidence.json \
    --policy-id apache --executable-path /usr/sbin/apache2 --output draft-sandbox.toml
tantular-cyber runtime policy validate --policy draft-sandbox.toml --declared apache

tantular-cyber runtime schema evidence|analysis
```

Make targets: `make runtime-eval`, `make runtime-eval-replay`, `make runtime-eval-live`,
`make runtime-suite` (regenerate fixtures), `make schema` (also writes both runtime schemas).

### Exit codes

| Code | `runtime import` | `runtime analyze` | `runtime eval` |
|---|---|---|---|
| 0 | complete | analysis and evidence complete | deterministic import checks passed |
| 2 | unsafe/unreadable/oversized input, bad output path | unreadable evidence | bad arguments / corpus |
| 3 | failed: nothing importable (reports still written) | — | — |
| 4 | degraded (see `status_reasons`) | model unavailable/invalid output/rejected refs, **or** evidence not complete | — |
| 5 | — | — | an import check failed |

## Status semantics (Stage 1)

* **complete** — at least one supported document counted, every counted interval has full
  metadata with `overflow = 0`, `lost = 0`, drained (not peek), no rejected
  documents/records, no unsupported documents, no schema drift, at least one record.
* **degraded** — any of: `rejected_documents`, `rejected_records`, `interval_overflow`,
  `interval_lost`, `missing_metadata` (coverage **unknown**, not zero loss;
  `coverage.loss_known = false`), `peek_only`, `identity_conflict`, `schema_drift`,
  `recognized_unsupported_map` / `unsupported_*` / `unknown_map` / `metric_document`,
  `record_limit`, `report_records_omitted`, `no_records` (no observations is **never**
  evidence of a clean host).
* **failed** — no supported document could be imported (empty, garbage, only unsupported).

`complete` describes the import, not the host: records are per-interval aggregates of
attempts at LSM hooks; the absence of a record does not mean the absence of activity.

## Duplicate and peek exports

The decoder output carries: upatch, map, schema, `interval.since_unix_nano`,
`interval.peek`, and its content. It does **not** carry the export time or global record ids.

* Exact repeats (same canonical SHA-256) → `duplicate`, counted once.
* A drain of `(upatch, map, schema, since_unix_nano)` supersedes peeks of it → `superseded_peek`.
* Peeks without a drain: the one with the largest `used` is counted and the import degrades
  (`peek_only`: partial, may reappear in a later drain).
* Two different drains of the same interval start → the first counts, the rest are
  `conflict` and the import degrades.
* Documents without `since_unix_nano` dedupe only by exact content. Separate import runs are
  not deduplicated against each other; compare `intervals[].content_sha256`.

## Bounds

64 MiB per input file, 16 MiB per NDJSON line, 10 000 documents, 200 000 records, nesting
depth 8, decoder string limits (`name` 32, paths 128, `priv.detail` 96, address 39 chars),
20 000 records listed per report (aggregates always cover all), 64 MiB output. Inputs must be
regular files (symlinks, FIFOs, devices refused; opened non-blocking with `O_NOFOLLOW`).
Strict UTF-8, no duplicate keys, no `NaN`/`Infinity`.

## Runtime evidence schema

`docs/runtime/runtime-evidence-schema.json` (`runtime_schema_version` 1.0.0,
`report_type: tantular-cyber.runtime-evidence`). Main parts:

* `upstream` — pinned project/revision and supported maps.
* `input` — sanitized base name, SHA-256, size, form, line and document counts. **Raw
  telemetry is not persisted.**
* `status`, `status_reasons`, `errors`, `verification` (always unverified).
* `privacy` — `path_mode` and notes.
* `coverage` — per map: documents and dispositions, records, rejected, overflow/lost totals,
  intervals missing metadata, peek-only intervals, exact `event_count_total` (decimal
  string), truncated / at-capacity paths, inferred callers, unknown fields; `maps_absent`,
  `unsupported`, `rejected_documents`, `loss_known`, `dedup_notes`.
* `intervals[]` — one per document: `document_ref`, interval metadata, `identity_key`,
  `content_sha256`, `disposition`.
* `records[]` — `ref` (`<document_ref>/<map>/<entry_index>`, used by model citations),
  executable `{dev, ino}`, `cgroup_id`, `pid`, first/last seen (ns strings), `count`
  (per-interval delta), `hook` semantics, `outcome: not_recorded`, `caller_inferred` (exec
  only; `null` = not exported by that decoder), and map-specific `fields`.
* `hook_semantics`, `interpretation_notes` (Indonesian, constant).

### Host-path privacy

Runtime paths are absolute **host** evidence, not repository paths, so they are not forced
into the repo-relative rule of `report.json`. Each is a `HostPath`: `display` per
`--path-mode`, unsalted `sha256` of the exported string, `truncated_by_decoder`,
`at_capacity`, `redactions`.

| `--path-mode` | `display` |
|---|---|
| `masked_user` (default) | `/home/<user>/…`, `/Users/<user>/…`; secrets redacted |
| `full` | redacted only |
| `basename` | last component |
| `hash_only` | `null` |

Executable names, `priv.detail` and paths are control/bidi-stripped and redacted before they
reach the report, the model or logs. Peer addresses, uids, pids and cgroup ids are kept.

## Advisory analysis (Stage 2)

`runtime-analysis.json` (`docs/runtime/runtime-analysis-schema.json`) references the evidence
by SHA-256 and copies its status; it cannot alter it. Contract (`RuntimeExplanation`):
`summary`, `observations[{text, evidence_refs}]`, `conditional_interpretations[{text,
conditions, evidence_refs}]`, `benign_alternatives`, `limitations`,
`recommended_investigation`, `mitigation_suggestions`, `citations`, `confidence`, `abstain`.

* Constant system prompt (`RUNTIME_SYSTEM_PROMPT`, hashed as `prompt_sha256`); evidence and
  reference chunks only inside the escaped `<untrusted_data>` JSON; at most 40 records
  (privilege, file, net, then exec; larger counts first).
* Output is parsed, bounded, sanitized and redacted. `evidence_refs` must name supplied
  records, `citations` must name supplied chunks; rejected ones are listed and degrade.
* **Runtime claim guard** (`runtime_claims`, lexicon in `runtime/analysis.py`, tests in
  `tests/test_runtime_analysis.py`): compromise / exfiltration / confirmed transfer /
  execution-or-mapping success / operation outcome / process ancestry / source location /
  "host clean" / TPM or remediation overclaims. Hits in observations move to conditional
  interpretations; certainty in the summary is flagged; CVE ids are stripped; mitigation
  items that claim a vulnerability is closed are removed; confidence drops to `low`. The
  source-scanner overclaim guard is unchanged.
* Transport errors, missing/remote models, invalid JSON → `degraded`, never `complete`.

## Reference retrieval (Stage 3)

* Bundled corpus `src/tantular_cyber/runtime/data/shirudo-reference-corpus.json`: 5
  documentation files at the pinned commit (BSD-2-Clause, see `THIRD_PARTY_NOTICES.md`),
  57 chunks, each with stable `chunk_id`
  (`shirudo@<rev12>:<path>#<section-slug>-<n>`), section, SHA-256; documents carry URL,
  SHA-256 and licence; the corpus has `corpus_sha256` and is verified on load.
* `corpus prepare` is the only way to build one: explicit, local checkout, reads `.git/HEAD`
  as text (never runs git), documentation files only.
* BM25 (k1 1.2, b 0.75), query from evidence **structure** (maps, kinds, status reasons;
  never from telemetry strings), top-k 4, 3600 characters total. Chunk ids, corpus hash,
  parameters and query terms are recorded in each analysis.
* Retrieved text explains mechanisms; it is never evidence of host activity and is treated as
  untrusted (a hostile chunk is part of the eval suite).

## Lab and mitigation tooling (Stage 6)

* `runtime lab manifest` prints the pinned target requirements (bpf-next signed-loader ABI,
  required commits and kernel patches, TPM 2.0 `/dev/tpmrm0`, preemption model, `shirudo
  init --check-only`), the upstream test-VM lifecycle, and what to record per run. A normal
  container or distribution kernel is not sufficient.
* `runtime lab plan|collect` builds `ssh` argument lists for an allowlist of read-only
  commands (`uname`, `/proc/version`, `shirudo-runtime status --json`, `upatch list`,
  `upatch map kernel-obs <map> --json --peek`). The target name must start with
  `lab-`/`test-`/`vm-`/`disposable-` and `--i-confirm-disposable-lab-vm` is required;
  `--execute` is required to run anything. load/unload/sign/push/reconcile/OTLP/shell words
  are refused. **Live validation is pending** (no authorized lab VM was available).
* `runtime policy propose|validate` writes/validates **draft** `app-sandbox` `sandbox.toml`
  proposals (format and capacities of the pinned README: 32 exec sandboxes, 128 networks, 64
  patterns, one `*`, ≤ 32 bytes per side) and explains allowed operations and residual gaps:
  name-based file rules and pre-opened descriptors, IPv4/IPv6 `connect`/`sendmsg` only,
  replaced executables lose labels, restart needed, not a boundary against root, not an
  agent sandbox, not a substitute for patching. Proposals match executables by `{dev, ino}`
  of exec records with the given path hash, never by name alone. Tantular never builds,
  signs, publishes or loads them; upstream `t9114` outcomes are not claimed.
* **Kernel upatch generation is deferred** and is not part of any default workflow. It would
  need a supported hook, build/verifier checks, behavioural and legitimate-workload tests,
  unload/rollback checks and human review before any signing; operator keys are never given
  to a model.
* Shirudo is **not** used as the third-party plugin sandbox; plugin execution stays disabled.

See [evaluation](../evaluation.md#runtime-telemetry-evaluation) for the runtime suite and
[the execution report](../SHIRUDO_INTEGRATION_EXECUTION_REPORT.md) for verification status.
