# Runtime telemetry tooling

Development-checkout tools for the Shirudo runtime integration ([docs/runtime/README.md](../../docs/runtime/README.md)).
This directory is outside the wheel. Nothing here downloads, calls a model, trains, or runs
Shirudo code.

* `gen_synthetic_suite.py` — regenerates the synthetic suite in `benchmarks/runtime`
  (deterministic; `make runtime-suite`).
* `gen_handwritten_replays.py` — regenerates the **hand-written** replay replies used by
  `make runtime-eval-replay`.
* `runtime_dataset.py` — validation and dry-run export for runtime-explanation examples.

## Runtime explanation dataset (`runtime_dataset.py`)

Each example stores the imported `RuntimeEvidenceReport`, retrieved chunk ids + corpus
SHA-256, the target (`RuntimeExplanation`), scenario family, split, licence, author/reviewer
pseudonyms, `review_state`, generator commit and prompt version. `input_sha256` must equal the
digest of the serving `build_runtime_messages` output.

```sh
PYTHONPATH=src .venv/bin/python tools/runtime/runtime_dataset.py validate EX.json --manifest NEW.json
PYTHONPATH=src .venv/bin/python tools/runtime/runtime_dataset.py export NEW.json --out NEW.jsonl --dry-run
PYTHONPATH=src .venv/bin/python tools/runtime/runtime_dataset.py run-manifest --out RUN.json
```

Validation rejects family/split crossings, duplicate targets, identical inputs across splits,
held-out/transfer runtime-suite families or input hashes, scanner-corpus case ids, claim-guard
hits, invalid refs/citations, canaries, invisible characters and copied material without
attribution. Only `reviewed` train records with passing human review reach the dry-run export;
nothing is trained. `run-manifest` records the unresolved prerequisites (editable checkpoint
with verified lineage, tokenizer, licence, compute budget, reviewed data, measured
insufficiency of prompt/retrieval) and which backends are unvalidated. `training_authorized`
is always `false`. Outputs never overwrite an existing file; CLI errors omit payloads.
