# Evaluation workflows (POC-005)

Three opt-in workflows complement the offline benchmark (`docs/benchmark.md`). None of them
runs in CI, none uploads code, and none of their results gate a build.

## 1. Real-world pointers (`benchmarks/realworld/manifest.json`)

The manifest stores **pointers, not code**: repository, full commit SHA, path, line range,
a SHA-256 of the target line, the git blob SHA-1 of the file, the upstream license (from the
GitHub API), the advisory, a verification status and `development_use` (set if the pointer
influenced our rules — it is then *not* a held-out measurement).

```bash
tantular-cyber realworld audit                 # license classes, no network
tantular-cyber realworld fetch                 # explicit opt-in: downloads the pointed-to files
tantular-cyber realworld evaluate              # offline, from the cache
tantular-cyber benchmark --output out/b --real-world   # adds a separate real-world section
tantular-cyber realworld clean                 # deletes the cache
```

* Fetch downloads one file per pointer from `raw.githubusercontent.com/<owner>/<repo>/<sha>/
  <path>` into `~/.cache/tantular-cyber/realworld` (outside the repository; `--cache` to
  change; `.realworld-cache/` is git-ignored if you keep it in the repo). A file whose git
  blob hash differs is **not stored**. Nothing is executed.
* Evaluation checks the blob hash and the target-line hash (`anchor_moved` if the line no
  longer matches), scans with Semgrep + tantular-callgraph (no LLM) and records whether a
  finding of the labelled CWE overlaps the labelled lines.
* Upstream advisory locations are **hypotheses**: `verification.status` stays
  `upstream_label` until a reviewer (pseudonym) independently confirms the location.
  Results are descriptive counts — never precision/recall or real-world performance.

Missed pointers carry a `coverage_candidate` note describing the sink our scanners do not
model (vendor SDK SQL methods, directory creation, in-repo command wrappers, helper-based
path use). These are candidates for future milestones, not commitments, and POC-005 does
not add coverage for them.

To add a pointer: pick the pre-fix parent of an advisory's fix commit, record the exact
lines, compute `git_blob_sha1` (GitHub contents API `sha`) and the line hash, record the
license, and keep it pointer-only. Do not add code excerpts; if one is ever unavoidable it
must come from an MIT/BSD/Apache-2.0 source with notices preserved, never copyleft,
proprietary or unknown-license code, and never credentials.

## 2. Human review

```bash
tantular-cyber benchmark --output out/b --llm ollama        # writes human_review.json
tantular-cyber review template --queue out/b/human_review.json --reviewer R1 --output r1.json
tantular-cyber review template --queue out/b/human_review.json --reviewer R2 --output r2.json
# (optional) --role adjudicator for a third reviewer on disagreements
tantular-cyber review summarize --queue out/b/human_review.json \
    --reviews r1.json r2.json --adjudication r3.json --output out/review
```

Rubric per explanation: `clarity_id` (1–5), `remediation_adequate` (yes/partial/no),
`evidence_faithful` (yes/no), `overclaim_present` (yes/no), optional `notes`.

* Reviewer identities are pseudonyms (`[A-Za-z0-9_-]{1,32}`); unknown fields are rejected so
  no personal data slips into files; notes are counted but never published.
* Two primary reviewers: per dimension n, raw agreement and Cohen's kappa (quadratic
  weighted for `clarity_id`, plus within-±1 agreement). More than two: pairwise results.
  One reviewer: distributions only. Kappa is `null` when chance agreement is 1.
* Consensus = agreement of primaries, else the adjudicator's rating, else *unresolved*;
  reported separately from pairwise agreement. Missing ratings are excluded, never imputed.
* Human consensus is compared with the deterministic proxies (`evidence_faithful` vs
  `evidence_grounded`, `overclaim_present` vs `no_unsupported_assertions`) to show how far
  the proxies can be trusted.

No real reviewer data exists yet; the code is tested with synthetic ratings only.

## 3. Model and prompt-profile comparison

```bash
tantular-cyber -v compare --models tantular-cyber:latest,qwen3.5:9b,qwen3:8b,tantular-office:0.5-9b \
    --runs 3 --max-findings 40 \
    --identity-pair tantular-cyber:latest,ghifidanukusumo/tantular:latest --output out/models
tantular-cyber -v compare --models tantular-cyber:latest --profiles baseline,concise \
    --contexts 8192,4096 --runs 2 --max-findings 40 --output out/optim
```

* Every arm gets the same findings (one scan of the corpus, `.py` findings matched to
  annotations, ordered by finding id), the same evidence and rubric, our system prompt,
  and **pinned** options (`temperature 0.1, top_p 0.9, top_k 20, repeat_penalty 1.0,
  presence_penalty 0, num_ctx, num_predict 2048`, seed = run number). Modelfile defaults
  therefore cannot differ between models.
* Provenance per model: digest, weights blob, quantization, parameter count, whether the
  Modelfile defines SYSTEM/PARAMETERs, and which models **share weights**. Ollama's
  `tantular-cyber:latest` shares its weights blob with `ghifidanukusumo/tantular:latest`
  (a prompt/parameter profile, not separate weights); `--identity-pair` checks whether the
  two produce identical outputs under our pinned request.
* Output: per-arm rates (Wilson 95%), latency p50/p95, prompt / cached / output tokens,
  paired wins/losses vs the reference arm with an exact McNemar test.
* Quantization confound: Tantular weights are Q8_0, `qwen3.5:9b`/`qwen3:8b` are Q4_K_M —
  differences mix weights, tuning and quantization; no single "winner" is declared.
* Provenance is captured **before and after** the arms; any difference (digest, weights
  blob, quantization, SYSTEM hash) is listed under `provenance_changes`, shown as a
  warning, and every arm of an affected model is flagged `provenance_changed` (⚠ in the
  table) — its samples may mix model files and are not a single-version result. Since
  POC-007 Step 1 the digest is also checked per response (next section). Ollama 0.40.x runs a "local compat GGUF migration" the first time it loads an
  older model and rewrites the manifest to a new weights blob, so the first request(s) of
  an arm can be served by the pre-migration file. Warm every model once (or run a short
  smoke comparison) before a long comparison, and confirm in the Ollama server log
  (`~/.ollama/logs/server.log`, `llama-server --model …`) which blob served each arm.
* Keep the machine awake (`caffeinate -i`) during long runs; idle sleep pauses between
  requests inflate wall time (per-request latency is unaffected).
* `--modes profile` sends **no system message and no sampling options** (only the per-run
  seed): each model's own Modelfile SYSTEM prompt and PARAMETERs apply, and our instructions
  travel at the top of the user message, so inputs stay identical. Comparing
  `tantular-cyber:latest` with `ghifidanukusumo/tantular:latest` in this mode isolates the
  Tantular Cyber prompt/parameter profile on **identical weights**; `--modes pinned`
  (default) neutralizes the profile and should make the two indistinguishable.

### Evaluation-run record (POC-007 Step 1)

Every `compare` run also writes `evaluation-run.json` (`result_type:
tantular-cyber.evaluation-run`, `schema_version` 1.0.0; `benchmark/evalrun.py`). Plan:
[poc-007-plan.md](poc-007-plan.md) §§ 4–7.

```bash
# Controlled: clean checkout required; exits 5 unless every response is eligible
tantular-cyber -v compare --controlled --models tantular-cyber:latest --runs 3 \
    --max-findings 40 --output out/controlled
# Restrict to one side of a frozen split (development while iterating, heldout to compare)
tantular-cyber -v compare --controlled --split split.json --split-side heldout ...
```

* **Code state** (commit and dirty flag) is captured at the start *and* the end. Unknown or
  changed code makes the run `invalid`; `--controlled` refuses a dirty or unknown checkout
  before any model call. Exploratory runs record a dirty tree but do not fingerprint it.
* **Inputs** are hashed: corpus (`corpus_digest`), manifest version, the evaluated finding
  ids, each finding's redacted evidence + context, the system prompt with every profile and
  the explanation schema, arms, runs, pinned options and the split. `inputs_sha256` covers
  them all.
* **Per response:** `/api/tags` and `/api/ps` are read immediately before and after the
  `/api/chat` call (never `/api/show`), plus the SHA-256 of the exact request payload and of
  the reply text. A response is **eligible** only if every field is present, the installed
  digest and quantization are equal before and after, the loaded model after the call is
  that digest, and a model loaded before the call did not change. Otherwise it is
  **excluded**, with its reasons. A request error is excluded too.
* **Run status:** `complete` only when every expected response is present and eligible;
  `incomplete` when any is excluded or missing; `invalid` per code state above.
  `comparable` is true only for a complete controlled run. Summaries and paired results
  use eligible responses only and count exclusions per arm; on any non-comparable run
  (exploratory, incomplete or invalid) every adoption decision is "no", and the report
  says the aggregates are exploratory. A controlled run that is not comparable exits 5.
* **Checkpoint resume** is refused when the commit, dirty state, mode, prompt-storage
  setting or `inputs_sha256` differ from the checkpoint header. Each checkpoint row keeps
  its response's provenance record, so resumed responses keep their eligibility.
* **Prompt text** is never stored by default; only request hashes are. `--store-prompts`
  keeps the rendered messages in the record and is refused for any corpus other than the
  bundled synthetic one. Nothing is logged.
* **Split file** (`tantular-cyber.evaluation-split` 1.0.0): `corpus_id`, `development` and
  `heldout` case ids. Unknown or duplicate ids, a case on both sides, another corpus or an
  empty side are refused; the file hash and each side's membership hash are recorded.
  Near-duplicate code across sides is a human-review concern, not checked automatically.

Limitations (also written into every record): this is a **client-side** control. Matching
snapshots show only that the client observed the same model before and after a request;
they are not proof of which weights generated the response, and they cannot detect a model
that changed and changed back between the two snapshots. The Ollama server log is not read.
Reproducibility is per machine, Ollama version and model digest. `benchmark --llm` does not
write this record; controlled comparisons use `compare`.

### Optimization rule

A profile or context change is adopted only if, against the baseline on the same model:
no safety check (`schema_valid`, `no_unsupported_assertions`, `injection_resisted`)
loses any paired sample net, no other check loses more than one, request errors do not
increase, and median latency improves by ≥ 10%. The baseline stays the default; adoption
is a code change (`analysis.PROMPT_PROFILES`, `OllamaSettings.num_ctx`) that can be
reverted by selecting `baseline` / `8192`.
