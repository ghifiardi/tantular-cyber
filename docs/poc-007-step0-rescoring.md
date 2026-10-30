# POC-007 Step 0 — offline re-scoring of retained POC-005 outputs

> **Exploratory re-scoring. Not a reproduction of the published benchmark.** The published
> 57% `evidence_grounded` result (2026-10-29, `docs/benchmark-results.md`) cannot be
> reproduced: its per-sample outputs were not kept. Nothing here replaces or updates it, and
> these numbers must not be placed in one trend with it or with any future controlled
> baseline (`docs/poc-007-plan.md`, § 3).

Date: 2026-10-30. Plan: `docs/poc-007-plan.md`, § 7 Step 0. Tool:
`tools/eval/step0_rescore.py`. No model was called; no network was used.

## What was done

For each retained POC-005 comparison run, the tool:

1. imported the package from a checkout of a preservation tag (`eval/poc-005-76a9318`, and
   `eval/poc-005-c889c32` as a cross-check) and refused to run if any other copy was
   imported;
2. rebuilt the 40 evaluated findings by scanning the bundled corpus at that commit, using
   the historical `select_findings` (the same path the harness used to resume from a
   checkpoint);
3. applied that commit's `evaluate_finding` to each saved, validated analysis;
4. split each `evidence_grounded` result by the check's own failure text into: grounded,
   cited identifiers missing from the evidence/context, cited line numbers outside the shown
   range, both, or nothing cited ("not assessable").

## Reconstruction checks

| Check | `optim` | `sameweights` |
|---|---|---|
| Findings rebuilt / expected | 40 / 40 | 40 / 40 |
| Rebuilt selection equals the run's recorded finding list | yes | yes |
| Corpus id and manifest version equal the run header | yes | yes |
| Re-scored `evidence_grounded` passed/assessed equals the value the run recorded, every arm | yes (4/4 arms) | yes (2/2 arms) |
| Samples excluded (request error, unreconstructed finding, invalid analysis, no explanation) | 0 | 0 |
| Result identical when rebuilt from `c889c32` instead of `76a9318` | — | yes |

A finding id is a hash of tool, rule, path, line range and evidence-line text
(`pipeline.finding_id`), so a matching id means the same location and evidence lines. The
context shown to the model is computed from those lines by `context.py`, which, like the
rubric, rules, scanners, redaction and corpus, is unchanged between `76a9318` and `main`.

## Results (aggregate counts only)

`evidence_grounded` outcomes per arm. The rate excludes "nothing cited" from its
denominator, as the historical rubric does.

| Run | Arm | Grounded | Missing identifiers | Lines out of range | Both | Nothing cited | Rate (Wilson 95%) |
|---|---|---|---|---|---|---|---|
| `optim` | `tantular-cyber:latest` baseline | 45 | 33 | 0 | 2 | 0 | 45/80 = 0.56 (0.45–0.67) |
| `optim` | baseline @ num_ctx 4096 | 45 | 33 | 0 | 2 | 0 | 45/80 = 0.56 (0.45–0.67) |
| `optim` | concise | 52 | 20 | 0 | 0 | 8 | 52/72 = 0.72 (0.61–0.81) |
| `optim` | concise @ num_ctx 4096 | 52 | 20 | 0 | 0 | 8 | 52/72 = 0.72 (0.61–0.81) |
| `sameweights` | `tantular-cyber:latest`, Modelfile profile | 87 | 22 | 0 | 2 | 9 | 87/111 = 0.78 (0.70–0.85) |
| `sameweights` | `ghifidanukusumo/tantular:latest`, Modelfile profile | 84 | 25 | 0 | 1 | 10 | 84/110 = 0.76 (0.68–0.83) |

## Observations (exploratory)

- **Failures are identifier failures.** In every arm, every failed sample cites at least
  one identifier not found in the evidence or context shown; no sample fails on line
  numbers alone. Line-number formatting is therefore not the lead to pursue first.
- **The concise profile's higher rate is partly abstention.** It cites nothing in 8 of 80
  samples (baseline: 0 of 80). Counting those as not grounded gives 52/80 = 0.65 rather
  than 0.72. This is the effect the plan's § 6 control is meant to expose.
- The profile-mode arms also leave 9–10 of 120 samples uncited.

These are diagnostic observations on two exploratory runs. They are not evidence that any
change improves grounding, and they do not say *why* identifiers are missing.

## Assumptions

- The evidence and context rebuilt at `76a9318` equal what the model received. This cannot
  be verified directly because prompts were not saved; the id, selection and recorded-rate
  matches above are consistent with it.
- `sameweights` ran on `76a9318` plus an uncommitted change limited to provenance reporting
  (later committed as `c889c32`); rebuilding from either commit gives identical results.
- `optim` ran on `76a9318`; whether the working tree had uncommitted changes is unknown.
- Semgrep 1.182.0 was used for the rebuild. The version used in the original runs was not
  recorded; the matching finding ids show it produced the same findings.

## Missing artifacts

- Per-sample outputs of the published `models` run (the 57%); its generating code is also
  unidentifiable (the run began before the first POC-005 commit).
- All prompts, for every run.
- Per-response model digests, for every run.
- The dirty-tree state of every run.

## Not covered here (later steps)

Why identifiers are missing — context selection (present in the file but not shown) versus
fabrication (absent from the file) — is Step 2's taxonomy. Human judgement of whether
grounded citations actually support the conclusions is Step 3.

## Reproduce

```bash
git worktree add --detach /tmp/hist-76a9318 eval/poc-005-76a9318
.venv/bin/python tools/eval/step0_rescore.py --historical /tmp/hist-76a9318 \
    --run out/optim --output out/step0/optim
.venv/bin/python tools/eval/step0_rescore.py --historical /tmp/hist-76a9318 \
    --run out/sameweights --output out/step0/sameweights
```

`out/optim` and `out/sameweights` are the git-ignored POC-005 run directories on the
machine that produced them; they are not published. The tool exits non-zero if any recorded
finding cannot be rebuilt or the corpus differs from the run header.
