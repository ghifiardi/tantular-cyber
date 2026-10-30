# Acceptance run — v0.3.0 — YYYY-MM-DD

Copy this file to `docs/acceptance-runs/YYYY-MM-DD-v0.3.0.md` (add `-2`, `-3` for further
runs on the same day). Fill in observed results only; expected results live in
[`../release-acceptance-v0.3.0.md`](../release-acceptance-v0.3.0.md) and are not repeated or
edited here. Redact local paths (use `$ACC`), user names, hostnames, secrets and any
non-sample repository content before committing.

* Checklist version: 1.0
* Tester / reviewer (pseudonym is fine):
* Started / finished (UTC):
* Overall: ACCEPTED / NOT ACCEPTED (any FAIL, or any BLOCKED test in A, B, N or C)

## Environment

| Item | Value |
|---|---|
| OS, version, architecture | |
| Python | |
| Semgrep (`semgrep --version`) | |
| Ollama (`ollama --version` or "not installed") | |
| Model used for `--ai` tests (name + digest from `ollama list`) | |
| `gh` | |
| Network setup (normal / offline for B11) | |

## Release assets

| Asset | SHA-256 (observed) | `SHA256SUMS` check |
|---|---|---|
| tantular_cyber-0.3.0-py3-none-any.whl | | |
| tantular_cyber-0.3.0.tar.gz | | |
| tantular-cyber-0.3.0-runtime.lock.txt | | |
| tantular-cyber-0.3.0.cdx.json | | |
| tantular-cyber-0.3.0-licenses.md | | |
| THIRD_PARTY_NOTICES.md | | |
| SHA256SUMS | | n/a |

Attestation (A3): wheel ___, sdist ___; workflow / ref / commit shown: ___

## Results

Status is PASS, FAIL or BLOCKED (with the missing prerequisite).

| ID | Actual result (exact output or summary of it) | Exit code | Evidence location | Status |
|---|---|---|---|---|
| A1 | | | | |
| A2 | | | | |
| A3 | | | | |
| A4 | | | | |
| A5 | | | | |
| A6 | | | | |
| A7 | | | | |
| A8 | | | | |
| A9 | | | | |
| A10 | | | | |
| A11 | | | | |
| B1 | | | | |
| B2 | | | | |
| B3 | | | | |
| B4 | | | | |
| B5 | | | | |
| B6 | | | | |
| B7 | | | | |
| B8 | | | | |
| B9 | | | | |
| B10 | | | | |
| B11 | | | | |
| N1 | | | | |
| N2 | | | | |
| N3 | | | | |
| N4 | | | | |
| N5 | | | | |
| C1 | | | | |
| C2 | | | | |
| D1 | | | | BLOCKED (no authorized repository selected) |
| D2 | | | | BLOCKED |
| D3 | | | | BLOCKED |
| D4 | | | | BLOCKED |
| D5 | | | | BLOCKED |

## Network observations (section N)

List every attempted destination, including failures. Keep local IPC (`AF_UNIX`, pipes)
in its own list with what it appears to be.

* External (`AF_INET`/`AF_INET6`) attempts:
* Loopback attempts:
* Local IPC:

## Section D counts (only when D is unblocked)

True positives / false positives / uncertain / known misses: __ / __ / __ / __.
No recall or detection rate is stated: the repository has no ground truth.

## Deviations and follow-ups

Any step done differently from the checklist, and an issue link for every FAIL.
