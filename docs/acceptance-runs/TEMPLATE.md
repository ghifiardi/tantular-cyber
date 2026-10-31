# Acceptance run — v0.3.0 — YYYY-MM-DD

Copy this file to `docs/acceptance-runs/YYYY-MM-DD-v0.3.0.md` (add `-2`, `-3` for further
runs on the same day). Fill in observed results only; expected results live in
[`../release-acceptance-v0.3.0.md`](../release-acceptance-v0.3.0.md) and are not repeated or
edited here. Redact local paths (use `$ACC`), user names, hostnames, secrets and any
non-sample repository content before committing.

* Checklist version: 1.0
* Tester / reviewer (pseudonym is fine):
* Started / finished (UTC):
* Overall (one of, using the rules in the checklist's "Test classes and the acceptance
  decision"):
  * **ACCEPTED** — every mandatory and conditional test PASS, no FAIL anywhere;
  * **CONDITIONALLY ACCEPTED** — every mandatory test PASS, no FAIL, one or more
    conditional tests BLOCKED (list them under "Blocked conditional tests" below);
  * **NOT ACCEPTED** — any mandatory test FAIL or BLOCKED, or any FAIL in any class.
* Decision:
* Section D (separate evaluation): BLOCKED (no authorized repository selected) / complete

### Blocked conditional tests

| ID | Missing prerequisite | Assurance not established |
|---|---|---|
| | | |

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

Status is PASS, FAIL, BLOCKED (with the missing prerequisite) or, for optional tests only, N/A (with the reason).

| ID | Class | Actual result (exact output or summary of it) | Exit code | Evidence location | Status |
|---|---|---|---|---|---|
| A1 | mandatory | | | | |
| A2 | mandatory | | | | |
| A3 | mandatory | | | | |
| A4 | mandatory | | | | |
| A5 | mandatory | | | | |
| A6 | mandatory | | | | |
| A7 | mandatory | | | | |
| A8 | mandatory | | | | |
| A9 | mandatory | | | | |
| A10 | mandatory | | | | |
| A11 | mandatory | | | | |
| B1 | mandatory | | | | |
| B2 | mandatory | | | | |
| B3 | mandatory | | | | |
| B4 | conditional | | | | |
| B5 | mandatory | | | | |
| B6 | mandatory | | | | |
| B7 | mandatory | | | | |
| B8 | mandatory | | | | |
| B9 | conditional | | | | |
| B10 | mandatory | | | | |
| B11 | mandatory | | | | |
| N1 | mandatory | | | | |
| N2 | conditional | | | | |
| N3 | conditional | | | | |
| N4 | conditional | | | | |
| N5 | optional | | | | |
| C1 | mandatory | | | | |
| C2 | mandatory | | | | |
| D1 | separate | | | | BLOCKED (no authorized repository selected) |
| D2 | separate | | | | BLOCKED |
| D3 | separate | | | | BLOCKED |
| D4 | separate | | | | BLOCKED |
| D5 | separate | | | | BLOCKED |

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
