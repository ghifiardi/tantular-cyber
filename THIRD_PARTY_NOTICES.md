# Third-party notices

Tantular Cyber's own application code is licensed under the **Apache License 2.0**
([LICENSE](LICENSE)). That license covers only Tantular Cyber's code and the synthetic data
shipped with it (rules, benchmark corpus, sample project). It does **not** cover the
components below, which keep their own licenses.

## Python runtime dependencies (installed from PyPI, not vendored)

Versions are pinned with hashes in `requirements/runtime.lock.txt`. A release lists the exact
set it was tested with in its license audit and SBOM.

| Package | License |
|---|---|
| httpx | BSD-3-Clause |
| httpcore | BSD-3-Clause |
| h11 | MIT |
| anyio | MIT |
| idna | BSD-3-Clause |
| certifi | MPL-2.0 (file-level copyleft; used unmodified, source at https://github.com/certifi/python-certifi) |
| pydantic | MIT |
| pydantic-core | MIT |
| annotated-types | MIT |
| typing-extensions | PSF-2.0 |
| typing-inspection | MIT |

## External tools (user-installed, never bundled)

| Tool | License | Role |
|---|---|---|
| [Semgrep](https://github.com/semgrep/semgrep) (Community Edition CLI) | LGPL-2.1-only | Required for **full** scans. Tantular Cyber does not bundle or redistribute Semgrep binaries; you install and update it yourself (for example `brew install semgrep` or `pipx install semgrep`). The optional `tantular-cyber[semgrep]` extra only asks pip to fetch Semgrep from PyPI under its own license. Tantular Cyber uses only its own bundled rules and never Semgrep Registry rules. |
| [Ollama](https://github.com/ollama/ollama) | MIT | Optional local LLM server for `scan --ai`. Not bundled. |

## Model weights

Tantular Cyber does **not** distribute any model weights. Any model you run with Ollama,
including the Tantular weights (`ghifidanukusumo/tantular`) and Qwen models, is governed by
the license on its model page; check it before use. Ollama cloud models (names ending in
`:cloud`) run on remote servers; Tantular Cyber refuses them unless you pass
`--allow-remote-ollama`.

## Data

* `benchmarks/corpus` (shipped in the wheel as `tantular_cyber/_data/corpus`) and the sample
  project are independently written synthetic code. They contain **fake** credentials
  (`tc-synthetic-…`) and deliberately vulnerable patterns; they are never executed.
* `benchmarks/realworld/manifest.json` contains pointers (repository, commit, path, hashes)
  only — no third-party source code.
