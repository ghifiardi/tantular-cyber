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
* `benchmarks/runtime` (shipped as `tantular_cyber/_data/runtime`) holds **synthetic**
  runtime-telemetry records written for this project (Apache-2.0), shaped like the decoder
  output of Shirudo kernel-obs at commit `44926eb77751495be6cc39cd745692355678956f`. They are
  not captured traces and contain no upstream code. The replay file in it is hand-written,
  not model output.

## Shirudo reference documentation excerpts

`src/tantular_cyber/runtime/data/shirudo-reference-corpus.json` contains excerpts (chunked,
control characters stripped, otherwise unmodified) of these documentation files of
[Shirudo](https://github.com/shirudo-dev/shirudo) at commit
`44926eb77751495be6cc39cd745692355678956f`; the corpus records each file's SHA-256 and URL:

| File | License as found |
|---|---|
| `upatches/kernel-obs/README.md` | no per-file SPDX header; repository `LICENSE` is BSD-2-Clause |
| `upatches/app-sandbox/README.md` | no per-file SPDX header; repository `LICENSE` is BSD-2-Clause |
| `Documentation/reference/formats/map-export.rst` | `SPDX-License-Identifier: BSD-2-Clause` |
| `Documentation/reference/cli/upatch-map.rst` | `SPDX-License-Identifier: BSD-2-Clause` |
| `Documentation/security/security-model.rst` | `SPDX-License-Identifier: BSD-2-Clause` |

These excerpts are **not** relicensed under Apache-2.0. No Shirudo BPF source, Lua decoder
(`GPL-2.0 OR BSD-2-Clause`) or script is copied, shipped or executed; the importer in
`src/tantular_cyber/runtime/` is independently written from reading the decoders' output
format. The two README licences rely on the repository-level `LICENSE`; confirm with the
upstream authors before relying on that for redistribution beyond this notice.

```text
BSD 2-Clause License

Copyright (c) 2026 Authors of shirudo

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
