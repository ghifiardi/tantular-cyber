# Installing Tantular Cyber

Tantular Cyber is a local-first, read-only security analyzer for Python repositories. You can
install it and run a **deterministic** scan without cloning this repository and without
downloading any AI model. Quickstart: [English](quickstart.md) ·
[Bahasa Indonesia](quickstart.id.md).

## Supported platforms

| Platform | Status | Scan capability |
|---|---|---|
| macOS (Apple silicon) | supported; tested in CI (`macos-latest`) and locally | **full** with Semgrep |
| Linux x86-64 | supported; tested in CI (Ubuntu, `ubuntu-latest`) | **full** with Semgrep |
| Other macOS/Linux architectures | expected to work wherever Semgrep and Python 3.11+ run; untested | **full** with Semgrep |
| Windows 10/11 via **WSL2** (Ubuntu) | supported (follow the Linux steps inside WSL2) | **full** with Semgrep |
| Native Windows | **not supported** | not tested; any scan is reported `degraded`, never `complete` |

Python **3.11 or 3.12** (tested; `requires-python >= 3.11`). Semgrep is required for full
scans; without it Tantular Cyber still runs its own call-graph scanner, but the report is
explicitly **PARTIAL / degraded** and the exit code is 4.

## 1. Install Semgrep (external, user-managed)

Semgrep is **not bundled**. It is a separate project licensed under LGPL-2.1; you install and
update it yourself. Tested range: `>= 1.90, < 2`.

```bash
# macOS
brew install semgrep
# Linux / WSL2
pipx install semgrep            # or: python3 -m pip install --user semgrep
semgrep --version
```

Tantular Cyber runs Semgrep with `--metrics=off --disable-version-check` and only its own
bundled rules (never Semgrep Registry rules).

## 2. Install Tantular Cyber

Download the release assets from the GitHub release page
(`https://github.com/ghifiardi/tantular-cyber/releases`) and **verify them first**
([release verification](release.md)):

```bash
VERSION=0.3.0
BASE=https://github.com/ghifiardi/tantular-cyber/releases/download/v$VERSION
curl -fLO "$BASE/tantular_cyber-$VERSION-py3-none-any.whl"
curl -fLO "$BASE/tantular-cyber-$VERSION-runtime.lock.txt"
curl -fLO "$BASE/SHA256SUMS"
shasum -a 256 --ignore-missing -c SHA256SUMS
gh attestation verify "tantular_cyber-$VERSION-py3-none-any.whl" --repo ghifiardi/tantular-cyber
```

**Option A — pipx (simplest).** Isolated environment, command on your `PATH`:

```bash
pipx install "./tantular_cyber-$VERSION-py3-none-any.whl"
```

pipx resolves the dependencies (httpx, pydantic and their dependencies) from PyPI within the
version ranges declared by the package.

**Option B — pinned and hash-verified dependencies.** Every dependency is installed at the
exact version and hash the release was tested with:

```bash
python3 -m venv ~/.local/share/tantular-cyber
~/.local/share/tantular-cyber/bin/pip install --require-hashes -r "tantular-cyber-$VERSION-runtime.lock.txt"
~/.local/share/tantular-cyber/bin/pip install --no-deps "./tantular_cyber-$VERSION-py3-none-any.whl"
ln -s ~/.local/share/tantular-cyber/bin/tantular-cyber ~/.local/bin/tantular-cyber
```

**Option C — from a tag without downloading assets** (builds from source; not
checksum-verified): `pipx install "git+https://github.com/ghifiardi/tantular-cyber@v$VERSION"`.

Developers working on Tantular Cyber itself: clone the repository and run `make install`.

## 3. Check your setup

```bash
tantular-cyber doctor          # human-readable; add --json for machines
```

`doctor` is read-only: it runs `semgrep --version`, may contact a **loopback** Ollama endpoint
with read-only requests (`/api/version`, `/api/tags`, `/api/show`) and nothing else. It never
downloads anything, pulls models or contacts remote hosts, and Tantular Cyber writes no files
(Semgrep may update its own state such as `~/.semgrep`). URLs containing credentials are
rejected. Exit code 0 = full
scan capability, 4 = partial, 3 = unavailable. Use `--no-ollama` to skip Ollama entirely.

## 4. Optional: local AI explanations (Ollama)

Scans never need a model. To add explanations with `scan --ai`:

1. Install Ollama from <https://ollama.com> (MIT-licensed, not bundled) and start it.
2. Choose a **local** model. Tantular Cyber was evaluated with the Tantular weights
   (`ghifidanukusumo/tantular`, Q8_0, about 10 GB on disk); other models are untested and
   behave differently (see [benchmark-results.md](benchmark-results.md)).
   `tantular-cyber:latest`, the default name, is a local Modelfile profile over those
   weights; under Tantular Cyber's pinned request the two produced identical outputs, so
   you can use `--model ghifidanukusumo/tantular:latest` directly.
3. Pulling a model is **your** decision and is never done by Tantular Cyber:
   `ollama pull ghifidanukusumo/tantular:latest` downloads about 10 GB from the Ollama
   registry and creates a local manifest. Tags can be updated upstream (the digest
   changes), and Ollama may migrate a model's files on first load. Check the model page
   for its license before use.

Cloud models (`*:cloud`) run on remote servers even though Ollama listens on localhost;
`scan --ai` refuses them unless you pass `--allow-remote-ollama`.

## Uninstall and cleanup

```bash
pipx uninstall tantular-cyber                 # Option A
rm -rf ~/.local/share/tantular-cyber ~/.local/bin/tantular-cyber   # Option B
rm -rf ~/.cache/tantular-cyber                # real-world pointer cache, if you used it
```

Reports live only where you wrote them (`--output`). Tantular Cyber never modifies scanned
repositories. Semgrep, Ollama and models are yours to keep or remove (`ollama rm <model>`).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `coverage: PARTIAL`, exit 4, "one or more scanners failed" | Semgrep not found or not runnable. Install it (step 1) or pass `--semgrep-bin /path/to/semgrep`. |
| doctor warns Semgrep is outside the tested range | Results may differ; install a version `>= 1.90, < 2`. |
| `scan --ai` → `degraded`, "LLM explanation unavailable" | Start Ollama, or drop `--ai` for a deterministic scan. |
| "model '…' is not available" | `ollama list`; pass an installed model with `--model`. |
| "runs on a remote host" | A cloud model was selected; choose a local model. |
| "not localhost" | `--ollama-url` points off this machine; it requires `--allow-remote-ollama` and sends redacted excerpts off-host. |
| "output directory is inside the analyzed repository" | Write reports outside the scanned repo, or pass `--allow-output-inside-repo`. |
| Native Windows reports `degraded` | Unsupported platform; use WSL2. |
