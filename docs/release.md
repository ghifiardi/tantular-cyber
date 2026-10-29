# Releases: verification, reproducibility and SBOM

## Release assets

A tag `vX.Y.Z` (equal to the `pyproject.toml` version) runs `.github/workflows/release.yml`,
which publishes a **draft** release that a maintainer reviews and publishes. Nothing is
uploaded to PyPI.

| Asset | Purpose |
|---|---|
| `tantular_cyber-X.Y.Z-py3-none-any.whl` | the package (pure Python) |
| `tantular_cyber-X.Y.Z.tar.gz` | source distribution |
| `tantular-cyber-X.Y.Z-runtime.lock.txt` | exact, hash-pinned runtime dependencies the release was tested with |
| `tantular-cyber-X.Y.Z.cdx.json` | CycloneDX 1.5 SBOM: wheel hash, locked dependencies with licenses, declared external tools (Semgrep, Ollama — not bundled) |
| `tantular-cyber-X.Y.Z-licenses.md` | license audit of the locked runtime dependencies |
| `THIRD_PARTY_NOTICES.md` | third-party and model-weight licensing notes |
| `SHA256SUMS` | SHA-256 of every asset |

Wheel and sdist carry a GitHub **build provenance attestation** (Sigstore-signed SLSA
provenance tied to the workflow run).

## Verify before installing

```bash
shasum -a 256 --ignore-missing -c SHA256SUMS
gh attestation verify tantular_cyber-X.Y.Z-py3-none-any.whl --repo ghifiardi/tantular-cyber
```

The attestation proves the file was built by this repository's release workflow from the
tagged commit; the checksum proves it was not altered after download.

## What the release workflow checks

1. Tag equals the package version.
2. Build tooling installed from `requirements/build.lock.txt` with `--require-hashes`.
3. Wheel and sdist are built **twice** with `SOURCE_DATE_EPOCH` = tagged commit time; the
   two builds must be byte-identical.
4. Runtime lock installed with `--require-hashes` into a clean venv; license audit
   (allowlist: MIT, BSD-2/3-Clause, Apache-2.0, ISC, PSF-2.0, MPL-2.0 for unmodified use;
   anything else fails) and SBOM generation.
5. Fresh-install smoke tests on Ubuntu and macOS × Python 3.11/3.12, each **without**
   Semgrep (must report PARTIAL/degraded, exit 4) and **with** a user-style Semgrep install
   (must be full/complete, exit 0, offline benchmark gates pass). On Linux the scan runs in
   a network namespace with no external interfaces.
6. Checksums verified, provenance attested, draft release created.

CI (`.github/workflows/ci.yml`, job `package`) runs steps 2–5 on every pull request.

## Reproducing a build

```bash
git checkout vX.Y.Z
python3.11 -m venv /tmp/b && /tmp/b/bin/pip install --require-hashes -r requirements/build.lock.txt
SOURCE_DATE_EPOCH=$(git log -1 --format=%ct) /tmp/b/bin/python -m build --no-isolation --outdir dist .
shasum -a 256 dist/*
```

The hashes should match `SHA256SUMS`. Builds are pure Python; differences usually mean a
different checkout (line endings, extra files) or different build tool versions.

## Updating locks

```bash
make lock    # uv pip compile --universal --generate-hashes (needs network; review the diff)
```

Runtime dependency ranges live in `pyproject.toml`; the lock pins what releases are tested
with. Semgrep is deliberately absent from the lock: it is an external, user-managed tool.
