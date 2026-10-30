PYTHON ?= python3.11
VENV ?= .venv
BIN := $(VENV)/bin
OUT ?= ./out

.PHONY: install install-semgrep test smoke bench bench-live bench-replay lint format schema scan scan-clean doctor clean lock dist package-smoke

install:  ## create venv and install with dev extras (semgrep from brew or `make install-semgrep`)
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e '.[dev]'

install-semgrep:
	$(BIN)/pip install -e '.[dev,semgrep]'

test:  ## unit + integration tests (no Ollama needed)
	$(BIN)/pytest -q

smoke:  ## live test against local Ollama tantular-cyber:latest
	TANTULAR_RUN_OLLAMA_SMOKE=1 $(BIN)/pytest -m ollama -q

bench:  ## offline deterministic benchmark (same as CI)
	$(BIN)/tantular-cyber benchmark --output $(OUT)/benchmark

bench-live:  ## opt-in: + local Ollama explanation quality, cold/warm perf, concurrency 1,2
	$(BIN)/tantular-cyber -v benchmark --output $(OUT)/benchmark-live --llm ollama \
		--llm-runs 2 --cold --concurrency 1,2 \
		$(if $(BASELINE),--baseline $(BASELINE),)

bench-replay:  ## LLM evaluation path with recorded outputs (no model needed)
	$(BIN)/tantular-cyber benchmark --output $(OUT)/benchmark-replay --llm replay \
		--replay-file tests/fixtures/bench_replay.json

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

schema:  ## regenerate the GATRA-facing report schema
	$(BIN)/tantular-cyber schema > docs/report-schema.json

scan:
	$(BIN)/tantular-cyber scan --repo tests/fixtures/vulnerable --output $(OUT)

scan-clean:
	$(BIN)/tantular-cyber scan --repo tests/fixtures/clean --output $(OUT)-clean

doctor:
	$(BIN)/tantular-cyber doctor

lock:  ## re-pin runtime/build dependencies with hashes (network; review the diff)
	uv pip compile pyproject.toml --universal --python-version 3.11 --generate-hashes \
		--no-header --annotation-style line -o requirements/runtime.lock.txt
	uv pip compile requirements/build.in --universal --python-version 3.11 --generate-hashes \
		--no-header --annotation-style line -o requirements/build.lock.txt

dist:  ## reproducible wheel + sdist from hash-locked build tooling
	$(PYTHON) -m venv .build-venv
	.build-venv/bin/pip install --require-hashes -r requirements/build.lock.txt
	SOURCE_DATE_EPOCH=$$(git log -1 --format=%ct) .build-venv/bin/python -m build --no-isolation --outdir dist .
	shasum -a 256 dist/*

package-smoke: dist  ## fresh-venv install of the wheel, with and without Semgrep
	PYTHON=$(PYTHON) tools/release/smoke.sh dist/tantular_cyber-*-py3-none-any.whl requirements/runtime.lock.txt none
	PYTHON=$(PYTHON) tools/release/smoke.sh dist/tantular_cyber-*-py3-none-any.whl requirements/runtime.lock.txt semgrep

clean:
	rm -rf $(OUT) $(OUT)-clean .pytest_cache .ruff_cache build dist .build-venv *.egg-info
