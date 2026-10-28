PYTHON ?= python3.11
VENV ?= .venv
BIN := $(VENV)/bin
OUT ?= ./out

.PHONY: install install-semgrep test smoke bench bench-live bench-replay lint format schema scan scan-clean doctor clean

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

clean:
	rm -rf $(OUT) $(OUT)-clean .pytest_cache .ruff_cache build dist *.egg-info
