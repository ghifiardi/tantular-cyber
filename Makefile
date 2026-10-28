PYTHON ?= python3.11
VENV ?= .venv
BIN := $(VENV)/bin
OUT ?= ./out

.PHONY: install install-semgrep test smoke lint format schema scan scan-clean doctor clean

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
