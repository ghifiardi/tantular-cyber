"""POC-003 benchmark and evaluation framework.

Measures scanner detection (deterministic, CI-gated) separately from LLM explanation
quality (sample-based, report-only). Benchmark artifacts use their own schema
(``tantular-cyber.benchmark``) and never change the POC-002 report schema.
"""

BENCHMARK_SCHEMA_VERSION = "1.1.0"  # 1.1: per-CWE gates, provenance, context metrics
BENCHMARK_RESULT_TYPE = "tantular-cyber.benchmark"
