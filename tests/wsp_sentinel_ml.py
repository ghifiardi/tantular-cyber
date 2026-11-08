"""Stand-in for an ML library in the per-arm process tests: it must only ever be imported in a
worker process, never in the test (parent) process."""
