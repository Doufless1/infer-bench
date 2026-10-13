"""infer-bench: a small, honest inference benchmarking harness.

It measures latency percentiles, throughput, accuracy relative to a reference
model and (where the hardware allows) power, and writes everything to a
self-describing JSON file so that two runs can be compared later.
"""

__version__ = "0.1.0"

# Version of the results-file format (see infer_bench/results.py).
SCHEMA_VERSION = "1.0"
