"""A fake backend with a known, fixed latency.

Its purpose is to test the *harness itself*: if the harness runs a backend
that takes exactly 5 ms and reports 7 ms, the harness is wrong. It is used by
the test-suite and by ``infer-bench selfcheck``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import Backend, TensorSpec


class SleepBackend(Backend):
    name = "sleep"

    def __init__(self, latency_ms: float = 5.0, spin: bool = True, output_size: int = 10):
        """
        latency_ms: how long every ``run`` takes.
        spin:       busy-wait (accurate, uses a CPU core) instead of ``time.sleep``
                    (releases the CPU and the GIL, so several streams can overlap).
        """
        self.latency_ms = float(latency_ms)
        self.spin = bool(spin)
        self.output_size = int(output_size)
        self._inputs = [TensorSpec("images", ("batch", 3, 32, 32), "float32")]
        self._outputs = [TensorSpec("logits", ("batch", self.output_size), "float32")]

    def load(self, model_path: str | Path | None = None) -> None:  # the path is ignored
        return None

    @property
    def inputs(self) -> list[TensorSpec]:
        return self._inputs

    @property
    def outputs(self) -> list[TensorSpec]:
        return self._outputs

    def run(self, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        duration_ns = int(self.latency_ms * 1e6)
        if self.spin:
            deadline = time.perf_counter_ns() + duration_ns
            while time.perf_counter_ns() < deadline:
                pass
        else:
            time.sleep(self.latency_ms / 1000.0)
        batch = next(iter(feed.values())).shape[0]
        return [np.zeros((batch, self.output_size), dtype=np.float32)]

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "latency_ms": self.latency_ms, "spin": self.spin}
