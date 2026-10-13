"""ONNX Runtime backend (CPU by default, any installed execution provider)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import Backend, TensorSpec

# ONNX Runtime reports types as "tensor(float)"; map them to numpy dtype names.
_ORT_TO_NUMPY = {
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(double)": "float64",
    "tensor(int8)": "int8",
    "tensor(uint8)": "uint8",
    "tensor(int16)": "int16",
    "tensor(int32)": "int32",
    "tensor(int64)": "int64",
    "tensor(bool)": "bool",
}

_GRAPH_OPT_LEVELS = ("disable", "basic", "extended", "all")


class OrtBackend(Backend):
    name = "onnxruntime"

    def __init__(
        self,
        providers: list[str] | None = None,
        intra_op_threads: int = 0,
        inter_op_threads: int = 0,
        graph_optimization: str = "all",
    ):
        """
        providers:         execution providers in priority order. Default: CPU only.
        intra_op_threads:  threads used inside one operator (0 = ONNX Runtime default).
        inter_op_threads:  threads used to run independent operators in parallel (0 = default).
        graph_optimization: "disable", "basic", "extended" or "all".
        """
        if graph_optimization not in _GRAPH_OPT_LEVELS:
            raise ValueError(f"graph_optimization must be one of {_GRAPH_OPT_LEVELS}")
        self.requested_providers = providers or ["CPUExecutionProvider"]
        self.intra_op_threads = int(intra_op_threads)
        self.inter_op_threads = int(inter_op_threads)
        self.graph_optimization = graph_optimization
        self._session = None
        self._inputs: list[TensorSpec] = []
        self._outputs: list[TensorSpec] = []
        self._output_names: list[str] = []
        self.active_providers: list[str] = []

    def load(self, model_path: str | Path) -> None:
        import onnxruntime as ort

        available = ort.get_available_providers()
        missing = [p for p in self.requested_providers if p not in available]
        if missing:
            # ONNX Runtime would silently fall back to the CPU. For a benchmark
            # that is dangerous (you would report CPU numbers as GPU numbers).
            raise RuntimeError(
                f"Execution provider(s) {missing} not available. Available here: {available}"
            )
        options = ort.SessionOptions()
        options.log_severity_level = 3  # errors only
        options.intra_op_num_threads = self.intra_op_threads
        options.inter_op_num_threads = self.inter_op_threads
        options.graph_optimization_level = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }[self.graph_optimization]
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=self.requested_providers
        )
        self.active_providers = list(self._session.get_providers())
        self._inputs = [
            TensorSpec(i.name, tuple(i.shape), _ORT_TO_NUMPY.get(i.type, i.type))
            for i in self._session.get_inputs()
        ]
        self._outputs = [
            TensorSpec(o.name, tuple(o.shape), _ORT_TO_NUMPY.get(o.type, o.type))
            for o in self._session.get_outputs()
        ]
        self._output_names = [o.name for o in self._outputs]

    @property
    def inputs(self) -> list[TensorSpec]:
        return self._inputs

    @property
    def outputs(self) -> list[TensorSpec]:
        return self._outputs

    def run(self, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        # InferenceSession.run is thread-safe, which the multi-stream mode relies on.
        return self._session.run(self._output_names, feed)

    def describe(self) -> dict[str, Any]:
        import onnxruntime as ort

        return {
            "name": self.name,
            "version": ort.__version__,
            "requested_providers": self.requested_providers,
            "active_providers": self.active_providers,
            "intra_op_threads": self.intra_op_threads,
            "inter_op_threads": self.inter_op_threads,
            "graph_optimization": self.graph_optimization,
        }

    def close(self) -> None:
        self._session = None
