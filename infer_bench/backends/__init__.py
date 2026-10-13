"""Inference backends. ``create_backend`` is the single place that knows the names."""

from __future__ import annotations

from typing import Any

from .base import Backend, TensorSpec
from .ort import OrtBackend
from .sleep import SleepBackend

BACKENDS = ("onnxruntime", "sleep")


def create_backend(name: str, **options: Any) -> Backend:
    """Build a backend by name. Unknown options are an error, not ignored."""
    if name in ("onnxruntime", "ort"):
        return OrtBackend(**options)
    if name == "sleep":
        return SleepBackend(**options)
    raise ValueError(f"unknown backend '{name}'. Choose from: {', '.join(BACKENDS)}")


__all__ = ["Backend", "TensorSpec", "OrtBackend", "SleepBackend", "BACKENDS", "create_backend"]
