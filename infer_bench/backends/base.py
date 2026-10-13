"""The interface every inference backend implements.

The benchmark runner only talks to this interface, so supporting another
runtime (OpenVINO, TensorRT, a vendor SDK such as an accelerator's own
Python API...) means writing one small subclass, not touching the harness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TensorSpec:
    """Name, shape and dtype of a model input or output.

    A shape entry is an ``int`` for a fixed dimension, or a ``str`` / ``None``
    for a dynamic one (for example the batch size).
    """

    name: str
    shape: tuple
    dtype: str  # numpy dtype name, e.g. "float32"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}


class Backend(ABC):
    """A loaded model that can run inference."""

    name = "base"

    @abstractmethod
    def load(self, model_path: str | Path) -> None:
        """Load the model. Must be called before anything else."""

    @property
    @abstractmethod
    def inputs(self) -> list[TensorSpec]:
        """Model inputs."""

    @property
    @abstractmethod
    def outputs(self) -> list[TensorSpec]:
        """Model outputs."""

    @abstractmethod
    def run(self, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Run one inference. ``feed`` maps input names to arrays.

        Must be safe to call from several threads at once if the backend is
        to be used with ``--streams > 1``.
        """

    def describe(self) -> dict[str, Any]:
        """Backend name, version and options, recorded in the results file."""
        return {"name": self.name}

    def close(self) -> None:
        """Release resources. Optional."""

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
