"""Shared fixtures. Models are built once per test session (it takes about a second)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from infer_bench.backends import OrtBackend, SleepBackend
from infer_bench.demo_model import build_classifier, build_detector
from infer_bench.inputs import synthetic_feeds
from infer_bench.quantize import quantize_int8
from infer_bench.runner import RunConfig, run_benchmark


@pytest.fixture(scope="session")
def models(tmp_path_factory) -> dict[str, Path]:
    """FP32 and INT8 demo models: classifier, detector, and the detector quantised naively and Conv-only."""
    root = tmp_path_factory.mktemp("models")
    paths = {
        "cls": build_classifier(root / "cls.onnx"),
        "det": build_detector(root / "det.onnx"),
    }
    for name in ("cls", "det"):
        backend = OrtBackend()
        backend.load(paths[name])
        calib = synthetic_feeds(backend.inputs, 1, count=32, seed=1)
        quantize_int8(paths[name], root / f"{name}_int8.onnx", calib)
        paths[f"{name}_int8"] = root / f"{name}_int8.onnx"
        if name == "det":
            quantize_int8(paths[name], root / "det_int8_conv.onnx", calib, op_types=["Conv"])
            paths["det_int8_conv"] = root / "det_int8_conv.onnx"
    return paths


def with_opset(path_in: Path, path_out: Path, version: int) -> Path:
    """Copy of a model that declares an older default operator set (its operators are valid in both)."""
    import onnx

    model = onnx.load(str(path_in))
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            imp.version = version
    onnx.checker.check_model(model)
    onnx.save(model, str(path_out))
    return path_out


@pytest.fixture()
def fake_rapl(tmp_path) -> Path:
    """A fake /sys/class/powercap tree with one package, two sub-domains and an MMIO duplicate."""
    root = tmp_path / "powercap"

    def domain(name_dir: str, name: str, energy_uj: int, max_range: int = 262_143_328_850) -> None:
        d = root / name_dir
        d.mkdir(parents=True)
        (d / "name").write_text(name + "\n")
        (d / "energy_uj").write_text(f"{energy_uj}\n")
        (d / "max_energy_range_uj").write_text(f"{max_range}\n")

    domain("intel-rapl:0", "package-0", 1_000_000)
    domain("intel-rapl:0:0", "core", 400_000)  # sub-domain: must not be counted twice
    domain("intel-rapl:0:1", "dram", 200_000)
    domain("intel-rapl-mmio:0", "package-0", 1_000_000)  # duplicate interface: ignored
    return root


def set_energy(root: Path, microjoules: int, domain: str = "intel-rapl:0") -> None:
    (root / domain / "energy_uj").write_text(f"{microjoules}\n")


@pytest.fixture(scope="session")
def sleep_result() -> dict:
    """A small real result document made with the fake backend (no model file needed)."""
    backend = SleepBackend(latency_ms=1.0, spin=True)
    backend.load()
    cfg = RunConfig(iterations=60, warmup=5, bootstrap_resamples=100, measure_power=False)
    return run_benchmark(backend, cfg, label="fixture-run", env={"timestamp_utc": "2026-01-01T00:00:00+00:00", "packages": {}})


def make_result(label: str, p50: float, ci: tuple[float, float], fps: float, **extra) -> dict:
    """A hand-made minimal result for compare/figure tests."""
    result = {
        "schema_version": "1.0",
        "run_id": f"id-{label}",
        "created_utc": "2026-01-01T00:00:00+00:00",
        "label": label,
        "model": {"file": f"{label}.onnx", "sha256": f"sha-{label}", "precision": "fp32"},
        "backend": {"name": "onnxruntime", "intra_op_threads": 1, "inter_op_threads": 0,
                    "graph_optimization": "all", "active_providers": ["CPUExecutionProvider"]},
        "config": {"mode": "model", "batch_size": 1, "streams": 1, "task": "none", "input_source": "synthetic"},
        "environment": {"cpu_model": "Test CPU", "machine": "x86_64", "python_version": "3.12.0",
                        "packages": {"onnxruntime": "1.20.0"}},
        "latency": {"n": 1000, "p50_ms": p50, "p90_ms": p50 * 1.1, "p95_ms": p50 * 1.2, "p99_ms": p50 * 1.5,
                    "mean_ms": p50, "std_ms": 0.1, "min_ms": p50 * 0.9, "max_ms": p50 * 2,
                    "median_ci95_ms": list(ci), "outliers": {"fraction": 0.0},
                    "drift": {"checked": True, "stable": True, "relative_change": 0.0}},
        "throughput": {"items_per_second": fps, "calls_per_second": fps, "wall_time_s": 1.0,
                       "n_calls": 1000, "n_items": 1000, "batch_size": 1},
        "warnings": [],
        "samples": {"latency_ms": [p50] * 50},
    }
    for key, value in extra.items():
        result[key] = value
    return result


def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
