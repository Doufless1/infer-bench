"""The results file: one self-describing JSON document per benchmark run.

Schema (version 1.0), top level keys:

    schema_version   "1.0"
    tool             {"name", "version"}
    run_id           unique id (UTC timestamp + random suffix)
    created_utc      ISO timestamp
    label            human name of the run
    model            file, sha256, size, precision, inputs/outputs, ...
    backend          runtime name/version/providers/threads
    config           everything that was set for the run (mode, batch, streams, ...)
    environment      machine + software versions (see env.py)
    latency          p50/p90/p95/p99, mean, CI of the median, outliers, drift
    throughput       items per second, wall time, item count
    stages           (pipeline mode) the same summary per stage
    streams          (multi-stream) per-stream throughput
    power            (optional) power / energy / efficiency
    accuracy         (optional) agreement with a reference model
    warnings         plain-language reasons not to over-trust the numbers
    samples          the raw per-call latencies in milliseconds

The raw samples are stored so that anyone can recompute the statistics
instead of trusting the summary.
"""

from __future__ import annotations

import json
import math
import re
import secrets
import datetime as _dt
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import SCHEMA_VERSION, __version__

REQUIRED_KEYS = (
    "schema_version",
    "run_id",
    "created_utc",
    "label",
    "model",
    "backend",
    "config",
    "environment",
    "latency",
    "throughput",
)
MAX_STORED_SAMPLES = 100_000


def new_run_id() -> str:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-").lower()
    return slug or "run"


def tool_block() -> dict[str, str]:
    return {"name": "infer-bench", "version": __version__}


def _sanitize(obj: Any) -> Any:
    """Make an object strictly JSON-serialisable (numpy -> python, NaN -> null)."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def limit_samples(samples_ms: np.ndarray | list[float]) -> tuple[list[float], bool]:
    """Round and, for very long runs, thin the samples evenly. Returns (samples, truncated)."""
    arr = np.asarray(samples_ms, dtype=np.float64)
    truncated = False
    if arr.size > MAX_STORED_SAMPLES:
        idx = np.linspace(0, arr.size - 1, MAX_STORED_SAMPLES).astype(np.int64)
        arr = arr[idx]
        truncated = True
    return [round(float(x), 6) for x in arr], truncated


def save_result(result: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_sanitize(result), handle, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def load_result(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    problems = validate_result(data)
    if problems:
        raise ValueError(f"{path}: not a valid results file: " + "; ".join(problems))
    return data


def validate_result(data: Any) -> list[str]:
    """List of problems with a loaded results document (empty list = valid)."""
    if not isinstance(data, dict):
        return ["top level is not an object"]
    problems = [f"missing key '{k}'" for k in REQUIRED_KEYS if k not in data]
    version = str(data.get("schema_version", ""))
    if version and version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        problems.append(f"unsupported schema_version {version} (this tool reads {SCHEMA_VERSION})")
    latency = data.get("latency")
    if isinstance(latency, dict):
        for key in ("n", "p50_ms", "p95_ms", "p99_ms"):
            if key not in latency:
                problems.append(f"latency is missing '{key}'")
    thr = data.get("throughput")
    if isinstance(thr, dict) and "items_per_second" not in thr:
        problems.append("throughput is missing 'items_per_second'")
    return problems


def default_output_path(results_dir: str | Path, label: str, run_id: str) -> Path:
    return Path(results_dir) / f"{slugify(label)}_{run_id}.json"


def iter_results(folder: str | Path) -> Iterator[tuple[Path, dict[str, Any] | None, str | None]]:
    """Yield (path, result, error) for every ``*.json`` in a folder."""
    for path in sorted(Path(folder).glob("*.json")):
        try:
            yield path, load_result(path), None
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            yield path, None, str(exc)


def flatten_result(result: dict[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    """One flat row per run: what the summary table and the dashboard show."""
    lat = result.get("latency", {})
    thr = result.get("throughput", {})
    cfg = result.get("config", {})
    model = result.get("model", {})
    env = result.get("environment", {})
    power = result.get("power") or {}
    acc = result.get("accuracy") or {}
    ci = lat.get("median_ci95_ms") or [None, None]
    det = acc.get("detection") or {}
    return {
        "label": result.get("label"),
        "run_id": result.get("run_id"),
        "created_utc": result.get("created_utc"),
        "file": str(path) if path is not None else None,
        "model": model.get("file"),
        "model_sha256": model.get("sha256"),
        "reference_sha256": (acc.get("reference") or {}).get("sha256"),
        "precision": model.get("precision"),
        "backend": (result.get("backend") or {}).get("name"),
        "mode": cfg.get("mode"),
        "batch": cfg.get("batch_size"),
        "streams": cfg.get("streams", 1),
        "n": lat.get("n"),
        "fps": thr.get("items_per_second"),
        "p50_ms": lat.get("p50_ms"),
        "p90_ms": lat.get("p90_ms"),
        "p95_ms": lat.get("p95_ms"),
        "p99_ms": lat.get("p99_ms"),
        "ci_lo_ms": ci[0],
        "ci_hi_ms": ci[1],
        "stable": (lat.get("drift") or {}).get("stable"),
        "avg_power_w": power.get("avg_power_w"),
        "fps_per_w": power.get("items_per_s_per_w"),
        "mj_per_item": power.get("energy_per_item_mj"),
        "cosine": acc.get("cosine_similarity_mean"),
        "top1_agree": acc.get("top1_agreement"),
        "det_f1": det.get("f1_iou50"),
        "cpu": env.get("cpu_model"),
        "n_warnings": len(result.get("warnings", [])),
    }
