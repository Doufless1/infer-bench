"""The benchmark loop.

What is measured, and how:

* Every call is timed individually with ``time.perf_counter_ns`` (a monotonic
  clock with nanosecond resolution). Only the call itself is inside the timed
  region; list appends, input selection and so on are outside it.
* A warm-up phase runs first and is discarded: the first calls pay for lazy
  initialisation, memory allocation and cold caches.
* Python's garbage collector is paused during the timed region so a
  collection cannot land in the middle of a measurement.
* Throughput is items processed divided by *wall-clock* time of the whole
  loop, not the sum of the latencies.
* Two modes: ``model`` times only the model call; ``pipeline`` times
  pre-processing, the model and post-processing separately, because in a
  real application the model is often not the only cost.
* ``streams > 1`` runs that many threads against the same loaded model for a
  fixed time (all streams busy for the whole window) and reports total and
  per-stream throughput.
"""

from __future__ import annotations

import contextlib
import gc
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import SCHEMA_VERSION
from .backends.base import Backend, TensorSpec
from .env import capture_environment, noise_warnings
from .inputs import image_feeds, model_image_size, synthetic_feeds
from .power import RaplMeter, build_power_report, manual_reading
from .postprocess import decode_yolo, softmax, topk_indices
from .preprocess import PreprocessConfig, preprocess, synthetic_image, unmap_boxes
from .results import limit_samples, new_run_id, tool_block
from .stats import drift_check, sample_size_warnings, summarize

MODES = ("model", "pipeline")
TASKS = ("none", "classification", "detection")
DEFAULT_MULTI_STREAM_SECONDS = 5.0
DRIFT_THRESHOLD = 0.05  # a >5% change between the two halves of a run counts as drift


@dataclass
class RunConfig:
    """Everything that defines one benchmark run (all of it ends up in the results file)."""

    mode: str = "model"  # "model" or "pipeline"
    task: str = "none"  # pipeline post-processing: "none", "classification" or "detection"
    batch_size: int = 1
    warmup: int = 50
    iterations: int = 500
    duration_s: float | None = None  # run for this long instead of a fixed iteration count
    streams: int = 1
    seed: int = 0
    pool_size: int = 8  # number of different inputs cycled through
    input_size: tuple[int, int] | None = None  # (height, width) for dynamic image models
    shape_overrides: dict[str, tuple[int, ...]] = field(default_factory=dict)
    preprocess: PreprocessConfig | None = None  # pipeline mode; derived from the model if None
    source_size: tuple[int, int] = (480, 640)  # (height, width) of synthetic source images
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    bootstrap_resamples: int = 1000
    pause_gc: bool = True
    measure_power: bool = True
    power_watts: float | None = None  # user-supplied average power (e.g. wall meter)
    idle_watts: float | None = None  # user-supplied idle power
    idle_seconds: float = 2.0  # RAPL idle baseline duration (0 = skip)

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.warmup < 0:
            raise ValueError("warmup cannot be negative")
        if self.streams < 1:
            raise ValueError("streams must be at least 1")
        if self.duration_s is None and self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.duration_s is not None and self.duration_s <= 0:
            raise ValueError("duration must be positive")
        if self.mode == "model" and self.task != "none":
            raise ValueError("--task only applies to pipeline mode (post-processing is not part of model mode)")

    def to_dict(self) -> dict[str, Any]:
        pre = self.preprocess
        return {
            "mode": self.mode,
            "task": self.task,
            "batch_size": self.batch_size,
            "warmup": self.warmup,
            "iterations": self.iterations,
            "duration_s": self.duration_s,
            "streams": self.streams,
            "seed": self.seed,
            "pool_size": self.pool_size,
            "input_size": list(self.input_size) if self.input_size else None,
            "shape_overrides": {k: list(v) for k, v in self.shape_overrides.items()},
            "preprocess": None
            if pre is None
            else {
                "height": pre.height, "width": pre.width, "mode": pre.mode, "scale": pre.scale,
                "mean": list(pre.mean), "std": list(pre.std), "pad_value": pre.pad_value,
            },
            "source_size": list(self.source_size),
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "bootstrap_resamples": self.bootstrap_resamples,
            "pause_gc": self.pause_gc,
            "power_watts": self.power_watts,
            "idle_watts": self.idle_watts,
            "idle_seconds": self.idle_seconds,
        }


# ------------------------------------------------------------------- workloads

class ModelWorkload:
    """Times only the model call."""

    def __init__(self, backend: Backend, feeds: list[dict[str, np.ndarray]]):
        self.backend = backend
        self.feeds = feeds

    def step(self, i: int) -> tuple[int, int, int, int]:
        feed = self.feeds[i % len(self.feeds)]
        t0 = time.perf_counter_ns()
        self.backend.run(feed)
        t1 = time.perf_counter_ns()
        return 0, t1 - t0, 0, t1 - t0


class PipelineWorkload:
    """Times pre-processing, the model call and post-processing separately."""

    def __init__(
        self,
        backend: Backend,
        images: list[np.ndarray],
        spec: TensorSpec,
        pre_cfg: PreprocessConfig,
        task: str,
        batch_size: int,
        conf_threshold: float,
        iou_threshold: float,
    ):
        self.backend, self.images, self.spec = backend, images, spec
        self.pre_cfg, self.task, self.batch = pre_cfg, task, batch_size
        self.conf, self.iou = conf_threshold, iou_threshold

    def _post(self, outputs: list[np.ndarray], metas: list[Any]) -> None:
        if self.task == "classification":
            probs = softmax(np.asarray(outputs[0]).reshape(self.batch, -1), axis=1)
            for row in probs:
                topk_indices(row, 5)
        elif self.task == "detection":
            for b in range(self.batch):
                dets = decode_yolo(
                    outputs[0][b : b + 1], conf_threshold=self.conf, iou_threshold=self.iou
                )
                unmap_boxes(dets.boxes, metas[b])

    def step(self, i: int) -> tuple[int, int, int, int]:
        n = len(self.images)
        t0 = time.perf_counter_ns()
        tensors, metas = [], []
        for j in range(self.batch):
            tensor, meta = preprocess(self.images[(i * self.batch + j) % n], self.pre_cfg)
            tensors.append(tensor)
            metas.append(meta)
        batch_tensor = tensors[0] if self.batch == 1 else np.concatenate(tensors, axis=0)
        batch_tensor = batch_tensor.astype(self.spec.dtype, copy=False)
        t1 = time.perf_counter_ns()
        outputs = self.backend.run({self.spec.name: batch_tensor})
        t2 = time.perf_counter_ns()
        self._post(outputs, metas)
        t3 = time.perf_counter_ns()
        return t1 - t0, t2 - t1, t3 - t2, t3 - t0


# ---------------------------------------------------------------------- loops

@dataclass
class _Timings:
    total: list[int] = field(default_factory=list)
    pre: list[int] = field(default_factory=list)
    infer: list[int] = field(default_factory=list)
    post: list[int] = field(default_factory=list)
    t_start: float = 0.0
    t_end: float = 0.0

    def add(self, step: tuple[int, int, int, int]) -> None:
        self.pre.append(step[0])
        self.infer.append(step[1])
        self.post.append(step[2])
        self.total.append(step[3])


def _timed_loop(
    workload: Any,
    *,
    iterations: int | None,
    deadline: float | None,
    t_start: float | None = None,
) -> _Timings:
    """Run ``workload.step`` until the iteration count or the deadline is reached."""
    out = _Timings()
    clock = time.perf_counter
    step = workload.step
    out.t_start = clock() if t_start is None else t_start
    i = 0
    while True:
        if iterations is not None and i >= iterations:
            break
        if deadline is not None and clock() >= deadline:
            break
        out.add(step(i))
        i += 1
    out.t_end = clock()
    return out


@contextlib.contextmanager
def _gc_paused(enabled: bool):
    was_enabled = gc.isenabled()
    if enabled:
        gc.collect()
        gc.disable()
    try:
        yield
    finally:
        if enabled and was_enabled:
            gc.enable()


def _run_single_stream(workload: Any, cfg: RunConfig, meter_ctx: Any) -> list[_Timings]:
    for i in range(cfg.warmup):
        workload.step(i)
    with _gc_paused(cfg.pause_gc), meter_ctx:
        t0 = time.perf_counter()
        deadline = None if cfg.duration_s is None else t0 + cfg.duration_s
        iterations = None if cfg.duration_s is not None else cfg.iterations
        timings = _timed_loop(workload, iterations=iterations, deadline=deadline, t_start=t0)
    return [timings]


def _run_multi_stream(workloads: list[Any], cfg: RunConfig, duration_s: float, meter_ctx: Any) -> list[_Timings]:
    n = len(workloads)
    ready = threading.Barrier(n + 1)
    go = threading.Event()
    shared: dict[str, float] = {}
    results: list[_Timings | None] = [None] * n
    errors: list[BaseException] = []

    def worker(k: int) -> None:
        try:
            for i in range(cfg.warmup):
                workloads[k].step(i)
            ready.wait()
            go.wait()
            results[k] = _timed_loop(
                workloads[k], iterations=None, deadline=shared["deadline"], t_start=shared["t_start"]
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised in the main thread
            errors.append(exc)
            ready.abort()
            go.set()

    threads = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(n)]
    for t in threads:
        t.start()
    try:
        ready.wait()
    except threading.BrokenBarrierError:
        for t in threads:
            t.join()
        raise errors[0] if errors else RuntimeError("a benchmark stream failed during warm-up")

    with _gc_paused(cfg.pause_gc), meter_ctx:
        shared["t_start"] = time.perf_counter()
        shared["deadline"] = shared["t_start"] + duration_s
        go.set()
        for t in threads:
            t.join()
    if errors:
        raise errors[0]
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------- driver

def _build_workloads(
    backend: Backend, cfg: RunConfig, images: list[np.ndarray] | None
) -> tuple[list[Any], dict[str, Any]]:
    """One workload per stream, plus a description of the input source."""
    specs = backend.inputs
    info: dict[str, Any] = {"input_source": "images" if images else "synthetic", "n_images": len(images or [])}

    if cfg.mode == "model":
        if images:
            if len(specs) != 1:
                raise ValueError("--images needs a model with exactly one input")
            pre = cfg.preprocess or _default_preprocess(specs[0], cfg)
            feeds = image_feeds(images, specs[0], pre, cfg.batch_size)
        else:
            feeds = synthetic_feeds(
                specs, cfg.batch_size, count=cfg.pool_size, seed=cfg.seed,
                overrides=cfg.shape_overrides, image_size=cfg.input_size,
            )
        return [ModelWorkload(backend, feeds) for _ in range(cfg.streams)], info

    # pipeline mode
    if len(specs) != 1 or len(specs[0].shape) != 4:
        raise ValueError("pipeline mode needs a model with exactly one 4-D (NCHW) image input")
    spec = specs[0]
    pre = cfg.preprocess or _default_preprocess(spec, cfg)
    if images:
        sources = images
    else:
        rng = np.random.default_rng(cfg.seed)
        h, w = cfg.source_size
        sources = [synthetic_image(rng, w, h) for _ in range(max(1, cfg.pool_size))]
        info["synthetic_source_size"] = [h, w]
    info["preprocess"] = {"height": pre.height, "width": pre.width, "mode": pre.mode}
    workloads = [
        PipelineWorkload(
            backend, sources, spec, pre, cfg.task, cfg.batch_size, cfg.conf_threshold, cfg.iou_threshold
        )
        for _ in range(cfg.streams)
    ]
    return workloads, info


def _default_preprocess(spec: TensorSpec, cfg: RunConfig) -> PreprocessConfig:
    h, w = model_image_size(spec, cfg.input_size)
    mode = "letterbox" if cfg.task == "detection" else "resize"
    return PreprocessConfig(height=h, width=w, mode=mode)


def run_benchmark(
    backend: Backend,
    cfg: RunConfig,
    *,
    label: str = "run",
    model_info: dict[str, Any] | None = None,
    images: list[np.ndarray] | None = None,
    meter: RaplMeter | None = None,
    env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one benchmark and return the results document (see results.py)."""
    cfg.validate()
    env = env if env is not None else capture_environment()
    workloads, input_info = _build_workloads(backend, cfg, images)

    # ---- power setup
    use_rapl = bool(cfg.measure_power and cfg.power_watts is None and meter is not None and meter.available)
    idle_w = cfg.idle_watts
    if use_rapl and idle_w is None and cfg.idle_seconds > 0:
        idle_w = meter.measure_idle(cfg.idle_seconds).avg_power_w  # type: ignore[union-attr]
    meter_ctx = meter.measure() if use_rapl else contextlib.nullcontext()  # type: ignore[union-attr]

    # ---- the measurement
    if cfg.streams == 1:
        runs = _run_single_stream(workloads[0], cfg, meter_ctx)
        duration_used = cfg.duration_s
    else:
        duration_used = cfg.duration_s if cfg.duration_s is not None else DEFAULT_MULTI_STREAM_SECONDS
        runs = _run_multi_stream(workloads, cfg, duration_used, meter_ctx)

    # ---- numbers
    t_start = min(r.t_start for r in runs)
    wall_s = max(r.t_end for r in runs) - t_start
    calls = sum(len(r.total) for r in runs)
    items = calls * cfg.batch_size
    total_ms = np.concatenate([np.asarray(r.total, dtype=np.float64) for r in runs]) / 1e6
    latency = summarize(total_ms, resamples=cfg.bootstrap_resamples, seed=cfg.seed)
    if cfg.streams > 1:
        # Samples of different streams are interleaved in time, so check drift per stream.
        drifts = [
            drift_check(np.asarray(r.total, dtype=np.float64) / 1e6, threshold=DRIFT_THRESHOLD)
            for r in runs
            if r.total
        ]
        checked = [d for d in drifts if d.get("checked")]
        if checked:
            latency["drift"] = max(checked, key=lambda d: abs(d["relative_change"]))
            latency["drift"]["scope"] = "worst of the per-stream checks"
    samples, truncated = limit_samples(total_ms)

    throughput = {
        "items_per_second": items / wall_s if wall_s > 0 else 0.0,
        "calls_per_second": calls / wall_s if wall_s > 0 else 0.0,
        "wall_time_s": wall_s,
        "n_calls": calls,
        "n_items": items,
        "batch_size": cfg.batch_size,
    }

    stages = None
    if cfg.mode == "pipeline":
        def stage(attr: str) -> dict[str, Any]:
            arr = np.concatenate([np.asarray(getattr(r, attr), dtype=np.float64) for r in runs]) / 1e6
            return summarize(arr, resamples=min(200, cfg.bootstrap_resamples), seed=cfg.seed)

        stages = {"preprocess": stage("pre"), "inference": stage("infer"), "postprocess": stage("post")}

    streams_block = None
    if cfg.streams > 1:
        streams_block = {
            "n_streams": cfg.streams,
            "duration_s": duration_used,
            "per_stream_items_per_second": [
                len(r.total) * cfg.batch_size / (r.t_end - t_start) for r in runs
            ],
            "per_stream_calls": [len(r.total) for r in runs],
        }

    # ---- power
    power: dict[str, Any]
    if cfg.power_watts is not None:
        power = build_power_report(
            manual_reading(cfg.power_watts, wall_s), items=items, idle_power_w=cfg.idle_watts,
            notes=["Power was supplied by the user (--power-watts), not measured by this tool."],
        )
    elif use_rapl:
        reading = meter_ctx.reading  # type: ignore[union-attr]
        power = build_power_report(reading, items=items, idle_power_w=idle_w)
    else:
        if not cfg.measure_power:
            reason = "power measurement disabled"
        elif meter is None:
            reason = "no RAPL meter given"
        else:
            reason = meter.reason_unavailable or "RAPL unavailable"
        power = {"measured": False, "reason": reason + ". Use --power-watts to supply a measured value."}

    # ---- warnings
    warnings = list(sample_size_warnings(latency["n"]))
    drift = latency["drift"]
    if drift.get("checked") and not drift["stable"]:
        warnings.append(
            f"Latency drifted {drift['relative_change'] * 100:+.1f}% between the first and second half of the run: "
            "possible throttling or background load. Do not trust a single summary number."
        )
    if latency["outliers"]["fraction"] > 0.05:
        warnings.append(
            f"{latency['outliers']['fraction'] * 100:.1f}% of the samples are outliers (spikes): check for background activity."
        )
    lo, hi = latency["median_ci95_ms"]
    if latency["p50_ms"] > 0 and (hi - lo) / latency["p50_ms"] > 0.05:
        warnings.append(
            "The 95% confidence interval of the median is wider than 5% of the median: "
            "differences smaller than that are not reliable."
        )
    if input_info["input_source"] == "synthetic":
        warnings.append("Synthetic inputs were used: fine for speed, not meaningful for accuracy.")
    if cfg.batch_size > 1:
        warnings.append("Latency is per call, and one call processes the whole batch.")
    if cfg.streams > 1 and cfg.mode == "pipeline":
        warnings.append(
            "Multi-stream runs use Python threads: pre- and post-processing share the GIL, so scaling is "
            "limited compared with a native pipeline."
        )
    warnings.extend(noise_warnings(env))

    model_block = dict(model_info or {})
    model_block["inputs"] = [s.to_dict() for s in backend.inputs]
    model_block["outputs"] = [s.to_dict() for s in backend.outputs]

    config = cfg.to_dict()
    config.update(input_info)
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": tool_block(),
        "run_id": new_run_id(),
        "created_utc": env.get("timestamp_utc"),
        "label": label,
        "model": model_block,
        "backend": backend.describe(),
        "config": config,
        "environment": env,
        "latency": latency,
        "throughput": throughput,
        "stages": stages,
        "streams": streams_block,
        "power": power,
        "accuracy": None,
        "warnings": warnings,
        "samples": {"latency_ms": samples, "truncated": truncated},
    }
