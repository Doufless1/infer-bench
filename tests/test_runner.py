import gc
import threading

import numpy as np
import pytest

from infer_bench import results
from infer_bench.backends import OrtBackend, SleepBackend
from infer_bench.backends.base import TensorSpec
from infer_bench.power import RaplMeter
from infer_bench.runner import RunConfig, run_benchmark

from conftest import set_energy

ENV = {"timestamp_utc": "2026-01-01T00:00:00+00:00", "packages": {}}


def run(backend, **kwargs):
    kwargs.setdefault("measure_power", False)
    kwargs.setdefault("bootstrap_resamples", 100)
    return run_benchmark(backend, RunConfig(**kwargs), label="t", env=ENV)


def sleeper(latency_ms=2.0, spin=True):
    backend = SleepBackend(latency_ms=latency_ms, spin=spin)
    backend.load()
    return backend


class CountingBackend(SleepBackend):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0
        self._lock = threading.Lock()

    def run(self, feed):
        with self._lock:
            self.calls += 1
        return super().run(feed)


# ------------------------------------------------------------- measurement accuracy

@pytest.mark.timing
def test_the_harness_measures_a_known_latency_accurately():
    """The most important test: a fake model that takes exactly 2 ms must be measured as ~2 ms."""
    result = run(sleeper(2.0), iterations=200, warmup=10)
    assert result["latency"]["p50_ms"] == pytest.approx(2.0, rel=0.10)
    assert result["latency"]["min_ms"] >= 2.0  # a call can never be faster than its busy-wait
    assert result["throughput"]["items_per_second"] == pytest.approx(500.0, rel=0.15)


def test_warmup_calls_are_not_timed():
    backend = CountingBackend(latency_ms=0.2, spin=True)
    backend.load()
    result = run(backend, iterations=40, warmup=15)
    assert backend.calls == 55
    assert result["latency"]["n"] == 40
    assert len(result["samples"]["latency_ms"]) == 40


@pytest.mark.timing
def test_duration_mode_runs_for_about_the_requested_time():
    result = run(sleeper(1.0), duration_s=0.5, warmup=2)
    assert 0.45 <= result["throughput"]["wall_time_s"] <= 0.75
    assert 250 <= result["latency"]["n"] <= 520


def test_batch_size_scales_items_but_not_calls():
    result = run(sleeper(1.0), iterations=30, warmup=2, batch_size=4)
    thr = result["throughput"]
    assert thr["n_calls"] == 30 and thr["n_items"] == 120 and thr["batch_size"] == 4
    assert thr["items_per_second"] == pytest.approx(4 * thr["calls_per_second"])
    assert any("whole batch" in w for w in result["warnings"])


def test_garbage_collector_state_is_restored():
    assert gc.isenabled()
    run(sleeper(0.2), iterations=20, warmup=2)
    assert gc.isenabled()
    gc.disable()
    try:
        run(sleeper(0.2), iterations=20, warmup=2)
        assert not gc.isenabled()  # was off before, stays off
    finally:
        gc.enable()


def test_garbage_collector_is_paused_during_the_timed_region():
    seen = []

    class Probe(SleepBackend):
        def run(self, feed):
            seen.append(gc.isenabled())
            return super().run(feed)

    backend = Probe(latency_ms=0.1)
    backend.load()
    run(backend, iterations=10, warmup=3)
    assert seen[:3] == [True] * 3 and seen[3:] == [False] * 10  # warm-up with GC, timed region without
    seen.clear()
    run(backend, iterations=5, warmup=0, pause_gc=False)
    assert seen == [True] * 5


# ------------------------------------------------------------------ result document

def test_result_document_is_complete_and_valid(tmp_path):
    result = run(sleeper(0.5), iterations=50, warmup=3)
    path = results.save_result(result, tmp_path / "r.json")
    assert results.validate_result(results.load_result(path)) == []
    for key in ("schema_version", "tool", "run_id", "model", "backend", "config", "environment", "latency",
                "throughput", "power", "warnings", "samples"):
        assert key in result
    assert result["backend"]["name"] == "sleep"
    assert result["config"]["iterations"] == 50 and result["config"]["input_source"] == "synthetic"
    assert result["power"]["measured"] is False
    assert result["model"]["inputs"][0]["name"] == "images"


def test_few_samples_trigger_a_warning():
    result = run(sleeper(0.2), iterations=20, warmup=2)
    assert any("timed samples" in w for w in result["warnings"])


def test_a_drifting_backend_is_flagged():
    class Slowing(SleepBackend):
        n = 0

        def run(self, feed):
            Slowing.n += 1
            self.latency_ms = 1.0 if Slowing.n <= 50 else 1.5  # a step change half-way through the run
            return super().run(feed)

    backend = Slowing()
    backend.load()
    result = run(backend, iterations=100, warmup=0)
    assert result["latency"]["drift"]["stable"] is False
    assert any("drifted" in w for w in result["warnings"])


# --------------------------------------------------------------------- multi-stream

@pytest.mark.timing
def test_several_streams_overlap_when_the_backend_releases_the_gil():
    single = run(sleeper(5.0, spin=False), duration_s=0.5, warmup=1)
    multi = run(sleeper(5.0, spin=False), duration_s=0.8, warmup=1, streams=4)
    ratio = multi["throughput"]["items_per_second"] / single["throughput"]["items_per_second"]
    assert ratio > 2.5  # ideal is 4
    block = multi["streams"]
    assert block["n_streams"] == 4 and len(block["per_stream_items_per_second"]) == 4
    assert sum(block["per_stream_calls"]) == multi["throughput"]["n_calls"]
    assert min(block["per_stream_items_per_second"]) > 0.7 * max(block["per_stream_items_per_second"])  # fair share


def test_streams_default_to_a_fixed_duration(monkeypatch):
    from infer_bench import runner

    assert runner.DEFAULT_MULTI_STREAM_SECONDS == 5.0  # what the README promises
    monkeypatch.setattr(runner, "DEFAULT_MULTI_STREAM_SECONDS", 0.3)  # keep the test fast
    result = run(sleeper(2.0, spin=False), streams=2, warmup=1)
    assert result["streams"]["duration_s"] == 0.3


def test_a_failing_stream_raises_instead_of_hanging():
    class Broken(SleepBackend):
        def run(self, feed):
            raise RuntimeError("boom")

    backend = Broken()
    backend.load()
    outcome: list = []

    def target():
        try:
            run(backend, streams=3, duration_s=0.2, warmup=1)
        except Exception as exc:  # noqa: BLE001
            outcome.append(exc)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=20)
    assert not t.is_alive(), "run_benchmark hung after a stream failed"
    assert outcome and "boom" in str(outcome[0])


# ----------------------------------------------------------------- pipeline mode

@pytest.mark.timing
def test_pipeline_mode_times_each_stage(models):
    backend = OrtBackend(intra_op_threads=1)
    backend.load(models["cls"])
    result = run(backend, mode="pipeline", task="classification", iterations=40, warmup=5)
    stages = result["stages"]
    pre, inf, post = (stages[k]["p50_ms"] for k in ("preprocess", "inference", "postprocess"))
    assert pre > 0 and inf > 0 and post > 0
    total = result["latency"]["p50_ms"]
    assert total == pytest.approx(pre + inf + post, rel=0.25)
    assert result["config"]["preprocess"] == {"height": 224, "width": 224, "mode": "resize"}


def test_pipeline_mode_with_detection_and_batches(models):
    backend = OrtBackend(intra_op_threads=1)
    backend.load(models["det"])
    result = run(backend, mode="pipeline", task="detection", iterations=15, warmup=2, batch_size=2)
    assert result["config"]["preprocess"]["mode"] == "letterbox"
    assert result["throughput"]["n_items"] == 30


def test_pipeline_mode_with_real_image_arrays(models):
    backend = OrtBackend(intra_op_threads=1)
    backend.load(models["cls"])
    images = [np.full((100, 150, 3), v, dtype=np.uint8) for v in (30, 120, 220)]
    cfg = RunConfig(mode="pipeline", task="classification", iterations=10, warmup=1, measure_power=False,
                    bootstrap_resamples=50)
    result = run_benchmark(backend, cfg, label="img", env=ENV, images=images)
    assert result["config"]["input_source"] == "images" and result["config"]["n_images"] == 3
    assert not any("Synthetic" in w for w in result["warnings"])


def test_model_mode_with_images_uses_them(models):
    from infer_bench.preprocess import PreprocessConfig

    backend = OrtBackend(intra_op_threads=1)
    backend.load(models["cls"])
    images = [np.full((64, 64, 3), 128, dtype=np.uint8)]
    cfg = RunConfig(iterations=10, warmup=1, measure_power=False, bootstrap_resamples=50,
                    preprocess=PreprocessConfig(224, 224, mode="resize"))
    result = run_benchmark(backend, cfg, label="img", env=ENV, images=images)
    assert result["config"]["input_source"] == "images"


def test_pipeline_mode_needs_an_image_model():
    backend = SleepBackend()
    backend.load()
    backend._inputs = [TensorSpec("a", ("batch", 10), "float32")]
    with pytest.raises(ValueError, match="4-D"):
        run(backend, mode="pipeline", iterations=2, warmup=0)


# ------------------------------------------------------------------ configuration

@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"mode": "fast"}, "mode must be"),
        ({"task": "segmentation"}, "task must be"),
        ({"batch_size": 0}, "batch_size"),
        ({"warmup": -1}, "warmup"),
        ({"streams": 0}, "streams"),
        ({"iterations": 0}, "iterations"),
        ({"duration_s": -1.0}, "duration"),
        ({"mode": "model", "task": "detection"}, "pipeline mode"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RunConfig(**kwargs).validate()


# ----------------------------------------------------------------------------- power

def test_manual_power_numbers_flow_into_the_result():
    result = run(sleeper(1.0), iterations=50, warmup=2, power_watts=10.0, idle_watts=4.0)
    p = result["power"]
    assert p["source"] == "manual" and p["avg_power_w"] == 10.0 and p["net_power_w"] == 6.0
    fps = result["throughput"]["items_per_second"]
    assert p["items_per_s_per_w"] == pytest.approx(fps / 10.0, rel=0.02)
    assert any("supplied by the user" in n for n in p["notes"])


def test_rapl_energy_is_measured_over_the_timed_region_only(fake_rapl):
    class Burner(SleepBackend):
        """Every call 'uses' 10 mJ: it bumps the fake energy counter."""

        def __init__(self):
            super().__init__(latency_ms=0.5, spin=True)
            self.energy_uj = 1_000_000

        def run(self, feed):
            self.energy_uj += 10_000
            set_energy(fake_rapl, self.energy_uj)
            return super().run(feed)

    backend = Burner()
    backend.load()
    cfg = RunConfig(iterations=40, warmup=10, idle_seconds=0.0, bootstrap_resamples=50)
    result = run_benchmark(backend, cfg, label="rapl", env=ENV, meter=RaplMeter(fake_rapl))
    p = result["power"]
    assert p["source"] == "rapl" and p["scope"] == "cpu_package_only"
    assert p["energy_j"] == pytest.approx(0.4)  # 40 timed calls x 10 mJ; the 10 warm-up calls are excluded
    assert p["energy_per_item_mj"] == pytest.approx(10.0)


def test_unavailable_meter_is_explained_not_hidden(tmp_path):
    cfg = RunConfig(iterations=20, warmup=1, bootstrap_resamples=50)
    result = run_benchmark(sleeper(0.2), cfg, label="x", env=ENV, meter=RaplMeter(tmp_path / "nope"))
    assert result["power"]["measured"] is False
    assert "RAPL" in result["power"]["reason"] and "--power-watts" in result["power"]["reason"]
