import time

import numpy as np
import pytest

from infer_bench.backends import BACKENDS, OrtBackend, SleepBackend, create_backend


@pytest.mark.timing
def test_sleep_backend_takes_the_requested_time_and_returns_the_right_shape():
    backend = SleepBackend(latency_ms=3.0, spin=True, output_size=7)
    backend.load()
    feed = {"images": np.zeros((2, 3, 32, 32), dtype=np.float32)}
    t0 = time.perf_counter()
    out = backend.run(feed)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert 3.0 <= elapsed_ms < 6.0
    assert out[0].shape == (2, 7)


@pytest.mark.timing
def test_sleep_backend_without_spin_releases_the_cpu():
    backend = SleepBackend(latency_ms=20.0, spin=False)
    t0 = time.process_time()
    backend.run({"images": np.zeros((1, 3, 32, 32), dtype=np.float32)})
    assert time.process_time() - t0 < 0.01  # almost no CPU time used while sleeping


def test_ort_backend_loads_and_runs_the_demo_classifier(models):
    backend = OrtBackend()
    backend.load(models["cls"])
    (inp,), (out,) = backend.inputs, backend.outputs
    assert inp.name == "images" and inp.dtype == "float32" and inp.shape == ("batch", 3, 224, 224)
    assert out.name == "logits" and out.shape == ("batch", 10)
    result = backend.run({"images": np.zeros((3, 3, 224, 224), dtype=np.float32)})
    assert result[0].shape == (3, 10)  # a dynamic batch dimension works


def test_ort_backend_describes_itself(models):
    backend = OrtBackend(intra_op_threads=2, graph_optimization="basic")
    backend.load(models["cls"])
    d = backend.describe()
    assert d["name"] == "onnxruntime" and d["intra_op_threads"] == 2 and d["graph_optimization"] == "basic"
    assert d["active_providers"] == ["CPUExecutionProvider"]
    assert d["version"]


def test_ort_backend_refuses_providers_that_are_not_installed(models):
    backend = OrtBackend(providers=["CUDAExecutionProvider"])
    with pytest.raises(RuntimeError, match="not available"):
        backend.load(models["cls"])  # silently falling back to the CPU would mislabel the results


def test_ort_backend_validates_the_optimisation_level():
    with pytest.raises(ValueError):
        OrtBackend(graph_optimization="turbo")


def test_ort_backend_can_be_used_from_several_threads(models):
    backend = OrtBackend(intra_op_threads=1)
    backend.load(models["cls"])
    feed = {"images": np.random.default_rng(0).random((1, 3, 224, 224), dtype=np.float32)}
    expected = backend.run(feed)[0]
    results = []

    import threading

    def work():
        results.append(backend.run(feed)[0])

    threads = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(results) == 4
    for r in results:
        np.testing.assert_allclose(r, expected, rtol=1e-5, atol=1e-6)


def test_create_backend_by_name_and_unknown_name():
    assert isinstance(create_backend("sleep", latency_ms=1.0), SleepBackend)
    assert isinstance(create_backend("onnxruntime"), OrtBackend)
    assert isinstance(create_backend("ort"), OrtBackend)
    assert set(BACKENDS) == {"onnxruntime", "sleep"}
    with pytest.raises(ValueError, match="unknown backend"):
        create_backend("tensorrt")


def test_backend_works_as_a_context_manager(models):
    with OrtBackend() as backend:
        backend.load(models["cls"])
        assert backend.run({"images": np.zeros((1, 3, 224, 224), dtype=np.float32)})
