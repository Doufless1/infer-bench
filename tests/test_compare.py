import pytest

from infer_bench.compare import compare_results

from conftest import make_result


def test_identical_runs_are_within_noise():
    a = make_result("a", 2.0, (1.95, 2.05), 500.0)
    cmp = compare_results(a, make_result("b", 2.0, (1.96, 2.04), 500.0))
    assert cmp["throughput_ratio"] == pytest.approx(1.0)
    assert cmp["latency_ratio"]["p50"] == pytest.approx(1.0)
    assert cmp["median_ci_overlap"] is True
    assert "within measurement noise" in cmp["verdict"] and "Do not claim" in cmp["verdict"]
    assert cmp["warnings"] == []


def test_clearly_faster_run_with_disjoint_intervals():
    base = make_result("fp32", 2.0, (1.95, 2.05), 500.0)
    other = make_result("int8", 1.0, (0.98, 1.02), 1000.0)
    cmp = compare_results(base, other)
    assert cmp["throughput_ratio"] == pytest.approx(2.0)
    assert cmp["latency_ratio"]["p50"] == pytest.approx(0.5)
    assert cmp["median_ci_overlap"] is False
    assert "2.00x lower" in cmp["verdict"]


def test_slower_run_is_described_as_higher_latency():
    cmp = compare_results(make_result("a", 1.0, (0.98, 1.02), 1000.0), make_result("b", 2.0, (1.95, 2.05), 500.0))
    assert "2.00x higher" in cmp["verdict"]


def test_setup_differences_produce_warnings():
    a = make_result("a", 1.0, (0.98, 1.02), 1000.0)
    b = make_result("b", 1.0, (0.98, 1.02), 1000.0)
    b["config"]["batch_size"] = 8
    b["config"]["mode"] = "pipeline"
    b["backend"]["intra_op_threads"] = 4
    b["environment"]["cpu_model"] = "Other CPU"
    b["environment"]["packages"]["onnxruntime"] = "1.21.0"
    text = " ".join(compare_results(a, b)["warnings"])
    for fragment in ("batch size", "mode", "intra_op_threads", "CPU model differs", "onnxruntime"):
        assert fragment in text


def test_few_samples_warning():
    a = make_result("a", 1.0, (0.98, 1.02), 1000.0)
    b = make_result("b", 1.0, (0.98, 1.02), 1000.0)
    b["latency"]["n"] = 30
    assert any("only 30 samples" in w for w in compare_results(a, b)["warnings"])


def test_efficiency_and_accuracy_blocks_are_carried_over():
    a = make_result("a", 1.0, (0.98, 1.02), 1000.0, power={"items_per_s_per_w": 50.0, "avg_power_w": 20.0, "source": "manual"})
    b = make_result("b", 1.0, (0.98, 1.02), 1000.0, power={"items_per_s_per_w": 100.0, "avg_power_w": 10.0, "source": "manual"},
                    accuracy={"cosine_similarity_mean": 0.99, "top1_agreement": 0.97, "detection": {"f1_iou50": 0.9}})
    cmp = compare_results(a, b)
    assert cmp["power"]["items_per_s_per_w_ratio"] == pytest.approx(2.0)
    assert cmp["accuracy"] == {"cosine_similarity_mean": 0.99, "top1_agreement": 0.97, "detection_f1": 0.9}
    assert compare_results(b, a)["accuracy"] is None


def test_missing_data_is_handled():
    a = make_result("a", 1.0, (0.98, 1.02), 1000.0)
    b = make_result("b", 1.0, (0.98, 1.02), 1000.0)
    del b["latency"]["median_ci95_ms"]
    del b["latency"]["p50_ms"]
    cmp = compare_results(a, b)
    assert cmp["verdict"] == "Not enough data to compare."
