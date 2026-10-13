import numpy as np
import pytest

from infer_bench import accuracy
from infer_bench.backends import OrtBackend
from infer_bench.backends.base import Backend, TensorSpec
from infer_bench.inputs import synthetic_feeds
from infer_bench.quantize import quantize_int8


class FnBackend(Backend):
    """A backend defined by a plain function, for building reference/candidate pairs."""

    name = "fn"

    def __init__(self, fn):
        self.fn = fn

    def load(self, model_path=None):
        pass

    @property
    def inputs(self):
        return [TensorSpec("x", (1, 4), "float32")]

    @property
    def outputs(self):
        return [TensorSpec("y", (1, 5), "float32")]

    def run(self, feed):
        return [self.fn(feed["x"])]


def feeds(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return [{"x": rng.standard_normal((1, 4)).astype(np.float32)} for _ in range(n)]


def logits(x):
    w = np.arange(20, dtype=np.float32).reshape(4, 5) / 10.0 - 1.0
    return x @ w


def test_helper_metrics():
    a = np.array([1.0, 0.0])
    assert accuracy.cosine_similarity(a, a) == pytest.approx(1.0)
    assert accuracy.cosine_similarity(a, np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert accuracy.cosine_similarity(np.zeros(3), np.zeros(3)) == 1.0
    assert accuracy.cosine_similarity(np.zeros(3), np.ones(3)) == 0.0
    assert accuracy.relative_l2_error(np.array([3.0, 4.0]), np.array([3.0, 4.0])) == 0.0
    assert accuracy.relative_l2_error(np.array([3.0, 4.0]), np.array([0.0, 0.0])) == pytest.approx(1.0)
    assert accuracy.relative_l2_error(np.zeros(2), np.ones(2)) == float("inf")


def test_identical_models_agree_perfectly():
    ref = FnBackend(logits)
    out = accuracy.compare_models(ref, FnBackend(logits), feeds(), task="classification")
    assert out["cosine_similarity_mean"] == pytest.approx(1.0)
    assert out["relative_l2_error_max"] == 0.0 and out["max_abs_error_max"] == 0.0
    assert out["top1_agreement"] == 1.0 and out["ref_top1_in_cand_top5"] == 1.0
    assert out["n_samples"] == out["n_compared"] == 30


def test_a_damaged_model_scores_lower():
    ref = FnBackend(logits)
    rng = np.random.default_rng(5)
    noisy = FnBackend(lambda x: logits(x) + rng.normal(0, 3.0, size=(1, 5)).astype(np.float32))
    out = accuracy.compare_models(ref, noisy, feeds(100), task="classification")
    assert out["cosine_similarity_mean"] < 0.95
    assert out["relative_l2_error_mean"] > 0.3
    assert out["top1_agreement"] < 0.8


def test_a_mildly_noisy_model_keeps_high_agreement_numbers():
    ref = FnBackend(logits)
    rng = np.random.default_rng(5)
    noisy = FnBackend(lambda x: logits(x) + rng.normal(0, 0.01, size=(1, 5)).astype(np.float32))
    out = accuracy.compare_models(ref, noisy, feeds(100), task="classification")
    assert out["cosine_similarity_mean"] > 0.999 and out["top1_agreement"] > 0.95


def test_auto_task_detection():
    ref = FnBackend(logits)
    assert accuracy.compare_models(ref, FnBackend(logits), feeds(3))["task"] == "classification"
    three_d = FnBackend(lambda x: np.zeros((1, 6, 8), dtype=np.float32) + x.sum())
    assert accuracy.compare_models(three_d, three_d, feeds(3))["task"] == "none"


def test_non_finite_candidate_outputs_are_counted_and_excluded():
    ref = FnBackend(logits)
    calls = {"n": 0}

    def sometimes_nan(x):
        calls["n"] += 1
        out = logits(x)
        if calls["n"] % 3 == 0:
            out = out * np.float32("nan")
        return out

    out = accuracy.compare_models(ref, FnBackend(sometimes_nan), feeds(9), task="classification")
    assert out["nonfinite_outputs"] == 3 and out["n_compared"] == 6
    assert any("NaN or infinity" in n for n in out["notes"])


def test_notes_are_honest_about_what_was_measured():
    out = accuracy.compare_models(FnBackend(logits), FnBackend(logits), feeds(3), input_source="synthetic")
    text = " ".join(out["notes"])
    assert "not ground-truth accuracy" in text and "synthetic" in text
    out = accuracy.compare_models(FnBackend(logits), FnBackend(logits), feeds(3), input_source="images")
    assert "synthetic" not in " ".join(out["notes"])


def test_bad_arguments():
    with pytest.raises(ValueError, match="task must be"):
        accuracy.compare_models(FnBackend(logits), FnBackend(logits), feeds(1), task="segmentation")
    with pytest.raises(ValueError, match="at least one"):
        accuracy.compare_models(FnBackend(logits), FnBackend(logits), [])


# ---------------------------------------------------------------- detection agreement

def yolo(boxes, n_classes=3, anchors=60):
    out = np.zeros((1, 4 + n_classes, anchors), dtype=np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(boxes):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + c, i] = s
    return out


SCENE = [(50, 50, 20, 20, 0, 0.9), (120, 80, 30, 30, 1, 0.8)]


def test_detection_agreement_identical_models():
    ref = FnBackend(lambda x: yolo(SCENE))
    out = accuracy.compare_models(ref, FnBackend(lambda x: yolo(SCENE)), feeds(5), task="detection")
    d = out["detection"]
    assert d["f1_iou50"] == 1.0 and d["precision"] == 1.0 and d["recall"] == 1.0
    assert d["true_positives"] == 10 and d["reference_detections"] == 10
    assert "not mAP" in d["note"]


def test_detection_agreement_partial_and_summed_over_images():
    ref = FnBackend(lambda x: yolo(SCENE))
    cand = FnBackend(lambda x: yolo(SCENE[:1] + [(200, 200, 20, 20, 2, 0.9)]))  # one right, one wrong
    out = accuracy.compare_models(ref, cand, feeds(4), task="detection")
    d = out["detection"]
    assert (d["true_positives"], d["false_positives"], d["false_negatives"]) == (4, 4, 4)
    assert d["f1_iou50"] == pytest.approx(0.5)


def test_detection_with_no_candidate_boxes_explains_the_usual_cause():
    ref = FnBackend(lambda x: yolo(SCENE))
    cand = FnBackend(lambda x: yolo([]))
    out = accuracy.compare_models(ref, cand, feeds(3), task="detection")
    assert out["detection"]["f1_iou50"] == 0.0 and out["detection"]["candidate_detections"] == 0
    assert any("no detections" in n and "--op-types Conv" in n for n in out["notes"])


def test_a_detector_without_a_task_gets_a_hint_instead_of_a_misleading_cosine():
    """Cosine similarity can stay near 1 while every detection is lost, so the missing task must be pointed out."""
    ref = FnBackend(lambda x: yolo(SCENE))
    cand = FnBackend(lambda x: yolo([]))
    out = accuracy.compare_models(ref, cand, feeds(4))  # task="auto"
    assert out["task"] == "none" and out["detection"] is None
    assert any("looks like a detector" in n and "box agreement" in n for n in out["notes"])
    # an explicit task needs no hint, and a classifier never gets one
    assert not any("looks like a detector" in n for n in accuracy.compare_models(ref, cand, feeds(4), task="detection")["notes"])
    assert not any("looks like a detector" in n for n in accuracy.compare_models(
        FnBackend(logits), FnBackend(logits), feeds(4))["notes"])


def test_detection_with_nothing_anywhere_has_no_f1():
    out = accuracy.compare_models(FnBackend(lambda x: yolo([])), FnBackend(lambda x: yolo([])), feeds(2), task="detection")
    assert out["detection"]["f1_iou50"] is None


# ------------------------------------------------------------------ with real models

def test_fp32_against_itself_through_onnx_runtime(models):
    a, b = OrtBackend(), OrtBackend()
    a.load(models["cls"])
    b.load(models["cls"])
    out = accuracy.compare_models(a, b, synthetic_feeds(a.inputs, 1, count=10, seed=3), task="classification")
    assert out["cosine_similarity_mean"] == pytest.approx(1.0) and out["top1_agreement"] == 1.0


def test_int8_classifier_stays_close_to_fp32(models, tmp_path):
    # 7-bit weights (reduce_range) on purpose. On x86 CPUs without VNNI, ONNX Runtime's U8S8 kernels add pairs
    # of uint8 x int8 products in 16 bits, which can saturate; with |weight| <= 64 a pair is at most
    # 2 * 255 * 64 = 32640 and cannot. The integer results, and so this test, are then the same on every CPU:
    # with full 8-bit weights the cosine was 0.998 on a VNNI laptop and 0.986 on some GitHub-hosted runners.
    int8 = tmp_path / "cls_int8_7bit.onnx"
    ref, cand = OrtBackend(), OrtBackend()
    ref.load(models["cls"])
    quantize_int8(models["cls"], int8, synthetic_feeds(ref.inputs, 1, count=32, seed=1), reduce_range=True)
    cand.load(int8)
    out = accuracy.compare_models(ref, cand, synthetic_feeds(ref.inputs, 1, count=60, seed=99), task="classification")
    assert out["cosine_similarity_mean"] > 0.99
    assert 0.8 <= out["top1_agreement"] <= 1.0
    assert out["ref_top1_in_cand_top5"] == 1.0


# ------------------------------------------------- warnings about meaningless comparisons

def notes_of(ref_fn, cand_fn=None, n=12, **kwargs):
    out = accuracy.compare_models(FnBackend(ref_fn), FnBackend(cand_fn or ref_fn), feeds(n), **kwargs)
    return " ".join(out["notes"])


def test_a_reference_that_outputs_almost_nothing_is_flagged():
    text = notes_of(lambda x: logits(x) * 1e-9, task="classification")
    assert "almost zero" in text and "--images" in text


def test_a_reference_that_ignores_its_input_is_flagged():
    constant = np.array([[0.1, 0.5, -0.3, 2.0, 0.0]], dtype=np.float32)
    text = notes_of(lambda x: constant + 0 * x.sum(), task="classification")
    assert "almost the same output" in text and "trivial" in text


def test_a_normal_reference_raises_no_signal_warnings():
    text = notes_of(logits, task="classification")
    assert "almost zero" not in text and "almost the same" not in text


def test_a_detection_reference_without_detections_is_flagged():
    text = notes_of(lambda x: yolo([]), task="detection")
    assert "no objects at all" in text


def test_the_reference_output_size_is_reported():
    out = accuracy.compare_models(FnBackend(lambda x: np.full((1, 5), 2.0, dtype=np.float32)),
                                  FnBackend(logits), feeds(3), task="classification")
    assert out["reference_mean_abs_output"] == pytest.approx(2.0)
