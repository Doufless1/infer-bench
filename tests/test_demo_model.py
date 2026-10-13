import numpy as np
import onnx
import pytest

from infer_bench import demo_model
from infer_bench.backends import OrtBackend
from infer_bench.inputs import synthetic_feeds
from infer_bench.modelinfo import describe_model, sha256_file


def test_demo_models_are_valid_onnx(models):
    for key in ("cls", "det"):
        onnx.checker.check_model(onnx.load(str(models[key])))


def test_demo_models_are_reproducible_for_a_seed(tmp_path):
    a = demo_model.build_classifier(tmp_path / "a.onnx", seed=1)
    b = demo_model.build_classifier(tmp_path / "b.onnx", seed=1)
    c = demo_model.build_classifier(tmp_path / "c.onnx", seed=2)
    assert sha256_file(a) == sha256_file(b)
    assert sha256_file(a) != sha256_file(c)


def test_classifier_shapes_and_precision(models):
    info = describe_model(models["cls"])
    assert info["precision"] == "fp32" and info["opset"] == 13
    backend = OrtBackend()
    backend.load(models["cls"])
    assert backend.outputs[0].shape == ("batch", 10)


def test_classifier_predicts_many_different_classes(models):
    """Without the bias balancing, one class wins for nearly every input and 'top-1 agreement' is meaningless."""
    backend = OrtBackend()
    backend.load(models["cls"])
    feeds = synthetic_feeds(backend.inputs, 1, count=120, seed=77)
    winners = [int(backend.run(f)[0].argmax()) for f in feeds]
    counts = np.bincount(winners, minlength=10)
    assert (counts > 0).sum() >= 5
    assert counts.max() < 0.5 * len(winners)


def test_detector_has_a_yolo_style_output(models):
    backend = OrtBackend()
    backend.load(models["det"])
    out = backend.run(synthetic_feeds(backend.inputs, 1, count=1)[0])[0]
    assert out.shape == (1, 8, 400)  # batch, 4 box values + 4 classes, 20 x 20 anchors
    assert out[0, :4].min() >= 0 and out[0, :4].max() <= 320  # boxes in input pixels
    assert out[0, 4:].min() >= 0 and out[0, 4:].max() <= 1  # class scores in 0..1


def test_detector_produces_some_but_not_absurdly_many_detections(models):
    from infer_bench.postprocess import decode_yolo

    backend = OrtBackend()
    backend.load(models["det"])
    counts = [len(decode_yolo(backend.run(f)[0])) for f in synthetic_feeds(backend.inputs, 1, count=20, seed=5)]
    assert 0 < np.mean(counts) < 100


def test_build_demo_model_dispatch(tmp_path):
    assert demo_model.build_demo_model("classifier", tmp_path / "c.onnx").is_file()
    with pytest.raises(ValueError, match="unknown demo model"):
        demo_model.build_demo_model("segmenter", tmp_path / "x.onnx")


def test_detector_needs_four_stages(tmp_path):
    with pytest.raises(ValueError, match="four"):
        demo_model.build_detector(tmp_path / "d.onnx", widths=(8, 8))


def test_model_info_for_a_real_file(models):
    info = describe_model(models["cls"])
    assert info["file"] == "cls.onnx" and info["n_nodes"] == 13 and info["n_parameters"] > 90_000
    assert len(info["sha256"]) == 64 and info["size_bytes"] > 0
    assert info["top_operators"]["Conv"] == 4
