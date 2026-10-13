from pathlib import Path

import numpy as np
import pytest

from infer_bench import quantize
from infer_bench.accuracy import compare_models
from infer_bench.backends import OrtBackend
from infer_bench.inputs import synthetic_feeds
from infer_bench.modelinfo import describe_model

from conftest import with_opset


def test_feed_reader_hands_out_feeds_then_none_and_can_rewind():
    reader = quantize.FeedReader([{"a": 1}, {"a": 2}])
    assert len(reader) == 2
    assert [reader.get_next(), reader.get_next(), reader.get_next()] == [{"a": 1}, {"a": 2}, None]
    reader.rewind()
    assert reader.get_next() == {"a": 1}


def test_quantised_classifier_is_int8_smaller_and_still_runs(models):
    fp32, int8 = describe_model(models["cls"]), describe_model(models["cls_int8"])
    assert fp32["precision"] == "fp32" and int8["precision"] == "int8"
    assert int8["size_bytes"] < fp32["size_bytes"] / 2
    assert "QuantizeLinear" in int8["top_operators"] or "DequantizeLinear" in int8["top_operators"]
    backend = OrtBackend()
    backend.load(models["cls_int8"])
    out = backend.run({"images": np.zeros((2, 3, 224, 224), dtype=np.float32)})
    assert out[0].shape == (2, 10) and np.all(np.isfinite(out[0]))


def test_quantize_writes_a_summary_and_creates_folders(models, tmp_path):
    backend = OrtBackend()
    backend.load(models["cls"])
    feeds = synthetic_feeds(backend.inputs, 1, count=8, seed=2)
    summary = quantize.quantize_int8(models["cls"], tmp_path / "nested" / "out.onnx", feeds, per_channel=False, method="percentile")
    assert (tmp_path / "nested" / "out.onnx").is_file()
    assert summary["calibration_samples"] == 8 and summary["per_channel"] is False
    assert summary["calibration_method"] == "percentile" and summary["format"] == "QDQ"
    assert summary["output_size_bytes"] < summary["input_size_bytes"]


@pytest.mark.parametrize(
    "kwargs, message",
    [({"activation_type": "int4"}, "activation/weight type"), ({"method": "magic"}, "method must be")],
)
def test_quantize_rejects_bad_options(models, tmp_path, kwargs, message):
    backend = OrtBackend()
    backend.load(models["cls"])
    feeds = synthetic_feeds(backend.inputs, 1, count=2)
    with pytest.raises(ValueError, match=message):
        quantize.quantize_int8(models["cls"], tmp_path / "x.onnx", feeds, **kwargs)


def test_entropy_calibration_is_diagnosed_as_identical_to_minmax(models, tmp_path):
    # Pins ONNX Runtime's behaviour: if this starts failing because the two files differ, a new ONNX Runtime
    # version runs a real entropy search and the warning (ENTROPY_SAME_AS_MINMAX) must go.
    backend = OrtBackend()
    backend.load(models["cls"])
    feeds = synthetic_feeds(backend.inputs, 1, count=8, seed=3)
    minmax = quantize.quantize_int8(models["cls"], tmp_path / "minmax.onnx", feeds, method="minmax")
    entropy = quantize.quantize_int8(models["cls"], tmp_path / "entropy.onnx", feeds, method="entropy")
    assert (tmp_path / "minmax.onnx").read_bytes() == (tmp_path / "entropy.onnx").read_bytes()
    assert quantize.ENTROPY_SAME_AS_MINMAX in entropy["warnings"]
    assert quantize.ENTROPY_SAME_AS_MINMAX not in minmax["warnings"]


def test_quantize_needs_calibration_data(models, tmp_path):
    with pytest.raises(ValueError, match="at least one calibration"):
        quantize.quantize_int8(models["cls"], tmp_path / "x.onnx", [])


def test_quantising_only_convolutions_keeps_the_detector_working(models):
    """The pitfall the accuracy tooling exists to catch.

    Quantising the whole detector puts pixel-scale box values and 0..1 class scores in one
    quantised tensor. Quantising only the convolutions keeps the output decoding in float.
    (How badly the naive version fails depends on the ONNX Runtime version, so only the
    guarantee that matters is asserted: the Conv-only model must be good, and not worse.)
    """
    ref = OrtBackend()
    ref.load(models["det"])
    feeds = synthetic_feeds(ref.inputs, 1, count=40, seed=99)
    scores = {}
    for name in ("det_int8", "det_int8_conv"):
        cand = OrtBackend()
        cand.load(models[name])
        out = compare_models(ref, cand, feeds, task="detection")
        scores[name] = out["detection"]["f1_iou50"] or 0.0
    assert scores["det_int8_conv"] > 0.8
    assert scores["det_int8_conv"] >= scores["det_int8"]


def test_nodes_to_exclude_keeps_those_nodes_in_float(models, tmp_path):
    backend = OrtBackend()
    backend.load(models["cls"])
    feeds = synthetic_feeds(backend.inputs, 1, count=8, seed=2)
    out = tmp_path / "excluded.onnx"
    quantize.quantize_int8(models["cls"], out, feeds, pre_process=False, nodes_to_exclude=["fc"])
    ops_all = describe_model(models["cls_int8"])["top_operators"]
    ops_excl = describe_model(out)["top_operators"]
    assert "Gemm" in ops_excl  # the excluded layer stays a float Gemm
    assert ops_excl.get("QuantizeLinear", 0) <= ops_all.get("QuantizeLinear", 0)


# ------------------------------------------------------------ warnings about hard-to-quantise models

def _weights_as_constant_nodes(path_in, path_out):
    """Rewrite a model so its weights are Constant nodes instead of initializers (as some exporters do)."""
    import onnx
    from onnx import helper

    model = onnx.load(str(path_in))
    constants = [
        helper.make_node("Constant", [], [init.name], name=f"const_{init.name}", value=init)
        for init in model.graph.initializer
    ]
    del model.graph.initializer[:]
    nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(constants + nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(path_out))


def test_weights_stored_as_constant_nodes_are_diagnosed(models, tmp_path):
    odd = tmp_path / "const.onnx"
    _weights_as_constant_nodes(models["cls"], odd)
    notes = quantize.quantization_warnings(odd)
    assert len(notes) == 1 and "Constant nodes" in notes[0] and "onnx-simplifier" in notes[0]
    assert quantize.quantization_warnings(models["cls"]) == []  # the normal layout raises nothing

    backend = OrtBackend()
    backend.load(odd)
    feeds = synthetic_feeds(backend.inputs, 1, count=4)

    # Without ONNX Runtime's pre-processing the Constant nodes reach the quantiser. It cannot quantise
    # those weights, so the "INT8" model does not shrink, and the tool has to say both things.
    raw = quantize.quantize_int8(odd, tmp_path / "raw.onnx", feeds, pre_process=False)
    assert any("Constant nodes" in w for w in raw["warnings"])
    assert any("hardly smaller" in w for w in raw["warnings"])

    # With pre-processing, some ONNX Runtime versions (1.31) turn those constants into initializers and the
    # quantisation then works; others (1.29) leave them alone. The tool must never be silent about a bad result.
    pre = quantize.quantize_int8(odd, tmp_path / "pre.onnx", feeds)
    shrank = pre["output_size_bytes"] < 0.5 * pre["input_size_bytes"]
    assert shrank or any("Constant nodes" in w for w in pre["warnings"])

    ok = quantize.quantize_int8(models["cls"], tmp_path / "ok.onnx", feeds)
    assert not any("hardly smaller" in w for w in ok["warnings"])


# --------------------------------------------------------------- operator set / loadable output

def test_models_with_an_old_opset_are_quantised_per_tensor_and_still_load(models, tmp_path):
    old = with_opset(models["cls"], tmp_path / "old.onnx", 12)
    assert quantize.default_opset(old) == 12 and quantize.default_opset(models["cls"]) == 13
    backend = OrtBackend()
    backend.load(old)
    feeds = synthetic_feeds(backend.inputs, 1, count=8)

    summary = quantize.quantize_int8(old, tmp_path / "old_int8.onnx", feeds)
    assert summary["per_channel"] is False and summary["opset"] == 12
    assert len(summary["notes"]) == 1 and "operator set 12" in summary["notes"][0]
    OrtBackend().load(tmp_path / "old_int8.onnx")  # the proof that matters: ONNX Runtime accepts the file

    modern = quantize.quantize_int8(models["cls"], tmp_path / "new_int8.onnx", feeds)
    assert modern["per_channel"] is True and modern["notes"] == []


def test_asking_for_per_channel_on_an_old_opset_is_never_silent(models, tmp_path):
    old = with_opset(models["cls"], tmp_path / "old.onnx", 12)
    backend = OrtBackend()
    backend.load(old)
    feeds = synthetic_feeds(backend.inputs, 1, count=8)
    try:
        summary = quantize.quantize_int8(old, tmp_path / "x.onnx", feeds, per_channel=True)
    except (RuntimeError, ValueError) as exc:
        # ONNX Runtime 1.18 refuses by itself (ValueError); 1.31 writes a model it cannot load and the tool
        # catches that (RuntimeError). Either way the message must say what to do.
        assert "--no-per-channel" in str(exc) and "operator set 13" in str(exc)
    else:  # a version in between copes, but the request is still flagged
        assert any("Per-channel quantisation was asked for" in w for w in summary["warnings"])


@pytest.mark.parametrize("opset, hint", [(12, "--no-per-channel"), (13, "--op-types Conv")])
def test_an_output_that_cannot_be_loaded_is_reported_with_a_hint(models, tmp_path, monkeypatch, opset, hint):
    source = with_opset(models["cls"], tmp_path / "src.onnx", opset)

    def broken_quantiser(*, model_output, **_ignored):
        Path(model_output).write_bytes(b"this is not an onnx model")

    monkeypatch.setattr("onnxruntime.quantization.quantize_static", broken_quantiser)
    feeds = [{"images": np.zeros((1, 3, 224, 224), dtype=np.float32)}]
    with pytest.raises(RuntimeError, match="cannot load it") as info:
        quantize.quantize_int8(source, tmp_path / "bad.onnx", feeds, per_channel=True)
    assert hint in str(info.value) and "kept for inspection" in str(info.value)
    assert (tmp_path / "bad.onnx").is_file()


def test_unfolded_batch_norm_is_diagnosed(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    ones, zeros = np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32)
    inits = [numpy_helper.from_array(a, n) for a, n in ((ones, "g"), (zeros, "b"), (zeros, "m"), (ones, "v"))]
    node = helper.make_node("BatchNormalization", ["x", "g", "b", "m", "v"], ["y"])
    graph = helper.make_graph([node], "bn", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 2, 2])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 2, 2])], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "bn.onnx"
    onnx.save(model, str(path))
    notes = quantize.quantization_warnings(path)
    assert len(notes) == 1 and "BatchNormalization" in notes[0]
