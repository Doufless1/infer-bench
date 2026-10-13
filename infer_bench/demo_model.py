"""Small demo ONNX models, built from scratch with ``onnx.helper``.

They exist so that the whole tool (benchmark, INT8 quantisation, accuracy
comparison, dashboard) can be tried in seconds without downloading anything
and without PyTorch. The weights are random (but seeded, so the files are
reproducible): these are **not** trained networks, and numbers measured on
them say nothing about real models. Use your own model for real work.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 13  # first opset with per-channel QuantizeLinear/DequantizeLinear
IR_VERSION = 8  # understood by every ONNX Runtime from 1.10 on


def _array(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> np.ndarray:
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _conv_relu(nodes, inits, rng, name: str, x: str, cin: int, cout: int, k: int, stride: int) -> str:
    inits.append(numpy_helper.from_array(_array(rng, (cout, cin, k, k), np.sqrt(2.0 / (cin * k * k))), f"{name}_w"))
    inits.append(numpy_helper.from_array(_array(rng, (cout,), 0.05), f"{name}_b"))
    nodes.append(
        helper.make_node(
            "Conv", [x, f"{name}_w", f"{name}_b"], [f"{name}_conv"], name=f"{name}_conv",
            kernel_shape=[k, k], strides=[stride, stride], pads=[k // 2] * 4,
        )
    )
    nodes.append(helper.make_node("Relu", [f"{name}_conv"], [f"{name}_out"], name=f"{name}_relu"))
    return f"{name}_out"


def _finish(graph: onnx.GraphProto, path: Path) -> Path:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def _classifier_graph(
    num_classes: int, input_size: int, widths: tuple[int, ...], seed: int, fc_bias: np.ndarray | None
) -> onnx.GraphProto:
    rng = np.random.default_rng(seed)
    nodes: list = []
    inits: list = [
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "norm_mean"),
        numpy_helper.from_array(np.array(4.0, dtype=np.float32), "norm_scale"),
    ]
    # Centre and stretch the input inside the graph, like a baked-in normalisation.
    nodes.append(helper.make_node("Sub", ["images", "norm_mean"], ["x_centered"], name="norm_sub"))
    nodes.append(helper.make_node("Mul", ["x_centered", "norm_scale"], ["x_norm"], name="norm_mul"))

    x, channels = "x_norm", 3
    for i, width in enumerate(widths):
        x = _conv_relu(nodes, inits, rng, f"conv{i + 1}", x, channels, width, k=3, stride=2)
        channels = width
    nodes.append(helper.make_node("GlobalAveragePool", [x], ["pooled"], name="global_pool"))
    nodes.append(helper.make_node("Flatten", ["pooled"], ["flat"], name="flatten", axis=1))
    fc_w = _array(rng, (num_classes, channels), 1.5 / np.sqrt(channels))
    drawn_bias = _array(rng, (num_classes,), 0.1)
    inits.append(numpy_helper.from_array(fc_w, "fc_w"))
    inits.append(numpy_helper.from_array(drawn_bias if fc_bias is None else fc_bias.astype(np.float32), "fc_b"))
    nodes.append(helper.make_node("Gemm", ["flat", "fc_w", "fc_b"], ["logits"], name="fc", transB=1))
    return helper.make_graph(
        nodes,
        "tinynet_classifier",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, input_size, input_size])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", num_classes])],
        initializer=inits,
    )


def build_classifier(
    path: str | Path,
    *,
    num_classes: int = 10,
    input_size: int = 224,
    widths: tuple[int, ...] = (16, 32, 64, 128),
    seed: int = 0,
) -> Path:
    """A 4-layer CNN classifier: input ``images`` (batch,3,S,S) -> ``logits`` (batch,classes).

    With random weights one class would win for almost every input, which makes "top-1
    agreement" meaningless. So the output bias is set so that every class has zero average
    logit over a fixed set of synthetic textures; the winning class then varies between inputs.
    """
    from .backends.ort import OrtBackend
    from .inputs import synthetic_feeds

    path = Path(path)
    first = _finish(_classifier_graph(num_classes, input_size, widths, seed, np.zeros(num_classes)), path)
    backend = OrtBackend()
    backend.load(first)
    feeds = synthetic_feeds(backend.inputs, 1, count=128, seed=12345)
    logits = np.concatenate([backend.run(f)[0] for f in feeds], axis=0)
    backend.close()
    return _finish(_classifier_graph(num_classes, input_size, widths, seed, -logits.mean(axis=0)), path)


def build_detector(
    path: str | Path,
    *,
    num_classes: int = 4,
    input_size: int = 320,
    widths: tuple[int, ...] = (16, 32, 64, 96),
    seed: int = 0,
) -> Path:
    """A detector-shaped network with a YOLOv8-style output.

    Output ``output0`` has shape (batch, 4 + num_classes, anchors): box centre x/y and
    width/height in input pixels, then per-class scores in 0..1. The values come from random
    weights, so the "detections" are meaningless - the point is the *shape*, which exercises
    the decoding, NMS and agreement code exactly like a real YOLOv8 export would.
    """
    if len(widths) != 4:
        raise ValueError("the demo detector uses exactly four stride-2 stages (stride 16)")
    rng = np.random.default_rng(seed)
    nodes: list = []
    inits: list = []
    x, channels = "images", 3
    for i, width in enumerate(widths):
        x = _conv_relu(nodes, inits, rng, f"conv{i + 1}", x, channels, width, k=3, stride=2)
        channels = width

    n_out = 4 + num_classes
    head_bias = _array(rng, (n_out,), 0.05)
    head_bias[4:] -= 1.5  # make most class scores fall below the usual 0.25 threshold
    inits.append(numpy_helper.from_array(_array(rng, (n_out, channels, 1, 1), 1.0 / np.sqrt(channels)), "head_w"))
    inits.append(numpy_helper.from_array(head_bias, "head_b"))
    nodes.append(
        helper.make_node("Conv", [x, "head_w", "head_b"], ["head"], name="head", kernel_shape=[1, 1])
    )
    inits.append(numpy_helper.from_array(np.array([4, num_classes], dtype=np.int64), "split_sizes"))
    nodes.append(helper.make_node("Split", ["head", "split_sizes"], ["box_raw", "cls_raw"], name="split", axis=1))
    box_scale = np.array([input_size, input_size, input_size / 2, input_size / 2], dtype=np.float32).reshape(1, 4, 1, 1)
    inits.append(numpy_helper.from_array(box_scale, "box_scale"))
    nodes.append(helper.make_node("Sigmoid", ["box_raw"], ["box_unit"], name="box_sigmoid"))
    nodes.append(helper.make_node("Mul", ["box_unit", "box_scale"], ["box_px"], name="box_scale_mul"))
    nodes.append(helper.make_node("Sigmoid", ["cls_raw"], ["cls_prob"], name="cls_sigmoid"))
    nodes.append(helper.make_node("Concat", ["box_px", "cls_prob"], ["merged"], name="merge", axis=1))
    inits.append(numpy_helper.from_array(np.array([0, n_out, -1], dtype=np.int64), "flat_shape"))
    nodes.append(helper.make_node("Reshape", ["merged", "flat_shape"], ["output0"], name="flatten_grid"))

    anchors = (input_size // 16) ** 2
    graph = helper.make_graph(
        nodes,
        "tinynet_detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, input_size, input_size])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, ["batch", n_out, anchors])],
        initializer=inits,
    )
    return _finish(graph, Path(path))


KINDS = {"classifier": build_classifier, "detector": build_detector}


def build_demo_model(kind: str, path: str | Path, **kwargs) -> Path:
    if kind not in KINDS:
        raise ValueError(f"unknown demo model '{kind}'. Choose from: {', '.join(KINDS)}")
    return KINDS[kind](path, **kwargs)
