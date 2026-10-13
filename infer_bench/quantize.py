"""INT8 post-training static quantisation with ONNX Runtime (QDQ format).

"Static" means the activation ranges are measured once, on a calibration set,
and baked into the model. The calibration set should look like the data the
model will see: use real images (``--images``) for a real model. Calibrating
on random noise only makes sense for the built-in demo models.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_QUANT_TYPES = {"int8": "QInt8", "uint8": "QUInt8"}
_METHODS = {"minmax": "MinMax", "entropy": "Entropy", "percentile": "Percentile"}

# Per-channel quantised weights need a DequantizeLinear with an ``axis`` attribute, which exists
# from operator-set 13. With ONNX Runtime 1.31 a model with an older opset quantised per channel is
# written without complaint and then cannot be loaded (found while testing a real OCR model).
MIN_OPSET_PER_CHANNEL = 13


class FeedReader:
    """Feeds calibration inputs to ONNX Runtime, one dict per call."""

    def __init__(self, feeds: Iterable[dict[str, np.ndarray]]):
        self._feeds = list(feeds)
        self._i = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._i >= len(self._feeds):
            return None
        feed = self._feeds[self._i]
        self._i += 1
        return feed

    def rewind(self) -> None:
        self._i = 0

    def __len__(self) -> int:
        return len(self._feeds)


def _make_reader(feeds: list[dict[str, np.ndarray]]):
    """Wrap FeedReader so it is a real ``CalibrationDataReader`` subclass."""
    from onnxruntime.quantization import CalibrationDataReader

    class _Reader(FeedReader, CalibrationDataReader):  # type: ignore[misc]
        def __init__(self, items: list[dict[str, np.ndarray]]):
            FeedReader.__init__(self, items)

    return _Reader(feeds)


def _pre_process(model_in: Path, workdir: Path) -> tuple[Path, str]:
    """ONNX Runtime's recommended clean-up before quantisation (shape inference, fusions).

    Returns the path to use and a note on what happened. Failure is not fatal:
    quantisation also works on the original model.
    """
    from onnxruntime.quantization.shape_inference import quant_pre_process

    out = workdir / "preprocessed.onnx"
    try:
        quant_pre_process(str(model_in), str(out))
        return out, "pre-processed with onnxruntime quant_pre_process"
    except Exception as first_error:  # noqa: BLE001 - e.g. sympy missing for symbolic shape inference
        try:
            quant_pre_process(str(model_in), str(out), skip_symbolic_shape=True)
            return out, f"pre-processed without symbolic shape inference ({type(first_error).__name__})"
        except Exception as second_error:  # noqa: BLE001
            return model_in, f"pre-processing skipped ({type(second_error).__name__}: {second_error})"


def default_opset(model_path: str | Path) -> int | None:
    """Version of the default ONNX operator set (domain "" / "ai.onnx") that a model uses."""
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            return int(imp.version)
    return None


def _per_channel_hint(opset: int | None) -> str:
    return (
        f"Per-channel quantisation needs operator set {MIN_OPSET_PER_CHANNEL} or higher and this model uses "
        f"{opset}: run again with --no-per-channel, or re-export the model with a newer opset."
    )


def _check_loadable(path: Path, *, per_channel: bool, opset: int | None) -> None:
    """Fail loudly if the model that was just written cannot be loaded.

    A quantiser can write a file and still produce an invalid graph. Finding that out here, with a
    hint, is much better than a cryptic error from the next command.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3  # errors only
    try:
        ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001 - whatever the reason, the output is unusable
        if per_channel and opset is not None and opset < MIN_OPSET_PER_CHANNEL:
            hint = _per_channel_hint(opset)
        else:
            hint = "Try --no-preprocess, --op-types Conv (quantise only the convolutions) or --exclude-nodes."
        raise RuntimeError(
            f"the quantised model was written to {path} (kept for inspection) but ONNX Runtime cannot load it: "
            f"{exc} -- HINT: {hint}"
        ) from exc


def quantization_warnings(model_path: str | Path) -> list[str]:
    """Properties of a model that make ONNX Runtime's INT8 quantisation work badly.

    Found while testing on a real third-party model (a PaddlePaddle export): the weights were
    stored as Constant nodes, ONNX Runtime skipped them, and the "INT8" model came out bigger,
    slower and wrong. The tool cannot fix every exporter, but it can say what is going on.
    """
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    large_constants = 0
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.HasField("t"):
                size = 1
                for d in attr.t.dims:
                    size *= int(d)
                large_constants += size >= 256
    notes: list[str] = []
    if large_constants:
        notes.append(
            f"{large_constants} large tensors are stored as Constant nodes instead of initializers (typical of some "
            "exporters, for example PaddlePaddle). ONNX Runtime cannot quantise those weights, so the INT8 model may "
            "be bigger, slower and much less accurate. Convert them to initializers first, for example with onnx-simplifier."
        )
    batch_norms = sum(1 for n in model.graph.node if n.op_type == "BatchNormalization")
    if batch_norms:
        notes.append(
            f"{batch_norms} BatchNormalization nodes are still in the graph. They are normally folded into the "
            "convolutions when a model is exported in inference mode; unfolded, they hurt INT8 quality and speed."
        )
    return notes


def quantize_int8(
    model_in: str | Path,
    model_out: str | Path,
    feeds: list[dict[str, np.ndarray]],
    *,
    per_channel: bool | None = None,
    activation_type: str = "uint8",
    weight_type: str = "int8",
    method: str = "minmax",
    reduce_range: bool = False,
    pre_process: bool = True,
    op_types: list[str] | None = None,
    nodes_to_exclude: list[str] | None = None,
) -> dict[str, Any]:
    """Quantise ``model_in`` to INT8 (QDQ) and write ``model_out``. Returns a summary.

    ``per_channel=None`` (the default) means "per channel when the model's opset allows it, otherwise
    per tensor". The written model is loaded once with ONNX Runtime to prove it is usable.
    """
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    if activation_type not in _QUANT_TYPES or weight_type not in _QUANT_TYPES:
        raise ValueError(f"activation/weight type must be one of {sorted(_QUANT_TYPES)}")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {sorted(_METHODS)}")
    if not feeds:
        raise ValueError("need at least one calibration input")

    model_in, model_out = Path(model_in), Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        source, pre_note = (_pre_process(model_in, Path(tmp)) if pre_process else (model_in, "pre-processing disabled"))
        warnings = quantization_warnings(source)
        notes: list[str] = []
        opset = default_opset(source)
        if per_channel is None:
            per_channel = opset is None or opset >= MIN_OPSET_PER_CHANNEL
            if not per_channel:
                notes.append(
                    f"The model uses operator set {opset}; per-channel INT8 needs {MIN_OPSET_PER_CHANNEL} or higher, so "
                    "the weights were quantised per tensor instead (usually a little less accurate). "
                    "Re-export the model with a newer opset to use per-channel."
                )
        elif per_channel and opset is not None and opset < MIN_OPSET_PER_CHANNEL:
            warnings.append(
                f"Per-channel quantisation was asked for, but the model uses operator set {opset} (below "
                f"{MIN_OPSET_PER_CHANNEL}): ONNX Runtime may write a model it cannot load."
            )
        try:
            quantize_static(
                model_input=str(source),
                model_output=str(model_out),
                calibration_data_reader=_make_reader(feeds),
                quant_format=QuantFormat.QDQ,
                per_channel=per_channel,
                reduce_range=reduce_range,
                activation_type=getattr(QuantType, _QUANT_TYPES[activation_type]),
                weight_type=getattr(QuantType, _QUANT_TYPES[weight_type]),
                calibrate_method=getattr(CalibrationMethod, _METHODS[method]),
                op_types_to_quantize=op_types or None,
                nodes_to_exclude=nodes_to_exclude or [],
            )
        except ValueError as exc:
            # Older ONNX Runtime versions refuse per-channel quantisation of an old opset themselves.
            if per_channel and opset is not None and opset < MIN_OPSET_PER_CHANNEL:
                raise ValueError(f"{exc} -- HINT: {_per_channel_hint(opset)}") from exc
            raise
    _check_loadable(model_out, per_channel=per_channel, opset=opset)
    in_size, out_size = model_in.stat().st_size, model_out.stat().st_size
    if out_size > 0.85 * in_size:
        warnings.append(
            f"The INT8 model is hardly smaller than the original ({in_size / 1e6:.2f} MB -> {out_size / 1e6:.2f} MB): "
            "most weights were probably not quantised."
        )
    return {
        "input": str(model_in),
        "output": str(model_out),
        "input_size_bytes": in_size,
        "output_size_bytes": out_size,
        "opset": opset,
        "notes": notes,
        "warnings": warnings,
        "calibration_samples": len(feeds),
        "format": "QDQ",
        "activation_type": activation_type,
        "weight_type": weight_type,
        "per_channel": per_channel,
        "calibration_method": method,
        "reduce_range": reduce_range,
        "op_types": op_types or "all supported",
        "nodes_to_exclude": nodes_to_exclude or [],
        "pre_processing": pre_note,
    }
