"""Describe an ONNX model file: identity (hash), size, precision, operators."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

# Operators that only appear in quantised (integer) graphs.
_QUANT_OPS = {
    "QuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeLinear",
    "ConvInteger",
    "MatMulInteger",
}


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def describe_model(path: str | Path) -> dict[str, Any]:
    """Facts about a model file, recorded in every results file.

    ``precision`` is a heuristic read from the graph: "int8" if it contains
    quantise/dequantise or integer operators, "fp16" / "bf16" if the weights
    are half precision, otherwise "fp32". It is a label for humans, not a
    guarantee.
    """
    import onnx
    from onnx import TensorProto

    path = Path(path)
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    op_counts = Counter(node.op_type for node in graph.node)
    is_quant = any(op in _QUANT_OPS or op.startswith("QLinear") for op in op_counts)

    # Weights normally live in graph initializers, but some exporters (for example from
    # PaddlePaddle) store them as Constant nodes instead, so count both.
    tensors = list(graph.initializer)
    for node in graph.node:
        if node.op_type == "Constant":
            tensors.extend(attr.t for attr in node.attribute if attr.name == "value" and attr.HasField("t"))
    params = 0
    dtypes: Counter[int] = Counter()
    for tensor in tensors:
        size = 1
        for d in tensor.dims:
            size *= int(d)
        params += size
        dtypes[tensor.data_type] += size

    if is_quant:
        precision = "int8"
    elif dtypes.get(TensorProto.FLOAT16, 0) > dtypes.get(TensorProto.FLOAT, 0):
        precision = "fp16"
    elif dtypes.get(TensorProto.BFLOAT16, 0) > dtypes.get(TensorProto.FLOAT, 0):
        precision = "bf16"
    elif dtypes.get(TensorProto.FLOAT, 0) > 0:
        precision = "fp32"
    else:
        precision = "unknown"

    opsets = {imp.domain or "ai.onnx": imp.version for imp in model.opset_import}
    # Only the file name is recorded, never the folder: results files get shared, and a folder
    # name can contain a user name. The SHA-256 is what identifies the model.
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "precision": precision,
        "ir_version": int(model.ir_version),
        "opset": opsets.get("ai.onnx", None),
        "n_nodes": len(graph.node),
        "n_parameters": params,  # elements in weights and other constant tensors
        "top_operators": dict(op_counts.most_common(8)),
    }
