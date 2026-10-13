"""Compare two benchmark results.

The comparison does not just divide two numbers. It also checks whether the
two runs were set up comparably (same batch size, mode, threads, CPU, ONNX
Runtime version...) and whether the difference is bigger than the measurement
noise, using the confidence intervals stored in each result.
"""

from __future__ import annotations

from typing import Any

from .env import comparability_warnings


def _get(result: dict[str, Any], *path: str) -> Any:
    node: Any = result
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _ratio(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return new / old


def compare_results(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """Compare ``other`` against ``base``. Ratios are other / base."""
    warnings: list[str] = []

    # --- was the setup comparable?
    for key, label in (
        ("mode", "mode"),
        ("batch_size", "batch size"),
        ("streams", "number of streams"),
        ("task", "post-processing task"),
        ("input_source", "input source"),
    ):
        a, b = _get(base, "config", key), _get(other, "config", key)
        if a != b:
            warnings.append(f"Different {label}: {a} vs {b}. The runs are not directly comparable.")
    for key in ("intra_op_threads", "inter_op_threads", "graph_optimization", "active_providers"):
        a, b = _get(base, "backend", key), _get(other, "backend", key)
        if a != b:
            warnings.append(f"Backend setting '{key}' differs: {a} vs {b}.")
    warnings.extend(
        comparability_warnings(base.get("environment", {}) or {}, other.get("environment", {}) or {})
    )

    # --- the numbers
    fps_a, fps_b = _get(base, "throughput", "items_per_second"), _get(other, "throughput", "items_per_second")
    lat_ratio = {q: _ratio(_get(other, "latency", f"{q}_ms"), _get(base, "latency", f"{q}_ms")) for q in ("p50", "p95", "p99")}
    ci_a, ci_b = _get(base, "latency", "median_ci95_ms"), _get(other, "latency", "median_ci95_ms")
    overlap: bool | None = None
    if ci_a and ci_b:
        overlap = not (ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0])

    speedup = _ratio(fps_b, fps_a)
    p50_ratio = lat_ratio["p50"]
    if p50_ratio is None:
        verdict = "Not enough data to compare."
    elif overlap:
        verdict = (
            "The median-latency confidence intervals overlap: the difference is within measurement "
            "noise. Do not claim a speed-up from these two runs."
        )
    elif p50_ratio < 1:
        verdict = f"Median latency is {1 / p50_ratio:.2f}x lower (intervals do not overlap)."
    else:
        verdict = f"Median latency is {p50_ratio:.2f}x higher (intervals do not overlap)."

    for result, name in ((base, "base"), (other, "other")):
        n = _get(result, "latency", "n")
        if isinstance(n, int) and n < 100:
            warnings.append(f"The {name} run has only {n} samples.")

    out: dict[str, Any] = {
        "base": base.get("label"),
        "other": other.get("label"),
        "throughput_ratio": speedup,
        "latency_ratio": lat_ratio,
        "median_ci_overlap": overlap,
        "verdict": verdict,
        "power": None,
        "accuracy": None,
        "warnings": warnings,
    }

    eff_a, eff_b = _get(base, "power", "items_per_s_per_w"), _get(other, "power", "items_per_s_per_w")
    if eff_a and eff_b:
        out["power"] = {
            "items_per_s_per_w_ratio": eff_b / eff_a,
            "avg_power_w": [_get(base, "power", "avg_power_w"), _get(other, "power", "avg_power_w")],
            "sources": [_get(base, "power", "source"), _get(other, "power", "source")],
        }
    acc = other.get("accuracy")
    if acc:
        out["accuracy"] = {
            "cosine_similarity_mean": acc.get("cosine_similarity_mean"),
            "top1_agreement": acc.get("top1_agreement"),
            "detection_f1": (acc.get("detection") or {}).get("f1_iou50"),
        }
    return out
