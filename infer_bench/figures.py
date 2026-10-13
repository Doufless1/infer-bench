"""Plotly figures for the dashboard.

Every function takes plain dictionaries (results or flattened rows) and
returns a ``plotly.graph_objects.Figure``, so the charts can be tested, and
reused in a notebook, without starting a web server.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go

PALETTE = ("#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#4d7c0f")
TEMPLATE = "plotly_white"
MAX_POINTS = 2000  # samples drawn per run in line charts


def color_for(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def _style(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        template=TEMPLATE,
        title={"text": title, "font": {"size": 15}, "x": 0.01, "xanchor": "left", "y": 0.97, "yanchor": "top"},
        height=height,
        margin={"l": 60, "r": 20, "t": 85, "b": 55},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
        font={"size": 12},
    )
    return fig


def empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
        font={"size": 13, "color": "#6b7280"}, align="center",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, title)


def throughput_figure(rows: Sequence[dict[str, Any]]) -> go.Figure:
    rows = [r for r in rows if r.get("fps") is not None]
    if not rows:
        return empty_figure("Throughput", "No runs selected.")
    fig = go.Figure(
        go.Bar(
            x=[r["label"] for r in rows],
            y=[r["fps"] for r in rows],
            marker_color=[color_for(i) for i in range(len(rows))],
            customdata=[[r.get("p50_ms"), r.get("p95_ms"), r.get("batch"), r.get("streams")] for r in rows],
            hovertemplate=(
                "<b>%{x}</b><br>%{y:.1f} items/s<br>p50 %{customdata[0]:.3f} ms, p95 %{customdata[1]:.3f} ms"
                "<br>batch %{customdata[2]}, streams %{customdata[3]}<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(title="items / second (higher is better)", rangemode="tozero")
    return _style(fig, "Throughput")


def latency_percentiles_figure(results: Sequence[dict[str, Any]]) -> go.Figure:
    if not results:
        return empty_figure("Latency percentiles", "No runs selected.")
    fig = go.Figure()
    for i, res in enumerate(results):
        lat = res["latency"]
        qs = ("p50_ms", "p90_ms", "p95_ms", "p99_ms")
        values = [lat[q] for q in qs]
        error_y = None
        lo, hi = lat.get("median_ci95_ms") or (None, None)
        if lo is not None:
            # Error bar on p50 only: the 95% confidence interval of the median.
            error_y = {
                "type": "data", "symmetric": False,
                "array": [hi - lat["p50_ms"], 0, 0, 0],
                "arrayminus": [lat["p50_ms"] - lo, 0, 0, 0],
            }
        fig.add_trace(
            go.Bar(name=res["label"], x=["p50", "p90", "p95", "p99"], y=values, marker_color=color_for(i), error_y=error_y)
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="latency per call, ms (lower is better)", rangemode="tozero")
    return _style(fig, "Latency percentiles (error bar: 95% CI of the median)")


def _samples(res: dict[str, Any]) -> np.ndarray:
    return np.asarray((res.get("samples") or {}).get("latency_ms") or [], dtype=np.float64)


def latency_distribution_figure(results: Sequence[dict[str, Any]]) -> go.Figure:
    pairs = [(i, r, _samples(r)) for i, r in enumerate(results)]
    pairs = [p for p in pairs if p[2].size]
    if not pairs:
        return empty_figure("Latency distribution", "No raw samples in the selected runs.")
    fig = go.Figure()
    # Clip the x-range at the 99.5th percentile of the slowest run so one spike does not flatten the plot.
    upper = max(float(np.percentile(s, 99.5)) for _, _, s in pairs)
    for i, res, s in pairs:
        fig.add_trace(
            go.Histogram(
                x=s, name=res["label"], marker_color=color_for(i), opacity=0.6, nbinsx=60,
                histnorm="probability",
            )
        )
        fig.add_vline(x=res["latency"]["p50_ms"], line={"color": color_for(i), "dash": "dot", "width": 1})
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="latency per call, ms (dotted line: median)", range=[0, upper * 1.05])
    fig.update_yaxes(title="share of calls")
    return _style(fig, "Latency distribution")


def latency_timeline_figure(results: Sequence[dict[str, Any]]) -> go.Figure:
    pairs = [(i, r, _samples(r)) for i, r in enumerate(results)]
    pairs = [p for p in pairs if p[2].size]
    if not pairs:
        return empty_figure("Latency over the run", "No raw samples in the selected runs.")
    fig = go.Figure()
    for i, res, s in pairs:
        step = max(1, s.size // MAX_POINTS)
        idx = np.arange(0, s.size, step)
        fig.add_trace(
            go.Scattergl(
                x=idx, y=s[idx], mode="markers", name=res["label"],
                marker={"color": color_for(i), "size": 3, "opacity": 0.45},
            )
        )
        window = max(5, s.size // 50)
        if s.size >= 2 * window:
            kernel = np.ones(window) / window
            rolling = np.convolve(s, kernel, mode="valid")
            fig.add_trace(
                go.Scatter(
                    x=np.arange(rolling.size) + window // 2, y=rolling, mode="lines",
                    name=f"{res['label']} (rolling mean)", line={"color": color_for(i), "width": 2},
                    showlegend=False,
                )
            )
    fig.update_xaxes(title="call number")
    fig.update_yaxes(title="latency per call, ms", rangemode="tozero")
    return _style(fig, "Latency over the run (look for warm-up, drift and spikes)")


def stage_breakdown_figure(results: Sequence[dict[str, Any]]) -> go.Figure:
    piped = [r for r in results if r.get("stages")]
    if not piped:
        return empty_figure("Pipeline stages", "No pipeline-mode runs selected.\nRun with  --mode pipeline  to see where the time goes.")
    labels = [r["label"] for r in piped]
    fig = go.Figure()
    for stage, color in (("preprocess", "#f59e0b"), ("inference", "#2563eb"), ("postprocess", "#10b981")):
        fig.add_trace(
            go.Bar(name=stage, y=labels, x=[r["stages"][stage]["p50_ms"] for r in piped], orientation="h", marker_color=color)
        )
    fig.update_layout(barmode="stack", legend={"traceorder": "normal"})
    fig.update_xaxes(title="median time per call, ms")
    return _style(fig, "Where the time goes (pipeline mode, median per stage)")


def _accuracy_points(rows: Sequence[dict[str, Any]], key: str) -> list[tuple[dict[str, Any], float, bool]]:
    """Runs that have the metric, plus the reference models themselves.

    A reference model agrees with itself perfectly (1.0), so a run whose model
    file is the reference of another run is added as a point at 1.0. Without
    that point the plot would show only the quantised model and nothing to
    compare it with.
    """
    points = [(r, float(r[key]), False) for r in rows if r.get(key) is not None and r.get("fps") is not None]
    reference_hashes = {r.get("reference_sha256") for r in rows if r.get(key) is not None} - {None}
    for r in rows:
        if r.get(key) is None and r.get("fps") is not None and r.get("model_sha256") in reference_hashes:
            points.append((r, 1.0, True))
    return points


def accuracy_tradeoff_figure(rows: Sequence[dict[str, Any]]) -> go.Figure:
    title = "Speed vs agreement with the reference"
    for key, label in (("det_f1", "detection F1 @ IoU 0.5"), ("top1_agree", "top-1 agreement"), ("cosine", "cosine similarity")):
        pts = _accuracy_points(rows, key)
        if not pts:
            continue
        ys = [v for _, v, _ in pts]
        fig = go.Figure(
            go.Scatter(
                x=[r["fps"] for r, _, _ in pts], y=ys, mode="markers+text",
                text=[r["label"] + (" (reference)" if is_ref else "") for r, _, is_ref in pts],
                textposition="top center", cliponaxis=False,  # labels of points near the edge must stay visible
                marker={"size": 13, "color": [color_for(i) for i in range(len(pts))]},
                hovertemplate="<b>%{text}</b><br>%{x:.1f} items/s<br>%{y:.4f}<extra></extra>",
            )
        )
        fig.update_xaxes(title="items / second (higher is better)", range=[0, max(r["fps"] for r, _, _ in pts) * 1.35])
        fig.update_yaxes(title=label + " (higher is better)", range=[min(ys) - 0.1, 1.15])  # room for the labels
        return _style(fig, title)
    return empty_figure(title, "No accuracy data yet.\nAdd  --reference FP32.onnx  to a run, or use  infer-bench accuracy ... --into RESULT.json")


def efficiency_figure(rows: Sequence[dict[str, Any]]) -> go.Figure:
    pts = [r for r in rows if r.get("fps_per_w") is not None]
    if not pts:
        return empty_figure(
            "Efficiency (items/s per watt)",
            "No power data.\nRAPL needs Linux + permission; otherwise pass  --power-watts  from a wall meter.",
        )
    fig = go.Figure(
        go.Bar(
            x=[r["label"] for r in pts], y=[r["fps_per_w"] for r in pts],
            marker_color=[color_for(i) for i in range(len(pts))],
            customdata=[[r.get("avg_power_w")] for r in pts],
            hovertemplate="<b>%{x}</b><br>%{y:.2f} items/s per W<br>average power %{customdata[0]:.2f} W<extra></extra>",
        )
    )
    fig.update_yaxes(title="items/s per watt (higher is better)", rangemode="tozero")
    fig.add_annotation(
        text="RAPL = CPU package only; watts from --power-watts = whatever you measured. See each run's power notes.",
        x=0, y=-0.2, xref="paper", yref="paper", showarrow=False, xanchor="left", font={"size": 11, "color": "#6b7280"},
    )
    return _style(fig, "Efficiency (items/s per watt)")
