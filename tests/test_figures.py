import numpy as np
import plotly.graph_objects as go
import pytest

from infer_bench import figures, results

from conftest import make_result


def rows_for(*docs):
    out = []
    for doc in docs:
        row = results.flatten_result(doc)
        row["id"] = doc["run_id"]
        out.append(row)
    return out


@pytest.fixture()
def two_runs():
    a = make_result("fp32", 2.0, (1.95, 2.05), 500.0, power={"avg_power_w": 10.0, "items_per_s_per_w": 50.0})
    b = make_result("int8", 1.0, (0.98, 1.02), 1000.0, power={"avg_power_w": 8.0, "items_per_s_per_w": 125.0},
                    accuracy={"cosine_similarity_mean": 0.998, "top1_agreement": 0.97, "reference": {"sha256": "sha-fp32"}})
    for doc in (a, b):
        doc["samples"]["latency_ms"] = list(np.random.default_rng(0).normal(doc["latency"]["p50_ms"], 0.05, 400))
    return a, b


def is_placeholder(fig: go.Figure) -> bool:
    return len(fig.data) == 0 and len(fig.layout.annotations) == 1


def test_every_figure_serialises_to_json(two_runs):
    a, b = two_runs
    rows = rows_for(a, b)
    piped = make_result("pipe", 3.0, (2.9, 3.1), 300.0,
                        stages={k: {"p50_ms": v} for k, v in (("preprocess", 1.0), ("inference", 1.8), ("postprocess", 0.2))})
    figs = [
        figures.throughput_figure(rows),
        figures.latency_percentiles_figure([a, b]),
        figures.latency_distribution_figure([a, b]),
        figures.latency_timeline_figure([a, b]),
        figures.stage_breakdown_figure([a, piped]),
        figures.accuracy_tradeoff_figure(rows),
        figures.efficiency_figure(rows),
    ]
    for fig in figs:
        assert isinstance(fig, go.Figure)
        assert fig.to_json()  # what Dash sends to the browser


def test_empty_inputs_give_explanatory_placeholders():
    for fig in (figures.throughput_figure([]), figures.latency_percentiles_figure([]),
                figures.latency_distribution_figure([]), figures.latency_timeline_figure([]),
                figures.stage_breakdown_figure([]), figures.accuracy_tradeoff_figure([]), figures.efficiency_figure([])):
        assert is_placeholder(fig)
    assert "--mode pipeline" in figures.stage_breakdown_figure([]).layout.annotations[0].text
    assert "--reference" in figures.accuracy_tradeoff_figure([]).layout.annotations[0].text
    assert "--power-watts" in figures.efficiency_figure([]).layout.annotations[0].text


def test_throughput_bars(two_runs):
    fig = figures.throughput_figure(rows_for(*two_runs))
    assert list(fig.data[0].x) == ["fp32", "int8"] and list(fig.data[0].y) == [500.0, 1000.0]


def test_percentile_bars_carry_the_confidence_interval_on_p50(two_runs):
    fig = figures.latency_percentiles_figure(list(two_runs))
    first = fig.data[0]
    assert list(first.x) == ["p50", "p90", "p95", "p99"]
    assert first.error_y.array[0] == pytest.approx(2.05 - 2.0)
    assert first.error_y.arrayminus[0] == pytest.approx(2.0 - 1.95)
    assert list(first.error_y.array[1:]) == [0, 0, 0]


def test_timeline_downsamples_long_runs():
    doc = make_result("long", 1.0, (0.9, 1.1), 100.0)
    doc["samples"]["latency_ms"] = list(np.random.default_rng(1).normal(1.0, 0.1, 50_000))
    fig = figures.latency_timeline_figure([doc])
    assert len(fig.data[0].x) <= figures.MAX_POINTS
    assert len(fig.data) == 2  # points plus the rolling mean


def test_timeline_and_distribution_skip_runs_without_samples(two_runs):
    a, b = two_runs
    b = dict(b)
    b["samples"] = {}
    assert len(figures.latency_distribution_figure([a, b]).data) == 1


def test_stage_breakdown_only_includes_pipeline_runs(two_runs):
    a, _ = two_runs
    piped = make_result("pipe", 3.0, (2.9, 3.1), 300.0,
                        stages={k: {"p50_ms": v} for k, v in (("preprocess", 1.0), ("inference", 1.8), ("postprocess", 0.2))})
    fig = figures.stage_breakdown_figure([a, piped])
    assert [t.name for t in fig.data] == ["preprocess", "inference", "postprocess"]
    assert list(fig.data[0].y) == ["pipe"] and fig.data[1].x[0] == 1.8


def test_tradeoff_adds_the_reference_model_at_perfect_agreement(two_runs):
    fig = figures.accuracy_tradeoff_figure(rows_for(*two_runs))
    trace = fig.data[0]
    assert sorted(trace.text) == ["fp32 (reference)", "int8"]
    values = dict(zip(trace.text, trace.y))
    assert values["fp32 (reference)"] == 1.0 and values["int8"] == 0.97  # top-1 agreement is preferred over cosine
    assert "top-1 agreement" in fig.layout.yaxis.title.text


def test_tradeoff_prefers_detection_f1():
    a = make_result("ref", 2.0, (1.9, 2.1), 100.0)
    b = make_result("cand", 1.0, (0.9, 1.1), 200.0,
                    accuracy={"cosine_similarity_mean": 0.99, "top1_agreement": None, "detection": {"f1_iou50": 0.9},
                              "reference": {"sha256": "sha-ref"}})
    fig = figures.accuracy_tradeoff_figure(rows_for(a, b))
    assert "detection F1" in fig.layout.yaxis.title.text


def test_efficiency_bars(two_runs):
    fig = figures.efficiency_figure(rows_for(*two_runs))
    assert list(fig.data[0].y) == [50.0, 125.0]


def test_colours_are_stable_per_position():
    assert figures.color_for(0) == figures.PALETTE[0]
    assert figures.color_for(len(figures.PALETTE)) == figures.PALETTE[0]
