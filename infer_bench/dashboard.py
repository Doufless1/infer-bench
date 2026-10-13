"""Results dashboard (Plotly Dash).

Reads every ``*.json`` results file in a folder and shows them in a table with
linked charts. Select rows to compare runs; with nothing selected all runs are
shown. Select exactly two runs to get a written comparison.

The server only ever reads the folder: it never runs benchmarks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dash import Dash, Input, Output, dash_table, dcc, html

from .compare import compare_results
from .figures import (
    accuracy_tradeoff_figure,
    efficiency_figure,
    latency_distribution_figure,
    latency_percentiles_figure,
    latency_timeline_figure,
    stage_breakdown_figure,
    throughput_figure,
)
from .results import flatten_result, iter_results

# (row key, column title, decimals or None for text)
TABLE_COLUMNS = (
    ("label", "run", None),
    ("precision", "precision", None),
    ("mode", "mode", None),
    ("batch", "batch", 0),
    ("streams", "streams", 0),
    ("fps", "items/s", 1),
    ("p50_ms", "p50 ms", 3),
    ("p95_ms", "p95 ms", 3),
    ("p99_ms", "p99 ms", 3),
    ("avg_power_w", "watts", 2),
    ("fps_per_w", "items/s/W", 2),
    ("top1_agree", "top-1 agr.", 3),
    ("cosine", "cosine", 5),
    ("det_f1", "det. F1", 3),
    ("n_warnings", "warn.", 0),
)

GRAPH_IDS = (
    "g-throughput", "g-percentiles", "g-distribution", "g-timeline", "g-stages", "g-tradeoff", "g-efficiency",
)

_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }
.wrap { max-width: 1400px; margin: 0 auto; padding: 20px 24px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { color: #475569; font-size: 13px; margin-bottom: 14px; }
.card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; margin-bottom: 16px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 16px; }
.grid .card { margin: 0; padding: 6px 8px; }
button { background: #2563eb; color: #fff; border: 0; border-radius: 6px; padding: 7px 14px; font-size: 13px; cursor: pointer; }
button:hover { background: #1d4ed8; }
.warn { color: #92400e; background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; padding: 6px 10px; margin: 4px 0; font-size: 13px; }
.ok { color: #065f46; }
code { background: #f1f5f9; padding: 1px 5px; border-radius: 4px; }
@media (max-width: 640px) { .grid { grid-template-columns: 1fr; } .wrap { padding: 12px; } }
"""


def load_state(results_dir: Path) -> dict[str, Any]:
    """Read the results folder into {'results': {id: result}, 'rows': [...], 'errors': [...]}."""
    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if results_dir.is_dir():
        for path, result, error in iter_results(results_dir):
            if result is None:
                errors.append(f"{path.name}: {error}")
                continue
            run_id = str(result["run_id"])
            if run_id in results:  # the same file copied twice
                run_id = f"{run_id}#{path.name}"
            row = flatten_result(result, path)
            row["id"] = run_id
            results[run_id] = result
            rows.append(row)
    rows.sort(key=lambda r: r.get("created_utc") or "")
    return {"results": results, "rows": rows, "errors": errors}


def table_data(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows rounded for display (kept numeric so the table sorts properly)."""
    data = []
    for r in rows:
        item: dict[str, Any] = {"id": r["id"]}
        for key, _title, digits in TABLE_COLUMNS:
            value = r.get(key)
            if digits is not None and isinstance(value, float):
                value = round(value, digits)
            item[key] = value
        data.append(item)
    return data


def _comparison_panel(state: dict[str, Any], ids: list[str]) -> Any:
    if len(ids) != 2:
        return html.Div("Select exactly two runs in the table to get a written comparison.", className="sub")
    base, other = state["results"][ids[0]], state["results"][ids[1]]
    cmp = compare_results(base, other)
    ratio = cmp["throughput_ratio"]
    lat = cmp["latency_ratio"]
    children: list[Any] = [
        html.Div([html.B("Base: "), cmp["base"], html.B("   Other: "), cmp["other"],
                  html.Span("   (ratios are other / base; the first listed or ticked run is the base)", className="sub")]),
        html.Div(
            f"Throughput x{ratio:.3f}   |   latency ratio p50 {lat['p50']:.3f}, p95 {lat['p95']:.3f}, p99 {lat['p99']:.3f} (below 1 = faster)"
            if ratio and all(v is not None for v in lat.values()) else "Not enough data for ratios."
        ),
        html.Div(html.B(cmp["verdict"]), style={"margin": "6px 0"}),
    ]
    if cmp["power"]:
        children.append(html.Div(f"Efficiency (items/s per W) x{cmp['power']['items_per_s_per_w_ratio']:.3f}"))
    children.extend(html.Div(w, className="warn") for w in cmp["warnings"])
    return html.Div(children)


def _warnings_panel(state: dict[str, Any], ids: list[str]) -> Any:
    blocks: list[Any] = []
    for run_id in ids:
        res = state["results"][run_id]
        warnings = res.get("warnings") or []
        header = f"{res['label']}: " + (f"{len(warnings)} warning(s)" if warnings else "no warnings")
        blocks.append(html.Details([html.Summary(header)] + [html.Div(w, className="warn") for w in warnings], open=bool(warnings) and len(ids) <= 3))
    for err in state["errors"]:
        blocks.append(html.Div(f"Skipped file {err}", className="warn"))
    return blocks or html.Div("No runs.", className="sub")


def render(state: dict[str, Any], selected_ids: list[str] | None) -> tuple:
    """Everything that depends on the selection: seven figures and two panels."""
    known = {r["id"] for r in state["rows"]}
    ids = [i for i in (selected_ids or []) if i in known] or [r["id"] for r in state["rows"]]
    rows = [r for i in ids for r in state["rows"] if r["id"] == i]
    results = [state["results"][i] for i in ids]
    return (
        throughput_figure(rows),
        latency_percentiles_figure(results),
        latency_distribution_figure(results),
        latency_timeline_figure(results),
        stage_breakdown_figure(results),
        accuracy_tradeoff_figure(rows),
        efficiency_figure(rows),
        _comparison_panel(state, ids),
        _warnings_panel(state, ids),
    )


def create_app(results_dir: str | Path = "results") -> Dash:
    results_dir = Path(results_dir)
    app = Dash(__name__, title="infer-bench dashboard")
    app.index_string = (
        "<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}"
        f"<style>{_CSS}</style></head><body>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body></html>"
    )
    state: dict[str, Any] = load_state(results_dir)

    def status_text() -> str:
        return f"{len(state['rows'])} run(s) in {results_dir}"  # as typed: no long absolute path in shared screenshots

    columns = [{"name": title, "id": key, **({"type": "numeric"} if digits is not None else {})} for key, title, digits in TABLE_COLUMNS]

    def layout() -> Any:
        state.update(load_state(results_dir))  # re-read the folder on every page load
        no_runs = not state["rows"]
        return html.Div(
            className="wrap",
            children=[
                html.H1("infer-bench dashboard"),
                html.Div(id="status", className="sub", children=status_text()),
                html.Div(
                    [html.Button("Reload results", id="reload", n_clicks=0)],
                    style={"marginBottom": "12px"},
                ),
                html.Div(
                    className="card",
                    children=[
                        html.Div(
                            [html.Span("No results yet in this folder. Run "), html.Code("infer-bench run MODEL.onnx"),
                             html.Span(" and press Reload.")] if no_runs else
                            "Tick rows to compare runs (nothing ticked = all runs). Click a column title to sort.",
                            className="sub",
                        ),
                        dash_table.DataTable(
                            id="runs",
                            columns=columns,
                            data=table_data(state["rows"]),
                            row_selectable="multi",
                            selected_row_ids=[],
                            sort_action="native",
                            page_size=12,
                            style_table={"overflowX": "auto"},
                            style_cell={"fontSize": 13, "padding": "6px 8px", "textAlign": "left", "minWidth": 50},
                            style_header={"fontWeight": "600", "backgroundColor": "#f1f5f9"},
                        ),
                    ],
                ),
                html.Div(className="card", children=[html.B("Comparison"), html.Div(id="compare-panel")]),
                html.Div(
                    className="grid",
                    children=[html.Div(className="card", children=dcc.Graph(id=g, config={"displaylogo": False})) for g in GRAPH_IDS],
                ),
                html.Div(className="card", style={"marginTop": "16px"}, children=[html.B("Warnings"), html.Div(id="warnings-panel")]),
            ],
        )

    app.layout = layout

    @app.callback(
        *(Output(g, "figure") for g in GRAPH_IDS),
        Output("compare-panel", "children"),
        Output("warnings-panel", "children"),
        Input("runs", "selected_row_ids"),
        Input("runs", "data"),
    )
    def update(selected_ids, _data):  # noqa: ANN001 - Dash callback
        return render(state, selected_ids)

    @app.callback(
        Output("runs", "data"),
        Output("runs", "selected_row_ids"),
        Output("status", "children"),
        Input("reload", "n_clicks"),
        prevent_initial_call=True,
    )
    def reload(_clicks):  # noqa: ANN001
        state.update(load_state(results_dir))
        return table_data(state["rows"]), [], status_text()

    return app
