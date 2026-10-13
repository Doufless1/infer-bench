import json

import pytest

pytest.importorskip("dash")

from infer_bench import dashboard, results  # noqa: E402

from conftest import make_result  # noqa: E402


@pytest.fixture()
def folder(tmp_path):
    for label, p50, fps in (("fp32", 2.0, 500.0), ("int8", 1.0, 1000.0)):
        doc = make_result(label, p50, (p50 * 0.97, p50 * 1.03), fps)
        doc["created_utc"] = f"2026-01-01T00:00:0{1 if label == 'fp32' else 2}+00:00"
        results.save_result(doc, tmp_path / f"{label}.json")
    return tmp_path


def walk_ids(node, found=None):
    """All component ids in a serialised Dash layout."""
    found = set() if found is None else found
    if isinstance(node, dict):
        props = node.get("props", {})
        if "id" in props:
            found.add(props["id"])
        walk_ids(props.get("children"), found)
    elif isinstance(node, list):
        for child in node:
            walk_ids(child, found)
    return found


def test_load_state_reads_valid_files_and_reports_broken_ones(folder):
    (folder / "broken.json").write_text("{nope")
    state = dashboard.load_state(folder)
    assert [r["label"] for r in state["rows"]] == ["fp32", "int8"]  # oldest first
    assert set(state["results"]) == {"id-fp32", "id-int8"}
    assert len(state["errors"]) == 1 and "broken.json" in state["errors"][0]


def test_load_state_with_a_missing_folder(tmp_path):
    state = dashboard.load_state(tmp_path / "missing")
    assert state == {"results": {}, "rows": [], "errors": []}


def test_the_same_run_copied_twice_is_kept_as_two_rows(folder):
    (folder / "copy.json").write_text((folder / "fp32.json").read_text())
    state = dashboard.load_state(folder)
    assert len(state["rows"]) == 3 and len({r["id"] for r in state["rows"]}) == 3


def test_table_data_rounds_numbers_and_keeps_ids(folder):
    state = dashboard.load_state(folder)
    state["rows"][0]["p50_ms"] = 1.23456789
    data = dashboard.table_data(state["rows"])
    assert data[0]["id"] == "id-fp32" and data[0]["p50_ms"] == 1.235
    assert set(data[0]) == {"id"} | {key for key, _, _ in dashboard.TABLE_COLUMNS}


def test_render_shows_every_run_when_nothing_is_selected(folder):
    state = dashboard.load_state(folder)
    outputs = dashboard.render(state, [])
    assert len(outputs) == len(dashboard.GRAPH_IDS) + 2
    assert list(outputs[0].data[0].x) == ["fp32", "int8"]


def test_render_respects_the_selection_and_ignores_unknown_ids(folder):
    state = dashboard.load_state(folder)
    only_int8 = dashboard.render(state, ["id-int8", "gone"])
    assert list(only_int8[0].data[0].x) == ["int8"]
    everything = dashboard.render(state, ["gone"])
    assert list(everything[0].data[0].x) == ["fp32", "int8"]


def text_of(component) -> str:
    return json.dumps(component.to_plotly_json(), default=lambda o: o.to_plotly_json() if hasattr(o, "to_plotly_json") else str(o))


def test_two_selected_runs_produce_a_written_comparison(folder):
    state = dashboard.load_state(folder)
    panel = dashboard.render(state, ["id-fp32", "id-int8"])[len(dashboard.GRAPH_IDS)]
    text = text_of(panel)
    assert "Median latency is 2.00x lower" in text and "Throughput x2.000" in text


def test_other_selection_sizes_get_a_hint_instead(folder):
    state = dashboard.load_state(folder)
    panel = dashboard.render(state, ["id-fp32"])[len(dashboard.GRAPH_IDS)]
    assert "exactly two" in text_of(panel)


def test_warnings_panel_lists_run_warnings_and_skipped_files(folder):
    doc = make_result("noisy", 1.0, (0.9, 1.1), 100.0)
    doc["warnings"] = ["Latency drifted +9.0%"]
    results.save_result(doc, folder / "noisy.json")
    (folder / "broken.json").write_text("{nope")
    state = dashboard.load_state(folder)
    panel = dashboard.render(state, ["id-noisy"])[-1]
    text = text_of(panel[0] if isinstance(panel, list) else panel)
    assert "Latency drifted" in text and "1 warning(s)" in text
    assert "broken.json" in text_of(dashboard.render(state, ["id-noisy"])[-1][-1])


def test_empty_folder_renders_without_error(tmp_path):
    state = dashboard.load_state(tmp_path)
    outputs = dashboard.render(state, [])
    assert len(outputs) == len(dashboard.GRAPH_IDS) + 2


def test_app_serves_the_page_layout_and_callbacks(folder):
    app = dashboard.create_app(folder)
    client = app.server.test_client()
    assert client.get("/").status_code == 200

    layout = client.get("/_dash-layout").get_json()
    ids = walk_ids(layout)
    assert {"runs", "reload", "status", "compare-panel", "warnings-panel", *dashboard.GRAPH_IDS} <= ids
    (table,) = [n for n in _find(layout, "runs")]
    assert [row["label"] for row in table["props"]["data"]] == ["fp32", "int8"]

    deps = client.get("/_dash-dependencies").get_json()
    assert len(deps) == 2
    wired = {part.split(".")[0] for dep in deps for part in dep["output"].strip(".").split("...")}
    assert wired <= ids  # every callback output refers to a component that exists in the layout


def _find(node, component_id):
    if isinstance(node, dict):
        if node.get("props", {}).get("id") == component_id:
            yield node
        yield from _find(node.get("props", {}).get("children"), component_id)
    elif isinstance(node, list):
        for child in node:
            yield from _find(child, component_id)


def test_the_folder_is_read_again_on_every_page_load(folder):
    app = dashboard.create_app(folder)
    client = app.server.test_client()
    before = list(_find(client.get("/_dash-layout").get_json(), "runs"))[0]["props"]["data"]
    results.save_result(make_result("new", 3.0, (2.9, 3.1), 300.0), folder / "new.json")
    after = list(_find(client.get("/_dash-layout").get_json(), "runs"))[0]["props"]["data"]
    assert len(before) == 2 and len(after) == 3
