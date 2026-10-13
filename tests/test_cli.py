import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from infer_bench import cli, results

from conftest import with_opset


def run_cli(capsys, *argv):
    code = cli.main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def sleep_run(capsys, out_dir, label, ms, iterations=40):
    code, out, err = run_cli(
        capsys, "run", "--backend", "sleep", "--sleep-ms", ms, "--iterations", iterations, "--warmup", 3,
        "--no-power", "--label", label, "--results-dir", out_dir,
    )
    assert code == 0, err
    return out


def only_file(folder: Path, prefix: str) -> Path:
    (match,) = [p for p in folder.glob("*.json") if p.name.startswith(prefix)]
    return match


@pytest.mark.timing
def test_selfcheck_passes(capsys):
    code, out, _ = run_cli(capsys, "selfcheck", "--iterations", 200)
    assert code == 0
    assert out.count("PASS") == 2 and "FAIL" not in out
    assert "trustworthy" in out


def test_demo_model_command(tmp_path, capsys):
    code, out, _ = run_cli(capsys, "demo-model", "detector", "-o", tmp_path / "d.onnx")
    assert code == 0 and (tmp_path / "d.onnx").is_file()
    assert "--task detection" in out


def test_run_with_the_sleep_backend_saves_a_valid_results_file(tmp_path, capsys):
    out = sleep_run(capsys, tmp_path, "demo", 1.0)
    saved = only_file(tmp_path, "demo_")
    result = results.load_result(saved)
    assert result["label"] == "demo" and result["latency"]["n"] == 40
    assert "Latency:" in out and "Throughput:" in out and f"Saved: {saved}" in out


def test_results_files_do_not_contain_local_paths(models, tmp_path, capsys):
    """Results get shared and published, so they must not carry folder names (which can hold a user name)."""
    import sys

    out_file = tmp_path / "private_folder" / "r.json"
    code, _, err = run_cli(
        capsys, "run", models["cls_int8"], "--reference", models["cls"], "--iterations", 10, "--warmup", 2,
        "--no-power", "--accuracy-samples", 4, "--threads", 1, "--out", out_file,
    )
    assert code == 0, err
    text = out_file.read_text(encoding="utf-8")
    for private in (str(models["cls"].parent), str(tmp_path), "private_folder", sys.executable):
        assert private not in text
    assert models["cls_int8"].name in text  # the file name is still recorded


def test_run_no_save_writes_nothing(tmp_path, capsys):
    code, out, _ = run_cli(capsys, "run", "--backend", "sleep", "--iterations", 20, "--warmup", 1, "--no-power",
                           "--no-save", "--results-dir", tmp_path)
    assert code == 0 and list(tmp_path.glob("*.json")) == [] and "Saved:" not in out


def test_run_exact_output_path(tmp_path, capsys):
    target = tmp_path / "deep" / "result.json"
    code, _, _ = run_cli(capsys, "run", "--backend", "sleep", "--iterations", 20, "--warmup", 1, "--no-power", "--out", target)
    assert code == 0 and target.is_file()


def test_run_onnx_model_with_a_reference_stores_accuracy(models, tmp_path, capsys):
    out_file = tmp_path / "int8.json"
    code, out, err = run_cli(
        capsys, "run", models["cls_int8"], "--reference", models["cls"], "--iterations", 30, "--warmup", 3,
        "--no-power", "--accuracy-samples", 20, "--threads", 1, "--out", out_file,
    )
    assert code == 0, err
    result = results.load_result(out_file)
    assert result["model"]["precision"] == "int8"
    assert result["accuracy"]["n_compared"] == 20 and result["accuracy"]["reference"]["precision"] == "fp32"
    assert "Accuracy:" in out and "top-1 agreement" in out


def test_run_with_a_detection_task_and_a_reference_scores_the_boxes(models, tmp_path, capsys):
    out_file = tmp_path / "det.json"
    code, out, err = run_cli(
        capsys, "run", models["det_int8"], "--mode", "pipeline", "--task", "detection", "--reference", models["det"],
        "--iterations", 10, "--warmup", 2, "--no-power", "--accuracy-samples", 10, "--threads", 1, "--out", out_file,
    )
    assert code == 0, err
    acc = results.load_result(out_file)["accuracy"]
    assert acc["task"] == "detection" and acc["detection"] is not None
    assert "detection F1" in out or "F1" in out


def test_run_pipeline_mode_with_an_image_folder(models, tmp_path, capsys):
    folder = tmp_path / "imgs"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        Image.fromarray(rng.integers(0, 255, (90, 120, 3), dtype=np.uint8)).save(folder / f"{i}.png")
    code, out, err = run_cli(
        capsys, "run", models["cls"], "--mode", "pipeline", "--task", "classification", "--images", folder,
        "--iterations", 12, "--warmup", 2, "--no-power", "--no-save", "--threads", 1,
    )
    assert code == 0, err
    assert "inputs=images" in out and "Stages:" in out and "mode=pipeline" in out


def test_run_with_several_streams(tmp_path, capsys):
    code, out, err = run_cli(
        capsys, "run", "--backend", "sleep", "--sleep-ms", 3, "--sleep-yield", "--streams", 2, "--duration", 0.3,
        "--warmup", 1, "--no-power", "--no-save",
    )
    assert code == 0, err
    assert "streams=2" in out and "items/s per stream" in out


def test_quantize_then_accuracy_into_a_results_file(models, tmp_path, capsys):
    q = tmp_path / "q.onnx"
    code, out, err = run_cli(capsys, "quantize", models["cls"], "-o", q, "--samples", 8)
    assert code == 0, err
    assert q.is_file() and "smaller" in out and "synthetic" in out

    result_file = tmp_path / "r.json"
    run_cli(capsys, "run", q, "--iterations", 20, "--warmup", 2, "--no-power", "--threads", 1, "--out", result_file)
    code, out, err = run_cli(capsys, "accuracy", models["cls"], q, "--samples", 15, "--into", result_file)
    assert code == 0, err
    assert "cosine similarity" in out and "top-1 agreement" in out and f"Stored in {result_file}" in out
    assert results.load_result(result_file)["accuracy"]["n_compared"] == 15


def test_quantize_a_model_with_an_old_opset_says_it_used_per_tensor(models, tmp_path, capsys):
    old = with_opset(models["cls"], tmp_path / "old.onnx", 12)
    code, out, err = run_cli(capsys, "quantize", old, "-o", tmp_path / "old_int8.onnx", "--samples", 4)
    assert code == 0, err
    assert "per_channel=False" in out and "note: The model uses operator set 12" in out


def test_quantize_failure_is_one_clear_line(models, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("onnxruntime.quantization.quantize_static",
                        lambda *, model_output, **_: Path(model_output).write_bytes(b"garbage"))
    code, out, err = run_cli(capsys, "quantize", models["cls"], "-o", tmp_path / "bad.onnx", "--samples", 2)
    assert code == 2 and err.startswith("error: the quantised model was written") and err.count("\n") == 1
    assert "Wrote" not in out  # it must not claim success


def test_quantize_with_op_types_option(models, tmp_path, capsys):
    code, _, err = run_cli(capsys, "quantize", models["det"], "-o", tmp_path / "d.onnx", "--samples", 8, "--op-types", "Conv",
                           "--task", "detection")
    assert code == 0, err


def test_accuracy_command_for_detection_prints_box_agreement(models, capsys):
    code, out, err = run_cli(capsys, "accuracy", models["det"], models["det_int8_conv"], "--task", "detection", "--samples", 20)
    assert code == 0, err
    assert "detection agreement" in out and "F1" in out


@pytest.mark.timing
def test_compare_summary_formats(tmp_path, capsys):
    sleep_run(capsys, tmp_path, "slow", 2.0)
    sleep_run(capsys, tmp_path, "fast", 1.0)
    slow, fast = only_file(tmp_path, "slow_"), only_file(tmp_path, "fast_")

    code, out, _ = run_cli(capsys, "compare", slow, fast)
    assert code == 0 and "Throughput ratio" in out and "Verdict:" in out and "lower" in out

    code, out, _ = run_cli(capsys, "summary", tmp_path, "--sort", "fps")
    lines = out.strip().splitlines()
    assert code == 0 and "items/s" in lines[0] and lines[2].startswith("fast")  # fastest first

    code, out, _ = run_cli(capsys, "summary", tmp_path, "--format", "markdown")
    assert out.startswith("| run |") and out.count("\n") == 4

    code, out, _ = run_cli(capsys, "summary", tmp_path, "--format", "csv")
    header = out.splitlines()[0].split(",")
    assert "label" in header and "fps" in header and len(out.strip().splitlines()) == 3


def test_summary_without_results(tmp_path, capsys):
    code, out, _ = run_cli(capsys, "summary", tmp_path)
    assert code == 1 and "no results found" in out


def test_summary_skips_broken_files_but_reports_them(tmp_path, capsys):
    sleep_run(capsys, tmp_path, "ok", 1.0)
    (tmp_path / "broken.json").write_text("{nope")
    code, out, err = run_cli(capsys, "summary", tmp_path)
    assert code == 0 and "ok" in out and "skipping" in err and "broken.json" in err


def test_info_command(models, capsys):
    code, out, _ = run_cli(capsys, "info", models["cls"])
    assert code == 0 and "precision=fp32" in out and "input   images" in out and "Conv x4" in out


def test_env_command_prints_json_and_power_status(capsys):
    code, out, _ = run_cli(capsys, "env")
    assert code == 0
    start = out.index("{")
    end = out.rindex("}") + 1
    assert "cpu_model" in json.loads(out[start:end])
    assert "Power:" in out


def test_errors_are_one_line_with_exit_code_2(tmp_path, capsys):
    code, out, err = run_cli(capsys, "run", tmp_path / "missing.onnx", "--no-save")
    assert code == 2 and err.startswith("error: model file not found") and err.count("\n") == 1


def test_missing_model_for_onnx_backend(capsys):
    code, _, err = run_cli(capsys, "run", "--no-save")
    assert code == 2 and "model file is required" in err


def test_bad_size_is_explained(models, capsys):
    code, _, err = run_cli(capsys, "run", models["cls"], "--input-size", "abc", "--no-save")
    assert code == 2 and "size must look like" in err


def test_invalid_batch_size_is_explained(models, capsys):
    code, _, err = run_cli(capsys, "run", models["cls"], "--batch", 0, "--no-save")
    assert code == 2 and "batch_size must be at least 1" in err


def test_fixed_batch_models_explain_how_to_proceed(tmp_path, capsys):
    import onnx
    from onnx import TensorProto, helper

    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "fixed.onnx"
    onnx.save(model, str(path))
    code, _, err = run_cli(capsys, "run", path, "--batch", 4, "--no-save", "--no-power")
    assert code == 2 and "fixed batch dimension of 1" in err
    code, out, _ = run_cli(capsys, "run", path, "--batch", 1, "--iterations", 10, "--warmup", 1, "--no-save", "--no-power")
    assert code == 0 and "Latency:" in out


def test_debug_flag_shows_the_traceback(tmp_path):
    with pytest.raises(FileNotFoundError):
        cli.main(["--debug", "run", str(tmp_path / "missing.onnx"), "--no-save"])


def test_all_console_output_is_plain_ascii(models, tmp_path, capsys):
    """Windows consoles with legacy code pages choke on characters like arrows or plus-minus."""
    outputs = []
    outputs.append(sleep_run(capsys, tmp_path, "a", 1.0))
    sleep_run(capsys, tmp_path, "b", 2.0)
    outputs.append(run_cli(capsys, "compare", only_file(tmp_path, "a_"), only_file(tmp_path, "b_"))[1])
    outputs.append(run_cli(capsys, "summary", tmp_path)[1])
    outputs.append(run_cli(capsys, "selfcheck", "--iterations", 100)[1])
    outputs.append(run_cli(capsys, "info", models["cls"])[1])
    outputs.append(run_cli(capsys, "accuracy", models["det"], models["det_int8"], "--task", "detection", "--samples", 5)[1])
    outputs.append(run_cli(capsys, "quantize", models["cls"], "-o", tmp_path / "q.onnx", "--samples", 4)[1])
    for text in outputs:
        assert text.isascii(), [c for c in text if not c.isascii()]


def test_dashboard_command_wires_arguments(tmp_path, monkeypatch, capsys):
    calls = {}

    class FakeApp:
        def run(self, **kwargs):
            calls["run"] = kwargs

    import infer_bench.dashboard as dashboard

    monkeypatch.setattr(dashboard, "create_app", lambda folder: calls.setdefault("folder", folder) and FakeApp())
    code, out, _ = run_cli(capsys, "dashboard", "--results", tmp_path, "--port", 9123, "--host", "0.0.0.0", "--dash-debug")
    assert code == 0
    assert calls["folder"] == str(tmp_path)
    assert calls["run"] == {"host": "0.0.0.0", "port": 9123, "debug": True}
    assert "http://0.0.0.0:9123" in out


def test_format_table_aligns_columns():
    text = cli.format_table(["a", "long header"], [["1", "x"], ["22222", "yy"]])
    lines = text.splitlines()
    assert len({len(line.rstrip()) for line in lines[:1]}) == 1
    assert lines[1].startswith("-----  -----------")


def test_ort_options_split_cores_between_streams(monkeypatch):
    import argparse

    monkeypatch.setattr(cli.os, "cpu_count", lambda: 8)
    args = argparse.Namespace(providers="CPUExecutionProvider", threads=0, inter_threads=0, graph_opt="all")
    assert cli._ort_options(args, streams=1)["intra_op_threads"] == 0  # ONNX Runtime default
    assert cli._ort_options(args, streams=4)["intra_op_threads"] == 2
    assert cli._ort_options(args, streams=16)["intra_op_threads"] == 1
    args.threads = 3
    assert cli._ort_options(args, streams=4)["intra_op_threads"] == 3  # an explicit choice wins
