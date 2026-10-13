import json

import numpy as np
import pytest

from infer_bench import results

from conftest import make_result


def test_roundtrip_through_a_file(tmp_path, sleep_result):
    path = results.save_result(sleep_result, tmp_path / "sub" / "r.json")
    loaded = results.load_result(path)
    assert loaded["label"] == sleep_result["label"]
    assert loaded["latency"]["p50_ms"] == pytest.approx(sleep_result["latency"]["p50_ms"])
    assert loaded["samples"]["latency_ms"] == sleep_result["samples"]["latency_ms"]


def test_generated_results_are_valid(sleep_result):
    assert results.validate_result(sleep_result) == []


def test_numpy_values_and_non_finite_numbers_become_valid_json(tmp_path):
    doc = make_result("x", 1.0, (0.9, 1.1), 100.0)
    doc["extra"] = {"a": np.float32(1.5), "b": np.int64(3), "c": np.array([1, 2]), "d": float("nan"), "e": float("inf"),
                    "f": np.bool_(True), "g": (1, 2), "h": tmp_path}
    path = results.save_result(doc, tmp_path / "r.json")
    raw = json.loads(path.read_text())  # strict parsing: would fail on a bare NaN with strict hooks below
    assert raw["extra"] == {"a": 1.5, "b": 3, "c": [1, 2], "d": None, "e": None, "f": True, "g": [1, 2], "h": str(tmp_path)}
    json.loads(path.read_text(), parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))


def test_validate_reports_problems():
    assert "top level is not an object" in results.validate_result([])
    doc = make_result("x", 1.0, (0.9, 1.1), 100.0)
    del doc["latency"]["p95_ms"]
    del doc["config"]
    doc["schema_version"] = "2.0"
    text = " ".join(results.validate_result(doc))
    assert "missing key 'config'" in text and "latency is missing 'p95_ms'" in text and "unsupported schema_version" in text


def test_load_result_rejects_invalid_files(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"hello": "world"}))
    with pytest.raises(ValueError, match="not a valid results file"):
        results.load_result(path)


def test_iter_results_reports_broken_files_instead_of_crashing(tmp_path):
    results.save_result(make_result("good", 1.0, (0.9, 1.1), 100.0), tmp_path / "good.json")
    (tmp_path / "broken.json").write_text("{ not json")
    (tmp_path / "other.json").write_text(json.dumps({"x": 1}))
    (tmp_path / "readme.txt").write_text("ignored")
    found = {p.name: (r is not None, e) for p, r, e in results.iter_results(tmp_path)}
    assert set(found) == {"good.json", "broken.json", "other.json"}
    assert found["good.json"][0] is True
    assert found["broken.json"][0] is False and found["other.json"][0] is False


def test_limit_samples_rounds_and_thins_very_long_runs():
    samples, truncated = results.limit_samples([1.23456789, 2.0])
    assert samples == [1.234568, 2.0] and not truncated
    big = np.arange(results.MAX_STORED_SAMPLES * 2, dtype=float)
    thinned, truncated = results.limit_samples(big)
    assert truncated and len(thinned) == results.MAX_STORED_SAMPLES
    assert thinned[0] == 0.0 and thinned[-1] == big[-1]  # evenly spread, ends included


def test_slugify_and_default_path(tmp_path):
    assert results.slugify("My Run: FP32 / b=1") == "my-run-fp32-b-1"
    assert results.slugify("***") == "run"
    p = results.default_output_path(tmp_path, "A b", "ID1")
    assert p == tmp_path / "a-b_ID1.json"


def test_run_ids_are_unique():
    assert len({results.new_run_id() for _ in range(50)}) == 50


def test_flatten_result_picks_the_columns_used_by_tables_and_dashboard():
    doc = make_result("fp32", 2.0, (1.9, 2.1), 500.0)
    doc["power"] = {"avg_power_w": 10.0, "items_per_s_per_w": 50.0, "energy_per_item_mj": 20.0}
    doc["accuracy"] = {"cosine_similarity_mean": 0.999, "top1_agreement": 0.97, "detection": {"f1_iou50": 0.9},
                       "reference": {"sha256": "abc"}}
    row = results.flatten_result(doc, "some/file.json")
    assert row["label"] == "fp32" and row["fps"] == 500.0 and row["p50_ms"] == 2.0
    assert (row["ci_lo_ms"], row["ci_hi_ms"]) == (1.9, 2.1)
    assert row["avg_power_w"] == 10.0 and row["fps_per_w"] == 50.0
    assert row["cosine"] == 0.999 and row["top1_agree"] == 0.97 and row["det_f1"] == 0.9
    assert row["model_sha256"] == "sha-fp32" and row["reference_sha256"] == "abc"
    assert row["file"] == "some/file.json"


def test_flatten_result_survives_missing_optional_blocks():
    row = results.flatten_result(make_result("x", 1.0, (0.9, 1.1), 100.0))
    assert row["avg_power_w"] is None and row["cosine"] is None and row["det_f1"] is None
