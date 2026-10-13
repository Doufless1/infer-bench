import platform

from infer_bench import __version__, env


def test_capture_environment_has_the_important_fields():
    e = env.capture_environment(measure_load=False)
    assert e["python_version"] == platform.python_version()
    assert e["harness_version"] == __version__
    assert e["cpu_count_logical"] and e["cpu_count_logical"] >= 1
    assert e["cpu_model"]
    assert "numpy" in e["packages"] and "onnxruntime" in e["packages"]
    assert "CPUExecutionProvider" in e["ort_available_providers"]
    assert e["timestamp_utc"].endswith("+00:00")
    assert e["cpu_busy_percent_at_start"] is None  # measure_load=False


def test_noise_warnings_for_a_quiet_performance_machine():
    quiet = {"cpu_governor": "performance", "on_battery": False, "cpu_busy_percent_at_start": 3.0, "in_container": False}
    assert env.noise_warnings(quiet) == []


def test_noise_warnings_flag_each_problem():
    noisy = {"cpu_governor": "powersave", "on_battery": True, "cpu_busy_percent_at_start": 60.0, "in_container": True}
    text = " ".join(env.noise_warnings(noisy))
    assert "powersave" in text and "battery" in text and "60%" in text and "container" in text


def test_noise_warnings_ignore_missing_information():
    assert env.noise_warnings({}) == []  # e.g. Windows: no governor, no battery info


def test_comparability_warnings_identical_environments():
    e = {"cpu_model": "A", "machine": "x86_64", "python_version": "3.12.1", "packages": {"onnxruntime": "1.20"}}
    assert env.comparability_warnings(e, dict(e)) == []


def test_comparability_warnings_report_differences():
    a = {"cpu_model": "A", "machine": "x86_64", "python_version": "3.12.1", "packages": {"onnxruntime": "1.20", "dash": "2"}}
    b = {"cpu_model": "B", "machine": "arm64", "python_version": "3.12.1", "packages": {"onnxruntime": "1.21", "dash": "3"}}
    text = " ".join(env.comparability_warnings(a, b))
    assert "CPU model differs" in text and "architecture differs" in text and "onnxruntime" in text
    assert "dash" not in text  # packages that do not affect speed are not reported
