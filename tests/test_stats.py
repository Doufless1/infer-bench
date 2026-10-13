import numpy as np
import pytest

from infer_bench import stats


def test_percentile_matches_numpy_linear_definition():
    data = np.arange(1, 101, dtype=float)  # 1..100
    assert stats.percentile(data, 50) == pytest.approx(50.5)
    assert stats.percentile(data, 95) == pytest.approx(95.05)
    assert stats.percentile(data, 99) == pytest.approx(99.01)


@pytest.mark.parametrize("bad", [[], [1.0, float("nan")], [float("inf")]])
def test_as_array_rejects_empty_and_non_finite(bad):
    with pytest.raises(ValueError):
        stats.as_array(bad)


def test_summarize_constant_samples_has_zero_spread():
    out = stats.summarize([2.0] * 200)
    assert out["n"] == 200
    assert out["p50_ms"] == out["p99_ms"] == out["mean_ms"] == 2.0
    assert out["std_ms"] == 0.0
    assert out["median_ci95_ms"] == [2.0, 2.0]
    assert out["drift"]["stable"] is True
    assert out["outliers"]["count_high"] == 0


def test_summarize_reports_the_expected_keys():
    out = stats.summarize(np.linspace(1, 2, 300))
    for key in ("p50_ms", "p90_ms", "p95_ms", "p99_ms", "mean_ms", "std_ms", "min_ms", "max_ms",
                "median_ci95_ms", "ci_method", "outliers", "drift", "unit", "n"):
        assert key in out
    assert out["p50_ms"] <= out["p90_ms"] <= out["p95_ms"] <= out["p99_ms"] <= out["max_ms"]
    lo, hi = out["median_ci95_ms"]
    assert lo <= out["p50_ms"] <= hi


def test_single_sample_is_handled():
    out = stats.summarize([3.0])
    assert out["p50_ms"] == 3.0 and out["std_ms"] == 0.0
    assert out["drift"]["checked"] is False


def test_bootstrap_ci_is_deterministic_for_a_seed():
    data = np.random.default_rng(0).normal(10, 1, 500)
    assert stats.bootstrap_median_ci(data, seed=7) == stats.bootstrap_median_ci(data, seed=7)
    assert stats.bootstrap_median_ci(data, seed=7) != stats.bootstrap_median_ci(data, seed=8)


def test_bootstrap_ci_covers_the_true_median_most_of_the_time():
    # 40 independent experiments; a 95% interval should cover the true median (10.0) in nearly all of them.
    hits = 0
    for seed in range(40):
        data = np.random.default_rng(seed).normal(10.0, 1.0, 300)
        lo, hi = stats.bootstrap_median_ci(data, resamples=300, seed=seed)
        hits += lo <= 10.0 <= hi
    assert hits >= 34  # a perfect 95% interval would give 38 on average; allow for chance


def test_block_bootstrap_is_wider_than_iid_for_correlated_samples():
    # An AR(1) series: consecutive samples are strongly correlated, so the median is less certain
    # than an independent bootstrap would claim. That is the reason for resampling blocks.
    rng = np.random.default_rng(3)
    x = np.zeros(2000)
    for i in range(1, x.size):
        x[i] = 0.95 * x[i - 1] + rng.normal()
    x = x - x.min() + 1.0
    lo1, hi1 = stats.bootstrap_median_ci(x, block=1, resamples=400, seed=0)
    lo2, hi2 = stats.bootstrap_median_ci(x, block=40, resamples=400, seed=0)
    assert (hi2 - lo2) > 1.5 * (hi1 - lo1)


def test_block_length_rule_of_thumb():
    assert stats.default_block_length(1) == 1
    assert stats.default_block_length(1000) == 10
    assert stats.default_block_length(8) == 2


def test_bootstrap_rejects_bad_confidence():
    with pytest.raises(ValueError):
        stats.bootstrap_median_ci([1.0, 2.0, 3.0], confidence=1.5)


def test_drift_check_detects_a_slowdown():
    data = np.concatenate([np.full(100, 1.0), np.full(100, 1.3)])
    d = stats.drift_check(data)
    assert d["checked"] and not d["stable"]
    assert d["relative_change"] == pytest.approx(0.3)


def test_drift_check_accepts_a_stable_run_and_small_changes():
    assert stats.drift_check(np.full(100, 1.0))["stable"]
    data = np.concatenate([np.full(100, 1.0), np.full(100, 1.04)])
    assert stats.drift_check(data)["stable"]  # +4% is below the 5% threshold


def test_drift_check_threshold_is_configurable():
    data = np.concatenate([np.full(100, 1.0), np.full(100, 1.04)])
    assert not stats.drift_check(data, threshold=0.02)["stable"]


def test_drift_check_with_too_few_samples():
    assert stats.drift_check([1.0, 2.0, 3.0]) == {"checked": False, "reason": "fewer than 4 samples"}


def test_iqr_outliers_counts_spikes_but_never_removes_anything():
    data = np.concatenate([np.full(98, 1.0) + np.linspace(0, 0.01, 98), [10.0, 12.0]])
    out = stats.iqr_outliers(data)
    assert out["count_high"] == 2 and out["count_low"] == 0
    assert out["fraction"] == pytest.approx(0.02)


def test_iqr_outliers_on_clean_uniform_data():
    out = stats.iqr_outliers(np.linspace(1, 2, 1000))
    assert out["count_high"] == 0 and out["count_low"] == 0


@pytest.mark.parametrize("n, fragment", [(10, "rough"), (50, "essentially the maximum"), (500, "fewer than 10")])
def test_sample_size_warnings(n, fragment):
    notes = stats.sample_size_warnings(n)
    assert len(notes) == 1 and fragment in notes[0]


def test_sample_size_warnings_silent_for_enough_samples():
    assert stats.sample_size_warnings(1000) == []
