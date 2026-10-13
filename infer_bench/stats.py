"""Statistics helpers for benchmark samples.

Everything here works on plain numbers (latencies in milliseconds) and has no
dependency on the rest of the package, so it is easy to test on its own.

Why these particular tools?

* Percentiles (p50/p90/p95/p99) because the mean hides tail latency, and the
  tail is what a real-time system has to survive.
* A confidence interval for the median, so that "A is faster than B" can be
  checked against measurement noise instead of being taken on faith.
* A drift check (first half vs second half), because thermal throttling or a
  background process makes a long run slower or faster over time and then a
  single summary number is misleading.
* IQR outlier counting, to show how spiky a run was without throwing samples
  away. Nothing is ever removed from the raw samples.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

PERCENTILES = (50, 90, 95, 99)


def as_array(samples: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("need at least one sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError("samples contain NaN or infinity")
    return arr


def percentile(samples: Sequence[float] | np.ndarray, q: float) -> float:
    """Percentile with linear interpolation (numpy's default definition)."""
    return float(np.percentile(as_array(samples), q))


def default_block_length(n: int) -> int:
    """Block length for the block bootstrap: roughly n**(1/3).

    This is the usual rule of thumb. It keeps blocks long enough to preserve
    short-range correlation between consecutive latencies, but short enough
    that there are many blocks to resample.
    """
    return max(1, int(round(n ** (1.0 / 3.0))))


def bootstrap_median_ci(
    samples: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    resamples: int = 1000,
    block: int | None = None,
    seed: int = 0,
) -> tuple[float, float]:
    """Confidence interval for the median using a moving-block bootstrap.

    Consecutive latency samples are not independent (a slow moment on the
    machine affects several runs in a row). A plain bootstrap would then give
    an interval that is too narrow. Resampling *blocks* of consecutive
    samples keeps that correlation inside each block. ``block=1`` is the
    ordinary (independent) bootstrap.

    The result is deterministic for a given ``seed``.
    """
    arr = as_array(samples)
    n = arr.size
    if n == 1:
        return float(arr[0]), float(arr[0])
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    length = default_block_length(n) if block is None else max(1, min(int(block), n))
    n_blocks = math.ceil(n / length)
    rng = np.random.default_rng(seed)
    offsets = np.arange(length)
    medians = np.empty(resamples, dtype=np.float64)

    # Process the resamples in chunks so the index array never gets huge.
    chunk = max(1, int(2_000_000 // n))
    done = 0
    while done < resamples:
        b = min(chunk, resamples - done)
        starts = rng.integers(0, n - length + 1, size=(b, n_blocks))
        idx = (starts[:, :, None] + offsets[None, None, :]).reshape(b, n_blocks * length)[:, :n]
        medians[done : done + b] = np.median(arr[idx], axis=1)
        done += b

    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.percentile(medians, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(lo), float(hi)


def drift_check(samples: Sequence[float] | np.ndarray, *, threshold: float = 0.05) -> dict[str, Any]:
    """Compare the median of the first half of the run with the second half.

    A stable run has nearly the same median in both halves. If the relative
    change is larger than ``threshold`` (default 5 %), the run drifted
    (thermal throttling, a background job, a warm-up that was too short...)
    and a single summary number should not be trusted.
    """
    arr = as_array(samples)
    n = arr.size
    if n < 4:
        return {"checked": False, "reason": "fewer than 4 samples"}
    half = n // 2
    first = float(np.median(arr[:half]))
    second = float(np.median(arr[n - half :]))
    if first > 0:
        rel = (second - first) / first
    else:
        rel = 0.0 if second == 0 else float("inf")
    return {
        "checked": True,
        "first_half_median": first,
        "second_half_median": second,
        "relative_change": rel,
        "threshold": threshold,
        "stable": bool(abs(rel) <= threshold),
    }


def iqr_outliers(samples: Sequence[float] | np.ndarray, *, k: float = 1.5) -> dict[str, Any]:
    """Count samples outside the Tukey fences (Q1 - k*IQR, Q3 + k*IQR).

    Outliers are counted and reported, never removed: for latency, the slow
    samples are exactly what the user wants to know about.
    """
    arr = as_array(samples)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = float(q3 - q1)
    lower = float(q1 - k * iqr)
    upper = float(q3 + k * iqr)
    high = int(np.sum(arr > upper))
    low = int(np.sum(arr < lower))
    return {
        "method": f"Tukey fences, k={k}",
        "lower_fence": lower,
        "upper_fence": upper,
        "count_high": high,
        "count_low": low,
        "fraction": (high + low) / arr.size,
    }


def summarize(
    samples_ms: Sequence[float] | np.ndarray,
    *,
    resamples: int = 1000,
    seed: int = 0,
    block: int | None = None,
    drift_threshold: float = 0.05,
) -> dict[str, Any]:
    """Full summary of a set of latency samples (milliseconds)."""
    arr = as_array(samples_ms)
    n = int(arr.size)
    block_used = default_block_length(n) if block is None else max(1, min(int(block), n))
    lo, hi = bootstrap_median_ci(arr, resamples=resamples, block=block_used, seed=seed)
    pct = {f"p{q}_ms": float(np.percentile(arr, q)) for q in PERCENTILES}
    return {
        "unit": "ms",
        "n": n,
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        **pct,
        "median_ci95_ms": [lo, hi],
        "ci_method": f"moving-block bootstrap, block={block_used}, resamples={resamples}, seed={seed}",
        "outliers": iqr_outliers(arr),
        "drift": drift_check(arr, threshold=drift_threshold),
    }


def sample_size_warnings(n: int) -> list[str]:
    """Plain-language warnings about how much to trust the tail percentiles."""
    notes: list[str] = []
    if n < 30:
        notes.append(f"Only {n} timed samples: all percentiles are rough. Use at least 100.")
    elif n < 100:
        notes.append(f"Only {n} timed samples: p99 is essentially the maximum. Use at least 1000 for p99.")
    elif n < 1000:
        notes.append(f"{n} timed samples: p99 rests on fewer than 10 samples. Use at least 1000 for a stable p99.")
    return notes
