"""Capture the machine and software environment of a benchmark run.

A latency number without the hardware and software it was measured on is
close to meaningless, so every results file carries this block. It is also
used to warn when two runs are being compared that are not really comparable
(different CPU, different ONNX Runtime version, ...).
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata as _md
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from . import __version__

# Packages whose versions are recorded when they are installed.
_PACKAGES = (
    "numpy",
    "onnx",
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "onnxruntime-openvino",
    "Pillow",
    "psutil",
    "dash",
    "plotly",
)

# Environment fields that must match for two runs to be comparable.
_COMPARABLE_FIELDS = (
    ("cpu_model", "CPU model"),
    ("machine", "CPU architecture"),
    ("python_version", "Python version"),
)


def _read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _cpu_model() -> str:
    """Best-effort marketing name of the CPU, on Linux, macOS and Windows."""
    system = platform.system()
    if system == "Linux":
        text = _read_text("/proc/cpuinfo") or ""
        for key in ("model name", "Model", "Hardware", "cpu model"):
            for line in text.splitlines():
                if line.lower().startswith(key.lower()) and ":" in line:
                    return line.split(":", 1)[1].strip()
    elif system == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3, check=False,
            ).stdout.strip()
            if out:
                return out
        except (OSError, subprocess.SubprocessError):
            pass
    elif system == "Windows":
        try:
            import winreg  # type: ignore[import-not-found]

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
        except (ImportError, OSError):
            pass
    return platform.processor() or platform.machine() or "unknown"


def _memory_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1024**3, 2)
    except Exception:  # noqa: BLE001 - psutil is optional
        pass
    text = _read_text("/proc/meminfo") or ""
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            kb = int(line.split()[1])
            return round(kb / 1024**2, 2)
    return None


def _physical_cores() -> int | None:
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001
        return None


def _in_container() -> bool:
    if Path("/.dockerenv").exists():
        return True
    cgroup = _read_text("/proc/1/cgroup") or ""
    return any(word in cgroup for word in ("docker", "kubepods", "containerd"))


def _package_versions() -> dict[str, str]:
    found: dict[str, str] = {}
    for name in _PACKAGES:
        try:
            found[name] = _md.version(name)
        except _md.PackageNotFoundError:
            continue
    return found


def _ort_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        return []


def _cpu_busy_percent() -> float | None:
    """How busy the CPU is right now (a short 0.3 s sample)."""
    try:
        import psutil

        return float(psutil.cpu_percent(interval=0.3))
    except Exception:  # noqa: BLE001
        pass
    try:
        load1 = os.getloadavg()[0]
        return 100.0 * load1 / (os.cpu_count() or 1)
    except (AttributeError, OSError):
        return None


def _on_battery() -> bool | None:
    try:
        import psutil

        battery = psutil.sensors_battery()
        if battery is None:
            return None
        return not battery.power_plugged
    except Exception:  # noqa: BLE001
        return None


def capture_environment(*, measure_load: bool = True) -> dict[str, Any]:
    """Collect everything worth knowing about the machine and the software."""
    governor = _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    env: dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "harness_version": __version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": _physical_cores(),
        "memory_gb": _memory_gb(),
        "cpu_governor": governor,
        "in_container": _in_container(),
        "on_battery": _on_battery(),
        "cpu_busy_percent_at_start": _cpu_busy_percent() if measure_load else None,
        "packages": _package_versions(),
        "ort_available_providers": _ort_providers(),
    }
    return env


def noise_warnings(env: dict[str, Any]) -> list[str]:
    """Conditions on the machine that make timing results noisy or low."""
    notes: list[str] = []
    governor = env.get("cpu_governor")
    if governor and governor != "performance":
        notes.append(
            f"CPU frequency governor is '{governor}', not 'performance': clock speed can change "
            "during the run and make latency noisier."
        )
    if env.get("on_battery"):
        notes.append("The machine is running on battery: many laptops lower the CPU speed on battery.")
    busy = env.get("cpu_busy_percent_at_start")
    if isinstance(busy, (int, float)) and busy > 25:
        notes.append(
            f"The CPU was already {busy:.0f}% busy before the run started: close other programs "
            "for cleaner numbers."
        )
    if env.get("in_container"):
        notes.append(
            "Running inside a container: CPU limits and shared hosts add noise, and power "
            "counters are usually not visible."
        )
    return notes


def comparability_warnings(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Reasons why two environments should not be compared one-to-one."""
    notes: list[str] = []
    for key, label in _COMPARABLE_FIELDS:
        if a.get(key) != b.get(key):
            notes.append(f"{label} differs: '{a.get(key)}' vs '{b.get(key)}'.")
    pkgs_a, pkgs_b = a.get("packages", {}), b.get("packages", {})
    for pkg in sorted(set(pkgs_a) | set(pkgs_b)):
        if pkg in ("dash", "plotly", "psutil", "Pillow"):
            continue  # not performance relevant
        if pkgs_a.get(pkg) != pkgs_b.get(pkg):
            notes.append(f"Package '{pkg}' differs: {pkgs_a.get(pkg)} vs {pkgs_b.get(pkg)}.")
    return notes
