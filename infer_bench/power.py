"""Power and energy measurement.

Two sources are supported:

``rapl``
    Intel/AMD RAPL energy counters exposed by the Linux kernel under
    ``/sys/class/powercap``. They count the energy used by the CPU *package*
    (cores, caches, memory controller) - **not** the whole machine, not a
    GPU and not an accelerator card on PCIe. Reading them usually needs root
    (or ``chmod a+r`` on the ``energy_uj`` files) on current kernels.

``manual``
    A number the user supplies, for example from a wall power meter or a
    smart plug (``--power-watts``). That is the only honest way to get
    whole-system or accelerator power on a laptop.

Idle power can be measured (RAPL) or supplied (``--idle-watts``) and is
subtracted to give the *net* power of the workload.

The sysfs root is a constructor argument so the counter logic, including the
wrap-around of the 32/64-bit counter, can be tested against a fake directory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RAPL_ROOT = "/sys/class/powercap"

# Shown when the counters exist but cannot be read.
PERMISSION_HINT = (
    "RAPL counters exist but are not readable. On Linux run as root, or make them readable with: "
    "sudo chmod -R a+r /sys/class/powercap/intel-rapl*"
)


@dataclass
class Domain:
    name: str
    energy_path: Path
    max_range_uj: int


@dataclass
class PowerReading:
    """Energy and average power over one measured interval."""

    source: str
    scope: str
    duration_s: float
    energy_j: float
    avg_power_w: float
    domains_j: dict[str, float] = field(default_factory=dict)


def counter_delta(start: int, end: int, max_range: int) -> int:
    """Energy difference in microjoules, handling counter wrap-around.

    The kernel counter counts up to ``max_energy_range_uj`` and then starts
    again from zero. If the end value is smaller than the start value the
    counter wrapped exactly once (a second wrap would need hours at the
    power levels of a CPU).
    """
    delta = end - start
    if delta < 0:
        delta += max_range
    return delta


class RaplMeter:
    """Reads CPU package energy from the Linux powercap interface."""

    def __init__(self, root: str | Path = RAPL_ROOT):
        self.root = Path(root)
        self.domains: list[Domain] = []
        self.reason_unavailable: str | None = None
        self._discover()

    def _discover(self) -> None:
        if not self.root.exists():
            self.reason_unavailable = "no /sys/class/powercap (RAPL is Linux-only, on Intel or AMD CPUs)"
            return
        permission_problem = False
        for entry in sorted(self.root.glob("intel-rapl:*")):
            # "intel-rapl:0" is a package; "intel-rapl:0:0" is a sub-domain
            # (cores, uncore, dram) that the package total already includes.
            if entry.name.count(":") != 1:
                continue
            try:
                name = (entry / "name").read_text().strip()
                if not name.startswith("package"):
                    continue
                max_range = int((entry / "max_energy_range_uj").read_text().strip())
                int((entry / "energy_uj").read_text().strip())  # readability probe
            except PermissionError:
                permission_problem = True
                continue
            except (OSError, ValueError):
                continue
            self.domains.append(Domain(name, entry / "energy_uj", max_range))
        if not self.domains:
            self.reason_unavailable = PERMISSION_HINT if permission_problem else "no readable RAPL package domain found"

    @property
    def available(self) -> bool:
        return bool(self.domains)

    def read_uj(self) -> dict[str, int]:
        """Current counter value of every package domain (microjoules)."""
        return {d.name: int(d.energy_path.read_text().strip()) for d in self.domains}

    def measure(self) -> "_Measurement":
        """Context manager: ``with meter.measure() as m: ...; m.reading``."""
        return _Measurement(self)

    def reading_between(self, start: dict[str, int], end: dict[str, int], duration_s: float) -> PowerReading:
        per_domain: dict[str, float] = {}
        for d in self.domains:
            per_domain[d.name] = counter_delta(start[d.name], end[d.name], d.max_range_uj) / 1e6
        energy = sum(per_domain.values())
        avg = energy / duration_s if duration_s > 0 else 0.0
        return PowerReading("rapl", "cpu_package_only", duration_s, energy, avg, per_domain)

    def measure_idle(self, seconds: float) -> PowerReading:
        """Average power while this process does nothing."""
        with self.measure() as m:
            time.sleep(seconds)
        assert m.reading is not None
        return m.reading


class _Measurement:
    def __init__(self, meter: RaplMeter):
        self._meter = meter
        self.reading: PowerReading | None = None
        self._t0 = 0.0
        self._e0: dict[str, int] = {}

    def __enter__(self) -> "_Measurement":
        self._e0 = self._meter.read_uj()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        t1 = time.perf_counter()
        e1 = self._meter.read_uj()
        self.reading = self._meter.reading_between(self._e0, e1, t1 - self._t0)


def manual_reading(avg_power_w: float, duration_s: float) -> PowerReading:
    """Wrap a user-supplied average power (watts) as a reading."""
    return PowerReading("manual", "user_supplied", duration_s, avg_power_w * duration_s, avg_power_w)


def build_power_report(
    reading: PowerReading,
    *,
    items: int,
    idle_power_w: float | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Turn a reading into the ``power`` block of a results file.

    Efficiency is reported twice: against the total power and, when an idle
    baseline exists, against the net power (total minus idle). The net figure
    is closer to "what the workload costs", but only as good as the idle
    measurement, which is why both the raw and the net values are kept.
    """
    duration = reading.duration_s
    avg = reading.avg_power_w
    items_per_s = items / duration if duration > 0 else 0.0
    report: dict[str, Any] = {
        "source": reading.source,
        "scope": reading.scope,
        "duration_s": duration,
        "avg_power_w": avg,
        "energy_j": reading.energy_j,
        "energy_per_item_mj": (reading.energy_j / items * 1000.0) if items else None,
        "items_per_s_per_w": (items_per_s / avg) if avg > 0 else None,
        "idle_power_w": idle_power_w,
        "net_power_w": None,
        "net_energy_per_item_mj": None,
        "items_per_s_per_net_w": None,
        "domains_j": reading.domains_j,
        "notes": list(notes or []),
    }
    if idle_power_w is not None:
        net = avg - idle_power_w
        report["net_power_w"] = net
        if net > 0 and items:
            report["net_energy_per_item_mj"] = net * duration / items * 1000.0
            report["items_per_s_per_net_w"] = items_per_s / net
        else:
            report["notes"].append(
                "Average power is not above the idle baseline, so net power and energy are not reported."
            )
    if reading.source == "rapl":
        report["notes"].append(
            "RAPL covers the CPU package only: not the whole system, not a GPU, not an accelerator on PCIe."
        )
    return report
