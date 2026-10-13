import pytest

from infer_bench import power

from conftest import set_energy


def test_counter_delta_without_wraparound():
    assert power.counter_delta(1_000, 5_000, 100_000) == 4_000


def test_counter_delta_handles_wraparound():
    # The counter went from 99_000 up to its maximum (100_000), wrapped to 0 and reached 2_000.
    assert power.counter_delta(99_000, 2_000, 100_000) == 3_000


def test_meter_finds_only_the_package_domain(fake_rapl):
    meter = power.RaplMeter(fake_rapl)
    assert meter.available
    assert [d.name for d in meter.domains] == ["package-0"]  # not core, not dram, not the mmio duplicate
    assert meter.read_uj() == {"package-0": 1_000_000}


def test_meter_measures_energy_and_average_power(fake_rapl):
    meter = power.RaplMeter(fake_rapl)
    start = meter.read_uj()
    set_energy(fake_rapl, 1_000_000 + 25_000_000)  # +25 J
    reading = meter.reading_between(start, meter.read_uj(), duration_s=2.0)
    assert reading.energy_j == pytest.approx(25.0)
    assert reading.avg_power_w == pytest.approx(12.5)
    assert reading.source == "rapl" and reading.scope == "cpu_package_only"


def test_meter_handles_a_counter_that_wrapped_during_the_measurement(fake_rapl):
    (fake_rapl / "intel-rapl:0" / "max_energy_range_uj").write_text("10000000\n")
    set_energy(fake_rapl, 9_000_000)
    meter = power.RaplMeter(fake_rapl)
    start = meter.read_uj()
    set_energy(fake_rapl, 2_000_000)  # wrapped: 1 J up to the max + 2 J
    reading = meter.reading_between(start, meter.read_uj(), duration_s=1.0)
    assert reading.energy_j == pytest.approx(3.0)


def test_measure_context_manager_reads_before_and_after(fake_rapl):
    meter = power.RaplMeter(fake_rapl)
    with meter.measure() as m:
        set_energy(fake_rapl, 1_000_000 + 5_000_000)
    assert m.reading is not None
    assert m.reading.energy_j == pytest.approx(5.0)
    assert m.reading.duration_s > 0


def test_meter_unavailable_without_powercap(tmp_path):
    meter = power.RaplMeter(tmp_path / "does-not-exist")
    assert not meter.available
    assert "RAPL is Linux-only" in meter.reason_unavailable


def test_meter_reports_permission_problems(fake_rapl, monkeypatch):
    original = power.Path.read_text

    def deny(self, *args, **kwargs):
        if self.name == "energy_uj":
            raise PermissionError(13, "Permission denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(power.Path, "read_text", deny)
    meter = power.RaplMeter(fake_rapl)
    assert not meter.available
    assert "chmod" in meter.reason_unavailable


def test_power_report_efficiency_numbers():
    reading = power.PowerReading("rapl", "cpu_package_only", duration_s=10.0, energy_j=100.0, avg_power_w=10.0)
    report = power.build_power_report(reading, items=500)
    assert report["avg_power_w"] == 10.0
    assert report["items_per_s_per_w"] == pytest.approx(5.0)  # 50 items/s / 10 W
    assert report["energy_per_item_mj"] == pytest.approx(200.0)  # 100 J / 500 items
    assert report["net_power_w"] is None
    assert any("CPU package only" in n for n in report["notes"])


def test_power_report_subtracts_idle_power():
    reading = power.PowerReading("rapl", "cpu_package_only", duration_s=10.0, energy_j=100.0, avg_power_w=10.0)
    report = power.build_power_report(reading, items=500, idle_power_w=4.0)
    assert report["net_power_w"] == pytest.approx(6.0)
    assert report["items_per_s_per_net_w"] == pytest.approx(50.0 / 6.0)
    assert report["net_energy_per_item_mj"] == pytest.approx(6.0 * 10.0 / 500 * 1000)


def test_power_report_when_idle_is_not_below_average():
    reading = power.manual_reading(5.0, duration_s=2.0)
    report = power.build_power_report(reading, items=10, idle_power_w=6.0)
    assert report["net_power_w"] == pytest.approx(-1.0)
    assert report["items_per_s_per_net_w"] is None
    assert any("not above the idle baseline" in n for n in report["notes"])


def test_manual_reading_is_marked_as_user_supplied():
    reading = power.manual_reading(12.0, duration_s=3.0)
    assert reading.source == "manual" and reading.scope == "user_supplied"
    assert reading.energy_j == pytest.approx(36.0)
