"""Hardware profiling and microbenchmark calibration."""

from __future__ import annotations

import json

import pytest

from aar.hardware import bytes_from_human, detect, human_bytes
from aar.hardware.calibrate import (CalibrationPoint, CalibrationStore,
                                     CostCurve, calibrate, _fit)


class TestByteHelpers:
    @pytest.mark.parametrize("text,expected", [
        ("32 GB", 32_000_000_000),
        ("512 MB", 512_000_000),
        ("8 GiB", 8 * 2**30),
        ("1 TB", 10**12),
    ])
    def test_parsing(self, text, expected):
        assert bytes_from_human(text) == expected

    def test_unparseable_returns_none(self):
        assert bytes_from_human("lots") is None

    def test_human_bytes_renders(self):
        assert human_bytes(1_500_000_000) == "1.5GB"


class TestProbes:
    """Every probe must return a usable object, never raise."""

    def test_os_probe(self):
        assert detect.probe_os().system

    def test_cpu_probe(self):
        cpu = detect.probe_cpu()
        assert cpu.logical_cores >= 1
        assert cpu.physical_cores >= 1
        assert cpu.simd_level in ("scalar", "avx", "avx2", "avx512")
        assert cpu.render()

    def test_memory_probe(self):
        mem = detect.probe_memory()
        assert mem.total_bytes > 0
        assert 0 < mem.available_bytes <= mem.total_bytes

    def test_gpu_probe_always_explains_itself(self):
        gpu = detect.probe_gpu()
        # Whether or not a GPU exists, we must be able to say why.
        assert gpu.reason
        assert gpu.render()

    def test_storage_probe(self):
        assert detect.probe_storage().media_type

    def test_network_probe_reports_deny_egress(self):
        net = detect.probe_network()
        assert net.egress_default == "deny"
        assert net.bandwidth_bytes_s and net.bandwidth_bytes_s > 0

    def test_software_probe(self):
        sw = detect.probe_software()
        assert sw.python
        assert sw.render()


class TestHardwareProfileFacade:
    def test_profile_is_memoised(self):
        p = detect.HardwareProfile()
        first = p.cpu
        assert p.cpu is first  # identical object => not re-probed

    def test_budgets_are_sane(self):
        p = detect.HardwareProfile()
        assert p.memory_budget_bytes > 0
        assert p.vram_budget_bytes >= 0
        assert p.memory_budget_bytes <= p.memory.available_bytes

    def test_fingerprint_is_stable(self):
        a = detect.HardwareProfile().fingerprint()
        b = detect.HardwareProfile().fingerprint()
        assert a == b
        assert len(a) == 16

    def test_no_gpu_means_zero_vram_budget(self, monkeypatch):
        p = detect.HardwareProfile()
        if not p.has_gpu:
            assert p.vram_budget_bytes == 0

    def test_render_lists_every_subsystem(self):
        text = detect.HardwareProfile().render()
        for label in ("os", "cpu", "memory", "gpu", "storage", "network",
                      "software", "budgets"):
            assert label in text



class TestCostCurveFitting:
    def _points(self, op, times, sizes=(1e6, 1e7, 1e8)):
        return [CalibrationPoint(op, "cpu", int(s), t) for s, t in zip(sizes, times)]

    def test_linear_fit_recovers_slope(self):
        pts = self._points("x", [0.011, 0.02, 0.11])
        curve = _fit(pts, "x", "cpu")
        assert curve.slope == pytest.approx(1e-9, rel=1e-3)
        assert curve.predict(1e8) == pytest.approx(0.11, rel=0.1)

    def test_monotonicity_is_never_violated(self):
        """A noisy measurement must not produce negative cost."""
        curve = _fit(self._points("noisy", [0.5, 0.2, 0.3]), "noisy", "cpu")
        assert curve.slope >= 0
        assert curve.power >= 0
        for n in (0, 1e6, 1e9, 1e12):
            assert curve.predict(n) >= 0

    def test_predict_never_negative_even_for_bad_curve(self):
        bad = CostCurve("op", "cpu", fixed=0.0, slope=-1e-9, exponent=1.0)
        assert bad.predict(1e12) >= 0

    def test_single_point_is_low_confidence(self):
        assert _fit([CalibrationPoint("x", "cpu", 1e6, 0.1)], "x", "cpu").low_confidence

    def test_empty_fit_is_low_confidence(self):
        assert _fit([], "x", "cpu").low_confidence

    def test_non_monotonic_input_is_conservative(self):
        curve = _fit(self._points("bad", [1.0, 0.1]), "bad", "cpu")
        assert curve.low_confidence
        assert curve.predict(1e8) >= 1.0


class TestCalibrationStore:
    def test_round_trip(self, tmp_profile):
        store = CalibrationStore(tmp_profile)
        store.fingerprint = "fp1"
        store.add_points([
            CalibrationPoint("groupby", "cpu", 1_000_000, 0.05),
            CalibrationPoint("groupby", "cpu", 10_000_000, 0.5),
        ])
        store.save()
        back = CalibrationStore.load(tmp_profile, fingerprint="fp1")
        assert back.get("groupby", "cpu") is not None
        assert back.get("groupby", "cpu").predict(1e6) == pytest.approx(0.05, rel=0.5)

    def test_profile_from_other_machine_is_discarded(self, tmp_profile):
        store = CalibrationStore(tmp_profile)
        store.fingerprint = "machine-A"
        store.add_points([CalibrationPoint("groupby", "cpu", 1e6, 0.05)])
        store.save()
        other = CalibrationStore.load(tmp_profile, fingerprint="machine-B")
        assert other.get("groupby", "cpu") is None
        assert other.meta.get("discarded_stale_profile")

    def test_missing_file_yields_empty_store(self, tmp_path):
        store = CalibrationStore.load(str(tmp_path / "nope.json"))
        assert not store.is_calibrated
        assert "No calibration" in store.render()

    def test_corrupt_file_does_not_crash(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        assert not CalibrationStore.load(str(path)).is_calibrated

    def test_written_file_is_valid_json(self, tmp_profile):
        store = CalibrationStore(tmp_profile)
        store.add_points([CalibrationPoint("filter", "cpu", 1e6, 0.01)])
        store.save()
        payload = json.loads(open(tmp_profile, encoding="utf-8").read())
        assert "curves" in payload and "version" in payload

    def test_best_device_falls_back_to_cpu_without_gpu(self, tmp_profile):
        store = CalibrationStore(tmp_profile)
        store.add_points([CalibrationPoint("groupby", "cpu", 1e6, 0.1)])
        assert store.best_device("groupby", 1e6)[0] == "cpu"


class TestLiveCalibration:
    @pytest.mark.slow
    def test_quick_calibration_measures_real_operations(self, tmp_profile):
        """The whole point: real measurements on this machine, in seconds."""
        store = calibrate(quick=True, include_gpu=False, include_disk=False,
                          path=tmp_profile, fingerprint="live")
        assert store.is_calibrated
        for op in ("scan", "filter", "sort", "groupby"):
            curve = store.get(op, "cpu")
            assert curve is not None, f"{op} was not calibrated"
            assert len(curve.points) >= 1
            assert curve.predict(1e6) >= 0
        assert CalibrationStore.load(tmp_profile, "live").is_calibrated

