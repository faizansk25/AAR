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


class TestDarwinMemoryFallback:
    """The no-psutil Darwin path, which is otherwise untestable on Linux.

    Two CI macOS jobs failed with PlanInfeasible on pipelines needing only
    ~40 MB. The cause was here: without psutil, ``probe_memory`` took
    ``Pages free`` from ``vm_stat`` as the available total. That counter
    excludes the file cache, so a healthy 7 GB Mac reports a few hundred MB
    free, and admitting against 80% of that refuses ordinary work.

    Adding psutil to ``[dev]`` would have turned CI green and left every real
    user on macOS without psutil still hitting it, so the stdlib path is
    fixed and tested directly against a recorded sample.
    """

    #: Recorded from a 7 GB Intel Mac, macOS 14, the shape GitHub's ARM
    #: runner produces as well. Verbatim, including the period after each
    #: count, which is why the parser strips it.
    SAMPLE = """\
Mach Virtual Memory Statistics: (page size of 4096 bytes)
Pages free:                               142896.
Pages active:                            396288.
Pages inactive:                          358912.
Pages speculative:                        18104.
Pages throttled:                              0.
Pages wired down:                        228544.
Pages purgeable:                          11520.
"Translation faults":                  88214560.
Pages copy-on-write:                    1184232.
Pages zero filled:                     18057344.
Pages reactivated:                       4421096.
Pages purged:                            1289416.
File-backed pages:                       301056.
Anonymous pages:                        302976.
Pages stored in compressor:              150528.
Pages occupied by compressor:            150528.
Decompressions:                          2150400.
Compressions:                            5146240.
Pageins:                                1189872.
Pageouts:                                 20864.
Swapins:                                      0.
Swapouts:                                     0.
"""

    def test_a_healthy_mac_reports_a_plausible_fraction_of_its_ram(self):
        available = detect.parse_darwin_vm_stat(self.SAMPLE)

        free_only = 142896 * 4096
        assert available is not None
        assert available > free_only * 3, (
            "available memory must include the reclaimable caches, not just "
            "untouched pages; this is the bug that failed two CI jobs")
        # 7 GB machine: the reclaimable pools should land in the high
        # hundreds of MB, not the tens.
        assert 200e6 < available < 7e9, available

    def test_it_sums_exactly_what_psutil_sums(self):
        """Parity with ``psutil/arch/osx/mem.c``, which computes::

            available = inactive + free;
            used      = active + wired;
            free     -= speculative;     // the *free field* only

        Speculative is subtracted from psutil's reported *free*, never from
        *available*, and purgeable is not counted at all. Matching that
        exactly is the requirement: ``probe_memory`` takes the psutil path
        when psutil is installed and this fallback when it is not, so a
        different formula would make ``pip install psutil`` a silent
        behaviour change.
        """
        free, inactive = 142896, 358912
        assert detect.parse_darwin_vm_stat(self.SAMPLE) == (
            free + inactive) * 4096

        # And the two differences from the previous version, stated
        # explicitly so a future edit has to argue with them.
        assert "Pages purgeable" not in detect._DARWIN_RECLAIMABLE, (
            "psutil does not include purgeable in available")
        assert "Pages speculative" not in detect._DARWIN_RECLAIMABLE, (
            "speculative is already inside Pages free, and psutil leaves it "
            "in available; only its separate `free` field subtracts it")

    def test_in_use_memory_is_never_counted_as_available(self):
        """The point of the whole exercise.

        Counting `active` would let the planner admit a workload that then
        gets OOM-killed, so this pins the exclusion rather than trusting it.
        """
        available = detect.parse_darwin_vm_stat(self.SAMPLE)
        for counter in ("Pages active", "Pages wired down",
                         "Pages occupied by compressor"):
            pages = int(
                [ln for ln in self.SAMPLE.splitlines()
                 if ln.startswith(counter)][0].split(":")[1].strip().rstrip("."))
            assert pages * 4096 < available, counter

    def test_unparseable_output_falls_back_rather_than_throwing(self):
        assert detect.parse_darwin_vm_stat("") is None
        assert detect.parse_darwin_vm_stat("garbage from a different OS") is None

    def test_the_page_size_is_read_from_the_header(self):
        scaled = self.SAMPLE.replace("page size of 4096 bytes",
                                     "page size of 16384 bytes")
        assert detect.parse_darwin_vm_stat(scaled) == (
            detect.parse_darwin_vm_stat(self.SAMPLE) * 4)

    def test_a_pipeline_needing_40mb_would_now_be_admitted(self):
        """The end-to-end symptom, as a regression test.

        Reproduces the shape of the CI failure without needing a Mac.

        The figure that matters is not the one in SAMPLE above. A 7 GB Mac
        with 142896 free pages still has ~585 MB free, which would admit a
        40 MB pipeline and hide the bug entirely. The failure needs a machine
        that is actually *busy* - which every CI runner is, since it is
        running a build while the tests run. There, ``Pages free`` sits in the
        low thousands of pages (tens of MB) and admitting against 80% of that
        refuses ordinary work.
        """
        from aar.hardware.detect import MemoryInfo

        # A loaded runner: same machine, caches long since allocated.
        loaded_free_pages = 6321
        broken = MemoryInfo(total_bytes=7_000_000_000,
                            available_bytes=loaded_free_pages * 4096)
        healthy = MemoryInfo(
            total_bytes=7_000_000_000,
            available_bytes=detect.parse_darwin_vm_stat(self.SAMPLE))

        peak = 40_000_000
        headroom = peak * 1.25  # the planner's admission multiplier
        assert headroom > broken.available_bytes * 0.8, (
            "precondition: the old figures really do reject a 40 MB "
            "pipeline, which is what CI reported")
        assert headroom <= healthy.available_bytes * 0.8, (
            "the corrected figures must admit it")

        # And the gap is far larger than a tuning difference could explain.
        ratio = healthy.available_bytes / broken.available_bytes
        assert ratio > 50, (
            f"the old figures understate a busy machine's reclaimable memory "
            f"by roughly {ratio:.0f}x, which is a wrong formula rather than "
            f"a margin to be tuned")


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


class TestGridSearchRecoversAKnownCurve:
    """The fitted curve must be the curve that was actually fitted.

    The third regressor was ``x * x`` for every candidate exponent while the
    resulting coefficient was stored as belonging to ``x ** exp`` - so the
    search returned a model it had never fit. Generating data from a known
    nonlinear curve is the only way to see it: at exp=2.0 the two
    expressions coincide, so a linear-dominated sample hides the bug.
    """

    def test_it_recovers_a_known_superlinear_curve(self):
        from aar.hardware.calibrate import _grid_search

        ns = [1e6, 2e6, 4e6, 8e6, 16e6]
        # A genuine n^1.3 term: fixed + linear + power law.
        truth = [1e-3 + 2e-9 * n + 5e-15 * (n ** 1.3) for n in ns]
        fixed, slope, power, exponent, r2 = _grid_search(ns, truth)
        assert r2 > 0.999, f"a perfect fit scored R2={r2:.4f}"
        assert 1.2 <= exponent <= 1.4, exponent

        # The returned coefficients must reproduce the data when used exactly
        # as ``CostCurve.predict`` uses them.
        for n, want in zip(ns, truth):
            got = fixed + slope * n + power * (n ** exponent)
            assert got == pytest.approx(want, rel=0.05), (n, got, want)

    def test_the_fitted_and_predicted_models_are_the_same_function(self):
        """Regression: the regressor must use the exponent under test."""
        from aar.hardware.calibrate import _grid_search

        ns = [1e6, 3e6, 9e6, 27e6]
        truth = [1e-3 + 1e-8 * n + 4e-14 * (n ** 1.2) for n in ns]
        fixed, slope, power, exponent, _ = _grid_search(ns, truth)
        assert exponent < 2.0, "expected a sub-quadratic exponent"
        for n, want in zip(ns, truth):
            got = fixed + slope * n + power * (n ** exponent)
            assert got == pytest.approx(want, rel=0.05)


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
        # A context manager, not `open(...).read()`: the bare form leaves the
        # handle to the garbage collector, and on Windows an unclosed handle
        # keeps the file locked, which turns this into a flaky failure the
        # moment anything else in the test touches `tmp_profile`.
        with open(tmp_profile, encoding="utf-8") as fh:
            payload = json.load(fh)
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

    def test_each_calibration_point_records_its_own_size(self, tmp_profile):
        """A point must report the size it was measured at, not the last one.

        The microbenchmark closures in `calibrate.py` are defined inside a
        loop and read a loop variable (`tbl`, `host`, `path`). They are
        correct only because `measure()` invokes each one before the loop
        advances. If a refactor ever collected the closures and called them
        afterwards, Python's late binding would make every point measure the
        *final* size - and the resulting cost curves would still look
        plausible, which is the worst way for this to fail.

        The invariant that catches it is simple: more than one distinct
        input size must appear in the points. That is why
        `src/aar/hardware/calibrate.py` is exempted from ruff's B023 rather
        than rewritten - this test is the guard, not the exclusion.

        Uses the CPU benchmark directly rather than a full `calibrate()` run
        so it stays fast; the full run is covered above.
        """
        pytest.importorskip("pyarrow")
        from aar.hardware.calibrate import _cpu_benchmarks

        points = [p for p in _cpu_benchmarks(sizes=(1_000_000, 8_000_000),
                                             repeats=1)
                  if p.rows > 0]
        if not points:
            pytest.skip("this machine could not allocate the test frames")

        sizes = {p.nbytes for p in points}
        assert len(sizes) > 1, (
            f"every calibration point used the same input size {sizes}; "
            "late binding in the benchmark closures would look exactly "
            "like this")

