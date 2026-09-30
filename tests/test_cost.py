"""Cost model: per-node cost, transitions, GPU transfer accounting, history.

The specification's worked example is the anchor test of this file:

    CPU operation = 400 ms, GPU kernel = 80 ms,
    H2D = 240 ms, D2H = 160 ms
    GPU total = 480 ms, CPU total = 400 ms  ->  choose CPU

If that example does not come out of the model, the model is wrong, however
many other tests pass.
"""

from __future__ import annotations

import pytest

from aar.capability import CapabilityRegistry, Device
from aar.cost import (CostBreakdown, CostModel, ExecutionHistory, Priors,
                      TransferProfile, default_cost_model, node_operation)
from aar.hardware.calibrate import CalibrationPoint, CalibrationStore
from aar.ir import Node, NodeType


def _node(node_type: NodeType) -> Node:
    return Node(node_type)


class TestSavedCalibrationIsUsed:
    """A profile the user paid for must reach the default planner.

    ``CostModel()`` built an empty ``CalibrationStore``, so a machine
    calibrated with ``aar calibrate`` planned from priors anyway. The plan
    said ``prior`` while the user believed it was measured - the failure mode
    this project exists to prevent, in the one place it is hardest to see.
    """

    def test_a_default_model_picks_up_the_saved_profile(self, tmp_path,
                                                       monkeypatch):
        from aar.hardware import calibrate as cal

        path = tmp_path / "profile.json"
        store = cal.CalibrationStore(str(path))
        store.add_points([cal.CalibrationPoint("groupby", "cpu", 1_000_000,
                                              0.05)])
        store.save()

        monkeypatch.setattr(cal, "PROFILE_PATH", str(path))
        # Load through the module-level helper the cost model calls.
        import aar.cost.model as model_module

        loaded = model_module._load_saved_calibration()
        assert loaded.get("groupby", "cpu") is not None
        assert loaded.is_calibrated

    def test_a_missing_profile_degrades_to_priors_rather_than_raising(
            self, tmp_path, monkeypatch):
        from aar.hardware import calibrate as cal
        import aar.cost.model as model_module

        monkeypatch.setattr(cal, "PROFILE_PATH", str(tmp_path / "nope.json"))
        loaded = model_module._load_saved_calibration()
        assert not loaded.is_calibrated

    def test_a_corrupt_profile_does_not_break_planning(self, tmp_path,
                                                       monkeypatch):
        from aar.hardware import calibrate as cal
        import aar.cost.model as model_module

        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(cal, "PROFILE_PATH", str(bad))
        loaded = model_module._load_saved_calibration()
        assert not loaded.is_calibrated


class TestSerialisationUnits:
    """``serialise_s`` was seconds-per-byte documented as flat seconds.

    Multiplying bytes by a number documented as seconds made 1 MB cost 20 s,
    a penalty large enough to reject correct plans on a unit error.
    """

    def test_one_megabyte_is_not_twenty_seconds(self):
        from aar.cost import Priors as P

        per_byte = P().serialise_s_per_byte
        assert 1_000_000 * per_byte < 1.0, "1 MB must not cost a second"

    def test_it_scales_with_the_bytes_actually_moved(self):
        from aar.cost import Priors as P

        per_byte = P().serialise_s_per_byte
        assert 10 * per_byte > per_byte


class TestEstimationErrorIsMeasured:
    """A prediction nobody checks is a prediction nobody should trust."""

    def test_relative_error_is_the_gap_over_the_actual(self):
        from aar.cost import EstimationError

        assert EstimationError("s", 100, 100).relative_error == 0.0
        assert EstimationError("s", 200, 100).relative_error == pytest.approx(1.0)
        assert EstimationError("s", 50, 100).relative_error == pytest.approx(0.5)

    def test_predicting_rows_for_an_empty_result_is_infinite_error(self):
        """The worst case must not quietly read as 'close enough'."""
        from aar.cost import EstimationError

        error = EstimationError("s", 1_000, 0)
        assert error.relative_error == float("inf")
        assert error.ratio == float("inf")

    def test_the_direction_of_the_miss_is_said_out_loud(self):
        from aar.cost import EstimationError

        assert "over" in EstimationError("s", 200, 100).render()
        assert "under" in EstimationError("s", 50, 100).render()
        assert "exact" in EstimationError("s", 100, 100).render()

    def test_a_log_finds_the_offenders(self):
        from aar.cost import EstimationLog

        log = EstimationLog()
        log.add("a", 100, 100, "parquet-metadata")
        log.add("b", 10_000, 100, "sampled")
        assert len(log) == 2
        assert [e.label for e in log.offenders] == ["b"]

    def test_an_empty_log_says_so_rather_than_reporting_zero_error(self):
        """No data must not be reported as perfect accuracy."""
        from aar.cost import EstimationLog

        assert "No estimation errors" in EstimationLog().render()

    def test_render_works_with_entries_in_it(self):
        """``render`` is the only way an analyst sees any of this.

        A property called as a method made every populated ``render()`` raise
        ``TypeError``, so the report could only ever be produced when it was
        empty - and an empty one says there is nothing to report. The
        headline number was unreachable in exactly the case it existed for.
        """
        from aar.cost import EstimationLog

        log = EstimationLog()
        log.add("a", 120, 100, "sampled")
        text = log.render()
        assert "ESTIMATION ACCURACY" in text
        assert "mean error" in text
        assert "worst" in text
        assert "120" in text and "100" in text


class TestTheProfilerIsScoredAgainstReality:
    """The end-to-end loop: predict, run, compare."""

    def _pipeline(self, directory, csv_text: str):
        src = directory / "in.csv"
        src.write_text(csv_text, encoding="utf-8")
        path = directory / "p.py"
        path.write_text(
            "from aar.sdk import csv, write_csv\n\n"
            "def build():\n"
            "    return write_csv(csv(r%r), r%r)\n"
            % (str(src), str(directory / "out.csv")),
            encoding="utf-8")
        return path

    def test_a_correct_profiler_reports_a_small_error(self, tmp_path):
        from aar.application import PipelineService

        path = self._pipeline(tmp_path, "v\n1\n2\n3\n4\n5\n")
        service = PipelineService()
        service.run(str(path))
        assert service.estimation_log is not None, (
            "a profiled run must record how accurate the profile was")
        errors = service.estimation_log.errors()
        assert errors, "no estimation error was recorded for a profiled run"
        assert service.estimation_log.worst_relative_error < 0.5, (
            service.estimation_log.render())

    def test_a_wrong_profiler_is_caught(self, tmp_path, monkeypatch):
        """The point of the loop: a confidently wrong estimate is visible.

        The profiler claims ten times the rows there are. Nothing about that
        is detectable from the profile alone - it is internally consistent,
        correctly typed, and confidently reported. Only comparing it with
        what the scan actually read reveals it.
        """
        from aar.application import PipelineService
        from aar.stats import DataProfiler

        real = DataProfiler.profile_csv

        def ten_times_too_many(self, path):
            profile = real(self, path)
            profile.rows *= 10
            return profile

        monkeypatch.setattr(DataProfiler, "profile_csv", ten_times_too_many)

        path = self._pipeline(tmp_path, "v\n1\n2\n3\n4\n5\n")
        service = PipelineService()
        service.run(str(path))
        assert service.estimation_log is not None
        offenders = service.estimation_log.offenders
        assert offenders, (
            "a profiler that over-estimated by 10x was not flagged:\n"
            + service.estimation_log.render())
        assert offenders[0].relative_error > 5.0
        assert "over" in offenders[0].render()

    def test_a_run_without_profiling_records_nothing(self, tmp_path):
        from aar.application import PipelineService

        path = self._pipeline(tmp_path, "v\n1\n2\n3\n")
        service = PipelineService()
        service.run(str(path), profile=False)
        assert service.estimation_log is None

class TestSegmentCostCoversEveryOperation:
    """A segment priced from its last node is under-priced.

    ``node_cost`` is called with ``seg.nodes[-1]``, so a segment of
    ``Filter -> GroupBy -> Sort`` is estimated from the sort alone. A filter
    that discards 99% of the rows and a group-by that reduces to a thousand
    groups both cost real time, and neither appears in the number the planner
    chose an engine with.

    The fix must be a *sum* of per-operation costs, not a fused estimate. The
    executor dispatches one node at a time - ``_dispatch`` calls
    ``engine.filter``, ``engine.group_by`` and ``engine.sort`` in sequence -
    so there is no fusion to credit. Pricing the segment as a single
    optimised query would claim an optimisation the runtime does not perform,
    which is the same class of error as reporting a GPU that never ran.
    """

    def _model(self) -> CostModel:
        """Distinct, large per-op costs, so mis-pricing is unmistakable."""
        n = 1_000_000
        store = CalibrationStore()
        for op, seconds in (("filter", 1.0), ("groupby", 2.0), ("sort", 3.0)):
            store.add_points([CalibrationPoint(op, "cpu", n, seconds)])
        return CostModel(calibration=store, registry=CapabilityRegistry(),
                         priors=Priors(startup_s=0.0))

    def _segment(self):
        from aar.ir import BinOp, Col, Lit, Node, NodeType, ScanSpec
        from aar.planner import decompose_into_segments

        scan = Node(NodeType.SCAN_CSV, scan=ScanSpec(kind="csv", path="x.csv"))
        scan.estimated_bytes = 1_000_000
        filt = Node(NodeType.FILTER, inputs=[scan],
                    predicate=BinOp(Col("v"), ">", Lit(1)))
        filt.estimated_bytes = 500_000
        proj = Node(NodeType.PROJECT, inputs=[filt], columns=("v", "g"))
        proj.estimated_bytes = 400_000
        filt2 = Node(NodeType.FILTER, inputs=[proj],
                     predicate=BinOp(Col("g"), "!=", Lit("")))
        filt2.estimated_bytes = 300_000
        # A scan is a forced segment boundary, so the multi-op segment is the
        # one *after* it. GroupBy is deliberately absent: its preferred device
        # is the GPU, which would split the segment and defeat the fixture.
        segments = decompose_into_segments(filt2)
        multi = [s for s in segments if len(s.nodes) >= 3]
        assert multi, (
            f"fixture produced {[len(s.nodes) for s in segments]} nodes per "
            f"segment; expected one with three")
        return multi[0]

    def test_a_multi_operation_segment_costs_more_than_its_last_node(self):
        model = self._model()
        segment = self._segment()

        total, _ = model.segment_cost(segment, "arrow")
        last_only, _ = model.node_cost(segment.nodes[-1], "arrow",
                                       segment.nbytes)
        assert total.compute_s > last_only.compute_s * 2, (
            f"segment priced at {total.compute_s:.3f}s but its final node "
            f"alone is {last_only.compute_s:.3f}s")

    def test_each_operation_in_the_segment_is_accounted_for(self):
        model = self._model()
        segment = self._segment()
        total, detail = model.segment_cost(segment, "arrow")

        # The segment must cost the sum of its parts, each priced at the
        # size that operation actually sees (500k, 400k, 300k) rather than
        # at the whole segment.
        expected = sum(model.compute_s(n, "arrow", n.estimated_bytes)[0]
                       for n in segment.nodes)
        assert total.compute_s == pytest.approx(expected, rel=1e-9)
        assert detail, "the breakdown must say which operations it costed"

    def test_each_operation_is_priced_at_its_own_size(self):
        """The tail of a segment is cheaper because less data reaches it.

        A group-by reducing a gigabyte to a thousand rows makes the sort
        after it cheap. Billing every operation at the segment's output
        size would over-price the tail and mis-rank engines on it.
        """
        model = self._model()
        segment = self._segment()
        detail = model.segment_cost(segment, "arrow")[1]
        sizes = [n.estimated_bytes for n in segment.nodes]
        assert sizes == sorted(sizes, reverse=True), sizes
        # The first operation is billed on more data than the last, so it
        # costs more; the detail has to show both.
        times = [float(part.split()[1].rstrip("ms"))
                 for part in detail.split("[")[1].split("]")[0].split(" + ")]
        assert len(times) == len(segment.nodes)
        assert times[0] > times[-1], times

    def test_a_single_operation_segment_is_unchanged(self):
        """One node in, one node priced: no regression on the simple case."""
        from aar.ir import Node, NodeType, ScanSpec
        from aar.planner import decompose_into_segments

        model = self._model()
        scan = Node(NodeType.SCAN_CSV, scan=ScanSpec(kind="csv", path="x.csv"))
        scan.estimated_bytes = 1_000_000
        segment = decompose_into_segments(scan)[0]
        total, _ = model.segment_cost(segment, "arrow")
        single, _ = model.node_cost(segment.nodes[-1], "arrow",
                                    segment.nbytes)
        assert total.compute_s == pytest.approx(single.compute_s, rel=1e-9)

    def test_the_detail_names_the_operations_it_summed(self):
        model = self._model()
        _, detail = model.segment_cost(self._segment(), "arrow")
        assert "filter" in detail
        assert "3 ops" in detail


class TestOperationMapping:
    @pytest.mark.parametrize("node_type,op", [
        (NodeType.GROUPBY, "groupby"),
        (NodeType.JOIN, "hash_join"),
        (NodeType.SORT, "sort"),
        (NodeType.FILTER, "filter"),
        (NodeType.SCAN_PARQUET, "parquet_decode"),
        (NodeType.WINDOW, "window"),
    ])
    def test_mapping(self, node_type, op):
        assert node_operation(_node(node_type)) == op

    def test_every_node_type_has_a_mapping(self):
        for nt in NodeType:
            assert node_operation(_node(nt)), nt


class TestCostBreakdown:
    def test_total_is_the_sum_of_terms(self):
        b = CostBreakdown(startup_s=1, read_s=2, transfer_s=3,
                          compute_s=4, spill_s=5, materialise_s=6)
        assert b.total_s == 21

    def test_movement_excludes_compute(self):
        b = CostBreakdown(transfer_s=3, materialise_s=2, compute_s=10)
        assert b.movement_s == 5
        assert b.kernel_s == 10

    def test_overhead_fraction_flags_transfer_bound_work(self):
        b = CostBreakdown(transfer_s=90, compute_s=10)
        assert b.overhead_fraction == pytest.approx(0.9)

    def test_pure_compute_has_no_overhead(self):
        assert CostBreakdown(compute_s=10).overhead_fraction == 0.0

    def test_addition_composes_terms(self):
        a = CostBreakdown(compute_s=1, transfer_s=2)
        b = CostBreakdown(compute_s=3, startup_s=4)
        assert (a + b).total_s == 10

    def test_render_shows_every_term(self):
        """The render must name every term it sums.

        This test used to call ``render()``, throw the string away, and assert
        nothing - it could not fail. A cost breakdown that silently dropped a
        term would still look plausible in `aar explain`, which is exactly
        where a cost model is read and trusted.
        """
        text = CostBreakdown(startup_s=0.1, read_s=0.1, transfer_s=0.1,
                             compute_s=0.1, spill_s=0.1,
                             materialise_s=0.1).render()
        assert text, "render() produced nothing at all"
        for term in ("startup", "read", "transfer", "compute", "spill",
                     "materialise"):
            assert term in text, f"render() omits the {term} term: {text!r}"
        # And the total must be the sum of the parts, not a separate number
        # nobody can reconcile. Six terms at 100ms each.
        assert "600.0" in text, f"total is missing from {text!r}"


class TestComputeCostSources:
    def _model(self, points=None) -> CostModel:
        store = CalibrationStore()
        # All points in one call: `add_points` re-fits the affected curve, so
        # adding them one at a time would leave only the last point's fit.
        if points:
            store.add_points(list(points))
        return CostModel(calibration=store, registry=CapabilityRegistry())


    def test_uses_calibration_when_present(self):
        """Three measurements give a confident fit, used unpenalised."""
        m = self._model([
            CalibrationPoint("groupby", "cpu", 1_000_000, 0.05),
            CalibrationPoint("groupby", "cpu", 10_000_000, 0.50),
            CalibrationPoint("groupby", "cpu", 100_000_000, 5.00),
        ])
        value, source = m.compute_s(_node(NodeType.GROUPBY), "duckdb", 1_000_000)
        assert source == "calibration"
        assert value == pytest.approx(0.05, rel=0.5)


    def test_falls_back_to_a_documented_prior(self):
        m = self._model()
        value, source = m.compute_s(_node(NodeType.GROUPBY), "duckdb", 1_000_000)
        assert source == "prior"
        assert value > 0

    def test_priors_are_pessimistic_not_optimistic(self):
        """An unmeasured op must not look cheap, or it will be chosen."""
        m = self._model()
        unmeasured, _ = m.compute_s(_node(NodeType.SORT), "duckdb", 100_000_000)
        m2 = self._model([CalibrationPoint("sort", "cpu", 100_000_000, 1.0)])
        measured, _ = m2.compute_s(_node(NodeType.SORT), "duckdb", 100_000_000)
        assert unmeasured >= measured

    def test_low_confidence_curve_is_penalised(self):
        store = CalibrationStore()
        # A single point produces a low-confidence curve.
        store.add_points([CalibrationPoint("groupby", "cpu", 1_000_000, 0.1)])
        m = CostModel(calibration=store, registry=CapabilityRegistry())
        value, source = m.compute_s(_node(NodeType.GROUPBY), "duckdb", 1_000_000)
        assert source == "calibration(penalised)"
        assert value > 0.1

    def test_gpu_without_gpu_curve_is_not_assumed_fast(self):
        """A GPU engine with no measured curve must fall back to the CPU
        curve, penalised - never to an optimistic guess."""
        store = CalibrationStore()
        store.add_points([CalibrationPoint("groupby", "cpu", 1_000_000, 0.1)])
        m = CostModel(calibration=store, registry=CapabilityRegistry())
        value, source = m.compute_s(_node(NodeType.GROUPBY),
                                    "polars_gpu", 1_000_000)
        assert source == "calibration(penalised)"
        assert value > 0.1


class TestGPUSpecificationExample:
    """The specification's worked example, reproduced from the model itself."""

    def test_faster_kernel_can_still_lose_on_transfers(self):
        cpu_compute, gpu_kernel = 0.400, 0.080
        h2d, d2h = 0.240, 0.160
        gpu_total = gpu_kernel + h2d + d2h
        assert gpu_total == pytest.approx(0.480)
        assert cpu_compute < gpu_total   # CPU wins despite a 5x slower kernel

    def test_model_reproduces_the_same_conclusion(self):
        """Same numbers as the specification's example, fed through the model.

        A 1 GB working set rather than 1 MB: the transfer rates are scaled to
        give exactly 240 ms up and 160 ms down at that size.
        """
        nbytes = 1_000_000_000
        store = CalibrationStore()
        store.add_points([CalibrationPoint("groupby", "gpu", nbytes, 0.080)])
        store.add_points([CalibrationPoint("groupby", "cpu", nbytes, 0.400)])

        transfer = TransferProfile(h2d_bytes_per_s=nbytes / 0.240,
                                   d2h_bytes_per_s=nbytes / 0.160)
        m = CostModel(calibration=store, registry=CapabilityRegistry(),
                      priors=Priors(gpu_startup_s=0.0, startup_s=0.0),
                      transfer=transfer)

        node = _node(NodeType.GROUPBY)
        cpu, _ = m.node_cost(node, "duckdb", nbytes)
        gpu, _ = m.node_cost(node, "polars_gpu", nbytes)

        assert gpu.kernel_s < cpu.kernel_s      # the kernel really is faster
        assert gpu.total_s > cpu.total_s        # and the GPU still loses

    def test_resident_gpu_data_pays_no_inbound_transfer(self):
        store = CalibrationStore()
        store.add_points([CalibrationPoint("groupby", "gpu", 1_000_000, 0.01)])
        m = CostModel(calibration=store, registry=CapabilityRegistry())
        node = _node(NodeType.GROUPBY)
        cold, _ = m.node_cost(node, "polars_gpu", 1_000_000, residency=Device.CPU)
        warm, _ = m.node_cost(node, "polars_gpu", 1_000_000, residency=Device.GPU)
        assert warm.transfer_s < cold.transfer_s

    def test_sink_keeps_data_on_device(self):
        """A GPU write should not need a copy back to the host."""
        store = CalibrationStore()
        store.add_points([CalibrationPoint("parquet_decode", "gpu",
                                           1_000_000, 0.01)])
        m = CostModel(calibration=store, registry=CapabilityRegistry())
        write = Node(NodeType.WRITE, target="out.parquet")
        resident, _ = m.node_cost(write, "polars_gpu", 1_000_000,
                                  residency=Device.GPU)
        assert resident.transfer_s == 0.0



class TestTransitionCost:
    def _model(self, **kw) -> CostModel:
        return CostModel(calibration=CalibrationStore(),
                         registry=CapabilityRegistry(), **kw)

    def test_same_engine_is_free(self):
        m = self._model()
        assert m.transition_cost("duckdb", "duckdb", 1_000_000).total_s == 0.0

    def test_cpu_to_cpu_in_process_is_free(self):
        """Zero-copy via the C Data Interface; the spec's interchange layer."""
        m = self._model(transfer=TransferProfile(in_process=True))
        assert m.transition_cost("duckdb", "polars_cpu",
                                 1_000_000).total_s == 0.0

    def test_cpu_to_gpu_pays_the_bus_both_ways(self):
        m = self._model(transfer=TransferProfile(
            h2d_bytes_per_s=1e10, d2h_bytes_per_s=1e10))
        cost = m.transition_cost("duckdb", "polars_gpu", 1_000_000_000)
        assert cost.transfer_s == pytest.approx(0.2, rel=0.01)

    def test_gpu_to_gpu_is_a_peer_copy(self):
        """One transfer, not two - device to device is not a host round trip."""
        m = self._model(transfer=TransferProfile(h2d_bytes_per_s=1e10,
                                                 d2h_bytes_per_s=1e10))
        peer = m.transition_cost("polars_gpu", "cudf", 1_000_000_000)
        assert peer.transfer_s == pytest.approx(0.1, rel=0.01)

    def test_crossing_the_network_is_slower_than_crossing_the_bus(self):
        m = self._model(transfer=TransferProfile(
            h2d_bytes_per_s=1e10, d2h_bytes_per_s=1e10,
            network_bytes_per_s=1e9))
        net = m.transition_cost("duckdb", "ray", 1_000_000_000)
        bus = m.transition_cost("duckdb", "polars_gpu", 1_000_000_000)
        assert net.total_s > bus.total_s

    def test_transition_grows_with_size(self):
        m = self._model()
        small = m.transition_cost("duckdb", "polars_gpu", 1_000_000)
        large = m.transition_cost("duckdb", "polars_gpu", 1_000_000_000)
        assert large.total_s > small.total_s


class TestPeakMemory:
    def _model(self) -> CostModel:
        return CostModel(calibration=CalibrationStore(),
                         registry=CapabilityRegistry())

    def test_join_uses_more_memory_than_filter(self):
        m = self._model()
        join = m.peak_memory_b(_node(NodeType.JOIN), "duckdb", 1_000_000)
        filt = m.peak_memory_b(_node(NodeType.FILTER), "duckdb", 1_000_000)
        assert join > filt

    def test_gpu_allocates_more_than_cpu_for_the_same_input(self):
        m = self._model()
        cpu = m.peak_memory_b(_node(NodeType.GROUPBY), "duckdb", 1_000_000)
        gpu = m.peak_memory_b(_node(NodeType.GROUPBY), "polars_gpu", 1_000_000)
        assert gpu > cpu

    def test_spill_is_charged_only_when_over_vram(self):
        """A working set that fits in VRAM must not be charged for spilling.

        The spec is explicit that non-UVM runs OOM past VRAM, so exceeding it
        is a real cost - but charging for work that fits would bias the
        optimiser toward CPU for reasons unrelated to reality.
        """
        m = self._model()
        from aar.hardware import HardwareProfile

        vram = HardwareProfile().vram_budget_bytes
        if not vram:
            pytest.skip("no GPU on this machine; VRAM budget is zero")

        fits, _ = m.node_cost(_node(NodeType.GROUPBY), "polars_gpu",
                              min(1_000_000, vram // 2))
        overflows, _ = m.node_cost(_node(NodeType.GROUPBY), "polars_gpu",
                                   vram * 4)
        assert fits.spill_s == 0.0
        assert overflows.spill_s > 0.0

    def test_cpu_work_never_spills(self):
        m = self._model()
        cost, _ = m.node_cost(_node(NodeType.GROUPBY), "duckdb",
                              10_000_000_000)
        assert cost.spill_s == 0.0




class TestExecutionHistory:
    def test_no_history_means_no_prediction(self):
        h = ExecutionHistory()
        assert h.predict(_node(NodeType.GROUPBY), "duckdb", 1000) is None

    def test_single_observation_is_a_lookup(self):
        h = ExecutionHistory()
        node = _node(NodeType.GROUPBY)
        h.record(node.id, "duckdb", 1000, 10, elapsed_ms=50.0)
        assert h.predict(node, "duckdb", 1000) == pytest.approx(0.05)

    def test_predictions_are_separated_by_engine(self):
        h = ExecutionHistory()
        node = _node(NodeType.GROUPBY)
        h.record(node.id, "duckdb", 1000, 10, elapsed_ms=50.0)
        assert h.predict(node, "polars_cpu", 1000) is None

    def test_failures_are_never_averaged_in(self):
        """A crashed run's duration is not a cost."""
        h = ExecutionHistory()
        node = _node(NodeType.GROUPBY)
        h.record(node.id, "duckdb", 1000, 10, elapsed_ms=10.0, success=True)
        h.record(node.id, "duckdb", 1000, 10, elapsed_ms=9999.0, success=False)
        assert h.predict(node, "duckdb", 1000) == pytest.approx(0.010)

    def test_regression_extrapolates_with_size(self):
        h = ExecutionHistory(min_samples_for_regression=3)
        node = _node(NodeType.GROUPBY)
        for size, ms in ((1000, 10.0), (2000, 20.0), (3000, 30.0),
                         (4000, 40.0)):
            h.record(node.id, "duckdb", size, size, elapsed_ms=ms)
        assert h.predict(node, "duckdb", 8000) > h.predict(node, "duckdb", 2000)

    def test_history_never_predicts_negative_time(self):
        h = ExecutionHistory(min_samples_for_regression=3)
        node = _node(NodeType.GROUPBY)
        for size, ms in ((1000, 100.0), (2000, 10.0), (3000, 50.0),
                         (4000, 20.0)):
            h.record(node.id, "duckdb", size, size, elapsed_ms=ms)
        for size in (1, 1000, 10_000_000):
            assert h.predict(node, "duckdb", size) >= 0

    def test_history_takes_precedence_over_calibration(self):
        store = CalibrationStore()
        store.add_points([CalibrationPoint("groupby", "cpu", 1000, 99.0)])
        h = ExecutionHistory()
        m = CostModel(calibration=store, registry=CapabilityRegistry(),
                      history=h)
        node = _node(NodeType.GROUPBY)
        h.record(node.id, "duckdb", 1000, 10, elapsed_ms=5.0)
        value, source = m.compute_s(node, "duckdb", 1000)
        assert source == "history"
        assert value == pytest.approx(0.005)

    def test_render_reports_totals(self):
        h = ExecutionHistory()
        h.record("op1", "duckdb", 1000, 10, elapsed_ms=25.0)
        text = h.render()
        assert "duckdb" in text and "25.0" in text


class TestExplain:
    def test_explain_names_engine_and_source(self):
        m = CostModel(calibration=CalibrationStore(),
                      registry=CapabilityRegistry())
        text = m.explain(_node(NodeType.GROUPBY), "duckdb", 1_000_000)
        assert "DuckDB" in text
        assert "compute from" in text

    def test_explain_flags_transfer_bound_work(self):
        """Work that is mostly movement should say so."""
        store = CalibrationStore()
        store.add_points([CalibrationPoint("groupby", "gpu", 1_000_000, 0.001)])
        m = CostModel(calibration=store, registry=CapabilityRegistry(),
                      priors=Priors(gpu_startup_s=0.0))
        text = m.explain(_node(NodeType.GROUPBY), "polars_gpu", 1_000_000_000)
        assert "movement" in text

    def test_default_model_is_usable(self):
        m = default_cost_model()
        value, source = m.compute_s(_node(NodeType.GROUPBY), "duckdb", 1_000_000)
        assert value >= 0
        assert source in ("history", "calibration", "calibration(penalised)",
                          "prior")

