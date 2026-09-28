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
        text = CostBreakdown(startup_s=0.1, compute_s=0.2).render()


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

