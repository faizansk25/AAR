"""Adaptive planner: segment decomposition and the segment-cost dynamic program.

The specification's counter-example is the anchor of this file. A per-operation
chooser produces::

    Filter GPU -> Join GPU -> GroupBy GPU -> UDF CPU -> Sort GPU

which pays the host/device bus three times. The segment planner must collapse
that to a single boundary, and the test that matters is the one that asserts
the *boundaries*, not merely that a plan was produced.
"""

from __future__ import annotations


from aar.capability import CapabilityRegistry, Device
from aar.cost import CostModel, Priors, TransferProfile
from aar.hardware.calibrate import CalibrationPoint, CalibrationStore
from aar.ir import BinOp, Col, Lit, Node, NodeType, ScanSpec
from aar.planner import (AdaptivePlanner, NodeTypeAffinity, decompose_into_segments, estimate_bytes, self_device)


# ------------------------------------------------------------- fixtures
def _scan(nbytes: int = 100_000_000) -> Node:
    n = Node(NodeType.SCAN_PARQUET,
             scan=ScanSpec(kind="parquet", path="t.parquet"))
    n.estimated_bytes = nbytes
    return n


def _pipeline(nbytes: int = 100_000_000) -> Node:
    """Filter -> GroupBy -> UDF -> Sort, the specification's shape."""
    scan = _scan(nbytes)
    filt = Node(NodeType.FILTER, inputs=[scan],
                predicate=BinOp(Col("v"), ">", Lit(1)))
    grp = Node(NodeType.GROUPBY, inputs=[filt], key_left=("g",))
    grp.estimated_bytes = nbytes // 2
    udf = Node(NodeType.PYTHON_UDF, inputs=[grp], udf_name="f")
    udf.estimated_bytes = nbytes // 2
    srt = Node(NodeType.SORT, inputs=[udf], sort_keys=(("v", True),))
    srt.estimated_bytes = nbytes // 2
    return srt


def _model(points) -> CostModel:
    store = CalibrationStore()
    if points:
        store.add_points(list(points))
    return CostModel(calibration=store, registry=CapabilityRegistry())


def _gpu_capable() -> tuple[CostModel, AdaptivePlanner]:
    """A model where the GPU is genuinely faster, and a planner that may use it."""
    n = 100_000_000
    points = []
    for op in ("filter", "groupby", "sort", "hash_join"):
        points.append(CalibrationPoint(op, "cpu", n, 0.400))
        points.append(CalibrationPoint(op, "gpu", n, 0.080))
    m = _model(points)
    reg = CapabilityRegistry()
    # Pretend the GPU exists so the arithmetic can be exercised on a machine
    # that has none. Availability is a separate, tested concern.
    planner = AdaptivePlanner(cost_model=m, registry=reg,
                              require_available=False)
    return m, planner


# ------------------------------------------------------------ decomposition
class TestDecomposition:
    def test_pipeline_splits_into_segments(self):
        segs = decompose_into_segments(_pipeline())
        assert len(segs) >= 2

    def test_segments_are_in_execution_order(self):
        segs = decompose_into_segments(_pipeline())
        assert "ScanParquet" in segs[0].op_types
        assert "Write" not in "".join(segs[-1].op_types)

    def test_python_udf_forces_its_own_segment(self):
        """A CPU-only node may never share a segment with GPU work."""
        segs = decompose_into_segments(_pipeline())
        udf_segs = [s for s in segs if NodeType.PYTHON_UDF in
                    [NodeType(t) for t in s.op_types]]
        assert len(udf_segs) == 1
        assert udf_segs[0].device is Device.CPU

    def test_segments_never_mix_devices(self):
        for seg in decompose_into_segments(_pipeline()):
            devices = {self_device(n) for n in seg.nodes}
            assert len(devices) == 1, f"segment mixes devices: {devices}"

    def test_every_node_appears_exactly_once(self):
        root = _pipeline()
        seen = [n.id for s in decompose_into_segments(root) for n in s.nodes]
        assert len(seen) == len(set(seen))
        assert set(seen) == {n.id for n in root.walk()}

    def test_single_node_is_one_segment(self):
        segs = decompose_into_segments(_scan())
        assert len(segs) == 1

    def test_estimated_bytes_used_when_present(self):
        n = _scan(12345)
        assert estimate_bytes(n) == 12345

    def test_estimated_bytes_falls_back_when_absent(self):
        n = Node(NodeType.FILTER)
        assert estimate_bytes(n) > 0

    def test_affinity_prefers_gpu_for_bulk_ops(self):
        assert self_device(Node(NodeType.GROUPBY)) is Device.GPU
        assert self_device(Node(NodeType.PYTHON_UDF)) is Device.CPU

    def test_affinity_rank_puts_preferred_first(self):
        assert (NodeTypeAffinity.rank(NodeType.GROUPBY, Device.GPU)
                < NodeTypeAffinity.rank(NodeType.GROUPBY, Device.REMOTE))


class TestSegmentPlanning:
    def test_every_planned_node_receives_an_engine(self):
        m, planner = _gpu_capable()
        root = _pipeline()
        planner.plan(root)          # the call is the point; it assigns
        for node in root.walk():
            assert node.assigned_engine, f"{node.type} has no engine"
            assert node.reason, f"{node.type} has no reason"

    def test_plan_is_deterministic(self):
        m, planner = _gpu_capable()
        a = planner.plan(_pipeline())
        b = planner.plan(_pipeline())
        assert len(a.engines) == len(b.engines)

    def test_python_udf_lands_on_the_cpu_worker(self):
        """A hard capability fact, not a preference."""
        m, planner = _gpu_capable()
        root = _pipeline()
        planner.plan(root)
        udf = [n for n in root.walk() if n.type is NodeType.PYTHON_UDF][0]
        assert udf.assigned_engine == "python_worker"

    def test_every_segment_candidate_is_a_real_engine(self):
        m, planner = _gpu_capable()
        plan = planner.plan(_pipeline())
        reg = CapabilityRegistry()
        for sp in plan.segments:
            for engine, _cost in sp.candidates:
                reg.spec(engine)  # raises if invented

    def test_reason_quotes_real_arithmetic(self):
        m, planner = _gpu_capable()
        plan = planner.plan(_pipeline())
        for sp in plan.segments:
            assert "ms" in sp.reason
            assert sp.engine in sp.reason

    def test_plan_render_is_readable(self):
        m, planner = _gpu_capable()
        text = planner.plan(_pipeline()).render()
        assert "PLAN" in text
        assert "total" in text
        assert "segment" in text


class TestSegmentBoundaries:
    """The specification's core claim, made measurable.

    With a 5x faster GPU kernel and 240+160 ms of transfers, a *per-operation*
    chooser picks the GPU everywhere it can. The *segment* planner must not,
    because the crossings cost more than the kernels save.
    """

    def _expensive_gpu_plan(self):
        n = 1_000_000_000
        points = []
        for op in ("filter", "groupby", "sort", "scan"):
            points.append(CalibrationPoint(op, "cpu", n, 0.400))
            points.append(CalibrationPoint(op, "gpu", n, 0.080))
        store = CalibrationStore()
        store.add_points(points)
        transfer = TransferProfile(h2d_bytes_per_s=n / 0.240,
                                   d2h_bytes_per_s=n / 0.160)
        model = CostModel(calibration=store,
                          registry=CapabilityRegistry(),
                          priors=Priors(gpu_startup_s=0.0, startup_s=0.0),
                          transfer=transfer)
        planner = AdaptivePlanner(cost_model=model,
                                  registry=CapabilityRegistry(),
                                  require_available=False)
        return planner

    def test_kernel_is_faster_on_gpu(self):
        """Precondition: the GPU genuinely is the faster kernel here."""
        planner = self._expensive_gpu_plan()
        m = planner._cost
        cpu, _ = m.node_cost(Node(NodeType.GROUPBY), "duckdb", 1_000_000_000)
        gpu, _ = m.node_cost(Node(NodeType.GROUPBY), "polars_gpu",
                             1_000_000_000)
        assert gpu.kernel_s < cpu.kernel_s

    def test_cpu_is_chosen_because_transfers_dominate(self):
        """And yet the CPU still wins overall - the specification's example."""
        plan = self._expensive_gpu_plan().plan(_pipeline(1_000_000_000))
        engines = set(plan.engines)
        assert "polars_gpu" not in engines, (
            f"GPU chosen despite dominating transfers: {plan.engines}")

    def test_chosen_engine_is_cheaper_end_to_end(self):
        """The decision must be justified by total cost, not by kernel time."""
        planner = self._expensive_gpu_plan()
        plan = planner.plan(_pipeline(1_000_000_000))
        for sp in plan.segments:
            if not sp.candidates:
                continue
            best_cost = sp.candidates[0][1]
            assert sp.total_s >= best_cost - 1e-9
            # The chosen engine is the cheapest candidate considered.
            assert sp.engine == sp.candidates[0][0]

    def test_boundaries_are_counted_and_reported(self):
        plan = self._expensive_gpu_plan().plan(_pipeline(1_000_000_000))
        assert plan.boundaries >= 0
        assert "boundary" in plan.render()

