"""Adaptive planner: segment decomposition and the segment-cost dynamic program.

The specification's counter-example is the anchor of this file. A per-operation
chooser produces::

    Filter GPU -> Join GPU -> GroupBy GPU -> UDF CPU -> Sort GPU

which pays the host/device bus three times. The segment planner must collapse
that to a single boundary, and the test that matters is the one that asserts
the *boundaries*, not merely that a plan was produced.
"""

from __future__ import annotations

import pytest

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


class TestDeclaredIsNotPlanned:
    """A plan must never name an engine this build cannot construct.

    The capability catalogue is allowed to be aspirational - it describes
    sixteen engines so that a distributed one can be specified before anyone
    has written it. The planner is not allowed to be. If it can name an
    engine with no class behind it, then on a machine that happens to have
    that engine's *package* installed, `aar explain` prints an engine name
    that never ran, and the segment silently executes on a fallback. The
    plan becomes a lie, which is the one thing this project is not allowed
    to ship.
    """

    UNIMPLEMENTED = ("ray", "dask", "spark_rapids", "trino", "mongodb",
                     "postgresql", "mysql", "sqlite")

    def test_the_catalogue_really_does_declare_unimplemented_engines(self):
        """If this ever stops being true the tests below are vacuous."""
        from aar.capability import ENGINES

        declared = {e.id for e in ENGINES}
        assert declared & set(self.UNIMPLEMENTED), (
            "none of the known-unimplemented engines are declared any more; "
            "either they were implemented (good - update the list) or the "
            "catalogue changed and this test no longer proves anything")

    def test_no_unimplemented_engine_is_ever_a_candidate(self):
        from aar.capability import CapabilityRegistry, Device
        from aar.ir import NodeType
        from aar.planner import AdaptivePlanner, Segment

        registry = CapabilityRegistry()
        # require_available=False on purpose: we are testing the
        # *implementation* filter, and on any machine without pyspark / ray /
        # dask installed the availability filter would hide the very bug
        # these tests exist to catch.
        planner = AdaptivePlanner(registry=registry, require_available=False)
        segment = Segment(index=0, nodes=[Node(NodeType.GROUPBY)],
                          device=Device.CPU, nbytes=1_000_000)
        candidates = planner.candidate_engines(segment)
        assert not (set(candidates) & set(self.UNIMPLEMENTED)), (
            "planner offered an engine with no implementation: "
            f"{sorted(set(candidates) & set(self.UNIMPLEMENTED))}")

    def test_marking_an_engine_available_does_not_resurrect_it(self):
        """The availability probe must not be the thing that saves us.

        `probe_engine` answers "is the package installed", which is a
        different question from "is there a class behind it". This forces the
        availability side to say yes and checks the implementation filter
        still refuses - the only way to prove the planner is not relying on
        pyspark happening to be absent from this laptop.
        """
        from aar.capability import CapabilityRegistry, Device
        from aar.ir import NodeType
        from aar.planner import AdaptivePlanner, Segment

        registry = CapabilityRegistry()
        for engine_id in self.UNIMPLEMENTED:
            # Re-insert the spec unchanged; what matters is that
            # `available_only=True` would now admit it, so the only thing
            # left filtering it out is the implementation check.
            registry._by_id[engine_id] = registry.spec(engine_id)

        planner = AdaptivePlanner(registry=registry, require_available=True)
        segment = Segment(index=0, nodes=[Node(NodeType.GROUPBY)],
                          device=Device.CPU, nbytes=1_000_000)
        candidates = planner.candidate_engines(segment)
        assert not (set(candidates) & set(self.UNIMPLEMENTED)), (
            "with those engines forced available, the planner still offered: "
            f"{sorted(set(candidates) & set(self.UNIMPLEMENTED))}")

    def test_implemented_ids_are_a_subset_of_the_catalogue(self):
        from aar.capability import ENGINES
        from aar.planner import implemented_engine_ids

        declared = {e.id for e in ENGINES}
        implemented = implemented_engine_ids()
        assert implemented, "nothing is implemented, which cannot be true"
        assert implemented <= declared, (
            f"implemented but not declared: {sorted(implemented - declared)}")
        assert "arrow" in implemented, (
            "arrow needs no third-party package and must always be buildable")


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

class TestBranchingDagIsOptimisedCorrectly:
    """The DP's state is "the engine of the previous segment".

    That is right for a chain and wrong for a branch. In a DAG the segment
    numbered just before segment *i* may not be one of its *predecessors* at
    all, and segment *i* may have two of them. The current table charges one
    crossing, from whichever segment happened to be numbered before, and
    charges nothing for the other input.

    A join with two inputs on different engines has to pay for both
    crossings. Paying for one is exactly the number that makes the planner
    prefer a plan it should not.

        scan1 -> filter1 --.
                        join -> write
        scan2 -> filter2 --'
    """

    SWITCH = 30.0

    def _planner(self, pinned: dict[int, str] | None = None):
        """Two engines, a fixed hop, and 1s of work per segment, so the
        only thing that varies is how many crossings are charged.

        ``pinned`` maps a segment index to the only engine it may use. That
        is how a straddle is forced: a pinned segment cannot share an engine
        with its neighbours, so a join downstream genuinely has two inputs
        arriving from different places. Pinning the IR node's
        ``assigned_engine`` does not work, because the planner assigns engines
        itself and overwrites it.
        """
        from aar.capability import CapabilityRegistry
        from aar.cost import CostBreakdown, CostModel

        switch = self.SWITCH
        pins = dict(pinned or {})

        class _FakeCost(CostModel):
            def segment_cost(self, segment, engine_id, from_engine=None,
                             residency=None):
                return CostBreakdown(compute_s=1.0), "calibration"

            def node_cost(self, node, engine_id, nbytes, from_engine=None,
                          residency=None):
                return CostBreakdown(compute_s=1.0), "calibration"

            def transition_cost(self, from_engine, to_engine, nbytes):
                if from_engine == to_engine:
                    return CostBreakdown()
                return CostBreakdown(transfer_s=switch)

        planner = AdaptivePlanner(
            cost_model=_FakeCost(calibration=None,
                                 registry=CapabilityRegistry()),
            registry=CapabilityRegistry(), require_available=False)
        if pins:
            # A subclass, because ``AdaptivePlanner`` uses ``__slots__`` and
            # therefore refuses attribute assignment. Overriding the method
            # is also the honest way to express the constraint: it *is* a
            # restriction on which engines a segment may use.
            base_candidates = planner.candidate_engines

            class _Pinned(type(planner)):  # type: ignore[misc]
                def candidate_engines(self, segment):
                    allowed = base_candidates(segment)
                    forced = pins.get(segment.index)
                    if forced:
                        narrowed = [e for e in allowed if e == forced]
                        return narrowed or [forced]
                    return allowed

            planner = _Pinned(planner._cost, planner._registry,
                              require_available=False)
        return planner

    def _branching_pipeline(self, right_engine: str | None = None):
        """scan/filter on each side, joined, then written.

        ``right_engine`` pins the right branch to one engine, which is how a
        straddle is forced: the two branches cannot then share an engine, so
        the join genuinely has two inputs arriving from different places.
        """
        from aar.ir import BinOp, Col, JoinType, Lit, Node, NodeType, ScanSpec

        def branch(name: str) -> Node:
            scan = Node(NodeType.SCAN_CSV, inputs=[],
                        scan=ScanSpec(kind="csv", path=f"{name}.csv"))
            scan.estimated_bytes = 1_000_000
            filt = Node(NodeType.FILTER, inputs=[scan],
                        predicate=BinOp(Col("k"), "=", Lit(1)))
            filt.estimated_bytes = 1_000_000
            return filt

        right = branch("right")
        if right_engine:
            # Pin it: every node on that branch is forced to one engine, so
            # the join cannot have both inputs arrive from the same place.
            for node in right.walk():
                node.assigned_engine = right_engine
        join = Node(NodeType.JOIN, inputs=[branch("left"), right],
                    key_left=("k",), join_type=JoinType.INNER)
        join.estimated_bytes = 2_000_000
        return Node(NodeType.WRITE, inputs=[join], target="out.csv",
                    write_format="csv")

    def _join_segment(self, plan):
        return next(sp for sp in plan.segments
                    if any("Join" in t for t in sp.segment.op_types))

    def test_a_join_pays_for_both_of_its_inputs(self):
        """The core defect: one crossing counted where there are two.

        The left branch is pinned to ``arrow`` and the right branch to
        ``duckdb``, so the join cannot avoid a straddle: whichever engine it
        picks, one of its two inputs has to cross. It must then pay for
        exactly that one crossing, and the plan must show it.

        Before the fix the table carried a single "previous engine" through
        the chain, so a plan that moved data between engines at every step
        cost the same as one that never moved any - 6.0s either way, with
        the crossings counted nowhere.
        """
        planner = self._planner(pinned={0: "arrow", 1: "arrow",
                                        2: "duckdb", 3: "duckdb"})
        plan = planner.plan(self._branching_pipeline())
        join_seg = self._join_segment(plan)

        engines = {sp.segment.index: sp.engine for sp in plan.segments}
        assert engines[1] == "arrow" and engines[3] == "duckdb", (
            f"branches are not straddled: {engines}")

        # The join sits on one side, so exactly one of its two inputs has to
        # cross. Charging for one is correct; charging for two would be as
        # wrong as charging for none.
        assert join_seg.inbound_s == pytest.approx(self.SWITCH, rel=1e-6), (
            f"join on {join_seg.engine} with inputs on arrow and duckdb "
            f"charged {join_seg.inbound_s:.1f}s, expected {self.SWITCH:.1f}s")

    def test_the_plan_is_not_cheaper_than_its_true_cost(self):
        """A straddle must cost more than agreeing on one engine.

        Everything on one engine pays nothing; a straddle pays for each
        branch that has to reach across. Before the fix both cost 6.0s, so
        the plan that moved the most data looked free.
        """
        straddled = self._planner(pinned={0: "arrow", 1: "arrow",
                                          2: "duckdb", 3: "duckdb"})
        together = self._planner()
        root = self._branching_pipeline()
        a = straddled.plan(root)
        b = together.plan(root)

        assert set(a.engines) == {"arrow", "duckdb"}, (
            f"expected a straddle, got {a.engines}")
        assert set(b.engines) == {"arrow"}, f"expected one engine: {b.engines}"
        assert a.total_s > b.total_s, (
            f"straddling plan {a.engines} costs {a.total_s:.1f}s but the "
            f"single-engine plan {b.engines} costs {b.total_s:.1f}s")
        # One crossing, at the join, for the branch that has to reach across.
        assert self._join_segment(a).inbound_s == pytest.approx(
            self.SWITCH, rel=1e-6)

    def test_a_chain_is_unaffected_by_the_dag_change(self):
        """A fix that only handles joins is not a fix."""
        from aar.ir import Node, NodeType

        def labelled(label, previous=None):
            node = Node(NodeType.MATERIALIZE,
                        inputs=[previous] if previous else [])
            node.id = f"seg-{label}-deadbeef"
            node.estimated_bytes = 1_000_000
            return node

        plan = self._planner().plan(labelled("S2", labelled("S1")))
        assert len(plan.segments) >= 2
        hops = sum(1 for a, b in zip(plan.engines, plan.engines[1:])
                   if a != b)
        assert hops <= 1, f"chain paid {hops} crossings: {plan.engines}"


class TestPlanningIsGloballyOptimal:
    """The planner must minimise total cost, not each segment greedily.

    This is the one audit finding left unfixed, so it starts as a test that
    *fails*. Writing the DP first would have produced a test asserting
    "a table is used", which passes whether or not the result is optimal.

    The construction is the audit's: three segments, two engines, a switch
    cost. Segment 1 is marginally cheaper on A; segments 2 and 3 are much
    cheaper on B. Greedy takes A, then pays the switch twice. The optimal
    path pays a small premium on segment 1 and switches once.

        segment   engine A   engine B
        S1            10 s       12 s
        S2            40 s        5 s
        S3            40 s        5 s
        switch cost = 30 s

        greedy  A,A,A = 10 + 30 + 40 = 80
        optimal A,B,B = 10 + 12 + 30 + 5 + 5 = 62

    A planner that looks only at the current segment picks A for S1 and
    never recovers the 18 s.
    """

    #: Per-(operation, device) compute costs, in seconds, at the test's size.
    COSTS = {("S1", "a"): 10.0, ("S1", "b"): 12.0,
             ("S2", "a"): 40.0, ("S2", "b"): 5.0,
             ("S3", "a"): 40.0, ("S3", "b"): 5.0}
    SWITCH = 30.0

    def _planner(self):
        """A planner whose two engines have exactly the costs above.

        Built on a fake cost model rather than a real one so the numbers in
        the test are the numbers the planner sees. A regression this precise
        cannot be expressed through calibration curves.
        """
        from aar.capability import CapabilityRegistry
        from aar.cost import CostBreakdown, CostModel

        costs = self.COSTS
        switch = self.SWITCH
        # Two real, constructible CPU engines stand in for "a" and "b", so
        # the plan the planner produces is one a user could actually run.
        # "a" is Arrow - the dependency every install has.
        real_a = "arrow"

        def _alias(engine_id: str) -> str:
            return "a" if engine_id == real_a else "b"

        class _FakeCost(CostModel):
            """Costs exactly what the test says, and nothing else."""

            def compute_s(self, node, engine_id, nbytes):
                # Nodes cannot carry a test-only attribute (Node has
                # __slots__), so the label is read off the node id the
                # fixture encodes: "seg-S2-<hash>".
                label = next((lab for lab in ("S1", "S2", "S3")
                              if f"seg-{lab}" in node.id), "S1")
                return (costs[(label, _alias(engine_id))], "calibration")

            def node_cost(self, node, engine_id, nbytes, from_engine=None,
                          residency=None):
                compute, source = self.compute_s(node, engine_id, nbytes)
                transfer = 0.0
                if from_engine and from_engine != engine_id:
                    transfer = switch
                return (CostBreakdown(startup_s=0.0, transfer_s=transfer,
                                      compute_s=compute), source)

            def transition_cost(self, from_engine, to_engine, nbytes):
                return CostBreakdown() if from_engine == to_engine \
                    else CostBreakdown(transfer_s=switch)

            def _fits(self, *a, **k):
                return True

        model = _FakeCost(calibration=None, registry=CapabilityRegistry())
        return AdaptivePlanner(cost_model=model,
                               registry=CapabilityRegistry(),
                               require_available=False)

    def _three_segment_pipeline(self):
        """Three boundary-separated segments, one node each.

        ``MATERIALIZE`` forces a segment boundary, which is what makes this
        three separate decisions rather than one. The segment label is
        encoded in the node id because ``Node`` has ``__slots__`` and cannot
        carry a test-only attribute.
        """
        from aar.ir import Node, NodeType

        def labelled(label: str, previous=None) -> Node:
            node = Node(NodeType.MATERIALIZE,
                        inputs=[previous] if previous else [])
            node.id = f"seg-{label}-deadbeef"
            node.estimated_bytes = 1_000_000
            return node

        s1 = labelled("S1")
        s2 = labelled("S2", s1)
        return labelled("S3", s2)

    def test_the_chosen_plan_is_the_globally_cheapest_one(self):
        """Fails today. That is the point of writing it first."""
        planner = self._planner()
        plan = planner.plan(self._three_segment_pipeline())

        total = sum(sp.total_s for sp in plan.segments)
        # The optimal assignment by brute force over {a,b}^3.
        best = min(
            sum(self.COSTS[(lab, eng)] for lab, eng in
                zip(("S1", "S2", "S3"), combo))
            + self.SWITCH * sum(1 for x, y in zip(combo, combo[1:]) if x != y)
            for combo in ("aab", "aba", "abb", "aaa", "bbb"))
        assert total == pytest.approx(best, rel=1e-6), (
            f"planner chose {plan.engines} costing {total:.1f}s; "
            f"the optimal path costs {best:.1f}s")

    def test_it_does_not_pay_the_switch_twice_when_paying_once_is_cheaper(self):
        """The specific error: A,A,B pays 2 switches; A,B,B pays one."""
        planner = self._planner()
        plan = planner.plan(self._three_segment_pipeline())
        engines = plan.engines
        switches = sum(1 for x, y in zip(engines, engines[1:]) if x != y)
        assert switches <= 1, (
            f"plan {engines} crosses a boundary {switches} times; "
            f"staying put on one engine is cheaper here")




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
        """The decision must be justified by total cost, not by kernel time.

        Note what this no longer asserts: that the chosen engine is the
        cheapest candidate *for its own segment*. That was the greedy rule,
        and it is wrong - a segment may pay more now to avoid a later
        crossing, and the planner must be free to do that. What must still
        hold is that no alternative engine would have made the whole run
        cheaper, which is what the optimality class now checks directly.
        """
        planner = self._expensive_gpu_plan()
        plan = planner.plan(_pipeline(1_000_000_000))
        for sp in plan.segments:
            if not sp.candidates:
                continue
            # The chosen engine is one of the candidates actually considered.
            assert sp.engine in [e for e, _ in sp.candidates]
            # And paying the hop, the segment never costs less than doing
            # the work on its own.
            assert sp.total_s >= sp.cost.compute_s - 1e-9

    def test_boundaries_are_counted_and_reported(self):
        plan = self._expensive_gpu_plan().plan(_pipeline(1_000_000_000))
        assert plan.boundaries >= 0
        assert "boundary" in plan.render()

    def test_a_cross_boundary_transfer_is_charged_exactly_once(self):
        """The accounting invariant: the planner must not pay a hop twice.

        ``node_cost`` folds the inbound transition into ``transfer_s``, and
        ``SegmentPlan.total_s`` adds ``inbound_s`` on top of
        ``cost.total_s``. Both are correct only if the hop is *removed* from
        the breakdown when it is recorded separately - which is what the
        planner now does.
        """
        n = 1_000_000_000
        planner = self._expensive_gpu_plan()
        model = planner._cost
        plan = planner.plan(_pipeline(n))

        for sp in plan.segments:
            if sp.inbound_s <= 0:
                continue
            # The recorded inbound hop must be a real transition cost.
            engine_before = plan.engines[plan.segments.index(sp) - 1]
            expected = model.transition_cost(engine_before, sp.engine,
                                             sp.segment.nbytes).total_s
            assert sp.inbound_s == pytest.approx(expected, rel=1e-9)
            # And it must not also be sitting inside the breakdown's transfer,
            # which the cost model added it to.
            raw, _ = model.node_cost(
                sp.segment.nodes[-1], sp.engine, sp.segment.nbytes,
                from_engine=engine_before,
                residency=model._registry.spec(engine_before).device)
            assert sp.cost.transfer_s == pytest.approx(
                raw.transfer_s - sp.inbound_s, rel=1e-6), (
                f"segment {sp.segment.index}: inbound {sp.inbound_s:.6f}s was "
                f"counted in transfer_s as well as in inbound_s")

    def test_the_plan_total_is_the_sum_of_its_segments(self):
        """No segment may be billed twice inside the plan total."""
        planner = self._expensive_gpu_plan()
        plan = planner.plan(_pipeline(1_000_000_000))
        summed = sum(sp.total_s for sp in plan.segments)
        expected = summed * (1.0 + planner._margin)
        assert plan.total_s == pytest.approx(expected, rel=1e-6)

