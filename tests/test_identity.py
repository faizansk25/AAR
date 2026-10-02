"""Semantic identity: the property that makes measurement history possible.

Every test here exists because a weaker version would pass while the subsystem
did nothing. The anchor is
:class:`TestHistoryCrossesAPipelineRebuild`: two separately constructed
pipelines, different ``Node.id`` values, one shared history entry found.
Without that, the rest is attractive hashing.

The negative tests matter at least as much. An identity that is *too* stable -
ignoring a changed join type, a changed literal, a changed schema - would pass
every "same pipeline, same ID" check while silently attributing one
operation's measurements to a different operation. That is worse than no
identity, so each is pinned explicitly.
"""

from __future__ import annotations


import dataclasses
import pytest

from aar.cost import ExecutionHistory
from aar.ir import (BinOp, Col, JoinType, Lit, Node, NodeType, ScanSpec,
                    WindowSpec, graph_node_id, semantic_operation_id)
from aar.ir.identity import (SEMANTIC_ID_VERSION, UnstableSemanticIdentity,
                             operation_payload, resource_snapshot, target_id)


# --------------------------------------------------------------- construction
def _filter(**kw) -> Node:
    n = Node(NodeType.FILTER, predicate=BinOp(Col("amount"), ">", Lit(100)))
    for k, v in kw.items():
        setattr(n, k, v)
    return n


def _scan(path: str = "a.parquet", **kw) -> Node:
    return Node(NodeType.SCAN_PARQUET,
                scan=ScanSpec(kind="parquet", path=path), **kw)


def _pipeline() -> Node:
    """Scan -> filter -> join -> groupby -> sort."""
    scan = _scan()
    filt = Node(NodeType.FILTER, inputs=[scan],
                predicate=BinOp(Col("amount"), ">", Lit(100)))
    other = _scan("b.parquet")
    join = Node(NodeType.JOIN, inputs=[filt, other], key_left=("k",),
                key_right=("k",), join_type=JoinType.INNER)
    grp = Node(NodeType.GROUPBY, inputs=[join], key_left=("region",))
    return Node(NodeType.SORT, inputs=[grp], sort_keys=(("region", True),))


class TestIdentityIsStableAcrossConstruction:
    def test_two_parses_produce_the_same_operation_ids(self):
        a = [semantic_operation_id(n) for n in _pipeline().walk()]
        b = [semantic_operation_id(n) for n in _pipeline().walk()]
        assert a == b
        # Six nodes: two scans (one feeds the filter, one the join), filter,
        # join, groupby, sort. Asserted so a change to the pipeline shape
        # cannot silently make this test vacuous.
        assert len(a) == 6

    def test_two_parses_produce_the_same_graph_node_ids(self):
        a = [graph_node_id(n) for n in _pipeline().walk()]
        b = [graph_node_id(n) for n in _pipeline().walk()]
        assert a == b

    def test_transient_ids_differ_while_semantic_ids_do_not(self):
        a = list(_pipeline().walk())
        b = list(_pipeline().walk())
        assert [n.id for n in a] != [n.id for n in b], (
            "precondition: the UUIDs really are different")
        assert [semantic_operation_id(n) for n in a] == [
            semantic_operation_id(n) for n in b]

    def test_the_identifier_is_prefixed_and_versioned(self):
        ident = semantic_operation_id(_filter())
        assert ident.startswith(f"aar-op-v{SEMANTIC_ID_VERSION}:")
        assert graph_node_id(_filter()).startswith("aar-node-v1:")
        # A digest, so a log line is recognisable rather than opaque.
        assert len(ident.split(":", 1)[1]) == 64

    def test_operation_and_graph_node_ids_are_different_spaces(self):
        """Domain separation. A collision here would cross-wire two lookups."""
        node = _filter()
        assert semantic_operation_id(node) != graph_node_id(node)


class TestOperationAndGraphNodeDiffer:
    def test_the_same_filter_under_different_sources_shares_its_operation(self):
        """Reusable performance history: one operation, many pipelines."""
        left = Node(NodeType.FILTER, inputs=[_scan("a.parquet")],
                    predicate=BinOp(Col("amount"), ">", Lit(100)))
        right = Node(NodeType.FILTER, inputs=[_scan("b.parquet")],
                     predicate=BinOp(Col("amount"), ">", Lit(100)))
        assert semantic_operation_id(left) == semantic_operation_id(right)

    def test_but_they_are_different_graph_nodes(self):
        """Different provenance chains, so measurements must not be pooled."""
        left = Node(NodeType.FILTER, inputs=[_scan("a.parquet")],
                    predicate=BinOp(Col("amount"), ">", Lit(100)))
        right = Node(NodeType.FILTER, inputs=[_scan("b.parquet")],
                     predicate=BinOp(Col("amount"), ">", Lit(100)))
        assert graph_node_id(left) != graph_node_id(right)

    def test_input_order_is_not_normalised_away(self):
        """A join's inputs[0] is its left side. Swapping them is a new node."""
        a, b = _scan("a.parquet"), _scan("b.parquet")
        left_first = Node(NodeType.JOIN, inputs=[a, b], key_left=("k",),
                          key_right=("k",))
        right_first = Node(NodeType.JOIN, inputs=[b, a], key_left=("k",),
                           key_right=("k",))
        assert semantic_operation_id(left_first) == semantic_operation_id(
            right_first)
        assert graph_node_id(left_first) != graph_node_id(right_first)


class _Profile:
    """A hardware profile with mutable free/available state."""

    def __init__(self, **over):
        self.os = type("os", (), {"system": "Linux", "release": "6.1",
                                  "architecture": "x86_64"})()
        self.cpu = type("cpu", (), {"model": "AMD EPYC 7B12",
                                    "physical_cores": 8,
                                    "logical_cores": 16, "simd_level": "avx2",
                                    "max_freq_mhz": 3400})()
        self.memory = type("mem", (), {"total_bytes": 34_000_000_000,
                                       "available_bytes": 20_000_000_000})()
        self.gpu = type("gpu", (), {"vendor": "", "model": "", "vram_bytes": 0,
                                    "compute_capability": "",
                                    "driver_version": ""})()
        self.software = type("sw", (), {"python": "3.12.1", "arrow": "17.0.0",
                                        "duckdb": "1.0.0", "polars": None,
                                        "cudf": None})()
        for k, v in over.items():
            setattr(self, k, v)


class TestTargetIdentity:
    def test_it_is_prefixed_and_versioned(self):
        assert target_id(_Profile()).startswith("aar-target-v1:")

    def test_the_same_machine_gives_the_same_id(self):
        assert target_id(_Profile()) == target_id(_Profile())

    def test_different_memory_pressure_gives_the_same_id(self):
        """Free RAM changes constantly. Keying a target on it would make every
        run a different target, which is the defect this split avoids."""
        busy = _Profile()
        busy.memory = type("mem", (), {"total_bytes": 34_000_000_000,
                                       "available_bytes": 500_000_000})()
        assert target_id(busy) == target_id(_Profile())

    def test_a_different_cpu_gives_a_different_id(self):
        other = _Profile()
        other.cpu = type("cpu", (), {"model": "Intel Xeon", "physical_cores": 4,
                                     "logical_cores": 8, "simd_level": "avx512",
                                     "max_freq_mhz": 2600})()
        assert target_id(other) != target_id(_Profile())

    def test_a_different_total_ram_gives_a_different_id(self):
        other = _Profile()
        other.memory = type("mem", (), {"total_bytes": 68_000_000_000,
                                        "available_bytes": 20_000_000_000})()
        assert target_id(other) != target_id(_Profile())


class TestResourceSnapshotIsSeparate:
    """``target_id`` names the machine; the snapshot records what it was
    allowed to use. Answering "why GPU yesterday, CPU today?" needs both."""

    def test_it_records_the_exact_budgets(self):
        from aar.cost import ResourceBudget

        snap = resource_snapshot(
            ResourceBudget(ram_bytes=64_000_000_000, vram_bytes=8_000_000_000))
        assert snap["ram_budget_bytes"] == 64_000_000_000
        assert snap["vram_budget_bytes"] == 8_000_000_000
        assert snap["v"] == SEMANTIC_ID_VERSION

    def test_the_same_machine_with_different_budgets_keeps_one_target(self):
        from aar.cost import ResourceBudget

        profile = _Profile()
        generous = resource_snapshot(
            ResourceBudget(ram_bytes=64_000_000_000, vram_bytes=20_000_000_000))
        tight = resource_snapshot(
            ResourceBudget(ram_bytes=1_000_000_000, vram_bytes=0))
        assert target_id(profile) == target_id(profile)
        assert generous != tight

    def test_it_is_a_record_not_a_digest(self):
        """Two differing snapshots must stay inspectable - the difference is
        the finding, and hashing it away would discard the explanation."""
        from aar.cost import ResourceBudget

        snap = resource_snapshot(ResourceBudget(ram_bytes=1_000_000_000))
        assert isinstance(snap, dict)
        assert snap["ram_budget_bytes"] == 1_000_000_000

    """Everything the planner writes must be invisible to the ID.

    AAR's IR keeps logical and physical state in one dataclass. If any of this
    leaked into the payload, a node would get a new identity every time it was
    planned, and history could never accumulate at all.
    """

    def test_estimates_do_not_change_it(self):
        base = semantic_operation_id(_filter())
        n = _filter()
        n.estimated_rows = 1_000_000
        n.estimated_bytes = 4_000_000_000
        n.estimated_selectivity = 0.25
        assert semantic_operation_id(n) == base

    def test_the_assigned_engine_and_reason_do_not_change_it(self):
        base = semantic_operation_id(_filter())
        n = _filter()
        n.assigned_engine = "duckdb"
        n.reason = "cheapest end to end"
        n.segment_id = 3
        n.estimated_ms = 12.5
        n.estimated_peak_memory = 999_999_999
        n.expected_saving_ms = 4.0
        n.fallback_engine = "polars"
        assert semantic_operation_id(n) == base

    def test_the_node_id_itself_does_not_change_it(self):
        a, b = _filter(), _filter()
        assert a.id != b.id
        assert semantic_operation_id(a) == semantic_operation_id(b)

    def test_a_measured_profile_does_not_change_it(self):
        """``aar_profile`` is evidence about the data, not a property of the
        operation. Including it would mean a fresh Parquet-footer read
        invalidated every stored measurement for the node."""
        class _Profile:
            nbytes = 12345
            num_rows = 99
        n = _filter()
        n.aar_profile = _Profile()
        assert semantic_operation_id(n) == semantic_operation_id(_filter())


class TestSemanticChangesChangeTheId:
    """The negative half. An over-stable identity is worse than none."""

    def test_a_changed_filter_literal_changes_it(self):
        other = Node(NodeType.FILTER,
                     predicate=BinOp(Col("amount"), ">", Lit(101)))
        assert semantic_operation_id(_filter()) != semantic_operation_id(other)

    def test_a_changed_filter_column_changes_it(self):
        other = Node(NodeType.FILTER,
                     predicate=BinOp(Col("cost"), ">", Lit(100)))
        assert semantic_operation_id(_filter()) != semantic_operation_id(other)

    def test_a_changed_operator_changes_it(self):
        other = Node(NodeType.FILTER,
                     predicate=BinOp(Col("amount"), ">=", Lit(100)))
        assert semantic_operation_id(_filter()) != semantic_operation_id(other)

    def test_a_changed_join_type_changes_it(self):
        def _join(kind):
            return Node(NodeType.JOIN, inputs=[_scan("a.parquet"),
                                               _scan("b.parquet")],
                        key_left=("k",), key_right=("k",), join_type=kind)
        assert semantic_operation_id(_join(JoinType.INNER)) != \
            semantic_operation_id(_join(JoinType.LEFT))

    def test_a_changed_sort_direction_changes_it(self):
        asc = Node(NodeType.SORT, inputs=[_scan()], sort_keys=(("v", True),))
        desc = Node(NodeType.SORT, inputs=[_scan()], sort_keys=(("v", False),))
        assert semantic_operation_id(asc) != semantic_operation_id(desc)

    def test_reordering_sort_keys_changes_it(self):
        """sort by (a, b) is not the same query as sort by (b, a)."""
        ab = Node(NodeType.SORT, sort_keys=(("a", True), ("b", True)))
        ba = Node(NodeType.SORT, sort_keys=(("b", True), ("a", True)))
        assert semantic_operation_id(ab) != semantic_operation_id(ba)

    def test_a_changed_scan_path_changes_it(self):
        assert semantic_operation_id(_scan("a.parquet")) != \
            semantic_operation_id(_scan("b.parquet"))

    def test_a_changed_scan_delimiter_changes_it(self):
        comma = _scan()
        comma.scan.delimiter = ","
        semi = _scan()
        semi.scan.delimiter = ";"
        assert semantic_operation_id(comma) != semantic_operation_id(semi)

    def test_a_changed_window_frame_changes_it(self):
        def _win(frame):
            return Node(NodeType.WINDOW, inputs=[_scan()],
                        window=WindowSpec(partition_by=("g",), frame=frame))
        assert semantic_operation_id(_win("rows between 1 preceding and "
                                          "current row")) != \
            semantic_operation_id(_win("rows between unbounded preceding and "
                                       "current row"))

    def test_a_changed_node_type_changes_it(self):
        assert semantic_operation_id(_filter()) != semantic_operation_id(
            Node(NodeType.PROJECT, columns=("a",)))

    def test_100_and_100_0_are_different_literals(self):
        """They render identically through str()."""
        ints = Node(NodeType.FILTER, predicate=BinOp(Col("v"), "=", Lit(100)))
        floats = Node(NodeType.FILTER,
                      predicate=BinOp(Col("v"), "=", Lit(100.0)))
        assert semantic_operation_id(ints) != semantic_operation_id(floats)

    def test_true_and_one_are_different_literals(self):
        """bool is an int subclass; True and 1 must not collide."""
        t = Node(NodeType.FILTER, predicate=BinOp(Col("v"), "=", Lit(True)))
        one = Node(NodeType.FILTER, predicate=BinOp(Col("v"), "=", Lit(1)))
        assert semantic_operation_id(t) != semantic_operation_id(one)

    def test_nan_infinity_and_signed_zero_are_distinguishable(self):
        """JSON cannot represent them, so they must be encoded explicitly.

        ``allow_nan=False`` means an unhandled NaN raises rather than emitting
        a token that hashes differently depending on the writer.
        """
        def _with(value):
            return Node(NodeType.FILTER,
                        predicate=BinOp(Col("v"), "=", Lit(value)))
        idents = {
            semantic_operation_id(_with(float("nan"))),
            semantic_operation_id(_with(float("inf"))),
            semantic_operation_id(_with(float("-inf"))),
            semantic_operation_id(_with(0.0)),
            semantic_operation_id(_with(-0.0)),
        }
        assert len(idents) == 5

    def test_and_is_not_reordered(self):
        """No commutativity normalisation in v1.

        ``a AND b`` and ``b AND a`` are mathematically equal, but null
        semantics, evaluation order and engine pushdown make rewriting them a
        source of subtle wrongness. Correctness first; deduplication later.
        """
        ab = Node(NodeType.FILTER, predicate=BinOp(
            BinOp(Col("a"), "=", Lit(1)), "AND", BinOp(Col("b"), "=", Lit(2))))
        ba = Node(NodeType.FILTER, predicate=BinOp(
            BinOp(Col("b"), "=", Lit(2)), "AND", BinOp(Col("a"), "=", Lit(1))))
        assert semantic_operation_id(ab) != semantic_operation_id(ba)

    def test_a_changed_write_target_changes_it(self):
        a = Node(NodeType.WRITE, target="out.parquet")
        b = Node(NodeType.WRITE, target="other.parquet")
        assert semantic_operation_id(a) != semantic_operation_id(b)

    """A UDF's name is not its identity."""

    def test_the_same_function_gives_the_same_id(self):
        def clean(x):
            return x + 1
        a = Node(NodeType.PYTHON_UDF, udf=clean, udf_name="clean")
        b = Node(NodeType.PYTHON_UDF, udf=clean, udf_name="clean")
        assert semantic_operation_id(a) == semantic_operation_id(b)

    def test_two_functions_with_the_same_name_differ(self):
        """The reason ``udf_name`` alone is useless: v1 and v2 of ``clean``
        share a name and do completely different work."""
        def clean_v1(x):
            return x + 1

        def clean_v2(x):
            return x * 10
        a = Node(NodeType.PYTHON_UDF, udf=clean_v1, udf_name="clean")
        b = Node(NodeType.PYTHON_UDF, udf=clean_v2, udf_name="clean")
        assert a.udf_name == b.udf_name, "precondition: same declared name"
        assert semantic_operation_id(a) != semantic_operation_id(b)

    def test_a_changed_default_changes_the_id(self):
        def scale(x, factor=1):
            return x * factor

        def scale_ten(x, factor=10):
            return x * factor
        a = Node(NodeType.PYTHON_UDF, udf=scale, udf_name="scale")
        b = Node(NodeType.PYTHON_UDF, udf=scale_ten, udf_name="scale")
        assert semantic_operation_id(a) != semantic_operation_id(b)

    def test_a_reformat_does_not_change_the_digest(self):
        """Reindentation and comments are not behaviour changes.

        Tested on the normaliser directly, because within one process you
        cannot hold two textually different versions of the *same* function
        and so cannot compare their IDs end to end. What this does prove is the
        property the ID depends on: formatting is stripped before hashing.
        """
        from aar.ir.identity import _normalise_source

        tight = "def f(x):\n    return x + 1\n"
        loose = ("def f(x):\n"
                 "    # a comment that cannot change behaviour\n"
                 "\n"
                 "        return x + 1\n")
        assert _normalise_source(tight) == _normalise_source(loose)

    def test_a_changed_body_does_change_the_digest(self):
        from aar.ir.identity import _normalise_source

        assert _normalise_source("def f(x):\n    return x + 1\n") != \
            _normalise_source("def f(x):\n    return x * 10\n")

    def test_renaming_the_function_changes_the_id(self):
        """Parameter names and the function name are kept, not stripped.

        Two differently named functions are different functions even when their
        bodies match, and the ``def`` line is what says so.
        """
        from aar.ir.identity import _normalise_source

        assert _normalise_source("def a(x):\n    return x\n") != \
            _normalise_source("def b(x):\n    return x\n")

    def test_an_unfingerprintable_callable_is_refused(self):
        class Opaque:
            def __call__(self, x):
                return x
        n = Node(NodeType.PYTHON_UDF, udf=Opaque(), udf_name="opaque")
        with pytest.raises(UnstableSemanticIdentity):
            semantic_operation_id(n)

    def test_the_escape_hatch_makes_it_identifiable(self):
        """A callable with no recoverable source can still be *declared*."""
        class Opaque:
            def __call__(self, x):
                return x
        n = Node(NodeType.PYTHON_UDF, udf=Opaque(), udf_name="opaque",
                 semantic_version="clean-v3")
        assert semantic_operation_id(n).startswith("aar-op-v1:")

    def test_two_versions_of_the_escape_hatch_differ(self):
        class Opaque:
            def __call__(self, x):
                return x
        a = Node(NodeType.PYTHON_UDF, udf=Opaque(), semantic_version="v1")
        b = Node(NodeType.PYTHON_UDF, udf=Opaque(), semantic_version="v2")
        assert semantic_operation_id(a) != semantic_operation_id(b)

    def test_a_builtin_is_identified_by_name_not_refused(self):
        """A builtin has no source but does have a stable identity."""
        n = Node(NodeType.PYTHON_UDF, udf=len, udf_name="len")
        assert semantic_operation_id(n).startswith("aar-op-v1:")


class TestRefusesRatherThanGuesses:
    def test_an_unknown_literal_type_raises(self):
        class Weird:
            def __repr__(self):
                return "<0x7fabc>"

        n = Node(NodeType.FILTER,
                 predicate=BinOp(Col("v"), "=", Lit(Weird())))
        with pytest.raises(UnstableSemanticIdentity):
            semantic_operation_id(n)

    def test_the_payload_explains_a_difference(self):
        """A caller must be able to see *why* two nodes differ, not just that
        they do - otherwise identity is undebuggable."""
        base = operation_payload(_filter())
        other = operation_payload(
            Node(NodeType.FILTER, predicate=BinOp(Col("amount"), ">", Lit(7))))
        assert base != other
        assert base["op"]["predicate"]["right"]["value"]["v"] == "100"
        assert other["op"]["predicate"]["right"]["value"]["v"] == "7"

    def test_the_payload_carries_its_version(self):
        assert operation_payload(_filter())["v"] == SEMANTIC_ID_VERSION


class TestPlanningStateIsNotIdentity:
    """Everything the planner writes must be invisible to the ID.

    AAR's IR keeps logical and physical state in one dataclass. If any of this
    leaked into the payload, a node would get a new identity every time it was
    planned, and history could never accumulate at all.
    """

    def test_estimates_do_not_change_it(self):
        base = semantic_operation_id(_filter())
        n = _filter()
        n.estimated_rows = 1_000_000
        n.estimated_bytes = 4_000_000_000
        n.estimated_selectivity = 0.25
        assert semantic_operation_id(n) == base

    def test_the_assigned_engine_and_reason_do_not_change_it(self):
        base = semantic_operation_id(_filter())
        n = _filter()
        n.assigned_engine = "duckdb"
        n.reason = "cheapest end to end"
        n.segment_id = 3
        n.estimated_ms = 12.5
        n.estimated_peak_memory = 999_999_999
        n.expected_saving_ms = 4.0
        n.fallback_engine = "polars"
        assert semantic_operation_id(n) == base

    def test_the_node_id_itself_does_not_change_it(self):
        a, b = _filter(), _filter()
        assert a.id != b.id
        assert semantic_operation_id(a) == semantic_operation_id(b)

    def test_a_measured_profile_does_not_change_it(self):
        """``aar_profile`` is evidence about the data, not a property of the
        operation. Including it would mean a fresh Parquet-footer read
        invalidated every stored measurement for the node."""
        class _Measured:
            nbytes = 12345
            num_rows = 99
        n = _filter()
        n.aar_profile = _Measured()
        assert semantic_operation_id(n) == semantic_operation_id(_filter())


class TestHistoryCrossesAPipelineRebuild:
    """The integration test: this subsystem's reason for existing.

    Everything above proves the hash is stable and discriminating. This proves
    that stability is *reached*: two independently constructed pipelines, no
    shared objects, and the second finds the first one's measurements.

    Every key here is a real :func:`semantic_operation_id`. The superseded
    branch proved the same property with a key function returning
    ``"aar-op-v1:" + node.type``, which is stable for the wrong reason - it
    ignores the predicate, the join keys and the scan spec, so it would pass
    these tests while pooling genuinely different work together.
    """

    TARGET = "aar-target-v1:test"

    def _history(self, **kw):
        return ExecutionHistory(target_id=self.TARGET, **kw)

    @staticmethod
    def _obs(key, size=1_000_000, total_ms=50.0, target="aar-target-v1:test"):
        from aar.cost.history import ExecutionRecord

        return ExecutionRecord(
            operation_id=key, graph_node_id=None,
            target_id=target, resource_snapshot="budget:4GiB",
            engine="duckdb", input_bytes=(size,), input_rows=(size // 1000,),
            output_rows=size // 1000, output_bytes=size // 10,
            actual_elapsed_total_ms=total_ms, actual_compute_ms=total_ms)

    def test_a_rebuilt_pipeline_finds_a_previous_runs_evidence(self):
        history = self._history()

        first = _pipeline()
        for node in first.walk():
            history.record(self._obs(semantic_operation_id(node)))

        second = _pipeline()
        first_ids = {n.id for n in first.walk()}
        second_ids = {n.id for n in second.walk()}
        assert not (first_ids & second_ids), (
            "precondition: the two pipelines share no node instances")

        found = [n for n in second.walk()
                 if history.predict_for_node(n, "duckdb").usable]
        assert len(found) == 6, (
            "a rebuilt pipeline must find every measurement the first one "
            f"recorded; found {len(found)} of 6")

    def test_the_operation_id_is_what_matched(self):
        """Not an accident of the graph-node id, and not the transient id."""
        history = self._history()
        node = Node(NodeType.FILTER,
                    predicate=BinOp(Col("amount"), ">", Lit(100)))
        history.record(self._obs(semantic_operation_id(node), size=1000,
                                 total_ms=42.0))
        rebuilt = Node(NodeType.FILTER,
                       predicate=BinOp(Col("amount"), ">", Lit(100)))
        assert rebuilt.id != node.id
        assert semantic_operation_id(rebuilt) == semantic_operation_id(node)
        got = history.predict_for_node(rebuilt, "duckdb")
        assert got.seconds == pytest.approx(0.042)

    def test_a_different_operation_is_not_served_from_that_history(self):
        """The guard that a type-only key function would fail.

        Both nodes are ``Filter``. Only the real identity - which hashes the
        predicate - tells them apart.
        """
        history = self._history()
        history.record(self._obs(semantic_operation_id(_filter()), size=1000,
                                 total_ms=42.0))
        changed = Node(NodeType.FILTER,
                       predicate=BinOp(Col("amount"), ">", Lit(999)))
        assert changed.type is _filter().type, (
            "precondition: same node type, different predicate")
        assert not history.predict_for_node(changed, "duckdb").usable

    def test_the_same_operation_on_two_machines_does_not_pool(self):
        """Evidence from machine A must not answer for machine B."""
        history = ExecutionHistory(target_id="aar-target-v1:aaa")
        node = Node(NodeType.FILTER,
                    predicate=BinOp(Col("amount"), ">", Lit(100)))
        key = semantic_operation_id(node)
        history.record(self._obs(key, size=1000, total_ms=50.0,
                                 target="aar-target-v1:aaa"))
        history.record(self._obs(key, size=1000, total_ms=5000.0,
                                 target="aar-target-v1:bbb"))

        assert history.predict(key, "duckdb", (1000,)).seconds == \
            pytest.approx(0.050)
        other = history.predict(key, "duckdb", (1000,),
                                target_id="aar-target-v1:bbb")
        assert other.seconds == pytest.approx(5.000)


class TestObservationsKeepEveryInput:
    def test_a_join_records_both_sides(self):
        """The defect: ``inputs[0]`` alone lost the right side entirely.

        A 2 GB + 8 GB join recorded 2 GB, so a per-byte cost was learned from a
        quarter of the bytes actually read.
        """
        from aar.runtime.executor import NodeOutcome

        outcome = NodeOutcome(
            node_id="n1", node_type="Join",
            engine_requested="duckdb", engine_used="duckdb",
            input_rows=(1_000, 4_000),
            input_bytes=(2_000_000_000, 8_000_000_000))
        assert outcome.input_bytes == (2_000_000_000, 8_000_000_000)
        assert outcome.bytes_in == 10_000_000_000
        assert outcome.max_bytes_in == 8_000_000_000
        assert outcome.input_rows == (1_000, 4_000)
        assert outcome.rows_in == 5_000

    def test_unmeasured_is_none_not_zero(self):
        """A zero peak-memory reading and "never measured" are different
        facts, and a store that conflates them averages absent data."""
        from aar.runtime.executor import NodeOutcome

        outcome = NodeOutcome(node_id="n", node_type="Filter",
                              engine_requested="duckdb", engine_used="duckdb")
        assert outcome.actual_peak_memory is None
        assert outcome.actual_transfer_bytes is None
        assert outcome.bytes_out == 0

    def test_a_record_also_keeps_none_for_unmeasured(self):
        """A zero peak-memory reading and "never measured" are different facts."""
        from aar.cost.history import ExecutionRecord

        rec = ExecutionRecord(
            operation_id="k", graph_node_id=None, target_id="t",
            resource_snapshot="b", engine="duckdb", input_bytes=(1000,),
            actual_elapsed_total_ms=5.0, actual_compute_ms=5.0)
        assert rec.actual_peak_memory is None
        assert rec.actual_transfer_bytes is None
        assert rec.output_bytes is None

    def test_a_measured_zero_is_still_distinguishable(self):
        from aar.cost.history import ExecutionRecord

        rec = ExecutionRecord(
            operation_id="k", graph_node_id=None, target_id="t",
            resource_snapshot="b", engine="duckdb", input_bytes=(1000,),
            actual_elapsed_total_ms=5.0, actual_compute_ms=5.0,
            actual_peak_memory=0)
        assert rec.actual_peak_memory == 0
        assert rec.actual_peak_memory is not None

    def test_a_record_is_the_full_contract(self):
        """Every field the subsystem needs, present and named.

        Asserted as a set because each of these was once missing: without a
        target an observation cannot be attributed, without a compute time it
        cannot be compared with a predicted kernel cost, and without a resource
        snapshot a 4 GB measurement sits beside a 64 GB one as though they were
        the same experiment.
        """
        from aar.cost.history import ExecutionRecord

        fields = {f.name for f in dataclasses.fields(ExecutionRecord)}
        required = {
            "operation_id", "graph_node_id", "target_id", "resource_snapshot",
            "engine", "engine_version", "input_rows", "input_bytes",
            "output_rows", "output_bytes", "actual_elapsed_total_ms",
            "actual_compute_ms", "acquire_ms", "actual_transfer_bytes",
            "actual_peak_memory", "cost_model_version", "schema_version",
            "success", "failure_kind", "timestamp",
        }
        assert required <= fields, f"missing: {sorted(required - fields)}"

    def test_the_schema_version_is_recorded_on_every_row(self):
        from aar.cost.history import HISTORY_SCHEMA_VERSION, ExecutionRecord

        rec = ExecutionRecord(
            operation_id="k", graph_node_id=None, target_id="t",
            resource_snapshot="b", engine="duckdb", input_bytes=(1,))
        assert rec.schema_version == HISTORY_SCHEMA_VERSION


class TestIdentityIsDiscriminating:
    """The negative half. An over-stable identity is worse than none."""

    def test_a_changed_filter_literal_changes_it(self):
        other = Node(NodeType.FILTER,
                     predicate=BinOp(Col("amount"), ">", Lit(101)))
        assert semantic_operation_id(_filter()) != semantic_operation_id(other)

    def test_a_changed_filter_column_changes_it(self):
        other = Node(NodeType.FILTER,
                     predicate=BinOp(Col("cost"), ">", Lit(100)))
        assert semantic_operation_id(_filter()) != semantic_operation_id(other)

    def test_a_changed_operator_changes_it(self):
        other = Node(NodeType.FILTER,
                     predicate=BinOp(Col("amount"), ">=", Lit(100)))
        assert semantic_operation_id(_filter()) != semantic_operation_id(other)

    def test_a_changed_join_type_changes_it(self):
        def _join(kind):
            return Node(NodeType.JOIN, inputs=[_scan("a.parquet"),
                                               _scan("b.parquet")],
                        key_left=("k",), key_right=("k",), join_type=kind)
        assert semantic_operation_id(_join(JoinType.INNER)) != \
            semantic_operation_id(_join(JoinType.LEFT))

    def test_a_changed_sort_direction_changes_it(self):
        asc = Node(NodeType.SORT, inputs=[_scan()], sort_keys=(("v", True),))
        desc = Node(NodeType.SORT, inputs=[_scan()], sort_keys=(("v", False),))
        assert semantic_operation_id(asc) != semantic_operation_id(desc)

    def test_reordering_sort_keys_changes_it(self):
        """sort by (a, b) is not the same query as sort by (b, a)."""
        ab = Node(NodeType.SORT, sort_keys=(("a", True), ("b", True)))
        ba = Node(NodeType.SORT, sort_keys=(("b", True), ("a", True)))
        assert semantic_operation_id(ab) != semantic_operation_id(ba)

    def test_a_changed_scan_path_changes_it(self):
        assert semantic_operation_id(_scan("a.parquet")) != \
            semantic_operation_id(_scan("b.parquet"))

    def test_a_changed_scan_delimiter_changes_it(self):
        comma = _scan()
        comma.scan.delimiter = ","
        semi = _scan()
        semi.scan.delimiter = ";"
        assert semantic_operation_id(comma) != semantic_operation_id(semi)

    def test_a_changed_window_frame_changes_it(self):
        def _win(frame):
            return Node(NodeType.WINDOW, inputs=[_scan()],
                        window=WindowSpec(partition_by=("g",), frame=frame))
        assert semantic_operation_id(_win("rows between 1 preceding and "
                                          "current row")) != \
            semantic_operation_id(_win("rows between unbounded preceding and "
                                       "current row"))

    def test_a_changed_node_type_changes_it(self):
        assert semantic_operation_id(_filter()) != semantic_operation_id(
            Node(NodeType.PROJECT, columns=("a",)))

    def test_100_and_100_0_are_different_literals(self):
        """They render identically through str()."""
        ints = Node(NodeType.FILTER, predicate=BinOp(Col("v"), "=", Lit(100)))
        floats = Node(NodeType.FILTER,
                      predicate=BinOp(Col("v"), "=", Lit(100.0)))
        assert semantic_operation_id(ints) != semantic_operation_id(floats)

    def test_true_and_one_are_different_literals(self):
        """bool is an int subclass; True and 1 must not collide."""
        t = Node(NodeType.FILTER, predicate=BinOp(Col("v"), "=", Lit(True)))
        one = Node(NodeType.FILTER, predicate=BinOp(Col("v"), "=", Lit(1)))
        assert semantic_operation_id(t) != semantic_operation_id(one)

    def test_nan_infinity_and_signed_zero_are_distinguishable(self):
        """JSON cannot represent them, so they must be encoded explicitly.

        ``allow_nan=False`` means an unhandled NaN raises rather than emitting
        a token that hashes differently depending on the writer.
        """
        def _with(value):
            return Node(NodeType.FILTER,
                        predicate=BinOp(Col("v"), "=", Lit(value)))
        idents = {
            semantic_operation_id(_with(float("nan"))),
            semantic_operation_id(_with(float("inf"))),
            semantic_operation_id(_with(float("-inf"))),
            semantic_operation_id(_with(0.0)),
            semantic_operation_id(_with(-0.0)),
        }
        assert len(idents) == 5

    def test_and_is_not_reordered(self):
        """No commutativity normalisation in v1.

        ``a AND b`` and ``b AND a`` are mathematically equal, but null
        semantics, evaluation order and engine pushdown make rewriting them a
        source of subtle wrongness. Correctness first; deduplication later.
        """
        ab = Node(NodeType.FILTER, predicate=BinOp(
            BinOp(Col("a"), "=", Lit(1)), "AND", BinOp(Col("b"), "=", Lit(2))))
        ba = Node(NodeType.FILTER, predicate=BinOp(
            BinOp(Col("b"), "=", Lit(2)), "AND", BinOp(Col("a"), "=", Lit(1))))
        assert semantic_operation_id(ab) != semantic_operation_id(ba)

    def test_a_changed_write_target_changes_it(self):
        a = Node(NodeType.WRITE, target="out.parquet")
        b = Node(NodeType.WRITE, target="other.parquet")
        assert semantic_operation_id(a) != semantic_operation_id(b)

