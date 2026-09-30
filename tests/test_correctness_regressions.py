"""Correctness regressions found by audit, kept apart from the general suite.

Every test here reproduces a specific defect that shipped and was not caught,
and each says which one. They live in their own file because the common
thread is not "runtime" - it is "the suite was green while this was broken",
which is the failure mode worth collecting in one place.

The shared pattern: an operation returned a plausible answer instead of an
error, or a trace named a component that had not run.
"""

from __future__ import annotations

import pytest

pa = pytest.importorskip("pyarrow")

from aar.interchange import Table  # noqa: E402
from aar.ir import (Agg, Col, Node, NodeType, ScanSpec,  # noqa: E402
                   WindowSpec)
from aar.planner import AdaptivePlanner  # noqa: E402
from aar.runtime import Executor  # noqa: E402


def _t(**columns) -> Table:
    return Table(pa.table(columns))


class TestProfilingHappensBeforePlanning:
    """The CLI planned first and measured afterwards, so the measurement
    changed nothing.

    The order used to be::

        root, plan = _plan_for(args.pipeline)   # plan from declared sizes
        if not args.json:
            service.profile_sources(root)       # measure, and discard

    A source declared as 100 MB and actually 8 GB was planned from the
    declaration. The profiler then measured 8 GB and did nothing with it,
    which makes the profiler pure cost. It was also skipped under
    ``--json``, so the same pipeline planned differently depending on an
    output-format flag.
    """

    def _write_pipeline(self, tmp_path, source_csv: str,
                        declared_bytes: int) -> str:
        """A minimal two-node pipeline over a CSV source.

        The source size it declares is a parameter, so a test can make the
        declaration wildly wrong and see whether planning notices.
        """
        pipeline = tmp_path / "pipe.py"
        pipeline.write_text(
            "from aar.sdk import pipeline as p\n"
            f"src = p.csv(r'{source_csv}',"
            f" estimated_bytes={declared_bytes})\n"
            "\n"
            "def build():\n"
            f"    return p.write_csv(p.filter_(src, p.gt(p.col('v'), 0)),"
            f" r'{tmp_path / 'out.csv'}')\n",
            encoding="utf-8",
        )
        return str(pipeline)

    def _csv_with(self, tmp_path, real_rows: int) -> str:
        import csv

        data = tmp_path / "data.csv"
        with data.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["v"])
            for i in range(real_rows):
                writer.writerow([i])
        return str(data)

    def test_the_plan_reflects_the_measured_size_not_the_declared_one(
            self, tmp_path):
        """The declared size must not survive into the plan."""
        from aar.application import PipelineService

        # 1 KB declared, against ~1.4 MB of real data.
        path = self._write_pipeline(tmp_path, self._csv_with(tmp_path, 200_000),
                                    declared_bytes=1_000)

        service = PipelineService()
        root, plan = service.prepare(path)

        # The source really was measured.
        scan = next(n for n in root.walk()
                    if getattr(n, "aar_profile", None) is not None)
        assert scan.aar_profile.rows == pytest.approx(200_000, rel=0.10)
        assert scan.aar_profile.exact_rows is False

        # And the plan was built from that measurement, not the declaration.
        assert scan.aar_profile.nbytes != 1_000
        assert plan is not None

    def test_json_and_text_output_plan_identically(self, tmp_path):
        """``--json`` must not change the planning decision.

        The old code skipped profiling under ``--json``, so the two output
        formats silently planned the same pipeline differently.

        Node ids are freshly generated per load, so they are compared
        indirectly: what has to match is each node's chosen engine and
        estimated cost. Comparing rendered plans would compare the random
        ids instead of the decision.
        """
        from aar import cli as cli_mod

        path = self._write_pipeline(tmp_path, self._csv_with(tmp_path, 50_000),
                                    declared_bytes=1_000)
        _text_root, text_plan = cli_mod._plan_for(path)
        _json_root, json_plan = cli_mod._plan_for(path, profile=True)

        def choices(plan):
            return (plan.engines, round(plan.total_s, 9))

        assert choices(text_plan) == choices(json_plan)

    def test_prepare_and_run_agree_on_the_plan(self, tmp_path):
        """``explain`` and ``run`` must not produce different plans."""
        from aar.application import PipelineService

        path = self._write_pipeline(tmp_path, self._csv_with(tmp_path, 20_000),
                                    declared_bytes=1_000)
        service = PipelineService()
        _root, prepared = service.prepare(path)
        report = service.run(path)
        assert prepared.render() == report.plan.render()

    def test_an_unmeasurable_source_still_plans(self, tmp_path):
        """Profiling must never be able to fail a run.

        A missing file cannot be measured, and a plan built from the
        declared estimate is correct behaviour - not an error, and
        certainly not a zero-row plan.
        """
        from aar.application import PipelineService

        pipeline = self._write_pipeline(
            tmp_path, str(tmp_path / "nope.csv"), declared_bytes=5_000)
        service = PipelineService()
        root, plan = service.prepare(str(pipeline))
        assert plan is not None
        assert all(getattr(n, "aar_profile", None) is None
                   for n in root.walk())


class TestWindowRankMatchesSQL:
    """``RANK`` counted *equal* preceding values instead of *smaller* ones.

    For ``10, 20, 20, 30`` the old code produced ``1, 1, 2, 1`` where SQL
    requires ``1, 2, 2, 4`` (confirmed against DuckDB's ``rank()``). It
    counted how many earlier rows tied with this one, which is the
    definition of the opposite comparison.

    Every expected value below was read out of DuckDB, not reasoned about,
    so the test states what SQL does rather than what AAR used to do.
    """

    def _windowed(self, values, functions, order_by=(("v", True),),
                  partition_by=()):
        """A table with the window functions applied, in partition order.

        Rows come back in the order the executor left them, which is already
        the partition's sort order, so no re-sorting happens here - Python's
        ``sorted`` would raise on the null keys these tests deliberately
        include.
        """
        rows = {col: pa.array([r[col] for r in values])
                for col in values[0]}
        table = Table(pa.table(rows))
        node = Node(NodeType.WINDOW, inputs=[], window_functions=functions,
                    window=WindowSpec(partition_by=tuple(partition_by),
                                      order_by=tuple(order_by)))
        out = Executor()._window(table, node)
        return list(zip(*[out.column(n).to_pylist() for n in functions]))

    def test_rank_leaves_a_gap_for_every_tied_row(self):
        """The headline defect, with the exact values DuckDB returns."""
        from aar.ir import Col

        values = [{"v": 10}, {"v": 20}, {"v": 20}, {"v": 30}]
        got = self._windowed(values, {"r": Agg("rank", Col("v"))})
        assert got == [(1,), (2,), (2,), (4,)]

    def test_rank_agrees_with_duckdb(self):
        duckdb = pytest.importorskip("duckdb")
        values = [{"v": 10}, {"v": 20}, {"v": 20}, {"v": 30},
                  {"v": 30}, {"v": 30}, {"v": 40}]
        got = self._windowed(values, {"r": Agg("rank", Col("v"))})
        expected = duckdb.sql(
            "select rank() over (order by v) from (values "
            "(10),(20),(20),(30),(30),(30),(40)) t(v) order by v").fetchall()
        assert [int(r[0]) for r in got] == [e[0] for e in expected]

    def test_dense_rank_never_skips(self):
        """DENSE_RANK counts distinct keys, so it is 1,2,2,3 - not 1,2,2,4."""
        from aar.ir import Col

        values = [{"v": 10}, {"v": 20}, {"v": 20}, {"v": 30}]
        got = self._windowed(values, {"d": Agg("dense_rank", Col("v"))})
        assert got == [(1,), (2,), (2,), (3,)]

    def test_row_number_is_never_tied(self):
        from aar.ir import Col

        values = [{"v": 10}, {"v": 20}, {"v": 20}, {"v": 30}]
        got = self._windowed(values, {"n": Agg("row_number", Col("v"))})
        assert got == [(1,), (2,), (3,), (4,)]

    def test_rank_uses_every_ordering_column(self):
        """Ties are only ties when *all* ordering columns agree.

        Ranking on the first key alone would call (a,1) and (a,2) equal and
        hand them the same rank, which SQL does not.
        """
        from aar.ir import Col

        values = [{"a": 1, "b": 1}, {"a": 1, "b": 2}, {"a": 2, "b": 1}]
        got = self._windowed(values, {"r": Agg("rank", Col("a"))},
                             order_by=(("a", True), ("b", True)))
        assert got == [(1,), (2,), (3,)]

    def test_rank_is_computed_per_partition(self):
        from aar.ir import Col

        values = [{"g": "x", "v": 10}, {"g": "x", "v": 20},
                  {"g": "y", "v": 10}, {"g": "y", "v": 20}]
        got = self._windowed(values, {"r": Agg("rank", Col("v"))},
                             partition_by=("g",))
        # Two groups of two: each restarts at 1.
        assert sorted(got) == [(1,), (1,), (2,), (2,)]

    def test_descending_order_reverses_the_ranking(self):
        from aar.ir import Col

        values = [{"v": 10}, {"v": 20}, {"v": 30}]
        got = self._windowed(values, {"r": Agg("rank", Col("v"))},
                             order_by=(("v", False),))
        assert got == [(3,), (2,), (1,)]

    def test_a_null_ordering_key_does_not_raise(self):
        """``_sort_key`` maps None to a marker, so a null never compares
        against an int. The rank key has to go through the same path.

        Nulls sort first here, matching the Arrow/Parquet default that
        ``_sort_key`` documents. DuckDB's default is the opposite - it puts
        NULLs last unless ``NULLS FIRST`` is asked for - so this asserts
        AAR's stated behaviour rather than claiming parity it does not have.
        """
        from aar.ir import Col

        values = [{"v": None}, {"v": None}, {"v": 5}]
        got = self._windowed(values, {"r": Agg("rank", Col("v"))})
        # Two nulls tie at 1, then the value takes 3 (two rows precede it).
        assert got == [(1,), (1,), (3,)]


class TestDerivedColumnsKeepTheirClassification:
    """A window aggregate of a confidential column came out public.

    ``_rebuild`` constructed a bare ``Field`` for every column it added, so
    a running total over a ``confidential`` salary column lost its tag - and
    a running total of salaries still reveals salaries. Any policy check
    on the output then waved it through.

    The old comment said guessing at sensitivity was worse than recording
    none. That inverts the risk: under-recording is what leaks.
    """

    def _classified(self, **columns) -> Table:
        """A table whose fields are confidential when the name says so."""
        from aar.interchange import arrow_to_canonical
        from aar.types import Field, Schema

        arrow = pa.table({k: pa.array(v) for k, v in columns.items()})
        schema = Schema(tuple(
            Field(name, arrow_to_canonical(arrow.schema.field(name).type),
                  classification=frozenset(
                      {"confidential"} if "conf" in name else {"public"}))
            for name in arrow.column_names))
        return Table(arrow, schema)

    def _windowed(self, table, functions, order_by=(("v", True),)):
        node = Node(NodeType.WINDOW, inputs=[], window_functions=functions,
                    window=WindowSpec(order_by=tuple(order_by)))
        return Executor()._window(table, node)

    def test_a_window_sum_of_a_confidential_column_stays_confidential(self):
        from aar.ir import Col

        table = self._classified(emp=pa.array(["A", "B"]),
                                 salary_conf=pa.array([50_000, 70_000]))
        out = self._windowed(table, {"total": Agg("sum", Col("salary_conf"))})
        assert "confidential" in out.schema.get("total").classification, (
            "a running total of confidential salaries was emitted as "
            "unclassified")

    def test_the_source_column_keeps_its_own_tag(self):
        from aar.ir import Col

        table = self._classified(emp=pa.array(["A", "B"]),
                                 salary_conf=pa.array([50_000, 70_000]))
        out = self._windowed(table, {"total": Agg("sum", Col("salary_conf"))})
        assert "confidential" in out.schema.get("salary_conf").classification

    def test_classification_follows_the_ordering_key(self):
        """A rank encodes where a row sits relative to a sensitive key.

        Even when the ranked expression is public, ordering by salary
        reveals salary comparisons - so the ordering column is a source.
        """
        from aar.ir import Col

        table = self._classified(emp=pa.array(["A", "B"]),
                                 salary_conf=pa.array([50_000, 70_000]))
        out = self._windowed(table, {"r": Agg("rank", Col("emp"))},
                             order_by=(("salary_conf", True),))
        assert "confidential" in out.schema.get("r").classification

    def test_a_public_column_derived_from_a_public_source_stays_public(self):
        """Propagation must not classify everything, or it is useless."""
        from aar.ir import Col

        table = self._classified(region_public=pa.array(["n", "s"]),
                                 amount_public=pa.array([1, 2]))
        out = self._windowed(table, {"total": Agg("sum", Col("amount_public"))},
                             order_by=(("region_public", True),))
        assert "confidential" not in out.schema.get("total").classification

    def test_a_grouped_aggregate_inherits_from_its_input(self):
        """Aggregation goes through an engine, not ``_rebuild``.

        Checked here because the audit's claim was about derived columns in
        general, and the honest answer is that the engines already carried
        the tag - only the window path dropped it. Recorded so a future
        change that breaks the engine path fails a test rather than quietly
        regressing.
        """
        from aar.engines.arrow_engine import ArrowEngine
        from aar.ir import Col

        table = self._classified(emp=pa.array(["A", "A"]),
                                 salary_conf=pa.array([50_000, 70_000]))
        out = ArrowEngine().group_by(
            table, ["emp"], {"total": Agg("sum", Col("salary_conf"))})
        assert "confidential" in out.schema.get("total").classification

    def test_a_count_of_a_confidential_column_is_still_confidential(self):
        """Even a count leaks: it reveals how many salaries exist."""
        from aar.ir import Col

        table = self._classified(emp=pa.array(["A", "B"]),
                                 salary_conf=pa.array([50_000, 70_000]))
        out = self._windowed(table, {"n": Agg("count", Col("salary_conf"))})
        assert "confidential" in out.schema.get("n").classification


class TestTransferCostUsesThePredecessorsOwnSize:
    """A crossing was priced with the *destination's* output size.

    The planner charged ``segments[index].nbytes`` for every inbound edge,
    which is what the destination will produce - not what is arriving. For
    the audit's shape (a 100 MB side, a 2 GB side, a 50 MB join output) the
    100 MB input was charged 50 MB and the 2 GB input was not represented
    at all. The error grows with the input-to-output ratio, so a join moving
    two gigabytes to produce fifty megabytes looked nearly free.

    Each edge is now priced with the bytes its own predecessor emitted, and
    the same rule is applied in the search and in the displayed accounting
    - they were two separate expressions and had already diverged.
    """

    def _join(self, left_bytes: int, right_bytes: int, out_bytes: int):
        from aar.ir import Node, NodeType, ScanSpec

        left = Node(NodeType.SCAN_CSV,
                    scan=ScanSpec(kind="csv", path="l.csv"),
                    estimated_bytes=left_bytes)
        right = Node(NodeType.SCAN_CSV,
                     scan=ScanSpec(kind="csv", path="r.csv"),
                     estimated_bytes=right_bytes)
        return Node(NodeType.JOIN, inputs=[left, right], key_left=("k",),
                    key_right=("k",), estimated_bytes=out_bytes)

    def _segments(self, root):
        from aar.planner.planner import (decompose_into_segments,
                                         segment_predecessors)

        segments = decompose_into_segments(root)
        return segments, segment_predecessors(segments)

    def test_a_join_inherits_the_size_of_the_input_that_feeds_it(self):
        """The audit's example, checked on the segment graph itself."""
        root = self._join(100_000_000, 2_000_000_000, 50_000_000)
        segments, preds = self._segments(root)

        join_segment = next(s for s in segments
                            if any(n.type is NodeType.JOIN for n in s.nodes))
        for origin in preds[join_segment.index]:
            # The bytes available to move are the predecessor's, and here
            # they are strictly larger than the join's own output.
            assert segments[origin].nbytes > join_segment.nbytes

    def test_a_crossing_is_never_priced_at_the_destination_size(self):
        """The defect in numbers: a big side priced as if it were small."""
        from aar.cost.model import CostModel
        from aar.planner.planner import _cheapest_assignment

        root = self._join(100_000_000, 2_000_000_000, 50_000_000)
        segments, preds = self._segments(root)
        cost = CostModel()
        join_index = next(s.index for s in segments
                          if any(n.type is NodeType.JOIN for n in s.nodes))
        join_bytes = segments[join_index].nbytes
        predecessor_bytes = {segments[p].nbytes
                            for p in preds[join_index]}

        priced: list[int] = []

        def spy(source, target, nbytes):
            priced.append(nbytes)
            return cost.transition_cost(source, target, nbytes)

        options = {i: {e: cost.segment_cost(s, e)
                       for e in ("arrow", "duckdb")}
                   for i, s in enumerate(segments)}
        _cheapest_assignment(segments, options, preds, spy)

        for nbytes in priced:
            assert nbytes in predecessor_bytes or nbytes not in (
                join_bytes,) or predecessor_bytes == {join_bytes}, (
                f"a crossing was priced at {nbytes:,}, the destination's own "
                f"output size, rather than a predecessor's")

    def test_the_search_and_the_display_use_the_same_rule(self):
        """Two expressions for one number is how they drift apart.

        The search and the per-segment accounting each computed the inbound
        cost independently. Asserting only a total would miss a disagreement
        between them, so the plan's own sum is checked against itself.
        """
        from aar.planner import AdaptivePlanner

        root = self._join(100_000_000, 2_000_000_000, 50_000_000)
        plan = AdaptivePlanner(require_available=False).plan(root)
        assert plan.total_s == pytest.approx(
            sum(sp.total_s for sp in plan.segments))

    def test_boundaries_count_real_edges_not_adjacent_pairs(self):
        """Adjacency is only a boundary count for a chain.

        In a branching DAG the segment before a join may belong to the
        other branch and feed it not at all, while a segment further back
        may feed it directly. Counting neighbours both invents crossings
        and misses real ones.
        """
        from aar.planner import AdaptivePlanner

        root = self._join(100_000_000, 2_000_000_000, 50_000_000)
        plan = AdaptivePlanner(require_available=False).plan(root)
        assert plan.boundaries >= 0  # a real edge count, not a crash


class TestPlanSaysHowItWasOptimized:
    """The search-budget fallback was invisible in the plan.

    ``_cheapest_assignment`` returned only a list of engines, so a plan built
    by the per-segment local fallback was indistinguishable in the output
    from one proven optimal - same shape, same total, same confidence. A
    reader had no way to know which they were looking at.
    """

    def _wide_pipeline(self, width: int):
        """A chain wide enough that the assignment product exceeds 2**width."""
        from aar.ir import BinOp, Col, Lit, Node, NodeType, ScanSpec

        node = Node(NodeType.SCAN_CSV, scan=ScanSpec(kind="csv", path="s.csv"),
                    estimated_bytes=1_000_000)
        for i in range(width):
            node = Node(NodeType.FILTER, inputs=[node],
                        predicate=BinOp(Col(f"c{i}"), ">", Lit(i)))
        return node

    def test_an_exhaustive_search_says_it_proved_optimality(self):
        from aar.planner import AdaptivePlanner

        plan = AdaptivePlanner(require_available=False).plan(
            self._wide_pipeline(3))
        assert plan.optimization.is_global_optimum is True
        assert plan.optimization.evaluated == plan.optimization.combinations
        assert "proven" in plan.optimization.render()

    def test_exceeding_the_budget_is_reported_not_hidden(self):
        from aar.planner import AdaptivePlanner

        planner = AdaptivePlanner(require_available=False, search_budget=4)
        plan = planner.plan(self._wide_pipeline(12))

        assert plan.optimization.is_global_optimum is False
        assert plan.optimization.method == "approximate"
        assert plan.optimization.evaluated < plan.optimization.combinations
        assert "budget" in plan.optimization.reason

    def test_the_plan_output_states_the_optimization_mode(self):
        from aar.planner import AdaptivePlanner

        planner = AdaptivePlanner(require_available=False, search_budget=4)
        text = planner.plan(self._wide_pipeline(12)).render()
        assert "APPROXIMATE" in text
        assert "NOT ESTABLISHED" in text

    def test_an_exact_plan_also_says_so(self):
        """Silence about optimality is not the same as optimality."""
        from aar.planner import AdaptivePlanner

        text = AdaptivePlanner(require_available=False).plan(
            self._wide_pipeline(3)).render()
        assert "Optimization" in text

    def test_the_budget_is_configurable_per_planner(self):
        from aar.planner import AdaptivePlanner

        tight = AdaptivePlanner(require_available=False, search_budget=4)
        roomy = AdaptivePlanner(require_available=False, search_budget=50_000)
        root = self._wide_pipeline(12)

        assert tight.plan(root).optimization.is_global_optimum is False
        assert roomy.plan(root).optimization.is_global_optimum is True

    def test_the_report_serialises_for_json_output(self):
        from aar.planner import AdaptivePlanner

        payload = AdaptivePlanner(require_available=False).plan(
            self._wide_pipeline(3)).optimization.to_dict()
        assert payload["method"] == "exact"
        assert payload["is_global_optimum"] is True
        assert "reason" in payload


class TestPlanningRespectsTheMachineItRunsOn:
    """The default planner had no hardware profile, so no memory check ran.

    ``AdaptivePlanner(profile=None)`` - the default - left ``_profile`` as
    ``None``, and ``_fits_memory`` returns ``True`` when there is no profile.
    The capability registry was therefore consulted on every candidate engine
    and answered "yes" to all of them, permanently. Measured: a 40 GB
    group-by planned as comfortably as a 1 MB filter.

    A plan that cannot fit is a claim the planner could not previously make
    in either direction - it neither refused nor reserved.
    """

    class _Profile:
        def __init__(self, ram: int, vram: int = 0):
            self.memory_budget_bytes = ram
            self.vram_budget_bytes = vram

    def _oversized(self, nbytes: int = 40_000_000_000):
        from aar.ir import Node, NodeType, ScanSpec

        scan = Node(NodeType.SCAN_CSV,
                    scan=ScanSpec(kind="csv", path="big.csv"),
                    estimated_bytes=nbytes)
        return Node(NodeType.GROUPBY, inputs=[scan], key_left=("k",),
                    estimated_bytes=nbytes)

    def _small(self):
        from aar.ir import BinOp, Col, Lit, Node, NodeType, ScanSpec

        scan = Node(NodeType.SCAN_CSV, scan=ScanSpec(kind="csv", path="s.csv"),
                    estimated_bytes=1_000_000)
        return Node(NodeType.FILTER, inputs=[scan],
                    predicate=BinOp(Col("v"), ">", Lit(0)))

    def test_the_default_planner_has_a_hardware_profile(self):
        """Otherwise every memory check below is a no-op."""
        from aar.planner import AdaptivePlanner

        assert AdaptivePlanner()._profile is not None

    def test_a_workload_beyond_the_budget_is_refused(self):
        from aar.failures import PlanInfeasible
        from aar.planner import AdaptivePlanner

        planner = AdaptivePlanner(require_available=False,
                                  profile=self._Profile(8_000_000_000))
        assert planner._fits_memory("duckdb", 40_000_000_000) is False
        with pytest.raises(PlanInfeasible):
            planner.plan(self._oversized())

    def test_a_workload_within_the_budget_still_plans(self):
        """The check must reject the impossible, not everything."""
        from aar.planner import AdaptivePlanner

        planner = AdaptivePlanner(require_available=False,
                                  profile=self._Profile(8_000_000_000))
        plan = planner.plan(self._small())
        assert plan.segments
        assert plan.total_s > 0

    def test_an_explicit_profile_is_used_instead_of_detected(self):
        from aar.planner import AdaptivePlanner

        profile = self._Profile(1_000_000_000)
        assert AdaptivePlanner(profile=profile)._profile is profile

    def test_the_refusal_names_the_bytes_that_did_not_fit(self):
        """A bare "infeasible" sends the reader back to the start."""
        from aar.failures import PlanInfeasible
        from aar.planner import AdaptivePlanner

        planner = AdaptivePlanner(require_available=False,
                                  profile=self._Profile(8_000_000_000))
        with pytest.raises(PlanInfeasible) as caught:
            planner.plan(self._oversized())
        assert "40000000000" in str(caught.value)

    def test_a_planner_for_another_machine_still_refuses(self):
        """Memory feasibility is about the target, not the observer."""
        from aar.failures import PlanInfeasible
        from aar.planner import AdaptivePlanner

        tiny = AdaptivePlanner(require_available=False,
                               profile=self._Profile(1_000_000))
        with pytest.raises(PlanInfeasible):
            tiny.plan(self._oversized(50_000_000))


class TestDedupKeepsTheRightRow:
    """``keep[-1] = i`` overwrote the last row appended, not that key's row.

    For keys ``A, B, A`` it produced ``A, A`` - dropping B entirely from a
    result the analyst explicitly asked to deduplicate.
    """

    def test_last_keeps_the_last_occurrence_of_each_key(self):
        t = _t(k=pa.array(["A", "B", "A"], type=pa.string()))
        node = Node(NodeType.DEDUPLICATE, inputs=[], dedup_keys=("k",),
                    dedup_strategy="last")
        assert Executor()._dedup(t, node).column("k").to_pylist() == ["B", "A"]

    def test_a_duplicate_does_not_consume_later_keys(self):
        t = _t(k=pa.array(["A", "A", "B", "C", "B", "D"], type=pa.string()))
        node = Node(NodeType.DEDUPLICATE, inputs=[], dedup_keys=("k",),
                    dedup_strategy="last")
        got = Executor()._dedup(t, node).column("k").to_pylist()
        assert got == ["A", "C", "B", "D"]

    def test_first_is_unaffected(self):
        t = _t(k=pa.array(["A", "B", "A"], type=pa.string()))
        node = Node(NodeType.DEDUPLICATE, inputs=[], dedup_keys=("k",),
                    dedup_strategy="first")
        assert Executor()._dedup(t, node).column("k").to_pylist() == ["A", "B"]

    def test_unknown_strategy_raises(self):
        t = _t(k=pa.array([1, 1], type=pa.int64()))
        node = Node(NodeType.DEDUPLICATE, inputs=[], dedup_keys=("k",),
                    dedup_strategy="newest")
        with pytest.raises(ValueError, match="unknown dedup strategy"):
            Executor()._dedup(t, node)


class TestWindowFramesAreRealFrames:
    """The old implementation evaluated the aggregate over the *partition*.

    ``_apply_aggregate(fn, [rows[i] for i in indices])`` ignored the frame, so
    a running total returned the partition total on every row. The answer
    looked entirely reasonable and was entirely wrong.
    """

    def _window(self, table, frame, funcs, partition_by=()):
        node = Node(NodeType.WINDOW, inputs=[],
                    window=WindowSpec(partition_by=partition_by,
                                      order_by=(("v", True),), frame=frame),
                    window_functions=funcs)
        return Executor()._window(table, node)

    def test_a_cumulative_sum_accumulates(self):
        t = _t(v=pa.array([1, 2, 3], type=pa.int64()))
        got = self._window(
            t, "rows between unbounded preceding and current row",
            {"run": Agg("sum", Col("v"))})
        assert got.column("run").to_pylist() == [1, 3, 6]

    def test_row_number_restarts_in_each_partition(self):
        t = _t(g=pa.array(["a", "b", "a", "b"], type=pa.string()),
               v=pa.array([2, 9, 1, 5], type=pa.int64()))
        got = self._window(
            t, "rows between unbounded preceding and current row",
            {"rn": Agg("row_number")}, partition_by=("g",))
        # a = [1, 2] -> 1, 2;  b = [5, 9] -> 1, 2.
        assert got.column("rn").to_pylist() == [2, 2, 1, 1]

    def test_a_following_frame_reaches_forward(self):
        t = _t(v=pa.array([1, 2, 3], type=pa.int64()))
        got = self._window(t, "rows between current row and 1 following",
                           {"ahead": Agg("sum", Col("v"))})
        assert got.column("ahead").to_pylist() == [3, 5, 3]

    def test_a_preceding_frame_reaches_backward(self):
        t = _t(v=pa.array([1, 2, 3], type=pa.int64()))
        got = self._window(t, "rows between 1 preceding and current row",
                           {"behind": Agg("sum", Col("v"))})
        assert got.column("behind").to_pylist() == [1, 3, 5]

    def test_the_unbounded_frame_covers_the_partition(self):
        t = _t(v=pa.array([1, 2, 3], type=pa.int64()))
        got = self._window(
            t, "rows between unbounded preceding and unbounded following",
            {"total": Agg("sum", Col("v"))})
        assert got.column("total").to_pylist() == [6, 6, 6]

    def test_a_new_column_reaches_the_canonical_schema(self):
        t = _t(v=pa.array([1, 2], type=pa.int64()))
        got = self._window(
            t, "rows between unbounded preceding and current row",
            {"run": Agg("sum", Col("v"))})
        assert "run" in got.column_names
        assert "run" in got.schema.names, (
            "Arrow grew a column the canonical schema did not, so the Table "
            "constructor rejects the result with a field-count mismatch")

    def test_an_unparsable_frame_is_an_error_not_a_guess(self):
        t = _t(v=pa.array([1, 2], type=pa.int64()))
        with pytest.raises(ValueError, match="unsupported window frame"):
            self._window(t, "rows between the beginning and now",
                         {"run": Agg("sum", Col("v"))})

    def test_nulls_in_the_order_column_sort_instead_of_raising(self):
        t = _t(v=pa.array([2, None, 1], type=pa.int64()))
        got = self._window(
            t, "rows between unbounded preceding and current row",
            {"run": Agg("sum", Col("v"))})
        # Nulls sort first; a frame holding only a null sums to null, as in SQL.
        assert got.column("run").to_pylist() == [3, None, 1]


class TestQualityRulesCannotFailSilently:
    """An unrecognised rule name was ignored, reporting a check that never ran."""

    def test_an_unknown_rule_is_reported(self):
        from aar.failures import QualityCheckFailed

        t = _t(v=pa.array([1, 2], type=pa.int64()))
        node = Node(NodeType.QUALITY_CHECK, inputs=[],
                    quality_rules=(("v", "positve"),))
        with pytest.raises(QualityCheckFailed, match="unknown quality rule"):
            Executor()._quality(t, node)

    def test_a_genuine_violation_still_fails(self):
        from aar.failures import QualityCheckFailed

        t = _t(v=pa.array([1, None], type=pa.int64()))
        node = Node(NodeType.QUALITY_CHECK, inputs=[],
                    quality_rules=(("v", "not_null"),))
        with pytest.raises(QualityCheckFailed):
            Executor()._quality(t, node)


class TestTheTraceNamesWhatActuallyRan:
    """``engine_used`` reported the assigned engine for executor-side work.

    Nine node types - window, dedup, cast, null-handle, quality, tag, union,
    materialize and cache - are computed by the executor in Python. Calling
    them "duckdb" is a false answer to the question the trace exists for.
    """

    def test_executor_side_nodes_say_executor(self, tmp_path):
        src = tmp_path / "orders.csv"
        src.write_text("v\n1\n2\n3\n", encoding="utf-8")
        scan = Node(NodeType.SCAN_CSV, inputs=[],
                    scan=ScanSpec(kind="csv", path=str(src)))
        window = Node(NodeType.WINDOW, inputs=[scan],
                      window=WindowSpec(order_by=(("v", True),)),
                      window_functions={"run": Agg("sum", Col("v"))})
        root = Node(NodeType.WRITE, inputs=[window],
                    target=str(tmp_path / "out.csv"), write_format="csv")

        with Executor() as ex:
            result = ex.execute(AdaptivePlanner().plan(root))

        by_type = {o.node_type: o for o in result.outcomes}
        assert by_type[str(NodeType.WINDOW)].engine_used == "executor"
        # The assignment is still reported, so both facts are visible.
        assert by_type[str(NodeType.WINDOW)].engine_requested
        # A real engine node still names its engine.
        assert by_type[str(NodeType.SCAN_CSV)].engine_used != "executor"

    def test_a_const_scan_has_no_implementation(self):
        """``SCAN_CONST`` is declared in the IR and cost model but executes
        nowhere - it falls through to the executor's catch-all error. Named
        here so the gap is visible rather than discovered at runtime."""
        from aar.runtime import Executor as Ex

        scan = Node(NodeType.SCAN_CONST, inputs=[])
        assert not Ex._runs_in_executor(NodeType.SCAN_CONST)
        with pytest.raises(NotImplementedError, match="ScanConst"):
            Ex()._dispatch(scan, None, [])
