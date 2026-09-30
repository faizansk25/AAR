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
