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