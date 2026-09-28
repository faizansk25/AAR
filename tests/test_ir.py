"""IR: node construction, expression rendering, DAG ordering."""

from __future__ import annotations

import pytest

from aar.ir import (Agg, BinOp, Col, JoinType, Lit, Node, NodeType, Privacy,
                    ScanSpec, topological_order)
from aar.types import INT32, INT64, UTF8, Field, Schema


def _scan(name: str = "t") -> Node:
    return Node(NodeType.SCAN_PARQUET,
                scan=ScanSpec(kind="parquet", path=f"{name}.parquet"),
                output_schema=Schema((Field("id", INT64), Field("v", INT32))))


class TestExpressions:
    def test_column_quoting_per_dialect(self):
        assert Col("amount").to_sql("duckdb") == '"amount"'
        assert Col("amount").to_sql("mysql") == "`amount`"

    def test_embedded_quote_is_escaped(self):
        assert Col('a"b').to_sql("duckdb") == '"a""b"'

    def test_string_literal_escapes_single_quote(self):
        assert Lit("O'Brien").to_sql() == "'O''Brien'"

    def test_string_literal_is_not_injectable(self):
        # A value containing SQL must be quoted, not executed. The literal is
        # rendered as a quoted string, which is what makes passing user data
        # through a pushdown safe.
        rendered = Lit("'; DROP TABLE users; --").to_sql()
        assert rendered == "'''; DROP TABLE users; --'"
        assert rendered.count("'") % 2 == 0

    def test_bool_literal(self):
        assert Lit(True).to_sql() == "TRUE"
        assert Lit(False).to_sql() == "FALSE"

    def test_null_literal(self):
        assert Lit(None).to_sql() == "NULL"

    def test_binop_parenthesised(self):
        e = BinOp(Col("a"), ">", Lit(5))
        assert e.to_sql() == '("a" > 5)'

    def test_columns_dependency_set(self):
        e = BinOp(Col("a"), "+", Col("b"))
        assert e.columns() == frozenset({"a", "b"})

    def test_agg_distinct(self):
        assert Agg("count", Col("x"), distinct=True).to_sql() == 'count(DISTINCT "x")'
        assert Agg("count").to_sql() == "count(*)"


class TestNodeMetadata:
    def test_source_and_sink_classification(self):
        scan = _scan()
        write = Node(NodeType.WRITE, inputs=[scan], target="out.parquet")
        assert scan.is_source and not scan.is_sink
        assert write.is_sink and not write.is_source

    def test_engine_assignment_is_recorded(self):
        n = _scan()
        assert n.assigned_engine is None
        n.assigned_engine = "duckdb"
        n.reason = "projection pushdown"
        assert "duckdb" in repr(n)

    def test_privacy_coercion_and_ordering(self):
        assert Privacy.coerce("confidential") is Privacy.CONFIDENTIAL
        assert Privacy.coerce(None) is Privacy.INTERNAL
        assert Privacy.PUBLIC < Privacy.RESTRICTED
        assert Privacy.RESTRICTED.rank == 3

    def test_invalid_privacy_rejected(self):
        with pytest.raises(Exception):
            Privacy.coerce("top-secret")

    def test_describe_is_human_readable(self):
        n = Node(NodeType.FILTER, inputs=[_scan()],
                 predicate=BinOp(Col("v"), ">", Lit(1)))
        assert "filter" in n.describe()

    def test_join_describe(self):
        a, b = _scan("a"), _scan("b")
        j = Node(NodeType.JOIN, inputs=[a, b], key_left=("id",),
                 key_right=("id",), join_type=JoinType.LEFT)
        assert "left join" in j.describe()

    def test_walk_yields_ancestors_once(self):
        a = _scan("a")
        b = _scan("b")
        j = Node(NodeType.JOIN, inputs=[a, b])
        top = Node(NodeType.AGGREGATE, inputs=[j])
        assert len(list(top.walk())) == 4


class TestTopologicalOrder:
    def test_children_precede_parents(self):
        a = _scan("a")
        b = _scan("b")
        j = Node(NodeType.JOIN, inputs=[a, b])
        top = Node(NodeType.WRITE, inputs=[j], target="out")
        order = topological_order(top)
        ids = [n.id for n in order]
        assert ids.index(j.id) < ids.index(top.id)
        assert ids.index(a.id) < ids.index(j.id)
        assert ids.index(b.id) < ids.index(j.id)

    def test_deterministic_across_runs(self):
        a, b = _scan("a"), _scan("b")
        j = Node(NodeType.JOIN, inputs=[a, b])
        top = Node(NodeType.WRITE, inputs=[j], target="out")
        first = [n.id for n in topological_order(top)]
        second = [n.id for n in topological_order(top)]
        assert first == second

    def test_cycle_is_detected(self):
        n = Node(NodeType.FILTER, inputs=[], predicate=BinOp(Col("a"), ">", Lit(1)))
        n.inputs.append(n)  # self-loop
        with pytest.raises(ValueError, match="cycle"):
            topological_order(n)

    def test_unreachable_branch_still_included(self):
        a = _scan("a")
        orphan = _scan("orphan")
        top = Node(NodeType.WRITE, inputs=[a], target="out")
        order = topological_order([top, orphan])
        assert any(n.id == orphan.id for n in order)
