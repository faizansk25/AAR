"""SQL and MongoDB connectors.

Two very different levels of verification, and the difference is stated in
each test's name rather than glossed:

* **SQLite is a real database.** It is in the standard library, so these
  tests execute real SQL against a real database file. A wrong pushdown
  returns the wrong rows and fails here.
* **MongoDB runs against ``mongomock``**, which implements the query,
  projection and aggregation semantics in Python. That genuinely verifies
  AAR's translation and typing - the part AAR owns - and says nothing about
  the wire protocol.
"""

from __future__ import annotations

import sqlite3

import pytest

pa = pytest.importorskip("pyarrow")

from aar.connectors import (  # noqa: E402
    MYSQL, POSTGRESQL, SQLITE, for_dialect, mock_mongo_connector,
    mongo_type_of, mysql_connector, postgresql_connector, quote_ident, quote_value, render_match, sqlite_connector,
    sqlite_type_name,
)
from aar.ir import (  # noqa: E402
    BinOp, Col, Lit, Node, NodeType, ScanSpec,
)
from aar.types import FLOAT64, INT64, TIMESTAMP  # noqa: E402


@pytest.fixture()
def db_path(tmp_path):
    """A real SQLite database file, created for real."""
    path = str(tmp_path / "orders.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE orders ("
                 "  id INTEGER, region TEXT, amount REAL, quantity INTEGER)")
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?)",
        [(1, "NA", 100.0, 1), (2, "EU", 200.0, 2), (3, "APAC", 300.0, 3),
         (4, "NA", 400.0, 4), (5, "EU", 500.0, 5)])
    conn.commit()
    conn.close()
    return path


def _scan(path, table="orders", **kwargs):
    return Node(NodeType.SCAN_SQL, scan=ScanSpec(
        kind="sql", path=path, table_name=table, **kwargs))


class TestDialects:
    def test_every_dialect_is_known(self):
        for name in ("sqlite", "postgresql", "postgres", "mysql"):
            assert for_dialect(name).name

    def test_an_unknown_dialect_is_refused(self):
        with pytest.raises(ValueError) as exc:
            for_dialect("oracle")
        assert "known dialects" in str(exc.value)

    def test_verification_is_a_field_not_a_comment(self):
        """A connector whose SQL generation is tested is a different thing
        from one that has moved a customer's rows, and the code says which."""
        assert SQLITE.verified_live is True
        assert POSTGRESQL.verified_live is False
        assert MYSQL.verified_live is False

    def test_mysql_quotes_with_backticks(self):
        assert quote_ident("a", MYSQL) == "`a`"
        assert quote_ident("a", SQLITE) == '"a"'

    def test_a_quote_inside_an_identifier_is_escaped(self):
        assert quote_ident('we"ird', SQLITE) == '"we""ird"'
        assert quote_ident("we`ird", MYSQL) == "`we``ird`"


class TestValueQuoting:
    def test_a_quote_in_a_value_cannot_break_out(self):
        assert quote_value("'; DROP TABLE orders; --", SQLITE) == \
            "'''; DROP TABLE orders; --'"
        assert quote_value("O'Brien", SQLITE) == "'O''Brien'"

    def test_numbers_and_booleans_are_not_quoted(self):
        assert quote_value(42, SQLITE) == "42"
        assert quote_value(True, SQLITE) == "TRUE"
        assert quote_value(None, SQLITE) == "NULL"


class TestTypeNames:
    def test_affinities_choose_the_right_comparison_behaviour(self):
        assert sqlite_type_name(INT64) == "INTEGER"
        assert sqlite_type_name(FLOAT64) == "REAL"

    def test_a_date_is_text_because_iso_sorts_correctly(self):
        """Storing a date as an integer would make `BETWEEN` wrong."""
        assert sqlite_type_name(TIMESTAMP("ms", None)) == "TEXT"


# ------------------------------------------------- pushdown into real SQL
class TestSQLiteConnector:
    def test_reads_a_real_database(self, db_path):
        with sqlite_connector(db_path) as connector:
            got = connector.read(_scan(db_path))
        assert got.num_rows == 5
        assert got.column_names == ("id", "region", "amount", "quantity")

    def test_types_come_back_as_canonical(self, db_path):
        with sqlite_connector(db_path) as connector:
            got = connector.read(_scan(db_path))
        assert got.schema.get("id").type == INT64
        assert got.schema.get("amount").type == FLOAT64

    def test_a_filter_is_pushed_into_the_statement(self, db_path):
        with sqlite_connector(db_path) as connector:
            node = _scan(db_path)
            node.predicate = BinOp(Col("amount"), ">", Lit(200.0))
            sql, params = connector.build_select(node)
            assert 'WHERE ("amount" > 200.0)' in sql
            assert params == []
            got = connector.read(node)
        assert got.num_rows == 3
        assert min(got.column("amount").to_pylist()) == 300.0

    def test_a_projection_reads_only_the_named_columns(self, db_path):
        """The point of pushdown: two columns, not four."""
        with sqlite_connector(db_path) as connector:
            node = _scan(db_path, columns=("region", "amount"))
            sql, _ = connector.build_select(node)
            assert sql.startswith('SELECT "region", "amount"')
            got = connector.read(node)
        assert got.column_names == ("region", "amount")

    def test_a_reserved_word_as_a_column_name_still_works(self, tmp_path):
        """`order` and `group` are reserved; unquoted they are a syntax error."""
        path = str(tmp_path / "reserved.db")
        conn = sqlite3.connect(path)
        conn.execute('CREATE TABLE "order" ("group" TEXT, n INTEGER)')
        conn.execute('INSERT INTO "order" VALUES (?, ?)', ("a", 1))
        conn.commit()
        conn.close()
        with sqlite_connector(path) as connector:
            got = connector.read(_scan(path, table="order",
                                       columns=("group", "n")))
        assert got.column("group").to_pylist() == ["a"]

    def test_an_aggregate_query_runs(self, db_path):
        with sqlite_connector(db_path) as connector:
            node = Node(NodeType.SCAN_SQL, scan=ScanSpec(
                kind="sql", path=db_path,
                query="SELECT region, SUM(amount) AS total "
                      "FROM orders GROUP BY region"))
            got = connector.read(node)
        totals = {r["region"]: r["total"] for r in got.arrow.to_pylist()}
        assert totals == {"NA": 500.0, "EU": 700.0, "APAC": 300.0}

    def test_an_empty_result_keeps_its_columns(self, db_path):
        with sqlite_connector(db_path) as connector:
            node = _scan(db_path)
            node.predicate = BinOp(Col("amount"), ">", Lit(1e9))
            got = connector.read(node)
        assert got.num_rows == 0
        assert "region" in got.column_names

    def test_schema_discovery_works_on_an_empty_table(self, tmp_path):
        """`SELECT * LIMIT 0` would report no schema for an empty table."""
        path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")
        conn.commit()
        conn.close()
        with sqlite_connector(path) as connector:
            assert connector.table_columns("t") == ["a", "b"]

    def test_an_injected_value_comes_back_as_data(self, tmp_path):
        """End to end: the value is returned, and the table still exists."""
        path = str(tmp_path / "inj.db")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE t (v TEXT)")
        conn.execute("INSERT INTO t VALUES (?)", ("'; DROP TABLE t; --",))
        conn.commit()
        conn.close()
        with sqlite_connector(path) as connector:
            node = _scan(path, table="t")
            node.predicate = BinOp(Col("v"), "=", Lit("'; DROP TABLE t; --"))
            got = connector.read(node)
            assert got.column("v").to_pylist() == ["'; DROP TABLE t; --"]
            assert connector.table_columns("t") == ["v"]

    def test_a_missing_table_is_reported_with_the_statement(self, db_path):
        with sqlite_connector(db_path) as connector:
            with pytest.raises(Exception) as exc:
                connector.read(_scan(db_path, table="ghosts"))
        assert "ghosts" in str(exc.value)


# ------------------------------------------- dialects without a server
class TestUnverifiedDialects:
    def test_postgresql_reports_itself_unverified(self):
        connector = postgresql_connector("postgresql://localhost/db")
        assert connector.verified_live is False
        assert "UNVERIFIED" in connector.verification

    def test_mysql_reports_itself_unverified(self):
        assert "UNVERIFIED" in mysql_connector("mysql://u@h/db").verification

    def test_constructing_a_connector_is_not_evidence_it_works(self):
        assert sqlite_connector(":memory:").verified_live is True
        assert postgresql_connector("postgresql://x/y").verified_live is False

    def test_postgresql_sql_generation_is_still_correct(self):
        """Tested, and distinguished from the wire protocol being untested."""
        connector = postgresql_connector("postgresql://localhost/db")
        node = _scan("x", table="orders", columns=("a", "b"))
        node.predicate = BinOp(Col("a"), "=", Lit(1))
        node.limit = 5
        sql, _ = connector.build_select(node)
        assert sql == 'SELECT "a", "b" FROM "orders" WHERE ("a" = 1) LIMIT 5'

    def test_mysql_backtick_quoting_reaches_the_statement(self):
        connector = mysql_connector("mysql://u@h/db")
        sql, _ = connector.build_select(_scan("x", table="orders",
                                              columns=("a",)))
        assert sql == "SELECT `a` FROM `orders`"



# ------------------------------------------- MongoDB, via mongomock
class TestMongoTypeNames:
    def test_types_use_the_driver_vocabulary(self):
        """A naming difference is visible; a silent reclassification is not."""
        assert mongo_type_of(1) == "int32"
        assert mongo_type_of(2 ** 40) == "int64"
        assert mongo_type_of(1.5) == "double"
        assert mongo_type_of("x") == "string"
        assert mongo_type_of(True) == "bool"
        assert mongo_type_of(None) == "null"
        assert mongo_type_of({"a": 1}) == "object"
        assert mongo_type_of([1]) == "array"


class TestPushdownIsVisible:
    """Pushdown is priority one. It must not be invisible.

    The SQL and MongoDB connectors genuinely push a filter and a projection
    into the source - verified against a real SQLite database elsewhere in
    this file - and the planner's own priority order puts source pushdown
    first, because it is the only optimisation whose saving is paid *before*
    any data moves. Until now the runtime did all of that and the analyst
    was never told, which made the largest win in the system invisible to
    the person it exists for.

    These run against a real database rather than a mock: a pushdown report
    that names a table and a predicate is only worth anything if the pushdown
    it describes actually happened.
    """

    def _db(self, tmp_path):
        import sqlite3

        path = tmp_path / "orders.db"
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE orders (region TEXT, amount REAL, note TEXT)")
        con.executemany(
            "INSERT INTO orders VALUES (?, ?, ?)",
            [("NA", 100.0, "a"), ("EU", 200.0, "b"), ("NA", 300.0, "c")],
        )
        con.commit()
        con.close()
        return path

    def _scan_node(self, db_path, predicate=None, columns=()):
        from aar.ir import Node, NodeType, ScanSpec

        spec = ScanSpec(kind="sqlite", path=str(db_path), table="orders",
                        columns=tuple(columns))
        node = Node(NodeType.SCAN_SQL, scan=spec)
        if predicate is not None:
            node.predicate = predicate
        return node

    def test_a_pushed_filter_is_shown_with_its_sql(self, tmp_path):
        from aar.ir import BinOp, Col, Lit
        from aar.planner import pushdown_report

        node = self._scan_node(
            self._db(tmp_path), predicate=BinOp(Col("amount"), ">", Lit(150)))
        text = "\n".join(pushdown_report(node))
        assert "pushed WHERE" in text, text
        assert "amount" in text, text
        assert "150" in text, text

    def test_a_pushed_projection_is_shown(self, tmp_path):
        from aar.planner import pushdown_report

        node = self._scan_node(self._db(tmp_path),
                               columns=("region", "amount"))
        text = "\n".join(pushdown_report(node))
        assert "pushed projection" in text, text
        assert "region" in text and "amount" in text, text
        # The column that was NOT asked for must not appear, or the report
        # is not describing the pushdown.
        assert "note" not in text, text

    def test_a_source_with_nothing_pushed_reports_nothing(self, tmp_path):
        """Silence here is correct, and is different from a wrong claim."""
        from aar.planner import pushdown_report

        node = self._scan_node(self._db(tmp_path))
        assert pushdown_report(node) == []

    def test_the_rendered_plan_shows_the_pushdown(self, tmp_path):
        """The pushdown appears above the engine lines, in a real render.

        The `Plan` is built by hand rather than by the planner, because a
        bare SQL scan has no feasible engine on this build: `sqlite` is
        declared in the catalogue but has no execution class, so the
        planner correctly refuses it. That refusal is a separate property
        with its own test; what matters here is that the render path shows
        the pushdown and shows it first.
        """
        from aar.ir import BinOp, Col, Lit
        from aar.planner import Plan

        node = self._scan_node(
            self._db(tmp_path), predicate=BinOp(Col("amount"), ">", Lit(150)),
            columns=("region", "amount"))
        text = Plan(root=node, segments=[], total_s=0.0).render()
        assert "PUSHED TO SOURCE" in text, text
        assert "pushed WHERE" in text, text
        assert "pushed projection" in text, text
        # It must come before the segment lines: it is the more useful
        # answer, and it is decided first.
        assert text.index("PUSHED TO SOURCE") < text.index("total"), text


class TestMongoMatchRendering:
    def test_a_simple_comparison_renders(self):
        got = render_match(BinOp(Col("amount"), ">", Lit(100)))
        assert got == {"amount": {"$gt": 100}}

    def test_and_becomes_a_list(self):
        expr = BinOp(BinOp(Col("a"), "=", Lit(1)), "AND",
                     BinOp(Col("b"), "<>", Lit(2)))
        assert render_match(expr) == {"$and": [{"a": {"$eq": 1}},
                                               {"b": {"$ne": 2}}]}

    def test_an_unsupported_operator_declines(self):
        """Declining is the safe answer, as in the SQL renderer."""
        assert render_match(BinOp(Col("a"), "~", Lit(1))) is None

    def test_a_comparison_of_two_columns_declines(self):
        """MongoDB has no field-to-field comparison in a match document."""
        assert render_match(BinOp(Col("a"), "=", Col("b"))) is None


class TestMongoConnectorWithMock:
    """Against mongomock: real semantics, not a server.

    A query AAR renders wrongly fails here, which is the property the tests
    need. What mongomock does *not* cover - BSON encoding, the wire
    protocol, indexes, the server's own optimiser - is what
    ``verification`` names as unverified.
    """

    @pytest.fixture()
    def connector(self):
        conn = mock_mongo_connector()
        conn._client["shop"]["orders"].insert_many([
            {"_id": 1, "region": "NA", "amount": 100, "tags": ["a"]},
            {"_id": 2, "region": "EU", "amount": 200, "tags": []},
            {"_id": 3, "region": "APAC", "amount": 300},
        ])
        yield conn
        conn.close()

    def _node(self, **kwargs):
        return Node(NodeType.SCAN_MONGO, scan=ScanSpec(
            kind="mongo", database="shop", collection="orders", **kwargs))

    def test_it_is_honest_about_what_it_has_verified(self, connector):
        assert connector.verified_live is False
        assert "UNVERIFIED" in connector.verification

    def test_documents_become_a_table(self, connector):
        got = connector.read(self._node(columns=("region", "amount")))
        assert got.num_rows == 3
        assert got.column_names == ("region", "amount")
        assert sorted(got.column("amount").to_pylist()) == [100, 200, 300]

    def test_a_missing_field_becomes_null(self, connector):
        got = connector.read(self._node(columns=("amount", "tags")))
        assert got.column("tags").to_pylist()[2] is None

    def test_a_filter_is_pushed_down(self, connector):
        node = self._node()
        node.predicate = BinOp(Col("amount"), ">", Lit(150))
        got = connector.read(node)
        assert sorted(got.column("amount").to_pylist()) == [200, 300]

    def test_a_projection_reads_only_the_named_fields(self, connector):
        got = connector.read(self._node(columns=("region",)))
        assert got.column_names == ("region",)

    def test_count_does_not_move_the_documents(self, connector):
        node = self._node()
        node.predicate = BinOp(Col("amount"), ">", Lit(150))
        assert connector.count(node) == 2

    def test_an_aggregation_pipeline_runs_as_written(self, connector):
        node = Node(NodeType.SCAN_MONGO, scan=ScanSpec(
            kind="mongo", database="shop", collection="orders",
            query=[{"$group": {"_id": "$region", "total": {"$sum": "$amount"}}},
                   {"$sort": {"_id": 1}}]))
        got = connector.read(node)
        assert [r["_id"] for r in got.arrow.to_pylist()] == ["APAC", "EU", "NA"]
        assert [r["total"] for r in got.arrow.to_pylist()] == [300, 200, 100]

    def test_a_nested_document_is_rendered_as_json(self, connector):
        """A document store's nesting is free-form; a fixed Arrow struct
        would either fail or invent a shape the data does not have."""
        conn = mock_mongo_connector()
        conn._client["d"]["c"].insert_one({"_id": 1, "meta": {"a": 1, "b": 2}})
        got = conn.read(Node(NodeType.SCAN_MONGO, scan=ScanSpec(
            kind="mongo", database="d", collection="c", columns=("meta",))))
        assert '"a": 1' in got.column("meta").to_pylist()[0]

    def test_explain_shows_the_query_that_would_run(self, connector):
        node = self._node(columns=("region",))
        node.predicate = BinOp(Col("amount"), ">", Lit(150))
        text = connector.explain_plan(node)
        assert "shop.orders.find" in text
        assert "$gt" in text
        assert "projection" in text


# ------------------------------------------------------------ source hygiene
class TestSourceParses:
    """Every module must parse, and no function may look truncated.

    Neither check is about style. A docstring split by an interrupted edit
    produces a module that imports nothing, so every other test in the file
    fails with a confusing error instead of pointing at the one wrong line -
    and a function that lost its body returns ``None`` where a ``Table`` was
    expected. Having these *inside* the suite means pytest cannot report a
    green run for code that does not compile.
    """

    @pytest.mark.parametrize("root", ["src", "tests", "tools", "pipelines"])
    def test_every_python_file_parses(self, root):
        import ast
        import pathlib

        base = pathlib.Path(__file__).resolve().parent.parent / root
        if not base.is_dir():
            pytest.skip(f"{root} is not present")
        broken = []
        for path in sorted(base.rglob("*.py")):
            try:
                ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                broken.append(f"{path.relative_to(base)}:{exc.lineno}: "
                              f"{exc.msg}")
        assert not broken, "modules that do not parse:\n  " + \
            "\n  ".join(broken)

    def test_no_function_is_truncated_to_a_docstring_and_imports(self):
        """A body of only a docstring and imports is almost always damage.

        The executor's guard for this catches it at run time for the one
        operation that returns a Table. This catches it at build time, for
        every function, including the ones no test happens to reach.

        The test is deliberately narrow: only *imports* count as harmless.
        `_RUN_CACHE.clear()` is a legitimate body with no return, so a
        check that merely looked for "no return statement" would fire on
        every well-written mutator in the codebase.
        """
        import ast
        import pathlib

        base = pathlib.Path(__file__).resolve().parent.parent / "src" / "aar"
        suspicious = []
        for path in sorted(base.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef,
                                         ast.AsyncFunctionDef)):
                    continue
                if any(isinstance(n, (ast.Return, ast.Yield, ast.YieldFrom,
                                      ast.Raise))
                       for n in ast.walk(node)):
                    continue
                # Everything after the docstring is an import or `pass`,
                # and there is more than just a docstring.
                rest = node.body[1:] if (node.body and isinstance(
                    node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)) \
                    else node.body
                if len(node.body) > 1 and all(
                        isinstance(n, (ast.Import, ast.ImportFrom, ast.Pass))
                        for n in rest):
                    suspicious.append(
                        f"{path.name}:{node.lineno} {node.name}()")
        assert not suspicious, (
            "functions that look truncated - body is only a docstring and "
            "imports, with no return:\n  " + "\n  ".join(suspicious))

    def test_mysql_backtick_quoting_reaches_the_statement(self):
        connector = mysql_connector("mysql://u@h/db")
        sql, _ = connector.build_select(_scan("x", table="orders",
                                              columns=("a",)))
        assert sql == "SELECT `a` FROM `orders`"


    def test_a_limit_is_rendered_in_the_statement(self, db_path):
        with sqlite_connector(db_path) as connector:
            node = _scan(db_path)
            node.limit = 2
            sql, _ = connector.build_select(node)
            assert "LIMIT 2" in sql
            assert connector.read(node).num_rows == 2

