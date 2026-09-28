"""Runtime: engines, interchange, connectors and the executor.

These tests run real data through real engines. Nothing here is mocked: a
filter test filters rows, a group-by test aggregates them, and a write test
writes a file and reads it back. A mock-based suite would prove the mocks
work.
"""

from __future__ import annotations

import os
import warnings

import pytest

pa = pytest.importorskip("pyarrow")

from aar.engines import create_engine                      # noqa: E402
from aar.engines.base import PredicateCompiler             # noqa: E402
from aar.interchange import Table, arrow_to_canonical, canonical_to_arrow  # noqa: E402
from aar.ir import (BinOp, Col, Lit, Node, NodeType, ScanSpec,  # noqa: E402
                    topological_order)
from aar.sdk import (col, csv, excel, filter_, group_by, limit, lit,  # noqa: E402
                     parquet, project, sort, sum_, udf, write_csv,
                     write_excel, write_parquet)
from aar.types import (FLOAT64, INT64, UTF8, Field, Schema)  # noqa: E402


# ------------------------------------------------------------------ fixtures
@pytest.fixture()
def orders(arrow):
    """A small, realistic orders table.

    Regions cycle NA, EU, APAC and end with NA, so the group sums are:

        NA   idx 0,3,6,9 -> 100.0 + 900.0 + 610.0 + 55.0   = 1665.0
        EU   idx 1,4,7   -> 250.5 + 310.5 + 88.0           =  649.0
        APAC idx 2,5,8   ->  75.25 + 42.0 + 1200.0         = 1317.25
    """
    return Table(pa.table({
        "id": pa.array(range(1, 11), type=pa.int64()),
        "region": pa.array(["NA", "EU", "APAC"] * 3 + ["NA"], type=pa.string()),
        "amount": pa.array([100.0, 250.5, 75.25, 900.0, 310.5,
                            42.0, 610.0, 88.0, 1200.0, 55.0],
                           type=pa.float64()),
        "quantity": pa.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=pa.int64()),
    }))


#: Expected group sums, derived from the fixture above.
GROUP_SUMS = {"NA": 1665.0, "EU": 649.0, "APAC": 1317.25}

#: Amounts strictly greater than 100: idx 1,3,4,6,8.
OVER_100 = [250.5, 900.0, 310.5, 610.0, 1200.0]

#: Group sums after filtering to `amount > 100`:
#:   NA 900.0 + 610.0 = 1510.0 | EU 250.5 + 310.5 = 561.0 | APAC 1200.0
FILTERED_SUMS = {"NA": 1510.0, "EU": 561.0, "APAC": 1200.0}



@pytest.fixture()
def parquet_file(tmp_path, orders):
    import pyarrow.parquet as pq

    path = tmp_path / "orders.parquet"
    pq.write_table(orders.arrow, str(path))
    return str(path)


@pytest.fixture()
def excel_file(tmp_path, orders):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Orders"
    names = list(orders.column_names)
    ws.append(names)
    for row in orders.arrow.to_pylist():
        ws.append([row[n] for n in names])
    path = tmp_path / "orders.xlsx"
    wb.save(str(path))
    return str(path)


# ------------------------------------------------------------ interchange
class TestTypeConversion:
    @pytest.mark.parametrize("arrow_type,expected", [
        (pa.int64(), INT64), (pa.float64(), FLOAT64),
        (pa.string(), UTF8), (pa.bool_(), UTF8),
    ])
    def test_arrow_to_canonical(self, arrow_type, expected):
        if arrow_type == pa.bool_():
            from aar.types import BOOLEAN
            assert arrow_to_canonical(arrow_type) == BOOLEAN
        else:
            assert arrow_to_canonical(arrow_type) == expected

    def test_timestamp_keeps_unit_and_timezone(self):
        t = arrow_to_canonical(pa.timestamp("ms", tz="UTC"))
        assert t.unit == "ms"
        assert t.timezone == "UTC"

    def test_decimal_keeps_precision(self):
        t = arrow_to_canonical(pa.decimal128(12, 2))
        assert (t.precision, t.scale) == (12, 2)

    def test_round_trip_preserves_the_type(self):
        for t in (pa.int64(), pa.float64(), pa.string(), pa.date32()):
            assert canonical_to_arrow(arrow_to_canonical(t)) == t

    def test_unmappable_arrow_type_raises(self):
        with pytest.raises(Exception):
            arrow_to_canonical(object())


class TestTable:
    def test_schema_is_derived_from_arrow(self, orders):
        assert orders.column_names == ("id", "region", "amount", "quantity")
        assert orders.schema.get("amount").type == FLOAT64
        assert orders.num_rows == 10

    def test_schema_length_must_match_columns(self, orders):
        with pytest.raises(ValueError):
            Table(orders.arrow, Schema((Field("only", INT64),)))

    def test_select_preserves_order_and_metadata(self, orders):
        got = orders.select(["amount", "id"])
        assert got.column_names == ("amount", "id")
        tagged = got.tagged("id", "PII")
        assert "PII" in tagged.schema.get("id").classification
        # Tagging must not mutate the original.
        assert "PII" not in orders.schema.get("id").classification


    def test_select_unknown_column_names_it(self, orders):
        with pytest.raises(KeyError) as exc:
            orders.select(["nope"])
        assert "have:" in str(exc.value)

    def test_empty_table_is_correctly_typed(self):
        s = Schema((Field("a", INT64), Field("b", UTF8)))
        t = Table.empty(s)
        assert t.num_rows == 0
        assert t.column_names == ("a", "b")
        assert t.schema == s

    def test_batches_partition_the_rows(self, orders):
        sizes = [b.num_rows for b in orders.batches(4)]
        assert sum(sizes) == orders.num_rows
        assert max(sizes) <= 4

    def test_batches_rejects_a_non_positive_size(self, orders):
        with pytest.raises(ValueError):
            list(orders.batches(0))

    def test_sort_indices_orders_descending(self, orders):
        idx = orders.sort_indices([("amount", False)])
        values = [orders.arrow.take(idx).column("amount").to_pylist()]
        assert values[0] == sorted(values[0], reverse=True)



# ----------------------------------------------------------------- engines
def _engine_importable(engine_id: str) -> bool:
    import importlib.util

    module = {"arrow": "pyarrow", "duckdb": "duckdb",
              "polars_cpu": "polars", "pandas": "pandas"}.get(engine_id)
    if module is None:
        return False
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


#: Every engine installed here must produce identical results for the core
#: operations. Cross-engine agreement is the property that matters: a
#: divergence between DuckDB and Polars is exactly the silent corruption the
#: Arrow interchange layer exists to prevent.
INSTALLED = [e for e in ("arrow", "duckdb", "polars_cpu", "pandas")
             if _engine_importable(e)]

#: Engine id -> the module it needs, for skip logic.
_MODULE_OF = {"arrow": "pyarrow", "duckdb": "duckdb",
              "polars_cpu": "polars", "pandas": "pandas",
              "python_worker": "builtins", "excel": "openpyxl"}

#: All engine ids the factory claims it can build.
ENGINE_FACTORIES = __import__("aar.engines", fromlist=["x"]).ENGINE_FACTORIES




@pytest.fixture(params=INSTALLED)
def engine(request):
    with create_engine(request.param) as eng:
        yield eng


class TestEngineAgreement:
    """Every engine must agree. A disagreement is a correctness bug."""

    def test_filter_keeps_the_same_rows(self, engine, orders):
        got = engine.filter(orders, BinOp(Col("amount"), ">", Lit(100.0)))
        assert sorted(got.column("amount").to_pylist()) == sorted(OVER_100)
        assert got.num_rows == len(OVER_100)


    def test_filter_on_strings(self, engine, orders):
        got = engine.filter(orders, BinOp(Col("region"), "=", Lit("EU")))
        assert got.num_rows == 3
        assert set(got.column("region").to_pylist()) == {"EU"}

    def test_filter_matching_nothing_returns_no_rows(self, engine, orders):
        got = engine.filter(orders, BinOp(Col("amount"), ">", Lit(1e12)))
        assert got.num_rows == 0

    def test_filter_preserves_the_schema(self, engine, orders):
        got = engine.filter(orders, BinOp(Col("amount"), ">", Lit(100.0)))
        assert got.column_names == orders.column_names

    def test_project_keeps_named_columns(self, engine, orders):
        got = engine.project(orders, ["region", "amount"])
        assert got.column_names == ("region", "amount")

    def test_group_by_sum_is_correct(self, engine, orders):
        got = engine.group_by(orders, ["region"], {"total": sum_("amount")})
        by_region = {r["region"]: r["total"] for r in got.arrow.to_pylist()}
        for region, expected in GROUP_SUMS.items():
            assert by_region[region] == pytest.approx(expected), region

    def test_group_by_returns_one_row_per_group(self, engine, orders):
        """Row count, checked independently of the values.

        A group-by that silently returns nothing fails every value assertion
        anyway, but it fails as "KeyError: 'NA'" with no hint of the cause.
        Asserting the count names the actual defect, and it is the assertion
        that catches an engine returning an empty result for the right
        reasons but the wrong data.
        """
        got = engine.group_by(orders, ["region"], {"total": sum_("amount")})
        assert got.num_rows == len(GROUP_SUMS)
        assert sorted(got.column("region").to_pylist()) == sorted(GROUP_SUMS)

    def test_group_by_keeps_the_input_key_column_type(self, engine, orders):
        """Keys must not be re-inferred from the result values.

        An engine that rebuilt the table from Python rows would let an Int64
        key silently become Float64 because one group happened to be null.
        """
        from aar.types import INT64, UTF8

        got = engine.group_by(orders, ["region"], {"total": sum_("amount")})
        assert got.schema.get("region").type == UTF8

        with_int_key = Table(pa.table({
            "k": pa.array([1, 1, 2], type=pa.int64()),
            "v": pa.array([1.0, 2.0, 3.0], type=pa.float64())}))
        out = engine.group_by(with_int_key, ["k"], {"total": sum_("v")})
        assert out.schema.get("k").type == INT64



    def test_group_by_over_no_rows_is_still_typed(self, engine, orders):
        empty = Table.empty(orders.schema)
        got = engine.group_by(empty, ["region"], {"total": sum_("amount")})
        assert got.num_rows == 0
        assert "total" in got.column_names
        assert "region" in got.column_names

    def test_sort_descending(self, engine, orders):
        got = engine.sort(orders, [("amount", False)])
        values = got.column("amount").to_pylist()
        assert values == sorted(values, reverse=True)

    def test_sort_by_string_column(self, engine, orders):
        got = engine.sort(orders, [("region", True)])
        values = got.column("region").to_pylist()
        assert values == sorted(values)

    def test_limit(self, engine, orders):
        assert engine.limit(orders, 3).num_rows == 3
        assert engine.limit(orders, 999).num_rows == orders.num_rows

    def test_limit_rejects_a_negative_count(self, engine, orders):
        with pytest.raises(ValueError):
            engine.limit(orders, -1)

    def test_read_parquet(self, engine, parquet_file):
        got = engine.read_scan(
            Node(NodeType.SCAN_PARQUET,
                 scan=ScanSpec(kind="parquet", path=parquet_file)))
        assert got.num_rows == 10
        assert "amount" in got.column_names

    @pytest.mark.parametrize("fmt,suffix", [("parquet", ".parquet"),
                                           ("csv", ".csv")])
    def test_write_then_read_back(self, engine, orders, tmp_path, fmt, suffix):
        """Every engine must be able to write every format it claims.

        Writing through an engine and reading the file back is the only test
        that catches a malformed output: a file the engine wrote but cannot
        itself read is still a broken pipeline.
        """
        target = str(tmp_path / f"out{suffix}")
        node = Node(NodeType.WRITE, target=target, write_format=fmt)
        engine.write(orders, node)
        assert os.path.exists(target)

        scan = Node(NodeType.SCAN_PARQUET if fmt == "parquet"
                    else NodeType.SCAN_CSV,
                    scan=ScanSpec(kind=fmt, path=target))
        back = engine.read_scan(scan)
        assert back.num_rows == orders.num_rows
        assert set(back.column_names) == set(orders.column_names)
        assert sorted(back.column("amount").to_pylist()) == \
            sorted(orders.column("amount").to_pylist())




# ------------------------------------------------------------------- joins
class TestJoin:
    def _left(self):
        return Table(pa.table({
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "name": pa.array(["a", "b", "c"], type=pa.string()),
        }))

    def _right(self):
        return Table(pa.table({
            "id": pa.array([2, 3, 4], type=pa.int64()),
            "score": pa.array([20, 30, 40], type=pa.int64()),
        }))

    @pytest.mark.parametrize("engine_id", INSTALLED)
    def test_inner_join_keeps_only_matches(self, engine_id):
        with create_engine(engine_id) as eng:
            got = eng.join(self._left(), self._right(), ["id"], "inner")
        rows = {r["id"]: r["score"] for r in got.arrow.to_pylist()}
        assert rows == {2: 20, 3: 30}

    @pytest.mark.parametrize("engine_id", INSTALLED)
    def test_left_join_keeps_unmatched_left_rows(self, engine_id):
        """A naive implementation silently drops these, which is wrong."""
        with create_engine(engine_id) as eng:
            got = eng.join(self._left(), self._right(), ["id"], "left")
        rows = {r["id"]: r["score"] for r in got.arrow.to_pylist()}
        assert rows[1] is None
        assert rows[2] == 20

    def test_null_keys_do_not_join_to_the_string_none(self):
        """The classic NULL-join bug, pinned shut."""
        left = Table(pa.table({"id": pa.array([None, 1], type=pa.int64())}))
        right = Table(pa.table({"id": pa.array([None], type=pa.int64()),
                                "v": pa.array([9], type=pa.int64())}))
        with create_engine("arrow") as eng:
            got = eng.join(left, right, ["id"], "inner")
        assert got.num_rows == 0

    def test_join_on_a_missing_key_is_refused(self):
        with create_engine("arrow") as eng:
            with pytest.raises(KeyError):
                eng.join(self._left(), self._right(), ["nope"], "inner")


# -------------------------------------------------------------------- UDF
class TestUDF:
    def test_row_udf_adds_a_column(self, orders):
        with create_engine("arrow") as eng:
            got = eng.udf(orders, lambda row: row["amount"] * 2)
        assert "result" in got.column_names
        assert got.column("result").to_pylist()[0] == 200.0

    def test_named_row_udf_names_its_column(self, orders):
        def doubled(row):
            return row["amount"] * 2
        with create_engine("arrow") as eng:
            got = eng.udf(orders, doubled)
        assert "doubled" in got.column_names

    def test_column_udf_receives_every_column(self, orders):
        """A column UDF is asked for explicitly, never inferred.

        It receives *all* columns, not the first: handing over one column
        without saying which would let it operate on the wrong data and
        produce a plausible wrong answer instead of an error.
        """
        seen = {}

        def collect(columns):
            seen.update(columns)
            return [v * 2 for v in columns["amount"]]

        with create_engine("arrow") as eng:
            got = eng.udf(orders, collect, mode="column")
        assert set(seen) == set(orders.column_names)
        assert got.column("collect").to_pylist() == \
            [v * 2 for v in orders.column("amount").to_pylist()]

    def test_row_mode_is_the_default_for_a_field_named_parameter(self, orders):
        """The ambiguity that motivated making ``mode`` explicit.

        ``def band(row)`` reads one field. A signature-based guess is free to
        call it with a whole column instead, and the failure then surfaces
        inside the user's function. Row mode is the documented default, so the
        common case needs no argument at all.
        """
        def band(row):
            return f'{row["region"]}:{row["amount"]}'

        with create_engine("arrow") as eng:
            got = eng.udf(orders, band)
        assert got.column("band").to_pylist()[0] == "NA:100.0"

    def test_column_udf_returning_the_wrong_length_is_refused(self, orders):
        """A length mismatch would silently truncate if not checked."""
        with create_engine("arrow") as eng:
            with pytest.raises(ValueError) as exc:
                eng.udf(orders, lambda cols: [1], mode="column")
        assert "one value per row" in str(exc.value)

    def test_unknown_udf_mode_is_refused(self, orders):
        with create_engine("arrow") as eng:
            with pytest.raises(ValueError):
                eng.udf(orders, lambda row: 1, mode="nonsense")

    def test_failing_udf_names_the_function_and_the_cause(self, orders):
        from aar.failures import UDFExecutionError

        def boom(row):
            raise ValueError("bad row")
        with create_engine("python_worker") as eng:
            with pytest.raises(UDFExecutionError) as exc:
                eng.udf(orders, boom)
        assert "boom" in str(exc.value)
        assert "bad row" in str(exc.value)

    def test_failing_udf_is_recorded_on_the_engine(self, orders):
        def boom(row):
            raise ValueError("nope")
        with create_engine("python_worker") as eng:
            with pytest.raises(Exception):
                eng.udf(orders, boom)
            assert eng.failures and eng.failures[0][0] == "boom"

    def test_python_worker_refuses_data_operations(self, orders):
        with create_engine("python_worker") as eng:
            with pytest.raises(NotImplementedError):
                eng.filter(orders, BinOp(Col("amount"), ">", Lit(1.0)))


# ---------------------------------------------------------------- factory
class TestFactory:
    def test_builds_the_requested_engine(self):
        with create_engine("arrow") as eng:
            assert eng.id == "arrow"

    @pytest.mark.parametrize("engine_id", sorted(ENGINE_FACTORIES))
    def test_every_registered_engine_can_actually_be_built(self, engine_id):
        """Each registered engine must be constructible and complete.

        This is the test that catches a class of mistake nothing else does: an
        engine whose methods ended up nested somewhere else, so the class looks
        fine to a reader and to a syntax check but is abstract. The factory
        would then quietly degrade every use of it, and the pipeline would
        still produce correct numbers - just on the wrong engine, every time,
        with the reason buried in the ledger.
        """
        if not _engine_importable(_MODULE_OF.get(engine_id, "")):
            pytest.skip(f"{engine_id} is not installed")
        engine = create_engine(engine_id)
        try:
            assert engine.id == engine_id
            # Nothing on the abstract contract may be missing.
            for op in ("read_scan", "filter", "project", "group_by",
                       "sort", "limit"):
                assert callable(getattr(engine, op)), f"{engine_id}.{op}"
        finally:
            engine.close()

    @pytest.mark.skipif(not _engine_importable("duckdb"),
                        reason="duckdb is not installed")
    def test_duckdb_opens_a_connection_and_reuses_it(self):
        """The connection is opened on first use and then reused.

        Opening it eagerly in ``__init__`` would make the engine pay a
        connection cost it might never need, and creating a new one per query
        would dominate the runtime of a small pipeline.
        """
        from aar.engines.duckdb_engine import DuckDBEngine

        engine = DuckDBEngine()
        try:
            assert engine._conn is None      # nothing opened yet
            first = engine.conn
            assert first is engine.conn      # the same object, not a new one
            assert engine._conn is not None
        finally:
            engine.close()
        assert engine._conn is None

    @pytest.mark.skipif(not _engine_importable("duckdb"),
                        reason="duckdb is not installed")
    def test_duckdb_accepts_a_thread_count(self):
        from aar.engines.duckdb_engine import DuckDBEngine

        engine = DuckDBEngine(threads=2)
        try:
            assert engine.conn is not None
        finally:
            engine.close()

    def test_unknown_engine_degrades_and_records(self):
        from aar.failures import DegradationLedger

        ledger = DegradationLedger()
        with create_engine("nope", ledger=ledger) as eng:
            assert eng.id in ("arrow", "duckdb", "polars_cpu", "pandas")
        assert len(ledger) == 1
        assert "nope" in ledger.entries[0].detail

    def test_absent_engine_degrades_and_records(self):
        from aar.failures import DegradationLedger

        ledger = DegradationLedger()
        # No GPU on this machine, so the factory must fall back and say so.
        with create_engine("polars_gpu", ledger=ledger) as eng:
            assert eng.device.value != "gpu"
        assert len(ledger) == 1
        assert ledger.entries[0].to_engine is not None


class TestSubstitutionsAreNeverSilent:
    """The defect that motivated all of this.

    `create_engine("cudf")` on a host without cuDF used to return an Arrow
    engine and record nothing. A verification script trusted the return
    value and wrote Arrow's timings to a file headed "cudf". These tests
    exist so that cannot come back.
    """

    def test_no_ledger_still_leaves_a_record(self):
        from aar.failures import process_ledger

        before = len(process_ledger())
        with create_engine("cudf") as eng:
            assert eng is not None
        entries = process_ledger().entries[before:]
        assert len(entries) == 1
        assert entries[0].from_engine == "cudf"
        assert entries[0].to_engine is not None

    def test_no_ledger_also_warns(self):
        """A record nobody reads is indistinguishable from silence."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with create_engine("cudf"):
                pass
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)
        assert any("cudf" in str(w.message) for w in caught)

    def test_a_ledger_suppresses_the_warning(self):
        """The caller supplied somewhere to put it, so they are listening."""
        from aar.failures import DegradationLedger

        ledger = DegradationLedger()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with create_engine("cudf", ledger=ledger):
                pass
        assert not [w for w in caught
                    if issubclass(w.category, RuntimeWarning)]
        assert len(ledger) == 1

    def test_a_present_engine_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with create_engine("arrow"):
                pass
        assert not [w for w in caught
                    if issubclass(w.category, RuntimeWarning)]

    def test_you_can_refuse_the_substitution(self):
        """When the specific engine is the point, there is no right answer
        to give but a different engine."""
        from aar.failures import CapabilityError

        with pytest.raises(CapabilityError) as excinfo:
            create_engine("cudf", allow_degradation=False)
        assert "cudf" in str(excinfo.value)
        assert "allow_degradation=True" in str(excinfo.value)

    def test_refusal_records_nothing(self):
        """A refused request is not a degradation - nothing ran."""
        from aar.failures import process_ledger

        before = len(process_ledger())
        with pytest.raises(Exception):
            create_engine("cudf", allow_degradation=False)
        assert len(process_ledger()) == before


class TestDeclaredIsNotImplemented:
    """The gap the GPU exercise exposed, kept visible rather than implied.

    The capability registry *declares* sixteen engines across six tiers.
    Only six have an implementation behind them. A declaration is a
    contract about what the planner may choose; an implementation is a
    class that can actually run. ``create_engine`` can only ever hand back
    a fallback for the other ten, so any test, benchmark or plan that
    treats a declared engine as a usable one is measuring the fallback.

    This is a tracked, deliberate gap, not a passing condition. When a
    real CudfEngine lands, this list is what must shrink.
    """

    #: Declared in the capability registry, deliberately not implemented.
    NOT_IMPLEMENTED = frozenset({
        "cudf", "polars_gpu", "ray", "dask", "spark_rapids",
        "postgresql", "mysql", "sqlite", "mongodb", "trino",
    })

    def test_the_known_gap_is_still_the_known_gap(self):
        """If this fails, an engine moved or was added - update the list."""
        from aar.capability import ENGINES
        from aar.engines.factory import ENGINE_FACTORIES

        # ENGINES is a tuple of EngineSpec, not a set of ids.
        declared = {spec.id for spec in ENGINES}
        implemented = set(ENGINE_FACTORIES)
        assert implemented <= declared, (
            f"engine factories with no capability declaration: "
            f"{sorted(implemented - declared)}")
        assert declared - implemented == self.NOT_IMPLEMENTED, (
            f"the unimplemented set changed. Missing now: "
            f"{sorted((declared - implemented) - self.NOT_IMPLEMENTED)}; "
            f"newly implemented: "
            f"{sorted(self.NOT_IMPLEMENTED - (declared - implemented))}")

    def test_an_unimplemented_engine_refuses_rather_than_implying(self):
        """The whole point of allow_degradation=False, on a real id."""
        from aar.failures import CapabilityError

        for engine_id in sorted(self.NOT_IMPLEMENTED):
            with pytest.raises(CapabilityError):
                create_engine(engine_id, allow_degradation=False)



# --------------------------------------------------------------- end to end
def _pipeline_for(path: str):
    """The specification's shape, on real files."""
    def risk_band(row):
        amount = row.get("total", 0) or 0
        return "high" if amount >= 1000 else (
            "medium" if amount >= 300 else "low")

    orders = parquet(path, estimated_bytes=2_000_000)
    big = filter_(orders, col("amount") > lit(100.0))
    by_region = group_by(big, "region", aggs={
        "total": sum_("amount"), "n": sum_("quantity")})
    banded = udf(by_region, risk_band, name="risk_band")
    ranked = sort(banded, "total desc")
    return limit(ranked, 2)


class TestEndToEnd:
    """The whole path: build -> plan -> execute -> rows."""

    def test_pipeline_runs_and_returns_the_right_rows(self, parquet_file):
        from aar.planner import AdaptivePlanner
        from aar.runtime import Executor

        root = _pipeline_for(parquet_file)
        plan = AdaptivePlanner().plan(root)
        with Executor() as ex:
            result = ex.execute(plan)

        assert result.ok, result.ledger.render()
        assert result.table is not None
        # `limit(2)` keeps only the two largest regions, so EU (561.0) is
        # correctly absent - the pipeline's own arithmetic, not a bug.
        assert result.table.num_rows == 2
        by_region = {r["region"]: r["total"]
                     for r in result.table.arrow.to_pylist()}
        assert by_region == {"NA": pytest.approx(1510.0),
                             "APAC": pytest.approx(1200.0)}
        totals = [r["total"] for r in result.table.arrow.to_pylist()]
        assert totals == sorted(totals, reverse=True)
        assert "risk_band" in result.table.column_names




    def test_pipeline_writes_a_real_file(self, parquet_file, tmp_path):
        import pyarrow.parquet as pq

        from aar.planner import AdaptivePlanner
        from aar.runtime import Executor

        out = str(tmp_path / "out.parquet")
        write = write_parquet(_pipeline_for(parquet_file), out)
        plan = AdaptivePlanner().plan(write)
        with Executor() as ex:
            result = ex.execute(plan)

        assert result.ok, result.ledger.render()
        assert os.path.exists(out)
        assert pq.read_table(out).num_rows == result.rows
        assert result.written and result.written[0][0] == out

    def test_every_node_is_recorded_with_its_engine(self, parquet_file):
        from aar.planner import AdaptivePlanner
        from aar.runtime import Executor

        root = _pipeline_for(parquet_file)
        plan = AdaptivePlanner().plan(root)
        with Executor() as ex:
            result = ex.execute(plan)
        # Every node the plan contains must have an outcome, and vice versa.
        assert len(result.outcomes) >= len(list(root.walk()))
        assert len(result.outcomes) == len(topological_order(root))
        for o in result.outcomes:
            assert o.engine_used
            assert o.error is None
        assert result.render().startswith("EXECUTION")


    def test_write_to_excel_round_trips(self, tmp_path):
        import openpyxl
        import pyarrow.parquet as pq

        from aar.planner import AdaptivePlanner
        from aar.runtime import Executor

        src = Table(pa.table({"a": pa.array([1, 2, 3], type=pa.int64()),
                              "b": pa.array(["x", "y", "z"],
                                            type=pa.string())}))
        path = str(tmp_path / "in.parquet")
        pq.write_table(src.arrow, path)

        target = str(tmp_path / "out.xlsx")
        plan = AdaptivePlanner().plan(
            write_excel(parquet(path), target, sheet="Result"))
        with Executor() as ex:
            result = ex.execute(plan)

        assert result.ok, result.ledger.render()
        assert os.path.exists(target)
        wb = openpyxl.load_workbook(target, read_only=True)
        assert "Result" in wb.sheetnames
        rows = list(wb["Result"].iter_rows(values_only=True))
        assert rows[0] == ("a", "b")
        assert len(rows) == 4          # header + 3 rows
        wb.close()

    def test_a_failing_node_raises_rather_than_returning_partials(
            self, tmp_path):
        import pyarrow.parquet as pq

        from aar.planner import AdaptivePlanner
        from aar.runtime import Executor

        def boom(row):
            raise RuntimeError("deliberate failure")

        src = Table(pa.table({"a": pa.array([1, 2], type=pa.int64())}))
        path = str(tmp_path / "in.parquet")
        pq.write_table(src.arrow, path)
        plan = AdaptivePlanner().plan(udf(parquet(path), boom, name="boom"))
        with Executor() as ex:
            with pytest.raises(Exception) as exc:
                ex.execute(plan)
        assert "boom" in str(exc.value) or "deliberate" in str(exc.value)


class TestExecutorOperations:
    def test_cast_converts_and_keeps_the_schema_length(self, orders):
        from aar.runtime import Executor
        from aar.types import DECIMAL

        # Decimal(38,2) has room for a 64-bit integer; Decimal(10,2) does not
        # and Arrow rightly refuses the cast.
        node = Node(NodeType.CAST, inputs=[], casts={"quantity": DECIMAL(38, 2)})
        got = Executor()._cast(orders, node)
        assert got.column_names == orders.column_names
        assert got.schema.get("quantity").type == DECIMAL(38, 2)

    def test_cast_that_cannot_be_represented_is_refused(self, orders):
        """A lossy cast must fail loudly rather than silently truncate."""
        from aar.runtime import Executor
        from aar.types import DECIMAL

        node = Node(NodeType.CAST, inputs=[], casts={"quantity": DECIMAL(4, 1)})
        with pytest.raises(ValueError) as exc:
            Executor()._cast(orders, node)
        assert "quantity" in str(exc.value)


    def test_null_handle_fills(self):
        from aar.runtime import Executor

        t = Table(pa.table({"id": pa.array([1, None, 3], type=pa.int64())}))
        node = Node(NodeType.NULL_HANDLE, inputs=[], null_strategy="fill",
                    fill_value=0)
        got = Executor()._null_handle(t, node)
        assert got.column("id").to_pylist() == [1, 0, 3]


# ------------------------------------------------------------- connectors
class TestExcelConnector:
    def test_reads_a_workbook_with_types_intact(self, excel_file):
        from aar.connectors.excel import read_excel
        from aar.ir import ScanSpec as _Spec

        got = read_excel(_Spec(kind="excel", path=excel_file, sheet="Orders"))
        assert got.num_rows == 10
        assert got.column_names == ("id", "region", "amount", "quantity")
        assert got.schema.get("id").type == INT64
        assert got.schema.get("amount").type == FLOAT64
        assert got.schema.get("region").type == UTF8

    def test_round_trips_through_write_and_read(self, orders, tmp_path):
        from aar.connectors.excel import read_excel, write_excel
        from aar.ir import ScanSpec as _Spec

        target = str(tmp_path / "out.xlsx")
        write_excel(orders, target, sheet="Data")
        back = read_excel(_Spec(kind="excel", path=target, sheet="Data"))
        assert back.num_rows == orders.num_rows
        assert back.column_names == orders.column_names
        assert sorted(back.column("amount").to_pylist()) == \
            sorted(orders.column("amount").to_pylist())

    def test_duplicate_headers_are_made_unique(self, tmp_path):
        """Two columns called `amount` must not produce an unselectable table."""
        openpyxl = pytest.importorskip("openpyxl")
        from aar.connectors.excel import read_excel
        from aar.ir import ScanSpec as _Spec

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["amount", "amount"])
        ws.append([1, 2])
        path = tmp_path / "dupe.xlsx"
        wb.save(str(path))

        got = read_excel(_Spec(kind="excel", path=str(path)))
        assert got.column_names == ("amount", "amount_1")
        assert got.num_rows == 1

    def test_blank_header_gets_a_usable_name(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        from aar.connectors.excel import read_excel
        from aar.ir import ScanSpec as _Spec

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["a", None])
        ws.append([1, 2])
        path = tmp_path / "blank.xlsx"
        wb.save(str(path))

        got = read_excel(_Spec(kind="excel", path=str(path)))
        assert got.column_names == ("a", "column_2")

    def test_error_cells_become_nulls_not_text(self, tmp_path):
        """`#DIV/0!` is not data; keeping it as text poisons every average."""
        openpyxl = pytest.importorskip("openpyxl")
        from aar.connectors.excel import read_excel
        from aar.ir import ScanSpec as _Spec

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["a"])
        ws.append([1])
        ws.append(["#DIV/0!"])
        path = tmp_path / "errors.xlsx"
        wb.save(str(path))

        got = read_excel(_Spec(kind="excel", path=str(path)))
        assert got.column("a").to_pylist() == [1, None]

    def test_missing_workbook_raises_source_unavailable(self, tmp_path):
        from aar.connectors.excel import read_excel
        from aar.failures import SourceUnavailable
        from aar.ir import ScanSpec as _Spec

        with pytest.raises(SourceUnavailable):
            read_excel(_Spec(kind="excel", path=str(tmp_path / "nope.xlsx")))

    def test_missing_sheet_lists_the_sheets_that_exist(self, excel_file):
        from aar.connectors.excel import read_excel
        from aar.failures import SchemaDriftError
        from aar.ir import ScanSpec as _Spec

        with pytest.raises(SchemaDriftError) as exc:
            read_excel(_Spec(kind="excel", path=excel_file, sheet="Ghost"))
        assert "Orders" in str(exc.value)


class TestCsvConnector:
    def test_reads_a_csv(self, orders, tmp_path):
        from aar.ir import ScanSpec as _Spec

        path = str(tmp_path / "orders.csv")
        with create_engine("arrow") as eng:
            eng.write(orders, Node(NodeType.WRITE, target=path,
                                   write_format="csv"))
            got = eng.read_scan(Node(NodeType.SCAN_CSV,
                                     scan=_Spec(kind="csv", path=path)))
        assert got.num_rows == 10
        assert "amount" in got.column_names


    def test_null_handle_drops_null_rows(self):
        from aar.runtime import Executor

        t = Table(pa.table({"id": pa.array([1, None, 3], type=pa.int64())}))
        node = Node(NodeType.NULL_HANDLE, inputs=[], null_strategy="drop")
        got = Executor()._null_handle(t, node)
        assert got.column("id").to_pylist() == [1, 3]

    def test_dedup_keeps_the_first_of_each_key(self):
        from aar.runtime import Executor

        dup = Table(pa.table({"k": pa.array([1, 1, 2, 2, 3], type=pa.int64())}))
        node = Node(NodeType.DEDUPLICATE, inputs=[], dedup_keys=("k",),
                    dedup_strategy="first")
        got = Executor()._dedup(dup, node)
        assert got.column("k").to_pylist() == [1, 2, 3]

    def test_quality_check_fails_on_a_null(self):
        """A not-null rule must trip when there really is a null."""
        from aar.failures import QualityCheckFailed
        from aar.runtime import Executor

        t = Table(pa.table({"amount": pa.array([1.0, None, 3.0],
                                              type=pa.float64())}))
        node = Node(NodeType.QUALITY_CHECK, inputs=[],
                    quality_rules=(("amount", "not_null"),))
        with pytest.raises(QualityCheckFailed) as exc:
            Executor()._quality(t, node)
        assert "amount" in str(exc.value)
        assert "1 null" in str(exc.value)


    def test_quality_check_passes_on_clean_data(self, orders):
        from aar.runtime import Executor

        node = Node(NodeType.QUALITY_CHECK, inputs=[],
                    quality_rules=(("amount", "positive"),))
        got = Executor()._quality(orders, node)
        assert got.num_rows == orders.num_rows

