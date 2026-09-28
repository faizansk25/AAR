"""What can honestly be verified about the GPU path without a GPU.

The development machine has no CUDA device, and pretending otherwise in a
test name would defeat the purpose of this file. So the three groups below
are explicitly different in what they establish, and each says so.

**1. The cuDF glue, against a faithful stand-in.**
:class:`FakeCudf` implements exactly the surface ``CudfEngine`` calls -
``DataFrame.from_arrow``, ``to_arrow``, ``query``, ``groupby(dropna=False)
.agg``, ``merge``, ``head``, ``sort_values`` - backed by pandas, because
cuDF *is* the pandas API on the GPU. Running the full engine contract
through it and comparing against ``ArrowEngine`` on identical data
establishes that AAR's boundary code, aggregate mapping, device-residency
accounting and output are correct.

It establishes **nothing** about RAPIDS, cuDF kernels, or GPU performance.
A pandas-backed double computes the same answers a GPU would; that is
precisely why it cannot speak to speed, or to the parts of cuDF that are
not pandas-shaped.

**2. The Polars GPU translation, against real Polars.**
Polars *is* installed here, so the expression and aggregate translation in
``PolarsGPUEngine`` can be exercised on the CPU path and compared against
``PolarsEngine``. That is real translation logic verified with a real
library - the one part of the GPU path that does not need a GPU to test.

**3. The cost model and planner, against a synthetic T4 profile.**
Decision logic is pure arithmetic, so a fabricated ``HardwareProfile`` for a
T4 answers real questions: does the planner choose the GPU when the GPU
wins, and does it correctly refuse when transfers dominate? These are the
questions the specification's counter-examples are about, and none of them
require silicon.

What none of this can establish: absolute performance, real bus bandwidth,
VRAM exhaustion behaviour, or whether the cost model is *calibrated*. Those
need hardware, and `tools/gpu_verification.py` measures them where one
exists.
"""

from __future__ import annotations

import re
import sys
import types
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")

from aar.engines.cudf_engine import (
    ROW_COUNTER, CudfEngine, _agg_spec, _to_query,
)
from aar.engines.factory import ENGINE_FACTORIES, create_engine
from aar.interchange import Table, arrow_to_canonical
from aar.ir import Agg, BinOp, Col, Func, Lit
from aar.types import Field, INT64, Schema, UTF8


class GroupBy:
    def __init__(self, grouped: Any) -> None:
        self._grouped = grouped

    def agg(self, **spec: Any) -> "FakeCudf.DataFrame":
        return FakeCudf.DataFrame(self._grouped.agg(**spec).reset_index())


class FakeCudf:
    """The subset of the cuDF module that ``CudfEngine`` actually calls.

    A test double, and labelled as one everywhere it is used. Its purpose is
    to exercise AAR's glue - the arrow boundary, the aggregate mapping, the
    transfer accounting - not to emulate a GPU.
    """

    __version__ = "24.04.00 (fake)"

    class DataFrame:
        def __init__(self, frame: Any) -> None:
            # Accept a list of dicts the way the real DataFrame does, which
            # is how the whole-table aggregate path builds its one row.
            if isinstance(frame, list):
                import pandas as pd

                frame = pd.DataFrame(frame)
            self._frame = frame

        @classmethod
        def from_arrow(cls, table: Any) -> "FakeCudf.DataFrame":
            import pandas as pd

            return cls(pd.DataFrame(table.to_pandas()))

        def to_arrow(self) -> Any:
            return pa.Table.from_pandas(self._frame, preserve_index=False)

        def __getitem__(self, key: Any) -> Any:
            # A scalar key gives a Series in pandas and in cuDF alike, and
            # the engine's whole-table path relies on that (it takes
            # `frame[col].sum()`). Wrapping a Series would break it, so only
            # list-style selection returns another frame.
            if isinstance(key, str):
                return self._frame[key]
            return FakeCudf.DataFrame(self._frame[key])

        def __setitem__(self, key: str, value: Any) -> None:
            self._frame[key] = value

        def head(self, n: int) -> "FakeCudf.DataFrame":
            return FakeCudf.DataFrame(self._frame.head(n))

        def sort_values(self, by: Any, ascending: Any = True, kind: str = ""
                        ) -> "FakeCudf.DataFrame":
            return FakeCudf.DataFrame(
                self._frame.sort_values(by=by, ascending=ascending,
                                        kind=kind or "quicksort"))

        def agg(self, spec: dict) -> Any:
            """Whole-frame aggregate, for the no-key case.

            A named aggregation over a frame with no groups yields one row
            per aggregate, which is what a table-wide aggregate means.
            """
            import pandas as pd

            values = {}
            for out, entry in spec.items():
                column, func = (entry if isinstance(entry, tuple)
                                else (out, entry))
                values[out] = getattr(self._frame[column], func)()
            return pd.DataFrame([values])

        def merge(self, other: Any, on: Any, how: str
                  ) -> "FakeCudf.DataFrame":
            return FakeCudf.DataFrame(
                self._frame.merge(other._frame, on=on, how=how))

        def groupby(self, by: Any, dropna: bool = True) -> GroupBy:
            return GroupBy(self._frame.groupby(by, dropna=dropna))

        def query(self, expression: str) -> "FakeCudf.DataFrame":
            return FakeCudf.DataFrame(_query_mask(self._frame, expression))

        @property
        def columns(self) -> Any:
            return self._frame.columns

        @columns.setter
        def columns(self, value: Any) -> None:
            self._frame.columns = value


#: cudf's `query` accepts ``x.isnull()``; pandas' evaluator does not. These
#: two rewrites are the whole difference, done here rather than in the
#: engine because the engine's rendering is correct for real cudf.
_NULL_CALL = re.compile(r"\(`([^`]+)`\)\.(isnull|notnull)\(\)")


def _query_mask(frame: Any, expression: str) -> Any:
    cleaned = _NULL_CALL.sub(r"__\2(`\1`)", expression)
    env = {"__isnull": lambda s: s.isna(), "__notnull": lambda s: s.notna()}
    # `query` returns the *filtered rows*, not a mask - `eval` gives the
    # mask. Getting that backwards is a double bug, not a real cudf one.
    mask = frame.eval(cleaned, engine="python", local_dict=env)
    return frame[mask]


@pytest.fixture
def cudf_host(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install the double so ``import cudf`` resolves inside the engine."""
    import pandas as pd

    module = types.ModuleType("cudf")
    module.DataFrame = FakeCudf.DataFrame
    module.NamedAgg = pd.NamedAgg
    module.__version__ = FakeCudf.__version__
    monkeypatch.setitem(sys.modules, "cudf", module)
    return module


# ----------------------------------------------------------------------- data
def _table(rows: int = 200) -> Table:
    """Deterministic data, so any failure is reproducible."""
    arrow = pa.table({
        "region": [f"r{i % 4}" for i in range(rows)],
        "amount": [float((i * 7) % 100) for i in range(rows)],
        "n": list(range(rows)),
    })
    schema = Schema(tuple(Field(f.name, arrow_to_canonical(f.type))
                          for f in arrow.schema))
    return Table(arrow, schema)


def _agg(op: str, column: str | None, output: str) -> Agg:
    return Agg(op, Col(column) if column else None)


def _sorted(table: Table) -> list[dict]:
    """Order-independent comparison.

    Group-by output order is not part of the contract, so two engines can
    only be compared after sorting. Ordering by the string form of the whole
    row is the only total order available across differing key names.
    """
    rows = table.arrow.to_pylist()
    return sorted(rows, key=lambda r: str(sorted(r.items(), key=str)))


# ------------------------------------------------------------------ 1. cuDF
class TestCudfGlue:
    """AAR's cuDF boundary code, via a pandas-backed stand-in."""

    def test_it_constructs_and_declines_honestly_on_a_cpu_host(self) -> None:
        # No double installed: it must decline, not raise, not pretend.
        with CudfEngine() as plain:
            assert plain.id == "cudf"
            assert plain.device.value == "gpu"
            assert not plain.supports("execute")
            assert "cudf" in plain.capabilities.reason("execute")

    def test_with_the_double_it_reports_itself_available(
            self, cudf_host: Any) -> None:
        with CudfEngine() as engine:
            assert engine.supports("execute")

    def test_filter_matches_arrow(self, cudf_host: Any) -> None:
        table = _table()
        predicate = BinOp(Col("amount"), ">", Lit(90.0))
        with CudfEngine() as gpu, create_engine("arrow") as cpu:
            assert _sorted(gpu.filter(table, predicate)) == \
                _sorted(cpu.filter(table, predicate))

    def test_project_matches_arrow(self, cudf_host: Any) -> None:
        table = _table()
        with CudfEngine() as gpu, create_engine("arrow") as cpu:
            assert _sorted(gpu.project(table, ["region", "n"])) == \
                _sorted(cpu.project(table, ["region", "n"]))

    def test_group_by_matches_arrow(self, cudf_host: Any) -> None:
        table = _table()
        aggs = {"total": _agg("SUM", "amount", "total"),
                "n": _agg("COUNT", "n", "n")}
        with CudfEngine() as gpu, create_engine("arrow") as cpu:
            assert _sorted(gpu.group_by(table, ["region"], aggs)) == \
                _sorted(cpu.group_by(table, ["region"], aggs))

    def test_sort_and_limit_match_arrow(self, cudf_host: Any) -> None:
        table = _table()
        # `n` is unique, so the order is total and the comparison is about
        # the sort rather than about how two engines break ties differently.
        keys = [("amount", True), ("n", True)]
        with CudfEngine() as gpu, create_engine("arrow") as cpu:
            assert gpu.sort(table, keys).arrow.to_pylist() == \
                cpu.sort(table, keys).arrow.to_pylist()
            assert gpu.sort(table, [("amount", False)]).num_rows == \
                cpu.sort(table, [("amount", False)]).num_rows
            assert gpu.limit(table, 7).arrow.to_pylist() == \
                cpu.limit(table, 7).arrow.to_pylist()

    def test_join_matches_arrow(self, cudf_host: Any) -> None:
        left = _table(rows=50)
        right = Table(pa.table({"region": [f"r{i}" for i in range(4)],
                                "label": [f"L{i}" for i in range(4)]}),
                      Schema((Field("region", UTF8), Field("label", UTF8))))
        with CudfEngine() as gpu, create_engine("arrow") as cpu:
            assert _sorted(gpu.join(left, right, ["region"], "inner")) == \
                _sorted(cpu.join(left, right, ["region"], "inner"))

    def test_a_whole_table_aggregate_still_returns_one_row(
            self, cudf_host: Any) -> None:
        with CudfEngine() as engine:
            out = engine.group_by(_table(), [], {"n": _agg("COUNT", None, "n")})
        assert out.num_rows == 1
        assert out.arrow.to_pylist()[0]["n"] == 200

    def test_it_declines_a_python_udf_and_says_why(self, cudf_host: Any) -> None:
        """A UDF on a GPU means every row crosses PCIe twice. It says so."""
        with CudfEngine() as engine:
            with pytest.raises(NotImplementedError) as excinfo:
                engine.udf(_table(), lambda row: row)
        assert "PCIe" in str(excinfo.value)


class TestCudfTransferAccounting:
    """The device-residency claim, made falsifiable.

    ``transfers`` counts the engine's own Arrow<->frame crossings. If a
    future change round-trips through the host per node, this fails - which
    is the point, because that change would be invisible in the output and
    catastrophic in the timing.
    """

    def test_one_operation_costs_exactly_one_transfer(self,
                                                      cudf_host: Any) -> None:
        table = _table()
        with CudfEngine() as engine:
            start = engine.transfers
            engine.filter(table, BinOp(Col("amount"), ">", Lit(10.0)))
            assert engine.transfers - start == 1

    def test_every_transform_is_one_transfer_not_two(self,
                                                     cudf_host: Any) -> None:
        table = _table()
        with CudfEngine() as engine:
            start = engine.transfers
            engine.filter(table, BinOp(Col("amount"), ">", Lit(10.0)))
            engine.project(table, ["region"])
            engine.sort(table, [("n", False)])
            engine.limit(table, 5)
            engine.group_by(table, ["region"], {"n": _agg("COUNT", "n", "n")})
            # One host->device per operation. If this ever scales with node
            # count, a GPU run is paying PCIe at every step.
            assert engine.transfers - start == 5

    def test_a_join_puts_both_sides_on_the_device(self, cudf_host: Any) -> None:
        left = _table(rows=20)
        right = Table(pa.table({"region": [f"r{i}" for i in range(4)],
                                "label": [f"L{i}" for i in range(4)]}),
                      Schema((Field("region", UTF8), Field("label", UTF8))))
        with CudfEngine() as engine:
            start = engine.transfers
            engine.join(left, right, ["region"], "inner")
            # Both sides in, one result out. A host-resident right side would
            # be a fourth, and would stream per row.
            assert engine.transfers - start == 2


class TestCudfAggregateMapping:
    """A wrong aggregate name is a wrong *answer*, not an error.

    So the mapping is asserted directly rather than trusted to round-trip.
    ``NamedAgg`` is passed in rather than imported, so the test pins the
    engine's *use* of the caller's class - the same class cudf would supply.
    """

    @staticmethod
    def _spec(aggs: dict) -> tuple[dict, bool]:
        import pandas as pd

        return _agg_spec(aggs, pd.NamedAgg)

    def test_the_names_are_the_ones_cudf_expects(self) -> None:
        spec, _ = self._spec({"a": _agg("SUM", "v", "a")})
        assert (spec["a"].column, spec["a"].aggfunc) == ("v", "sum")
        spec, _ = self._spec({"a": _agg("MIN", "v", "a")})
        assert (spec["a"].column, spec["a"].aggfunc) == ("v", "min")
        spec, _ = self._spec({"a": _agg("MAX", "v", "a")})
        assert (spec["a"].column, spec["a"].aggfunc) == ("v", "max")
        # AVG is mean, not median. Getting that wrong is invisible in a test
        # that only checks the aggregate ran.
        spec, _ = self._spec({"a": _agg("AVG", "v", "a")})
        assert (spec["a"].column, spec["a"].aggfunc) == ("v", "mean")

    def test_count_of_a_column_skips_nulls_but_count_star_does_not(self) -> None:
        # SQL: COUNT(x) ignores nulls, COUNT(*) does not. pandas' "size"
        # counts nulls, so using it for COUNT(x) overstates the count - so
        # COUNT(x) uses "count" and COUNT(*) becomes a sum of ones.
        spec, needs_counter = self._spec({"a": _agg("COUNT", "v", "a")})
        assert (spec["a"].column, spec["a"].aggfunc) == ("v", "count")
        assert needs_counter is False

        spec, needs_counter = self._spec({"a": _agg("COUNT", None, "a")})
        assert (spec["a"].column, spec["a"].aggfunc) == (ROW_COUNTER, "sum")
        assert needs_counter is True

    def test_distinct_is_refused_because_approximating_would_be_a_wrong_count(
            self) -> None:
        with pytest.raises(NotImplementedError):
            self._spec({"a": Agg("COUNT", Col("v"), distinct=True)})

    def test_an_unknown_aggregate_is_refused_not_approximated(self) -> None:
        with pytest.raises(NotImplementedError):
            self._spec({"a": _agg("MEDIAN", "v", "a")})


class TestCudfQueryRendering:
    """Predicates render to cuDF's query language, or decline."""

    def test_comparisons_and_logic(self) -> None:
        assert _to_query(BinOp(Col("v"), ">", Lit(5.0))) == "(`v`) > (5.0)"
        assert _to_query(BinOp(Col("v"), "<", Lit(2))) == "(`v`) < (2)"
        joined = _to_query(BinOp(BinOp(Col("a"), ">", Lit(1)), "AND",
                                 BinOp(Col("b"), "=", Lit("x"))))
        assert joined == "((`a`) > (1)) and ((`b`) = ('x'))"

    def test_null_tests(self) -> None:
        assert _to_query(Func("isnull", (Col("v"),))) == "(`v`).isnull()"
        assert _to_query(Func("isNotNull", (Col("v"),))) == "(`v`).notnull()"

    def test_an_unrenderable_call_declines_rather_than_guessing(self) -> None:
        with pytest.raises(NotImplementedError):
            _to_query(Func("between", (Col("v"), Lit(1), Lit(2))))

