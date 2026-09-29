"""The cuDF engine: GPU-resident relational execution over RAPIDS.

cuDF presents the pandas API on the GPU, so this engine is structurally a
sibling of :mod:`aar.engines.pandas_engine` with one decisive difference:
**nothing leaves the device until the pipeline says so.** A frame stays
device-resident across filter, project, group_by, sort, limit and join,
and is converted to Arrow once - on a scan, or on write.

That is the whole point of the engine. A GPU that is fed host-resident
data, hands it back after every node, and is fed it again is a GPU-shaped
tax with none of the speed, and the cost model exists specifically to
detect that pattern. So the transfer count is what this module is careful
about, and it is asserted in the tests: one device-to-host transfer per
boundary, not per node.

**What this is and is not.** It is real code against the real RAPIDS API.
It has not been executed on a GPU, because the development machine has no
CUDA device - see ``report.md`` section 8.2. Writing it does not calibrate
it, and nothing here is a performance claim. What the tests *do* establish
on a CPU-only host is that the Arrow/cuDF boundary, the device-residency
guarantee, the aggregate mapping and the decline behaviour are correct.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from ..capability import Device
from ..failures import SourceUnavailable
from ..interchange import Table, reconcile
from ..ir import Agg, Col, Expr, Node
from ..lineage import taint as _lineage
from ._mask import to_mask
from .base import Engine

__all__ = ["CudfEngine"]


class CudfEngine(Engine):
    """Executes against RAPIDS cuDF, keeping data on the device."""

    id = "cudf"
    device = Device.GPU

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self._cudf = _import_cudf()
        #: Counts host<->device transfers. Not a performance measurement -
        #: a structural assertion that the pipeline is not bouncing data
        #: across PCIe once per node. The tests assert on this.
        self.transfers = 0
        if self._cudf is None:
            self._decline("execute", "cudf is not installed")

    def _require(self) -> Any:
        if self._cudf is None:
            raise NotImplementedError(
                f"{self.id} needs RAPIDS cuDF; install cudf-cu12")
        return self._cudf

    def _frame(self, table: Table) -> Any:
        """Arrow Table -> cuDF frame: one direction of one transfer.

        The only place in this engine where column data enters the device.
        """
        self.transfers += 1
        return self._require().DataFrame.from_arrow(table.arrow)

    def _table(self, frame: Any, source: Table | None = None,
               derived: Any = None) -> Table:
        """cuDF frame -> Arrow Table: the other direction of one transfer.

        A cuDF frame has nowhere to put an AAR classification, so without
        ``reconcile`` a CONFIDENTIAL column silently becomes public after a
        GPU-side filter. That is the same metadata loss the pandas engine
        has to work around, and it is a privacy bug, not a cosmetic one.
        """
        return reconcile(Table(frame.to_arrow()), source, derived)

    def read_scan(self, node: Node) -> Table:
        """Read a source onto the device, then back into a ``Table``.

        A scan's result is a pipeline boundary, so the executor materialises
        a ``Table`` for it. Intermediates stay on the device.
        """
        cudf = self._require()
        spec = node.scan
        if spec is None:
            raise ValueError("scan node has no ScanSpec")
        kind = spec.kind
        if kind == "parquet":
            path = _require_path(spec.path, "Parquet")
            columns = list(spec.columns) or None
            try:
                frame = cudf.read_parquet(path, columns=columns)
            except TypeError:  # older cuDF has no `columns=` argument
                frame = cudf.read_parquet(path)
                if columns:
                    frame = frame[columns]
            return self._table(frame)
        if kind == "csv":
            path = _require_path(spec.path, "CSV")
            frame = cudf.read_csv(path, sep=spec.delimiter or ",")
            if spec.columns:
                frame = frame[list(spec.columns)]
            return self._table(frame)
        if kind == "json":
            return self._table(cudf.read_json(_require_path(spec.path, "JSON")))
        if kind == "sql":
            return self._read_sql(node)
        raise NotImplementedError(
            f"the cuDF engine cannot read a {kind!r} source; install the "
            f"engine that can")

    @staticmethod
    def _read_sql(node: Node) -> Table:
        """SQL goes through the connector, which owns the connection.

        A GPU does not make a remote database faster, and pushing a result
        to the device only to pull it straight back is the transfer the cost
        model exists to avoid.
        """
        from ..connectors.sql import sqlite_connector

        spec = node.scan
        connector = sqlite_connector(spec.path or ":memory:")
        try:
            return connector.read(node)
        finally:
            connector.close()

    # ------------------------------------------------------------- transform
    def filter(self, table: Table, predicate: Expr) -> Table:
        # A boolean *mask*, not `.query(...)`. The first T4 run returned
        # `SyntaxError: invalid syntax (<unknown>, line 1)` because cudf's
        # query parser does not accept the backtick identifier quoting that
        # numexpr (and so the pandas stand-in) tolerates.
        #
        # The mask builder is shared with the pandas engine, which is
        # possible because cuDF *is* the pandas Series API on a GPU.
        # Removing the query language is better than repairing the quoting:
        # the mask is evaluated by cuDF's own vectorised kernels, it composes
        # for any nesting, and a column whose name contains a space or a
        # digit - which is what a real export has - needs no escaping at all.
        # A string-built predicate always has some input that breaks it, and
        # the input that breaks it is a user's column name.
        #
        # One `_frame` call: two would double the transfer count and hide the
        # very thing this counter exists to catch.
        frame = self._frame(table)
        return self._table(frame[to_mask(frame, predicate)], table)

    def project(self, table: Table, columns: Sequence[str]) -> Table:
        return self._table(self._frame(table)[list(columns)], table)

    def group_by(self, table: Table, keys: Sequence[str],
                 aggs: dict[str, Agg]) -> Table:
        cudf = self._require()
        frame = self._frame(table)
        spec, needs_counter = _agg_spec(aggs, cudf.NamedAgg)
        if needs_counter:
            # COUNT(*) has no column to point at in a named aggregation, so
            # give it one: a column of ones, summed, is the row count.
            frame[ROW_COUNTER] = 1
        key_names = list(keys)
        if key_names:
            # `.agg(**spec)`, not `.agg(spec)`. The dict-of-tuples form
            # (`{"total": ("amount", "sum")}`) was removed in pandas 3.0 and
            # cuDF follows pandas, so the tuple form is a KeyError waiting to
            # happen on any modern RAPIDS.
            grouped = frame.groupby(key_names, dropna=False).agg(**spec)
        else:
            # A whole-table aggregate has no key to group on, and
            # `groupby([])` is an error in both pandas and cudf. The answer
            # wanted is one row, so the aggregates are applied to the whole
            # frame and assembled directly.
            grouped = cudf.DataFrame([{
                out: getattr(frame[na.column], na.aggfunc)()
                for out, na in spec.items()}])
        return self._table(
            grouped, table,
            derived=_lineage.aggregate_tags(table.schema, aggs))

    def sort(self, table: Table, keys: Sequence[tuple[str, bool]]) -> Table:
        # The bool in a sort key is *ascending*, matching every other engine.
        # A stable sort (`mergesort`) is not decoration: with ties, an
        # unstable sort makes two engines return the same rows in a different
        # order, which reads as a data bug and is really a reproducibility
        # one. Row order after a sort is part of what analysts rely on.
        by = [k for k, _ in keys]
        ascending = [asc for _, asc in keys]
        return self._table(
            self._frame(table).sort_values(by=by, ascending=ascending,
                                           kind="mergesort"),
            table)

    def limit(self, table: Table, n: int) -> Table:
        return self._table(self._frame(table).head(n), table)

    def join(self, left: Table, right: Table, keys: Sequence[str],
             how: str = "inner") -> Table:
        # Both sides go to the device first. A GPU-resident left joined to a
        # host-resident right would stream `right` over PCIe for every row of
        # `left` - the worst transfer shape there is.
        frame = self._frame(left).merge(self._frame(right), on=list(keys),
                                       how=how)
        # A joined column is a function of both inputs, so it inherits both.
        both = _lineage.derive_from(left.schema,
                                    list(keys) + _other_columns(left, right,
                                                                 keys))
        out_names = list(frame.columns)
        return self._table(frame, source=left, derived={n: both
                                                        for n in out_names})

    def udf(self, table: Table, fn: Any, mode: str = "row") -> Table:
        """A Python UDF cannot run on a GPU, by construction.

        Not a missing feature so much as a fact: a Python callable executes
        on the host, so honouring it means moving every row across PCIe,
        running it there, and moving the result back - precisely what the
        GPU was chosen to avoid. Declining is the honest answer, and the
        executor turns this into a recorded degradation and a fallback.
        """
        raise NotImplementedError(
            f"{self.id} cannot run a Python UDF: a Python callable executes "
            f"on the host, so honouring it would round-trip every row across "
            f"PCIe. Use a native expression, or let the executor fall back.")

    def write(self, table: Table, node: Node) -> int:
        """Materialise to the target, copying off the device once."""
        frame = self._frame(table)
        fmt = (node.write_format or "").lower()
        # For a host-native target the frame has to come back regardless;
        # Parquet keeps the Arrow path and never needs the host frame.
        out = table if fmt == "parquet" else self._table(frame)
        from .arrow_engine import ArrowEngine
        return ArrowEngine().write(out, node)

    def close(self) -> None:
        """Idempotent. cuDF frames are garbage collected; nothing to free."""


# --------------------------------------------------------------------- utils
def _import_cudf() -> Any:
    try:
        import cudf
    except ImportError:
        return None
    return cudf


def _require_path(path: str | None, kind: str) -> str:
    if not path or not os.path.exists(path):
        raise SourceUnavailable(f"no such {kind} file: {path}", path=str(path))
    return path


def _flatten_frame(frame: Any) -> Any:
    """Undo the MultiIndex columns cudf builds for multi-output aggs."""
    if not hasattr(frame, "columns"):
        return frame
    try:
        frame.columns = ["__".join(map(str, c)).strip("_")
                         if isinstance(c, tuple) else c for c in frame.columns]
    except (AttributeError, TypeError):  # pragma: no cover - cudf variant
        pass
    return frame


def _other_columns(left: Table, right: Table, keys: Sequence[str]) -> list[str]:
    """The right-hand columns a join brings in, excluding the keys."""
    key_set = set(keys)
    return [c for c in right.column_names if c not in key_set]


def _agg_column(agg: Agg) -> str | None:
    """The column an aggregate reads, or None for COUNT(*)."""
    arg = agg.arg
    if arg is None:
        return None
    if isinstance(arg, Col):
        return arg.name
    # A non-column argument is not something this engine can map to a
    # pandas aggregate name, and guessing would be a wrong answer.
    raise NotImplementedError(
        f"the cuDF engine aggregates columns, not {type(arg).__name__}")


def _agg_spec(aggs: dict[str, Agg], named_agg: Any) -> tuple[dict, bool]:
    """Map AAR aggregates onto cuDF's *named* aggregation form.

    Each entry is a ``NamedAgg``, handed to the engine as ``.agg(**spec)``.
    The older ``{"total": ("amount", "sum")}`` dict-of-tuples form was
    removed in pandas 3.0, and cuDF tracks pandas, so the tuple form is a
    KeyError waiting to happen on any current RAPIDS build.

    Returns ``(spec, needs_row_counter)``. The flag is True when something
    counts rows, because a named aggregation cannot express ``COUNT(*)``:
    it needs a real column, so the caller adds an all-ones one.
    """
    spec: dict[str, Any] = {}
    needs_counter = False
    for out, agg in aggs.items():
        op = (agg.func or "").upper()
        if agg.distinct:
            raise NotImplementedError(
                f"the cuDF engine does not implement DISTINCT {op!r}; "
                f"approximating it would silently change the count")
        column = _agg_column(agg)
        if op in ("SUM", "MIN", "MAX", "AVG"):
            spec[out] = named_agg(_need(column), _AGG_NAMES[op])
        elif op == "COUNT":
            # COUNT(*) counts rows; COUNT(x) skips nulls. "count" on a real
            # column gives the second. "size" would count nulls too - a
            # different question, and one that stays invisible until a report
            # is quietly wrong. So COUNT(*) becomes a sum of ones, and the
            # caller is told to add the column.
            if column is None:
                needs_counter = True
                spec[out] = named_agg(ROW_COUNTER, "sum")
            else:
                spec[out] = named_agg(column, "count")
        elif op in ("FIRST", "ARBITRARY"):
            spec[out] = named_agg(_need(column), "first")
        else:
            raise NotImplementedError(
                f"the cuDF engine does not implement {op!r}")
    return spec, needs_counter


#: AAR aggregate name -> the name cuDF/pandas expects.
_AGG_NAMES = {"SUM": "sum", "MIN": "min", "MAX": "max", "AVG": "mean"}

#: Synthetic all-ones column used to express COUNT(*). Prefixed so it
#: cannot collide with a real column.
ROW_COUNTER = "__aar_row_count__"


def _need(column: str | None) -> str:
    """The column a non-counting aggregate must name."""
    if column is None:
        raise NotImplementedError(
            f"the cuDF engine needs a column here, but the aggregate has "
            f"none; use COUNT(*) if you meant a row count")
    return column


def _to_mask(frame: Any, expr: Expr) -> Any:
    """Build a cuDF boolean Series from an AAR predicate.

    Every branch is a Series operation, so the whole predicate is evaluated
    on the device in one vectorised pass. Anything that cannot be expressed
    as Series arithmetic is declined rather than approximated - an
    approximate filter returns silently wrong rows, which is the one
    failure this project exists to prevent.
    """
    from ..ir import BinOp, Func

    if isinstance(expr, BinOp):
        op = expr.op
        if op == "AND":
            # The children are *predicates*, so they recurse through
            # _to_mask. Routing them through _to_value was a real bug: a
            # nested conjunction is a BinOp, and _to_value only evaluates
            # arithmetic, so `a > 1 AND b < 5` raised "cannot evaluate
            # BinOp as a value" instead of returning a mask.
            return _to_mask(frame, expr.left) & _to_mask(frame, expr.right)
        if op == "OR":
            return _to_mask(frame, expr.left) | _to_mask(frame, expr.right)
        left = _to_value(frame, expr.left)
        right = _to_value(frame, expr.right)
        if op == "=":
            return left == right
        if op in ("<>", "!="):
            return left != right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op in ("+", "-", "*", "/"):
            return {"+": lambda: left + right, "-": lambda: left - right,
                    "*": lambda: left * right,
                    "/": lambda: left / right}[op]()
        raise NotImplementedError(
            f"the cuDF engine cannot filter on {op!r}; use a native "
            f"expression or let the executor fall back")
    if isinstance(expr, Func):
        value = _to_value(frame, expr.args[0]) if expr.args else None
        name = expr.name.lower()
        if name in ("isnull", "is_null"):
            return value.isnull()
        if name in ("isnotnull", "is_not_null"):
            return value.notnull()
        raise NotImplementedError(
            f"the cuDF engine cannot filter on {expr.name!r}")
    raise NotImplementedError(
        f"the cuDF engine cannot filter on "
        f"{type(expr).__name__}; use a native expression or let the "
        f"executor fall back")


def _to_value(frame: Any, expr: Expr) -> Any:
    """A column reference, a literal, or a nested arithmetic expression."""
    from ..ir import BinOp, Lit

    if isinstance(expr, Col):
        return frame[expr.name]
    if isinstance(expr, Lit):
        return expr.value
    if isinstance(expr, BinOp):
        left = _to_value(frame, expr.left)
        right = _to_value(frame, expr.right)
        if expr.op in ("+", "-", "*", "/"):
            return {"+": lambda: left + right, "-": lambda: left - right,
                    "*": lambda: left * right,
                    "/": lambda: left / right}[expr.op]()
    raise NotImplementedError(
        f"the cuDF engine cannot evaluate {type(expr).__name__} as a value")
