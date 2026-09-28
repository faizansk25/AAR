"""The Arrow engine: the reference implementation of the engine contract.

It is also the fallback. Every operation is implemented against
``pyarrow.compute`` or, where Arrow has no kernel, against a small
correct Python path - so if DuckDB, Polars and pandas are all absent, AAR can
still run a pipeline, and says that is what it did.

That fallback is the specification's ninth principle made concrete:
graceful degradation, with the degradation recorded.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from ..capability import Device
from ..failures import SourceUnavailable
from ..interchange import Table, require_arrow
from ..ir import Agg, Col, Expr, JoinType, Lit, Node, NodeType
from .base import Engine, PredicateCompiler

__all__ = ["ArrowEngine"]


class ArrowEngine(Engine):
    """Executes directly against Arrow, with no third-party engine."""

    id = "arrow"
    device = Device.CPU

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self._pa = require_arrow()

    # ------------------------------------------------------------------ read
    def read_scan(self, node: Node) -> Table:
        spec = node.scan
        if spec is None:
            raise ValueError("scan node has no ScanSpec")
        kind = spec.kind
        if kind == "parquet":
            return self._read_parquet(spec.path, spec.columns)
        if kind == "csv":
            return self._read_csv(spec.path, spec.delimiter, spec.columns)
        if kind == "json":
            return self._read_json(spec.path)
        if kind == "excel":
            return self._read_excel(node)
        raise NotImplementedError(
            f"the Arrow engine cannot read a {kind!r} source; "
            f"install the engine that can")

    def _read_parquet(self, path: str | None,
                      columns: Sequence[str]) -> Table:
        import pyarrow.parquet as pq

        if not path or not os.path.exists(path):
            raise SourceUnavailable(f"no such Parquet file: {path}",
                                    path=str(path))
        table = pq.read_table(path, columns=list(columns) or None)
        return Table(table)

    def _read_csv(self, path: str | None, delimiter: str,
                 columns: Sequence[str]) -> Table:
        import pyarrow.csv as pacsv

        if not path or not os.path.exists(path):
            raise SourceUnavailable(f"no such CSV file: {path}", path=str(path))
        parse = pacsv.ParseOptions(delimiter=delimiter or ",")
        read = pacsv.ReadOptions(encoding="utf-8")
        table = pacsv.read_csv(path, parse_options=parse, read_options=read)
        if columns:
            table = table.select(list(columns))
        return Table(table)

    def _read_json(self, path: str | None) -> Table:
        import pyarrow.json as pajson

        if not path or not os.path.exists(path):
            raise SourceUnavailable(f"no such JSON file: {path}", path=str(path))
        return Table(pajson.read_json(path))

    def _read_excel(self, node: Node) -> Table:
        """Excel is the connector's job, not the engine's.

        Keeping one implementation of range addressing, header detection and
        error-cell handling matters more than the convenience of inlining it
        here: a second copy of that logic is a second set of bugs.
        """
        from ..connectors.excel import read_excel

        return read_excel(node.scan)


    # --------------------------------------------------------------- filter
    def filter(self, table: Table, predicate: Expr) -> Table:
        """Filter by evaluating the predicate.

        The common comparison forms are vectorised through an Arrow kernel;
        anything more exotic takes the row path, which is slower but always
        correct.
        """
        if predicate is None:
            return table
        if table.num_rows == 0:
            return table
        vectorised = self._vector_filter(table, predicate)
        if vectorised is not None:
            return vectorised
        return self._row_filter(table, predicate)

    def _vector_filter(self, table: Table, predicate: Expr) -> "Table | None":
        """Handle the usual comparison forms with an Arrow kernel."""
        import pyarrow.compute as pc
        from ..ir import BinOp

        if not isinstance(predicate, BinOp):
            return None
        if predicate.op not in ("=", "<>", ">", ">=", "<", "<="):
            return None
        left, right = predicate.left, predicate.right
        if not isinstance(left, Col) or not table.schema.has(left.name):
            return None
        column = table.column(left.name)

        if isinstance(right, Lit):
            value = right.value
            if value is None:
                return table.slice(0, 0)
            if not isinstance(value, str):
                sample = column[0].as_py() if len(column) else None
                if sample is not None and type(sample) is not type(value):
                    try:
                        value = type(sample)(value)
                    except (TypeError, ValueError):
                        return None
        elif isinstance(right, Col):
            if not table.schema.has(right.name):
                return None
            value = table.column(right.name)
        else:
            return None

        kernels = {
            "=": pc.equal, "<>": pc.not_equal, ">": pc.greater,
            ">=": pc.greater_equal, "<": pc.less, "<=": pc.less_equal,
        }
        try:
            mask = kernels[predicate.op](column, value)
        except Exception:  # noqa: BLE001 - any kernel refusal means "use the row path"
            return None
        return Table(table.arrow.filter(mask), table.schema)

    def _row_filter(self, table: Table, predicate: Expr) -> Table:
        rows = table.arrow.to_pylist()
        test = PredicateCompiler.compile(predicate)
        keep = [r for r in rows if test(r)]
        if len(keep) == len(rows):
            return table
        return Table(self._pa.Table.from_pylist(keep,
                                                 schema=table.arrow.schema),
                     table.schema)

    # -------------------------------------------------------------- project
    def project(self, table: Table, columns: Sequence[str]) -> Table:
        return table.select(list(columns))


    # -------------------------------------------------------------- group by
    def group_by(self, table: Table, keys: Sequence[str],
                 aggs: dict[str, Any]) -> Table:
        """Group and aggregate.

        Grouping runs in Python over row dicts. It is the slowest path in AAR
        and is reached only when the planner has nothing better - which is
        exactly when a slow, obviously-correct answer beats a fast wrong one.
        """
        key_names = list(keys)
        if table.num_rows == 0:
            return self._empty_group(table, key_names, aggs)

        buckets: dict[tuple, list[dict]] = {}
        order: list[tuple] = []
        for row in table.arrow.to_pylist():
            k = tuple(_hashable(row.get(name)) for name in key_names)
            if k not in buckets:
                buckets[k] = []
                order.append(k)
            buckets[k].append(row)

        out_rows: list[dict[str, Any]] = []
        for k in order:
            group = buckets[k]
            record = {name: group[0].get(name) for name in key_names}
            for name, agg in aggs.items():
                record[name] = _apply_aggregate(agg, group)
            out_rows.append(record)

        return _build_result(out_rows, key_names, list(aggs), table)

    def _empty_group(self, table: Table, keys: list[str],
                     aggs: dict[str, Any]) -> Table:
        """A group-by over no rows still yields the right columns and types."""
        import pyarrow as pa

        from ..interchange import canonical_to_arrow
        from ..types import Field, Schema

        fields = [pa.field(k, canonical_to_arrow(table.schema.get(k).type)
                           if table.schema.has(k) else pa.string())
                  for k in keys]
        arrow_schema = pa.schema(fields + [pa.field(a, pa.float64())
                                           for a in aggs])
        return Table(pa.Table.from_pylist([], schema=arrow_schema),
                     Schema(tuple(
                         Field(k, table.schema.get(k).type
                               if table.schema.has(k) else _canonical_string())
                         for k in keys)
                         + tuple(Field(a, _canonical_float()) for a in aggs)))

    # ----------------------------------------------------------------- sort
    def sort(self, table: Table, keys: Sequence[tuple[str, bool]]) -> Table:
        if not keys or table.num_rows == 0:
            return table
        return table.take(table.sort_indices(list(keys)))

    # ---------------------------------------------------------------- limit
    def limit(self, table: Table, n: int) -> Table:
        if n < 0:
            raise ValueError("limit must be non-negative")
        return table.slice(0, min(n, table.num_rows))


    # ----------------------------------------------------------------- join
    def join(self, left: Table, right: Table, keys: Sequence[str],
             how: str) -> Table:
        """Hash join in Python.

        Correct for every join type the IR can express, including the outer
        joins where a non-matching row must still be emitted with nulls - the
        case a naive implementation silently drops.
        """
        key_names = list(keys)
        for k in key_names:
            if not left.schema.has(k) or not right.schema.has(k):
                raise KeyError(f"join key {k!r} missing from an input")

        right_rows = right.arrow.to_pylist()
        right_by_key: dict[tuple, list[dict]] = {}
        for rrow in right_rows:
            right_by_key.setdefault(
                tuple(_hashable(rrow.get(k)) for k in key_names), []).append(rrow)

        left_rows = left.arrow.to_pylist()
        out: list[dict] = []
        for lrow in left_rows:
            k = tuple(_hashable(lrow.get(name)) for name in key_names)
            # SQL: NULL never equals NULL, not even to itself. Matching them
            # here would fabricate join rows that no engine agrees on.
            hits = right_by_key.get(k) if None not in k else None
            if hits:
                for rrow in hits:
                    merged = dict(lrow)
                    for name, value in rrow.items():
                        if name not in key_names:
                            merged[name] = value
                    out.append(merged)
            elif how in (JoinType.LEFT, JoinType.FULL):
                merged = dict(lrow)
                for name in right.column_names:
                    if name not in key_names:
                        merged[name] = None
                out.append(merged)

        if how in (JoinType.RIGHT, JoinType.FULL):
            left_keys = {tuple(_hashable(r.get(n)) for n in key_names)
                         for r in left_rows if None not in
                         tuple(_hashable(r.get(n)) for n in key_names)}
            for rrow in right_rows:
                k = tuple(_hashable(rrow.get(n)) for n in key_names)
                if k not in left_keys:
                    merged = {n: None for n in left.column_names
                              if n not in key_names}
                    for name in key_names:
                        merged[name] = rrow.get(name)
                    for name, value in rrow.items():
                        if name not in key_names:
                            merged[name] = value
                    out.append(merged)


        return _build_joined(out, left, right, key_names, how)

    # ------------------------------------------------------------------ udf
    def udf(self, table: Table, fn: Any, mode: str = "row") -> Table:
        """Apply a Python function to the batch.

        ``mode`` is explicit:

        * ``"row"`` (default) - ``fn(row_dict)`` once per record, returning one
          value. The record holds every column, so nothing is guessed.
        * ``"column"`` - ``fn(columns_dict)`` once, where ``columns_dict``
          maps each column name to its full list of values, returning one
          value per record.
        * ``"auto"`` - inspect the signature.

        The column form passes *all* columns rather than the first. Handing a
        caller one column and not saying which would let a UDF quietly
        operate on the wrong data, and a function like
        ``lambda v: [x * 2 for x in v]`` would produce a confidently wrong
        answer rather than an error.

        Neither default is inferred, because both conventions are
        one-parameter functions: ``def band(amount)`` is a row function that
        reads one field, and a column function with a short name. Any name- or
        arity-based rule picks the wrong one regularly.
        """
        rows = table.arrow.to_pylist()
        if not rows:
            return table
        out_name = _udf_output_name(fn)
        resolved = (mode or "row").lower()
        if resolved == "auto":
            resolved = "column" if _udf_wants_column(fn) else "row"
        if resolved == "column":
            columns = {name: [r.get(name) for r in rows]
                       for name in table.column_names}
            produced = list(fn(columns))
            if len(produced) != len(rows):
                raise ValueError(
                    f"column UDF {out_name!r} returned {len(produced)} "
                    f"value(s) for {len(rows)} input row(s); a column UDF "
                    f"must return exactly one value per row")
            for r, v in zip(rows, produced):
                r[out_name] = v
        elif resolved == "row":
            for r in rows:
                r[out_name] = fn(r)
        else:
            raise ValueError(
                f"unknown UDF mode {mode!r}; use 'row', 'column' or 'auto'")
        return _build_result(rows, list(table.column_names), [out_name], table)

    # ---------------------------------------------------------------- write
    def write(self, table: Table, node: Node) -> int:
        """Write to the target in ``node``. Returns rows written."""
        target = node.target
        if not target:
            raise ValueError("write node has no target")
        fmt = (node.write_format or "").lower()
        if fmt == "parquet":
            import pyarrow.parquet as pq

            parent = os.path.dirname(os.path.abspath(target))
            if parent:
                os.makedirs(parent, exist_ok=True)
            pq.write_table(table.arrow, target)
            return table.num_rows
        if fmt == "csv":
            import pyarrow.csv as pacsv

            parent = os.path.dirname(os.path.abspath(target))
            if parent:
                os.makedirs(parent, exist_ok=True)
            pacsv.write_csv(table.arrow, target)
            return table.num_rows
        if fmt == "excel":
            from ..connectors.excel import write_excel

            return write_excel(table, target, sheet=node.scan.sheet
                               if node.scan else None)
        raise NotImplementedError(
            f"the Arrow engine cannot write {fmt or 'that format'!r}")



# ------------------------------------------------------------------ helpers
def _hashable(value: Any) -> Any:
    """Make a value usable as a dict key.

    ``None`` stays ``None`` rather than becoming a string, so a SQL-style
    ``NULL`` does not join to the literal text ``"None"`` - a classic source
    of silently duplicated join rows.
    """
    if value is None:
        return None
    if isinstance(value, (list, dict, set)):
        return str(value)
    return value


def _apply_aggregate(agg: Any, group: list[dict]) -> Any:
    """Evaluate one aggregate over a group of row dicts."""
    from ..ir import Agg as AggExpr
    from ..ir import Col as ColExpr

    if not isinstance(agg, AggExpr):
        raise TypeError(f"expected an aggregate, got {type(agg).__name__}")
    name = agg.func.upper()

    if name == "COUNT":
        if agg.arg is None:
            return len(group)
        source = _values(agg.arg, group)
        if agg.distinct:
            return len({_hashable(v) for v in source if v is not None})
        return sum(1 for v in source if v is not None)

    values = [v for v in _values(agg.arg, group) if v is not None]
    if agg.distinct:
        seen: list[Any] = []
        for v in values:
            if v not in seen:
                seen.append(v)
        values = seen
    if not values:
        return None

    if name in ("SUM",):
        return sum(values)
    if name in ("AVG", "MEAN"):
        return sum(values) / len(values)
    if name == "MIN":
        return min(values)
    if name == "MAX":
        return max(values)
    if name in ("ANY", "FIRST"):
        return values[0]
    if name in ("ARBITRARY", "LAST"):
        return values[-1]
    if name == "LIST":
        return values
    raise NotImplementedError(f"unsupported aggregate {agg.func!r}")


def _values(expr: Any, group: list[dict]) -> list[Any]:
    from ..ir import Col, Lit

    if isinstance(expr, Col):
        return [r.get(expr.name) for r in group]
    if isinstance(expr, Lit):
        return [expr.value] * len(group)
    raise NotImplementedError(
        f"an aggregate argument must be a column or a literal, "
        f"got {type(expr).__name__}")


def _infer_arrow_type(values: list[Any]) -> Any:
    """Pick an Arrow type for Python values.

    Widening, never narrowing: a column containing one float keeps float
    even if the sample that was inspected happened to be an int.
    """
    import pyarrow as pa

    present = [v for v in values if v is not None]
    if not present:
        return pa.string()
    if all(isinstance(v, bool) for v in present):
        return pa.bool_()
    if all(isinstance(v, int) and not isinstance(v, bool) for v in present):
        return pa.int64()
    if all(isinstance(v, (int, float)) and not isinstance(v, bool)
           for v in present):
        return pa.float64()
    if all(isinstance(v, (int, float, str)) and not isinstance(v, bool)
           for v in present):
        return pa.string()
    return pa.string()


def _from_pa(arrow_type: Any) -> Any:
    from ..interchange import arrow_to_canonical
    return arrow_to_canonical(arrow_type)


def _canonical_string() -> Any:
    from ..types import UTF8
    return UTF8


def _canonical_float() -> Any:
    from ..types import FLOAT64
    return FLOAT64


def _field(name: str, arrow_type: Any) -> Any:
    from ..types import Field
    return Field(name, _from_pa(arrow_type))



def _udf_output_name(fn: Any) -> str:
    """The column a UDF writes to: its own name, or ``result`` if anonymous."""
    name = getattr(fn, "__name__", "") or ""
    if not name or name == "<lambda>":
        return "result"
    return name


def _udf_wants_column(fn: Any) -> bool:
    """Whether a UDF takes a column rather than a row.

    A function with exactly one *named* parameter is treated as a column
    function, which is the common vectorised-UDF convention. A function
    taking several named parameters (``fn(a, b)``) needs a row, so the
    convention cannot be guessed from arity alone.
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    positional = [
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) != 1:
        return False
    only = positional[0]
    # `fn(rows)` is ambiguous; `fn(*, ...)` and `fn(x=...)` are not.
    if only.kind is only.VAR_POSITIONAL:
        return False
    return only.name not in ("row", "record", "item", "r", "_row")


def _build_result(
    rows: list[dict],
    key_names: list[str],
    value_names: list[str],
    table: Table,
) -> Table:
    """Build a result table, keeping the input's key column types.

    Key types come from the input rather than being inferred from Python
    values, so an Int64 key does not silently become Float64 because one
    group happened to contain a null.
    """
    import pyarrow as pa

    from ..interchange import canonical_to_arrow
    from ..types import Field, Schema

    names = list(key_names) + [v for v in value_names if v not in key_names]
    fields = []
    for name in names:
        if name in key_names and table.schema.has(name):
            fields.append(pa.field(name,
                                   canonical_to_arrow(
                                       table.schema.get(name).type)))
        else:
            fields.append(pa.field(name, _infer_arrow_type(
                [r.get(name) for r in rows])))
    arrow_schema = pa.schema(fields)
    return Table(pa.Table.from_pylist(rows, schema=arrow_schema),
                 Schema(tuple(Field(f.name, _from_pa(f.type),
                                    nullable=f.nullable)
                              for f in arrow_schema)))


def _build_joined(out: list[dict], left: Table, right: Table,
                  keys: list[str], how: Any) -> Table:
    """Assemble a join result with a stable column order and inherited types.

    The key columns keep the *left* input's type (a join key should be one
    type, not two), and every other column takes the type of the side it came
    from, so a right-hand Int64 does not become Float64 just because the
    left-hand column was text.
    """
    import pyarrow as pa

    from ..interchange import canonical_to_arrow
    from ..types import Field, Schema

    names: list[str] = []
    for name in left.column_names:
        if name not in names:
            names.append(name)
    for name in right.column_names:
        if name not in names:
            names.append(name)

    fields = []
    for name in names:
        if name in keys and left.schema.has(name):
            fields.append(pa.field(name,
                                   canonical_to_arrow(
                                       left.schema.get(name).type),
                                   nullable=False))
        elif left.schema.has(name):
            fields.append(pa.field(name,
                                   canonical_to_arrow(
                                       left.schema.get(name).type)))
        elif right.schema.has(name):
            fields.append(pa.field(name, canonical_to_arrow(
                right.schema.get(name).type)))
        else:
            fields.append(pa.field(name, pa.string()))
    arrow_schema = pa.schema(fields)
    return Table(pa.Table.from_pylist(out, schema=arrow_schema),
                 Schema(tuple(Field(f.name, _from_pa(f.type),
                                    nullable=f.nullable)
                              for f in arrow_schema)))

