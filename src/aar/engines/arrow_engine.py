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
from ..lineage import taint as _lineage
from .base import Engine, PredicateCompiler

__all__ = ["ArrowEngine"]


class _Unsupported(Exception):
    """The predicate has no Arrow kernel here; take the row path.

    Internal and deliberately not a ``NotImplementedError``: this is not a
    missing feature, it is the normal hand-off between a fast path and a
    slow one that is always correct.
    """


#: Comparison op -> the Arrow compute kernel. A missing entry is the
#: difference between a millisecond and a Python loop over every row.
_KERNELS: dict[str, Any] = {}

#: Arithmetic op -> kernel, used for the nested case ``a > 1 AND b * 2 > 3``.
_ARITHMETIC: dict[str, Any] = {}


def _install_kernels() -> None:
    """Bind the kernels once, lazily, so importing this module is cheap."""
    import pyarrow.compute as pc

    _KERNELS.update({
        "=": pc.equal, "<>": pc.not_equal, ">": pc.greater,
        ">=": pc.greater_equal, "<": pc.less, "<=": pc.less_equal,
    })
    _ARITHMETIC.update({
        "+": pc.add_checked, "-": pc.subtract_checked,
        "*": pc.multiply_checked, "/": pc.divide,
    })


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
        if kind == "sql":
            return self._read_sql(node)
        if kind == "mongo":
            return self._read_mongo(node)
        raise NotImplementedError(
            f"the Arrow engine cannot read a {kind!r} source; "
            f"install the engine that can")

    @staticmethod
    def _read_sql(node: Node) -> Table:
        """SQL sources go through the connector, which owns the connection.

        A SQLite path needs no server, so this path is exercised against a
        real database in the test suite rather than against a stub.
        """
        from ..connectors.sql import sqlite_connector

        spec = node.scan
        connector = sqlite_connector(spec.path or ":memory:")
        try:
            return connector.read(node)
        finally:
            connector.close()

    @staticmethod
    def _read_mongo(node: Node) -> Table:
        from ..connectors.mongo import mock_mongo_connector, mongo_connector

        spec = node.scan
        connector = (mongo_connector(spec.dsn, spec.database)
                     if spec.dsn else mock_mongo_connector())
        try:
            return connector.read(node)
        finally:
            connector.close()

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
        """Build a boolean mask with Arrow kernels, recursively.

        This used to handle exactly one shape: a single comparison with a
        column on the left. Anything else - notably a conjunction - fell
        through to ``_row_filter``, which materialises every row as a Python
        dict and calls a Python predicate per row. On 500,000 rows a nested
        ``AND`` measured **20,114 ms** against DuckDB's 37 ms: a 3,936x
        gap, reproducible to within 0.4% spread. The single-comparison case
        looked fine at 17 ms, which is exactly why it survived - the
        benchmark had to try the *nested* shape to see it.

        Now recursive, so ``a AND b``, ``a OR b``, null tests and nested
        arithmetic all take the kernel path. Returning ``None`` still means
        "use the row path", so an exotic predicate is slower rather than
        wrong.
        """
        import pyarrow as pa
        import pyarrow.compute as pc

        from ..ir import BinOp, Col, Func, Lit

        if not _KERNELS:
            _install_kernels()
        arrow = table.arrow

        def value_of(expr: Any) -> Any:
            if isinstance(expr, Col):
                if not table.schema.has(expr.name):
                    raise _Unsupported()
                return table.column(expr.name)
            if isinstance(expr, Lit):
                if expr.value is None:
                    # A null operand in a comparison is an IS NULL test, not
                    # a value; a null in arithmetic has no Arrow answer here.
                    raise _Unsupported()
                literal = expr.value
                if not isinstance(literal, str) and len(arrow):
                    sample = arrow.column(0)[0].as_py()
                    if sample is not None and \
                            type(sample) is not type(literal):
                        try:
                            literal = type(sample)(literal)
                        except (TypeError, ValueError):
                            pass
                return literal
            if isinstance(expr, BinOp) and expr.op in _ARITHMETIC:
                return _ARITHMETIC[expr.op](
                    pc.cast(value_of(expr.left), pa.float64()),
                    pc.cast(value_of(expr.right), pa.float64()))
            raise _Unsupported()

        def mask_of(expr: Any) -> Any:
            if isinstance(expr, BinOp):
                op = expr.op
                if op == "AND":
                    # kleene, so a null operand stays unknown rather than
                    # silently becoming false.
                    return pc.and_kleene(mask_of(expr.left),
                                          mask_of(expr.right))
                if op == "OR":
                    return pc.or_kleene(mask_of(expr.left),
                                        mask_of(expr.right))
                if op in _KERNELS:
                    return _KERNELS[op](value_of(expr.left),
                                        value_of(expr.right))
                raise _Unsupported()
            if isinstance(expr, Func):
                name = expr.name.lower()
                if name in ("isnull", "is_null"):
                    return pc.is_null(value_of(expr.args[0]))
                if name in ("isnotnull", "is_not_null"):
                    return pc.is_valid(value_of(expr.args[0]))
                raise _Unsupported()
            raise _Unsupported()

        try:
            mask = mask_of(predicate)
        except _Unsupported:
            return None
        except Exception:  # noqa: BLE001 - any kernel refusal means row path
            return None
        if len(mask) != arrow.num_rows:
            return None
        return Table(arrow.filter(mask), table.schema)

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

        The vectorised path is tried first. ``pa.TableGroupBy`` is Arrow's
        own hash aggregation and runs in C, which is the difference between
        seconds and minutes on a few million rows.

        This matters more than a typical optimisation. The Arrow engine is
        the *fallback*: it is what runs when DuckDB, Polars and pandas are
        all absent. A fallback that takes ten minutes on three million rows
        is not a fallback, it is a hang - and "graceful degradation" (the
        specification's ninth principle) is a promise that the degraded path
        is slower, not that it never finishes.

        The Python path is kept for the aggregates Arrow cannot express,
        which is the honest split: use the fast kernel where it applies,
        and stay correct where it does not.
        """
        key_names = list(keys)
        if table.num_rows == 0:
            return self._empty_group(table, key_names, aggs)

        fast = _arrow_group_by(table, key_names, aggs)
        if fast is not None:
            return fast

        return self._group_by_python(table, key_names, aggs)

    def _group_by_python(self, table: Table, key_names: list,
                         aggs: dict[str, Any]) -> Table:
        """The row-dict path: always correct, and no longer reckless.

        Reached only when Arrow's grouping kernel is unusable. It is retained
        rather than deleted because correctness must not depend on which
        kernel happens to be available on a given machine.

        The important change is the projection. The obvious implementation
        calls ``table.arrow.to_pylist()``, which materialises *every* column
        as a Python object - on the 3.07M-row NYC taxi file that is 19
        columns, roughly 58 million Python objects, and a group-by that needs
        three of them took over ten minutes. Selecting the key columns and
        the aggregate arguments first, and converting only those, is the
        difference between unusable and slow.
        """
        from ..ir import Col as ColExpr

        # Project first. Converting 19 columns to Python when a group-by
        # needs three is the difference between slow and unusable.
        needed = list(key_names)
        for agg in aggs.values():
            if isinstance(getattr(agg, "arg", None), ColExpr):
                needed.append(agg.arg.name)
        keep = [n for n in dict.fromkeys(needed) if n in table.column_names]
        source = table.arrow.select(keep) if keep else table.arrow

        key_values = {n: source.column(n).to_pylist() for n in key_names}
        arg_values = {n: source.column(n).to_pylist()
                      for n in dict.fromkeys(needed)
                      if n not in key_values}

        order: list = []
        buckets: dict = {}
        for index in range(source.num_rows):
            k = tuple(_hashable(key_values[n][index]) for n in key_names)
            if k not in buckets:
                buckets[k] = []
                order.append(k)
            buckets[k].append(index)

        columns: dict[str, list] = {n: [] for n in key_names}
        for k in order:
            first = buckets[k][0]
            for n in key_names:
                columns[n].append(key_values[n][first])
        for name, agg in aggs.items():
            columns[name] = _aggregate_by_index(agg, arg_values, buckets,
                                                 order)

        import pyarrow as pa

        from ..types import Field, Schema

        arrow = pa.table(columns)
        schema = Schema(tuple(
            Field(f.name, _from_pa(f.type), nullable=f.nullable,
                  classification=(_key_tag(table.schema, f.name)
                                  if f.name in key_names
                                  else _tag_for(table.schema, f.name, aggs)))
            for f in arrow.schema))
        return Table(arrow, schema)





    def _empty_group(self, table: Table, keys: list[str],
                     aggs: dict[str, Any]) -> Table:
        """A group-by over no rows still yields the right columns and types.

        It must also yield the right *classification*. This path used to
        build the aggregate columns with no tags, so a group-by over an
        empty table returned `SUM(confidential)` as an unlabelled column on
        the Arrow and pandas engines while DuckDB and Polars labelled it
        correctly. The data is empty either way, which is exactly why it
        went unnoticed - but a policy that masks on classification would
        disagree with itself depending on which engine the planner chose.
        """
        import pyarrow as pa

        from ..interchange import canonical_to_arrow
        from ..types import Field, Schema

        fields = [pa.field(k, canonical_to_arrow(table.schema.get(k).type)
                           if table.schema.has(k) else pa.string())
                  for k in keys]
        arrow_schema = pa.schema(fields + [pa.field(a, pa.float64())
                                           for a in aggs])
        derived = _aggregate_tags(table.schema, aggs)
        return Table(pa.Table.from_pylist([], schema=arrow_schema),
                     Schema(tuple(
                         Field(k, table.schema.get(k).type
                               if table.schema.has(k)
                               else _canonical_string())
                         for k in keys)
                     + tuple(Field(a, _canonical_float(),
                                   classification=derived.get(a, frozenset()))
                             for a in aggs)))

    # ----------------------------------------------------------------- sort
    def sort(self, table: Table, keys: Sequence[tuple[str, bool]]) -> Table:
        """Order rows, keeping every column's classification.

        Sorting is a kernel, not a method: ``pa.Table`` has no
        ``sort_indices`` - it is ``pyarrow.compute.sort_indices`` - and its
        sort order is an enum whose *spelling* differs from Arrow's docs in a
        way that raises rather than falling back. Calling it on the wrapper
        was a latent crash on any non-empty sort: the reference engine
        failing at the one operation every other engine could do.
        """
        if not keys or table.num_rows == 0:
            return table
        import pyarrow.compute as pc

        order = [(str(name), "ascending" if asc else "descending")
                 for name, asc in keys]
        return table.take(pc.sort_indices(table.arrow, order))

    # ---------------------------------------------------------------- limit
    def limit(self, table: Table, n: int) -> Table:
        if n < 0:
            raise ValueError("limit must be non-negative")
        return table.slice(0, min(n, table.num_rows))


    # ----------------------------------------------------------------- join
    def join(self, left: Table, right: Table, keys: Sequence[str],
             how: str) -> Table:
        """Join with Arrow's native hash join, or the Python path.

        The Python path is correct for every join type the IR can express,
        including the outer joins where a non-matching row must still be
        emitted with nulls - the case a naive implementation silently drops.
        It is also *slow*: it materialises both sides as Python dicts, so
        1,000,000 rows by 19 columns is roughly 19 million Python objects.
        Measured on real NYC taxi data it took **4,462 ms** against polars'
        100 ms, a 45x gap that the isolated benchmark is now precise enough
        to trust.

        So the native ``pyarrow.Table.join`` is tried first, and the Python
        path is kept for the two cases where Arrow's answer is not the
        answer this system owes:

        1. **NULL in a key.** SQL says NULL never equals NULL, not even to
           itself; the Python path implements that deliberately, and Arrow's
           hash join matches null keys to null keys. Matching them would
           fabricate rows no other engine produces, so a null key anywhere
           in a key column routes to the Python path.
        2. **A non-key column present on both sides.** Arrow disambiguates
           by appending ``_left``/``_right``; this system's contract is that
           the right side wins. Rather than change the contract to match the
           library, the collision routes to the Python path.

        Both guards are cheap - a null count and a set intersection - and
        both describe dirty or unusual data, which is not the case a
        performance fix is for.
        """
        key_names = list(keys)
        for k in key_names:
            if not left.schema.has(k) or not right.schema.has(k):
                raise KeyError(f"join key {k!r} missing from an input")

        native = self._native_join(left, right, key_names, how)
        if native is not None:
            return native
        return self._python_join(left, right, key_names, how)

    def _native_join(self, left: Table, right: Table, key_names: list[str],
                     how: str) -> "Table | None":
        """``pyarrow.Table.join``, or ``None`` where its semantics differ."""
        import pyarrow as pa

        from ..types import Field, Schema

        # SEMI/ANTI/CROSS/ASOF have no Arrow equivalent here, and an
        # approximation would be a wrong answer rather than a slow one.
        #
        # `how` arrives as a JoinType member, and JoinType is a `str` mixin
        # enum: `str(JoinType.INNER)` is "JoinType.INNER" on Python 3.11+,
        # not "inner", so the lookup has to unwrap `.value` explicitly. The
        # first version of this compared `str(how)` against plain names,
        # never matched, and silently ran the Python path for every join -
        # which is exactly the kind of no-op that looks like a successful
        # change because the tests still pass.
        kind = getattr(how, "value", how)
        mapped = {"inner": "inner", "left": "left outer",
                  "right": "right outer", "full": "full outer"}
        join_type = mapped.get(str(kind))
        if join_type is None:
            return None

        for side, name in ((left, "left"), (right, "right")):
            for key in key_names:
                if side.column(key).null_count:
                    return None
        overlapping = (set(left.column_names) & set(right.column_names)
                       - set(key_names))
        if overlapping:
            return None

        try:
            result = left.arrow.join(
                right.arrow, keys=key_names, join_type=join_type,
                coalesce_keys=True)
        except Exception:  # noqa: BLE001 - any refusal means the slow path
            return None

        # A joined column is a function of both inputs, so it inherits both
        # sides' tags - the same union DuckDB and Polars apply, which is
        # what the cross-engine metadata tests assert. Every non-key output
        # column gets the same union because every one of them is a value
        # produced by a row of each side.
        from ..lineage.taint import derive_from

        left_cols = [c for c in left.column_names if c not in key_names]
        right_cols = [c for c in right.column_names if c not in key_names]
        # From *both* schemas. Passing the right side's column names to the
        # left side's schema silently drops them - a column absent from a
        # schema does not raise, it is simply not found - which is how the
        # first version lost `pii` on `bonus` and the cross-engine metadata
        # test caught it.
        both = (derive_from(left.schema, left_cols)
                | derive_from(right.schema, right_cols))
        schema = Schema(tuple(
            Field(
                f.name, _from_pa(f.type), nullable=f.nullable,
                # A key is a value taken from the left input, so it keeps its
                # own tags; anything else is joined, and inherits.
                classification=(
                    left.schema.get(f.name).classification
                    if f.name in key_names and left.schema.has(f.name)
                    else both)
            ) for f in result.schema))
        return Table(result, schema)

    def _python_join(self, left: Table, right: Table, key_names: list[str],
                     how: str) -> Table:
        """The original implementation: always correct, and slow."""
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
        # A UDF is opaque: the only sound assumption is that it read every
        # column it was handed, so the result inherits all of them.
        return _build_result(rows, list(table.column_names), [out_name],
                             table, {out_name: _lineage.inherit_all(
                                 table.schema)})


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


def _aggregate_by_index(agg: Any, values: dict[str, list],
                        buckets: dict, order: list) -> list:
    """Apply one aggregate to each group, from projected column lists.

    The same semantics as :func:`_apply_aggregate`, but driven by row
    indices over already-projected columns rather than by row dicts. The
    semantics are the part that must not drift: nulls are skipped by every
    aggregate except COUNT(*), and an all-null group yields ``None`` rather
    than zero.
    """
    from ..ir import Agg as AggExpr
    from ..ir import Col, Lit

    name = agg.func.upper()
    out: list = []
    for key in order:
        indices = buckets[key]
        if name == "COUNT" and agg.arg is None:
            out.append(len(indices))
            continue
        if isinstance(agg.arg, Lit):
            group = [agg.arg.value] * len(indices)
        elif isinstance(agg.arg, Col):
            column = values.get(agg.arg.name)
            group = [None if column is None else column[i] for i in indices]
        else:
            raise NotImplementedError(
                f"an aggregate argument must be a column or a literal, "
                f"got {type(agg.arg).__name__}")
        if agg.distinct:
            seen: list = []
            for v in group:
                if v is not None and v not in seen:
                    seen.append(v)
            group = seen
        else:
            group = [v for v in group if v is not None]
        if name == "COUNT":
            out.append(len(group))
        elif not group:
            out.append(None)
        elif name == "SUM":
            out.append(sum(group))
        elif name in ("AVG", "MEAN"):
            out.append(sum(group) / len(group))
        elif name == "MIN":
            out.append(min(group))
        elif name == "MAX":
            out.append(max(group))
        elif name in ("ANY", "FIRST"):
            out.append(group[0])
        elif name in ("ARBITRARY", "LAST"):
            out.append(group[-1])
        elif name == "LIST":
            out.append(group)
        else:
            raise NotImplementedError(
                f"unsupported aggregate {agg.func!r}")
    return out


#: Whether Arrow's own grouping kernel is usable, decided once.
#:
#: pyarrow 25.0.1 raises ``AttributeError: 'tuple' object has no attribute
#: 'startswith'`` for *every* documented spelling of ``TableGroupBy.aggregate``
#: - the legacy list-of-tuples form, the dict form, a bare string key, and
#: every count/min/max spelling. That is a defect in the installed library,
#: not a usage error, so AAR works around it rather than assuming it away.
#:
#: The verdict is probed on a two-row synthetic table rather than on real
#: data, and cached, because the probe on a multi-million-row table is not
#: cheap: it does a great deal of work *before* it fails. Probing per call
#: turned a fast failure into a hang.
_GROUP_BY_KERNEL: bool | None = None


def _group_by_kernel_works() -> bool:
    """One-time probe: can this pyarrow's grouping kernel be used at all?"""
    global _GROUP_BY_KERNEL
    if _GROUP_BY_KERNEL is not None:
        return _GROUP_BY_KERNEL
    try:
        import pyarrow as pa

        probe = pa.table({"k": ["a", "b"], "v": [1, 2]})
        out = probe.group_by(["k"]).aggregate({"total": ("sum", "v")})
        _GROUP_BY_KERNEL = out.num_rows == 2
    except Exception:  # noqa: BLE001
        _GROUP_BY_KERNEL = False
    return _GROUP_BY_KERNEL


def _arrow_group_by(table: Table, keys: list[str],
                    aggs: dict[str, Any]) -> Table | None:
    """Aggregate with Arrow's own hash kernel, or decline.

    ``None`` means "Arrow cannot express this" and the caller should use the
    Python path. Declining is always the safe answer for the same reason it
    is in the SQL connector: a near-miss aggregation returns subtly wrong
    numbers, and nothing downstream can tell.
    """
    import pyarrow as pa

    from ..ir import Agg as AggExpr
    from ..ir import Col

    if not _group_by_kernel_works():
        return None

    for name, agg in aggs.items():
        if not isinstance(agg, AggExpr) or agg.distinct:
            return None
        if not isinstance(agg.arg, (Col, type(None))):
            return None

    try:
        grouped = table.arrow.group_by(keys)
        spec = [(name, (agg.func.lower(),) if agg.arg is None
                 else (agg.func.lower(), agg.arg.name))
                for name, agg in aggs.items()]
        result = grouped.aggregate(spec)
    except Exception:  # noqa: BLE001
        return None

    # Arrow names the key column "key_0"/"key_0" in some versions and keeps
    # the real name in others; normalise to the caller's names.
    if list(result.schema.names[:len(keys)]) != keys:
        result = result.rename_columns(
            [*keys, *[n for n in result.schema.names[len(keys):]]])

    canonical = Schema(tuple(
        Field(f.name, _from_pa(f.type), nullable=f.nullable,
              classification=(_tag_for(table.schema, f.name, aggs)
                              if f.name in aggs
                              else _key_tag(table.schema, f.name)))
        for f in result.schema))
    return Table(result, canonical)


def _tag_for(schema: Any, name: str, aggs: dict[str, Any]) -> frozenset:
    """A derived column inherits its aggregate's argument's tags."""
    return _aggregate_tags(schema, aggs).get(name, frozenset())


def _key_tag(schema: Any, name: str) -> frozenset:
    """A group key is a value from the input, so it keeps its own tags."""
    if schema.has(name):
        return schema.get(name).classification
    return frozenset()


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



def _aggregate_tags(schema: Any,
                    aggs: dict[str, Any]) -> dict[str, frozenset[str]]:
    """What each aggregate output column inherits from its argument.

    Delegates to the lineage layer so every engine applies the same rule; a
    second implementation here would eventually be a second answer.
    """
    return _lineage.aggregate_tags(schema, aggs)


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
    value_tags: dict[str, frozenset[str]] | None = None,
) -> Table:
    """Build a result table, keeping the input's key column types.

    Key types come from the input rather than being inferred from Python
    values, so an Int64 key does not silently become Float64 because one
    group happened to contain a null.

    ``value_tags`` carries the classification each *derived* column inherits.
    Without it a ``SUM(salary)`` would arrive unlabelled and a policy
    trusting classification would have nothing to act on - which is exactly
    how a derived column leaks past a working privacy layer.
    """
    import pyarrow as pa

    from ..interchange import canonical_to_arrow
    from ..types import Field, Schema

    tags = value_tags or {}
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

    def classification_for(name: str) -> frozenset[str]:
        """A group key is the value itself; a derived column is a function of
        its inputs. Both keep their sensitivity, for different reasons."""
        if name in key_names and table.schema.has(name):
            return table.schema.get(name).classification
        return tags.get(name, frozenset())

    return Table(pa.Table.from_pylist(rows, schema=arrow_schema),
                 Schema(tuple(
                     Field(f.name, _from_pa(f.type), nullable=f.nullable,
                           classification=classification_for(f.name))
                     for f in arrow_schema)))



def _build_joined(out: list[dict], left: Table, right: Table,
                  keys: list[str], how: Any) -> Table:
    """Assemble a join result with a stable column order and inherited types.

    The key columns keep the *left* input's type (a join key should be one
    type, not two), and every other column takes the type of the side it came
    from. Classification is the union of both sides: either input can
    contribute to a joined row, so a right-hand ``CONFIDENTIAL`` column is as
    sensitive as if it had arrived alone.
    """
    import pyarrow as pa

    from ..interchange import canonical_to_arrow
    from ..lineage import taint as _lineage
    from ..types import Field, Schema

    both = _lineage.merge_schemas(left.schema, right.schema)
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
                 Schema(tuple(
                     Field(f.name, _from_pa(f.type), nullable=f.nullable,
                           classification=both)
                     for f in arrow_schema)))


