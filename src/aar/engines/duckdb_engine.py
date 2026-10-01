"""DuckDB engine.

DuckDB is the specification's tier-2 workhorse: an in-process analytical
database with real projection and filter pushdown onto Parquet. It is also
the most useful fallback on a machine with no GPU, which is why the factory
tries it before the slower engines.

The connection is reused across every node in a run. Opening a DuckDB
connection costs milliseconds, and paying that per node would dominate small
pipelines.

Anything DuckDB cannot express as SQL falls back to the Arrow engine rather
than approximating it. A filter that meant something slightly different
would be far worse than one that merely ran slower.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Sequence

from ..capability import Device
from ..failures import SourceUnavailable
from ..interchange import Table, reconcile, require_arrow
from ..ir import Col, Expr, Lit, Node
from ..lineage import taint as _lineage
from .base import Engine

__all__ = ["DuckDBEngine"]

#: Join kinds DuckDB genuinely implements, and the SQL keyword for each.
#:
#: An **allow-list**, not a deny-list. The earlier code was
#: ``{"inner": "INNER", ...}.get(str(how), "INNER")``, so any kind not named
#: became an inner join - which is how SEMI, ANTI and ASOF quietly returned
#: inner-join rows. A missing entry now means "raise", because an unknown join
#: kind is a question AAR cannot answer correctly on the user's behalf.
#:
#: ASOF is deliberately absent. DuckDB has no ASOF join of this shape, and
#: pretending otherwise is the exact bug being fixed.
_DUCKDB_JOIN_KINDS: dict[str, str] = {
    "inner": "INNER",
    "left": "LEFT",
    "right": "RIGHT",
    "full": "FULL",
    "semi": "SEMI",
    "anti": "ANTI",
    "cross": "CROSS",
}


class DuckDBEngine(Engine):
    """Executes through an embedded DuckDB connection."""

    id = "duckdb"
    device = Device.CPU

    def __init__(self, threads: int | None = None,
                 memory_limit: str | None = None, **options: Any) -> None:
        super().__init__(threads=threads, memory_limit=memory_limit, **options)
        self._duckdb = _import_duckdb()
        self._lock = threading.Lock()
        self._conn: Any = None
        self._registered: set[str] = set()
        self._counter = 0
        if memory_limit:
            os.environ.setdefault("AAR_DUCKDB_MEMORY_LIMIT", memory_limit)

    @property
    def conn(self) -> Any:
        """The in-process connection, opened on first use."""
        if self._conn is None:
            threads = self._options.get("threads")
            # DuckDB rejects `config=None`; the kwarg must be omitted
            # entirely when there is nothing to configure.
            if threads:
                self._conn = self._duckdb.connect(
                    database=":memory:", config={"threads": str(threads)})
            else:
                self._conn = self._duckdb.connect(database=":memory:")
        return self._conn


    def close(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.close()
        finally:
            self._conn = None
            self._registered.clear()

    # --------------------------------------------------------- registration
    def _register(self, table: Table) -> "_Relation":
        with self._lock:
            self._counter += 1
            name = f"aar_t{self._counter}"
            self.conn.register(name, table.arrow)
            self._registered.add(name)
        return _Relation(self.conn, name)

    def _to_table(self, result: Any, source: Table | None = None,
                  derived: Any = None) -> Table:
        """A DuckDB result -> an AAR Table, keeping AAR's metadata.

        DuckDB exposes results as its own relation type, not Arrow. Calling
        ``.arrow()`` on it is not a real method, so the result is materialised
        through ``fetch_arrow_table`` - which is also the zero-copy path
        rather than a row-by-row conversion.

        ``source`` and ``derived`` are not optional politeness. A DuckDB
        result is a bare Arrow table, and ``Table(arrow)`` builds a schema
        whose every field is unclassified - so without reconciliation a
        CONFIDENTIAL column silently becomes public the moment a filter runs
        on DuckDB, and a policy trusting classification has nothing to act
        on. The numbers stay right, which is what makes it dangerous.
        """
        pa = require_arrow()
        if isinstance(result, pa.Table):
            arrow = result
        else:
            fetch = getattr(result, "fetch_arrow_table", None)
            if fetch is None:
                raise TypeError(
                    f"DuckDB returned {type(result).__name__}, which has no "
                    f"fetch_arrow_table(); this build of duckdb is not "
                    f"supported")
            arrow = fetch()
        return reconcile(Table(_denormalise_decimals(arrow, pa)),
                         source, derived)


    # ------------------------------------------------------------------ read
    def read_scan(self, node: Node) -> Table:
        """Read a source.

        Parquet and CSV go through DuckDB's own readers, which is where the
        specification's pushdown numbers come from. Excel and the remote
        sources are delegated to the Arrow engine rather than reimplemented -
        duplicating them would be two implementations to keep correct.
        """
        spec = node.scan
        if spec is None:
            raise ValueError("scan node has no ScanSpec")
        if spec.kind == "parquet":
            if not spec.path or not os.path.exists(spec.path):
                raise SourceUnavailable(f"no such Parquet file: {spec.path}")
            cols = ", ".join(_quote_ident(c) for c in spec.columns) or "*"
            return self._to_table(self.conn.execute(
                f"SELECT {cols} FROM read_parquet(?)", [spec.path]))
        if spec.kind == "csv":
            if not spec.path or not os.path.exists(spec.path):
                raise SourceUnavailable(f"no such CSV file: {spec.path}")
            delim = (spec.delimiter or ",").replace("'", "''")
            cols = ", ".join(_quote_ident(c) for c in spec.columns) or "*"
            return self._to_table(self.conn.execute(
                f"SELECT {cols} FROM read_csv(?, delim='{delim}', "
                f"header=true)", [spec.path]))
        from .arrow_engine import ArrowEngine
        return ArrowEngine().read_scan(node)

    # --------------------------------------------------------------- filter
    def filter(self, table: Table, predicate: Expr) -> Table:
        """Push the predicate into DuckDB as a WHERE clause where possible.

        A predicate is compiled to SQL only when every term is a simple
        comparison. Anything more complex takes the Arrow path, because
        emitting SQL that does not mean the same thing as the predicate would
        silently change the filter.
        """
        sql_where = _to_sql_where(predicate)
        if sql_where is None:
            from .arrow_engine import ArrowEngine
            return ArrowEngine().filter(table, predicate)
        rel = self._register(table)
        try:
            return self._to_table(
                rel.sql(f'SELECT * FROM "{rel.name}" WHERE {sql_where}'),
                source=table)
        finally:
            rel.release()

    # -------------------------------------------------------------- project
    def project(self, table: Table, columns: Sequence[str]) -> Table:
        return table.select(list(columns))

    # -------------------------------------------------------------- group by
    def group_by(self, table: Table, keys: Sequence[str],
                 aggs: dict[str, Any]) -> Table:
        """Group in SQL.

        This is where DuckDB earns its place: the aggregation runs in its
        vectorised engine rather than in Python.
        """
        from .arrow_engine import ArrowEngine

        if not aggs:
            return ArrowEngine().group_by(table, keys, {})
        key_sql = ", ".join(_quote_ident(k) for k in keys)
        parts = [_quote_ident(k) for k in keys]
        for name, agg in aggs.items():
            rendered = _agg_sql(agg)
            if rendered is None:
                return ArrowEngine().group_by(table, keys, aggs)
            parts.append(f'{rendered} AS {_quote_ident(name)}')
        rel = self._register(table)
        try:
            result = rel.sql(
                f"SELECT {', '.join(parts)} FROM {_quote_ident(rel.name)} "
                f"GROUP BY {key_sql}")
            # Materialise *before* releasing. Unregistering the relation
            # invalidates the pending result, and fetching afterwards returns
            # an empty table rather than an error - a silent wrong answer.
            return self._to_table(
                result, source=table,
                derived=_lineage.aggregate_tags(table.schema, aggs))
        finally:
            rel.release()


    # ----------------------------------------------------------------- sort
    def sort(self, table: Table, keys: Sequence[tuple[str, bool]]) -> Table:
        if not keys or table.num_rows == 0:
            return table
        order = ", ".join(f'{_quote_ident(k)} {"ASC" if asc else "DESC"}'
                          for k, asc in keys)
        rel = self._register(table)
        try:
            return self._to_table(
                rel.sql(f'SELECT * FROM {_quote_ident(rel.name)} ORDER BY '
                        f'{order}'), source=table)
        finally:
            rel.release()

    # ---------------------------------------------------------------- limit
    def limit(self, table: Table, n: int) -> Table:
        if n < 0:
            raise ValueError("limit must be non-negative")
        if table.num_rows <= n:
            return table
        rel = self._register(table)
        try:
            return self._to_table(
                rel.sql(f'SELECT * FROM {_quote_ident(rel.name)} '
                        f'LIMIT {int(n)}'), source=table)
        finally:
            rel.release()

    # ----------------------------------------------------------------- join
    def join(self, left: Table, right: Table, keys: Sequence[str],
             how: str) -> Table:
        """Join, with the join kind honoured rather than approximated.

        **This used to end in ``.get(str(how), "INNER")``**, which meant any
        join kind DuckDB was not explicitly listed for silently became an
        inner join. Measured on left={k:1,2,3} and right={k:2,3,4}:

            semi  -> returned [2, 3]  (correct rows, wrong columns)
            anti  -> returned [2, 3]  (the *exact complement* of [1])
            asof  -> returned [2, 3]  (inner again)

        Anti is the dangerous one: it returned exactly the rows it was defined
        to exclude, with no error and no warning. A pipeline that filtered with
        an anti-join would report success and produce the opposite of the
        intended answer. That is worse than a crash, because a crash is at
        least visible.

        DuckDB supports SEMI, ANTI, LEFT, RIGHT, FULL and INNER natively, so
        the honest implementation maps what it genuinely supports and
        **raises** for ASOF rather than pretending. An unsupported kind is now
        a visible error at the boundary, not a silent rewrite.
        """
        rel_l = self._register(left)
        rel_r = self._register(right)
        # ``how`` arrives as a JoinType member, and JoinType is a ``str``
        # mixin enum: ``str(JoinType.INNER)`` is "JoinType.INNER" on Python
        # 3.11+, not "inner", so the lookup has to unwrap ``.value``.
        kind = getattr(how, "value", how)
        how_sql = _DUCKDB_JOIN_KINDS.get(str(kind))
        if how_sql is None:
            raise ValueError(
                f"duckdb cannot perform a {str(kind)!r} join; it was not "
                f"silently converted to an inner join. Supported: "
                f"{sorted(_DUCKDB_JOIN_KINDS)}")
        try:
            on = " AND ".join(f'l.{_quote_ident(k)} = r.{_quote_ident(k)}'
                              for k in keys)
            if how_sql == "CROSS":
                # A cross join has no ON clause at all; emitting `ON` is a
                # syntax error rather than a silent behaviour change.
                on = "" if not keys else on
            # SEMI and ANTI keep only the left side's columns by definition, so
            # the projection has to differ too - emitting the right columns
            # would fabricate data that a semi-join is defined not to produce.
            if how_sql in ("SEMI", "ANTI"):
                cols = ", ".join(f"l.{_quote_ident(c)}"
                                 for c in left.column_names)
            else:
                cols = ", ".join(
                    [f"l.{_quote_ident(c)}" for c in left.column_names]
                    + [f"r.{_quote_ident(c)}" for c in right.column_names
                       if c not in keys])
            # Either input can contribute to a joined row, so every output
            # column is as sensitive as the more sensitive of the two sides.
            both = _lineage.merge_schemas(left.schema, right.schema)
            if how_sql in ("SEMI", "ANTI"):
                # Only the left side's rows survive, so only its lineage can.
                out_names = list(left.column_names)
            else:
                out_names = list(left.column_names) + [
                    c for c in right.column_names if c not in keys]
            join_sql = f'{how_sql} JOIN {_quote_ident(rel_r.name)} r'
            if how_sql != "CROSS":
                join_sql += f" ON {on}"
            return self._to_table(
                rel_l.sql(
                    f'SELECT {cols} FROM {_quote_ident(rel_l.name)} l '
                    f'{join_sql}'),
                source=left, derived=dict.fromkeys(out_names, both))
        finally:
            rel_l.release()
            rel_r.release()

    # ------------------------------------------------------------------ udf
    def udf(self, table: Table, fn: Any, mode: str = "row") -> Table:
        """A Python UDF cannot run inside DuckDB, so it runs outside it."""
        from .arrow_engine import ArrowEngine
        return ArrowEngine().udf(table, fn, mode)

    # ---------------------------------------------------------------- write
    def write(self, table: Table, node: Node) -> int:
        """Write to the target in ``node``.

        Only Parquet and CSV go through DuckDB's ``COPY``. ``HEADER`` is a CSV
        option and passing it for Parquet is a syntax error, so the option
        list is built per format rather than shared.
        """
        target = node.target
        if not target:
            raise ValueError("write node has no target")
        fmt = (node.write_format or "").lower()
        if fmt not in ("parquet", "csv"):
            from .arrow_engine import ArrowEngine
            return ArrowEngine().write(table, node)
        parent = os.path.dirname(os.path.abspath(target))
        if parent:
            os.makedirs(parent, exist_ok=True)
        literal = "'" + target.replace("'", "''") + "'"
        options = ("FORMAT CSV, HEADER true" if fmt == "csv"
                   else "FORMAT PARQUET")
        rel = self._register(table)
        try:
            rel.sql(f"COPY {_quote_ident(rel.name)} TO {literal} ({options})")
        finally:
            rel.release()
        return table.num_rows



# ------------------------------------------------------------------ helpers
def _import_duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The DuckDB engine needs the duckdb package. Install it with:\n"
            "  pip install duckdb\n"
            f"(import failed: {exc})") from exc
    return duckdb


class _Relation:
    """A named DuckDB relation for one Arrow table.

    Arrow tables are registered under a unique name and dropped when the
    relation is released, so a long run does not accumulate relations in the
    connection and slow every later query down.
    """

    __slots__ = ("conn", "name", "_alive")

    def __init__(self, conn: Any, name: str) -> None:
        self.conn = conn
        self.name = name
        self._alive = True

    def sql(self, query: str, params: Sequence[Any] = ()) -> Any:
        return self.conn.execute(query, list(params) if params else None)

    def release(self) -> None:
        if self._alive:
            try:
                self.conn.unregister(self.name)
            except Exception:  # noqa: BLE001 - already gone is fine
                pass
            self._alive = False

    def __enter__(self) -> "_Relation":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


def _quote_ident(name: str) -> str:
    """Quote an identifier for DuckDB, escaping embedded quotes."""
    return '"' + str(name).replace('"', '""') + '"'


def _denormalise_decimals(arrow: Any, pa: Any) -> Any:
    """Collapse DuckDB's ``DECIMAL`` results to the canonical numeric type.

    DuckDB types ``SUM(INTEGER)`` as ``DECIMAL(38, 0)`` and returns
    ``Decimal('400')``, where Arrow returns the integer ``400``. Both are
    "the right number", so nothing is wrong in a narrow sense - but AAR
    exists to make engine differences invisible, and this one changes the
    Python type a caller receives and the canonical type of the output
    schema depending on which engine the planner chose. A pipeline that
    returns ``int`` on one run and ``Decimal`` on the next, purely because
    the cost model had a different opinion, is precisely the surprise the
    interchange layer is meant to remove.

    A decimal with a zero scale is an integer; anything else becomes a
    float. ``NaN``/infinity are left to Arrow, which already represents
    them in the float domain.
    """
    arrays = []
    changed = False
    for field, column in zip(arrow.schema, arrow.columns):
        if not pa.types.is_decimal(field.type):
            arrays.append(column)
            continue
        if field.type.scale == 0:
            arrays.append(pa.array(
                [None if v is None else int(v) for v in column.to_pylist()],
                type=pa.int64()))
        else:
            arrays.append(pa.array(
                [None if v is None else float(v) for v in column.to_pylist()],
                type=pa.float64()))
        changed = True
    if not changed:
        return arrow
    return pa.Table.from_arrays(
        arrays,
        schema=pa.schema([pa.field(f.name, a.type, nullable=f.nullable)
                          for f, a in zip(arrow.schema, arrays)]))



def _quote_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"




def _to_sql_where(expr: Any) -> str | None:
    """Render a predicate as a DuckDB WHERE clause, or None if too complex.

    Returning ``None`` is the important behaviour: the caller falls back to
    the row-wise path rather than emitting SQL that does not mean the same
    thing as the predicate. A filter that quietly changed would be far worse
    than one that ran slower.
    """
    from ..ir import BinOp

    if expr is None:
        return None
    if isinstance(expr, Col):
        return _quote_ident(expr.name)
    if isinstance(expr, Lit):
        return _quote_value(expr.value)
    if not isinstance(expr, BinOp):
        return None
    op = expr.op
    if op not in ("=", "<>", ">", ">=", "<", "<=", "AND", "OR", "IS",
                  "IS NOT"):
        return None
    left = _to_sql_where(expr.left)
    right = _to_sql_where(expr.right)
    if left is None or right is None:
        return None

    # A null test must stay a null test. SQL's three-valued logic makes any
    # comparison involving NULL evaluate to UNKNOWN, not TRUE, so rewriting
    # `x IS NOT NULL` as `x <> NULL` matches *zero rows* rather than every
    # non-null one - a silently wrong answer, on the fastest engine, with no
    # error anywhere. The same reasoning applies to `x IS NULL`.
    if op in ("IS", "IS NOT"):
        if not (isinstance(expr.right, Lit) and expr.right.value is None):
            return None
        return f"{left} IS{' NOT' if op == 'IS NOT' else ''} NULL"

    if op == "IS NOT":
        # `col IS NOT 'literal'` is not valid SQL; express it as <>.
        return f"({left} <> {right})"
    return f"({left} {op} {right})"



def _agg_sql(agg: Any) -> str | None:
    """Render an aggregate as SQL, or None if AAR's form is richer."""
    from ..ir import Agg as AggExpr

    if not isinstance(agg, AggExpr):
        return None
    if agg.custom:
        return None      # a user aggregate has no DuckDB equivalent
    func = agg.func.upper()
    sql_func = {"SUM": "SUM", "COUNT": "COUNT", "AVG": "AVG", "MEAN": "AVG",
                "MIN": "MIN", "MAX": "MAX"}.get(func)
    if sql_func is None:
        return None
    if agg.arg is None:
        return "COUNT(*)"
    if not isinstance(agg.arg, Col):
        return None
    inner = (f"DISTINCT {_quote_ident(agg.arg.name)}" if agg.distinct
             else _quote_ident(agg.arg.name))
    return f"{sql_func}({inner})"

