"""Polars engine.

Polars is the specification's tier-2 CPU engine: lazy, parallel, streaming.
This wrapper uses its eager DataFrame API rather than its lazy one because the
planner has already decided the execution order - re-deriving it in Polars
would give AAR two optimisers disagreeing about the same plan.

Data crosses into and out of Polars as Arrow. That is the whole point of
``polars.from_arrow`` and ``to_arrow``: the interchange layer's rule holds even
here.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from ..capability import Device
from ..failures import SourceUnavailable
from ..interchange import Table, reconcile
from ..ir import Expr, Node
from ..lineage import taint as _lineage
from .base import Engine

__all__ = ["PolarsEngine"]


class PolarsEngine(Engine):
    """Executes through Polars."""

    id = "polars_cpu"
    device = Device.CPU

    def __init__(self, threads: int | None = None, **options: Any) -> None:
        super().__init__(threads=threads, **options)
        self._pl = _import_polars()

    @staticmethod
    def _wrap(table: Table) -> Any:
        return table.arrow

    def _table(self, frame: Any, source: Table | None = None,
               derived: Any = None) -> Table:
        """Polars -> Arrow -> AAR, with the schema preserved.

        ``source``/``derived`` restore what Polars cannot carry. A Polars
        DataFrame has nowhere to put an AAR classification, so the round trip
        returns a table whose fields are all unclassified - which would make a
        CONFIDENTIAL column silently public after any Polars filter.
        """
        return reconcile(Table(frame.to_arrow()), source, derived)

    # ------------------------------------------------------------------ read
    def read_scan(self, node: Node) -> Table:
        """Read a source into Arrow.

        Every result goes through :meth:`_table`. Handing ``Table()`` a Polars
        DataFrame works right up until the moment it walks the frame's
        ``schema`` dict as if it were an Arrow schema - and it then fails
        deep inside type conversion instead of at the boundary.
        """
        spec = node.scan
        if spec is None:
            raise ValueError("scan node has no ScanSpec")
        if spec.kind == "parquet":
            self._require(spec.path, "Parquet")
            return self._table(self._pl.read_parquet(
                spec.path, columns=list(spec.columns) or None))
        if spec.kind == "csv":
            self._require(spec.path, "CSV")
            return self._table(self._pl.read_csv(
                spec.path, separator=spec.delimiter or ","))
        if spec.kind == "json":
            self._require(spec.path, "JSON")
            return self._table(self._pl.read_json(spec.path))
        from .arrow_engine import ArrowEngine
        return ArrowEngine().read_scan(node)


    @staticmethod
    def _require(path: str | None, kind: str) -> None:
        if not path or not os.path.exists(path):
            raise SourceUnavailable(f"no such {kind} file: {path}")

    # --------------------------------------------------------------- filter
    def filter(self, table: Table, predicate: Expr) -> Table:
        """Filter through Polars, or Arrow when the predicate is exotic."""
        expr = _to_polars_expr(self._pl, predicate)
        if expr is None:
            from .arrow_engine import ArrowEngine
            return ArrowEngine().filter(table, predicate)
        frame = self._pl.from_arrow(table.arrow).filter(expr)
        return self._table(frame, source=table)

    def project(self, table: Table, columns: Sequence[str]) -> Table:
        return table.select(list(columns))

    def group_by(self, table: Table, keys: Sequence[str],
                 aggs: dict[str, Any]) -> Table:
        from .arrow_engine import ArrowEngine

        if not aggs:
            return ArrowEngine().group_by(table, keys, {})
        spec = _to_polars_aggs(self._pl, aggs)
        if spec is None:
            return ArrowEngine().group_by(table, keys, aggs)
        frame = (self._pl.from_arrow(table.arrow)
                 .group_by(list(keys)).agg(**spec).sort(list(keys)))
        return self._table(
            frame, source=table,
            derived=_lineage.aggregate_tags(table.schema, aggs))


    def sort(self, table: Table, keys: Sequence[tuple[str, bool]]) -> Table:
        """Order rows by ``keys``.

        Uses ``by``/``descending`` rather than a list of ``(column, ascending)``
        pairs. Polars parses a pair-list as if it were column *values*, which
        is confusing to look at and fails at run time; the two-list form is
        explicit about which key is which direction.
        """
        if not keys or table.num_rows == 0:
            return table
        by = [self._pl.col(str(k)) for k, _ in keys]
        descending = [not bool(asc) for _, asc in keys]
        return self._table(self._pl.from_arrow(table.arrow)
                           .sort(by, descending=descending), source=table)



    def limit(self, table: Table, n: int) -> Table:
        if n < 0:
            raise ValueError("limit must be non-negative")
        if table.num_rows <= n:
            return table
        return self._table(self._pl.from_arrow(table.arrow).head(n),
                           source=table)

    def join(self, left: Table, right: Table, keys: Sequence[str],
             how: str) -> Table:
        from .arrow_engine import ArrowEngine

        how_map = {"inner": "inner", "left": "left", "right": "right",
                   "full": "full"}
        polars_how = how_map.get(str(how))
        if polars_how is None:
            return ArrowEngine().join(left, right, keys, how)
        lf = self._pl.from_arrow(left.arrow)
        rf = self._pl.from_arrow(right.arrow)
        joined = lf.join(rf, on=list(keys), how=polars_how)
        # A joined row can draw from either input, so every output column is
        # as sensitive as the more sensitive of the two sides. Left alone,
        # this boundary dropped both sides' tags on the floor.
        both = _lineage.merge_schemas(left.schema, right.schema)
        out_names = list(left.column_names) + [
            c for c in right.column_names if c not in keys]
        return self._table(joined, source=left,
                           derived=dict.fromkeys(out_names, both))

    def udf(self, table: Table, fn: Any, mode: str = "row") -> Table:
        from .arrow_engine import ArrowEngine
        return ArrowEngine().udf(table, fn, mode)

    def write(self, table: Table, node: Node) -> int:
        target = node.target
        fmt = (node.write_format or "").lower()
        if not target:
            raise ValueError("write node has no target")
        parent = os.path.dirname(os.path.abspath(target))
        if parent:
            os.makedirs(parent, exist_ok=True)
        frame = self._pl.from_arrow(table.arrow)
        if fmt == "parquet":
            frame.write_parquet(target)
        elif fmt == "csv":
            frame.write_csv(target)
        else:
            from .arrow_engine import ArrowEngine
            return ArrowEngine().write(table, node)
        return table.num_rows


def _import_polars() -> Any:
    try:
        import polars
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The Polars engine needs the polars package. Install it with:\n"
            "  pip install polars\n"
            f"(import failed: {exc})") from exc
    return polars


# --------------------------------------------------------- expression glue
def _to_polars_expr(pl: Any, expr: Any) -> Any:
    """Render an IR predicate as a Polars expression, or None if too complex."""
    from ..ir import BinOp, Col, Lit

    if expr is None:
        return None
    if isinstance(expr, Col):
        return pl.col(expr.name)
    if isinstance(expr, Lit):
        return pl.lit(expr.value)
    if not isinstance(expr, BinOp):
        return None
    op = expr.op
    if op == "AND":
        return pl.all_horizontal(
            [e for e in (_to_polars_expr(pl, expr.left),
                         _to_polars_expr(pl, expr.right)) if e is not None])
    if op == "OR":
        return pl.any_horizontal(
            [e for e in (_to_polars_expr(pl, expr.left),
                         _to_polars_expr(pl, expr.right)) if e is not None])
    left = _to_polars_expr(pl, expr.left)
    right = _to_polars_expr(pl, expr.right)
    if left is None or right is None:
        return None
    try:
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
    except Exception:  # noqa: BLE001
        return None
    return None


def _to_polars_aggs(pl: Any, aggs: dict[str, Any]) -> dict[str, Any] | None:
    """Render AAR aggregates as Polars aggregations, or None if unsupported."""
    from ..ir import Agg as AggExpr
    from ..ir import Col

    spec: dict[str, Any] = {}
    for name, agg in aggs.items():
        if not isinstance(agg, AggExpr) or agg.custom or not isinstance(
                agg.arg, Col):
            return None
        func = agg.func.upper()
        col = pl.col(agg.arg.name)
        # `DISTINCT` has to be honoured for every aggregate, not just COUNT.
        # A plain `col.sum()` silently ignores it, so `SUM(DISTINCT x)` would
        # quietly return the sum *with duplicates* - the right number for a
        # different question, which is the worst kind of wrong. Polars has no
        # `sum(distinct=...)`, so the distinct values are taken first.
        if agg.distinct:
            col = col.unique()
        if func == "SUM":
            spec[name] = col.sum()
        elif func == "AVG":
            spec[name] = col.mean()
        elif func == "MIN":
            spec[name] = col.min()
        elif func == "MAX":
            spec[name] = col.max()
        elif func == "COUNT":
            spec[name] = col.n_unique() if agg.distinct else col.count()
        else:
            return None
    return spec


