from __future__ import annotations

import os
from typing import Any, Sequence

from ..capability import Device
from ..failures import SourceUnavailable
from ..interchange import Table
from ..ir import Agg, Col, Expr, Node
from .base import Engine

__all__ = ["PolarsGPUEngine"]


class PolarsGPUEngine(Engine):
    """Polars with RAPIDS as its GPU engine.

    A different strategy from :class:`~aar.engines.cudf_engine.CudfEngine`,
    worth being precise about the difference. cuDF owns the device and
    executes natively there. This engine keeps Polars' own lazy
    optimisations - projection and predicate pushdown into Parquet, its
    streaming executor - and only materialises on the device when a
    collect happens, via ``collect(engine="cudf")``.

    The trade: more pushdown than cuDF on a scan, and a crossing point at
    collect time that must be counted. The cost model prices that crossing
    explicitly, which is why these are two engines rather than one engine
    with a flag.
    """

    id = "polars_gpu"
    device = Device.GPU

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self._pl = _import_polars()
        self._cudf = _import_cudf()
        #: Which collect call this Polars build actually accepts, recorded
        #: rather than assumed. See `_find_collect`.
        self.collect_api: str | None = None
        #: Collects that crossed to the device. Asserted in the tests.
        self.device_collects = 0
        if self._pl is None:
            self._decline("execute", "polars is not installed")
        elif self._cudf is None:
            self._decline("execute",
                          "polars GPU needs cudf; install cudf-cu12")
        else:
            self.collect_api = _find_collect(self._pl)
            if self.collect_api is None:
                # Constructed, but there is no way to reach the device on this
                # Polars build. Declining with the *real* reason is far better
                # than constructing and failing on the first collect, which is
                # what the first T4 run hit: `ValueError: Invalid engine
                # argument engine='cudf'`.
                self._decline(
                    "execute",
                    f"polars {getattr(self._pl, '__version__', '?')} accepts "
                    f"no known GPU collect API; tried "
                    f"collect(engine='cudf'), collect(engine='gpu') and "
                    f"collect_cudf()")

    def _lazy(self, table: Table) -> Any:
        return self._pl.from_arrow(table.arrow).lazy()

    def _collect(self, frame: Any) -> Any:
        """Materialise on the device, then bring the result back.

        The single crossing point: everything before it is host-side lazy
        planning, everything after is a device-resident result.
        """
        api = self.collect_api or _find_collect(self._pl)
        if api is None:
            raise NotImplementedError(
                f"{self.id}: this Polars build accepts no known GPU collect "
                f"API, so the data cannot reach the device")
        self.device_collects += 1
        return _apply_collect(api, frame)

    def _to_table(self, collected: Any, source: Table | None = None) -> Table:
        out = Table(collected.to_arrow())
        return out.with_schema(source.schema) if source is not None else out

    def read_scan(self, node: Node) -> Table:
        if self._cudf is None:
            raise NotImplementedError(
                "polars_gpu needs cudf; install cudf-cu12")
        spec = node.scan
        if spec is None:
            raise ValueError("scan node has no ScanSpec")
        if spec.kind != "parquet":
            # A CSV/JSON scan on a GPU is a host parse plus a copy. Polars
            # parses those on the CPU far faster, so declining is faster
            # than complying.
            self._decline(f"scan_{spec.kind}",
                          "CSV/JSON parsing is faster on the CPU than copying "
                          "to the device first")
            raise NotImplementedError(
                f"polars_gpu scans Parquet only; {spec.kind!r} is faster on "
                f"the CPU. Let the executor fall back.")
        if not spec.path or not os.path.exists(spec.path):
            raise SourceUnavailable(
                f"no such Parquet file: {spec.path}", path=str(spec.path))
        lazy = self._pl.scan_parquet(spec.path)
        if spec.columns:
            lazy = lazy.select(list(spec.columns))
        return self._to_table(self._collect(lazy))

    def filter(self, table: Table, predicate: Expr) -> Table:
        lazy = self._expr(self._lazy(table), predicate)
        return self._to_table(self._collect(lazy), table)

    def project(self, table: Table, columns: Sequence[str]) -> Table:
        lazy = self._lazy(table).select(list(columns))
        return self._to_table(self._collect(lazy), table)

    def group_by(self, table: Table, keys: Sequence[str],
                 aggs: dict[str, Agg]) -> Table:
        lazy = self._lazy(table)
        exprs = [_pl_agg(a) for a in aggs.values()]
        if keys:
            lazy = lazy.group_by(list(keys)).agg(exprs)
        else:
            # A whole-table aggregate has no key to group on. Selecting the
            # aggregates off the bare frame is the count-one-row answer,
            # rather than an empty frame.
            lazy = lazy.select(exprs)
        return self._to_table(self._collect(lazy), table)

    def sort(self, table: Table, keys: Sequence[tuple[str, bool]]) -> Table:
        by = [k for k, _ in keys]
        descending = [desc for _, desc in keys]
        lazy = self._lazy(table).sort(by, descending=descending)
        return self._to_table(self._collect(lazy), table)

    def limit(self, table: Table, n: int) -> Table:
        return self._to_table(self._collect(self._lazy(table).limit(n)), table)

    def join(self, left: Table, right: Table, keys: Sequence[str],
             how: str = "inner") -> Table:
        lazy = self._lazy(left).join(self._lazy(right), on=list(keys), how=how)
        return self._to_table(self._collect(lazy), left)

    def udf(self, table: Table, fn: Any, mode: str = "row") -> Table:
        raise NotImplementedError(
            f"{self.id} cannot run a Python UDF: a Python callable executes "
            f"on the host, so honouring it would round-trip every row across "
            f"PCIe. Let the executor fall back.")

    def write(self, table: Table, node: Node) -> int:
        out = self._to_table(self._collect(self._lazy(table)))
        from .arrow_engine import ArrowEngine


# --------------------------------------------------------------------- utils
def _apply_collect(api: str, frame: Any) -> Any:
    """Cross to the device using whichever spelling this build accepts."""
    if api == "collect_cudf_method":
        return frame.collect_cudf()
    if api == "collect_cudf_function":
        return _import_polars().collect_cudf(frame)
    if api == "engine_gpu":
        return frame.collect(engine="gpu")
    return frame.collect(engine="cudf")


def _find_collect(pl: Any) -> str | None:
    """Which GPU collect spelling this Polars build supports.

    Polars has moved this API more than once: `collect(engine="cudf")` was
    valid for some releases and is rejected outright by others - the first
    T4 run hit `ValueError: Invalid engine argument engine='cudf'` on
    Polars 1.35. Guessing one spelling is how that happened in the first
    place, so the spellings are probed and the winner recorded in
    ``PolarsGPUEngine.collect_api`` for the run report to show.

    A feature *presence* check is used rather than running a collect,
    because probing by doing real work would either be slow or would need a
    frame. Where presence is ambiguous, `engine_gpu` is preferred over the
    older names: it is the spelling the project is moving toward.
    """
    if hasattr(pl, "collect_cudf"):
        return "collect_cudf_function"
    lazy = getattr(pl, "LazyFrame", None)
    if lazy is not None and hasattr(lazy, "collect_cudf"):
        return "collect_cudf_method"
    if _accepts_engine(pl, "gpu"):
        return "engine_gpu"
    if _accepts_engine(pl, "cudf"):
        return "engine_cudf"
    return None


def _accepts_engine(pl: Any, name: str) -> bool:
    """Whether `collect(engine=name)` is a valid call on this build.

    Introspects the signature rather than calling it, so the answer is a
    fact about the build rather than the result of a trial run.
    """
    import inspect

    lazy = getattr(pl, "LazyFrame", None)
    if lazy is None or not hasattr(lazy, "collect"):
        return False
    try:
        signature = inspect.signature(lazy.collect)
    except (TypeError, ValueError):  # pragma: no cover - C extension
        return False
    engine = signature.parameters.get("engine")
    if engine is None:
        return False
    if engine.default is inspect.Parameter.empty:
        return False
    # A Literal type is the strongest signal available; otherwise assume a
    # plain str annotation and let the first real collect report a mismatch.
    annotation = str(engine.annotation)
    if "Literal" in annotation:
        return name in annotation
    return True


def _import_polars() -> Any:
    try:
        import polars
    except ImportError:
        return None
    return polars


def _import_cudf() -> Any:
    try:
        import cudf
    except ImportError:
        return None
    return cudf


def _out(column: str | None, agg: Agg) -> str:
    """The output column name for an aggregate.

    Polars has no third positional for a name, so `custom` is where AAR
    carries it; falling back to the source column keeps `SUM(amount)`
    arriving as `amount`, which is what every other engine does too.
    """
    return getattr(agg, "custom", None) or column or "value"


def _pl_agg(agg: Agg) -> Any:
    """Map an AAR aggregate onto a Polars expression.

    ``AVG`` maps to ``mean``: both are sum over non-null values divided by
    the count of non-null values. Silently using ``median`` here would be a
    wrong answer rather than an error.
    """
    import polars as pl

    from ..ir import Agg, Col

    # Agg is (func, arg, distinct, custom). `arg` is the expression, so a
    # column name comes off a Col - and `agg.column` does not exist, which
    # the first T4 run found.
    arg = agg.arg
    col = arg.name if isinstance(arg, Col) else None
    op = (agg.func or "").upper()
    if col is None and op != "COUNT":
        raise NotImplementedError(
            f"the polars GPU engine needs a column for {op!r}")
    if op == "SUM":
        return pl.col(col).sum().alias(_out(col, agg))
    if op == "MIN":
        return pl.col(col).min().alias(_out(col, agg))
    if op == "MAX":
        return pl.col(col).max().alias(_out(col, agg))
    if op == "AVG":
        return pl.col(col).mean().alias(_out(col, agg))
    if op == "COUNT":
        # COUNT(*) counts rows; COUNT(x) skips nulls. `len()` and `count()`
        # are different questions, and conflating them is invisible until a
        # report is quietly wrong.
        return (pl.len().alias(_out(col, agg)) if col is None
                else pl.col(col).count().alias(_out(col, agg)))
    if op in ("FIRST", "ARBITRARY"):
        return pl.col(col).first().alias(_out(col, agg))
    raise NotImplementedError(
        f"the polars GPU engine does not implement {op!r}")


def _out(column: str | None, agg: Agg) -> str:
    return getattr(agg, "custom", None) or column or "value"


def _to_pl_expr(expr: Expr) -> Any:
    """Render a predicate as a Polars expression."""
    import polars as pl

    from ..ir import BinOp, Call, Lit

    if isinstance(expr, Col):
        return pl.col(expr.name)
    if isinstance(expr, Lit):
        return pl.lit(expr.value)
    if isinstance(expr, Call):
        name = expr.name.lower()
        args = [_to_pl_expr(a) for a in expr.args]
        if name in ("isnull", "is_null"):
            return args[0].is_null()
        if name in ("isnotnull", "is_not_null"):
            return args[0].is_not_null()
        raise NotImplementedError(
            f"the polars GPU engine cannot render {expr.name!r}")
    if isinstance(expr, BinOp):
        left, right = _to_pl_expr(expr.left), _to_pl_expr(expr.right)
        mapping = {"AND": lambda: left & right,
                   "OR": lambda: left | right,
                   "=": lambda: left == right,
                   "<>": lambda: left != right,
                   "!=": lambda: left != right,
                   ">": lambda: left > right,
                   ">=": lambda: left >= right,
                   "<": lambda: left < right,
                   "<=": lambda: left <= right,
                   "+": lambda: left + right,
                   "-": lambda: left - right,
                   "*": lambda: left * right,
                   "/": lambda: left / right}
        if expr.op in mapping:
            return mapping[expr.op]()
    raise NotImplementedError(
        f"the polars GPU engine cannot render "
        f"{getattr(expr, 'op', expr)!r}")

