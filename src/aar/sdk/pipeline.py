"""Analyst-facing pipeline construction.

A pipeline file is ordinary Python. It defines a function (or a module-level
``PIPELINE``) that returns an IR root node, which the planner then schedules::

    # pipelines/orders.py
    from aar.sdk import parquet, col, gt, group_by, write_parquet

    def build():
        orders = parquet("orders.parquet", estimated_bytes=2_000_000_000)
        big = orders.filter(gt(col("amount"), 100)).project("id", "region")
        return write_parquet(group_by(big, "region", sum_=("amount",)), "out.parquet")

Nothing here is a mock. Every builder returns a real :class:`~aar.ir.Node`
that the planner costs for real and the executor will run. What is *not* here
is the executor itself - see the status note in ``README.md``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Callable, Mapping, Sequence

from ..ir import Agg, BinOp, Col, Expr, JoinType, Lit, Node, NodeType, ScanSpec
from ..types import Schema

__all__ = [
    "Context", "load_pipeline", "build_pipeline",
    "col", "lit", "eq", "ne", "gt", "ge", "lt", "le", "and_", "or_", "not_",
    "sum_", "count", "mean", "min_", "max_",
    "parquet", "csv", "excel", "sql", "mongo",
    "filter_", "project", "join", "group_by", "aggregate", "sort", "limit",
    "write_parquet", "write_csv", "write_excel", "udf", "classify",
]


# ------------------------------------------------------------- expressions
def col(name: str) -> Col:
    """Reference a column."""
    return Col(name)


def lit(value: Any) -> Lit:
    """Reference a constant. Rendered escaped, never interpolated."""
    return Lit(value)


def eq(a: Expr, b: Expr) -> BinOp:
    return BinOp(a, "=", b)


def ne(a: Expr, b: Expr) -> BinOp:
    return BinOp(a, "<>", b)


def gt(a: Expr, b: Expr) -> BinOp:
    return BinOp(a, ">", b)


def ge(a: Expr, b: Expr) -> BinOp:
    return BinOp(a, ">=", b)


def lt(a: Expr, b: Expr) -> BinOp:
    return BinOp(a, "<", b)


def le(a: Expr, b: Expr) -> BinOp:
    return BinOp(a, "<=", b)


def and_(*terms: Expr) -> BinOp:
    out = terms[0]
    for t in terms[1:]:
        out = BinOp(out, "AND", t)
    return out


def or_(*terms: Expr) -> BinOp:
    out = terms[0]
    for t in terms[1:]:
        out = BinOp(out, "OR", t)
    return out


def not_(term: Expr) -> BinOp:
    return BinOp(term, "IS NOT", Lit(True))


def sum_(name: str) -> Agg:
    return Agg("SUM", Col(name))


def count(name: str | None = None) -> Agg:
    return Agg("COUNT", Col(name) if name else None)


def mean(name: str) -> Agg:
    return Agg("AVG", Col(name))


def min_(name: str) -> Agg:
    return Agg("MIN", Col(name))


def max_(name: str) -> Agg:
    return Agg("MAX", Col(name))


# ------------------------------------------------------------------ sources
def classify(input_node: Node, *columns: str, tags: Sequence[str] = (),
             descriptions: Mapping[str, str] | None = None) -> Node:
    """Mark columns as sensitive, so propagation has something to carry.

    Formats do not carry AAR's classification - Parquet has no place for it,
    and an Excel header is just a string. This node is where an analyst says
    what a column *is*, and everything downstream inherits it: an aggregate
    over a classified column is classified, a join takes the union of both
    sides, and a UDF inherits whatever it could see.

    ```python
    orders = classify(excel("FY26-orders.xlsx"),
                      "customer", "ssn", tags=["PII"],
                      descriptions={"ssn": "National insurance number"})
    ```

    Tag names are free text; the sensitivity ladder recognises ``PUBLIC``,
    ``INTERNAL``, ``PII``, ``PHI``, ``FINANCIAL``, ``CONFIDENTIAL`` and
    ``RESTRICTED``. An unrecognised tag is treated as ``INTERNAL`` rather than
    ignored, so a new label is conservative by default.
    """
    if not columns:
        raise ValueError("classify() needs at least one column name")
    normalised = [t.strip().upper() for t in tags if t and t.strip()]
    if not normalised:
        raise ValueError(
            f"classify({', '.join(columns)}) was given no tags; a column "
            f"marked without a tag protects nothing")
    n = Node(NodeType.TAG, inputs=[input_node],
             columns=tuple(columns), tag_values=tuple(normalised))
    n.descriptions = dict(descriptions or {})
    n.estimated_bytes = input_node.estimated_bytes
    return _inherit(input_node, n)


def _source(node_type: NodeType, spec: ScanSpec, schema: Schema | None,
            estimated_bytes: int) -> Node:
    node = Node(node_type, scan=spec)
    if schema is not None:
        node.output_schema = schema
    node.estimated_bytes = estimated_bytes
    return node


def parquet(path: str, estimated_bytes: int = 64_000_000,
            columns: Sequence[str] = ()) -> Node:
    """Read a Parquet file.

    ``estimated_bytes`` is a *declaration* the planner costs against. The
    runtime replaces it with the real file size, and the history layer
    replaces it with a measurement, so an early estimate costs accuracy but
    never correctness.
    """
    spec = ScanSpec(kind="parquet", path=path, columns=tuple(columns))
    return _source(NodeType.SCAN_PARQUET, spec, None, estimated_bytes)


def csv(path: str, estimated_bytes: int = 64_000_000,
        delimiter: str = ",") -> Node:
    spec = ScanSpec(kind="csv", path=path, delimiter=delimiter)
    return _source(NodeType.SCAN_CSV, spec, None, estimated_bytes)


def excel(path: str, sheet: str | None = None,
          cell_range: str | None = None, named_range: str | None = None,
          table: str | None = None, header_row: int = 1,
          formula_handling: str = "values",
          estimated_bytes: int = 16_000_000) -> Node:
    """Read an Excel workbook, sheet, table, named range or cell range."""
    spec = ScanSpec(kind="excel", path=path, sheet=sheet,
                    cell_range=cell_range, named_range=named_range,
                    table=table, header_row=header_row,
                    formula_handling=formula_handling)
    return _source(NodeType.SCAN_EXCEL, spec, None, estimated_bytes)


def sql(connection: str, table: str | None = None,
        query: str | None = None, estimated_bytes: int = 64_000_000) -> Node:
    """Read from a SQL database, by table or by query."""
    if not table and not query:
        raise ValueError("sql() needs either a table or a query")
    spec = ScanSpec(kind="sql", connection=connection, table_name=table,
                    query=query)
    return _source(NodeType.SCAN_SQL, spec, None, estimated_bytes)


def mongo(connection: str, collection: str,
          estimated_bytes: int = 64_000_000) -> Node:
    spec = ScanSpec(kind="mongo", connection=connection, collection=collection)
    return _source(NodeType.SCAN_MONGO, spec, None, estimated_bytes)



# -------------------------------------------------------------- transforms
def _inherit(input_node: Node, node: Node) -> Node:
    """Propagate schema and size downstream unless the caller set them."""
    if not node.input_schemas:
        node.input_schemas = (input_node.output_schema,)
    if node.estimated_bytes is None:
        node.estimated_bytes = input_node.estimated_bytes
    return node


def filter_(input_node: Node, predicate: Expr,
            estimated_bytes: int | None = None) -> Node:
    """Keep rows matching ``predicate``.

    ``pushdown_capable`` marks the node as something a source can absorb, which
    is what lets the planner push it into PostgreSQL rather than pulling the
    table across the wire and filtering locally.
    """
    n = Node(NodeType.FILTER, inputs=[input_node], predicate=predicate)
    n.pushdown_capable = True
    n.estimated_bytes = estimated_bytes
    return _inherit(input_node, n)


def project(input_node: Node, *columns: str,
            estimated_bytes: int | None = None) -> Node:
    """Keep and rename columns. A projection pushdown candidate."""
    names: list[str] = []
    for c in columns:
        if isinstance(c, str):
            names.append(c)
        else:
            raise TypeError(f"project() takes column names, got {type(c).__name__}")
    n = Node(NodeType.PROJECT, inputs=[input_node], columns=tuple(names))
    n.pushdown_capable = True
    n.estimated_bytes = estimated_bytes
    _inherit(input_node, n)
    # Preserve tags and lineage on the surviving columns.
    if input_node.output_schema:
        kept = input_node.output_schema.select(
            [c for c in names if input_node.output_schema.has(c)])
        if len(kept) == len(names):
            n.output_schema = kept
    return n


def join(left: Node, right: Node, on: Sequence[str],
         how: str = "inner", estimated_bytes: int | None = None) -> Node:
    """Join two inputs on shared key columns."""
    keys = tuple(on)
    if not keys:
        raise ValueError("join() needs at least one key column")
    n = Node(NodeType.JOIN, inputs=[left, right],
             key_left=keys, key_right=keys, join_type=JoinType(how))
    n.pushdown_capable = False   # a cross-source join must happen locally
    n.estimated_bytes = estimated_bytes
    n.input_schemas = (left.output_schema, right.output_schema)
    if n.estimated_bytes is None:
        a = left.estimated_bytes or 0
        b = right.estimated_bytes or 0
        n.estimated_bytes = int((a + b) * 1.3)
    return n


def group_by(input_node: Node, *keys: str,
             aggs: dict[str, Agg] | None = None,
             estimated_bytes: int | None = None) -> Node:
    """Group by ``keys`` and optionally aggregate.

    A group with aggregations produces fewer rows than its input; a plain
    group-by is modelled as a distinct operation, so a caller that wants a
    reduction should pass ``aggs``.
    """
    n = Node(NodeType.GROUPBY, inputs=[input_node], key_left=tuple(keys))
    if aggs:
        n.agg_functions = {k: (v,) for k, v in aggs.items()}
        if estimated_bytes is None:
            n.estimated_bytes = max(1024, int((input_node.estimated_bytes or 0) * 0.1))
    else:
        n.estimated_bytes = estimated_bytes
    return _inherit(input_node, n)


def aggregate(input_node: Node, aggs: dict[str, Agg]) -> Node:
    """Aggregate the whole input, with no grouping."""
    n = Node(NodeType.AGGREGATE, inputs=[input_node],
             agg_functions={k: (v,) for k, v in aggs.items()})
    n.estimated_bytes = 1024
    return _inherit(input_node, n)


def sort(input_node: Node, *keys: str, descending: bool = False,
         estimated_bytes: int | None = None) -> Node:
    """Sort. Pass ``key='col desc'`` style names for direction."""
    parsed: list[tuple[str, bool]] = []
    for k in keys:
        k = k.strip()
        if k.lower().endswith(" desc"):
            parsed.append((k[: -len(" desc")].strip(), False))
        else:
            parsed.append((k.removesuffix(" asc").strip(), True))
    n = Node(NodeType.SORT, inputs=[input_node], sort_keys=tuple(parsed))
    n.estimated_bytes = estimated_bytes
    return _inherit(input_node, n)


def limit(input_node: Node, n: int) -> Node:
    n_node = Node(NodeType.LIMIT, inputs=[input_node], limit=int(n))
    n_node.estimated_bytes = min(input_node.estimated_bytes or 0, n)
    return _inherit(input_node, n_node)


def udf(input_node: Node, fn: Callable[..., Any] | None = None,
        name: str = "", mode: str = "row",
        estimated_bytes: int | None = None) -> Node:
    """Apply a Python function. Always CPU-resident; never a GPU candidate.

    ``mode`` says how ``fn`` is called and is never inferred:

    * ``"row"`` (default) - ``fn(row_dict)`` returns one value per record.
      The record holds every column, so the function's parameter name does
      not have to encode a convention.
    * ``"column"`` - ``fn(columns_dict)`` returns a list of one value per
      record, where ``columns_dict`` maps each column name to its full list
      of values.
    * ``"auto"`` - inspect the signature. Offered, but not the default,
      because a one-argument function is ambiguous between the first two.

    ```python
    def risk_band(row):                     # row mode: the whole record
        return "high" if row["total"] > 1000 else "low"

    def as_ints(columns):                  # column mode: every column
        return [int(v) for v in columns["id"]]
    ```

    ```python
    udf(orders, risk_band)                              # row
    udf(orders, as_ints, mode="column")                 # column
    ```
    """
    if mode not in ("row", "column", "auto"):
        raise ValueError(
            f"mode must be 'row', 'column' or 'auto'; got {mode!r}")
    n = Node(NodeType.PYTHON_UDF, inputs=[input_node], udf=fn,
             udf_name=name or getattr(fn, "__name__", "udf"),
             udf_mode=mode)
    n.deterministic = False     # unknown until measured
    n.estimated_bytes = estimated_bytes
    return _inherit(input_node, n)



# ------------------------------------------------------------------- writes
def _write(input_node: Node, node_type: NodeType, target: str,
           fmt: str, mode: str, estimated_bytes: int | None) -> Node:
    n = Node(node_type, inputs=[input_node], target=target, write_format=fmt,
             write_mode=mode)
    n.estimated_bytes = estimated_bytes
    return _inherit(input_node, n)


def write_parquet(input_node: Node, path: str, mode: str = "overwrite") -> Node:
    return _write(input_node, NodeType.WRITE, path, "parquet", mode, 0)


def write_csv(input_node: Node, path: str, mode: str = "overwrite") -> Node:
    return _write(input_node, NodeType.WRITE, path, "csv", mode, 0)


def write_excel(input_node: Node, path: str, sheet: str | None = None,
                mode: str = "overwrite") -> Node:
    n = _write(input_node, NodeType.WRITE, path, "excel", mode, 0)
    n.scan = ScanSpec(kind="excel", path=path, sheet=sheet)
    return n



# ----------------------------------------------------------------- loading
#: A pipeline module may expose any of these, in order of preference.
_ENTRY_NAMES = ("build", "pipeline", "PIPELINE", "build_pipeline")


def load_pipeline(path: str, entry: str | None = None) -> Node:
    """Import a pipeline file and return the IR root it builds.

    The file is ordinary Python and is executed, so it is trusted input: run
    a pipeline only if you are willing to run the code in it. AAR's privacy
    guarantees cover the network and the data path, not arbitrary Python the
    analyst chose to point at.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    mod_name = "aar_pipeline_" + os.path.splitext(
        os.path.basename(path))[0].replace("-", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import a pipeline from {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module can import itself, and always remove
    # it afterwards so a reload cannot inherit stale state.
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise

    if entry is not None:
        fn = getattr(module, entry, None)
        if fn is None:
            raise AttributeError(
                f"{path} has no attribute {entry!r}; "
                f"looked for a callable among {list(_ENTRY_NAMES)}")
        if not callable(fn):
            raise TypeError(f"{path}:{entry} is not callable")
    else:
        fn = None
        for name in _ENTRY_NAMES:
            candidate = getattr(module, name, None)
            if callable(candidate):
                fn = candidate
                break
        if fn is None:
            raise AttributeError(
                f"{path} defines no pipeline entry point. Expected a callable "
                f"named one of {', '.join(_ENTRY_NAMES)}.")

    result = fn()
    if not isinstance(result, Node):
        raise TypeError(
            f"{path}: the pipeline entry point returned "
            f"{type(result).__name__}, expected an aar.ir.Node")
    return result


def build_pipeline(path: str, entry: str | None = None) -> Node:
    """Alias of :func:`load_pipeline`, for readability at call sites."""
    return load_pipeline(path, entry)


class Context:
    """A small namespace for building pipelines with an environment.

    Equivalent to calling the module-level builders directly; provided because
    the specification names ``ctx.excel()`` and a pipeline that binds
    ``from aar.sdk import excel`` reads the same without the indirection.
    """

    excel = staticmethod(excel)
    parquet = staticmethod(parquet)
    csv = staticmethod(csv)
    sql = staticmethod(sql)
    mongo = staticmethod(mongo)
    col = staticmethod(col)
    filter_ = staticmethod(filter_)
    project = staticmethod(project)
    join = staticmethod(join)
    group_by = staticmethod(group_by)
    aggregate = staticmethod(aggregate)
    sort = staticmethod(sort)
    limit = staticmethod(limit)
    udf = staticmethod(udf)
    write_parquet = staticmethod(write_parquet)
    write_csv = staticmethod(write_csv)
    write_excel = staticmethod(write_excel)

