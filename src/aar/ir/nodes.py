"""Internal Analytics IR - node types and the operator metadata contract.

The IR is AAR's *stable* internal representation. It is deliberately **not**
Substrait: Substrait is still pre-1.0, and this IR additionally models things
Substrait has no vocabulary for - Excel ranges and named ranges, Python UDFs
with GPU-hostility metadata, privacy classifications, governance policy
constraints, materialisation and cache boundaries, and NoSQL aggregation
stages. Substrait import/export adapters live in :mod:`aar.ir.substrait` and
translate in both directions without leaking Substrait's shape into the core.

Every node carries the metadata the adaptive planner reasons over. The
metadata is *declared*, not measured: measured values (real cardinalities,
real elapsed times) arrive from the execution history and are reconciled
against these estimates.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from ..types import EMPTY_SCHEMA, DataType, Schema, UnmappableType

__all__ = [
    "NodeType", "Privacy", "JoinType", "Node", "ScanSpec", "Expr",
    "Col", "Lit", "BinOp", "UnaryOp", "Func", "Agg", "CastExpr",
    "WindowSpec", "is_source", "is_sink", "topological_order",
]


class NodeType(str, enum.Enum):
    """The closed set of IR node types."""

    SCAN_EXCEL = "ScanExcel"
    SCAN_SQL = "ScanSQL"
    SCAN_MONGO = "ScanMongo"
    SCAN_PARQUET = "ScanParquet"
    SCAN_CSV = "ScanCSV"
    SCAN_JSON = "ScanJSON"
    SCAN_ARROW = "ScanArrow"
    SCAN_CONST = "ScanConst"

    FILTER = "Filter"
    PROJECT = "Project"
    JOIN = "Join"
    GROUPBY = "GroupBy"
    AGGREGATE = "Aggregate"
    SORT = "Sort"
    WINDOW = "Window"
    DEDUPLICATE = "Deduplicate"
    CAST = "Cast"
    NULL_HANDLE = "NullHandle"
    LIMIT = "Limit"
    UNION = "Union"

    PYTHON_UDF = "PythonUDF"
    QUALITY_CHECK = "QualityCheck"
    TAG = "Tag"

    WRITE = "Write"
    CACHE = "Cache"
    MATERIALIZE = "Materialize"

    def __str__(self) -> str:  # pragma: no cover - display
        return self.value


class Privacy(str, enum.Enum):
    """Sensitivity of a column or pipeline stage.

    The planner treats ``RESTRICTED`` as ineligible for any engine or
    destination that is not on an explicit allow-list, which is what makes
    "the secure path is the easy path" enforceable rather than aspirational.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

    @property
    def rank(self) -> int:
        return {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}[self.value]

    def __lt__(self, other: "Privacy") -> bool:
        if not isinstance(other, Privacy):
            return NotImplemented
        return self.rank < other.rank

    @staticmethod
    def coerce(value: "Privacy | str | None") -> "Privacy":
        if isinstance(value, Privacy):
            return value
        if value is None:
            return Privacy.INTERNAL
        try:
            return Privacy(str(value).lower())
        except ValueError as exc:
            raise UnmappableType(
                f"unknown privacy level {value!r}; "
                f"expected one of {[p.value for p in Privacy]}") from exc


class JoinType(str, enum.Enum):
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    FULL = "full"
    SEMI = "semi"
    ANTI = "anti"
    CROSS = "cross"
    #: AAR-specific: probe the local side after a pushdown, stream the remote.
    ASOF = "asof"

    def __str__(self) -> str:  # pragma: no cover - display
        return self.value


#: Nodes that read from an external system.
_SOURCE_NODES = frozenset({
    NodeType.SCAN_EXCEL, NodeType.SCAN_SQL, NodeType.SCAN_MONGO,
    NodeType.SCAN_PARQUET, NodeType.SCAN_CSV, NodeType.SCAN_JSON,
    NodeType.SCAN_ARROW, NodeType.SCAN_CONST,
})

#: Nodes that write to an external system.
_SINK_NODES = frozenset({NodeType.WRITE})


def is_source(node: "Node") -> bool:
    return node.type in _SOURCE_NODES


def is_sink(node: "Node") -> bool:
    return node.type in _SINK_NODES


# ---------------------------------------------------------------- expressions
class Expr:
    """Base class for IR expressions.

    Expressions are a tiny, closed algebra rather than a string DSL. That
    lets the planner analyse them, lets pushdown render them into SQL or a
    MongoDB aggregation stage, and keeps operator-precedence bugs out of the
    problem entirely.
    """

    __slots__ = ()

    def to_sql(self, dialect: str = "duckdb") -> str:  # pragma: no cover - overridden
        raise NotImplementedError(type(self).__name__)

    def columns(self) -> frozenset[str]:
        """Every column name this expression reads."""
        out: set[str] = set()
        for child in _expr_children(self):
            out |= child.columns()
        return frozenset(out)


def _expr_children(e: "Expr") -> tuple["Expr", ...]:
    match e:
        case BinOp(left, _, right):
            return (left, right)
        case UnaryOp(_, operand):
            return (operand,)
        case Func(_, args, _):
            return tuple(args)
        case Agg(_, arg, _, _):
            return (arg,) if arg is not None else ()
        case CastExpr(arg, _):
            return (arg,)
        case _:
            return ()


def _quote(name: str, dialect: str) -> str:
    """Quote an identifier for the target dialect.

    Identifiers reaching this function have already been quoted and escaped
    at construction, so a column name can never break out of its literal.
    """
    if dialect in ("postgres", "duckdb", "sqlite"):
        return '"' + name.replace('"', '""') + '"'
    if dialect == "mysql":
        return "`" + name.replace("`", "``") + "`"
    return name


@dataclass(frozen=True, slots=True)
class Col(Expr):
    """A reference to a column of the current input."""

    name: str

    def columns(self) -> frozenset[str]:
        return frozenset({self.name})

    def to_sql(self, dialect: str = "duckdb") -> str:
        return _quote(self.name, dialect)

    # Comparisons and boolean combination, so a pipeline reads as
    # `col("a") > 100` rather than `gt(col("a"), lit(100))`. The operator
    # forms build exactly the same IR nodes, so the planner, pushdown and
    # executor need to know only one spelling.
    def __gt__(self, other: Any) -> "BinOp":
        return BinOp(self, ">", _as_expr(other))

    def __ge__(self, other: Any) -> "BinOp":
        return BinOp(self, ">=", _as_expr(other))

    def __lt__(self, other: Any) -> "BinOp":
        return BinOp(self, "<", _as_expr(other))

    def __le__(self, other: Any) -> "BinOp":
        return BinOp(self, "<=", _as_expr(other))

    def __eq__(self, other: Any) -> "BinOp":  # type: ignore[override]
        return BinOp(self, "=", _as_expr(other))

    def __ne__(self, other: Any) -> "BinOp":  # type: ignore[override]
        return BinOp(self, "<>", _as_expr(other))

    def __and__(self, other: Any) -> "BinOp":
        return BinOp(self, "AND", _as_expr(other))

    def __or__(self, other: Any) -> "BinOp":
        return BinOp(self, "OR", _as_expr(other))

    def __hash__(self) -> int:
        return hash(("Col", self.name))


def _as_expr(value: Any) -> Expr:
    """Coerce a raw Python value into an expression.

    Required because ``col("a") > 5`` has to mean exactly what
    ``gt(col("a"), lit(5))`` means. Without this coercion the operator
    spelling would either not exist or would bypass the escaping that
    :class:`Lit` performs, and a value containing a quote would reach SQL
    unescaped.
    """
    if isinstance(value, Expr):
        return value
    return Lit(value)



@dataclass(frozen=True, slots=True)
class Lit(Expr):
    """A constant. ``value`` is a Python object; ``dtype`` pins the type."""

    value: Any
    dtype: DataType | None = None

    def to_sql(self, dialect: str = "duckdb") -> str:
        v = self.value
        if v is None:
            return "NULL"
        if isinstance(v, bool):
            return "TRUE" if v else "FALSE"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            return "'" + v.replace("'", "''") + "'"
        if isinstance(v, (list, tuple)):
            inner = ", ".join(Lit(x, self.dtype).to_sql(dialect) for x in v)
            return f"[{inner}]" if dialect in ("duckdb", "postgres") else f"({inner})"
        return "'" + str(v) + "'"


@dataclass(frozen=True, slots=True)
class BinOp(Expr):
    """A binary operation; ``op`` is a SQL-ish infix operator.

    Raw Python values on either side are lifted to :class:`Lit` on
    construction. Every consumer - the predicate compiler, the SQL renderers,
    the planner's pushdown analysis - assumes its operands are expressions, and
    a bare ``100`` arriving from a caller would otherwise fail deep inside
    one of them, far from the line that created it.
    """

    left: Expr
    op: str
    right: Expr

    def __post_init__(self) -> None:
        if not isinstance(self.left, Expr):
            object.__setattr__(self, "left", _as_expr(self.left))
        if not isinstance(self.right, Expr):
            object.__setattr__(self, "right", _as_expr(self.right))

    def to_sql(self, dialect: str = "duckdb") -> str:
        return f"({self.left.to_sql(dialect)} {self.op} {self.right.to_sql(dialect)})"


@dataclass(frozen=True, slots=True)
class UnaryOp(Expr):
    """A prefix operation, e.g. ``NOT x`` or ``-x``."""

    op: str
    operand: Expr

    def __post_init__(self) -> None:
        if not isinstance(self.operand, Expr):
            object.__setattr__(self, "operand", _as_expr(self.operand))

    def to_sql(self, dialect: str = "duckdb") -> str:
        return f"({self.op} {self.operand.to_sql(dialect)})"



@dataclass(frozen=True, slots=True)
class Func(Expr):
    """A scalar function call."""

    name: str
    args: tuple[Expr, ...] = ()
    kwargs: tuple[tuple[str, Any], ...] = ()

    def to_sql(self, dialect: str = "duckdb") -> str:
        parts = [a.to_sql(dialect) for a in self.args]
        parts += [f"{k}={v!r}" for k, v in self.kwargs]
        return f"{self.name}({', '.join(parts)})"


@dataclass(frozen=True, slots=True)
class Agg(Expr):
    """An aggregate, e.g. ``SUM(x)`` or ``COUNT(DISTINCT y)``."""

    func: str
    arg: Expr | None = None
    distinct: bool = False
    custom: str | None = None

    def to_sql(self, dialect: str = "duckdb") -> str:
        inner = "DISTINCT " if self.distinct else ""
        arg = self.arg.to_sql(dialect) if self.arg is not None else "*"
        return f"{self.func}({inner}{arg})"


@dataclass(frozen=True, slots=True)
class CastExpr(Expr):
    """An explicit cast. ``to`` is always a canonical type."""

    arg: Expr
    to: DataType

    def to_sql(self, dialect: str = "duckdb") -> str:
        names = {
            "int8": "TINYINT", "int16": "SMALLINT", "int32": "INT",
            "int64": "BIGINT", "float32": "FLOAT", "float64": "DOUBLE",
            "utf8": "VARCHAR", "bool": "BOOLEAN", "date32": "DATE",
            "date64": "DATE", "binary": "BLOB", "null": "VARCHAR",
            "timestamp": "TIMESTAMP", "decimal": "DECIMAL",
        }
        target = names.get(self.to.kind.value, "VARCHAR")
        if self.to.kind.value == "timestamp" and self.to.unit == "ms":
            target = "TIMESTAMP_MS"
        return f"CAST({self.arg.to_sql(dialect)} AS {target})"


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """A window frame definition."""

    partition_by: tuple[str, ...] = ()
    order_by: tuple[tuple[str, bool], ...] = ()  # (column, ascending)
    frame: str = "rows between unbounded preceding and current row"
    kind: str = "rows"  # rows | range



# --------------------------------------------------------------- scan specs
@dataclass(slots=True)
class ScanSpec:
    """Everything a connector needs to open a source.

    One spec type covers every source system. Fields that do not apply to a
    given ``kind`` stay ``None``; the connector validates what it requires and
    raises rather than reading something subtly different from what was asked
    for.
    """

    kind: str = "parquet"  # excel | sql | mongo | parquet | csv | json | arrow | const

    # --- Excel
    path: str | None = None
    sheet: str | None = None
    table: str | None = None
    named_range: str | None = None
    cell_range: str | None = None
    header_row: int = 1
    formula_handling: str = "values"  # values | evaluate | formulas
    include_hidden: bool = False

    # --- SQL
    connection: str | None = None
    dsn: str | None = None
    table_name: str | None = None
    query: str | None = None

    # --- Mongo
    collection: str | None = None
    database: str | None = None
    pipeline: tuple[Mapping[str, Any], ...] = ()

    # --- files
    delimiter: str = ","
    encoding: str = "utf-8"
    columns: tuple[str, ...] = ()
    row_filter: "Expr | None" = None
    projection: tuple[str, ...] = ()

    # --- general
    streaming: bool = True
    batch_size: int = 8192
    options: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        if self.kind == "excel":
            loc = self.path or "<in-memory>"
            bits = [f"sheet={self.sheet}" if self.sheet else "",
                    f"table={self.table}" if self.table else "",
                    f"range={self.named_range or self.cell_range}"
                    if (self.named_range or self.cell_range) else "",
                    f"header_row={self.header_row}"]
            joined = ", ".join(b for b in bits if b)
            return f"Excel({loc}" + (", " + joined if joined else "") + ")"
        if self.kind == "sql":
            return f"SQL({self.connection}: {self.table_name or self.query})"
        if self.kind == "mongo":
            return f"Mongo({self.connection}: {self.database}.{self.collection})"
        if self.kind == "parquet":
            return f"Parquet({self.path})"
        if self.kind == "csv":
            return f"CSV({self.path})"
        if self.kind == "json":
            return f"JSON({self.path})"
        return f"{self.kind}({self.path or self.connection or ''})"

    def cache_key(self) -> tuple[Any, ...]:
        """Identity of the *data*, not of the request.

        Used for cache invalidation: a changed file mtime/size or a changed
        query text yields a different key, so results cannot go stale.
        """
        import os

        stamp: Any = None
        if self.path and os.path.exists(self.path):
            st = os.stat(self.path)
            stamp = (st.st_mtime_ns, st.st_size)
        return (
            self.kind, self.path, self.sheet, self.table, self.named_range,
            self.cell_range, self.header_row, self.formula_handling,
            self.include_hidden, self.connection, self.table_name, self.query,
            self.collection, self.database, tuple(self.columns),
            self.delimiter, self.encoding, stamp,
        )



# ------------------------------------------------------------------- nodes
@dataclass(slots=True)
class Node:
    """One operation in the analytics DAG.

    The same dataclass models a logical operation *and* a physical decision:
    ``assigned_engine`` is empty until the planner fills it, and ``reason``
    records why. Keeping one node type (rather than separate logical and
    physical trees) is deliberate - it makes "why did this run on the GPU" a
    field you can read rather than a diff you have to compute.
    """

    type: NodeType
    inputs: list["Node"] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"op_{uuid.uuid4().hex[:12]}")

    # --- logical payload
    scan: ScanSpec | None = None
    predicate: Expr | None = None
    columns: tuple[str, ...] = ()
    expressions: dict[str, Expr] = field(default_factory=dict)
    key_left: tuple[str, ...] = ()
    key_right: tuple[str, ...] = ()
    join_type: JoinType = JoinType.INNER
    agg_functions: dict[str, tuple[Agg, ...]] = field(default_factory=dict)
    sort_keys: tuple[tuple[str, bool], ...] = ()
    window: WindowSpec | None = None
    window_functions: dict[str, Expr] = field(default_factory=dict)
    dedup_keys: tuple[str, ...] = ()
    dedup_strategy: str = "first"
    #: Classification tags applied by a TAG node, and the free-text
    #: descriptions that go with them.
    tag_values: tuple[str, ...] = ()
    descriptions: dict[str, str] = field(default_factory=dict)
    casts: dict[str, DataType] = field(default_factory=dict)
    null_strategy: str = "fill"
    fill_value: Any = None
    udf: Any = None
    udf_name: str = ""
    udf_mode: str = "row"
    #: Declared identity for a UDF whose source cannot be hashed - a C
    #: extension, a callable object, a function defined in a REPL. Setting
    #: this is an assertion by the author that the string captures everything
    #: that affects behaviour and cost. When absent, identity is derived from
    #: the function's source and raises if that is impossible; it is never
    #: silently faked from a name or an address.
    semantic_version: str | None = None
    limit: int | None = None
    target: str | None = None
    write_format: str | None = None
    write_mode: str = "overwrite"
    quality_rules: tuple[tuple[str, str], ...] = ()

    # --- declared metadata (estimates; reconciled with history at plan time)
    output_schema: Schema = EMPTY_SCHEMA
    input_schemas: tuple[Schema, ...] = ()
    estimated_rows: int | None = None
    estimated_bytes: int | None = None
    estimated_selectivity: float | None = None
    #: A measured :class:`aar.stats.TableProfile` for this node's source, when
    #: one has been taken. This is *evidence* where ``estimated_bytes`` is a
    #: declaration: the planner prefers it, and it is what lets a plan say
    #: "measured from a Parquet footer" rather than "assumed". Typed as
    #: ``Any`` to keep the IR free of a dependency on the statistics layer.
    aar_profile: Any = None
    privacy: Privacy = Privacy.INTERNAL
    deterministic: bool = True
    streamable: bool = True
    parallelizable: bool = True
    spillable: bool = True
    pushdown_capable: bool = False
    #: Engines known to implement this node type at all. Intersected with the
    #: capability registry, which then intersects with what is installed.
    supported_engines: frozenset[str] = frozenset()
    #: Non-engine constraints, e.g. {"requires_gpu": True}.
    requires: frozenset[str] = frozenset()
    hint: str | None = None

    # --- physical decision, filled by the planner
    assigned_engine: str | None = None
    reason: str = ""
    estimated_ms: float = 0.0
    estimated_peak_memory: int = 0
    expected_saving_ms: float = 0.0
    fallback_engine: str | None = None
    fallback_reason: str = ""
    segment_id: int = 0
    pushed: tuple[str, ...] = ()

    @property
    def is_source(self) -> bool:
        return self.type in _SOURCE_NODES

    @property
    def is_sink(self) -> bool:
        return self.type in _SINK_NODES

    @property
    def arity(self) -> int:
        return len(self.inputs)

    def input(self, i: int = 0) -> "Node":
        return self.inputs[i]

    def walk(self):
        """Yield this node then all ancestors, each once, in DFS order."""
        seen: set[int] = set()
        stack = [self]
        while stack:
            n = stack.pop()
            if id(n) in seen:
                continue
            seen.add(id(n))
            yield n
            stack.extend(reversed(n.inputs))

    def with_engine(self, engine: str, reason: str = "", **kw: Any) -> "Node":
        return replace(self, assigned_engine=engine, reason=reason, **kw)



    def describe(self) -> str:
        """One-line human description, used by the explain panel."""
        t = self.type
        if self.scan is not None:
            return self.scan.describe()
        if t is NodeType.FILTER:
            return f"filter {self.predicate}"
        if t is NodeType.PROJECT:
            return f"project {', '.join(self.columns) or '*'}"
        if t is NodeType.JOIN:
            return f"{self.join_type} join on {'='.join(self.key_left)}"
        if t is NodeType.GROUPBY:
            return f"group by {', '.join(self.key_left)}"
        if t is NodeType.AGGREGATE:
            return f"aggregate {list(self.agg_functions)}"
        if t is NodeType.SORT:
            return "sort by " + ", ".join(
                f"{c}{'' if asc else ' desc'}" for c, asc in self.sort_keys)
        if t is NodeType.WINDOW:
            return f"window {list(self.window_functions)}"
        if t is NodeType.DEDUPLICATE:
            return f"dedup on {', '.join(self.dedup_keys)} ({self.dedup_strategy})"
        if t is NodeType.CAST:
            return "cast " + ", ".join(f"{k}->{v}" for k, v in self.casts.items())
        if t is NodeType.NULL_HANDLE:
            return f"null-handle ({self.null_strategy})"
        if t is NodeType.PYTHON_UDF:
            return f"udf {self.udf_name or 'anonymous'}"
        if t is NodeType.QUALITY_CHECK:
            return f"quality {list(dict(self.quality_rules))}"
        if t is NodeType.WRITE:
            return f"write {self.target} [{self.write_format or 'native'}]"
        if t is NodeType.CACHE:
            return "cache boundary" + (f" ({self.target})" if self.target else "")
        if t is NodeType.MATERIALIZE:
            return "materialize" + (f" {self.target}" if self.target else "")
        if t is NodeType.LIMIT:
            return f"limit {self.limit}"
        if t is NodeType.UNION:
            return "union"
        return str(t)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        eng = f"@{self.assigned_engine}" if self.assigned_engine else ""
        return f"<{self.type.value} {self.id}{eng}>"


def topological_order(nodes: "list[Node] | Node") -> list[Node]:
    """Deterministic topological order, children before parents.

    **Input order is the tie-break, not ``Node.id``.**

    This used to sort by ``c.id``, which is a UUID. That made the order
    deterministic *within one process* and arbitrary *between* two: parsing the
    same pipeline twice produced different traversal orders, so the planner
    saw a different graph and reached a different plan. It has already caused
    a real defect - "the last scan in the segment" alternated between a 100 MB
    and a 2 GB input, so the wrong one was priced.

    Sorting by ``semantic_operation_id`` would be worse, not better: two
    genuinely identical operations in one graph legitimately share an
    operation ID, and the sort would then be arbitrary again while looking
    deterministic. ``Node.inputs`` is in *declared* order, which is real
    information - for a JOIN, ``inputs[0]`` is the left side - so it is used
    directly.

    Roots are likewise taken in the order given, falling back to the order they
    appear in ``nodes``.
    """
    if isinstance(nodes, Node):
        nodes = [nodes]
    roots = [n for n in nodes if n.is_sink] or list(nodes)
    out: list[Node] = []
    state: dict[int, int] = {}  # 0 unvisited, 1 open, 2 done

    def visit(n: Node) -> None:
        st = state.get(id(n), 0)
        if st == 2:
            return
        if st == 1:
            raise ValueError(f"cycle detected at {n.id} ({n.type})")
        state[id(n)] = 1
        # Declared input order. Not sorted: see the docstring.
        for child in n.inputs:
            visit(child)
        state[id(n)] = 2
        out.append(n)

    for r in roots:
        visit(r)
    for n in nodes:
        if state.get(id(n), 0) != 2:
            visit(n)
    return out
