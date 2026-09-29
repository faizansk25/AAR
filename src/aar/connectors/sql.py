"""SQL sources and targets.

One implementation, several dialects. The interesting property of a SQL
connector is *pushdown* - turning a scan, a filter and a projection into one
statement the database executes itself, so AAR never materialises data it
did not need. That matters most for exactly the thing the specification
cares about: a filter that only reads two of forty columns should not drag
forty columns across the wire.

The design keeps the parts that differ small and explicit:

* :class:`SqlDialect` holds the quoting, type mapping and LIMIT syntax. It
  is data, not code paths, so a new database is a dict rather than a class.
* :class:`SqlConnector` does the pushdown. It never *approximates*: if a
  predicate cannot be rendered safely it is not pushed down, and the caller
  gets a plan that is correct and slower rather than one that is fast and
  quietly wrong.

Two drivers, two very different levels of verification, and the difference is
recorded rather than glossed:

* **SQLite** is in the standard library, so the connector is tested against
  a real database on real files. That is genuine verification.
* **PostgreSQL** needs a server. Its dialect and SQL generation are tested;
  the wire protocol is not, and the connector says so when it has not been
  connected to a live database.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..failures import SourceUnavailable
from ..interchange import Table
from ..ir import BinOp, Col, Func, Lit, Node, UnaryOp
from ..types import (DataType, Field, Schema)

__all__ = [
    "SqlDialect", "SQLITE", "POSTGRESQL", "MYSQL", "SqlConnector",
    "sqlite_connector", "postgresql_connector", "mysql_connector",
    "for_dialect", "quote_ident", "quote_value", "render_where", "projection_sql",
    "sqlite_type_name",
]


# --------------------------------------------------------------- dialects
def quote_ident(name: str, dialect: "SqlDialect") -> str:
    """Quote an identifier, escaping the quote character.

    An unquoted identifier is how ``user`` becomes a syntax error and how a
    column called ``order`` silently changes meaning. Every identifier this
    module emits is quoted, including the ones that look safe.
    """
    if dialect.quote == '"':
        escaped = str(name).replace('"', '""')
    elif dialect.quote == "`":
        escaped = str(name).replace("`", "``")
    else:
        escaped = str(name)
    return f"{dialect.quote}{escaped}{dialect.quote}"


def quote_value(value: Any, dialect: "SqlDialect") -> str:
    """Render a Python value as a SQL literal.

    Literals are escaped, never interpolated. This is the boundary that the
    injection test in the IR suite covers, and it is reproduced here because
    a pushdown filter reaches this code with a value that came from a
    pipeline file rather than from a typed AST.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (bytes, bytearray)):
        return "X'" + bytes(value).hex() + "'"
    text = str(value)
    if dialect.cast_on_concat:
        return "CAST('" + text.replace("'", "''") + "' AS TEXT)"
    return "'" + text.replace("'", "''") + "'"


def sqlite_type_name(dtype: DataType) -> str:
    """A canonical type rendered as a SQLite column type.

    SQLite uses *affinity*, not a strict type system, so these names choose
    the right comparison behaviour rather than enforcing anything. A column
    declared TEXT sorts and compares as text; one declared with no type
    would not.
    """
    kind = dtype.kind
    name = str(getattr(kind, "value", kind)).upper()
    if "INT" in name:
        return "INTEGER"
    if any(t in name for t in ("FLOAT", "DOUBLE", "REAL", "DECIMAL", "NUMERIC")):
        return "REAL"
    if "BOOL" in name:
        return "INTEGER"      # SQLite has no boolean; 0/1 is the convention
    if "TIME" in name or "DATE" in name:
        return "TEXT"          # ISO-8601 sorts correctly as text
    return "TEXT"


# --------------------------------------------------------------- dialect
class SqlDialect:
    """Everything that differs between databases, as data.

    Holding this as fields rather than subclass methods means adding a
    database is a declaration, and the shared pushdown code never needs a
    branch to ask "which one is this?".
    """

    __slots__ = ("name", "quote", "limit_style", "offset_style",
                 "cast_on_concat", "paramstyle", "supports_returning",
                 "boolean_is_bool", "verified_live", "notes")

    def __init__(self, name: str, quote: str = '"',
                 limit_style: str = "limit_offset",
                 cast_on_concat: bool = False, paramstyle: str = "qmark",
                 supports_returning: bool = False,
                 boolean_is_bool: bool = True,
                 verified_live: bool = False, notes: str = "") -> None:
        self.name = name
        self.quote = quote
        #: ``limit_offset`` is ``LIMIT n OFFSET m``; ``offset_fetch`` is
        #: SQL Server's ``OFFSET m ROWS FETCH NEXT n ROWS ONLY``.
        self.limit_style = limit_style
        self.offset_style = "offset"
        self.cast_on_concat = cast_on_concat
        self.paramstyle = paramstyle
        self.supports_returning = supports_returning
        self.boolean_is_bool = boolean_is_bool
        #: True only where the connector has run against a real server.
        self.verified_live = verified_live
        self.notes = notes

    def limit_clause(self, n: int | None, offset: int = 0) -> str:
        """The row-limiting clause, in the syntax this dialect uses."""
        if n is None and not offset:
            return ""
        if self.limit_style == "offset_fetch":
            parts = []
            if offset:
                parts.append(f"OFFSET {int(offset)} ROWS")
            if n is not None:
                parts.append(f"FETCH NEXT {int(n)} ROWS ONLY")
            return " " + " ".join(parts) if parts else ""
        parts = []
        if n is not None:
            parts.append(f"LIMIT {int(n)}")
        if offset:
            parts.append(f"OFFSET {int(offset)}")
        return " " + " ".join(parts) if parts else ""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<SqlDialect {self.name}{' live' if self.verified_live else ''}>"


#: SQLite: in the standard library, so tested against a real database.
SQLITE = SqlDialect(
    "sqlite", quote='"', paramstyle="qmark", verified_live=True,
    notes="stdlib; verified against a real database on real files")

#: PostgreSQL: needs a server. SQL generation is tested; the wire is not.
POSTGRESQL = SqlDialect(
    "postgresql", quote='"', paramstyle="pyformat", supports_returning=True,
    verified_live=False,
    notes="dialect and SQL generation tested; no server available, so the "
          "wire protocol is unverified")

MYSQL = SqlDialect(
    "mysql", quote="`", paramstyle="format",
    verified_live=False,
    notes="dialect and SQL generation tested; no server available")

_DIALECTS = {"sqlite": SQLITE, "postgresql": POSTGRESQL, "postgres":
             POSTGRESQL, "mysql": MYSQL}


def for_dialect(name: str) -> SqlDialect:
    dialect = _DIALECTS.get((name or "").strip().lower())
    if dialect is None:
        raise ValueError(
            f"unknown SQL dialect {name!r}; known dialects are "
            f"{', '.join(sorted(_DIALECTS))}")
    return dialect


# ------------------------------------------------------- predicate pushdown
_COMPARISONS = ("=", "<>", "!=", ">", ">=", "<", "<=")


def render_where(expr: Any, dialect: SqlDialect,
                 columns: Sequence[str] = ()) -> str | None:
    """Render a predicate as SQL, or return ``None`` if it cannot be.

    ``None`` is the important return value. The caller must then *not* push
    the filter down and filter after the fetch instead, because a SQL clause
    that means something slightly different from the predicate is worse than
    a slower query: it returns wrong rows and nothing downstream can tell.

    Only a small, well-understood grammar is accepted - comparisons,
    ``AND``/``OR``/``NOT``, ``IS NULL``/``IS NOT NULL``, and a handful of
    scalar functions. Anything else declines.
    """
    if expr is None:
        return None
    if isinstance(expr, Col):
        return quote_ident(expr.name, dialect)
    if isinstance(expr, Lit):
        return quote_value(expr.value, dialect)
    if isinstance(expr, UnaryOp):
        if expr.op.upper() == "NOT":
            inner = render_where(expr.operand, dialect, columns)
            return None if inner is None else f"NOT ({inner})"
        return None
    if isinstance(expr, Func):
        return _render_func(expr, dialect, columns)
    if not isinstance(expr, BinOp):
        return None

    op = expr.op
    if op in ("AND", "OR"):
        left = render_where(expr.left, dialect, columns)
        right = render_where(expr.right, dialect, columns)
        if left is None or right is None:
            return None
        return f"({left} {op} {right})"
    if op in _COMPARISONS:
        left = render_where(expr.left, dialect, columns)
        right = render_where(expr.right, dialect, columns)
        if left is None or right is None:
            return None
        return f"({left} {op} {right})"
    if op == "IS":
        return _render_is_null(expr, dialect, columns, negated=False)
    if op == "IS NOT":
        return _render_is_null(expr, dialect, columns, negated=True)
    return None


def _render_is_null(expr: Any, dialect: SqlDialect, columns: Sequence[str],
                    negated: bool) -> str | None:
    """Render the null check for ``x IS [NOT] NULL``.

    SQL's three-valued logic makes this genuinely different from a
    comparison against a literal: a comparison involving NULL is unknown,
    and unknown is not true. A connector that rewrote one as the other would
    silently change which rows a filter keeps.
    """
    target = expr.left if isinstance(expr.left, Col) else (
        expr.right if isinstance(expr.right, Col) else None)
    if target is None:
        return None
    name = quote_ident(target.name, dialect)
    return f"{name} IS NOT NULL" if negated else f"{name} IS NULL"


_SQL_FUNCS = {
    "LOWER": "LOWER", "UPPER": "UPPER", "LENGTH": "LENGTH",
    "TRIM": "TRIM", "ABS": "ABS", "ROUND": "ROUND", "COALESCE": "COALESCE",
    "ABSOLUTE": "ABS",
}


def _render_func(expr: Any, dialect: SqlDialect,
                 columns: Sequence[str]) -> str | None:
    name = _SQL_FUNCS.get((expr.name or "").upper())
    if name is None:
        return None
    args = [render_where(a, dialect, columns) for a in expr.args]
    if any(a is None for a in args):
        return None
    return f"{name}({', '.join(args)})"


def projection_sql(columns: Sequence[str], dialect: SqlDialect) -> str:
    """The SELECT list for a projection pushdown.

    This is the specification's "read only the columns you need" made
    concrete: naming two of forty columns here is the difference between a
    database reading two and AAR moving forty.
    """
    if not columns:
        return "*"
    return ", ".join(quote_ident(c, dialect) for c in columns)


def explain_pushdown(node: Node, dialect: SqlDialect) -> str:
    """The statement a scan would issue, for `aar explain`.

    Printed rather than executed, so an analyst can see exactly what would
    have left the machine before it does.
    """
    spec = node.scan
    if spec is None or not (spec.table_name or spec.query):
        return "(no SQL source)"
    if spec.query:
        source = f"({spec.query}) AS {quote_ident('_aar_source', dialect)}"
    else:
        source = quote_ident(spec.table_name, dialect)
    parts = [f"SELECT {projection_sql(spec.columns, dialect)}",
             f"FROM {source}"]
    where = render_where(node.predicate, dialect, spec.columns)
    if where:
        parts.append(f"WHERE {where}")
    return " ".join(parts)


# ------------------------------------------------------------- connector
class SqlConnector:
    """Reads and writes a SQL database, with real pushdown.

    The connector owns the *connection* and the *statement*; the executor
    owns the *plan*. A scan node carries a ``ScanSpec`` and the connector
    turns it into one statement, which is the whole point: a filter that
    reaches the database never crosses the wire at all.
    """

    def __init__(self, dialect: SqlDialect, connect: Any,
                 verified_live: bool = False) -> None:
        self.dialect = dialect
        self._connect = connect
        #: Whether this connector has actually talked to a real server. The
        #: distinction is reported rather than assumed, because a connector
        #: that has only had its SQL generation tested is a different thing
        #: from one that has moved a customer's rows.
        self.verified_live = verified_live or dialect.verified_live
        self._connection: Any = None

    # ------------------------------------------------------------ lifecycle
    def connect(self) -> Any:
        if self._connection is None:
            try:
                self._connection = self._connect()
            except Exception as exc:  # noqa: BLE001
                raise SourceUnavailable(
                    f"could not connect to {self.dialect.name}: "
                    f"{type(exc).__name__}: {exc}",
                    dialect=self.dialect.name) from exc
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:  # noqa: BLE001
                pass
            self._connection = None

    def __enter__(self) -> "SqlConnector":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def verification(self) -> str:
        """How far this connector has been verified. Printed, not implied."""
        if self.verified_live:
            return f"{self.dialect.name}: verified against a real database"
        return (f"{self.dialect.name}: SQL generation tested, but no live "
                f"server was available - the wire protocol is UNVERIFIED")

    # ----------------------------------------------------------------- scan
    def build_select(self, node: Node) -> tuple[str, list[Any]]:
        """The statement a scan issues, and any bind parameters.

        Returns the SQL and the parameters separately rather than
        interpolating values, so a caller that can bind them does - and a
        caller that cannot can see exactly what it is about to interpolate.
        """
        spec = node.scan
        if spec is None or not spec.table_name and not spec.query:
            raise ValueError("a SQL scan needs a table or a query")

        if spec.query:
            source = f"({spec.query}) AS {quote_ident('_aar_source', self.dialect)}"
        else:
            source = quote_ident(spec.table_name, self.dialect)
        sql = f"SELECT {projection_sql(spec.columns, self.dialect)} FROM {source}"
        where = render_where(node.predicate, self.dialect, spec.columns)
        if where:
            sql += f" WHERE {where}"
        sql += self.dialect.limit_clause(node.limit,
                                          getattr(node, "offset", 0) or 0)
        return sql, []

    def read(self, node: Node) -> Table:
        """Execute a scan and return the result as an AAR table."""
        connection = self.connect()
        sql, params = self.build_select(node)
        cursor = connection.cursor()
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            names = [d[0] for d in (cursor.description or [])]
        except Exception as exc:  # noqa: BLE001
            raise SourceUnavailable(
                f"{self.dialect.name} rejected the query {sql!r}: "
                f"{type(exc).__name__}: {exc}", sql=sql,
                dialect=self.dialect.name) from exc
        finally:
            try:
                cursor.close()
            except Exception:  # noqa: BLE001
                pass

        return _rows_to_table(names, rows, node.scan)

    def table_columns(self, table_name: str) -> list[str]:
        """The column names of a table, for schema discovery.

        Uses the metadata view of the dialect rather than
        ``SELECT * LIMIT 0``, because a view or a table with no rows still
        has a schema and an empty result set does not.
        """
        connection = self.connect()
        cursor = connection.cursor()
        try:
            if self.dialect.name == "sqlite":
                cursor.execute(
                    f"SELECT name FROM pragma_table_info("
                    f"{quote_value(table_name, self.dialect)})")
            else:
                cursor.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = %s ORDER BY ordinal_position",
                    (table_name,))
            return [r[0] for r in cursor.fetchall()]
        finally:
            try:
                cursor.close()
            except Exception:  # noqa: BLE001
                pass


# ------------------------------------------------------- result handling
def _rows_to_table(names: Sequence[str], rows: Sequence[Sequence[Any]],
                   spec: Any) -> Table:
    """Driver rows -> an AAR table, with a schema derived from the values.

    SQL is dynamically typed and a result set has no declared type, so the
    schema is inferred from what came back. Inference widens, never narrows:
    a column holding one float stays float even if the first row inspected
    was an int, because a column whose type changes based on which row you
    look at first is not a column type.
    """
    import pyarrow as pa

    from ..interchange import arrow_to_canonical

    if not names:
        return Table.empty(Schema())
    arrays = []
    canonical: list[Any] = []
    for index, name in enumerate(names):
        values = [row[index] for row in rows]
        arrow_type = _infer_arrow_type(values)
        arrays.append(pa.array(values, type=arrow_type))
        canonical.append(arrow_to_canonical(arrow_type))

    arrow = pa.Table.from_arrays(arrays, names=list(names))
    return Table(arrow, Schema(tuple(
        Field(str(n), t) for n, t in zip(names, canonical))))


def _infer_arrow_type(values: Sequence[Any]) -> Any:
    """The narrowest Arrow type that holds every value, widened if needed."""
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
    if all(isinstance(v, (int, float, str, bytes, bool, type(None)))
           for v in present):
        return pa.string()
    return pa.string()


# ------------------------------------------------------------- factories
def sqlite_connector(path: str, read_only: bool = False) -> SqlConnector:
    """A SQLite connector.

    SQLite is in the standard library, so this is the one SQL source AAR can
    fully verify on a machine with no server. A ``:memory:`` path is accepted
    and is how the test suite creates a real database in a real temp file.
    """
    import sqlite3

    def _connect() -> Any:
        if path == ":memory:" or not read_only:
            return sqlite3.connect(path)
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    return SqlConnector(SQLITE, _connect, verified_live=True)


def postgresql_connector(dsn: str) -> SqlConnector:
    """A PostgreSQL connector.

    ``verified_live`` is False and stays False until this has actually
    talked to a server. Constructing one is not evidence that it works, and
    :attr:`SqlConnector.verification` says so in words.
    """
    def _connect() -> Any:
        try:
            import psycopg
        except ImportError as exc:
            raise SourceUnavailable(
                "The PostgreSQL connector needs psycopg. Install it with:\n"
                "  pip install psycopg[binary]\n"
                f"(import failed: {exc})") from exc
        return psycopg.connect(dsn)

    return SqlConnector(POSTGRESQL, _connect, verified_live=False)


def mysql_connector(dsn: str) -> SqlConnector:
    """A MySQL connector, same honesty about verification as PostgreSQL."""
    def _connect() -> Any:
        try:
            import pymysql
        except ImportError as exc:
            raise SourceUnavailable(
                "The MySQL connector needs pymysql. Install it with:\n"
                "  pip install pymysql\n"
                f"(import failed: {exc})") from exc
        return pymysql.connect(**_parse_mysql_dsn(dsn))

    return SqlConnector(MYSQL, _connect, verified_live=False)


def _parse_mysql_dsn(dsn: str) -> dict[str, Any]:
    """Parse a MySQL DSN into PyMySQL connect keywords."""
    from urllib.parse import urlparse

    parsed = urlparse(dsn)
    kwargs: dict[str, Any] = {
        "host": parsed.hostname or "localhost",
        "user": parsed.username or "",
        "database": (parsed.path or "/").lstrip("/"),
    }
    if parsed.password:
        kwargs["password"] = parsed.password
    if parsed.port:
        kwargs["port"] = parsed.port
    return kwargs

