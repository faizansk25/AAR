"""How a source is located, resolved, and parsed.

Every consumer of a :class:`~aar.ir.ScanSpec` - the profiler, the planner,
each engine - used to work out for itself which field held the path or DSN.
They disagreed. The profiler read ``connection`` while the Arrow engine read
``path`` and defaulted to ``:memory:``, so a pipeline built with the public
``sdk.sql()`` API profiled one database and executed against an empty one.
The same split applied to CSV parsing, where the profiler accepted embedded
newlines and execution did not.

Both are defined once here and imported by both halves. A source that is
measured must be the source that is read, and a file that parses during
profiling must parse during execution.
"""

from __future__ import annotations

from typing import Any

__all__ = ["resolve_sql_connection", "csv_read_options",
           "is_sqlite_connection", "sql_file_path"]


def sql_file_path(connection: Any) -> str | None:
    """The SQLite file a connection names, or ``None`` if it names no file.

    SQLite is the only dialect whose connection identifies a file rather
    than a server, so it is the only case where a path can be recovered
    from a connection string. Server DSNs, in-memory databases, and an open
    ``sqlite3.Connection`` (whose type exposes no path attribute) all
    return ``None``, which callers must treat as "cannot be profiled by
    touching the file" rather than as an error.
    """
    if not connection:
        return None
    if isinstance(connection, str):
        text = connection.strip()
        # A URI (file:...?mode=ro) still names a file, but the prefix has to
        # come off before sqlite3.connect will accept it.
        if text.startswith("file:"):
            return text[len("file:"):].split("?", 1)[0]
        # A host:port DSN names a server, not a file.
        if (not text or text == ":memory:"
                or text.startswith(("http", "postgres", "mysql", "trino",
                                    "mssql", "oracle"))):
            return None
        return text
    # An open connection has no public way to report the file it was opened
    # with. Guessing a path could profile a different database from the one
    # the pipeline reads, so this declines.
    return None


def is_sqlite_connection(connection: Any) -> bool:
    """True when this connection names a SQLite file we can open directly."""
    return sql_file_path(connection) is not None


def resolve_sql_connection(spec: Any) -> str | None:
    """The connection a SQL scan should use, from whichever field holds it.

    ``connection`` is what the SDK sets and what every other dialect needs.
    ``path`` is retained because a hand-built spec may set only that, and
    ``dsn`` because the Mongo scan uses the name. Order matters: the
    declared connection wins, so a spec carrying both never has its
    connection silently replaced by a path.

    Returns ``None`` when nothing resolves, and the caller must refuse
    rather than substituting a default. Defaulting to ``:memory:`` is how
    a typo in a connection string turns into "no such table" three layers
    away from the mistake.
    """
    for field in ("connection", "path", "dsn"):
        value = getattr(spec, field, None)
        if value:
            return value
    return None


def csv_read_options(delimiter: str | None = None):
    """``(ParseOptions, ReadOptions)`` shared by CSV profiling and execution.

    Delegates to the engine so there is genuinely one definition: the
    profiler imported this, the Arrow engine called it, and they still
    disagreed until both were pointed here. A module-level function that
    reimplemented the options would be a third place to drift.
    """
    from .engines.arrow_engine import ArrowEngine

    return ArrowEngine.csv_read_options(delimiter)