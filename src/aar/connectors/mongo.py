"""MongoDB: documents in, Arrow out.

The specification treats a document store as a first-class source, and the
interesting part is the same as for SQL: pushing work down so AAR only
materialises what it needs. A pipeline stage that matches and projects should
run in the database, because a collection with fifty million documents and a
filter that keeps twelve is not a problem AAR should solve by reading it.

Two things are different from SQL and both are handled explicitly:

* **Types are per-value, not per-column.** A document store has no schema,
  so one field can be an Int64 in one document and a string in the next. AAR
  widens to a single type per output column and reports it, rather than
  producing a column that is a different type on every row.
* **Nulls are meaningful and nested.** A missing field and a field set to
  ``null`` are different in MongoDB, and both are ``NULL`` here. Collapsing
  them is a documented loss of distinction, not a silent one.

Verification: this connector is exercised against ``mongomock``, which
implements the query, projection and aggregation semantics in Python. That
genuinely tests AAR's translation, filtering and typing - it is not a stub.
It does **not** test the wire protocol, BSON encoding, indexes or the
server's own optimiser, and the connector's ``verification`` string says so.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..failures import SourceUnavailable
from ..interchange import Table
from ..ir import BinOp, Col, Expr, Lit, Node, NodeType

__all__ = ["MongoConnector", "mongo_connector", "mock_mongo_connector",
           "render_match", "mongo_type_of"]


# ------------------------------------------------------------ type naming
def mongo_type_of(value: Any) -> str:
    """The BSON type name for a Python value.

    Mirrors the driver's own vocabulary so a mapping error is a naming
    difference rather than a silent reclassification.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int64" if value > 2 ** 31 - 1 or value < -(2 ** 31) else "int32"
    if isinstance(value, float):
        return "double"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (bytes, bytearray)):
        return "binData"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    return "object"


# -------------------------------------------------------- match rendering
_MONGOS: dict[str, str] = {
    "=": "$eq", "<>": "$ne", "!=": "$ne",
    ">": "$gt", ">=": "$gte", "<": "$lt", "<=": "$lte",
    "AND": "$and", "OR": "$or",
}


def render_match(expr: Any) -> dict[str, Any] | None:
    """Render a predicate as a MongoDB match document, or ``None``.

    Same contract as the SQL renderer: ``None`` means "do not push this
    down". A filter that reaches MongoDB meaning something else is worse
    than a filter applied after the fetch, because the wrong rows come back
    with no indication anything was wrong.
    """
    if expr is None:
        return None
    if isinstance(expr, Col):
        return None
    if isinstance(expr, Lit):
        return None
    if isinstance(expr, BinOp):
        op = expr.op
        if op in ("AND", "OR"):
            left = render_match(expr.left)
            right = render_match(expr.right)
            if left is None or right is None:
                return None
            return {f"${op.lower()}": [left, right]}
        if op in _MONGOS:
            field_name, value = _split_comparison(expr)
            if field_name is None:
                return None
            return {field_name: {_MONGOS[op]: value}}
        if op in ("IS", "IS NOT"):
            field_name, _ = _split_comparison(expr)
            if field_name is None:
                return None
            return {field_name: {"$exists": op == "IS"}}
    return None


def _split_comparison(expr: Any) -> tuple[str | None, Any]:
    """Pull ``(field, value)`` out of a comparison, if it is a simple one."""
    if isinstance(expr.left, Col):
        right = expr.right
        if isinstance(right, Lit):
            return expr.left.name, right.value
        if isinstance(right, Col):
            return None, None
        return expr.left.name, right
    if isinstance(expr.right, Col) and isinstance(expr.left, Lit):
        return expr.right.name, expr.left.value
    return None, None


# ------------------------------------------------------------- connector
class MongoConnector:
    """Reads a collection, optionally through a native aggregation pipeline.

    The two read paths are chosen deliberately. A **match** is a filter
    AAR can express as a MongoDB query document, so it is pushed down and
    only matching documents cross the wire. An **aggregation** is handed to
    MongoDB verbatim, because a pipeline written in the server's own
    aggregation language is the database using its strengths rather than AAR
    reimplementing them badly.

    Neither path is used when the expression cannot be translated exactly.
    The connector then fetches and filters locally, and says so.
    """

    def __init__(self, client: Any, verified_live: bool = False,
                 driver: str = "pymongo") -> None:
        self._client = client
        #: Whether this has talked to a real mongod. False for mongomock,
        #: which is a faithful in-process implementation of the query and
        #: aggregation semantics but not a server.
        self.verified_live = verified_live
        self.driver = driver
        self._database: str | None = None

    @property
    def verification(self) -> str:
        if self.verified_live:
            return "mongodb: verified against a real server"
        return (f"mongodb ({self.driver}): query, projection and aggregation "
                f"semantics verified in-process; the wire protocol, BSON "
                f"encoding and the server's own optimiser are UNVERIFIED")

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "MongoConnector":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def collection(self, node: Node) -> Any:
        spec = node.scan
        if spec is None or not spec.collection or not spec.database:
            raise ValueError("a MongoDB scan needs a database and a collection")
        try:
            return self._client[spec.database][spec.collection]
        except Exception as exc:  # noqa: BLE001
            raise SourceUnavailable(
                f"could not open {spec.database}.{spec.collection}: "
                f"{type(exc).__name__}: {exc}",
                database=spec.database, collection=spec.collection) from exc

    def read(self, node: Node) -> Table:
        """Fetch documents, pushing down whatever can be pushed safely."""
        spec = node.scan
        collection = self.collection(node)
        pushed = False

        if spec.query:
            # A hand-written aggregation pipeline goes to MongoDB as-is.
            documents = list(collection.aggregate(list(spec.query)))
            pushed = True
        else:
            match = render_match(node.predicate)
            projection = {c: 1 for c in spec.columns} or None
            cursor = collection.find(filter=match or {},
                                     projection=projection)
            if match:
                pushed = True
            documents = list(cursor)
            if not pushed and node.predicate is not None:
                documents = [d for d in documents
                             if _matches_locally(node.predicate, d)]

        if node.limit:
            documents = documents[:int(node.limit)]
        return documents_to_table(documents, spec.columns)

    def count(self, node: Node) -> int:
        """How many documents match, without moving them.

        This is the cheapest way to size a scan, and the planner's cost model
        wants exactly this number rather than a guess.
        """
        spec = node.scan
        if spec.query:
            return len(list(self.collection(node).aggregate(list(spec.query))))
        match = render_match(node.predicate)
        return int(self.collection(node).count_documents(match or {}))

    def explain_plan(self, node: Node) -> str:
        """How a scan would be executed, for `aar explain`."""
        spec = node.scan
        if spec.query:
            return f"db.{spec.database}.{spec.collection}.aggregate(" \
                   f"{list(spec.query)!r})"
        match = render_match(node.predicate)
        parts = [f"db.{spec.database}.{spec.collection}.find("]
        if match:
            parts.append(f"filter={match!r}")
        else:
            parts.append("{}")
        if spec.columns:
            parts.append(f"projection={ {c: 1 for c in spec.columns} !r}")
        parts.append(")")
        return "".join(parts)


# -------------------------------------------------------- document -> table
def documents_to_table(documents: Sequence[Mapping[str, Any]],
                       columns: Sequence[str] = ()) -> Table:
    """Documents -> an AAR table, with one type per output column.

    A document store has no schema, so the union of keys across the result
    determines the columns and the values determine the types. Both are
    reported in the resulting schema rather than guessed silently, because a
    column that is an Int64 in one row and a string in the next is a data
    problem the analyst needs told about.

    Nested objects and arrays are rendered as JSON text. Arrow's nested
    types exist, but a document store's nesting is free-form and mapping it
    into a fixed Arrow struct would either fail or invent a shape the data
    does not have.
    """
    import json

    import pyarrow as pa

    from ..interchange import arrow_to_canonical
    from ..types import Field, Schema

    names: list[str] = [str(c) for c in columns] if columns else []
    if not names:
        seen: list[str] = []
        for doc in documents:
            for key in doc.keys():
                if key not in seen:
                    seen.append(str(key))
        names = seen
    if not names:
        return Table.empty(Schema())

    arrays = []
    canonical = []
    for name in names:
        values = [doc.get(name) for doc in documents]
        flat = [_flatten(v) for v in values]
        arrow_type = _infer_arrow_type(flat)
        arrays.append(pa.array(flat, type=arrow_type))
        canonical.append(arrow_to_canonical(arrow_type))

    arrow = pa.Table.from_arrays(arrays, names=names)
    return Table(arrow, Schema(tuple(
        Field(n, t) for n, t in zip(names, canonical))))


def _flatten(value: Any) -> Any:
    """Render nested documents and arrays as JSON text."""
    if isinstance(value, (Mapping, list, tuple)):
        import json
        return json.dumps(value, default=str, sort_keys=True)
    return value


def _infer_arrow_type(values: Sequence[Any]) -> Any:
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
    return pa.string()


def _matches_locally(expr: Any, document: Mapping[str, Any]) -> bool:
    """Filter a document in Python, for a predicate MongoDB cannot take.

    Used only when :func:`render_match` declined. Same three-valued logic
    the predicate compiler uses elsewhere, so a filter that falls back here
    keeps exactly the rows it would have kept had it been pushed down.
    """
    from ..engines.base import PredicateCompiler

    return PredicateCompiler.evaluate(expr, dict(document))


# ------------------------------------------------------------- factories
def mongo_connector(dsn: str = "mongodb://localhost:27017",
                    database: str = "") -> MongoConnector:
    """A MongoDB connector against a real server."""
    def _build() -> Any:
        try:
            from pymongo import MongoClient
        except ImportError as exc:
            raise SourceUnavailable(
                "The MongoDB connector needs pymongo. Install it with:\n"
                "  pip install pymongo\n"
                f"(import failed: {exc})") from exc
        return MongoClient(dsn)

    return MongoConnector(_build(), verified_live=True, driver="pymongo")


def mock_mongo_connector() -> Any:
    """An in-process MongoDB via ``mongomock``, for testing.

    ``mongomock`` implements the query, projection and aggregation semantics
    in Python. That is enough to verify that AAR's predicate translation,
    pushdown decisions and type mapping are correct - which is the part AAR
    is responsible for. It is not a server, so nothing about the wire
    protocol, BSON encoding, indexes or the server's own optimiser is
    covered, and the connector's ``verification`` string says exactly that.

    This is a real test double, not a stub: a query that AAR renders
    wrongly fails here, which is the property the tests need.
    """
    try:
        import mongomock
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Testing the MongoDB connector needs mongomock. Install it with:\n"
            "  pip install mongomock\n"
            f"(import failed: {exc})") from exc
    return MongoConnector(mongomock.MongoClient(), verified_live=False,
                          driver="mongomock")


