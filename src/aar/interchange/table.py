"""Arrow-native interchange.

The specification is blunt about this: never convert

    Pandas -> Python objects -> JSON -> Polars -> Python list -> cuDF

and always use

    Arrow -> Arrow -> Arrow -> Arrow

So the *only* thing that moves between engines in AAR is a
:class:`Table`, which wraps an ``pyarrow.Table`` and nothing else. Engines
convert into Arrow and out of Arrow at their own boundary; they never convert
into each other's type system. That is what makes the interchange lossless by
construction rather than by careful coding.

The other job here is keeping the *metadata* attached across those hops.
``pyarrow.Table`` carries types but not classification or lineage, and AAR's
privacy guarantees would evaporate at the first engine boundary without it.
So :class:`Table` carries a canonical :class:`~aar.types.Schema` alongside
the Arrow schema and reconciles the two on every conversion.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

from ..types import (Field, LineageRef, Schema, TypeKind, UnmappableType,
                     from_source, lossy)

__all__ = ["Table", "arrow_to_canonical", "canonical_to_arrow", "reconcile",
           "require_arrow"]


def require_arrow():
    """Return ``pyarrow``, or explain precisely why it is unavailable."""
    try:
        import pyarrow
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Apache Arrow is required to move data between engines. "
            "Install it with:  pip install pyarrow\n"
            f"(import failed: {exc})") from exc
    return pyarrow

# ------------------------------------------------------- type conversion
def arrow_to_canonical(pa_type: Any) -> "Any":
    """Map an Arrow type onto a canonical :class:`DataType`.

    Total over Arrow's documented type space. Anything unmappable raises
    rather than degrading to text, because a silent UTF-8 fallback is how a
    numeric column quietly becomes a string column.
    """
    import pyarrow as pa
    import pyarrow.types as pat

    if pat.is_null(pa_type):
        from ..types import NULL
        return NULL
    if pat.is_boolean(pa_type):
        from ..types import BOOLEAN
        return BOOLEAN
    if pat.is_int8(pa_type):
        from ..types import INT8
        return INT8
    if pat.is_int16(pa_type):
        from ..types import INT16
        return INT16
    if pat.is_int32(pa_type):
        from ..types import INT32
        return INT32
    if pat.is_int64(pa_type):
        from ..types import INT64
        return INT64
    if pat.is_uint8(pa_type):
        from ..types import UINT8
        return UINT8
    if pat.is_uint16(pa_type):
        from ..types import UINT16
        return UINT16
    if pat.is_uint32(pa_type):
        from ..types import UINT32
        return UINT32
    if pat.is_uint64(pa_type):
        from ..types import UINT64
        return UINT64
    if pat.is_float32(pa_type):
        from ..types import FLOAT32
        return FLOAT32
    if pat.is_float64(pa_type) or pat.is_float16(pa_type):
        from ..types import FLOAT64
        return FLOAT64
    if pat.is_decimal128(pa_type):
        from ..types import DECIMAL
        return DECIMAL(pa_type.precision, pa_type.scale)
    if pat.is_decimal256(pa_type):
        from ..types import DECIMAL
        return DECIMAL(min(pa_type.precision, 76), pa_type.scale)
    if pat.is_string(pa_type) or pat.is_large_string(pa_type):
        from ..types import UTF8
        return UTF8
    if pat.is_binary(pa_type) or pat.is_fixed_size_binary(pa_type):
        from ..types import BINARY
        return BINARY
    if pat.is_large_binary(pa_type):
        from ..types import LARGE_BINARY
        return LARGE_BINARY
    if pat.is_date32(pa_type):
        from ..types import DATE32
        return DATE32
    if pat.is_date64(pa_type):
        from ..types import DATE64
        return DATE64
    if pat.is_time32(pa_type):
        from ..types import TIME32
        return TIME32.replace(unit=pa_type.unit)
    if pat.is_time64(pa_type):
        from ..types import TIME64
        return TIME64.replace(unit=pa_type.unit)
    if pat.is_timestamp(pa_type):
        from ..types import TIMESTAMP
        return TIMESTAMP(pa_type.unit, pa_type.tz or None)
    if pat.is_duration(pa_type):
        from ..types import DURATION
        return DURATION.replace(unit=pa_type.unit)
    if pat.is_dictionary(pa_type):
        return arrow_to_canonical(pa_type.value_type)
    if pat.is_list(pa_type):
        from ..types import list_of
        return list_of(arrow_to_canonical(pa_type.value_type))
    if pat.is_map(pa_type):
        from ..types import map_of
        return map_of(arrow_to_canonical(pa_type.key_type),
                      arrow_to_canonical(pa_type.item_type))
    if pat.is_struct(pa_type):
        from ..types import struct_of
        return struct_of(*[
            Field(f.name, arrow_to_canonical(f.type)) for f in pa_type])
    raise UnmappableType(
        f"Arrow type {pa_type} has no canonical representation")


def canonical_to_arrow(dt: Any) -> Any:
    """Map a canonical type onto an Arrow type."""
    import pyarrow as pa

    k = dt.kind
    # Arrow type constructors are functions: `pa.int64`, not `pa.int64()`.
    # Mapping to the constructor and calling it once keeps the table honest.
    simple = {
        TypeKind.NULL: lambda: pa.null(),
        TypeKind.BOOLEAN: lambda: pa.bool_(),
        TypeKind.INT8: lambda: pa.int8(),
        TypeKind.INT16: lambda: pa.int16(),
        TypeKind.INT32: lambda: pa.int32(),
        TypeKind.INT64: lambda: pa.int64(),
        TypeKind.UINT8: lambda: pa.uint8(),
        TypeKind.UINT16: lambda: pa.uint16(),
        TypeKind.UINT32: lambda: pa.uint32(),
        TypeKind.UINT64: lambda: pa.uint64(),
        TypeKind.FLOAT32: lambda: pa.float32(),
        TypeKind.FLOAT64: lambda: pa.float64(),
        TypeKind.UTF8: lambda: pa.string(),
        TypeKind.BINARY: lambda: pa.binary(),
        TypeKind.LARGE_BINARY: lambda: pa.large_binary(),
        TypeKind.DATE32: lambda: pa.date32(),
        TypeKind.DATE64: lambda: pa.date64(),
    }
    if k in simple:
        return simple[k]()

    if k is TypeKind.TIMESTAMP:
        return pa.timestamp(dt.unit or "us", tz=dt.timezone)
    if k is TypeKind.TIME32:
        return pa.time32(dt.unit or "ms")
    if k is TypeKind.TIME64:
        return pa.time64(dt.unit or "us")
    if k is TypeKind.DURATION:
        return pa.duration(dt.unit or "us")
    if k is TypeKind.DECIMAL:
        if (dt.precision or 38) > 38:
            return pa.decimal256(dt.precision or 38, dt.scale or 0)
        return pa.decimal128(dt.precision or 38, dt.scale or 0)
    if k is TypeKind.CATEGORICAL:
        return pa.dictionary(pa.int32(), canonical_to_arrow(dt.dictionary_values
                                                           or _utf8()))
    if k is TypeKind.LIST:
        return pa.list_(canonical_to_arrow(dt.value_type or _utf8()))
    if k is TypeKind.MAP:
        return pa.map_(canonical_to_arrow(dt.key_type or _utf8()),
                       canonical_to_arrow(dt.value_type or _utf8()))
    if k is TypeKind.STRUCT:
        return pa.struct([pa.field(f.name, canonical_to_arrow(f.type))
                          for f in dt.fields])
    raise UnmappableType(f"canonical type {dt} has no Arrow representation")


def _utf8():
    from ..types import UTF8
    return UTF8



# ------------------------------------------------------------------ table
class Table:
    """An Arrow table plus the AAR metadata that Arrow cannot carry.

    This is the only currency that crosses an engine boundary. Construct it
    from Arrow, and read it back with :meth:`to_arrow`; classification and
    lineage travel with it in between.
    """

    __slots__ = ("_arrow", "_schema")

    def __init__(self, arrow_table: Any, schema: Schema | None = None) -> None:
        self._arrow = arrow_table
        if schema is None:
            schema = self._schema_from_arrow(arrow_table)
        elif len(schema) != arrow_table.num_columns:
            raise ValueError(
                f"schema has {len(schema)} field(s) but the table has "
                f"{arrow_table.num_columns} column(s)")
        self._schema = schema

    @staticmethod
    def _schema_from_arrow(arrow_table: Any) -> Schema:
        return Schema(tuple(
            Field(f.name, arrow_to_canonical(f.type), nullable=f.nullable)
            for f in arrow_table.schema))

    # ------------------------------------------------------------- factories
    @classmethod
    def from_arrow(cls, arrow_table: Any) -> "Table":
        return cls(arrow_table)

    @classmethod
    def empty(cls, schema: Schema) -> "Table":
        """A zero-row table with a declared schema.

        Needed for correctness, not tidiness: a filter that matches nothing,
        or a group-by over an empty input, must produce a correctly typed
        result rather than an untyped empty frame that breaks the schema
        contract downstream.
        """
        import pyarrow as pa

        arrow_schema = pa.schema([
            pa.field(f.name, canonical_to_arrow(f.type), nullable=f.nullable)
            for f in schema.fields
        ])
        return cls(pa.Table.from_pylist([], schema=arrow_schema), schema)

    # ------------------------------------------------------------ properties
    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def arrow(self) -> Any:
        return self._arrow

    @property
    def num_rows(self) -> int:
        return self._arrow.num_rows

    @property
    def num_columns(self) -> int:
        return self._arrow.num_columns

    @property
    def nbytes(self) -> int:
        return int(self._arrow.nbytes)

    @property
    def column_names(self) -> tuple[str, ...]:
        """Column names in order.

        A tuple, to match :attr:`Schema.names` and the IR. Returning Arrow's
        list here would make every caller's comparison against a declared
        schema fail for no reason other than container type.
        """
        return tuple(self._arrow.schema.names)


    def column(self, name: str) -> Any:
        return self._arrow.column(name)

    def to_arrow(self) -> Any:
        return self._arrow

    def __len__(self) -> int:
        return self._arrow.num_rows

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Table({self.num_rows} rows x {self.num_columns} cols, "
                f"{self.nbytes} bytes)")


    # ------------------------------------------------------------- transform
    def with_schema(self, schema: Schema) -> "Table":
        """The same rows under a different declared schema.

        Used after an operation that reorders, renames or retypes columns but
        not the data. The column *count* is checked; anything else would let a
        schema lie about the data it claims to describe.
        """
        if len(schema) != self.num_columns:
            raise ValueError(
                f"cannot relabel {self.num_columns} column(s) as "
                f"{len(schema)} field(s)")
        return Table(self._arrow, schema)

    def select(self, names: Sequence[str]) -> "Table":
        """Project columns, keeping their metadata."""
        arrow_names = list(self._arrow.schema.names)
        unknown = [n for n in names if n not in arrow_names]
        if unknown:
            raise KeyError(
                f"no such column(s): {', '.join(unknown)} "
                f"(have: {', '.join(arrow_names)})")
        index = [arrow_names.index(n) for n in names]
        return Table(self._arrow.select(index), self._schema.select(names))

    def slice(self, offset: int = 0, length: int | None = None) -> "Table":
        if offset == 0 and length is None:
            return self
        return Table(self._arrow.slice(offset, length), self._schema)

    def take(self, indices: Any) -> "Table":
        return Table(self._arrow.take(indices), self._schema)

    def sort_indices(self, keys: Sequence[tuple[str, bool]]) -> Any:
        """A permutation that orders the table by ``keys``.

        Returns Arrow indices rather than a sorted table, so an engine that
        sorts natively can do it in its own space and AAR does not have to
        round-trip the data through Arrow to find out the order.
        """
        import pyarrow.compute as pc

        order = [(k, "ascending" if asc else "descending") for k, asc in keys]
        missing = [k for k, _ in order if k not in self._arrow.schema.names]
        if missing:
            raise KeyError(f"no such column(s): {', '.join(missing)}")
        return pc.sort_indices(self._arrow, sort_keys=order)

    # ---------------------------------------------------------------- batches
    def batches(self, batch_size: int = 8192) -> Iterator["Table"]:
        """Iterate in batches, for streaming rather than materialising.

        A pipeline that only aggregates never needs the whole input at once,
        and materialising it is how a laptop runs out of memory on a file the
        specification would call small.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        total = self.num_rows
        if total == 0:
            yield self
            return
        offset = 0
        while offset < total:
            yield self.slice(offset, min(batch_size, total - offset))
            offset += batch_size

    # -------------------------------------------------------------- metadata
    def tagged(self, column: str, *tags: str) -> "Table":
        """A copy with classification applied to one column."""
        if not self._schema.has(column):
            raise KeyError(f"no such column: {column!r}")
        return Table(self._arrow, Schema(tuple(
            f.with_classification(*tags) if f.name == column else f
            for f in self._schema)))

    def with_lineage(self, ref: LineageRef) -> "Table":
        """A copy whose every column records the given provenance."""
        return Table(self._arrow, Schema(tuple(
            f.with_lineage(ref) for f in self._schema)))

    def classifications(self) -> dict[str, frozenset[str]]:
        return {f.name: f.classification for f in self._schema}



# ------------------------------------------------------- boundary reconcile
def reconcile(result: Table, source: Table | None = None,
              derived: Mapping[str, frozenset[str]] | None = None) -> Table:
    """Re-attach AAR metadata that an engine's own conversion discarded.

    This exists because of a specific, silent privacy bug. An engine that
    round-trips through its native frame - DuckDB's relation, a Polars
    DataFrame, a pandas DataFrame - comes back as a bare Arrow table, and
    ``Table(arrow)`` derives a schema whose every field is *unclassified*.
    The data is right, the numbers are right, the plan is right, and a
    CONFIDENTIAL column has silently become public. A policy that trusts
    classification then has nothing to act on and the data reaches the sink
    unmasked, with no error anywhere to notice.

    The rule, applied per output column:

    * a column that also exists in ``source`` keeps that column's tags -
      filter, sort, limit and join preserve columns, so the tags still
      describe the same data;
    * otherwise a column named in ``derived`` gets the tags the lineage
      rules computed for it - an aggregate over a confidential column, a
      UDF that could have read every column, a join that saw both sides;
    * otherwise it is unclassified, which is the honest answer for a column
      that genuinely came from nowhere.

    Note the first rule is *not* a general identity. It holds only because
    the caller passes the table the operation was derived from, and
    ``reconcile`` cannot verify that, so being right about the ``source`` is
    the caller's job.
    """
    if source is None and not derived:
        return result
    derived = derived or {}
    fields = []
    for name, field in zip(result.column_names, result.schema.fields):
        classification: frozenset[str] = frozenset()
        if source is not None and source.schema.has(name):
            classification = source.schema.get(name).classification
        elif name in derived:
            classification = frozenset(derived[name])
        fields.append(Field(name, field.type, nullable=field.nullable,
                            classification=classification))
    return result.with_schema(Schema(tuple(fields)))


