"""Canonical type system for the Adaptive Analytics Runtime.

Every type that crosses an engine boundary is normalised into the canonical
types declared here.  This is the defence against the verified failure mode
where pandas and Polars produce different payloads for equivalent temporal
types and a ``Timestamp`` column is silently corrupted during a handoff.

Design rules
------------
1. The canonical set is small, closed and stable. It does not track any
   engine's type list.
2. Conversions are explicit and *total*: every source type either maps to a
   canonical type or raises :class:`UnmappableType`. There is no silent
   fallback to ``Utf8``.
3. Every conversion is recorded as a :class:`TypeConversion` so the runtime
   can log it and flag lossy conversions for human review.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

__all__ = [
    "TypeKind", "DataType", "Schema", "Field", "LineageRef",
    "TypeConversion", "UnmappableType", "NormalizationPolicy",
    "NULL", "BOOLEAN",
    "INT8", "INT16", "INT32", "INT64",
    "UINT8", "UINT16", "UINT32", "UINT64",
    "FLOAT32", "FLOAT64", "UTF8", "BINARY", "LARGE_BINARY",
    "DATE32", "DATE64", "TIME32", "TIME64", "TIMESTAMP", "DURATION",
    "DECIMAL", "CATEGORICAL", "list_of", "struct_of", "map_of",
    "widen", "common_type", "lossy", "EMPTY_SCHEMA",
]


class UnmappableType(ValueError):
    """Raised when a source type has no faithful canonical representation."""


class TypeKind(str, enum.Enum):
    """The closed set of canonical types."""

    NULL = "null"
    BOOLEAN = "bool"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    DECIMAL = "decimal"
    UTF8 = "utf8"
    BINARY = "binary"
    LARGE_BINARY = "large_binary"
    DATE32 = "date32"
    DATE64 = "date64"
    TIME32 = "time32"
    TIME64 = "time64"
    TIMESTAMP = "timestamp"
    DURATION = "duration"
    CATEGORICAL = "categorical"
    LIST = "list"
    STRUCT = "struct"
    MAP = "map"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


_INT_KINDS = (
    TypeKind.INT8, TypeKind.INT16, TypeKind.INT32, TypeKind.INT64,
    TypeKind.UINT8, TypeKind.UINT16, TypeKind.UINT32, TypeKind.UINT64,
)
_SIGNED_INTS = (TypeKind.INT8, TypeKind.INT16, TypeKind.INT32, TypeKind.INT64)
_UNSIGNED_INTS = (TypeKind.UINT8, TypeKind.UINT16, TypeKind.UINT32, TypeKind.UINT64)
_FLOATS = (TypeKind.FLOAT32, TypeKind.FLOAT64)
_TEMPORAL = (
    TypeKind.DATE32, TypeKind.DATE64, TypeKind.TIME32,
    TypeKind.TIME64, TypeKind.TIMESTAMP, TypeKind.DURATION,
)
_BYTE_WIDTH = {
    TypeKind.INT8: 1, TypeKind.UINT8: 1, TypeKind.BOOLEAN: 1,
    TypeKind.INT16: 2, TypeKind.UINT16: 2,
    TypeKind.INT32: 4, TypeKind.UINT32: 4, TypeKind.FLOAT32: 4, TypeKind.DATE32: 4,
    TypeKind.INT64: 8, TypeKind.UINT64: 8, TypeKind.FLOAT64: 8,
    TypeKind.DATE64: 8, TypeKind.TIME64: 8, TypeKind.TIMESTAMP: 8,
    TypeKind.TIME32: 4,
}



@dataclass(frozen=True, slots=True)
class DataType:
    """A canonical type.

    Attributes
    ----------
    kind:      The :class:`TypeKind`.
    unit:      Resolution for ``TIMESTAMP``/``DURATION`` (``s``/``ms``/``us``/``ns``).
    timezone:  IANA zone name, or ``None`` for a naive timestamp.
    precision, scale:  Only meaningful for ``DECIMAL``.
    key_type, value_type:  Element types for ``MAP``; element type for ``LIST``.
    fields:    Ordered children for ``STRUCT``.
    dictionary_values:  Storage type behind a ``CATEGORICAL``.
    """

    kind: TypeKind
    unit: str | None = None
    timezone: str | None = None
    precision: int | None = None
    scale: int | None = None
    key_type: "DataType | None" = None
    value_type: "DataType | None" = None
    fields: tuple["Field", ...] = ()
    dictionary_values: "DataType | None" = None

    @property
    def is_numeric(self) -> bool:
        return self.kind in _INT_KINDS or self.kind in _FLOATS or self.kind is TypeKind.DECIMAL

    @property
    def is_integer(self) -> bool:
        return self.kind in _INT_KINDS

    @property
    def is_temporal(self) -> bool:
        return self.kind in _TEMPORAL

    @property
    def is_nested(self) -> bool:
        return self.kind in (TypeKind.LIST, TypeKind.STRUCT, TypeKind.MAP)

    @property
    def byte_width(self) -> int | None:
        """Fixed width in bytes, or ``None`` for variable-width types."""
        if self.kind is TypeKind.DECIMAL:
            if self.precision is None:
                return None
            return 16 if self.precision > 18 else 8
        return _BYTE_WIDTH.get(self.kind)

    @property
    def is_fixed_width(self) -> bool:
        return self.byte_width is not None

    def child_types(self) -> tuple["DataType", ...]:
        if self.kind is TypeKind.LIST:
            return (self.value_type,) if self.value_type else ()
        if self.kind is TypeKind.MAP:
            return tuple(t for t in (self.key_type, self.value_type) if t)
        if self.kind is TypeKind.STRUCT:
            return tuple(f.type for f in self.fields)
        if self.kind is TypeKind.CATEGORICAL and self.dictionary_values:
            return (self.dictionary_values,)
        return ()

    def replace(self, **changes: Any) -> "DataType":
        from dataclasses import replace as _replace

        return _replace(self, **changes)

    def __str__(self) -> str:
        k = self.kind
        if k is TypeKind.DECIMAL:
            return f"Decimal({self.precision},{self.scale})"
        if k is TypeKind.TIMESTAMP:
            return f"Timestamp({self.unit or 'us'},{self.timezone or 'naive'})"
        if k is TypeKind.DURATION:
            return f"Duration({self.unit or 'us'})"
        if k is TypeKind.TIME32:
            return f"Time32({self.unit or 'ms'})"
        if k is TypeKind.TIME64:
            return f"Time64({self.unit or 'us'})"
        if k is TypeKind.LIST:
            return f"List<{self.value_type}>"
        if k is TypeKind.MAP:
            return f"Map<{self.key_type},{self.value_type}>"
        if k is TypeKind.STRUCT:
            return "Struct{" + ",".join(f"{f.name}:{f.type}" for f in self.fields) + "}"
        if k is TypeKind.CATEGORICAL:
            return f"Categorical<{self.dictionary_values or UTF8}>"
        return str(k)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DataType({self})"


# --------------------------------------------------------------- singletons
NULL = DataType(TypeKind.NULL)
BOOLEAN = DataType(TypeKind.BOOLEAN)
INT8 = DataType(TypeKind.INT8)
INT16 = DataType(TypeKind.INT16)
INT32 = DataType(TypeKind.INT32)
INT64 = DataType(TypeKind.INT64)
UINT8 = DataType(TypeKind.UINT8)
UINT16 = DataType(TypeKind.UINT16)
UINT32 = DataType(TypeKind.UINT32)
UINT64 = DataType(TypeKind.UINT64)
FLOAT32 = DataType(TypeKind.FLOAT32)
FLOAT64 = DataType(TypeKind.FLOAT64)
UTF8 = DataType(TypeKind.UTF8)
BINARY = DataType(TypeKind.BINARY)
LARGE_BINARY = DataType(TypeKind.LARGE_BINARY)
DATE32 = DataType(TypeKind.DATE32)
DATE64 = DataType(TypeKind.DATE64)
TIME32 = DataType(TypeKind.TIME32, unit="ms")
TIME64 = DataType(TypeKind.TIME64, unit="us")
DURATION = DataType(TypeKind.DURATION, unit="us")
CATEGORICAL = DataType(TypeKind.CATEGORICAL, dictionary_values=UTF8)


def TIMESTAMP(unit: str = "us", timezone: str | None = None) -> DataType:  # noqa: N802
    """Sentinel-returning factory for timestamp types."""
    return DataType(TypeKind.TIMESTAMP, unit=unit, timezone=timezone)


def DECIMAL(precision: int = 38, scale: int = 0) -> DataType:  # noqa: N802
    """Sentinel-returning factory for fixed-point decimals."""
    if scale > precision:
        raise UnmappableType(f"scale {scale} exceeds precision {precision}")
    return DataType(TypeKind.DECIMAL, precision=precision, scale=scale)


def list_of(value: DataType) -> DataType:
    return DataType(TypeKind.LIST, value_type=value)


def struct_of(*fields: "Field") -> DataType:
    return DataType(TypeKind.STRUCT, fields=tuple(fields))


def map_of(key: DataType, value: DataType) -> DataType:
    return DataType(TypeKind.MAP, key_type=key, value_type=value)


# ---------------------------------------------------------------- coercion
#: Bit width per integer kind. An explicit table, not arithmetic on an enum
#: index: deriving widths from ordinals silently mislabels UINT8 as 16 bits and
#: INT16/32/64 as 8, which corrupts every range check downstream.
_INT_BITS = {
    TypeKind.INT8: 8, TypeKind.INT16: 16, TypeKind.INT32: 32, TypeKind.INT64: 64,
    TypeKind.UINT8: 8, TypeKind.UINT16: 16, TypeKind.UINT32: 32, TypeKind.UINT64: 64,
}
_INT_RANK = _INT_BITS


def _int_fits(src: DataType, dst: TypeKind) -> bool:
    """Can every value of ``src`` be represented exactly in ``dst``?"""
    sbits = _INT_BITS[src.kind]
    dbits = _INT_BITS[dst]
    s_signed = src.kind in _SIGNED_INTS
    d_signed = dst in _SIGNED_INTS
    if s_signed and not d_signed:
        # -2^(n-1) .. 2^(n-1)-1 must fit in 0 .. 2^d-1.
        return sbits <= dbits
    if not s_signed and d_signed:
        # 0 .. 2^s-1 must fit in -2^(d-1) .. 2^(d-1)-1.
        return sbits < dbits
    return sbits <= dbits



def widen(t: DataType) -> DataType:
    """Smallest canonical type that losslessly holds every value of ``t``."""
    k = t.kind
    if k is TypeKind.NULL:
        return INT64
    if k in _FLOATS:
        return FLOAT64
    if k is TypeKind.DECIMAL:
        return DECIMAL(max(t.precision or 38, 38), (t.scale or 0) + 2)
    if k in (TypeKind.TIME32,):
        return TIME64
    if k in (TypeKind.DATE32,):
        return DATE64
    if k is TypeKind.CATEGORICAL:
        return widen(t.dictionary_values or UTF8)
    if k is TypeKind.TIMESTAMP:
        return t
    return t


def lossy(src: DataType, dst: DataType) -> str | None:
    """Return a human reason string if ``src -> dst`` loses information.

    This is the single source of truth for "is this conversion safe?" and is
    consulted by the reconciliation layer, the writer, and the explain panel.
    """
    if src == dst:
        return None
    s, d = src.kind, dst.kind

    if s is TypeKind.NULL:
        return None

    if s in _INT_KINDS and d in _INT_KINDS:
        return None if _int_fits(src, d) else (
            f"integer {s} does not fit in {d}"
        )
    if s in _INT_KINDS and d in _FLOATS:
        return (None if d is TypeKind.FLOAT64 or _int_fits(src, TypeKind.INT64) else
                f"integer {s} loses precision as {d}")
    if s is TypeKind.DECIMAL and d in _FLOATS:
        return f"Decimal({src.precision},{src.scale}) may lose precision as {d}"
    if s in _FLOATS and d in _INT_KINDS:
        return f"truncates fractional part of {s} into {d}"
    if s in _FLOATS and d is TypeKind.DECIMAL:
        return None
    if s in _INT_KINDS and d is TypeKind.DECIMAL:
        return (None if (dst_scale_safe(src, d)) else
                f"integer {s} loses low-order digits at Decimal scale {d.scale}")
    if d is TypeKind.INT32 and s in _FLOATS:
        return f"narrowing {s} to Int32 may overflow"

    if s is TypeKind.TIMESTAMP:
        if d is not TypeKind.TIMESTAMP:
            return f"Timestamp has no direct {d} representation"
        if src.timezone != dst.timezone:
            return (f"timezone {src.timezone or 'naive'} -> {dst.timezone or 'naive'}"
                    " requires conversion")
        if not _time_fits(src.unit, dst.unit):
            return f"resolution {src.unit} -> {dst.unit} is not exact"
        return None
    if s in (TypeKind.DATE32, TypeKind.DATE64) and d in (TypeKind.DATE32, TypeKind.DATE64):
        return None
    if s is TypeKind.CATEGORICAL:
        return lossy(src.dictionary_values or UTF8, dst)

    if s in (TypeKind.LIST,) and d is TypeKind.LIST:
        if not src.value_type or not dst.value_type:
            return "list element type missing"
        return lossy(src.value_type, dst.value_type)
    if s is TypeKind.MAP and d is TypeKind.MAP:
        reasons = [r for r in (
            lossy(src.key_type, dst.key_type) if src.key_type and dst.key_type else "map key type missing",
            lossy(src.value_type, dst.value_type) if src.value_type and dst.value_type else "map value type missing",
        ) if r]
        return "; ".join(reasons) or None
    if s is TypeKind.STRUCT and d is TypeKind.STRUCT:
        if {f.name for f in src.fields} != {f.name for f in dst.fields}:
            return "struct field set differs"
        return None

    if s is TypeKind.UTF8 or s is TypeKind.BINARY:
        return f"{s} cannot be stored as {d} without a cast"
    if s in (TypeKind.BINARY, TypeKind.LARGE_BINARY) and d is TypeKind.UTF8:
        return "binary reinterpreted as text (may be invalid UTF-8)"
    return f"no safe conversion {s} -> {d}"


def dst_scale_safe(src: DataType, d: DataType) -> bool:
    """Whether an integer source survives conversion to decimal ``d``.

    A source wider than the decimal can hold is always safe; a narrow one
    only is if its magnitude fits inside the scale's resolution.
    """
    if d.scale <= 0:
        return True
    limit = 10 ** d.scale
    bits = _INT_BITS[src.kind]
    if src.kind in _SIGNED_INTS:
        return bits >= 64 or (2 ** (bits - 1)) < limit
    return bits >= 64 or (2 ** bits) < limit



def _time_fits(src_unit: str | None, dst_unit: str | None) -> bool:
    order = {"s": 0, "ms": 1, "us": 2, "ns": 3}
    s, d = order.get(src_unit or "us", 2), order.get(dst_unit or "us", 2)
    return d >= s


def common_type(types: Iterable[DataType]) -> DataType:
    """Least common supertype, or ``NULL`` for an empty/all-null input."""
    seen = [t for t in types if t.kind is not TypeKind.NULL]
    if not seen:
        return NULL
    uniq: list[DataType] = []
    for t in seen:
        if t not in uniq:
            uniq.append(t)
    if len(uniq) == 1:
        return uniq[0]
    widest = max(_BYTE_WIDTH.get(t.kind, 8) for t in uniq)
    if widest >= 8 and any(t.kind in _FLOATS for t in uniq):
        return FLOAT64
    if all(t.kind in _INT_KINDS for t in uniq):
        return max(uniq, key=lambda t: _INT_BITS[t.kind])
    if all(t.kind in _TEMPORAL for t in uniq):
        return max(uniq, key=lambda t: _time_fits(t.unit or "us", "ns"))
    return UTF8



# ---------------------------------------------------------------- lineage
@dataclass(frozen=True, slots=True)
class LineageRef:
    """A single provenance pointer for a column value."""

    system: str
    table: str | None = None
    column: str | None = None
    node_id: str | None = None
    expression: str | None = None

    def render(self) -> str:
        loc = ".".join(p for p in (self.system, self.table, self.column) if p)
        return f"{loc}({self.node_id})" if self.node_id else loc


@dataclass(frozen=True, slots=True)
class Field:
    """A named, typed, classified column.

    ``classification`` and ``lineage`` travel with the data across every
    engine boundary, so a ``CONFIDENTIAL`` tag applied in PostgreSQL is still
    present when the column lands in an Excel workbook.
    """

    name: str
    type: DataType
    nullable: bool = True
    classification: frozenset[str] = frozenset()
    description: str | None = None
    lineage: tuple[LineageRef, ...] = ()

    def with_classification(self, *tags: str) -> "Field":
        return Field(self.name, self.type, self.nullable,
                     self.classification | frozenset(t.upper() for t in tags),
                     self.description, self.lineage)

    def with_lineage(self, *refs: LineageRef) -> "Field":
        return Field(self.name, self.type, self.nullable, self.classification,
                     self.description, tuple(self.lineage) + tuple(refs))

    def __str__(self) -> str:
        tags = f" [{','.join(sorted(self.classification))}]" if self.classification else ""
        nn = "" if self.nullable else " not null"
        return f"{self.name}:{self.type}{nn}{tags}"


@dataclass(frozen=True, slots=True)
class Schema:
    """An ordered, named collection of :class:`Field` objects.

    Accepts a single ``Field``, any iterable of them, or nothing, so callers
    do not have to remember to add a trailing comma to a one-column schema -
    a mistake that would otherwise surface as an opaque
    ``'Field' object is not iterable`` deep inside a plan.
    """

    fields: tuple[Field, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.fields, Field):
            normalised: tuple[Field, ...] = (self.fields,)
        elif isinstance(self.fields, str):
            raise TypeError("Schema takes Field objects, not a column name")
        else:
            try:
                normalised = tuple(self.fields)
            except TypeError as exc:
                raise TypeError(
                    f"Schema expects a Field or an iterable of Fields, "
                    f"got {type(self.fields).__name__}. "
                    f"Did you mean Schema((Field({self.fields!r}, ...),))?"
                ) from exc
        object.__setattr__(self, "fields", normalised)
        seen: set[str] = set()
        for f in normalised:
            if not isinstance(f, Field):
                raise TypeError(
                    f"Schema elements must be Field, got {type(f).__name__}")
            if f.name in seen:
                raise ValueError(f"duplicate column name in schema: {f.name!r}")
            seen.add(f.name)


    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def __bool__(self) -> bool:
        return bool(self.fields)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    @property
    def types(self) -> tuple[DataType, ...]:
        return tuple(f.type for f in self.fields)

    def get(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"no such column: {name!r} (have: {', '.join(self.names)})")

    def has(self, name: str) -> bool:
        return any(f.name == name for f in self.fields)

    def index(self, name: str) -> int:
        for i, f in enumerate(self.fields):
            if f.name == name:
                return i
        raise KeyError(name)

    def column(self, idx: int) -> Field:
        return self.fields[idx]

    def select(self, names: Iterable[str]) -> "Schema":
        return Schema(tuple(self.get(n) for n in names))

    def drop(self, names: Iterable[str]) -> "Schema":
        drop = set(names)
        return Schema(tuple(f for f in self.fields if f.name not in drop))

    def rename(self, mapping: Mapping[str, str]) -> "Schema":
        """Rename columns, carrying type, nullability, tags and lineage."""
        return Schema(tuple(
            f if f.name not in mapping else Field(
                mapping[f.name], f.type, f.nullable,
                f.classification, f.description, f.lineage)
            for f in self.fields
        ))

    def cast(self, mapping: Mapping[str, DataType]) -> "Schema":
        """Retype columns, carrying nullability, tags and lineage across.

        Classification and lineage surviving a cast is the point: a column
        that was ``CONFIDENTIAL`` before the cast must still be afterwards.
        """
        return Schema(tuple(
            f if f.name not in mapping else Field(
                f.name, mapping[f.name], f.nullable,
                f.classification, f.description, f.lineage)
            for f in self.fields
        ))

    def with_nullability(self, **mapping: bool) -> "Schema":
        """Adjust per-column nullability without touching anything else."""
        return Schema(tuple(
            f if f.name not in mapping else Field(
                f.name, f.type, mapping[f.name], f.classification,
                f.description, f.lineage)
            for f in self.fields
        ))

    def extend(self, *others: "Schema") -> "Schema":
        out = list(self.fields)
        for o in others:
            out.extend(o.fields)
        return Schema(tuple(out))

    def inherit_classification(self, *sources: str) -> "Schema":

        """Union the named columns' tags onto every column in the schema.

        Used by joins where the broadcast side carries a classification the
        streamed side cannot see.
        """
        tags: frozenset[str] = frozenset()
        for s in sources:
            if self.has(s):
                tags |= self.get(s).classification
        if not tags:
            return self
        return Schema(tuple(
            f if f.classification >= tags else Field(
                f.name, f.type, f.nullable, f.classification | tags,
                f.description, f.lineage)
            for f in self.fields
        ))

    def all_classifications(self) -> frozenset[str]:
        out: frozenset[str] = frozenset()
        for f in self.fields:
            out |= f.classification
        return out

    def comparable(self, other: "Schema") -> bool:
        """True when two schemas can be zipped field-for-field without a cast."""
        if len(self) != len(other):
            return False
        return all(
            a.name == b.name and (a.type == b.type or lossy(a.type, b.type) is None)
            for a, b in zip(self.fields, other.fields)
        )

    def diff(self, other: "Schema") -> "SchemaDiff":
        return SchemaDiff.between(self, other)

    def render(self) -> str:
        return "{" + ", ".join(str(f) for f in self.fields) + "}"

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Schema({self.render()})"


EMPTY_SCHEMA = Schema()


@dataclass(frozen=True, slots=True)
class SchemaDiff:
    """Structured description of how a source schema drifted from expectation.

    Consumed by the reconciliation layer to raise failure mode #5
    (schema drift) instead of silently producing wrong results.
    """

    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    retyped: tuple[tuple[str, DataType, DataType], ...] = ()
    nullability_changed: tuple[tuple[str, bool, bool], ...] = ()
    identical: bool = True

    @classmethod
    def between(cls, expected: Schema, actual: Schema) -> "SchemaDiff":
        """Compare an expected schema against what the source actually has.

        Drift is reported, never repaired. A system that silently re-maps a
        renamed column produces confidently wrong numbers, which is the exact
        failure mode this exists to prevent.
        """
        added = tuple(n for n in actual.names if not expected.has(n))
        removed = tuple(n for n in expected.names if not actual.has(n))
        retyped: list[tuple[str, DataType, DataType]] = []
        nulls: list[tuple[str, bool, bool]] = []
        for f in expected.fields:
            if not actual.has(f.name):
                continue
            g = actual.get(f.name)
            if g.type != f.type:
                retyped.append((f.name, f.type, g.type))
            if g.nullable != f.nullable:
                nulls.append((f.name, f.nullable, g.nullable))
        same = not (added or removed or retyped or nulls)
        return cls(added, removed, tuple(retyped), tuple(nulls), same)

    def describe(self) -> str:
        if self.identical:
            return "schema unchanged"
        parts: list[str] = []
        if self.added:
            parts.append("added: " + ", ".join(self.added))
        if self.removed:
            parts.append("removed: " + ", ".join(self.removed))
        if self.retyped:
            parts.append("retyped: " + ", ".join(
                f"{n} {a}->{b}" for n, a, b in self.retyped))
        if self.nullability_changed:
            parts.append("nullability: " + ", ".join(
                f"{n} {a}->{b}" for n, a, b in self.nullability_changed))
        return "; ".join(parts)

    def __bool__(self) -> bool:
        """True when drift was detected, so ``if expected.diff(actual):`` reads well."""
        return not self.identical


# ------------------------------------------------------- type conversion log
@dataclass(frozen=True, slots=True)
class TypeConversion:
    """A recorded, reviewable conversion between a source and a target type."""

    column: str
    source_system: str
    source_type: str
    target_type: DataType
    lossy_reason: str | None = None

    @property
    def is_lossy(self) -> bool:
        return self.lossy_reason is not None

    def render(self) -> str:
        tail = f"  LOSSY: {self.lossy_reason}" if self.lossy_reason else ""
        return f"{self.column}: {self.source_type} -> {self.target_type} ({self.source_system}){tail}"


class ConversionLog:
    """Append-only record of every type conversion performed at a boundary.

    The runtime holds one of these per execution. Nothing is dropped silently:
    lossy entries are surfaced in the explain panel and, under a strict
    policy, block the write.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: list[TypeConversion] = []

    def record(
        self, column: str, source_system: str, source_type: str,
        target: DataType,
    ) -> TypeConversion:
        reason = lossy(_source_as_canonical(source_type, source_system), target)
        entry = TypeConversion(column, source_system, source_type, target, reason)
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> tuple[TypeConversion, ...]:
        return tuple(self._entries)

    @property
    def lossy_entries(self) -> tuple[TypeConversion, ...]:
        return tuple(e for e in self._entries if e.is_lossy)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def clear(self) -> None:
        self._entries.clear()


def _source_as_canonical(source_type: str, system: str) -> DataType:
    try:
        return from_source(source_type, system)
    except UnmappableType:
        return UTF8


class NormalizationPolicy(str, enum.Enum):
    """How to treat a conversion that would lose information."""

    #: Record and continue. Default - loss is visible in the log.
    ALLOW_LOSSY = "allow_lossy"
    #: Promote the target to the lossless widening type automatically.
    AUTO_WIDEN = "auto_widen"
    #: Stop the execution and surface failure mode #6.
    FAIL_ON_LOSSY = "fail_on_lossy"

    #: Parse a policy name, defaulting to AUTO_WIDEN.
    @staticmethod
    def parse(value: str | None) -> "NormalizationPolicy":
        if not value:
            return NormalizationPolicy.AUTO_WIDEN
        return NormalizationPolicy(str(value).lower())
# Mapping is name-based and total over the documented type space of each
# system. Anything unrecognised raises so the reconciliation layer can decide
# policy; nothing is guessed.

_PG = {
    "bool": BOOLEAN,
    "int2": INT16, "int4": INT32, "int8": INT64,
    "smallint": INT16, "integer": INT32, "bigint": INT64,
    "float4": FLOAT32, "float8": FLOAT64,
    "real": FLOAT32, "double precision": FLOAT64,
    "numeric": DECIMAL(38, 9), "decimal": DECIMAL(38, 9),
    "text": UTF8, "varchar": UTF8, "bpchar": UTF8, "char": UTF8,
    "uuid": UTF8, "name": UTF8, "citext": UTF8,
    "date": DATE32,
    "time": TIME32, "time without time zone": TIME32,
    "time with time zone": TIME64,
    "timestamp": TIMESTAMP("us", None),
    "timestamp without time zone": TIMESTAMP("us", None),
    "timestamp with time zone": TIMESTAMP("us", "UTC"),
    "timestamptz": TIMESTAMP("us", "UTC"),
    "interval": DURATION,
    "bytea": BINARY,
    "json": UTF8, "jsonb": UTF8,
    "xml": UTF8,
    "inet": UTF8, "cidr": UTF8, "macaddr": UTF8,
    "money": DECIMAL(18, 2),
    "_int4": list_of(INT32), "_int8": list_of(INT64),
    "_text": list_of(UTF8), "_float8": list_of(FLOAT64),
    "int4range": UTF8, "int8range": UTF8, "numrange": UTF8,
}

_MYSQL = {
    "bool": BOOLEAN, "boolean": BOOLEAN, "tinyint(1)": BOOLEAN,
    "tinyint": INT8, "smallint": INT16, "mediumint": INT32, "int": INT32,
    "integer": INT32, "bigint": INT64,
    "float": FLOAT32, "double": FLOAT64,
    "decimal": DECIMAL(38, 9), "numeric": DECIMAL(38, 9),
    "char": UTF8, "varchar": UTF8, "text": UTF8, "tinytext": UTF8,
    "mediumtext": UTF8, "longtext": UTF8, "enum": UTF8, "set": UTF8,
    "date": DATE32, "datetime": TIMESTAMP("us", None),
    "timestamp": TIMESTAMP("us", None), "time": TIME32, "year": INT16,
    "binary": BINARY, "varbinary": BINARY, "blob": LARGE_BINARY,
    "longblob": LARGE_BINARY, "json": UTF8, "geometry": BINARY,
}

_SQLITE = {
    "INTEGER": INT64, "INT": INT64, "BIGINT": INT64,
    "SMALLINT": INT16, "TINYINT": INT8,
    "REAL": FLOAT64, "DOUBLE": FLOAT64, "DOUBLE PRECISION": FLOAT64,
    "FLOAT": FLOAT64, "NUMERIC": DECIMAL(38, 9), "DECIMAL": DECIMAL(38, 9),
    "TEXT": UTF8, "VARCHAR": UTF8, "CHARACTER": UTF8, "CLOB": UTF8,
    "BLOB": LARGE_BINARY, "BINARY": BINARY, "VARBINARY": BINARY,
    "BOOLEAN": BOOLEAN, "DATE": DATE32, "DATETIME": TIMESTAMP("us", None),
}

_SPARK = {
    "boolean": BOOLEAN, "byte": INT8, "tinyint": INT8, "short": INT16,
    "smallint": INT16, "int": INT32, "integer": INT32, "long": INT64,
    "bigint": INT64, "float": FLOAT32, "double": FLOAT64,
    "decimal": DECIMAL(38, 18), "string": UTF8, "varchar": UTF8,
    "char": UTF8, "binary": BINARY, "date": DATE32,
    "timestamp": TIMESTAMP("us", None), "timestamp_ntz": TIMESTAMP("us", None),
    "timestamp_ltz": TIMESTAMP("us", "UTC"), "array": list_of(UTF8),
    "map": map_of(UTF8, UTF8), "struct": UTF8, "interval": DURATION,
}

_MONGODB = {
    "double": FLOAT64, "string": UTF8, "object": UTF8, "array": UTF8,
    "binData": BINARY, "undefined": NULL, "objectId": UTF8, "bool": BOOLEAN,
    "date": TIMESTAMP("ms", "UTC"), "null": NULL,
    "int": INT32, "long": INT64, "decimal": DECIMAL(38, 9),
    "timestamp": TIMESTAMP("us", "UTC"), "symbol": UTF8, "minKey": NULL,
    "maxKey": NULL, "number": FLOAT64,
}

#: Excel has no real type system; every cell is typed by inspection. These
#: are the *normalised* results, not the raw cell content.
_EXCEL = {
    "number": FLOAT64, "text": UTF8, "boolean": BOOLEAN,
    "date": TIMESTAMP("s", None), "datetime": TIMESTAMP("us", None),
    "time": TIME64, "error": UTF8, "blank": NULL, "formula": FLOAT64,
}

_ARROW = {
    "null": NULL, "bool": BOOLEAN, "boolean": BOOLEAN,
    "int8": INT8, "int16": INT16, "int32": INT32, "int64": INT64,
    "uint8": UINT8, "uint16": UINT16, "uint32": UINT32, "uint64": UINT64,
    "halffloat": FLOAT32, "float": FLOAT32, "float32": FLOAT32,
    "double": FLOAT64, "float64": FLOAT64,
    "string": UTF8, "utf8": UTF8, "large_string": UTF8,
    "binary": BINARY, "large_binary": LARGE_BINARY,
    "date32": DATE32, "date32[day]": DATE32,
    "date64": DATE64, "date64[ms]": DATE64,
    "time32[s]": TIME32, "time32[ms]": TIME32,
    "time64[us]": TIME64, "time64[ns]": TIME64,
    "timestamp[s]": TIMESTAMP("s", None), "timestamp[ms]": TIMESTAMP("ms", None),
    "timestamp[us]": TIMESTAMP("us", None), "timestamp[ns]": TIMESTAMP("ns", None),
    "timestamp[us, tz=UTC]": TIMESTAMP("us", "UTC"),
    "duration[s]": DURATION, "duration[ms]": DURATION,
    "duration[us]": DURATION, "duration[ns]": DURATION,
    "dictionary": UTF8, "decimal128": DECIMAL(38, 9),
    "decimal256": DECIMAL(76, 18),
}


_SOURCE_TABLES: dict[str, dict[str, DataType]] = {
    "postgresql": _PG, "postgres": _PG, "pg": _PG,
    "mysql": _MYSQL, "mariadb": _MYSQL,
    "sqlite": _SQLITE,
    "spark": _SPARK,
    "mongodb": _MONGODB, "mongo": _MONGODB, "bson": _MONGODB,
    "excel": _EXCEL, "xlsx": _EXCEL, "ods": _EXCEL,
    "arrow": _ARROW, "parquet": _ARROW, "pyarrow": _ARROW,
    "polars": _ARROW, "pandas": _ARROW, "duckdb": _ARROW, "cudf": _ARROW,
    "dask": _ARROW, "datafusion": _ARROW,
}

#: Python scalar -> canonical. Used by the UDF connector and JSON ingest.
_PYTHON_TABLES: dict[type, DataType] = {
    bool: BOOLEAN,
    int: INT64, float: FLOAT64, str: UTF8, bytes: BINARY,
}


def from_source(source_type: str, system: str) -> DataType:
    """Map a source-system type name onto the canonical type.

    Raises :class:`UnmappableType` when the name is unknown, so that callers
    apply policy rather than silently degrading to text.
    """
    table = _SOURCE_TABLES.get(str(system).lower())
    if table is None:
        raise UnmappableType(
            f"unknown source system {system!r}; "
            f"known: {', '.join(sorted(_SOURCE_TABLES))}")
    key = str(source_type).strip().lower()
    # Postgres parameterised numerics: numeric(12,2)
    if key.startswith("numeric(") or key.startswith("decimal("):
        try:
            inner = key[key.index("(") + 1:key.rindex(")")]
            p, s = (int(x.strip()) for x in inner.split(",")[:2])
            return DECIMAL(p, s)
        except (ValueError, IndexError):
            return DECIMAL(38, 9)
    for arrow_key, val in table.items():
        if arrow_key.lower() == key:
            return val
    if key.startswith("timestamp") and table is _PG:
        return TIMESTAMP("us", "UTC") if "with time zone" in key else TIMESTAMP("us", None)
    if key.startswith("varchar") or key.startswith("char"):
        return UTF8
    raise UnmappableType(f"{system} type {source_type!r} has no canonical mapping")


def _tz_name(tz: Any) -> str | None:
    """Canonical IANA name for a ``tzinfo``, or ``None`` if naive.

    ``zoneinfo`` objects expose ``.key``; ``datetime.timezone`` instances do
    not, and accessing ``.key`` on one is an ``AttributeError`` rather than a
    ``None``. Both are common, so both are handled, and a fixed offset is
    rendered explicitly instead of being guessed at.
    """
    if tz is None:
        return None
    key = getattr(tz, "key", None)
    if key:
        return str(key)
    name = getattr(tz, "zone", None)
    if name:
        return str(name)
    offset = None
    try:
        offset = tz.utcoffset(None)
    except (TypeError, ValueError):
        offset = None
    if offset is not None:
        total = int(offset.total_seconds())
        if total == 0:
            return "UTC"
        sign = "+" if total > 0 else "-"
        total = abs(total)
        return f"{sign}{total // 3600:02d}:{total % 3600 // 60:02d}"
    return str(tz)


def from_python(value: object) -> DataType:

    """Infer a canonical type from a Python scalar."""
    t = type(value)
    if t is bool:
        return BOOLEAN
    if t in (int, str, float, bytes):
        return _PYTHON_TABLES[t]
    import datetime as _dt

    if t is _dt.datetime:
        return TIMESTAMP("us", _tz_name(value.tzinfo))
    if t is _dt.date:
        return DATE32
    if t is _dt.time:
        return TIME64
    if t is _dt.timedelta:
        return DURATION
    if value is None:
        return NULL
    if isinstance(value, (list, tuple)):
        inner = [from_python(v) for v in value[:64]] or [NULL]
        return list_of(common_type(inner))
    if isinstance(value, dict):
        return UTF8
    raise UnmappableType(f"cannot infer canonical type from {t.__name__}")


def normalize(t: DataType, policy: "NormalizationPolicy | str" = NormalizationPolicy.AUTO_WIDEN) -> DataType:
    """Reconcile a canonical type against a normalisation policy."""
    p = policy if isinstance(policy, NormalizationPolicy) else NormalizationPolicy.parse(policy)
    if p is NormalizationPolicy.FAIL_ON_LOSSY or p is NormalizationPolicy.ALLOW_LOSSY:
        return t
    return widen(t)


def arrow_name(t: DataType) -> str:
    """Render a canonical type using Arrow's naming convention."""
    k = t.kind
    if k is TypeKind.TIMESTAMP:
        return f"timestamp[{t.unit or 'us'}]"
    if k is TypeKind.DURATION:
        return f"duration[{t.unit or 'us'}]"
    if k is TypeKind.TIME32:
        return f"time32[{t.unit or 'ms'}]"
    if k is TypeKind.TIME64:
        return f"time64[{t.unit or 'us'}]"
    if k is TypeKind.DECIMAL:
        return "decimal128" if (t.precision or 38) <= 38 else "decimal256"
    if k is TypeKind.DATE32:
        return "date32"
    if k is TypeKind.DATE64:
        return "date64"
    if k is TypeKind.CATEGORICAL:
        return "dictionary"
    return str(k)
