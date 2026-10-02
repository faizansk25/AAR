"""Stable semantic identity for IR operations and graph nodes.

Two different questions need two different identifiers, and conflating them is
why measurement history never worked.

``Node.id``
    Transient *instance* identity: a UUID, unique within one constructed DAG.
    Correct for tracing one run. Useless for learning, because two parses of
    the same pipeline produce different UUIDs and so can never share evidence.

``operation_id``
    What the operation *means*. ``filter(amount > 100)`` is the same operation
    in every pipeline on every machine, so it can share measurements.

``graph_node_id``
    What the operation means *at this position in this pipeline*. Two nodes can
    have the same operation and still be different things:

        CSV A -> filter(amount > 100)
        CSV B -> filter(amount > 100)

    Same operation, different provenance chains, different costs. Sharing
    history between them would be wrong; sharing none between two parses of one
    pipeline would be worse. So ``graph_node_id = operation_id + ordered
    parent graph_node_ids``, where position is meaningful (``JOIN`` input 0 is
    the left side) and never normalised away.

Design rules, in priority order:

1. **An ID that cannot be computed must refuse to exist.** Anything not
   stably encodable raises :class:`UnstableSemanticIdentity` rather than
   silently hashing a memory address or a ``repr``. A silently unstable ID is
   worse than no ID: it splits history in two without saying so.
2. **Explicit whitelist, never generic serialisation.** ``repr(node)``,
   ``dataclasses.asdict(node)`` and ``pickle`` all break the moment an unrelated
   field is added, and the IR deliberately mixes logical payload with physical
   planning decisions in one dataclass. Every field here is named on purpose.
3. **No mathematical normalisation in v1.** ``a AND b`` is not reordered to
   ``b AND a``, join inputs are not swapped, unions are not sorted. Null
   semantics, evaluation order and engine pushdown all make those rewrites
   unsafe, and a deduplication win is not worth a wrong answer.
4. **Versioned from day one.** The prefix is part of the hash, so a change to
   what goes into a payload produces new IDs rather than colliding with old
   ones in a store that has no way to tell them apart.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from datetime import date, datetime, time as _time
from decimal import Decimal
from typing import Any

__all__ = [
    "SEMANTIC_ID_VERSION",
    "UnstableSemanticIdentity",
    "operation_payload",
    "graph_node_payload",
    "semantic_operation_id",
    "graph_node_id",
    "target_id",
    "resource_snapshot",
]

#: Bump when the *meaning* of any payload changes. Included in every hash, so
#: identifiers from different versions can never be confused in one store.
SEMANTIC_ID_VERSION = 1

#: Domain separators. Without these, an operation payload and a graph-node
#: payload that happened to serialise identically would collide, and a
#: measurement recorded under one would silently answer for the other.
_DOMAIN_OPERATION = "aar:semantic-operation:v1"
_DOMAIN_GRAPH_NODE = "aar:semantic-graph-node:v1"
_DOMAIN_TARGET = "aar:target:v1"

#: The readable half of each identifier's prefix, keyed by domain.
_PREFIX = {"aar:semantic-operation:v1": "aar-op-v1",
           "aar:semantic-graph-node:v1": "aar-node-v1",
           "aar:target:v1": "aar-target-v1"}


class UnstableSemanticIdentity(Exception):
    """A node cannot be given a stable identity, so it is refused one.

    Raised rather than falling back to something plausible. The whole value of
    a semantic ID is that it means the same thing tomorrow; an ID derived from
    a ``repr`` containing a memory address satisfies none of that while
    appearing to work.
    """


# ------------------------------------------------------------- canonical JSON
def _canonical(payload: Any) -> bytes:
    """Serialise a payload to bytes that mean exactly one thing.

    Every flag here is load-bearing. ``sort_keys`` and the compact separators
    remove formatting differences. ``ensure_ascii=False`` keeps non-ASCII text
    as UTF-8 rather than escaping it. ``allow_nan=False`` is the important one:
    JSON has no representation for NaN or Infinity, so Python would otherwise
    emit bare ``NaN`` tokens that are not valid JSON and hash differently
    depending on the writer.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(domain: str, payload: Any) -> str:
    """Hash a payload under a domain separator, with a readable prefix.

    The version prefix is inside the digest *and* outside it, so an identifier
    is self-describing when read in a log or a database row: seeing
    ``aar-op-v1:`` tells a reader which rules produced it without consulting
    the schema.
    """
    blob = _canonical(payload)
    inner = hashlib.sha256(
        domain.encode("ascii") + b"\0" + blob).hexdigest()
    return f"{_PREFIX[domain]}:{inner}"


# ------------------------------------------------------------ literal encoding
def _literal(value: Any) -> dict[str, Any]:
    """Encode a literal value with its type, or refuse.

    Every branch names a type explicitly, because ``100`` and ``100.0`` and
    ``Decimal("100")`` are different literals that render identically through
    ``str``. Floats are special-cased for NaN and the signed zeros, which are
    equal under ``==`` but not under ``repr`` - so ``-0.0`` and ``0.0`` must
    not collapse into one ID.
    """
    if value is None:
        return {"t": "null"}
    # bool before int: bool is an int subclass, and True/1 must not collide.
    if isinstance(value, bool):
        return {"t": "bool", "v": value}
    if isinstance(value, int):
        return {"t": "int", "v": str(value)}
    if isinstance(value, float):
        if math.isnan(value):
            return {"t": "float", "v": "nan"}
        if math.isinf(value):
            return {"t": "float", "v": "inf" if value > 0 else "-inf"}
        if value == 0.0:
            # Distinguish -0.0 from 0.0: equal in Python, distinct in repr.
            return {"t": "float",
                    "v": "-0.0" if math.copysign(1, value) < 0 else "0.0"}
        return {"t": "float", "v": repr(value)}
    if isinstance(value, Decimal):
        if value.is_nan():
            return {"t": "decimal", "v": "nan"}
        if value.is_infinite():
            return {"t": "decimal", "v": "inf" if value > 0 else "-inf"}
        return {"t": "decimal", "v": str(value)}
    if isinstance(value, str):
        return {"t": "str", "v": value}
    if isinstance(value, (bytes, bytearray)):
        return {"t": "bytes", "v": bytes(value).hex()}
    if isinstance(value, datetime):
        return {"t": "datetime", "v": value.isoformat()}
    if isinstance(value, date):
        return {"t": "date", "v": value.isoformat()}
    if isinstance(value, _time):
        return {"t": "time", "v": value.isoformat()}
    if isinstance(value, (list, tuple)):
        # Order is preserved: a list literal is ordered data, not a set.
        return {"t": "list", "v": [_literal(v) for v in value]}
    if isinstance(value, (set, frozenset)):
        # Sets are unordered, so they are sorted by their encoded form. This is
        # a genuine normalisation and it is safe: two equal sets always encode
        # to the same sorted list of the same elements.
        return {"t": "set",
                "v": sorted((_canonical(_literal(v)).decode("utf-8")
                             for v in value))}
    if isinstance(value, dict):
        items = sorted((str(k), _canonical(_literal(v)).decode("utf-8"))
                       for k, v in value.items())
        return {"t": "dict", "v": [[k, v] for k, v in items]}
    raise UnstableSemanticIdentity(
        f"cannot encode a literal of type {type(value).__name__!r} into a "
        f"stable identity. Add an explicit case for it; do not fall back to "
        f"repr(), which is not stable across processes or versions.")


# ------------------------------------------------------- expression encoding
def _expr(e: Any) -> dict[str, Any]:
    """Encode an IR expression structurally.

    Never ``to_sql()``. Rendered SQL is the *output* of a dialect-specific
    compiler: quoting differs per engine, and a rendering change would silently
    invalidate every stored measurement without any operation having changed.
    The IR's expression algebra is closed, so encoding it directly is both
    possible and stable.
    """
    from .nodes import Agg, BinOp, CastExpr, Col, Func, Lit, UnaryOp

    if isinstance(e, Col):
        return {"kind": "column", "name": e.name}
    if isinstance(e, Lit):
        payload: dict[str, Any] = {"kind": "literal",
                                   "value": _literal(e.value)}
        if e.dtype is not None:
            payload["dtype"] = str(getattr(e.dtype, "kind", e.dtype))
        return payload
    if isinstance(e, BinOp):
        return {"kind": "binop", "op": e.op,
                "left": _expr(e.left), "right": _expr(e.right)}
    if isinstance(e, UnaryOp):
        return {"kind": "unary", "op": e.op, "operand": _expr(e.operand)}
    if isinstance(e, Func):
        return {"kind": "func", "name": e.name,
                "args": [_expr(a) for a in e.args],
                "kwargs": sorted([[str(k), _literal(v)]
                                  for k, v in e.kwargs])}
    if isinstance(e, Agg):
        return {"kind": "agg", "func": e.func,
                "arg": _expr(e.arg) if e.arg is not None else None,
                "distinct": bool(e.distinct), "custom": e.custom}
    if isinstance(e, CastExpr):
        return {"kind": "cast", "to": str(getattr(e.to, "kind", e.to)),
                "arg": _expr(e.arg)}
    raise UnstableSemanticIdentity(
        f"cannot encode expression of type {type(e).__name__!r}. Add an "
        f"explicit case; rendering to SQL is not an identity because it "
        f"depends on the dialect.")


# --------------------------------------------------------------- schema shape
def _schema(schema: Any) -> list[list[Any]]:
    """Encode a schema as ordered ``[name, type, privacy]`` triples.

    A schema change is a semantic change - the same bytes mean something
    different once a column is int64 rather than utf8 - so it is part of the
    identity. Names are included because a planner reading different columns is
    doing different work.
    """
    fields = getattr(schema, "fields", None)
    if not fields:
        return []
    out: list[list[Any]] = []
    for f in fields:
        kind = getattr(getattr(f, "type", None), "kind", None)
        out.append([str(getattr(f, "name", "")), str(kind),
                    str(getattr(f, "privacy", ""))])
    return out


# --------------------------------------------------------------------- UDFs
def _normalise_source(source: str) -> str:
    """Canonical form of a UDF's body: indentation and comments removed.

    Exposed for testing because the property is not otherwise reachable from
    outside: within one process you cannot have two textually different
    versions of the *same* function, so the only honest way to ask "does a
    reformat change the digest?" is to normalise two strings directly.

    Comments go because they cannot change behaviour. The ``def`` line stays
    because parameter names and the function name can.
    """
    body = [ln.strip() for ln in source.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    return "\n".join(body)


def _udf(node: Any) -> dict[str, Any]:
    """Identify a Python UDF by what it *does*, not by what it is called.

    ``udf_name`` alone is worthless here. Two different functions routinely
    share a name across pipeline versions::

        def clean(x): return x + 1     # v1
        def clean(x): return x * 10    # v2

    Same name, same column, different work and different cost. The digest is
    over the source text, plus defaults.

    A callable whose source cannot be recovered - a C extension, a builtin, an
    object with only ``__call__`` - raises. That is the intended behaviour: an
    ID that changes whenever the module is reloaded is not an identity.
    """
    explicit = getattr(node, "semantic_version", None)
    payload: dict[str, Any] = {
        "udf_name": node.udf_name,
        "udf_mode": node.udf_mode,
    }
    if explicit:
        # The documented escape hatch: the author has asserted that this
        # version string captures everything that matters.
        payload["explicit_version"] = str(explicit)
        return payload

    fn = node.udf
    if fn is None:
        payload["udf_kind"] = "unresolved"
        return payload

    if not callable(fn):
        payload["udf_kind"] = "non-callable"
        payload["udf_value"] = _literal(fn)
        return payload

    target = fn
    if not (inspect.isfunction(target) or inspect.ismethod(target)):
        # functools.partial, a lambda held in an object, a decorated callable.
        inner = getattr(target, "__wrapped__", None)
        if inner is not None:
            target = inner
        elif inspect.isbuiltin(target):
            # A builtin has a name and no source. It has *some* stable
            # identity, so it is recorded as such rather than refused.
            payload["udf_kind"] = "builtin"
            payload["udf_module"] = str(getattr(target, "__module__", "") or "")
            payload["udf_qualname"] = str(getattr(target, "__qualname__", "")
                                          or getattr(target, "__name__", ""))
            return payload
        else:
            # Anything else that is callable - an instance with __call__, a
            # class used as a factory, a proxy object - has behaviour that
            # cannot be recovered here. Refused rather than guessed at.
            raise UnstableSemanticIdentity(
                f"UDF {node.udf_name!r} is a callable object of type "
                f"{type(target).__name__!r} with no recoverable source. Give "
                f"the node an explicit semantic_version=... so its identity is "
                f"declared rather than inferred.")

    try:
        source = inspect.getsource(target)
    except (OSError, TypeError) as exc:
        raise UnstableSemanticIdentity(
            f"cannot read the source of UDF {node.udf_name!r} "
            f"({getattr(target, '__qualname__', target)!r}). A UDF defined in "
            f"a REPL or a C extension has no stable identity; pass "
            f"semantic_version=... instead.") from exc

    # Comments and blank lines are dropped and indentation normalised, so a
    # reformat does not invalidate stored measurements while any change to the
    # body does. The def line is kept: it carries parameter names, and those
    # do change behaviour when they change.
    code = _normalise_source(source)

    payload["udf_kind"] = "source"
    payload["udf_module"] = str(getattr(target, "__module__", "") or "")
    payload["udf_qualname"] = str(getattr(target, "__qualname__", "") or "")
    payload["udf_code_sha256"] = hashlib.sha256(
        code.encode("utf-8")).hexdigest()
    try:
        sig = inspect.signature(target)
        payload["udf_defaults"] = [
            _literal(p.default) for p in sig.parameters.values()
            if p.default is not inspect.Parameter.empty]
    except (TypeError, ValueError):
        payload["udf_defaults"] = None
    return payload


# ------------------------------------------------- per-node semantic payload
def _scan_payload(scan: Any) -> dict[str, Any]:
    """Identify a scan by the request it makes of the source.

    ``cache_key`` is deliberately *not* reused: it mixes the file's mtime and
    size into its identity, which is correct for cache invalidation and wrong
    for semantic identity. A file whose bytes are rewritten in place is a
    different request, but it is still the same operation.
    """
    payload: dict[str, Any] = {"kind": scan.kind}
    for name in ("path", "sheet", "table", "named_range", "cell_range",
                 "header_row", "formula_handling", "include_hidden",
                 "connection", "dsn", "table_name", "query", "collection",
                 "database", "delimiter", "encoding", "streaming"):
        value = getattr(scan, name, None)
        if value is not None:
            payload[name] = _literal(value)
    payload["columns"] = list(scan.columns)
    payload["projection"] = list(scan.projection)
    payload["row_filter"] = (_expr(scan.row_filter)
                             if scan.row_filter is not None else None)
    if scan.options:
        # Options are free-form caller input, so they are encoded rather than
        # trusted - but they *are* part of what the source is asked to do.
        payload["options"] = _literal(dict(scan.options))
    return payload


def _semantic_body(node: Any) -> dict[str, Any]:
    """The logical payload of one node, as a whitelist.

    Everything excluded here describes *a planning attempt*, not an operation:
    ``id``, ``assigned_engine``, ``reason``, ``segment_id``, ``estimated_ms``,
    ``estimated_peak_memory``, ``expected_saving_ms``, ``fallback_engine``,
    ``pushed``, ``estimated_rows``, ``estimated_bytes``,
    ``estimated_selectivity``, ``aar_profile``. AAR's IR deliberately keeps
    logical and physical state in one dataclass, which makes an explicit list
    the only safe way to separate them - a generic ``asdict`` would sweep the
    planner's own output into the operation's identity and change every ID the
    moment a plan was assigned.
    """
    from .nodes import NodeType

    t = node.type
    body: dict[str, Any] = {"type": t.value}

    if node.scan is not None:
        body["scan"] = _scan_payload(node.scan)

    if node.predicate is not None:
        body["predicate"] = _expr(node.predicate)
    if getattr(node, "security_rule", ""):
        # A barrier's *provenance* is part of its identity. Two runs of the
        # same query under different row-level policies produce different
        # results from identical IR text, so an identity that ignored the rule
        # would let a cached result computed under a permissive policy satisfy
        # a request made under a restrictive one.
        body["security_rule"] = node.security_rule
        body["source_scope"] = getattr(node, "source_scope", "")
    if node.columns:
        body["columns"] = list(node.columns)
    if node.expressions:
        # Sorted by name: this is a mapping of named projections, and a dict
        # has no meaningful order of its own. The *expressions* keep their own
        # internal order, which is where sequence semantics actually live.
        body["expressions"] = [[k, _expr(node.expressions[k])]
                               for k in sorted(node.expressions)]
    if node.key_left:
        body["key_left"] = list(node.key_left)
    if node.key_right:
        body["key_right"] = list(node.key_right)
    if t is NodeType.JOIN:
        body["join_type"] = str(node.join_type)
    if node.agg_functions:
        body["agg_functions"] = [
            [name, [_expr(a) for a in node.agg_functions[name]]]
            for name in sorted(node.agg_functions)]
    if node.sort_keys:
        # (column, ascending) pairs, order-significant: sorting by (a, b) is
        # not the same query as sorting by (b, a).
        body["sort_keys"] = [[c, bool(asc)] for c, asc in node.sort_keys]
    if node.window is not None:
        w = node.window
        body["window"] = {
            "partition_by": list(w.partition_by),
            "order_by": [[c, bool(asc)] for c, asc in w.order_by],
            "frame": w.frame,
            "kind": w.kind,
        }
    if node.window_functions:
        body["window_functions"] = [[k, _expr(node.window_functions[k])]
                                    for k in sorted(node.window_functions)]
    if node.dedup_keys:
        body["dedup_keys"] = list(node.dedup_keys)
        body["dedup_strategy"] = node.dedup_strategy
    if node.tag_values:
        body["tag_values"] = list(node.tag_values)
    if node.casts:
        body["casts"] = sorted([[k, str(getattr(v, "kind", v))]
                                for k, v in node.casts.items()])
    if t is NodeType.NULL_HANDLE:
        body["null_strategy"] = node.null_strategy
        body["fill_value"] = _literal(node.fill_value)
    if t is NodeType.PYTHON_UDF:
        body.update(_udf(node))
    if node.limit is not None:
        body["limit"] = int(node.limit)
    if t is NodeType.WRITE:
        # The destination is part of the operation: writing to a different
        # target is different work with a different cost.
        body["target"] = node.target
        body["write_format"] = node.write_format
        body["write_mode"] = node.write_mode
    if node.quality_rules:
        body["quality_rules"] = sorted([list(r) for r in node.quality_rules])

    # Schemas, where known. A node whose input schema is unknown and one whose
    # input schema is known-but-empty are genuinely different situations, so
    # the distinction is recorded rather than collapsed.
    body["input_schemas"] = [_schema(s) for s in node.input_schemas]
    body["output_schema"] = _schema(node.output_schema)
    return body


def operation_payload(node: Any) -> dict[str, Any]:
    """The canonical payload an ``operation_id`` is computed from.

    Exposed so a caller can inspect *why* two nodes differ instead of only
    seeing that they do - which is the difference between a debuggable system
    and an opaque one.
    """
    return {
        "v": SEMANTIC_ID_VERSION,
        "op": _semantic_body(node),
    }


def semantic_operation_id(node: Any) -> str:
    """The stable identity of what this node *means*.

    Independent of position, engine, estimates and of the random UUID in
    ``Node.id``.
    """
    return _digest(_DOMAIN_OPERATION, operation_payload(node))


def graph_node_payload(node: Any, parent_ids: tuple[str, ...]) -> dict[str, Any]:
    """A graph-node payload: the operation plus its ordered ancestry.

    ``parent_ids`` must be in *declared input order*, not sorted. Input
    position is semantics for a join (``inputs[0]`` is the left side) and for a
    union, so sorting them would merge genuinely different nodes.
    """
    return {
        "v": SEMANTIC_ID_VERSION,
        "operation": semantic_operation_id(node),
        # Position included explicitly, so the digest distinguishes
        # "left = A, right = B" from "left = B, right = A" even if a future
        # change ever made their operation IDs equal.
        "parents": [[i, pid] for i, pid in enumerate(parent_ids)],
    }


def graph_node_id(node: Any, parent_ids: tuple[str, ...] | None = None) -> str:
    """The stable identity of this node *in its pipeline*.

    Defaults to the node's own inputs, computed in declared order. Pass
    ``parent_ids`` explicitly when the caller already has them, which is the
    normal case in a walk and avoids recomputing every ancestor.
    """
    if parent_ids is None:
        parent_ids = tuple(graph_node_id(i) for i in node.inputs)
    return _digest(_DOMAIN_GRAPH_NODE, graph_node_payload(node, parent_ids))


# ------------------------------------------------------------ target identity
#: Fields of a hardware profile that describe the *machine*, as opposed to
#: what it is doing right now. Free/available RAM and free VRAM are excluded on
#: purpose: they change minute to minute, so a target keyed on them would be a
#: different target every time the plan ran, which is precisely the confusion
#: that separates ``target_id`` from ``resource_snapshot``.
_TARGET_FIELDS = (
    ("os", "system"), ("os", "release"), ("os", "architecture"),
    ("cpu", "model"), ("cpu", "physical_cores"), ("cpu", "logical_cores"),
    ("cpu", "simd_level"), ("cpu", "max_freq_mhz"),
    ("memory", "total_bytes"),
    ("gpu", "vendor"), ("gpu", "model"), ("gpu", "vram_bytes"),
    ("gpu", "compute_capability"), ("gpu", "driver_version"),
    ("software", "python"), ("software", "arrow"), ("software", "duckdb"),
    ("software", "polars"), ("software", "cudf"),
)


def _dig(profile: Any, section: str, attr: str) -> Any:
    obj = getattr(profile, section, None)
    if obj is None:
        return None
    return getattr(obj, attr, None)


def target_id(profile: Any) -> str:
    """Stable identity of the *machine* a measurement belongs to.

    Deliberately separate from ``HardwareProfile.fingerprint()``, which is not
    changed here because it already keys stored calibration files. This is a
    new, versioned identity for measurement records, and it answers a different
    question: not "is this the same calibration?" but "would these two runs
    have executed on comparable hardware?".

    The distinction from :func:`resource_snapshot` is not academic. Since the
    cost model takes an explicit ``ResourceBudget``, two plans for the same
    machine can legitimately differ if one had 2 GB of free VRAM and the other
    20 GB - and answering "why did yesterday choose GPU and today chose CPU?"
    requires seeing both the identical ``target_id`` and the differing
    snapshot.
    """
    payload: dict[str, Any] = {"v": SEMANTIC_ID_VERSION}
    for section, attr in _TARGET_FIELDS:
        value = _dig(profile, section, attr)
        if value is not None:
            payload[f"{section}.{attr}"] = _literal(value)
    return _digest(_DOMAIN_TARGET, payload)


def resource_snapshot(budget: Any) -> dict[str, Any]:
    """The exact budgets a plan was priced against.

    Recorded *separately* from ``target_id`` and never folded into it, because
    these numbers change continuously while the machine does not. Folding them
    in would make every measurement its own target, which is the same defect
    as keying on ``Node.id``.

    Returned as a plain dict rather than a digest: this is a record of state to
    be stored and compared, not an identifier. Two snapshots that differ are
    the finding, so collapsing them to a hash would discard the very
    information that explains the difference.
    """
    return {
        "v": SEMANTIC_ID_VERSION,
        "ram_budget_bytes": int(getattr(budget, "ram_bytes", 0) or 0),
        "vram_budget_bytes": int(getattr(budget, "vram_bytes", 0) or 0),
    }
