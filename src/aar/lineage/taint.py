"""Lineage: how a classification travels from a source column to a result.

The gap this closes is specific and it is a privacy hole. A ``CONFIDENTIAL``
tag already survives every engine boundary, so a projected or cast column
keeps it. But an operation that *creates* a column - an aggregate, a UDF, a
join - has nothing to carry the tag forward, so ``SUM(salary)`` arrived
unlabelled and a policy trusting classification missed it entirely.

The rule is one sentence: **a derived column is at least as sensitive as
everything it was derived from.** The interesting part is the edges, and each
is a case where being wrong leaks data:

* An aggregate inherits its *argument's* tags. ``SUM(salary)`` is as
  sensitive as ``salary``.
* ``COUNT(*)`` inherits nothing. It reports how many rows there are, which is
  a property of the table, not of any column's value.
* A group key inherits its own tags, because a group key *is* the value.
  Bucketing customers by a quasi-identifier is a disclosure, not an
  anonymisation.
* A UDF inherits the union of every column it could see. A function is
  opaque, so the only sound assumption is that it read everything.
* A join inherits from both sides, since either can contribute.

There is one escape hatch, :func:`declassify`, and it requires a written
justification. An aggregate that provably cannot disclose a single value
still needs a way out, or analysts will simply strip the tags off the source.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..types import Field, Schema, Sensitivity, sensitivity_of

__all__ = [
    "LineageEvent", "derive_from", "inherit_all", "merge_schemas",
    "declassify", "is_derived_from", "describe", "aggregate_tags",
    "AGGREGATE_RULE", "COUNT_STAR_RULE", "UDF_RULE", "JOIN_RULE",
]


#: Rule names, recorded on every derivation so an audit can say *why* a
#: column is sensitive rather than only that it is.
AGGREGATE_RULE = "aggregate.inherits_argument"
COUNT_STAR_RULE = "aggregate.count_star_derives_nothing"
UDF_RULE = "udf.inherits_whole_row"
JOIN_RULE = "join.inherits_both_sides"


class LineageEvent:
    """One recorded derivation, for the audit trail.

    A plain class rather than a dataclass so the events can be compared in
    tests by content, which is the only way anyone reads an audit trail.
    """

    __slots__ = ("output", "sources", "rule", "justification")

    def __init__(self, output: str, sources: Sequence[str], rule: str,
                 justification: str = "") -> None:
        self.output = output
        self.sources = tuple(sources)
        self.rule = rule
        self.justification = justification

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LineageEvent):
            return NotImplemented
        return (self.output, self.sources, self.rule,
                self.justification) == \
            (other.output, other.sources, other.rule, other.justification)

    def __repr__(self) -> str:
        parts = [f"{self.output} <- {', '.join(self.sources) or '(nothing)'}",
                 self.rule]
        if self.justification:
            parts.append(f"justified: {self.justification}")
        return " | ".join(parts)



def derive_from(schema: Any, sources: Iterable[str]) -> frozenset[str]:
    """The union of the classifications of ``sources``.

    A source name that is not in the schema contributes nothing rather than
    raising. That is deliberate: an aggregate over a column an earlier
    projection already dropped is a plan bug, and failing here would replace
    a clear error at plan time with a confusing one at run time. The taint
    rules stay sound because an absent column cannot carry a value.
    """
    names = set(sources)
    if schema is None:
        return frozenset()
    tags: set[str] = set()
    for field in _fields(schema):
        if field.name in names:
            tags |= set(field.classification)
    return frozenset(tags)


def inherit_all(schema: Any) -> frozenset[str]:
    """Every tag anywhere in a schema.

    The right answer for an opaque operation - a UDF, a raw SQL passthrough -
    where the engine genuinely cannot know which columns were read.
    """
    if schema is None:
        return frozenset()
    tags: set[str] = set()
    for field in _fields(schema):
        tags |= set(field.classification)
    return frozenset(tags)


def merge_schemas(left: Any, right: Any) -> frozenset[str]:
    """Everything from both sides of a join or a union."""
    return inherit_all(left) | inherit_all(right)


def for_aggregate(schema: Any, agg: Any) -> frozenset[str]:
    """What an aggregate inherits, including the COUNT(*) exception.

    COUNT with no argument is the one aggregate that derives from nothing: it
    reports how many rows exist, which is a property of the table rather than
    of the value of any column. Masking it would be theatre. Every other
    aggregate inherits the tags of its argument, resolved by name, so an
    expression over several columns takes the union.
    """
    from ..ir import Agg as AggExpr
    from ..ir import Col

    if not isinstance(agg, AggExpr):
        return frozenset()
    if agg.arg is None:
        return frozenset()
    if isinstance(agg.arg, Col):
        return derive_from(schema, (agg.arg.name,))
    return inherit_all(schema)


def aggregate_tags(schema: Any,
                   aggs: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    """The tags every aggregate output column inherits, by output name.

    Without this, ``SUM(salary)`` is born unlabelled and a policy that
    trusts classification has nothing to act on - a derived column leaking
    straight past a privacy layer that looks like it is working.

    It lives here rather than in an engine because every engine needs the
    same answer, and a per-engine copy would eventually be a second and
    different answer.
    """
    return {name: for_aggregate(schema, agg) for name, agg in aggs.items()}



def declassify(field: Field, justification: str) -> Field:
    """Remove classification from a derived column, with a written reason.

    A justified declassification is legitimate - ``COUNT(*)`` over a salary
    table reveals no individual salary - and a system with no escape hatch
    pushes analysts toward deleting the source tags instead, which is
    strictly worse. Doing it quietly is the unacceptable part, so the
    justification is mandatory and is stored on the field.
    """
    if not field.classification:
        return field
    if not justification or not justification.strip():
        raise ValueError(
            f"column {field.name!r} is classified "
            f"({', '.join(sorted(field.classification))}); declassifying it "
            f"requires a written justification, because the result of a "
            f"classified computation is classified by default")
    note = f"declassified: {justification.strip()}"
    description = f"{field.description}; {note}" if field.description else note
    return Field(field.name, field.type, field.nullable, frozenset(),
                 description, field.lineage)


def is_derived_from(field: Field, tag: str) -> bool:
    """Whether a column is tainted by a given classification tag."""
    return str(tag).upper() in field.classification


def describe(schema: Any) -> str:
    """A human-readable account of what is sensitive in a schema."""
    fields = _fields(schema)
    if not fields:
        return "No columns."
    width = max(len(f.name) for f in fields)
    rows = [f"  {f.name:<{width}}  "
            f"{', '.join(sorted(f.classification)) or '-':<26}"
            f"  {sensitivity_of(f.classification).name}" for f in fields]
    return "\n".join(rows)


def _fields(schema: Any) -> tuple[Field, ...]:
    """The fields of a schema, or of a table that carries one."""
    if schema is None:
        return ()
    fields = getattr(schema, "fields", None)
    if fields is not None:
        return tuple(fields)
    inner = getattr(schema, "schema", None)
    if inner is not None:
        return tuple(getattr(inner, "fields", ()) or ())
    return ()
