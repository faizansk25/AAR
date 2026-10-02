"""The evidence contract: what an event is, and what makes it verifiable.

Deliberately small and deliberately boring. This is the layer every other part
of the audit subsystem agrees on, so it should change less often than anything
else in AAR - which is exactly what a verification surface should do.

The central decision is that **there is one event type, not one per
subsystem**. Making run start, policy decision, RLS application, node execution
and degradation into separate record kinds is how audit systems end up with four
inconsistent ledgers that agree with none of each other. One type, one
sequence, one hash chain: everything is an event, and ``event_type`` says what
kind.

Three things every event carries, because each is a question a compliance buyer
asks and a plain database row cannot answer:

* **Who** - ``subject``, including the identity provider that vouched for them.
  AAR does not authenticate anyone; it records what an external system
  asserted, and names which system that was.
* **Under what authority** - ``policy_id`` and ``policy_hash``. A decision is
  only auditable if the policy in force at the time is recoverable, and a hash
  is what makes it recoverable after the file is edited.
* **What actually happened** - ``engine_planned`` beside ``engine_used``. An
  approved plan and an executed plan are different things, and an answer that
  cannot distinguish them is not an answer.

``EvidenceEvent`` is immutable and hashable. Evidence you can mutate is not
evidence.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .canonical import GENESIS_HASH, chain_digest

__all__ = [
    "EVIDENCE_SCHEMA_VERSION", "EventType", "SubjectIdentity", "EvidenceEvent",
    "new_event_id", "new_run_id",
]

#: Bumped when the *meaning* of a field changes. Not cosmetic: an event written
#: under an older version describes different facts, and a verifier that cannot
#: tell them apart is not verifying.
EVIDENCE_SCHEMA_VERSION = "aar-evidence-v1"


class EventType(str, enum.Enum):
    """What happened. The closed set.

    Deliberately coarse. A compliance question is "was this row removed by
    RLS", not "which of eleven internal reasons triggered it" - and a
    vocabulary that grows a member per internal detail becomes a vocabulary
    nobody can write a query against.
    """

    RUN_STARTED = "run.started"
    RUN_FINISHED = "run.finished"
    NODE_EXECUTED = "node.executed"
    ROWS_RESTRICTED = "rows.restricted"
    COLUMNS_RESTRICTED = "columns.restricted"
    POLICY_ALLOWED = "policy.allowed"
    POLICY_DENIED = "policy.denied"
    EGRESS_ATTEMPTED = "egress.attempted"
    PLAN_DEVIATED = "plan.deviated"

    def __str__(self) -> str:  # pragma: no cover - display
        return self.value


@dataclass(frozen=True, slots=True)
class SubjectIdentity:
    """Who acted, and who vouched for them.

    AAR does not authenticate. It records an assertion made by something
    outside itself - an SSO provider, a directory, a notebook the analyst
    controls - and names that source in ``identity_provider``. A claim with no
    provider recorded is a claim AAR invented, and that distinction is the
    whole point of the field.
    """

    #: Stable identifier within the provider's namespace.
    subject_id: str
    #: Human-readable name. May not be unique; nothing relies on it.
    display_name: str = ""
    #: Sorted, because set iteration order is not stable across processes.
    roles: tuple[str, ...] = ()
    #: ``"entra"``, ``"okta"``, ``"ldap"``, ``"local"``... Empty means AAR was
    #: not told, which is a fact worth recording rather than a default.
    identity_provider: str = ""

    @classmethod
    def coerce(cls, value: Any) -> "SubjectIdentity | None":
        """Accept a :class:`~aar.governance.Subject`, or pass one through."""
        if value is None or isinstance(value, SubjectIdentity):
            return value
        return cls(
            subject_id=str(getattr(value, "name", "") or "anonymous"),
            display_name=str(getattr(value, "name", "") or ""),
            roles=tuple(sorted(getattr(value, "roles", ()) or ())),
            identity_provider="local",
        )

    def as_payload(self) -> dict[str, Any]:
        """The hashed and stored form. Flat, typed, and complete."""
        return {
            "subject_id": self.subject_id,
            "display_name": self.display_name,
            "roles": list(self.roles),
            "identity_provider": self.identity_provider,
        }


@dataclass(frozen=True, slots=True)
class EvidenceEvent:
    """One immutable, hashable fact about a governed execution.

    Every field is a plain scalar, a tuple of scalars, or a string-keyed
    mapping of the same - because the whole value must pass through
    :func:`~aar.audit.canonical.canonical_bytes` without coercion. That
    constraint is the contract working: it turns an uncanonicalisable field
    into a construction error rather than a hash that quietly disagrees with
    what was written to disk.

    Fields left empty are recorded as empty, not omitted. "This run had no
    rules" and "we did not record whether it had rules" are different claims,
    and only the first is safe to make in an audit.
    """

    run_id: str
    sequence: int
    event_type: EventType
    recorded_at: str
    schema_version: str = EVIDENCE_SCHEMA_VERSION
    event_id: str = ""

    # --- who, and under what authority
    subject: SubjectIdentity | None = None
    policy_id: str = ""
    policy_hash: str = ""

    # --- what was asked for, and what ran
    logical_plan_hash: str = ""
    physical_plan_hash: str = ""
    operation_id: str = ""
    graph_node_id: str = ""
    engine_planned: str = ""
    engine_used: str = ""

    # --- what it did to the data
    rows_in: tuple[int, ...] = ()
    rows_out: int | None = None
    #: Duration in whole microseconds. Integer because a float cannot be
    #: canonically hashed, and because sub-microsecond precision on a wall
    #: clock is not a fact about the execution.
    duration_us: int | None = None

    # --- why
    rule_id: str = ""
    reason: str = ""
    degradation: str = ""

    # --- environment
    aar_version: str = ""
    target_id: str = ""

    #: Event-specific detail, flattened into the hashed payload. Free-form on
    #: purpose: encoding every future audit concept into this dataclass means
    #: changing the contract for a new field, and the contract should change as
    #: rarely as possible.
    attributes: Mapping[str, Any] = field(default_factory=dict)

    # --- the chain
    previous_hash: str = GENESIS_HASH
    event_hash: str = ""

    def __post_init__(self) -> None:
        if not self.event_id:
            object.__setattr__(self, "event_id", new_event_id())
        # A dict passed in is still mutable from outside, which would let a
        # recorded event change *after* its hash was taken. Freezing it is what
        # makes the hash mean anything.
        object.__setattr__(self, "attributes", _freeze(self.attributes))

    def payload(self) -> dict[str, Any]:
        """The exact structure that is hashed.

        ``event_hash`` is excluded - it cannot cover itself - and everything
        else is included, schema version first. A verifier recomputes from this
        and gets the same answer, which is the only property that matters.
        """
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "event_type": self.event_type.value,
            "recorded_at": self.recorded_at,
            "subject": self.subject.as_payload() if self.subject else None,
            "policy_id": self.policy_id,
            "policy_hash": self.policy_hash,
            "logical_plan_hash": self.logical_plan_hash,
            "physical_plan_hash": self.physical_plan_hash,
            "operation_id": self.operation_id,
            "graph_node_id": self.graph_node_id,
            "engine_planned": self.engine_planned,
            "engine_used": self.engine_used,
            "rows_in": list(self.rows_in),
            "rows_out": self.rows_out,
            "duration_us": self.duration_us,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "degradation": self.degradation,
            "aar_version": self.aar_version,
            "target_id": self.target_id,
            "attributes": dict(self.attributes),
            "previous_hash": self.previous_hash,
        }

    def sealed(self, previous_hash: str) -> "EvidenceEvent":
        """This event with its hash computed over ``previous_hash``.

        Returns a new object rather than mutating, so an unsealed event can be
        held and inspected without the risk of it quietly becoming the sealed
        one.
        """
        pending = replace(self, previous_hash=previous_hash, event_hash="")
        return replace(pending, event_hash=chain_digest(previous_hash,
                                                        pending.payload()))

    def verify(self) -> bool:
        """Whether this event's own hash matches its contents.

        Says nothing about its *place* in the chain; :func:`verify_chain` in
        :mod:`aar.audit.recorder` covers that. Split because a per-event check
        and a sequencing check fail for different reasons and are diagnosed
        differently.
        """
        if not self.event_hash:
            return False
        return chain_digest(self.previous_hash,
                            self.payload()) == self.event_hash

    def row(self) -> dict[str, Any]:
        """The stored form: indexed columns beside the canonical payload."""
        return {
            "run_id": self.run_id,
            "sequence": self.sequence,
            "event_type": self.event_type.value,
            "subject_id": (self.subject.subject_id if self.subject else None),
            "rule_id": self.rule_id or None,
            "node_id": self.graph_node_id or None,
            "recorded_at": self.recorded_at,
            "event_hash": self.event_hash,
            "previous_hash": self.previous_hash,
            "payload": self.payload(),
        }


def new_run_id() -> str:
    """A run identifier. Random, because a run is an event in time, not content."""
    return f"run_{uuid.uuid4().hex}"


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def _freeze(value: Any) -> Any:
    """Deep-freeze a mapping into plain dicts of tuples and scalars."""
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"evidence attribute key {key!r} is not a string; a "
                    f"non-string key has no canonical order")
            out[key] = _freeze(item)
        return out
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        raise TypeError(
            "evidence attributes must not contain sets; their iteration order "
            "is not stable, so the same event would hash differently twice")
    return value
