"""Authorized *views* over immutable evidence.

The store keeps the exact facts. This module decides who sees which of them.

That separation is the whole design, and it exists because of a specific
failure mode: redaction applied at record time produces an audit trail that is
*complete for whoever can read it* and silently incomplete for everyone else.
So the chain records ``rows_before: 18_241`` and ``rows_removed: 17_627``
regardless of who is watching, and the withholding happens here.

Four properties follow, each a decision rather than an accident:

* **Redaction never re-hashes.** A view is not an ``EvidenceEvent`` with fields
  removed and a fresh hash. It is an :class:`EvidenceView` carrying the
  ``source_event_id`` and ``source_event_hash`` it came from, so AAR can say
  *"the evidence verifies; three fields were withheld from this viewer"*
  instead of passing off a doctored object as the proof.
* **Capabilities, not role names.** ``"compliance"`` is one company's word for
  it; another calls the same person ``"auditor"``. Authorization asks about
  :class:`Capability` values and a deployment maps its own roles onto them.
* **The viewer is not the subject.** ``explain_access("dana")`` says whose run
  is examined, not who is asking. Letting a caller pass ``role="compliance"``
  would be authorization by assertion.
* **Redaction is not ``None``.** The evidence semantics already distinguish
  "never measured" from "measured as zero". Collapsing "you may not see this"
  into the same bucket would destroy that, so a withheld value is a
  :class:`FieldValue` in state ``REDACTED`` with a reason.

Counts are withheld by default because they are *data about data*: "614 of
18,241 rows survived" is itself a disclosure, and repeated counts across
varying filters permit a differencing attack that recovers protected rows one
at a time.

## The trust boundary

This layer enforces visibility in the Workbench, the CLI and the API. It is
**not** strong isolation: AAR is local-first, so anyone who can read the
evidence file directly can read what this module hides. See
:func:`trust_boundary_notice`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from .contract import EvidenceEvent, SubjectIdentity

__all__ = [
    "Capability", "EvidenceAuthorizer", "EvidenceView", "FieldState",
    "FieldValue", "RESTRICTED_COUNT_FIELDS", "project",
    "trust_boundary_notice",
]


class Capability(str, enum.Enum):
    """What a viewer may do. Never a role name.

    A deployment maps its own roles onto these. Comparing a role string inside
    the authorization code would mean trusting a value the caller supplied
    about itself.
    """

    AUDIT_READ = "audit.read"
    AUDIT_READ_POLICY_DETAILS = "audit.read_policy_details"
    #: May see row counts for data they were not authorised to read. Scoped
    #: separately because it is the dangerous one: a count of excluded rows is
    #: data about data.
    AUDIT_READ_RESTRICTED_COUNTS = "audit.read_restricted_counts"
    AUDIT_READ_OTHER_SUBJECTS = "audit.read_other_subjects"
    AUDIT_VERIFY_INTEGRITY = "audit.verify_integrity"

    def __str__(self) -> str:  # pragma: no cover - display
        return self.value


#: Fields carrying counts of rows the subject was not authorised to see.
RESTRICTED_COUNT_FIELDS: tuple[str, ...] = (
    "rows_before", "rows_after", "rows_removed",
)


class FieldState(str, enum.Enum):
    """Why a field has the value it has. Never collapsed into ``None``."""

    MEASURED = "measured"
    #: Never measured - a different fact from "withheld".
    UNKNOWN = "unknown"
    #: Measured, and withheld from this viewer.
    REDACTED = "redacted"

    def __str__(self) -> str:  # pragma: no cover - display
        return self.value


class RestrictionState(str, enum.Enum):
    """Whether row-level security actually governed a run. Four states, not two.

    The tempting implementation is "look for a ``rows.restricted`` event; if
    there isn't one, no RLS was applied". That is wrong, and it is wrong in the
    direction that matters: absence of evidence is not evidence of absence.

    Four situations collapse into two under that shortcut, and the two are
    opposites:

    * policy evaluated, no rule matched this subject - genuinely unrestricted;
    * governance was bypassed, or the run died before determination - we do not
      know, and must not say "unrestricted".

    So the determination is recorded explicitly on every event
    (:attr:`EvidenceEvent.restriction_state`) and rendered as one of four
    values. :attr:`UNKNOWN` is the load-bearing one: it is what an old evidence
    version, or a run interrupted before the policy was consulted, resolves to.
    """

    #: A matching rule was found and executed.
    APPLIED = "applied"
    #: Policy evaluated successfully; no rule applied to this subject or source.
    NOT_APPLICABLE = "not_applicable"
    #: Governance was explicitly bypassed (``--unsafe-disable-policy``).
    NOT_EVALUATED = "not_evaluated"
    #: The run ended, or predates this field, before a determination.
    UNKNOWN = "unknown"

    def __str__(self) -> str:  # pragma: no cover - display
        return self.value

    @property
    def is_determined(self) -> bool:
        """Whether this is a claim about the run rather than about AAR."""
        return self not in (self.UNKNOWN,)

    @property
    def label(self) -> str:
        """How to say it to a person, without leaking the authorization model."""
        return self.name.replace("_", " ").upper()


@dataclass(frozen=True, slots=True)
class FieldValue:
    """A value together with why it is or is not available.

    Returning a bare ``None`` for a withheld field would make three situations
    indistinguishable - not measured, measured as zero, and withheld - which
    is exactly the confusion an audit product cannot afford.
    """

    state: FieldState
    value: Any = None
    reason: str = ""

    @classmethod
    def measured(cls, value: Any) -> "FieldValue":
        return cls(FieldState.MEASURED, value)

    @classmethod
    def unknown(cls, reason: str = "not measured") -> "FieldValue":
        return cls(FieldState.UNKNOWN, None, reason)

    @classmethod
    def redacted(cls, reason: str = "insufficient_privilege") -> "FieldValue":
        return cls(FieldState.REDACTED, None, reason)

    @property
    def is_visible(self) -> bool:
        return self.state is FieldState.MEASURED

    def __bool__(self) -> bool:
        return self.is_visible

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"state": self.state.value}
        if self.state is FieldState.MEASURED:
            out["value"] = self.value
        if self.reason:
            out["reason"] = self.reason
        return out

    def render(self) -> str:
        if self.state is FieldState.MEASURED:
            return str(self.value)
        if self.state is FieldState.REDACTED:
            return "[restricted audit detail]"
        return f"[{self.reason or 'unknown'}]"


@dataclass(frozen=True, slots=True)
class EvidenceAuthorizer:
    """Decides what one viewer may see. Holds no policy of its own.

    Constructed from an externally authenticated identity - today whatever
    the embedding application has, later whatever Entra, Okta or LDAP
    asserted. AAR does not authenticate anyone, so this is deliberately just a
    value: the capability set has to come from somewhere AAR controls.
    """

    viewer: SubjectIdentity
    capabilities: frozenset[Capability] = frozenset()

    def may(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def may_read(self, event: EvidenceEvent) -> bool:
        """Whether this viewer may read this event at all.

        Reading *your own* runs is the base case; reading someone else's is a
        capability, because "who is this about" and "who is asking" are
        different questions.
        """
        if not self.may(Capability.AUDIT_READ):
            return False
        if event.subject is None or \
                event.subject.subject_id == self.viewer.subject_id:
            return True
        return self.may(Capability.AUDIT_READ_OTHER_SUBJECTS)

    def may_see_restricted_counts(self) -> bool:
        return self.may(Capability.AUDIT_READ_RESTRICTED_COUNTS)

    def may_see_predicate(self) -> bool:
        return self.may(Capability.AUDIT_READ_POLICY_DETAILS)

    def may_verify(self) -> bool:
        return self.may(Capability.AUDIT_VERIFY_INTEGRITY)


@dataclass(frozen=True, slots=True)
class EvidenceView:
    """An authorized projection of one event. Not an event; never re-hashed.

    The provenance fields are the point. ``source_event_id`` and
    ``source_event_hash`` name the event this came from, and ``chain_verified``
    reports whether *that* event verifies - so a caller can tell "the evidence
    is sound and you are seeing less of it" from "the evidence is unsound",
    which a view that silently dropped fields would make indistinguishable
    from a doctored one.
    """

    source_event_id: str
    source_event_hash: str
    run_id: str
    event_type: str
    sequence: int
    chain_verified: bool
    visible: dict[str, Any]
    #: Names of fields withheld, so the omission is stated rather than inferred
    #: from a missing key. Machine-readable and deterministic, because a client
    #: needs to know exactly what it did not get.
    redacted_fields: tuple[str, ...] = ()
    #: Names of fields never measured - a different fact again.
    unknown_fields: tuple[str, ...] = ()
    #: :class:`~aar.audit.view.RestrictionState` value for this run, recorded
    #: rather than inferred from the presence of a restriction event.
    restriction_state: str = RestrictionState.UNKNOWN
    #: Whether the *result* of integrity verification is safe to show this
    #: viewer. It is, for anyone who may read the run: a valid chain reveals
    #: no protected business data, and suppressing it would just teach readers
    #: to distrust the word VALID. The diagnostics behind a *break* are a
    #: different matter and need :attr:`Capability.AUDIT_VERIFY_INTEGRITY`.
    integrity_detail_permitted: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "source_event_id": self.source_event_id,
            "source_event_hash": self.source_event_hash,
            "run_id": self.run_id,
            "event_type": self.event_type,
            "sequence": self.sequence,
            "chain_verified": self.chain_verified,
            "visible": {k: (v.to_json() if isinstance(v, FieldValue) else v)
                        for k, v in self.visible.items()},
            "redacted_fields": list(self.redacted_fields),
            "unknown_fields": list(self.unknown_fields),
        }

    def render(self) -> str:
        """The analyst-facing rendering: no counts, proof intact.

        Prose, because that is the product. "RLS applied: yes" plus a rule name
        is a complete and honest answer to *was my access restricted*; the
        figures are what the compliance view is for.
        """
        lines = [f"Run {self.run_id}, event {self.sequence}: {self.event_type}"]
        for name, value in self.visible.items():
            shown = value.render() if isinstance(value, FieldValue) else value
            lines.append(f"  {name}: {shown}")
        lines.append(f"  Evidence event: {self.source_event_id}")
        lines.append(f"  Evidence hash:  {self.source_event_hash[:16]}...")
        state = "VALID" if self.chain_verified else "BROKEN"
        lines.append(f"  Evidence integrity: {state}")
        if not self.chain_verified and not self.integrity_detail_permitted:
            lines.append("  Break details require "
                         "audit.verify_integrity")
        if self.redacted_fields:
            lines.append("  Restricted audit detail withheld: "
                         + summarise_withheld(self.redacted_fields))
        return "\n".join(lines)


#: Withheld fields described in the abstract rather than by name. Enumerating
#: every one leaks the shape of the privileged evidence: a viewer who learns
#: that ``restricted_group_count`` exists has been told something about the
#: system that is not theirs to know, and the list grows every time a field is
#: added. The machine-readable ``redacted_fields`` keeps the exact names for
#: clients that need deterministic semantics.
_WITHHELD_SUMMARY = "additional audit detail is restricted for your current access"


def summarise_withheld(fields: "tuple[str, ...]") -> str:
    """The human-facing phrase. Names nothing, and names no role."""
    return _WITHHELD_SUMMARY if fields else ""


def project(event: EvidenceEvent, authorizer: EvidenceAuthorizer,
            chain_verified: bool = True) -> EvidenceView:
    """Build the view one viewer is entitled to.

    ``rows_after`` is withheld by default even though it counts *authorised*
    rows, and the reason is worth stating: an aggregate-only query may return a
    single row although hundreds of authorised source rows participated, so the
    figure is a disclosure the analyst did not necessarily learn from their own
    result.
    """
    if not authorizer.may_read(event):
        return EvidenceView(
            source_event_id=event.event_id,
            source_event_hash=event.event_hash,
            run_id=event.run_id,
            event_type=event.event_type.value,
            sequence=event.sequence,
            chain_verified=chain_verified,
            visible={"access": "not permitted for this viewer"},
            redacted_fields=("event contents",),
            restriction_state=event.restriction_state or
            RestrictionState.UNKNOWN,
            integrity_detail_permitted=authorizer.may_verify(),
        )

    visible: dict[str, Any] = {}
    redacted: list[str] = []
    unknown: list[str] = []
    may_counts = authorizer.may_see_restricted_counts()

    def put(name: str, value: Any) -> None:
        """Record a field, distinguishing *absent* from *present*.

        An empty string counts as absent. The contract models these fields as
        ``str`` with ``""`` meaning "not applicable to this event", so treating
        ``""`` as a measurement would render ``engine_planned: measured ("")``
        for a node that has no planned engine - a field claiming to be evidence
        while carrying none. That is the same conflation this module exists to
        avoid, one level down.
        """
        if value is None or (isinstance(value, str) and not value.strip()):
            visible[name] = FieldValue.unknown()
            unknown.append(name)
        else:
            visible[name] = FieldValue.measured(value)

    put("rule", event.rule_id)
    put("source", event.attributes.get("source"))
    put("reason", event.reason)
    put("engine_planned", event.engine_planned)
    put("engine_used", event.engine_used)
    put("degradation", event.degradation)
    put("applied_before_aggregation",
        event.attributes.get("applied_before_aggregation"))

    for name in RESTRICTED_COUNT_FIELDS:
        raw = event.attributes.get(name)
        if name == "rows_after" and raw is None:
            raw = event.rows_out
        if raw is None:
            visible[name] = FieldValue.unknown()
            unknown.append(name)
        elif may_counts:
            visible[name] = FieldValue.measured(raw)
        else:
            visible[name] = FieldValue.redacted()
            redacted.append(name)

    if "predicate" in event.attributes:
        # Shown or withheld, never dropped: an absent key and a redacted one
        # are different facts, and the permitted case must not be the same
        # shape as the forbidden case.
        if authorizer.may_see_predicate():
            visible["predicate"] = FieldValue.measured(
                event.attributes["predicate"])
        else:
            visible["predicate"] = FieldValue.redacted()
            redacted.append("predicate")

    return EvidenceView(
        source_event_id=event.event_id,
        source_event_hash=event.event_hash,
        run_id=event.run_id,
        event_type=event.event_type.value,
        sequence=event.sequence,
        chain_verified=chain_verified,
        visible=visible,
        redacted_fields=tuple(redacted),
        unknown_fields=tuple(unknown),
        # Read from the record, never deduced from the presence of a
        # restriction event. An event absent because governance was bypassed
        # must not be reported as one absent because no rule applied.
        restriction_state=event.restriction_state or RestrictionState.UNKNOWN,
        # Integrity *status* is safe for any reader; the diagnostics behind a
        # break are what the capability actually gates.
        integrity_detail_permitted=authorizer.may_verify(),
    )


TRUST_BOUNDARY = """\
AAR's audit-detail visibility is enforced by the Workbench, the CLI and the
query API. It is NOT strong isolation: the evidence database is local-first and
lives on the user's machine, so anyone who can read that file directly can read
the counts this layer withholds from the API.

Strong separation requires one of: restricted filesystem permissions on the
evidence store; a separate audit service or process holding the store; a
centralised evidence server; or an encrypted store whose key is held
separately. None is built yet, and AAR does not claim otherwise."""


def trust_boundary_notice() -> str:
    """The limitation above, as a string, for a CLI or an explain panel."""
    return TRUST_BOUNDARY