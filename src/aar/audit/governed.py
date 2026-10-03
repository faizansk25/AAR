"""The governed execution lifecycle: one object that closes every run it opens.

AAR already had four independent recorders - ``ExecutionResult.outcomes``,
``PolicyEngine.decisions``, ``DegradationLedger``, ``LineageEvent`` - written at
different points in different units with no shared ordering, which is how an
audit trail stops being one. :mod:`aar.audit.recorder` fixed that by making one
canonical stream. The obvious next mistake is to add a fifth: a
``GovernedRun.events`` list, populated from the others afterwards, in a
different order.

:class:`GovernedRun` is deliberately *not* that. It is a lifecycle wrapper::

    GovernedRun
        |
        +-- EvidenceRecorder   <- canonical truth, sole owner of the chain
                 ^
                 |
          Executor / Policy / Disclosure

It holds no event list. ``events()`` delegates, so there is exactly one chain,
one ordering and one place where a hash is taken. Everything this class adds is
sequencing and classification - things the recorder cannot know on its own.

Three decisions live here, and each would be wrong somewhere else.

**The run opens before the policy is read.** A run that fails to load its
policy still happened: somebody attempted it, and a compliance system that
records only successful runs cannot answer "was anything attempted?". So the
recorder opens first, binds the authority when it learns it, and closes in
every path including the exception paths.

**Authority is bound by a later event, never by editing the first.** The
opening event is hashed when written. Writing the ``policy_id`` onto it later
would either break the chain or force a rehash, leaving a ``run.started`` that
describes facts nobody possessed at the time. So ``run.started`` says
``policy_state = unresolved`` and a separate ``policy.bound`` records the
authority when it became known - preserving the chronology that "which policy
governed this?" needs.

**Denied is not failed.** A run stopped by policy did its job; a run that
crashed is a fault. Both were recorded as ``failed``, which reported correctly-
refused attempts as errors and trained people to read denials as noise.

Every disclosure verdict is *projected* here rather than accumulated: the
executor's :class:`~aar.governance.disclosure.Suppression` is a working return
value, and the canonical record is the ``disclosure.evaluated`` event it
becomes. There is deliberately no suppression *collection* to reconstruct
ordering from later - that would be a second timeline.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from ..failures import PolicyDenied, PrivacyViolation
from .contract import EventType, EvidenceEvent, RunStatus
from .recorder import EvidenceRecorder, RunContext, event_fields

__all__ = [
    "GovernedRun", "BASELINE_POLICY_ID", "BASELINE_POLICY_HASH",
    "policy_identity", "disclosure_attributes",
]

#: The identity of the built-in policy used when no file is supplied.
#: Deterministic and versioned on purpose: "no policy file" is itself a
#: governance state, and AAR applies its own baseline rules in that case. An
#: empty ``policy_id`` would be ambiguous between "no policy" and "we forgot to
#: record it", so the baseline names itself.
BASELINE_POLICY_ID = "aar.baseline-policy-v1"

#: A hash of the *name*, not of policy contents. There are no contents: the
#: baseline is the compiled-in default in :mod:`aar.governance.policy`. It is
#: still a stable identifier so "which policy governed this" has an answer, and
#: so a future store can tell a v1 baseline from a v2 rather than colliding.
BASELINE_POLICY_HASH = hashlib.sha256(
    f"aar.baseline-policy-v1:{BASELINE_POLICY_ID}".encode("utf-8")).hexdigest()[:16]


def policy_identity(path: str | None, loaded: Any = None) -> tuple[str, str]:
    """The ``(policy_id, policy_hash)`` for a run, given its policy file.

    A supplied file is identified by its name and a hash of its *contents*, so
    two policies sharing a name but differing in rules are distinguishable -
    which is the question an auditor actually asks ("was this the finance
    policy in force in March?").

    Contents rather than mtime: an edited-then-restored file looks unchanged
    under mtime granularity on some filesystems, and a policy hash that misses
    an edit is worse than no hash at all.
    """
    if not path:
        return BASELINE_POLICY_ID, BASELINE_POLICY_HASH
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
    except OSError:
        # Unreadable *right now* is not the same as "there is no policy". The
        # id is still the path; the caller emits policy.bind_failed, so the run
        # records the attempt rather than silently claiming the baseline.
        return str(path), ""
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    return name, digest.hexdigest()[:16]


def _verdict(suppression: Any, refused: str) -> str:
    """``passed`` | ``suppressed`` | ``refused``.

    Three states, not two, and the third matters most. "The control ran and
    removed nothing" and "the control could not measure the groups so the run
    was refused" are opposite facts; collapsing them would let a refusal report
    as a pass - which is the exact confusion Round 30 removed for RLS.
    """
    if refused:
        return "refused"
    groups = int(getattr(suppression, "suppressed_groups", 0) or 0)
    return "suppressed" if groups else "passed"


def disclosure_attributes(suppression: Any, *, refused: str = "") -> dict:
    """Project a :class:`~aar.governance.disclosure.Suppression` into event fields.

    The adaptor, and it lives here rather than in the executor so the executor
    does not grow a dependency on the audit package - the same separation
    :func:`aar.audit.recorder.event_fields` uses for ``NodeOutcome``.

    Every contributing rule is recorded as structured data rather than prose.
    When two policies applied and the stricter one bound, "k=10 because policy B
    overrode policy A" is a claim a reader must trust; a list of
    ``{rule_id, min_group_size}`` pairs is a claim they can check. Provenance in
    a sentence is not evidence.

    ``applied_before_exposure`` is explicit rather than implied by the event
    type: a reader asking "before aggregation or after?" - the question Round 26
    existed to answer - should not have to infer it from sequencing.
    """
    rule = getattr(suppression, "rule", "") or ""
    contributing = getattr(suppression, "contributing", ()) or ()
    fields: dict[str, Any] = {
        "rule_id": rule,
        "reason": refused or getattr(suppression, "reason", "") or "",
        "graph_node_id": getattr(suppression, "node_id", "") or "",
        "attributes": {
            "control": "min_group_size",
            "verdict": _verdict(suppression, refused),
            "suppressed_groups": int(
                getattr(suppression, "suppressed_groups", 0) or 0),
            "smallest_group": getattr(suppression, "smallest_group", None),
            "applied_before_exposure": True,
            "contributing_rules": [
                {"rule_id": r.rule, "min_group_size": int(r.min_group_size)}
                for r in contributing],
        },
    }
    minimum = getattr(suppression, "minimum_required", None)
    if minimum is not None:
        fields["attributes"]["effective_minimum"] = int(minimum)
        fields["attributes"]["effective_rule_id"] = rule
    return fields


class GovernedRun:
    """Opens an evidence run, binds its authority, and always closes it.

    A context manager, because the failure mode this exists to prevent is an
    exception path that skips ``finish_run`` - leaving a chain open forever and
    every later ``emit`` failing with "run is still open". The guarantee is
    therefore structural rather than a convention each caller must remember::

        with GovernedRun(subject="dana") as run:
            run.bind_policy(policy_path)
            ...                       # may raise anything

    and the run is closed with the right status whichever way the block exits.

    It stores no events. ``events()`` and ``verify()`` delegate to the
    recorder, so there is exactly one chain and one ordering.
    """

    __slots__ = ("recorder", "context", "_closed", "_disclosure_seen")

    def __init__(self, subject: Any = None,
                 recorder: EvidenceRecorder | None = None,
                 policy_path: str | None = None, target_id: str = "",
                 aar_version: str = "") -> None:
        self.recorder = recorder or EvidenceRecorder()
        #: Opened before the policy is read, deliberately. The authority is
        #: bound by a later event; see the module docstring for why that is not
        #: the same as filling in the opening event.
        self.context: RunContext = self.recorder.start_run(
            subject=subject, target_id=target_id, aar_version=aar_version)
        self._closed = False
        #: Node ids whose disclosure verdict has been recorded. Only to refuse a
        #: *duplicate* event for the same aggregate - not as a store. The
        #: canonical facts live in the recorder; this cannot reconstruct them.
        self._disclosure_seen: set[str] = set()
        if policy_path is not None:
            self.bind_policy(policy_path)

    # ------------------------------------------------------------- authority
    def bind_policy(self, policy_path: str | None,
                    policy_id: str = "", policy_hash: str = "") -> EvidenceEvent:
        """Bind the authority this run executes under.

        With a path, the identity is derived from the file's name and contents;
        without one, the built-in baseline names itself. Either way it is a
        stable, honest answer to "which policy governed this?" rather than an
        empty string that could mean anything.
        """
        if policy_path or not policy_id:
            policy_id, policy_hash = policy_identity(policy_path)
        return self.recorder.bind_policy(policy_id, policy_hash)

    def policy_bind_failed(self, reason: str) -> EvidenceEvent:
        """Record that no authority could be established for this run.

        Its own event type, not a ``policy.denied``: nothing was refused, the
        run simply never obtained the authority it would have needed to refuse
        anything with. A refusal implies a judgement, and this is the absence of
        the means to make one.
        """
        return self.recorder.emit(EventType.POLICY_BIND_FAILED, reason=reason)

    # ------------------------------------------------------------- governance
    def record_policy_denial(self, reason: str, rule_id: str = "") -> EvidenceEvent:
        return self.recorder.emit(EventType.POLICY_DENIED, reason=reason,
                                  rule_id=rule_id)

    def record_disclosure(self, suppression: Any,
                          refused: str = "") -> EvidenceEvent:
        """Project one disclosure verdict into the canonical chain.

        Called for **every** guarded aggregate, including the ones that passed.
        An event emitted only when something was suppressed would repeat the
        mistake Round 30 fixed for RLS: absence would have to mean "nothing was
        removed", when it could equally mean the control never ran.

        A second verdict for the same aggregate is refused rather than recorded:
        a duplicate would let a caller supersede an earlier verdict with a later,
        softer one, and which one governed would then depend on emission order.
        """
        node_id = getattr(suppression, "node_id", "") or ""
        if node_id and node_id in self._disclosure_seen:
            raise RuntimeError(
                f"aggregate {node_id} already has a disclosure verdict in this "
                f"run; a second event would let one verdict supersede another, "
                f"and which one governed would depend on emission order")
        self._disclosure_seen.add(node_id)
        fields = disclosure_attributes(suppression, refused=refused)
        return self.recorder.emit(EventType.DISCLOSURE_EVALUATED, **fields)

    def record_nodes(self, outcomes: Sequence[Any]) -> list[EvidenceEvent]:
        """Project executor outcomes, in the executor's order.

        Returns the same event objects the recorder holds rather than copies,
        so a caller holding one cannot diverge from the chain.
        """
        return [self.recorder.emit(EventType.NODE_EXECUTED, **event_fields(o))
                for o in outcomes]

    def record_restriction_state(self, state: str,
                                 rows_before: int | None = None,
                                 rows_after: int | None = None) -> EvidenceEvent:
        """Record how row-level security actually governed this run.

        Always emitted, with one of the four
        :class:`~aar.audit.view.RestrictionState` values. A reader must never
        have to infer "no restriction" from a missing event, because "evaluated
        and nothing matched" and "never evaluated" demand opposite answers.
        """
        removed = (None if rows_before is None or rows_after is None
                   else max(0, rows_before - rows_after))
        return self.recorder.emit(
            EventType.ROWS_RESTRICTED, restriction_state=str(state),
            rows_in=((rows_before,) if rows_before is not None else ()),
            rows_out=rows_after,
            attributes={"rows_removed": removed,
                        "applied_before_aggregation": True})

    # ------------------------------------------------------------- delegation
    def events(self, event_type: EventType | None = None) -> tuple[EvidenceEvent, ...]:
        """The chain, straight from the recorder.

        Delegated, not mirrored, and returned as the recorder's own tuple - so a
        caller holding one cannot diverge from the chain. The optional filter is
        applied here rather than by the recorder because the recorder stores
        one run at a time and does not need it; the shape is kept identical so
        this is a pure pass-through rather than a second accessor with its own
        semantics.
        """
        events = self.recorder.events()
        if event_type is None:
            return events
        return tuple(e for e in events if e.event_type is event_type)

    def verify(self) -> Any:
        return self.recorder.verify()

    @property
    def status(self) -> str:
        return self.context.status

    # ------------------------------------------------------------- lifecycle
    def close(self, status: Any = RunStatus.SUCCESS, reason: str = "") -> RunContext:
        """Seal the run. Idempotent, because two seals would be a lie.

        A second call returns the same context rather than emitting a second
        ``run.finished``: the chain would then contain two terminal events and
        a reader could not tell which one governed.
        """
        if self._closed:
            return self.context
        self._closed = True
        return self.recorder.finish_run(status=str(status), reason=reason)

    def denied(self, reason: str, rule_id: str = "") -> RunContext:
        """Close as ``DENIED``: governance deliberately prevented this.

        Also records the denial itself, so the chain says *which* control
        refused. A ``DENIED`` status alone tells an auditor that something was
        blocked and nothing about what.
        """
        self.record_policy_denial(reason, rule_id)
        return self.close(RunStatus.DENIED, reason)

    def failed(self, reason: str) -> RunContext:
        """Close as ``FAILED``: AAR could not complete the run."""
        return self.close(RunStatus.FAILED, reason)

    def __enter__(self) -> "GovernedRun":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Close on every path, classifying the exception.

        The ordering matters: a governance refusal is recorded as a refusal
        *and* the run is closed as denied, while anything else is a failure. A
        caller that wants to handle the denial itself still sees it re-raised -
        recording an attempt is not the same as swallowing it.
        """
        if self._closed:
            return False
        if exc is None:
            self.close(RunStatus.SUCCESS)
        elif isinstance(exc, (PolicyDenied, PrivacyViolation)):
            self.denied(str(exc), rule_id=getattr(exc, "rule", "") or "")
        else:
            self.failed(f"{type(exc).__name__}: {exc}")
        return False