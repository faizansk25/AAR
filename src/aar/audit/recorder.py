"""One evidence sink. Every subsystem emits into it; none keeps its own list.

The codebase already had four independent recorders -
``ExecutionResult.outcomes``, ``PolicyEngine.decisions``,
``DegradationLedger``, ``LineageEvent`` - which is exactly how an audit trail
stops being one. They are written at different points, in different units,
with no shared ordering, so *reconstructing* a run means guessing which
happened first. Adding a fifth called ``GovernedRun.events`` and copying the
others into it afterwards would have made it five.

So the direction is reversed: the recorder is the canonical stream, and those
objects become ergonomic projections of it. A caller that wants the node
outcomes still gets them; a caller that wants "what happened, in order, under
what authority" gets the chain.

What this class guarantees:

* **one order** - a single monotonic sequence per run, assigned here, not by
  each caller;
* **one chain** - every event sealed against its predecessor's hash;
* **one clock** - every event stamped from one source, so two events cannot
  disagree about order by disagreeing about time;
* **refuses to guess** - emitting outside a run raises rather than producing
  an event that cannot be placed in the chain.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from .canonical import GENESIS_HASH
from .contract import (EventType, EvidenceEvent, SubjectIdentity, new_run_id)

__all__ = [
    "EvidenceRecorder", "RunContext", "VerificationResult", "stamp",
    "verify_chain", "event_fields", "rows_removed",
]


def stamp() -> str:
    """The one timestamp format, from the one clock.

    ISO-8601 UTC, millisecond precision, explicit ``Z``. Fixed precision
    because a timestamp is hashed: if the format varies with the caller's
    locale or Python version, two identical events written on two machines hash
    differently and the chain cannot be verified across them.
    """
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass(slots=True)
class RunContext:
    """One governed execution, open or sealed."""

    run_id: str
    subject: SubjectIdentity | None
    policy_id: str
    policy_hash: str
    logical_plan_hash: str = ""
    physical_plan_hash: str = ""
    target_id: str = ""
    aar_version: str = ""
    started_at: str = ""
    finished_at: str = ""
    status: str = "running"
    first_hash: str = GENESIS_HASH
    final_hash: str = GENESIS_HASH
    event_count: int = 0



@dataclass(frozen=True, slots=True)
class VerificationResult:
    """The outcome of checking a chain.

    Reports *where* it broke, because "the evidence is invalid" is not an
    answer an auditor can act on - "event 47 does not match its own contents"
    is.
    """

    valid: bool
    checked: int
    first_bad_sequence: int | None = None
    reason: str = ""

    def __bool__(self) -> bool:
        return self.valid

    def render(self) -> str:
        if self.valid:
            return f"EVIDENCE CHAIN VALID ({self.checked} events verified)"
        return (f"EVIDENCE CHAIN BROKEN at event {self.first_bad_sequence}: "
                f"{self.reason}")


class EvidenceRecorder:
    """The single sink for governed-execution evidence.

    Cheap by default: with no ``on_event`` callback it keeps events in memory
    and writes nothing, so the subsystems that emit evidence do not force a
    database into existence. Pass a callback - or an evidence store - when the
    evidence must outlive the process.
    """

    __slots__ = ("_lock", "_run", "_recorded", "_on_event", "_clock")

    def __init__(self, on_event: Any = None, clock: Any = stamp) -> None:
        self._lock = threading.Lock()
        self._run: RunContext | None = None
        #: Named ``_recorded`` rather than ``_events`` because the public
        #: accessor is ``events()``. The earlier name collided with it, so
        #: ``_events()`` resolved to the list and ``verify`` died on a call.
        self._recorded: list[EvidenceEvent] = []
        self._on_event = on_event
        self._clock = clock

    # ------------------------------------------------------------- lifecycle
    @property
    def active_run(self) -> str:
        return self._run.run_id if self._run else ""

    @property
    def is_recording(self) -> bool:
        return self._run is not None

    def start_run(self, subject: Any = None, policy_id: str = "",
                  policy_hash: str = "", logical_plan_hash: str = "",
                  physical_plan_hash: str = "", target_id: str = "",
                  aar_version: str = "", run_id: str = "") -> RunContext:
        """Open a run. A recorder records one run at a time.

        Sequential runs are a deliberate limit: a single chain is a far
        stronger claim than a set of interleaved ones. Parallel governed
        executions become separate runs referencing a common parent, once
        there is a real reason to support that.

        The run is installed *before* the opening event is emitted, so no lock
        is held across the call that will take it again.
        """
        with self._lock:
            if self._run is not None:
                raise RuntimeError(
                    f"run {self._run.run_id} is still open; finish it before "
                    f"starting another. Two runs in one chain would assert an "
                    f"ordering that no execution had.")
            context = RunContext(
                run_id=run_id or new_run_id(),
                subject=SubjectIdentity.coerce(subject),
                policy_id=policy_id,
                policy_hash=policy_hash,
                logical_plan_hash=logical_plan_hash,
                physical_plan_hash=physical_plan_hash,
                target_id=target_id,
                aar_version=aar_version,
                started_at=self._clock(),
            )
            self._run = context
            self._recorded = []
        self.emit(EventType.RUN_STARTED)
        return context

    def finish_run(self, status: str = "ok", reason: str = "") -> RunContext:
        """Seal the run with a ``run.finished`` carrying the terminal status."""
        with self._lock:
            if self._run is None:
                raise RuntimeError("no run is open")
            context = self._run
        self.emit(EventType.RUN_FINISHED, reason=reason,
                  attributes={"status": status})
        with self._lock:
            self._run.status = status
            self._run.finished_at = self._clock()
            sealed = context
            self._run = None
        return sealed

    # ----------------------------------------------------------------- emit
    def emit(self, event_type: EventType, **fields: Any) -> EvidenceEvent:
        """Record one event. The only way anything enters the chain.

        ``fields`` are :class:`EvidenceEvent` attributes; ``attributes=``
        carries event-specific detail. Run-level facts - subject, policy, plan
        hashes - are filled in here so no caller can forget them, because a
        node event with no subject attached is exactly the kind of hole nobody
        notices until an auditor finds it.

        Takes the lock itself. It is deliberately *not* called from inside
        another locked region: an earlier version had ``start_run`` call this
        while holding the same non-reentrant lock, which deadlocked the very
        first run - a silent hang rather than an error, because every caller
        was innocent.
        """
        with self._lock:
            if self._run is None:
                raise RuntimeError(
                    f"cannot record {event_type.value}: no run is open. "
                    f"Evidence emitted outside a run has no place in a chain.")
            run = self._run
            attributes = fields.pop("attributes", None) or {}
            defaults = {
                "run_id": run.run_id,
                "sequence": run.event_count,
                "recorded_at": self._clock(),
                "policy_id": run.policy_id,
                "policy_hash": run.policy_hash,
                "logical_plan_hash": run.logical_plan_hash,
                "physical_plan_hash": run.physical_plan_hash,
                "target_id": run.target_id,
                "aar_version": run.aar_version,
            }
            defaults.update(fields)
            # The subject is a run fact and cannot be overridden per event.
            # Allowing ``subject=None`` through would let any emitter strip
            # the identity from the record of what they did - which is
            # precisely the record an auditor reads first.
            defaults["subject"] = run.subject
            event = EvidenceEvent(event_type=event_type, **defaults,
                                  attributes=attributes)
            event = event.sealed(
                self._recorded[-1].event_hash if self._recorded else GENESIS_HASH)
            self._recorded.append(event)
            run.event_count += 1
            run.final_hash = event.event_hash
            if run.event_count == 1:
                run.first_hash = event.event_hash
            if self._on_event is not None:
                self._on_event(event)
            return event

    # ----------------------------------------------------------------- read
    def events(self) -> tuple[EvidenceEvent, ...]:
        """Every event of the current or most recent run, in order."""
        with self._lock:
            return tuple(self._recorded)

    def of_type(self, *types: EventType) -> list[EvidenceEvent]:
        wanted = set(types)
        return [e for e in self.events() if e.event_type in wanted]

    def verify(self) -> "VerificationResult":
        """Check the in-memory chain without touching storage."""
        return verify_chain(self.events())


def verify_chain(events: Sequence[EvidenceEvent]) -> VerificationResult:
    """Verify a run's chain: each event's own hash, and the links between.

    Two failure modes are reported separately because they mean different
    things. An event failing :meth:`EvidenceEvent.verify` was *altered*. An
    event whose ``previous_hash`` does not match its predecessor was
    *reordered, removed or replaced* - the kind of tampering that leaves every
    event individually intact, and the kind a per-event check alone would miss
    entirely.
    """
    previous = GENESIS_HASH
    for expected_sequence, event in enumerate(events):
        if event.sequence != expected_sequence:
            return VerificationResult(
                False, len(events), event.sequence,
                f"expected sequence {expected_sequence}; there is a gap, which "
                f"means an event was removed")
        if not event.verify():
            return VerificationResult(
                False, len(events), event.sequence,
                "this event's contents do not match its own hash; it was "
                "altered after being recorded")
        if event.previous_hash != previous:
            return VerificationResult(
                False, len(events), event.sequence,
                "this event does not follow the one before it; the sequence "
                "was reordered or an earlier event was replaced")
        previous = event.event_hash
    return VerificationResult(True, len(events))


def event_fields(outcome: Any) -> dict[str, Any]:
    """Project a :class:`~aar.runtime.NodeOutcome` into evidence fields.

    The adapter between the executor's working representation and the
    contract. Kept here rather than in the executor so the executor does not
    grow a dependency on the audit package, and so the mapping is one auditable
    function rather than a scattering of keyword arguments at every call site.

    Note what is *not* carried: ``elapsed_ms`` as a float. It becomes
    ``duration_us`` as an integer, because a float cannot be canonically hashed
    and because a duration is not the fact an auditor needs anyway.
    """
    elapsed = getattr(outcome, "elapsed_ms", None)
    return {
        "graph_node_id": getattr(outcome, "node_id", "") or "",
        "operation_id": getattr(outcome, "operation_id", "") or "",
        "engine_planned": getattr(outcome, "engine_requested", "") or "",
        "engine_used": getattr(outcome, "engine_used", "") or "",
        "rows_in": tuple(getattr(outcome, "input_rows", ()) or ()),
        "rows_out": getattr(outcome, "rows_out", None),
        "duration_us": (int(round(float(elapsed) * 1000))
                        if elapsed is not None else None),
        "degradation": ("degraded" if getattr(outcome, "degraded", False)
                        else ""),
    }


def rows_removed(before: int | None, after: int | None) -> int | None:
    """Rows a restriction removed, or ``None`` if either count is unknown.

    ``None`` rather than a guess: "we cannot say how many rows were excluded"
    and "zero rows were excluded" are very different sentences in a compliance
    answer, and only one of them is true when a count was not taken.
    """
    if before is None or after is None:
        return None
    return max(0, before - after)
