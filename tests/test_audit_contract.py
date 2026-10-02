"""The evidence contract: canonical form, immutability, and the hash chain.

This is the layer the whole audit subsystem hashes against, so its tests are
about *tampering* rather than about features. Each property corresponds to a
way a "this evidence has not been modified" claim could be quietly false:

* an encoding that is not reproducible makes the claim unverifiable;
* ``True`` encoding as ``1`` lets a forged field hash to the original's value;
* a string encoding without a length prefix lets content imitate structure;
* a chain that only checks each event in isolation misses reordering and
  deletion, which leave every event individually intact.

The deadlock in :meth:`EvidenceRecorder.start_run` is also pinned here. It was
a real bug: the first version called :meth:`~EvidenceRecorder.emit` while
holding the same non-reentrant lock, so *every* run hung silently and no error
was raised anywhere.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from aar.audit import (EVIDENCE_SCHEMA_VERSION, CanonicalisationError,
                       EventType, EvidenceEvent, EvidenceRecorder, Subject,
                       canonical_bytes, chain_digest, digest, stamp,
                       verify_chain)
from aar.governance import Subject as GovernanceSubject


# ------------------------------------------------------------------ canonical
class TestCanonicalForm:
    def test_the_same_value_always_encodes_identically(self):
        assert canonical_bytes({"a": 1}) == canonical_bytes({"a": 1})

    def test_key_order_does_not_change_the_hash(self):
        assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})

    def test_a_boolean_is_not_an_integer(self):
        """``True == 1`` in Python, so a naive encoder lets a forged value
        hash to the same digest as the original."""
        assert digest({"x": True}) != digest({"x": 1})
        assert digest({"x": False}) != digest({"x": 0})

    def test_none_is_not_the_empty_string(self):
        assert digest({"x": None}) != digest({"x": ""})

    def test_a_string_cannot_impicate_structure(self):
        """A length prefix stops ``"a:1"`` from parsing as ``a -> 1``."""
        assert digest({"a": "1"}) != digest({"a": 1})
        assert digest(["a;b"]) != digest(["a", "b"])

    def test_nesting_is_unambiguous(self):
        assert digest([[1, 2], [3]]) != digest([[1], [2, 3]])

    def test_floats_are_refused(self):
        with pytest.raises(CanonicalisationError):
            canonical_bytes({"x": 1.5})

    def test_the_float_refusal_explains_the_alternative(self):
        with pytest.raises(CanonicalisationError) as exc:
            canonical_bytes(1.5)
        assert "duration_us" in str(exc.value)

# --------------------------------------------------------------------- events
class TestEvidenceEvent:
    def _event(self, **kw):
        defaults = {
            "run_id": "run_1", "sequence": 0,
            "event_type": EventType.RUN_STARTED,
            "recorded_at": "2026-10-01T00:00:00.000Z",
            "event_id": "evt_1",
        }
        defaults.update(kw)
        return EvidenceEvent(**defaults)

    def test_an_event_is_immutable(self):
        with pytest.raises(FrozenInstanceError):
            self._event().rows_out = 5

    def test_attributes_cannot_be_mutated_through_the_original_dict(self):
        attributes = {"source": "employees"}
        event = self._event(attributes=attributes)
        attributes["source"] = "payments"
        assert event.attributes["source"] == "employees"

    def test_a_set_in_attributes_is_refused(self):
        """Set order is not stable, so the same event would hash twice."""
        with pytest.raises(TypeError):
            self._event(attributes={"roles": {"a", "b"}})

    def test_the_schema_version_is_recorded(self):
        assert self._event().schema_version == EVIDENCE_SCHEMA_VERSION

    def test_a_sealed_event_verifies(self):
        assert self._event().sealed("0" * 64).verify()

    def test_an_unsealed_event_does_not_verify(self):
        assert not self._event().verify()

    def test_altering_any_field_breaks_verification(self):
        event = self._event(rows_out=10).sealed("0" * 64)
        assert event.verify()
        assert not replace(event, rows_out=11).verify()

    def test_altering_the_reason_breaks_verification(self):
        """The most tempting edit in an audit trail is rewriting a reason."""
        event = self._event(reason="because").sealed("0" * 64)
        assert not replace(event, reason="because I said so").verify()

    def test_the_schema_version_is_hashed_not_merely_stored(self):
        """Evidence from one version must not verify as another."""
        event = self._event().sealed("0" * 64)
        assert not replace(event, schema_version="aar-evidence-v0").verify()

    def test_the_row_carries_indexed_columns_and_a_payload(self):
        row = self._event(rule_id="rls.x").sealed("0" * 64).row()
        assert row["run_id"] == "run_1"
        assert row["rule_id"] == "rls.x"
        assert row["payload"]["rule_id"] == "rls.x"


# ------------------------------------------------------------------- subject
class TestSubjectIdentity:
    def test_a_governance_subject_is_accepted(self):
        identity = Subject.coerce(
            GovernanceSubject(name="dana", roles=frozenset({"analyst"})))
        assert identity.subject_id == "dana"
        assert identity.roles == ("analyst",)
        assert identity.identity_provider == "local"

# ------------------------------------------------------------------ recorder
class TestEvidenceRecorder:
    def _run(self, **kw):
        recorder = EvidenceRecorder()
        defaults = {
            "subject": GovernanceSubject(name="dana",
                                         roles=frozenset({"analyst"})),
            "policy_id": "finance-2026-10", "policy_hash": "abc123",
        }
        defaults.update(kw)
        recorder.start_run(**defaults)
        return recorder

    def test_a_run_records_its_opening_and_closing(self):
        recorder = self._run()
        context = recorder.finish_run()
        types = [e.event_type for e in recorder.events()]
        assert types[0] is EventType.RUN_STARTED
        assert types[-1] is EventType.RUN_FINISHED
        assert context.status == "ok"

    def test_sequence_is_assigned_by_the_recorder(self):
        recorder = self._run()
        recorder.emit(EventType.NODE_EXECUTED)
        recorder.emit(EventType.NODE_EXECUTED)
        assert [e.sequence for e in recorder.events()] == [0, 1, 2]

    def test_run_level_facts_are_attached_to_every_event(self):
        """A node event with no subject is exactly the hole nobody notices."""
        recorder = self._run()
        event = recorder.emit(EventType.NODE_EXECUTED)
        assert event.subject.subject_id == "dana"
        assert event.policy_id == "finance-2026-10"

    def test_a_caller_may_not_strip_the_subject(self):
        recorder = self._run()
        event = recorder.emit(EventType.NODE_EXECUTED, subject=None)
        assert event.subject.subject_id == "dana"

    def test_emitting_outside_a_run_is_refused(self):
        with pytest.raises(RuntimeError) as exc:
            EvidenceRecorder().emit(EventType.NODE_EXECUTED)
        assert "no run is open" in str(exc.value)

    def test_two_runs_at_once_are_refused(self):
        with pytest.raises(RuntimeError):
            self._run().start_run()

    def test_starting_a_run_does_not_deadlock(self):
        """Regression: ``start_run`` used to call ``emit`` under the same lock.

        Every run hung, silently, with no error - the worst shape a bug can
        take, because every caller was innocent.
        """
        context = EvidenceRecorder().start_run(
            subject=GovernanceSubject(name="dana"))
        assert context.event_count == 1

    def test_a_callback_receives_every_event(self):
        seen = []
        recorder = EvidenceRecorder(on_event=seen.append)
        recorder.start_run()
        recorder.emit(EventType.NODE_EXECUTED)
        recorder.finish_run()
        assert len(seen) == 3

    def test_a_failed_run_records_its_status(self):
        recorder = self._run()
        context = recorder.finish_run(status="failed", reason="out of memory")
        assert context.status == "failed"
        assert recorder.events()[-1].attributes["status"] == "failed"

    def test_timestamps_are_iso_utc(self):
        assert stamp().endswith("Z")
        assert stamp()[4] == "-"


# ------------------------------------------------------------------- the chain
class TestTheChainDetectsTampering:
    def _chain(self):
        recorder = EvidenceRecorder()
        recorder.start_run(subject=GovernanceSubject(name="dana"))
        recorder.emit(EventType.ROWS_RESTRICTED, rule_id="rls.analyst",
                      rows_in=(84_291,), rows_out=(12_415,))
        recorder.emit(EventType.NODE_EXECUTED, engine_used="arrow",
                      rows_in=(12_415,), rows_out=12)
        recorder.finish_run()
        assert recorder.verify().valid
        return list(recorder.events())

    def test_an_untouched_chain_verifies(self):
        assert verify_chain(self._chain()).valid

    def test_an_altered_event_is_detected(self):
        events = self._chain()
        forged = replace(events[1], rows_out=999_999)
        result = verify_chain(events[:1] + [forged] + events[2:])
        assert not result.valid
        assert result.first_bad_sequence == 1
        assert "altered" in result.reason

    def test_a_rewritten_reason_is_detected(self):
        """The most tempting edit in an audit trail."""
        events = self._chain()
        forged = replace(events[1], reason="approved by nobody")
        assert not verify_chain(events[:1] + [forged] + events[2:]).valid

    def test_a_removed_event_is_detected(self):
        events = self._chain()
        result = verify_chain([events[0], events[1], events[3]])
        assert not result.valid
        assert "removed" in result.reason

    def test_a_reordered_chain_is_detected(self):
        events = self._chain()
        assert not verify_chain(
            [events[0], events[2], events[1], events[3]]).valid

    def test_a_run_spliced_into_another_is_detected_by_the_link(self):
        """Each replacement is internally valid; the break is in the links.

        This is the case a per-event check alone would miss entirely, and it
        is why the chain carries ``previous_hash`` at all.
        """
        first, second = self._chain(), self._chain()
        result = verify_chain([first[0], second[1], second[2], second[3]])
        assert not result.valid
        assert "does not follow" in result.reason

    def test_the_result_says_where_it_broke(self):
        events = self._chain()
        forged = replace(events[2], engine_used="duckdb")
        result = verify_chain(events[:2] + [forged] + events[3:])
        assert result.first_bad_sequence == 2
        assert "BROKEN at event 2" in result.render()

    def test_a_valid_chain_renders_as_valid(self):
        assert "VALID" in verify_chain(self._chain()).render()
    def test_roles_are_sorted_because_set_order_is_not_stable(self):
        identity = Subject.coerce(
            GovernanceSubject(name="d", roles=frozenset({"x", "a"})))
        assert identity.roles == ("a", "x")

    def test_an_identity_may_name_its_provider(self):
        """AAR does not authenticate; it records what vouched for the subject."""
        identity = Subject(subject_id="dana@corp", roles=("analyst",),
                           identity_provider="entra")
        assert identity.as_payload()["identity_provider"] == "entra"

    def test_an_unasserted_provider_is_empty_not_assumed(self):
        assert Subject(subject_id="dana").identity_provider == ""
    def test_raw_bytes_are_refused(self):
        with pytest.raises(CanonicalisationError):
            canonical_bytes(b"payload")

    def test_a_non_string_key_is_refused(self):
        with pytest.raises(CanonicalisationError):
            canonical_bytes({1: "a"})

    def test_unicode_is_normalised(self):
        """NFC, so the same visible text written two ways is one value."""
        assert digest("café") == digest("café")

    def test_an_unbounded_integer_is_refused(self):
        with pytest.raises(CanonicalisationError):
            canonical_bytes(2 ** 70)

    def test_the_digest_is_domain_separated(self):
        """A digest here must not be replayable as an identity elsewhere."""
        assert digest({"a": 1}) != chain_digest("0" * 64, {"a": 1})