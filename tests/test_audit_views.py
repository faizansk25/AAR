"""Authorized views over immutable evidence.

The rule these tests defend: **the evidence is complete and the *view* is
partial, never the other way round.** A redacted field is present in the chain
with its true value; only the rendering withholds it. Redacting at record time
would produce a trail silently incomplete for everyone except whoever can read
the raw file.

The counting tests exist because of a specific attack. "614 of 18,241 rows" is
itself a disclosure, and an analyst who can vary a filter and compare counts
recovers protected rows one at a time - 614 for A, 613 for A-with-filter-X,
and exactly one row satisfies X. That differencing attack is why the counts are
a capability rather than a default.
"""

from __future__ import annotations

import pytest

from aar.audit import (Capability, EvidenceAuthorizer, EventType,
                       EvidenceEvent, EvidenceRecorder, FieldState,
                       FieldValue, RestrictionState, Subject, project,
                       summarise_withheld, trust_boundary_notice)
from aar.governance import Subject as GovernanceSubject

ANALYST_CAPS = (Capability.AUDIT_READ,)
COMPLIANCE_CAPS = (Capability.AUDIT_READ,
                   Capability.AUDIT_READ_RESTRICTED_COUNTS,
                   Capability.AUDIT_READ_OTHER_SUBJECTS,
                   Capability.AUDIT_READ_POLICY_DETAILS,
                   Capability.AUDIT_VERIFY_INTEGRITY)


def _viewer(name: str, *caps: Capability) -> EvidenceAuthorizer:
    return EvidenceAuthorizer(
        viewer=Subject(subject_id=name, identity_provider="entra"),
        capabilities=frozenset(caps))


ANALYST = _viewer("dana", *ANALYST_CAPS)
COMPLIANCE = _viewer("priya", *COMPLIANCE_CAPS)


@pytest.fixture
def rls_event():
    """One recorded restriction, carrying the full counts in its attributes."""
    recorder = EvidenceRecorder()
    recorder.start_run(
        GovernanceSubject(name="dana", roles=frozenset({"analyst"})),
        policy_id="finance-2026-10", policy_hash="abc123")
    event = recorder.emit(
        EventType.ROWS_RESTRICTED, rule_id="rls.finance.eu",
        engine_used="arrow", rows_in=(18_241,), rows_out=614,
        restriction_state=RestrictionState.APPLIED,
        attributes={"source": "employees", "rows_before": 18_241,
                    "rows_removed": 17_627,
                    "applied_before_aggregation": True,
                    "predicate": "department = 'FINANCE'"})
    recorder.finish_run()
    return event


def _run_started(state: str) -> EvidenceEvent:
    """An event carrying a determination and no restriction rule attached.

    ``start_run`` already emits a ``run.started`` of its own, so this records
    a second event and returns that one. Using a distinct event type keeps the
    fixture honest: the point is a determination *without* a ``rows.restricted``
    event, which is precisely the case that must not be read as "no RLS".
    """
    recorder = EvidenceRecorder()
    recorder.start_run(
        GovernanceSubject(name="dana"), policy_id="corp", policy_hash="h")
    recorder.emit(EventType.POLICY_ALLOWED, restriction_state=state,
                  reason="policy evaluated")
    recorder.finish_run()
    return recorder.events()[1]


class TestTheStoreKeepsTheExactFacts:
    def test_counts_are_recorded_not_redacted(self, rls_event):
        """Redaction belongs to the view, never to the record.

        Redacting here would leave a compliance viewer nothing to see and the
        audit trail permanently incomplete for everyone.
        """
        assert rls_event.attributes["rows_before"] == 18_241
        assert rls_event.attributes["rows_removed"] == 17_627
        assert rls_event.verify()

    def test_producing_a_view_does_not_touch_the_event(self, rls_event):
        project(rls_event, ANALYST)
        assert rls_event.attributes["rows_before"] == 18_241
        assert rls_event.verify(), "a view must not alter its source"


class TestTheAnalystView:
    def test_counts_are_withheld(self, rls_event):
        view = project(rls_event, ANALYST)
        for name in ("rows_before", "rows_after", "rows_removed"):
            assert view.visible[name].state is FieldState.REDACTED, name

    def test_the_rule_name_is_visible(self, rls_event):
        """Knowing *that* you were restricted is not a disclosure."""
        got = project(rls_event, ANALYST).visible["rule"].value
        assert got == "rls.finance.eu"

    def test_the_source_is_visible(self, rls_event):
        got = project(rls_event, ANALYST).visible["source"].value
        assert got == "employees"

    def test_that_rls_ran_before_aggregation_is_visible(self, rls_event):
        got = project(rls_event, ANALYST).visible
        assert got["applied_before_aggregation"].value is True

    def test_the_predicate_is_withheld_without_the_capability(self, rls_event):
        view = project(rls_event, ANALYST)
        assert view.visible["predicate"].state is FieldState.REDACTED

    def test_the_predicate_is_visible_with_the_capability(self, rls_event):
        got = project(rls_event, COMPLIANCE).visible["predicate"].value
        assert got == "department = 'FINANCE'"

    def test_the_underlying_proof_is_still_stated(self, rls_event):
        """The view is a projection, not a substitute for the evidence."""
        view = project(rls_event, ANALYST)
        assert view.source_event_id == rls_event.event_id
        assert view.source_event_hash == rls_event.event_hash
        assert view.chain_verified is True

    def test_withheld_fields_are_named_not_silently_dropped(self, rls_event):
        """A missing key and a withheld key must be distinguishable."""
        view = project(rls_event, ANALYST)
        assert "rows_before" in view.redacted_fields
        assert "rows_before" in view.visible


class TestTheComplianceView:
    def test_counts_are_visible(self, rls_event):
        view = project(rls_event, COMPLIANCE)
        assert view.visible["rows_before"].value == 18_241
        assert view.visible["rows_removed"].value == 17_627

    def test_one_trail_two_authorized_readings(self, rls_event):
        """Not two trails: both views come from the same event."""
        analyst = project(rls_event, ANALYST)
        compliance = project(rls_event, COMPLIANCE)
        assert analyst.source_event_hash == compliance.source_event_hash
        assert analyst.chain_verified == compliance.chain_verified
class TestRedactionIsNotNone:
    def test_a_withheld_field_is_not_bare_none(self, rls_event):
        """None would be ambiguous between three different situations."""
        got = project(rls_event, ANALYST).visible["rows_removed"]
        assert got.value is None
        assert got.state is FieldState.REDACTED

    def test_redacted_and_unknown_are_different(self, rls_event):
        view = project(rls_event, ANALYST)
        assert view.visible["rows_removed"].state is FieldState.REDACTED
        # `degradation` was never recorded at all - a third state.
        assert view.visible["degradation"].state is FieldState.UNKNOWN

    def test_a_measured_zero_is_still_visible(self):
        """Zero is a measurement; withholding it would be indistinguishable
        from never having measured it."""
        zero = FieldValue.measured(0)
        assert zero.is_visible
        assert zero.to_json() == {"state": "measured", "value": 0}

    def test_the_json_form_names_the_reason(self, rls_event):
        payload = project(rls_event, ANALYST).to_json()
        assert payload["visible"]["rows_removed"] == {
            "state": "redacted", "reason": "insufficient_privilege"}
        assert payload["redacted_fields"]

    def test_an_absent_field_is_unknown_not_measured(self, rls_event):
        """An empty string means "not applicable", so it is not evidence."""
        got = project(rls_event, ANALYST).visible["engine_planned"]
        assert got.state is FieldState.UNKNOWN


class TestCapabilitiesNotRoles:
    def test_capabilities_are_not_role_names(self):
        """A deployment maps its own vocabulary onto these."""
        names = {c.value for c in Capability}
        for role in ("compliance", "auditor", "admin",
                     "data_governance_admin"):
            assert role not in names

    def test_one_capability_does_not_grant_another(self, rls_event):
        """Holding AUDIT_READ must not imply restricted counts."""
        viewer = _viewer("dana", *ANALYST_CAPS)
        assert viewer.may_read(rls_event)
        assert not viewer.may_see_restricted_counts()

    def test_counts_capability_does_not_grant_other_subjects(self, rls_event):
        viewer = _viewer("priya", Capability.AUDIT_READ,
                         Capability.AUDIT_READ_RESTRICTED_COUNTS)
        assert not viewer.may_read(rls_event)


class TestViewerIsNotSubject:
    def test_reading_your_own_run_is_the_base_case(self, rls_event):
        assert _viewer("dana", *ANALYST_CAPS).may_read(rls_event)

    def test_reading_another_subject_needs_a_capability(self, rls_event):
        assert not _viewer("priya", *ANALYST_CAPS).may_read(rls_event)
        assert _viewer("priya", *COMPLIANCE_CAPS).may_read(rls_event)

    def test_a_viewer_with_no_read_capability_sees_nothing(self, rls_event):
        view = project(rls_event, _viewer("dana"))
        assert view.visible["access"] == "not permitted for this viewer"
        assert view.redacted_fields == ("event contents",)

    def test_a_denied_view_still_reports_chain_integrity(self, rls_event):
        """Withholding content must not become "everything is fine"."""
        view = project(rls_event, _viewer("dana"), chain_verified=False)
        assert view.chain_verified is False


class TestRendering:
    def test_the_analyst_rendering_names_no_counts(self, rls_event):
        text = project(rls_event, ANALYST).render()
        assert "18241" not in text and "17627" not in text
        assert "restricted audit detail" in text
        assert "VALID" in text

    def test_the_compliance_rendering_has_the_figures(self, rls_event):
        text = project(rls_event, COMPLIANCE).render()
        assert "18241" in text and "17627" in text

    def test_a_withheld_predicate_does_not_leak(self, rls_event):
        assert "FINANCE'" not in project(rls_event, ANALYST).render()


def test_the_trust_boundary_is_stated_not_hidden():
    """AAR is local-first, so view-level authorization is not isolation."""
    notice = trust_boundary_notice()
    assert "NOT strong isolation" in notice
    assert "filesystem" in notice
class TestAbsenceOfAnEventIsNotAbsenceOfRestriction:
    """The same lesson as ``unknown != zero != redacted``, one level up.

    Concluding "no RLS was applied" from the absence of a ``rows.restricted``
    event cannot distinguish two opposite truths: policy evaluated and no rule
    matched, versus governance never being evaluated at all. The determination
    is therefore recorded, not inferred.
    """

    def test_applied_is_reported_as_applied(self, rls_event):
        view = project(rls_event, ANALYST)
        assert view.restriction_state == RestrictionState.APPLIED
        assert view.restriction_state.is_determined

    def test_not_applicable_is_not_the_same_as_unknown(self):
        event = _run_started(RestrictionState.NOT_APPLICABLE)
        view = project(event, ANALYST)
        assert view.restriction_state == RestrictionState.NOT_APPLICABLE
        assert view.restriction_state != RestrictionState.UNKNOWN
        assert view.restriction_state.is_determined

    def test_not_evaluated_is_not_reported_as_unrestricted(self):
        """Governance was bypassed, so AAR must not claim the run was clean."""
        event = _run_started(RestrictionState.NOT_EVALUATED)
        view = project(event, ANALYST)
        assert view.restriction_state == RestrictionState.NOT_EVALUATED
        assert view.restriction_state != RestrictionState.NOT_APPLICABLE

    def test_an_undetermined_run_says_unknown(self):
        """An interrupted run, or one predating the field, must not guess."""
        event = _run_started("")
        assert project(event, ANALYST).restriction_state == \
            RestrictionState.UNKNOWN

    def test_the_state_is_not_inferred_from_event_absence(self):
        """This event has no rule attached, and says so explicitly."""
        event = _run_started(RestrictionState.NOT_APPLICABLE)
        assert not event.rule_id
        assert project(event, ANALYST).restriction_state == \
            RestrictionState.NOT_APPLICABLE

    def test_the_state_is_part_of_the_hash(self):
        """A rewritten determination must break verification."""
        from dataclasses import replace as dc_replace

        event = _run_started(RestrictionState.NOT_APPLICABLE)
        assert event.verify()
        assert not dc_replace(
            event, restriction_state=RestrictionState.UNKNOWN).verify()

    def test_only_unknown_is_undetermined(self):
        for state in (RestrictionState.APPLIED, RestrictionState.NOT_APPLICABLE,
                      RestrictionState.NOT_EVALUATED):
            assert state.is_determined, state
        assert not RestrictionState.UNKNOWN.is_determined


class TestIntegrityStatusIsSafeButDiagnosticsAreNot:
    def test_any_reader_sees_valid_or_broken(self, rls_event):
        """Status reveals no protected business data, and suppressing it would
        only teach readers to distrust the word VALID."""
        assert project(rls_event, ANALYST, chain_verified=True).\
            chain_verified is True

    def test_the_verify_capability_gates_the_diagnostics(self, rls_event):
        analyst = project(rls_event, ANALYST, chain_verified=False)
        auditor = project(rls_event, COMPLIANCE, chain_verified=False)
        assert analyst.integrity_detail_permitted is False
        assert auditor.integrity_detail_permitted is True

    def test_a_break_without_permission_names_no_diagnostics(self, rls_event):
        text = project(rls_event, ANALYST, chain_verified=False).render()
        assert "BROKEN" in text
        assert "audit.verify_integrity" in text

    def test_the_capability_is_not_decorative(self):
        """It gates something real rather than existing and doing nothing."""
        assert _viewer("dana", *ANALYST_CAPS).may_verify() is False
        assert _viewer("priya", *COMPLIANCE_CAPS).may_verify() is True


class TestTheWithholdingMessageNamesNoRole:
    def test_the_human_message_names_no_role(self):
        message = summarise_withheld(("rows_before", "rows_removed"))
        for word in ("compliance", "auditor", "admin", "role"):
            assert word not in message.lower()

    def test_the_human_message_names_no_internal_field(self):
        """Enumerating them reveals what privileged evidence exists."""
        message = summarise_withheld(("rows_before", "restricted_group_count",
                                      "suppression_reason"))
        assert "restricted_group_count" not in message
        assert "suppression_reason" not in message

    def test_the_human_message_says_detail_is_restricted(self):
        message = summarise_withheld(("rows_before",))
        assert "restricted" in message
        assert "current access" in message

    def test_nothing_withheld_means_nothing_announced(self):
        assert summarise_withheld(()) == ""

    def test_the_json_still_names_every_field(self, rls_event):
        """Clients need deterministic semantics even when people do not."""
        view = project(rls_event, ANALYST)
        assert "rows_before" in view.redacted_fields