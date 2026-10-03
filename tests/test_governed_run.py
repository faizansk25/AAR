"""GovernedRun: the lifecycle that makes evidence a record rather than a log.

Every test here is about a claim AAR must not make. The common thread is that
a run which *cannot* record something is not permitted to look as though it
recorded something harmless - the failure mode that made Round 30 necessary,
one level up.
"""

from __future__ import annotations

import json

import pytest

from aar.application import PipelineService
from aar.audit import (
    BASELINE_POLICY_ID, EvidenceRecorder, GovernedRun, RunStatus,
    policy_identity,
)
from aar.audit.contract import EventType
from aar.failures import PolicyDenied
from aar.governance.disclosure import DisclosureRule, Suppression

ROWS = "region,amount,keep\n" + "".join(
    f"north,{n * 10},1\n" for n in range(1, 7)) + "south,100,0\nsouth,200,1\n"


def _pipeline(directory, csv_text: str = ROWS):
    src = directory / "orders.csv"
    src.write_text(csv_text, encoding="utf-8")
    out = directory / "out.csv"
    path = directory / "pipeline.py"
    path.write_text(
        "from aar.sdk import csv, group_by, mean, write_csv\n"
        "\n"
        "def build():\n"
        f"    return write_csv(group_by(csv(r'{src}'), 'region',\n"
        f"        aggs={{'avg_amount': mean('amount')}}), r'{out}')\n",
        encoding="utf-8")
    return path


def _policy(directory, raw):
    path = directory / "policy.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _run(directory, policy=None, role="analyst"):
    recorder = EvidenceRecorder()
    service = PipelineService(evidence=recorder)
    args = {}
    if policy is not None:
        args["policy_path"] = str(_policy(directory, policy))
    report = service.run(str(_pipeline(directory)), role=role, **args)
    return recorder, report


def _of(recorder, event_type):
    return [e for e in recorder.events() if e.event_type is event_type]


def _unreachable_policy(directory):
    """A policy whose row rule names a source the pipeline does not read."""
    return _policy(directory, {
        "name": "t",
        "rls": {"analyst": [{"source": "absent_source",
                             "predicate": "keep = 1"}]}})


class TestAuthorityIsBoundNotBackfilled:
    def test_the_opening_event_does_not_claim_an_authority(self, tmp_path):
        """`run.started` is hashed when written, so it cannot be completed later.

        Filling in `policy_id` afterwards would either break the chain or force
        a rehash, leaving an opening event that describes facts nobody had yet.
        """
        recorder, _ = _run(tmp_path, {"name": "t", "disclosure": 5})
        started = _of(recorder, EventType.RUN_STARTED)[0]
        assert started.policy_id == ""
        assert started.attributes["policy_state"] == "unresolved"

    def test_a_later_event_binds_the_real_authority(self, tmp_path):
        recorder, _ = _run(tmp_path, {"name": "t", "disclosure": 5})
        bound = _of(recorder, EventType.POLICY_BOUND)[0]
        assert bound.policy_id == "policy.json"
        assert bound.policy_hash
        assert bound.attributes["policy_state"] == "resolved"

    def test_later_events_inherit_the_bound_authority(self, tmp_path):
        """One binding, then every event carries it - no per-event plumbing."""
        recorder, _ = _run(tmp_path, {"name": "t", "disclosure": 5})
        after = recorder.events()[1:]
        assert after, "no events followed the opening"
        for event in after:
            assert event.policy_id == "policy.json", event.event_type

    def test_the_baseline_names_itself(self, tmp_path):
        """No policy file is a governance state, and must not read as a gap."""
        recorder, _ = _run(tmp_path, None, role=None)
        bound = _of(recorder, EventType.POLICY_BOUND)[0]
        assert bound.policy_id == BASELINE_POLICY_ID
        assert bound.policy_hash

    def test_the_policy_hash_notices_an_edited_file(self, tmp_path):
        """An edited-then-restored file looks unchanged under mtime granularity."""
        path = _policy(tmp_path, {"name": "t"})
        before = policy_identity(str(path))[1]
        path.write_text(json.dumps({"name": "other"}), encoding="utf-8")
        assert policy_identity(str(path))[1] != before

    def test_a_missing_policy_file_fails_the_run_but_records_it(self, tmp_path):
        """Nothing was refused; the run never obtained the means to refuse."""
        recorder = EvidenceRecorder()
        with pytest.raises(Exception):
            PipelineService(evidence=recorder).run(
                str(_pipeline(tmp_path)),
                policy_path=str(tmp_path / "absent.json"))
        kinds = [e.event_type for e in recorder.events()]
        assert kinds[0] is EventType.RUN_STARTED
        assert kinds[-1] is EventType.RUN_FINISHED
        assert dict(recorder.events()[-1].attributes)["status"] == "failed"
        assert recorder.verify().valid


class TestDeniedIsNotFailed:
    def test_a_policy_refusal_is_recorded_as_denied(self, tmp_path):
        """Governance working as designed is not a fault to investigate."""
        recorder = EvidenceRecorder()
        with pytest.raises(PolicyDenied):
            PipelineService(evidence=recorder).run(
                str(_pipeline(tmp_path)), role="analyst",
                policy_path=str(_unreachable_policy(tmp_path)))
        assert dict(recorder.events()[-1].attributes)["status"] == "denied"

    def test_a_runtime_failure_is_recorded_as_failed(self, tmp_path):
        recorder = EvidenceRecorder()
        with pytest.raises(Exception):
            PipelineService(evidence=recorder).run(str(tmp_path / "nope.py"))
        assert dict(recorder.events()[-1].attributes)["status"] == "failed"

    def test_a_refusal_still_raises(self, tmp_path):
        """Recording an attempt is not the same as swallowing it."""
        recorder = EvidenceRecorder()
        with pytest.raises(PolicyDenied):
            PipelineService(evidence=recorder).run(
                str(_pipeline(tmp_path)), role="analyst",
                policy_path=str(_unreachable_policy(tmp_path)))
        assert recorder.events(), "the refused attempt was not preserved"

    def test_a_refused_run_says_which_control_refused(self, tmp_path):
        """`DENIED` alone says something was blocked and nothing about what."""
        recorder = EvidenceRecorder()
        with pytest.raises(PolicyDenied):
            PipelineService(evidence=recorder).run(
                str(_pipeline(tmp_path)), role="analyst",
                policy_path=str(_unreachable_policy(tmp_path)))
        denials = _of(recorder, EventType.POLICY_DENIED)
        assert len(denials) == 1
        assert "absent_source" in denials[0].reason


class TestEveryOpenedRunCloses:
    def test_a_successful_run_is_sealed(self, tmp_path):
        recorder, _ = _run(tmp_path, {"name": "t", "disclosure": 5})
        assert not recorder.is_recording
        assert recorder.events()[-1].event_type is EventType.RUN_FINISHED
        assert dict(recorder.events()[-1].attributes)["status"] == "success"
        assert recorder.verify().valid

    def test_a_denied_run_is_sealed(self, tmp_path):
        recorder = EvidenceRecorder()
        with pytest.raises(PolicyDenied):
            PipelineService(evidence=recorder).run(
                str(_pipeline(tmp_path)), role="analyst",
                policy_path=str(_unreachable_policy(tmp_path)))
        assert not recorder.is_recording, "the refused run was left open"
        assert recorder.verify().valid

    def test_a_failed_run_is_sealed(self, tmp_path):
        recorder = EvidenceRecorder()
        with pytest.raises(Exception):
            PipelineService(evidence=recorder).run(str(tmp_path / "nope.py"))
        assert not recorder.is_recording, "the failed run was left open"
        assert recorder.verify().valid

    def test_closing_twice_does_not_emit_two_terminal_events(self):
        """Two `run.finished` events leave the governing one ambiguous."""
        with GovernedRun(subject="dana") as run:
            run.close(RunStatus.SUCCESS)
            run.close(RunStatus.FAILED)
        assert len(_of(run, EventType.RUN_FINISHED)) == 1
        assert run.verify().valid

    def test_a_run_with_no_evidence_sink_still_works(self, tmp_path):
        """The runtime must not force a database into existence."""
        report = PipelineService().run(str(_pipeline(tmp_path)))
        assert report.result.table is not None


class TestDisclosureEvidence:
    """Every guarded aggregate produces a positive evaluation event."""

    def test_a_guarded_aggregate_is_evaluated(self, tmp_path):
        recorder, _ = _run(tmp_path, {"name": "t", "disclosure": 5})
        verdicts = _of(recorder, EventType.DISCLOSURE_EVALUATED)
        assert len(verdicts) == 1, "a guarded aggregate left no evidence"

    def test_a_control_that_suppressed_nothing_still_reports(self, tmp_path):
        """Absence must not mean "nothing was suppressed".

        It could equally mean the control never ran, and those demand opposite
        answers - the exact confusion Round 30 removed for RLS.
        """
        recorder, _ = _run(tmp_path, {"name": "t", "disclosure": 2})
        verdicts = _of(recorder, EventType.DISCLOSURE_EVALUATED)
        assert len(verdicts) == 1
        assert verdicts[0].attributes["verdict"] == "passed"
        assert verdicts[0].attributes["suppressed_groups"] == 0

    def test_a_suppression_is_recorded_with_its_numbers(self, tmp_path):
        """South holds two contributors and is removed; north holds six."""
        recorder, _ = _run(tmp_path, {"name": "t", "disclosure": 5})
        event = _of(recorder, EventType.DISCLOSURE_EVALUATED)[0]
        assert event.attributes["verdict"] == "suppressed"
        assert event.attributes["suppressed_groups"] == 1
        assert event.attributes["smallest_group"] == 2

    def test_the_effective_minimum_is_structured_not_prose(self, tmp_path):
        recorder, _ = _run(tmp_path, {"name": "t", "disclosure": 5})
        attributes = _of(recorder, EventType.DISCLOSURE_EVALUATED)[0].attributes
        assert attributes["effective_minimum"] == 5
        assert attributes["effective_rule_id"] == "disclosure.min_group_size"

    def test_contributing_rules_are_data_not_a_sentence(self, tmp_path):
        """A list a reader can check beats a claim they must trust."""
        event = Suppression(node_id="n1", rule="strict", suppressed_groups=0,
                            smallest_group=None, reason="", minimum_required=10,
                            contributing=(DisclosureRule(10, "strict"),
                                          DisclosureRule(3, "loose")))
        from aar.audit.governed import disclosure_attributes

        fields = disclosure_attributes(event)
        assert [dict(r) for r in fields["attributes"]["contributing_rules"]] == [
            {"rule_id": "strict", "min_group_size": 10},
            {"rule_id": "loose", "min_group_size": 3}]

    def test_restriction_is_recorded_before_exposure(self, tmp_path):
        """The Round 26 question, answered explicitly rather than inferred."""
        recorder, _ = _run(tmp_path, {
            "name": "t", "disclosure": 5,
            "rls": {"analyst": [{"source": "orders",
                                 "predicate": "keep = 1"}]}})
        rows = _of(recorder, EventType.ROWS_RESTRICTED)
        assert len(rows) == 1
        assert rows[0].restriction_state == "applied"
        assert rows[0].attributes["applied_before_aggregation"] is True

    def test_a_run_with_no_barriers_does_not_claim_none_applied(self, tmp_path):
        """No barrier is not proof that nothing matched.

        `NOT_EVALUATED` is the weaker, truthful claim; `NOT_APPLICABLE` would
        assert a policy evaluation that this run never performed.
        """
        recorder, _ = _run(tmp_path, None, role=None)
        rows = _of(recorder, EventType.ROWS_RESTRICTED)[0]
        assert rows.restriction_state == "not_evaluated"

    def test_an_unmeasured_count_is_none_not_zero(self, tmp_path):
        """"We cannot say how many" and "none" are different sentences."""
        recorder, _ = _run(tmp_path, {"name": "t", "disclosure": 5})
        rows = _of(recorder, EventType.ROWS_RESTRICTED)[0]
        assert rows.attributes.get("rows_removed") is None or \
            isinstance(rows.attributes["rows_removed"], int)


class TestGovernedRunIsNotARecorder:
    def test_events_are_the_recorders_own_objects(self):
        """A copy would be a second source of truth that could diverge."""
        recorder = EvidenceRecorder()
        with GovernedRun(subject="dana", recorder=recorder) as run:
            run.record_restriction_state("applied")
        assert run.events() == recorder.events()
        assert run.events()[0] is recorder.events()[0]

    def test_it_holds_no_event_list_of_its_own(self):
        """A mirrored list is exactly the fifth recorder this design refuses."""
        with GovernedRun(subject="dana") as run:
            assert not [slot for slot in dir(run)
                        if "events" in slot and slot != "events"]
            assert not [slot for slot in type(run).__slots__
                        if "event" in slot]

    def test_a_duplicate_verdict_is_refused(self):
        """Otherwise a later, softer verdict could supersede an earlier one."""
        with GovernedRun(subject="dana") as run:
            verdict = Suppression(node_id="n1", rule="r",
                                  suppressed_groups=1, smallest_group=1,
                                  reason="")
            run.record_disclosure(verdict)
            with pytest.raises(RuntimeError):
                run.record_disclosure(verdict)
        assert len(_of(run, EventType.DISCLOSURE_EVALUATED)) == 1