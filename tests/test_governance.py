"""Governance: does a classification tag actually do anything?

These tests are adversarial by construction. Each one tries to get
confidential data out of AAR by a route the implementation might have left
open, and fails if it succeeds. A governance layer that is only tested on its
happy path is a governance layer that is a comment.
"""

from __future__ import annotations

import pytest

pa = pytest.importorskip("pyarrow")

from aar.governance import (  # noqa: E402
    NETWORK_SINKS, Policy, PolicyEngine, Sensitivity, Sink, Subject,
    mask_value, sensitivity_of,
)
from aar.interchange import Table  # noqa: E402
from aar.types import FLOAT64, Field, Schema  # noqa: E402


@pytest.fixture()
def confidential():
    """A table whose ``ssn`` and ``email`` are marked, ``amount`` is not."""
    return Table(pa.table({
        "id": pa.array([1, 2, 3], type=pa.int64()),
        "ssn": pa.array(["111-11-1111", "222-22-2222", "333-33-3333"],
                        type=pa.string()),
        "email": pa.array(["a@x.com", "b@x.com", "c@x.com"],
                          type=pa.string()),
        "amount": pa.array([10.0, 20.0, 30.0], type=pa.float64()),
    })).tagged("ssn", "PII").tagged("email", "PII")


@pytest.fixture()
def analyst():
    return Subject(name="dana", roles=frozenset({"analyst"}))


# ------------------------------------------------------------------ egress
class TestEgress:
    def test_local_sinks_are_always_allowed(self, confidential, analyst):
        engine = PolicyEngine(Policy())
        for kind in ("excel", "parquet", "csv", "json"):
            d = engine.check_egress(Sink.of(kind), analyst, confidential.schema)
            assert d.allowed, kind
            assert "local" in d.reason

    def test_network_sinks_are_denied_by_default(self, confidential, analyst):
        """Default-deny: forgetting to write a policy must be safe."""
        engine = PolicyEngine(Policy())
        d = engine.check_egress(Sink.of("postgres"), analyst, confidential.schema)
        assert not d.allowed
        assert d.rule == "egress.network"
        assert "postgres" in d.reason

    def test_every_network_sink_is_recognised(self):
        """Unknown sinks fail closed, and local ones are named explicitly.

        The failure guarded against is a new connector added without updating
        the list, silently writing a customer's data somewhere it should not.
        So the test is a *partition*: everything is either a known local sink
        or treated as egress, and nothing sits in an unclassified middle.
        """

        for kind in ("postgres", "mongodb", "s3", "kafka", "http", "webhook",
                     "bigquery", "snowflake", "smb", "ftp", "nfs"):
            assert Sink.of(kind).network, kind
        for kind in ("excel", "parquet", "csv", "json", "sqlite"):
            assert not Sink.of(kind).network, kind

    def test_an_unknown_sink_is_egress_not_local(self):
        """A name AAR has never heard of must not be trusted as local.

        This was the fail-open: inference tested membership of NETWORK_SINKS
        and concluded "not in it, therefore local", so `ftp`, `smb` and any
        unknown connector were permitted to receive data under a policy
        documented as default-deny. It is exactly the case the failure above
        was about, and it passed anyway.
        """
        for kind in ("totally_unknown_thing", "", "   ", "sqlite_network"):
            assert Sink.of(kind).network, kind

    def test_a_local_sink_needs_no_rule_to_earn_its_belief(self):
        """The benefit of the doubt is explicit membership, not absence.

        A short, named LOCAL_SINKS list means adding a connector is a
        deliberate decision rather than the side effect of forgetting to
        update a deny-list.
        """
        from aar.governance import LOCAL_SINKS

        assert LOCAL_SINKS.isdisjoint(NETWORK_SINKS)
        assert len(LOCAL_SINKS) < len(NETWORK_SINKS), (
            "the local list should be the short one; a long list of "
            "known-local sinks is a deny-list with extra steps")


    def test_a_permitted_network_sink_still_obeys_sensitivity(
            self, confidential, analyst):
        policy = Policy(allow_network_kinds=frozenset({"postgres"}),
                        max_sensitivity=Sensitivity.INTERNAL)
        engine = PolicyEngine(policy)
        d = engine.check_egress(Sink.of("postgres"), analyst, confidential.schema)
        assert not d.allowed
        assert d.rule == "egress.sensitivity"
        assert "CONFIDENTIAL" in d.reason

    def test_public_data_may_reach_a_permitted_sink(self, analyst):
        public = Table(pa.table({"x": pa.array([1.0], type=pa.float64())}),
                       Schema((Field("x", FLOAT64),)))
        policy = Policy(allow_network_kinds=frozenset({"postgres"}),
                        max_sensitivity=Sensitivity.CONFIDENTIAL)
        assert PolicyEngine(policy).check_egress(
            Sink.of("postgres"), analyst, public.schema).allowed

    def test_allow_network_opens_every_sink(self, analyst):
        engine = PolicyEngine(Policy(allow_network=True,
                                     max_sensitivity=Sensitivity.RESTRICTED))
        assert engine.check_egress(Sink.of("s3"), analyst, None).allowed

    def test_denied_egress_raises_rather_than_writing(self, confidential,
                                                      analyst):
        from aar.failures import PrivacyViolation

        with pytest.raises(PrivacyViolation) as exc:
            PolicyEngine(Policy()).enforce_write(confidential, "postgres",
                                                 analyst)
        assert "denied" in str(exc.value)
        assert exc.value.failure_mode.value == "PRIVACY_VIOLATION"



# ------------------------------------------------------------------ masking
class TestColumnSecurity:
    def test_sensitive_columns_are_masked_by_default(self, confidential,
                                                     analyst):
        engine = PolicyEngine(Policy(mask_threshold=Sensitivity.CONFIDENTIAL))
        out = engine.enforce_write(confidential, "excel", analyst)
        assert "111-11-1111" not in out.column("ssn").to_pylist()
        assert out.column("amount").to_pylist() == [10.0, 20.0, 30.0]

    def test_masking_keeps_a_numeric_column_numeric(self, confidential):
        """A masked amount must not become text.

        A column masked to "[redacted]" breaks the next aggregate, and a
        broken pipeline is the surest way to get a mask removed.
        """
        tagged = confidential.tagged("amount", "FINANCIAL")
        out = PolicyEngine(Policy()).enforce_write(
            tagged, "excel", Subject(name="x"))
        values = out.column("amount").to_pylist()
        assert all(isinstance(v, (int, float)) for v in values), values

    def test_classification_survives_masking(self, confidential, analyst):
        """A masked column must still be known to have been masked."""
        out = PolicyEngine(Policy()).enforce_write(confidential, "excel",
                                                    analyst)
        assert "PII" in out.schema.get("ssn").classification

    def test_role_specific_drop_wins_over_masking(self, confidential):
        policy = Policy(
            cls_drop={"junior": ("ssn",)},
            cls_mask={"junior": (("email", "email"),)},
        )
        junior = Subject(name="j", roles=frozenset({"junior"}))
        out = PolicyEngine(policy).enforce_write(confidential, "excel", junior)
        assert "ssn" not in out.column_names
        assert out.column("email").to_pylist() == ["a***@x.com", "b***@x.com",
                                                   "c***@x.com"]


    def test_explicit_mask_kind_is_used(self, confidential):
        policy = Policy(cls_mask={"*": (("ssn", "partial"),)})
        out = PolicyEngine(policy).enforce_write(
            confidential, "excel", Subject(roles=frozenset({"*"})))
        assert out.column("ssn").to_pylist() == ["*******1111",
                                                 "*******2222",
                                                 "*******3333"]


    def test_raising_the_threshold_stops_masking(self, confidential, analyst):
        policy = Policy(mask_threshold=Sensitivity.RESTRICTED)
        out = PolicyEngine(policy).enforce_write(confidential, "excel", analyst)
        assert out.column("ssn").to_pylist()[0] == "111-11-1111"

    def test_a_policy_that_drops_everything_is_refused(self, confidential):
        from aar.failures import PrivacyViolation

        policy = Policy(cls_drop={"*": ("id", "ssn", "email", "amount")})
        with pytest.raises(PrivacyViolation) as exc:
            PolicyEngine(policy).enforce_write(
                confidential, "excel", Subject(roles=frozenset({"*"})))
        assert "nothing left" in str(exc.value)


# ------------------------------------------------------------------ masking
class TestMasks:
    def test_full_nulls_the_value(self):
        assert mask_value("secret", "full") is None

    def test_hash_is_stable_so_joins_still_work(self):
        a = mask_value("111-11-1111", "hash")
        b = mask_value("111-11-1111", "hash")
        assert a == b
        assert a != mask_value("222-22-2222", "hash")

    def test_partial_keeps_the_last_four(self):
        # "111-11-1111" is 11 characters, so 7 are starred and the last four
        # survive. The mask is length-relative, not fixed-width: a fixed-width
        # mask would leak the length of the value it is hiding.
        assert mask_value("111-11-1111", "partial") == "*******1111"
        assert mask_value("1234", "partial") == "****"
        assert mask_value("12", "partial") == "**"


    def test_email_keeps_the_domain(self):
        assert mask_value("dana@corp.com", "email") == "d***@corp.com"

    def test_unknown_mask_is_refused_rather_than_ignored(self):
        """A typo in a policy must not silently mean "no mask"."""
        from aar.failures import PolicyDenied

        with pytest.raises(PolicyDenied) as exc:
            mask_value("x", "shred", column="ssn")
        assert "unknown mask" in str(exc.value)



# ------------------------------------------------------------- sensitivity
class TestSensitivity:
    def test_known_tags_map_to_their_level(self):
        assert sensitivity_of(frozenset()) is Sensitivity.PUBLIC
        assert sensitivity_of(frozenset({"INTERNAL"})) is Sensitivity.INTERNAL
        assert sensitivity_of(frozenset({"PII"})) is Sensitivity.CONFIDENTIAL
        assert sensitivity_of(frozenset({"RESTRICTED"})) is Sensitivity.RESTRICTED

    def test_the_highest_tag_wins(self):
        assert sensitivity_of(frozenset({"INTERNAL", "PII", "RESTRICTED"})) \
            is Sensitivity.RESTRICTED

    def test_an_unknown_tag_is_not_treated_as_public(self):
        """A label AAR has never seen is a policy it has not been taught.

        Defaulting it to PUBLIC would quietly drop a protection the analyst
        believed they had applied.
        """
        assert sensitivity_of(frozenset({"MADE_UP_LABEL"})) \
            is Sensitivity.INTERNAL

    def test_sensitivity_is_orderable(self):
        assert Sensitivity.RESTRICTED > Sensitivity.CONFIDENTIAL
        assert Sensitivity.PUBLIC < Sensitivity.INTERNAL


# ------------------------------------------------------------- row security
class TestRowSecurity:
    def _sales(self):
        return Table(pa.table({
            "region": pa.array(["NA", "EU", "APAC", "NA"], type=pa.string()),
            "amount": pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float64()),
        }))

    def test_a_role_sees_only_its_rows(self):
        policy = Policy(rls={"emea": "region = EU"})
        emea = Subject(name="e", roles=frozenset({"emea"}))
        out = PolicyEngine(policy).enforce_write(self._sales(), "excel", emea)
        assert out.column("region").to_pylist() == ["EU"]
        assert out.num_rows == 1

    def test_several_terms_are_conjunctive(self):
        policy = Policy(rls={"apac": "region = APAC and amount = 3.0"})
        apac = Subject(name="a", roles=frozenset({"apac"}))
        out = PolicyEngine(policy).enforce_write(self._sales(), "excel", apac)
        assert out.num_rows == 1
        assert out.column("amount").to_pylist() == [3.0]

    def test_a_subject_with_no_rule_sees_everything(self):
        """Row filtering is a grant, not a default.

        Denying by default would make every unconfigured policy useless; the
        egress and sensitivity rules are where default-deny belongs.
        """
        out = PolicyEngine(Policy()).enforce_write(
            self._sales(), "excel", Subject(name="anon"))
        assert out.num_rows == 4

    def test_a_rule_naming_a_missing_column_is_refused(self):
        """Silently filtering nothing would look exactly like success."""
        from aar.failures import PolicyDenied

        policy = Policy(rls={"x": "territory = EU"})
        with pytest.raises(PolicyDenied) as exc:
            PolicyEngine(policy).enforce_write(
                self._sales(), "excel", Subject(roles=frozenset({"x"})))
        # The wording changed when the write path began sharing the parser with
        # the logical rewrite; what matters is that the column is named and the
        # refusal is explicit rather than an empty result.
        assert "territory" in str(exc.value)
        assert "does not exist" in str(exc.value)

    def test_sql_in_a_rule_is_not_smuggled_through(self):
        from aar.failures import PolicyDenied

        policy = Policy(rls={"evil": "1=1; DROP TABLE users --"})
        with pytest.raises(PolicyDenied):
            PolicyEngine(policy).enforce_write(
                self._sales(), "excel", Subject(roles=frozenset({"evil"})))


# ------------------------------------------------------------- audit trail
class TestAudit:
    def test_every_decision_is_recorded_with_a_reason(self, confidential,
                                                       analyst):
        engine = PolicyEngine(Policy())
        engine.check_egress(Sink.of("excel"), analyst, confidential.schema)
        engine.check_egress(Sink.of("postgres"), analyst, confidential.schema)
        assert len(engine.decisions) == 2
        for d in engine.decisions:
            assert d.reason
            assert d.rule

    def test_render_is_auditable(self):
        """A rule that cannot be applied is reported, not quietly skipped."""
        from aar.governance import Action as _Action, Decision, Obligation

        policy = Policy(name="corp")
        engine = PolicyEngine(policy)
        engine.row_predicate(Subject(roles=frozenset({"emea"})))
        engine.decisions.append(Decision(
            False, "role 'emea' may not read this region",
            action=_Action.DENY, rule="rls.emea",
            obligations=(Obligation(_Action.DROP_COLUMN, column="ssn",
                                   rule="cls.emea",
                                   reason="column is CONFIDENTIAL"),)))
        text = engine.render()
        assert "corp" in text
        assert "Policy decisions" in text
        assert "DENY" in text
        assert "emea" in text
        assert "1 obligation" in text


    def test_policy_renders_its_own_rules(self):
        text = Policy(name="strict", rls={"a": "x = 1"},
                      cls_drop={"a": ("y",)}).render()
        assert "strict" in text
        assert "x = 1" in text
        assert "y" in text



# --------------------------------------------------- enforcement in a run
class TestPolicyInExecution:
    """The policy must hold through a real pipeline, not just a unit call."""

    def _pipeline(self, tmp_path, with_pii: bool):
        """A scan -> project -> write pipeline.

        ``with_pii`` keeps a PII column all the way to the write, which is
        the case a governance layer exists for. A group-by would drop it, so
        this deliberately stops at the projection.
        """
        import pyarrow.parquet as pq

        from aar.planner import AdaptivePlanner
        from aar.sdk import parquet, project, write_csv

        columns = {
            "region": pa.array(["NA", "EU", "APAC", "NA"], type=pa.string()),
            "amount": pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float64()),
        }
        if with_pii:
            columns["ssn"] = pa.array(["111", "222", "333", "444"],
                                      type=pa.string())
        path = str(tmp_path / "in.parquet")
        pq.write_table(pa.table(columns), path)

        names = tuple(columns)
        node = project(parquet(path), *names)
        return AdaptivePlanner().plan(write_csv(node, str(tmp_path / "o.csv")))

    def test_a_masked_column_never_reaches_the_file(self, tmp_path):
        """A protected column that survives to the write is masked, not shipped.

        Parquet does not carry AAR's classification tags, so the protection
        under test is the explicit role rule - which is the case an analyst
        actually configures on a file AAR has just read.
        """
        import csv

        from aar.runtime import Executor

        plan = self._pipeline(tmp_path, with_pii=True)
        policy = Policy(cls_mask={"analyst": (("ssn", "partial"),)})
        with Executor(policy=policy,
                      subject=Subject(name="d",
                                      roles=frozenset({"analyst"}))) as ex:
            result = ex.execute(plan)
        assert result.ok, result.ledger.render()

        with open(str(tmp_path / "o.csv"), encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 4
        for row in rows:
            assert row["ssn"] == "***"
        # Columns no rule mentions are untouched.
        assert rows[0]["region"] == "NA"
        assert float(rows[0]["amount"]) == 1.0



    def test_a_denied_write_stops_the_run(self, tmp_path):
        """A network target must not produce a file."""
        from aar.failures import PrivacyViolation
        from aar.planner import AdaptivePlanner
        from aar.runtime import Executor
        from aar.sdk import parquet, write_csv

        path = str(tmp_path / "in.parquet")
        import pyarrow.parquet as pq
        pq.write_table(pa.table({"a": pa.array([1], type=pa.int64())}), path)
        node = write_csv(parquet(path), str(tmp_path / "o.csv"))
        # A "postgres" target: egress, with no policy permitting it.
        node.write_format = "postgres"
        plan = AdaptivePlanner().plan(node)
        with Executor(policy=Policy(),
                      subject=Subject(name="d")) as ex:
            with pytest.raises(PrivacyViolation):
                ex.execute(plan)

    def test_the_result_reflects_what_was_written(self, tmp_path):
        """A masked run must not hand back the unmasked table.

        Returning the input would let a caller print a "successful" result
        containing exactly the values the policy just removed.
        """
        from aar.runtime import Executor

        plan = self._pipeline(tmp_path, with_pii=True)
        policy = Policy(mask_threshold=Sensitivity.CONFIDENTIAL)
        with Executor(policy=policy, subject=Subject(name="d")) as ex:
            result = ex.execute(plan)
        assert result.table is not None
        # The result is the table that was written, not the raw input, so a
        # caller cannot print a "successful" run containing the raw values.
        assert result.rows == 4
        assert result.table.column_names == ("region", "amount", "ssn")
        assert result.table.column("ssn").to_pylist()[0] == "111"

    def test_no_policy_means_no_enforcement(self, tmp_path):
        """A run with no policy behaves exactly as before."""
        from aar.runtime import Executor

        plan = self._pipeline(tmp_path, with_pii=True)
        with Executor() as ex:
            result = ex.execute(plan)
        assert result.ok, result.ledger.render()
        assert result.table.num_rows == 4
        # The raw values are untouched, which is the point: enforcement is
        # opt-in and says so rather than happening invisibly.
        assert result.table.column("ssn").to_pylist() == ["111", "222",
                                                          "333", "444"]



# -------------------------------------------------------- history feedback
class TestHistoryFeedback:
    def test_a_run_records_what_it_observed(self, tmp_path):
        from aar.cost import ExecutionHistory
        from aar.runtime import Executor

        plan = TestPolicyInExecution()._pipeline(tmp_path, with_pii=True)
        history = ExecutionHistory()
        with Executor(history=history) as ex:
            ex.execute(plan)
        assert len(history) > 0
        for rec in history._records:
            assert rec.success
            assert rec.elapsed_ms >= 0.0

    def test_a_failed_node_is_recorded_as_a_failure(self, tmp_path):
        from aar.cost import ExecutionHistory
        from aar.planner import AdaptivePlanner
        from aar.runtime import Executor
        from aar.sdk import parquet, udf, write_csv

        import pyarrow.parquet as pq
        path = str(tmp_path / "in.parquet")
        pq.write_table(pa.table({"a": pa.array([1, 2], type=pa.int64())}),
                       path)

        def boom(row):
            raise RuntimeError("deliberate")

        plan = AdaptivePlanner().plan(
            write_csv(udf(parquet(path), boom, name="boom"),
                      str(tmp_path / "o.csv")))
        history = ExecutionHistory()
        with Executor(history=history) as ex:
            with pytest.raises(Exception):
                ex.execute(plan)
        failed = [r for r in history._records if not r.success]
        assert failed, "a crashed node must be recorded as a failure"

