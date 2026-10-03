"""Small-cell suppression: RLS's answer is not the whole answer.

Every test here is a *disclosure that must not happen*, or a *policy that must
be refused*. The happy path - a big enough group, correctly reported - is
asserted too, because a control that suppressed everything would satisfy every
other test in this file.

The subject of all of it is one specific failure: RLS can be applied perfectly,
every row the analyst may see is the only row they see, and the answer still
discloses a colleague's salary. A row filter cannot fix that, because nothing
was filtered.
"""

from __future__ import annotations

import json

import pytest

from aar.application import PipelineService
from aar.failures import PolicyDenied
from aar.governance.disclosure import (
    GROUP_COUNT_COLUMN, DisclosureRule, apply_disclosure_control,
    disclosure_guards, effective_rule, enforced_rule, merge_rules,
    strip_group_counts,
)
from aar.ir import Agg, Col, Node, NodeType, ScanSpec
from aar.runtime.executor import Executor

# Two regions. North has six contributors, south has two: enough for one
# boundary and not the other, which is what makes a mistaken k visible.
ROWS = "region,amount\n" + "".join(
    f"north,{n * 10}\n" for n in range(1, 7)) + "south,100\nsouth,200\n"

# A `keep` column, so an RLS barrier can thin one group without removing the
# other. RLS predicates are equality-only by design, so the barrier selects a
# value rather than a range - which means thinning a single group needs a
# column the other group's rows do not share.
ROWS_WITH_KEEP = ("region,amount,keep\n" + "".join(
    f"north,{n * 10},1\n" for n in range(1, 7))
    + "south,100,0\nsouth,200,1\n")


def _pipeline(directory, csv_text: str = ROWS):
    """A real, runnable aggregate pipeline - not a mock of one."""
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
    """Write a policy file and return its path."""
    path = directory / "policy.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _run(directory, policy=None):
    if policy is None:
        return PipelineService().run(str(_pipeline(directory)))
    path = directory / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    return PipelineService().run(str(_pipeline(directory)),
                                 policy_path=str(path))


def _rows(report):
    return report.result.table.arrow.to_pylist()


def _regions(report):
    """The group names a run returned, as a set."""
    return {r["region"] for r in _rows(report)}


class TestTheDisclosureItPrevents:
    def test_an_average_over_two_people_is_withheld(self, tmp_path):
        """The whole reason this exists, as a concrete number.

        South is two people at 100 and 200. Its average is 150, which is not a
        summary of a group - it is one arithmetic step from each of them.
        """
        report = _run(tmp_path, {"name": "t", "disclosure": 5})
        rows = {r["region"]: r["avg_amount"] for r in _rows(report)}
        assert "south" not in rows, f"a 2-person average leaked: {rows}"
        assert rows["north"] == pytest.approx(35.0)

    def test_the_group_that_is_big_enough_still_appears(self, tmp_path):
        """A control that suppressed everything would pass the test above."""
        report = _run(tmp_path, {"name": "t", "disclosure": 5})
        rows = {r["region"]: r["avg_amount"] for r in _rows(report)}
        assert set(rows) == {"north"}
        assert rows["north"] == pytest.approx(35.0)

    def test_the_boundary_is_the_rule_not_an_approximation(self, tmp_path):
        """k=2 permits south; k=3 does not. The count decides, nothing else."""
        assert _regions(_run(tmp_path, {"name": "t", "disclosure": 2})) == \
            {"north", "south"}
        assert _regions(_run(tmp_path, {"name": "t", "disclosure": 3})) == \
            {"north"}


class TestTheGuardIsHonestAboutWhatItDid:
    def test_it_reports_what_was_suppressed(self, tmp_path):
        """Silence would be indistinguishable from a guard that never ran."""
        report = _run(tmp_path, {"name": "t", "disclosure": 5})
        verdicts = report.result.suppressions
        assert len(verdicts) == 1
        assert verdicts[0].suppressed_groups == 1
        assert verdicts[0].smallest_group == 2

    def test_it_says_so_when_nothing_was_suppressed(self, tmp_path):
        """A clean run is a fact worth recording, not an absence of one."""
        report = _run(tmp_path, {"name": "t", "disclosure": 2})
        verdicts = report.result.suppressions
        assert verdicts and verdicts[0].suppressed_groups == 0
        assert "enough contributors" in verdicts[0].describe()

    def test_the_evidence_reaches_the_execution_record(self, tmp_path):
        report = _run(tmp_path, {"name": "t", "disclosure": 5})
        rendered = report.result.render()
        assert "disclosure.min_group_size" in rendered
        assert "suppressed 1 group" in rendered

    def test_suppressions_do_not_leak_between_runs(self, tmp_path):
        """A reused service must not carry the last run's verdicts.

        Otherwise a run that suppressed nothing reports the previous run's
        suppression, and the evidence says something false about this one.
        """
        service = PipelineService()
        policy = tmp_path / "policy.json"
        pipeline = _pipeline(tmp_path)
        policy.write_text(json.dumps({"name": "t", "disclosure": 5}),
                          encoding="utf-8")
        first = service.run(str(pipeline), policy_path=str(policy))
        assert first.result.suppressions[0].suppressed_groups == 1

        policy.write_text(json.dumps({"name": "t", "disclosure": 2}),
                          encoding="utf-8")
        second = service.run(str(pipeline), policy_path=str(policy))
        assert second.result.suppressions[0].suppressed_groups == 0


class TestTheCountIsNotAThingYouCanSee:
    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_the_hidden_count_never_reaches_the_result(self, tmp_path, k):
        """An analyst who could read the count would learn the group sizes.

        That is the same information the rule protects, delivered with a
        column name instead of an aggregate.
        """
        table = _run(tmp_path, {"name": "t", "disclosure": k}).result.table
        assert GROUP_COUNT_COLUMN not in table.column_names, k
        assert all(GROUP_COUNT_COLUMN not in row
                   for row in table.arrow.to_pylist())

    def test_stripping_is_idempotent(self, tmp_path):
        table = _run(tmp_path).result.table
        assert strip_group_counts(table) is table

    def test_stripping_leaves_the_original_alone(self, tmp_path):
        """The check has to stay re-runnable after it has run once."""
        table = _run(tmp_path).result.table
        before = list(table.column_names)
        strip_group_counts(table)
        assert list(table.column_names) == before


class TestNoPolicyMeansNoChange:
    def test_without_a_rule_nothing_is_suppressed(self, tmp_path):
        """A control nobody asked for must not silently change results."""
        assert _regions(_run(tmp_path, {"name": "t"})) == {"north", "south"}

    def test_without_a_policy_nothing_is_suppressed(self, tmp_path):
        assert _regions(_run(tmp_path)) == {"north", "south"}

    def test_no_verdicts_are_recorded_when_no_rule_applies(self, tmp_path):
        assert _run(tmp_path, {"name": "t"}).result.suppressions == []


class TestItIsAPerGroupRuleNotAGlobalOne:
    def test_a_large_table_still_loses_its_small_group(self, tmp_path):
        """The case a global row count gets wrong.

        500 contributors, 495 of them in one group. Every whole-table measure
        - rows_before, rows_after, an RLS summary - says "this table is large
        and safe". Only the per-group count sees the group of five.
        """
        rows = "region,amount\n" + "".join(
            f"big,{n}\n" for n in range(500)) + "tiny,7\n"
        policy = _policy(tmp_path, {"name": "t", "disclosure": 10})
        report = PipelineService().run(str(_pipeline(tmp_path, rows)),
                                       policy_path=str(policy))
        names = {r["region"] for r in report.result.table.arrow.to_pylist()}
        assert names == {"big"}

    def test_counts_are_measured_after_rls_not_before(self, tmp_path):
        """A barrier that removes rows must change what counts.

        If the count were taken above the barrier, a subject whose RLS leaves
        them one row in a group of two would be judged on the two.
        """
        policy = _policy(tmp_path, {
            "name": "t",
            # Drops south's 100 row and keeps everything else, so south is
            # left with one contributor while north still has six.
            "rls": {"analyst": [{"source": "orders",
                                 "predicate": "keep = 1"}]},
            "disclosure": 2,
        })
        pipeline = _pipeline(tmp_path, ROWS_WITH_KEEP)
        report = PipelineService().run(str(pipeline), role="analyst",
                                       policy_path=str(policy))
        assert _regions(report) == {"north"}, (
            "the group size was counted above the RLS barrier")

    def test_the_run_records_the_post_rls_count(self, tmp_path):
        """The smallest group reported is the post-barrier one."""
        policy = _policy(tmp_path, {
            "name": "t",
            "rls": {"analyst": [{"source": "orders",
                                 "predicate": "keep = 1"}]},
            "disclosure": 2,
        })
        report = PipelineService().run(str(_pipeline(tmp_path, ROWS_WITH_KEEP)),
                                       role="analyst", policy_path=str(policy))
        assert report.result.suppressions[0].smallest_group == 1


class TestTheRuleIsAttachedToThePlan:
    """The obligation lives on the node, so it survives planning.

    A rule held only in a rewrite pass's return value would be lost by any
    tree rebuild, pushdown or clone - and would be lost silently, because a
    plan with no rule attached is a plan that looks unprotected and is.
    """

    def test_a_guarded_aggregate_carries_its_rule(self, tmp_path):
        pipeline = _pipeline(tmp_path)
        policy = _policy(tmp_path, {"name": "t", "disclosure": 5})
        root = PipelineService().run(str(pipeline),
                                     policy_path=str(policy)).root
        guards = disclosure_guards(root)
        assert len(guards) == 1
        assert guards[0].rule.min_group_size == 5

    def test_the_strictest_rule_wins_when_several_apply(self):
        """Larger k is stricter: k is a floor, so raising it hides more.

        `min` here would pick the *weaker* rule and silently weaken protection
        every time two policies applied to one aggregate.
        """
        node = Node(NodeType.GROUPBY, agg_functions={"a": (Agg("AVG", Col("x")),)})
        node.disclosure_rules = (DisclosureRule(10, "strict"),
                                 DisclosureRule(3, "loose"))
        assert enforced_rule(node).min_group_size == 10
        assert enforced_rule(node).rule == "strict"

    def test_an_unguarded_node_has_no_rule(self):
        node = Node(NodeType.GROUPBY, agg_functions={"a": (Agg("AVG", Col("x")),)})
        assert enforced_rule(node) is None

    def test_two_policies_same_name_different_k_both_survive(self):
        """The name-collapse bug, as a regression.

        Deduplicating by ``rule`` alone dropped the stricter of two rules that
        shared an id, so a k=10 policy could vanish with no trace and the
        weaker k=5 would silently govern.
        """
        merged = merge_rules((DisclosureRule(5),), (DisclosureRule(10),))
        assert {r.min_group_size for r in merged} == {5, 10}
        assert effective_rule(merged).min_group_size == 10

    def test_identical_rules_merge_rather_than_accumulate(self):
        merged = merge_rules((DisclosureRule(5, "r"),), (DisclosureRule(5, "r"),))
        assert len(merged) == 1

    def test_applying_twice_does_not_weaken_the_rule(self):
        """Idempotent: re-running the rewrite pass must not lower k."""
        from aar.governance.disclosure import apply_disclosure_control

        scan = Node(NodeType.SCAN_CSV, scan=ScanSpec(kind="csv", path="x.csv"))
        node = Node(NodeType.GROUPBY, inputs=[scan], key_left=("region",),
                    agg_functions={"a": (Agg("AVG", Col("x")),)})
        apply_disclosure_control(node, [DisclosureRule(10)], None)
        apply_disclosure_control(node, [DisclosureRule(5)], None)
        assert enforced_rule(node).min_group_size == 10

    def test_the_guard_names_what_also_applied(self):
        """Provenance is retained, not collapsed to a bare number."""
        from aar.governance.disclosure import apply_disclosure_control

        scan = Node(NodeType.SCAN_CSV, scan=ScanSpec(kind="csv", path="x.csv"))
        node = Node(NodeType.GROUPBY, inputs=[scan], key_left=("region",),
                    agg_functions={"a": (Agg("AVG", Col("x")),)})
        guards = apply_disclosure_control(
            node, [DisclosureRule(10, "strict"), DisclosureRule(3, "loose")],
            None)
        assert guards[0].rule.min_group_size == 10
        assert len(guards[0].contributing) == 2
        assert "loose" in guards[0].reason


class TestKMustActuallyProtectSomething:
    @pytest.mark.parametrize("k", [0, 1, -1])
    def test_a_k_that_suppresses_nothing_is_not_constructible(self, k):
        """k=1 permits every non-empty group.

        Omitting the rule already means "protection off", so a k=1 policy is a
        second spelling of that which *reads* as protection in an explain panel.
        """
        with pytest.raises(ValueError) as caught:
            DisclosureRule(min_group_size=k)
        assert "not a disclosure control" in str(caught.value)

    def test_two_is_accepted(self):
        assert DisclosureRule(min_group_size=2).min_group_size == 2


class TestItFailsClosed:
    def test_a_missing_group_key_is_refused(self):
        """A group-by key absent from the input is an ambiguity, not a pass.

        "I could not tell how many rows this group had" must never become
        "so I allowed the group".
        """
        scan = Node(NodeType.SCAN_CSV, scan=ScanSpec(kind="csv", path="x.csv"))
        grouped = Node(NodeType.GROUPBY, inputs=[scan],
                       key_left=("absent_column",),
                       agg_functions={"avg": (Agg("AVG", Col("amount")),)})
        grouped.disclosure_rules = (DisclosureRule(min_group_size=5),)
        # One real input whose columns genuinely do not include the group key.
        import pyarrow as pa

        from aar.interchange import Table

        table = Table.from_arrow(pa.table({"region": ["north"]}))
        with pytest.raises(PolicyDenied) as caught:
            Executor()._group_counts(grouped, [table])
        assert "absent_column" in str(caught.value)
        assert "refused" in str(caught.value)

    def test_a_non_columnar_input_is_refused(self):
        node = Node(NodeType.AGGREGATE,
                    agg_functions={"avg": (Agg("AVG", Col("amount")),)})
        node.disclosure_rules = (DisclosureRule(min_group_size=5),)

        class _Opaque:
            column_names = ("region",)

        with pytest.raises(PolicyDenied):
            Executor()._group_counts(node, [_Opaque()])

    def test_a_multi_input_aggregate_is_refused(self):
        node = Node(NodeType.GROUPBY, key_left=("region",),
                    agg_functions={"avg": (Agg("AVG", Col("amount")),)})
        node.disclosure_rules = (DisclosureRule(min_group_size=5),)
        with pytest.raises(PolicyDenied):
            Executor()._group_counts(node, [object(), object()])

    def test_a_zero_minimum_is_rejected_at_construction(self):
        """A rule that suppresses nothing must not be expressible."""
        with pytest.raises(ValueError):
            DisclosureRule(min_group_size=0)

    def test_a_rule_that_cannot_be_attached_is_refused(self):
        """Attaching to a node that cannot hold rules must not pass quietly."""

        node = Node(NodeType.GROUPBY,
                    agg_functions={"avg": (Agg("AVG", Col("amount")),)})

        class _RefuseRules:
            """A descriptor that refuses every write to the field.

            A data descriptor, so it wins over the dataclass default: the node
            reads as having no rules and cannot be given any. That is the shape
            of "a plan object that cannot carry an obligation", and the only
            safe response to it is a refusal.

            Installed on a *subclass* on purpose. Putting it on ``Node`` itself
            makes every later node in the process unwritable, which is a far
            worse failure than the one under test - and one that only shows up
            in whichever unrelated test happens to run next.
            """

            def __get__(self, instance, owner):
                return ()

            def __set__(self, instance, value):
                raise AttributeError("disclosure_rules")

        class _Unattachable(Node):
            """Built normally, then made unwritable.

            The descriptor is installed *after* construction, because a data
            descriptor on the class also fires during the dataclass
            ``__init__`` - so declaring it on the class breaks the very node
            the test needs to create.
            """

            __slots__ = ()

        node = Node(NodeType.GROUPBY,
                    agg_functions={"avg": (Agg("AVG", Col("amount")),)})
        node.__class__ = _Unattachable
        _Unattachable.disclosure_rules = _RefuseRules()
        with pytest.raises(PolicyDenied) as caught:
            apply_disclosure_control(node, [DisclosureRule(5)], None)
        assert "cannot be attached" in str(caught.value)
        assert "disclosure.min_group_size" in str(caught.value)
        del _Unattachable.disclosure_rules  # inherited default is back

    def test_the_refusing_double_did_not_leak_onto_other_nodes(self):
        """Every other node in the process must still be writable.

        Asserted directly because the original version of this test replaced
        the descriptor on ``Node`` itself, which silently broke three
        unrelated Workbench tests that ran after it. A test double that
        damages global state is a defect in the test, not in what it tests.
        """
        plain = Node(NodeType.GROUPBY,
                     agg_functions={"avg": (Agg("AVG", Col("amount")),)})
        apply_disclosure_control(plain, [DisclosureRule(5)], None)
        assert len(plain.disclosure_rules) == 1
