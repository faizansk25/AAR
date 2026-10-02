"""Row-level security as a pre-computation barrier.

The defect these tests exist to prevent is specific, and it was measured rather
than imagined: RLS used to be applied by ``enforce_write``, at the end of a run.
An analyst asking "what is the average salary in my region?" got the average
over *every* region, because by the time the predicate was applied the ``AVG``
had already consumed all of them - and filtering the single output row would
either drop the result or raise on a column the aggregate no longer had.

Each class below asserts one property the architecture depends on:

* the barrier is placed **before** every combining operation;
* an ambiguous rule is **refused**, never guessed at;
* the restriction is **identical on every engine**, which is the differential
  check that a correct answer on Arrow and a leak on DuckDB would fail;
* a barrier that disappears, weakens, or is hoisted is caught;
* ``policy=None`` is a *baseline*, not "governance off".
"""

from __future__ import annotations

import json

import pyarrow as pa
import pytest

from aar.failures import PolicyDenied
from aar.governance import (Policy, PolicyEngine, Subject, apply_row_security,
                            assert_barriers_intact, parse_row_predicate)
from aar.interchange import Table
from aar.ir import (Agg, BinOp, Col, Lit, Node, NodeType, ScanSpec,
                    semantic_operation_id, topological_order)


# ------------------------------------------------------------------ fixtures
def _scan(name: str) -> Node:
    return Node(NodeType.SCAN_PARQUET,
                scan=ScanSpec(kind="parquet", path=f"{name}.parquet",
                              source_name=name))


def _employees() -> Node:
    """Scan -> group by region -> AVG(salary): the shape that used to fail."""
    scan = _scan("employees")
    grp = Node(NodeType.GROUPBY, inputs=[scan], key_left=("region",))
    grp.agg_functions = {"avg_salary": (Agg("AVG", Col("salary")),)}
    return grp


def _two_sources() -> Node:
    return Node(NodeType.JOIN, inputs=[_scan("orders"), _scan("users")],
                key_left=("k",))


def _analyst() -> Subject:
    return Subject(name="dana", roles=frozenset({"analyst"}))


def _scoped(source: str, predicate: str) -> Policy:
    return Policy(rls={"analyst": [{"source": source, "predicate": predicate}]})


def _table() -> Table:
    return Table(pa.table({
        "region": ["EU", "NA", "EU", "APAC", "NA"],
        "salary": [100.0, 200.0, 300.0, 400.0, 500.0],
    }))


# ------------------------------------------------------- placement is the point
class TestTheBarrierPrecedesEverything:
    def test_the_restriction_is_injected_below_the_aggregate(self):
        barriers = apply_row_security(_employees(),
                                      _scoped("employees", "region = EU"),
                                      _analyst())
        assert len(barriers) == 1
        assert barriers[0].rule == "rls.analyst"
        assert barriers[0].source == "employees"

    def test_the_plan_order_puts_security_before_the_group_by(self):
        root = _employees()
        apply_row_security(root, _scoped("employees", "region = EU"), _analyst())
        order = [n.type.value for n in topological_order(root)]
        assert order.index("SecurityFilter") < order.index("GroupBy"), order

    def test_the_barrier_is_a_distinct_type_not_an_ordinary_filter(self):
        """If it were a plain FILTER, the optimiser would treat it as optional."""
        root = _employees()
        apply_row_security(root, _scoped("employees", "region = EU"), _analyst())
        kinds = {n.type for n in topological_order(root)}
        assert NodeType.SECURITY_FILTER in kinds
        assert NodeType.FILTER not in kinds

    def test_the_barrier_carries_its_provenance(self):
        barriers = apply_row_security(_employees(),
                                      _scoped("employees", "region = EU"),
                                      _analyst())
        node = barriers[0].node
        assert node.security_rule == "rls.analyst"
        assert node.source_scope == "employees"
        assert "SECURITY" in node.describe()

    def test_the_aggregate_sees_only_authorized_rows(self):
        """The end-to-end answer, measured rather than assumed from shape."""
        from aar.engines import create_engine

        barriers = apply_row_security(_employees(),
                                      _scoped("employees", "region = EU"),
                                      _analyst())
        engine = create_engine("arrow")
        secured = engine.filter(_table(), barriers[0].node.predicate)
        assert secured.to_arrow().column("region").to_pylist() == ["EU", "EU"]

        # 200.0 over EU only. Over everything it would be 300.0, the number the
        # old write-time filter could never have produced.
        grouped = engine.group_by(secured, ["region"], {
            "avg_salary": Agg("AVG", Col("salary"))})
        assert grouped.to_arrow().to_pylist() == [
            {"region": "EU", "avg_salary": 200.0}]

    def test_a_role_with_no_rule_is_left_alone(self):
        """Row filtering is a grant, not a default - unchanged semantics."""
        root = _employees()
        before = [n.type for n in topological_order(root)]
        assert apply_row_security(root, Policy(rls={"other": "x = 1"}),
                                  _analyst()) == []
        assert [n.type for n in topological_order(root)] == before
class TestAmbiguityIsRefusedNotGuessed:
    def test_an_unscoped_rule_over_two_sources_is_refused(self):
        """The dangerous case: which input does `region = EU` belong to?"""
        with pytest.raises(PolicyDenied) as exc:
            apply_row_security(_two_sources(),
                               Policy(rls={"analyst": "region = EU"}),
                               _analyst())
        message = str(exc.value)
        assert "unscoped" in message
        assert "orders" in message and "users" in message

    def test_the_refusal_says_how_to_fix_it(self):
        with pytest.raises(PolicyDenied) as exc:
            apply_row_security(_two_sources(),
                               Policy(rls={"analyst": "region = EU"}),
                               _analyst())
        assert "'source'" in str(exc.value)

    def test_a_rule_naming_an_absent_source_is_refused(self):
        """A typo must not read as 'no restriction needed'."""
        with pytest.raises(PolicyDenied) as exc:
            apply_row_security(_employees(), _scoped("custmers", "region = EU"),
                               _analyst())
        assert "does not read" in str(exc.value)

    def test_an_unscoped_rule_over_one_source_is_fine(self):
        """The legacy form stays legal when the answer is unambiguous."""
        barriers = apply_row_security(
            _employees(), Policy(rls={"analyst": "region = EU"}), _analyst())
        assert len(barriers) == 1

    def test_both_sides_of_a_join_are_secured_independently(self):
        barriers = apply_row_security(_two_sources(), Policy(rls={"analyst": [
            {"source": "orders", "predicate": "region = EU"},
            {"source": "users", "predicate": "tenant_id = 42"}]}), _analyst())
        assert {b.source for b in barriers} == {"orders", "users"}

    def test_each_side_of_a_join_gets_its_own_predicate(self):
        barriers = apply_row_security(_two_sources(), Policy(rls={"analyst": [
            {"source": "orders", "predicate": "region = EU"},
            {"source": "users", "predicate": "tenant_id = 42"}]}), _analyst())
        by_source = {b.source: str(b.node.predicate) for b in barriers}
        assert "region" in by_source["orders"]
        assert "tenant_id" in by_source["users"]

    def test_sql_is_not_smuggled_through_a_policy_string(self):
        with pytest.raises(PolicyDenied):
            apply_row_security(_employees(),
                               Policy(rls={"analyst": "1=1; DROP TABLE users --"}),
                               _analyst())

    def test_an_unsupported_operator_is_refused_not_approximated(self):
        with pytest.raises(PolicyDenied):
            apply_row_security(_employees(),
                               Policy(rls={"analyst": "region LIKE 'E%'"}),
                               _analyst())

    def test_a_rule_naming_an_undeclared_column_is_refused(self):
        """Silently filtering nothing looks exactly like success."""
        from aar.types import Field, Schema, UTF8

        scan = _scan("employees")
        scan.output_schema = Schema(fields=(Field("region", UTF8),))
        root = Node(NodeType.PROJECT, inputs=[scan], columns=("region",))
        with pytest.raises(PolicyDenied) as exc:
            apply_row_security(root, Policy(rls={"analyst": "territory = EU"}),
                               _analyst())
        assert "territory" in str(exc.value)


# --------------------------------------------------------- the barrier holds
class TestTheBarrierCannotBeWeakenedOrMoved:
    def _secured(self):
        root = _employees()
        barriers = apply_row_security(root, _scoped("employees", "region = EU"),
                                      _analyst())
        return root, barriers

    def test_an_intact_plan_passes_the_check(self):
        root, barriers = self._secured()
        assert_barriers_intact(root, barriers)  # must not raise

    def test_no_barriers_means_no_check(self):
        assert_barriers_intact(_employees(), [])  # must not raise

    def test_a_removed_barrier_is_caught(self):
        root, barriers = self._secured()
        barrier_node = barriers[0].node
        grp = next(n for n in root.walk() if n.type is NodeType.GROUPBY)
        grp.inputs = [barrier_node.input(0)]  # bypass the security filter
        with pytest.raises(PolicyDenied) as exc:
            assert_barriers_intact(root, barriers)
        assert "no longer part of the plan" in str(exc.value)

    def test_a_hoisted_barrier_is_caught(self):
        """Filter above the aggregate: the original defect, reintroduced.

        Rewired as ``scan -> GroupBy -> SECURITY_FILTER -> Write`` - the shape
        AAR had before this change - and then checked from the write, which is
        the node a run actually ends at.
        """
        root, barriers = self._secured()
        barrier_node = barriers[0].node
        scan = barrier_node.input(0)
        grp = next(n for n in root.walk() if n.type is NodeType.GROUPBY)

        barrier_node.inputs = [grp]   # now fed by the aggregate...
        grp.inputs = [scan]           # ...and the scan feeds the aggregate
        write = Node(NodeType.WRITE, inputs=[barrier_node], target="out.csv")
        with pytest.raises(PolicyDenied) as exc:
            assert_barriers_intact(write, barriers)
        assert "after a GroupBy" in str(exc.value)

    def test_a_weakened_predicate_is_caught(self):
        root, barriers = self._secured()
        barriers[0].node.predicate = BinOp(Col("region"), "=", Lit("NA"))
        with pytest.raises(PolicyDenied) as exc:
            assert_barriers_intact(root, barriers)
        assert "different predicate" in str(exc.value)


# --------------------------------------------------------------- differential
class TestEveryEngineAgreesOnTheAuthorizedRows:
    """The check that catches "secure on Arrow, leaking on DuckDB".

    Discovery is behavioural and refuses degradation, for the reason
    ``test_cross_engine_semantics`` documents: ``create_engine`` *silently
    substitutes* a fallback, so a naive loop would compare DuckDB with DuckDB
    and pass while proving nothing at all.
    """

    @staticmethod
    def _engines():
        from aar.engines import create_engine

        found = []
        for engine_id in ("arrow", "duckdb", "pandas", "polars_cpu"):
            try:
                found.append((engine_id,
                              create_engine(engine_id, allow_degradation=False)))
            except Exception:  # noqa: BLE001 - not installed on this machine
                continue
        return found

    def test_the_secured_rowset_is_identical_on_every_engine(self):
        engines = self._engines()
        assert len(engines) >= 2, "need at least two engines to compare"
        barriers = apply_row_security(_employees(),
                                      _scoped("employees", "region = EU"),
                                      _analyst())
        predicate = barriers[0].node.predicate

        seen = {}
        for engine_id, engine in engines:
            rows = engine.filter(_table(), predicate).to_arrow().to_pylist()
            # The guarantee itself, per engine...
            assert [r["region"] for r in rows] == ["EU", "EU"], engine_id
            seen[engine_id] = sorted(rows, key=lambda r: r["salary"])
        # ...and that no engine disagrees about *which* rows those are.
        values = list(seen.values())
        assert all(v == values[0] for v in values), seen

    def test_the_secured_aggregate_matches_across_engines(self):
        engines = self._engines()
        if len(engines) < 2:
            pytest.skip("need at least two engines to compare")
        barriers = apply_row_security(_employees(),
                                      _scoped("employees", "region = EU"),
                                      _analyst())
        predicate = barriers[0].node.predicate

        results = {}
        for engine_id, engine in engines:
            secured = engine.filter(_table(), predicate)
            results[engine_id] = engine.group_by(secured, ["region"], {
                "avg_salary": Agg("AVG", Col("salary"))}).to_arrow().to_pylist()
        first = next(iter(results.values()))
        assert all(v == first for v in results.values()), results
        assert first[0]["avg_salary"] == pytest.approx(200.0)

    def test_the_unsecured_aggregate_is_larger_so_the_test_is_real(self):
        """Guard against a vacuous comparison: EU is not the whole table."""
        from aar.engines import create_engine

        engine = create_engine("arrow")
        everything = engine.group_by(_table(), ["region"], {
            "avg_salary": Agg("AVG", Col("salary"))}).to_arrow().to_pylist()
        overall = sum(r["avg_salary"] for r in everything) / len(everything)
        assert overall != pytest.approx(200.0), \
            "the secured figure must differ from the unrestricted one"


# --------------------------------------------------------- the default matters
class TestNoPolicyIsABaselineNotAnAbsence:
    def test_a_missing_policy_builds_a_restrictive_engine(self):
        """`None` used to mean 'governance off'."""
        from aar.application import PipelineService

        engine = PipelineService._load_policy(None)
        assert isinstance(engine, PolicyEngine)

    def test_the_baseline_denies_network_egress(self):
        from aar.application import PipelineService

        decision = PipelineService._load_policy(None).check_egress(
            "postgres", Subject())
        assert not decision.allowed
        assert decision.rule == "egress.network"

    def test_the_baseline_still_allows_local_output(self):
        from aar.application import PipelineService

        assert PipelineService._load_policy(None).check_egress(
            "parquet", Subject()).allowed

    def test_the_baseline_applies_no_row_rules(self):
        """No RLS configured means no RLS - not deny-all rows."""
        from aar.application import PipelineService

        assert PipelineService._load_policy(None).row_predicate(
            Subject()).rule == "rls.none"

    def test_a_missing_policy_file_is_still_a_hard_error(self):
        from aar.application import PipelineService

        with pytest.raises(FileNotFoundError):
            PipelineService._load_policy("no/such/policy.json")

    def test_the_explicit_bypass_returns_none(self):
        from aar import cli

        assert cli._load_policy(None, disable=True) is None

    def test_the_bypass_announces_itself_on_stderr(self, capsys):
        from aar import cli

        cli._load_policy(None, disable=True)
        assert "DISABLED" in capsys.readouterr().err

    def test_a_secured_plan_does_not_re_apply_rls_at_the_write(self):
        """Re-filtering the aggregate's output would refuse a correct query.

        `region = EU` names a column `AVG(salary)` does not produce, so a
        second application either raises or drops the result - punishing
        exactly the analytic query this change exists to enable.
        """
        engine = PolicyEngine(Policy(rls={"analyst": "region = EU"}))
        aggregate = Table(pa.table({"avg_salary": [200.0]}))
        out = engine.enforce_write(aggregate, "parquet", _analyst(),
                                   rls_applied=True)
        assert out.to_arrow().to_pylist() == [{"avg_salary": 200.0}]

    def test_an_unsecured_plan_still_filters_at_the_write(self):
        """The fallback keeps its old meaning when there are no barriers."""
        engine = PolicyEngine(Policy(rls={"analyst": "region = EU"}))
        assert engine.enforce_write(
            _table(), "parquet", _analyst()).num_rows == 2

    def test_scoped_rules_refuse_the_unsecured_write_path(self):
        """Better to refuse than to apply one rule of two and look compliant."""
        engine = PolicyEngine(Policy(rls={"analyst": [
            {"source": "orders", "predicate": "region = EU"}]}))
        with pytest.raises(PolicyDenied) as exc:
            engine.enforce_write(_table(), "parquet", _analyst())
        assert "no security barriers" in str(exc.value)


# ------------------------------------------------------------ policy loading
class TestPolicyFileFormat:
    def test_the_bare_string_form_still_loads(self, tmp_path):
        from aar.governance import load_policy

        path = tmp_path / "p.json"
        path.write_text(json.dumps({"rls": {"analyst": "region = EU"}}))
        policy = load_policy(str(path))
        assert policy.rules_for_role("analyst")[0].predicate == "region = EU"

    def test_the_scoped_form_loads(self, tmp_path):
        from aar.governance import load_policy

        path = tmp_path / "p.json"
        path.write_text(json.dumps({"rls": {"analyst": [
            {"source": "orders", "predicate": "region = EU"}]}}))
        rule = load_policy(str(path)).rules_for_role("analyst")[0]
        assert (rule.source, rule.predicate) == ("orders", "region = EU")

    def test_an_unknown_key_in_a_rule_is_refused(self, tmp_path):
        from aar.governance import load_policy

        path = tmp_path / "p.json"
        path.write_text(json.dumps({"rls": {"analyst": [
            {"src": "orders", "predicate": "region = EU"}]}}))
        with pytest.raises(PolicyDenied) as exc:
            load_policy(str(path))
        assert "src" in str(exc.value)

    def test_an_empty_predicate_is_refused(self, tmp_path):
        from aar.governance import load_policy

        path = tmp_path / "p.json"
        path.write_text(json.dumps({"rls": {"analyst": [
            {"source": "orders", "predicate": "  "}]}}))
        with pytest.raises(PolicyDenied):
            load_policy(str(path))

    def test_the_example_policy_is_source_scoped(self):
        """The shipped example must model the safe form."""
        from aar.governance import EXAMPLE_POLICY

        for role, value in EXAMPLE_POLICY["rls"].items():
            assert isinstance(value, list), role
            assert all("source" in rule for rule in value), role

    def test_the_example_policy_still_validates(self, tmp_path):
        from aar.governance import EXAMPLE_POLICY, load_policy

        path = tmp_path / "p.json"
        path.write_text(json.dumps(EXAMPLE_POLICY))
        assert load_policy(str(path)).name == "example-strict"

    def test_a_policy_renders_its_scoped_rules(self):
        text = _scoped("orders", "region = EU").render()
        assert "orders" in text and "region = EU" in text


# ------------------------------------------------------------- literal typing
class TestPolicyLiteralsAreTyped:
    def test_a_numeric_literal_is_not_left_as_text(self):
        """"3" against an int column matches nothing, silently."""
        value = parse_row_predicate("tenant_id = 42").right.value
        assert value == 42 and isinstance(value, int)

    def test_a_float_literal_parses_as_a_float(self):
        assert parse_row_predicate("amount = 3.0").right.value == 3.0

    def test_a_quoted_literal_stays_text(self):
        assert parse_row_predicate("region = 'EU'").right.value == "EU"

    def test_conjunctions_compose(self):
        assert parse_row_predicate("region = 'EU' and tenant_id = 7").op == "AND"

    def test_column_matching_is_case_insensitive(self):
        assert parse_row_predicate("REGION = EU",
                                   columns=("region",)).left.name == "region"

    def test_an_unparseable_term_is_refused(self):
        with pytest.raises(PolicyDenied):
            parse_row_predicate("region <> EU")


# -------------------------------------------------- identity must not collide
class TestSecurityIsPartOfIdentity:
    """A cached result must not cross a policy boundary.

    Identity exists so two runs of the same query share measurements. If the
    security rule were not part of that identity, a result computed under a
    permissive policy would satisfy a later, more restrictive request - and
    because the data is *wrong* rather than missing, nothing downstream could
    notice.
    """

    def _secured(self, predicate: str) -> Node:
        root = _employees()
        apply_row_security(root, _scoped("employees", predicate), _analyst())
        return next(n for n in root.walk()
                    if n.type is NodeType.SECURITY_FILTER)

    def test_two_policies_do_not_share_an_identity(self):
        assert (semantic_operation_id(self._secured("region = EU"))
                != semantic_operation_id(self._secured("region = NA")))

    def test_the_same_policy_does_share_an_identity(self):
        assert (semantic_operation_id(self._secured("region = EU"))
                == semantic_operation_id(self._secured("region = EU")))

    def test_the_same_predicate_on_a_different_source_differs(self):
        a, b = self._secured("region = EU"), self._secured("region = EU")
        b.source_scope = "payments"
        assert semantic_operation_id(a) != semantic_operation_id(b)

    def test_a_secured_operation_differs_from_an_unsecured_one(self):
        """A plain FILTER with the same predicate is not the same operation."""
        secured = self._secured("region = EU")
        plain = Node(NodeType.FILTER, inputs=[_scan("employees")],
                     predicate=secured.predicate)
        assert semantic_operation_id(secured) != semantic_operation_id(plain)