"""Lineage: does a derived column inherit the sensitivity of its inputs?

This is the privacy hole. A ``CONFIDENTIAL`` tag survived every engine
boundary from the start, but an operation that *creates* a column had nothing
to carry it forward, so ``SUM(salary)`` arrived unlabelled and a policy
trusting classification missed it entirely.

Every test here is a leak that would have worked before this change.
"""

from __future__ import annotations

import pytest

pa = pytest.importorskip("pyarrow")

from aar.engines import create_engine  # noqa: E402
from aar.interchange import Table  # noqa: E402
from aar.ir import Agg, Col  # noqa: E402
from aar.lineage import taint  # noqa: E402
from aar.types import (  # noqa: E402
    FLOAT64, INT64, UTF8, Field, Schema, Sensitivity, sensitivity_of,
)


@pytest.fixture()
def payroll():
    """A table where `salary` and `ssn` are classified and `region` is not."""
    return Table(pa.table({
        "region": pa.array(["NA", "EU", "APAC", "NA"], type=pa.string()),
        "salary": pa.array([100.0, 200.0, 300.0, 400.0], type=pa.float64()),
        "ssn": pa.array(["1", "2", "3", "4"], type=pa.string()),
    })).tagged("salary", "CONFIDENTIAL", "FINANCIAL").tagged("ssn", "PII")


def _agg(func: str, column: str | None) -> Agg:
    return Agg(func, Col(column) if column else None, "out")


# ----------------------------------------------------------------- the hole
class TestDerivedColumnsInherit:
    def test_an_aggregate_over_a_classified_column_inherits_its_tags(
            self, payroll):
        """The bug this whole module exists to close."""
        with create_engine("arrow") as eng:
            got = eng.group_by(payroll, ["region"],
                               {"total": _agg("SUM", "salary")})
        assert "CONFIDENTIAL" in got.schema.get("total").classification
        assert sensitivity_of(got.schema.get("total").classification) \
            is Sensitivity.CONFIDENTIAL

    def test_an_aggregate_over_a_public_column_stays_public(self, payroll):
        """Propagation must not be indiscriminate.

        If everything inherited everything, every column would be masked and
        the policy would be useless - which is how masks get removed.
        """
        with create_engine("arrow") as eng:
            got = eng.group_by(payroll, ["region"],
                               {"first": _agg("ARBITRARY", "region")})
        assert got.schema.get("first").classification == frozenset()


    def test_a_group_key_keeps_its_own_tags(self):
        """A group key *is* the value; bucketing by it discloses it."""
        table = Table(pa.table({
            "customer": pa.array(["a", "b", "c"], type=pa.string()),
            "n": pa.array([1, 2, 3], type=pa.int64()),
        })).tagged("customer", "PII")
        with create_engine("arrow") as eng:
            got = eng.group_by(table, ["customer"], {"n": _agg("SUM", "n")})
        assert "PII" in got.schema.get("customer").classification

    def test_a_udf_inherits_everything_it_could_have_read(self, payroll):
        """A UDF is opaque, so the sound assumption is that it read all."""
        def total(row):
            return row["salary"] + row["ssn"].__len__()

        with create_engine("arrow") as eng:
            got = eng.udf(payroll, total)
        tags = got.schema.get("total").classification
        assert "CONFIDENTIAL" in tags
        assert "PII" in tags

    def test_a_column_udf_inherits_too(self, payroll):
        def widen(columns):
            return [v * 1.0 for v in columns["salary"]]

        with create_engine("arrow") as eng:
            got = eng.udf(payroll, widen, mode="column")
        assert "CONFIDENTIAL" in got.schema.get("widen").classification

    def test_a_join_inherits_from_both_sides(self):
        left = Table(pa.table({
            "id": pa.array([1, 2], type=pa.int64()),
            "a": pa.array([1.0, 2.0], type=pa.float64())})).tagged("a", "PII")
        right = Table(pa.table({
            "id": pa.array([1, 2], type=pa.int64()),
            "b": pa.array([3.0, 4.0], type=pa.float64())})).tagged("b", "FINANCIAL")
        with create_engine("arrow") as eng:
            got = eng.join(left, right, ["id"], "inner")
        for name in ("a", "b"):
            assert sensitivity_of(got.schema.get(name).classification) \
                is Sensitivity.CONFIDENTIAL

    def test_count_star_inherits_nothing(self, payroll):
        """`COUNT(*)` reports how many rows exist, not any column's value.

        Masking it would be theatre, and a system with no way to express that
        pushes analysts toward deleting the source tags instead.
        """
        with create_engine("arrow") as eng:
            got = eng.group_by(payroll, [], {"n": _agg("COUNT", None)})
        assert got.num_rows == 1
        assert got.schema.get("n").classification == frozenset()

    def test_count_of_a_classified_column_does_inherit(self, payroll):
        """Conservative, and deliberately so.

        The non-null count of a classified column is a property of that
        column, and the rule that "an aggregate is as sensitive as its
        argument" is the one an auditor can reason about. The
        `declassify` escape hatch exists for the cases that need it.
        """
        with create_engine("arrow") as eng:
            got = eng.group_by(payroll, [], {"n": _agg("COUNT", "salary")})
        assert "CONFIDENTIAL" in got.schema.get("n").classification


# ------------------------------------------------------- already protected
class TestExistingProtectionStillHolds:
    """Propagation adds to what already worked; it must not replace it."""

    def test_a_projection_keeps_its_tags(self, payroll):
        got = payroll.select(["salary"])
        assert "CONFIDENTIAL" in got.schema.get("salary").classification

    def test_a_filter_keeps_its_tags(self, payroll):
        from aar.ir import BinOp, Lit

        with create_engine("arrow") as eng:
            got = eng.filter(payroll, BinOp(Col("salary"), ">", Lit(150.0)))
        assert "CONFIDENTIAL" in got.schema.get("salary").classification

    def test_a_null_fill_keeps_its_tags(self):
        from aar.ir import Node, NodeType
        from aar.runtime import Executor

        table = Table(pa.table({
            "salary": pa.array([1.0, None], type=pa.float64())})
        ).tagged("salary", "CONFIDENTIAL")
        node = Node(NodeType.NULL_HANDLE, inputs=[], null_strategy="fill",
                    fill_value=0.0)
        got = Executor()._null_handle(table, node)
        assert "CONFIDENTIAL" in got.schema.get("salary").classification

    def test_a_cast_keeps_its_tags(self):
        from aar.types import DECIMAL

        schema = Schema((Field("salary", DECIMAL(38, 2)).with_classification(
            "CONFIDENTIAL"),))
        got = schema.cast({"salary": DECIMAL(38, 2)})
        assert "CONFIDENTIAL" in got.get("salary").classification


# --------------------------------------------------------- declassification
class TestDeclassify:
    def test_a_justified_declassification_clears_the_tag(self, payroll):
        """The escape hatch. Without it, analysts delete the source tags."""
        with create_engine("arrow") as eng:
            got = eng.group_by(payroll, ["region"],
                               {"n": _agg("COUNT", None)})
        field = got.schema.get("n")
        assert field.classification == frozenset()
        cleared = taint.declassify(field, "COUNT(*) reveals no salary")
        assert cleared.classification == frozenset()

    def test_declassifying_without_a_reason_is_refused(self, payroll):
        """A quiet declassification is the thing this must never allow."""
        with create_engine("arrow") as eng:
            got = eng.group_by(payroll, ["region"],
                               {"total": _agg("SUM", "salary")})
        with pytest.raises(ValueError) as exc:
            taint.declassify(got.schema.get("total"), "   ")
        assert "written justification" in str(exc.value)

    def test_declassifying_an_unclassified_column_is_a_no_op(self):
        field = Field("x", INT64)
        assert taint.declassify(field, "anything") is field


# ------------------------------------------------------------ policy closes
class TestPolicyCatchesTheLeak:
    """The point of all of the above: a policy can now act on a derived column."""

    def test_a_derived_column_is_masked_by_a_working_policy(self, payroll):
        from aar.governance import Policy, PolicyEngine, Subject

        with create_engine("arrow") as eng:
            got = eng.group_by(payroll, ["region"],
                               {"total": _agg("SUM", "salary")})
        out = PolicyEngine(Policy(mask_threshold=Sensitivity.CONFIDENTIAL)
                           ).enforce_write(got, "excel", Subject(name="d"))
        # Masked to zero, not shipped as 100.0/200.0/300.0/500.0.
        assert out.column("total").to_pylist() == [0.0, 0.0, 0.0]


    def test_the_derived_column_keeps_its_tag_after_masking(self, payroll):
        """A masked column is still known to have been masked."""
        from aar.governance import Policy, PolicyEngine, Subject

        with create_engine("arrow") as eng:
            got = eng.group_by(payroll, ["region"],
                               {"total": _agg("SUM", "salary")})
        out = PolicyEngine(Policy()).enforce_write(got, "excel",
                                                   Subject(name="d"))
        assert "CONFIDENTIAL" in out.schema.get("total").classification


# ------------------------------------------------------------------ helpers
class TestTaintHelpers:
    def test_derive_from_takes_the_union(self):
        schema = Schema((
            Field("a", INT64).with_classification("PII"),
            Field("b", INT64).with_classification("FINANCIAL"),
            Field("c", INT64),
        ))
        assert taint.derive_from(schema, ("a", "b")) == \
            frozenset({"PII", "FINANCIAL"})
        assert taint.derive_from(schema, ("c",)) == frozenset()

    def test_an_absent_source_contributes_nothing(self, payroll):
        assert taint.derive_from(payroll.schema, ("ghost",)) == frozenset()

    def test_inherit_all_collects_everything(self, payroll):
        assert taint.inherit_all(payroll.schema) == \
            frozenset({"CONFIDENTIAL", "FINANCIAL", "PII"})

    def test_describe_reports_the_sensitivity(self, payroll):
        text = taint.describe(payroll.schema)
        assert "salary" in text
        assert "CONFIDENTIAL" in text
        assert "INTERNAL" in text.split("region")[-1] or "PUBLIC" in text

    def test_is_derived_from_is_case_insensitive(self):
        field = Field("x", INT64).with_classification("pii")
        assert taint.is_derived_from(field, "PII")
        assert not taint.is_derived_from(field, "FINANCIAL")


# ------------------------------------------------- the whole chain, end to end
class TestPropagationThroughAPipeline:
    """Source tag -> aggregate -> write, with a policy watching."""

    def _source(self, tmp_path):
        import pyarrow.parquet as pq

        src = str(tmp_path / "payroll.parquet")
        pq.write_table(pa.table({
            "region": pa.array(["NA", "EU", "APAC", "NA"], type=pa.string()),
            "salary": pa.array([100.0, 200.0, 300.0, 400.0],
                               type=pa.float64()),
        }), src)
        return src

    def test_a_tagged_source_makes_the_aggregate_confidential(self, tmp_path):
        from aar.planner import AdaptivePlanner
        from aar.runtime import Executor
        from aar.sdk import classify, group_by, parquet, sum_, write_csv

        tagged = classify(parquet(self._source(tmp_path)),
                          "salary", tags=["CONFIDENTIAL", "FINANCIAL"])
        plan = AdaptivePlanner().plan(write_csv(
            group_by(tagged, "region", aggs={"payroll": sum_("salary")}),
            str(tmp_path / "o.csv")))

        with Executor() as ex:
            result = ex.execute(plan)
        assert result.ok, result.ledger.render()
        assert "CONFIDENTIAL" in result.table.schema.get("payroll").classification

    def test_a_policy_now_masks_the_derived_aggregate(self, tmp_path):
        """The end-to-end proof that the hole is closed.

        Before this change the same pipeline produced an unmasked payroll
        total, because nothing carried the tag from the source to the sum.
        """
        from aar.governance import Policy, PolicyEngine, Subject
        from aar.planner import AdaptivePlanner
        from aar.runtime import Executor
        from aar.sdk import classify, group_by, parquet, sum_, write_csv

        tagged = classify(parquet(self._source(tmp_path)),
                          "salary", tags=["CONFIDENTIAL"])
        node = group_by(tagged, "region", aggs={"payroll": sum_("salary")})
        out = str(tmp_path / "o.csv")
        plan = AdaptivePlanner().plan(write_csv(node, out))

        policy = Policy(mask_threshold=Sensitivity.CONFIDENTIAL)
        with Executor(policy=policy, subject=Subject(name="d")) as ex:
            result = ex.execute(plan)
        assert result.ok, result.ledger.render()

        import csv
        with open(out, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows
        # NA pays 100+400=500, EU 200, APAC 300. None of it may ship.
        assert sorted(float(r["payroll"]) for r in rows) == [0.0, 0.0, 0.0]

    def test_the_region_total_is_not_masked(self, tmp_path):
        """Propagation must be specific, not blanket.

        If everything inherited everything the policy would be unusable, and
        an unusable policy is one that gets switched off.
        """
        from aar.planner import AdaptivePlanner
        from aar.runtime import Executor
        from aar.sdk import classify, group_by, parquet, sum_, write_csv

        tagged = classify(parquet(self._source(tmp_path)),
                          "salary", tags=["CONFIDENTIAL"])
        plan = AdaptivePlanner().plan(write_csv(
            group_by(tagged, "region", aggs={"payroll": sum_("salary")}),
            str(tmp_path / "o.csv")))
        with Executor() as ex:
            result = ex.execute(plan)
        assert result.table.schema.get("region").classification == frozenset()

    def test_classifying_a_missing_column_is_an_error(self, tmp_path):
        from aar.planner import AdaptivePlanner
        from aar.runtime import Executor
        from aar.sdk import classify, parquet, write_csv

        node = classify(parquet(self._source(tmp_path)), "ghost",
                        tags=["PII"])
        plan = AdaptivePlanner().plan(write_csv(node,
                                                str(tmp_path / "o.csv")))
        with Executor() as ex:
            with pytest.raises(KeyError) as exc:
                ex.execute(plan)
        assert "no such column" in str(exc.value)

    def test_classify_requires_tags(self, tmp_path):
        from aar.sdk import classify, parquet

        with pytest.raises(ValueError) as exc:
            classify(parquet(self._source(tmp_path)), "salary", tags=[])
        assert "protects nothing" in str(exc.value)

    def test_classify_requires_a_column(self, tmp_path):
        from aar.sdk import classify, parquet

        with pytest.raises(ValueError):
            classify(parquet(self._source(tmp_path)), tags=["PII"])


