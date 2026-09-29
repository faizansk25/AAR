"""Cross-engine metadata preservation and fail-closed governance.

Both of these encode bugs that were found and fixed rather than behaviours
that were designed in, so each test says what the failure looked like.

The first family is a *silent* privacy failure: an engine that round-trips a
table through its own native representation comes back as a bare Arrow table,
Arrow has nowhere to put an AAR classification, and the result is
unclassified - while the data, the plan and the numbers all stay correct. A
CONFIDENTIAL column becomes public and a policy that trusts classification
has nothing to act on.
"""

from __future__ import annotations

import pytest

pa = pytest.importorskip("pyarrow")

from aar.engines.factory import create_engine  # noqa: E402
from aar.interchange import Table, reconcile  # noqa: E402
from aar.ir import Agg, BinOp, Col, Lit  # noqa: E402
from aar.types import Field, INT64, Schema, UTF8  # noqa: E402

CONF = frozenset({"confidential"})
ENGINE_IDS = ("arrow", "duckdb", "polars_cpu", "pandas")


def tagged_table(rows=None):
    """A two-column table in which `salary` is CONFIDENTIAL."""
    schema = Schema((Field("region", UTF8),
                     Field("salary", INT64, classification=CONF)))
    data = rows if rows is not None else {
        "region": ["NA", "EU", "NA"], "salary": [100, 200, 300]}
    return Table(pa.table(data), schema)


def available_engines():
    """Only engines that are actually installed on this machine."""
    found = []
    for engine_id in ENGINE_IDS:
        try:
            create_engine(engine_id)
        except Exception:  # noqa: BLE001
            continue
        found.append(engine_id)
    return found


def engine_of(engine_id):
    return create_engine(engine_id)


def _sorted_rows(table):
    """A table's rows, order-independent.

    Group-by row order differs by engine by design, so parity is compared on
    content. Anything that needs a particular order has to ask for a sort.
    """
    return sorted(repr(row) for row in table.arrow.to_pylist())


class TestTagPreservation:
    def test_filter_keeps_the_tag(self):
        predicate = BinOp(Col("salary"), ">", Lit(100))
        for engine_id in available_engines():
            out = engine_of(engine_id).filter(tagged_table(), predicate)
            assert out.schema.get("salary").classification == CONF, engine_id

    def test_project_keeps_the_tag(self):
        for engine_id in available_engines():
            out = engine_of(engine_id).project(tagged_table(),
                                               ["region", "salary"])
            assert out.schema.get("salary").classification == CONF, engine_id

    def test_sort_keeps_the_tag(self):
        for engine_id in available_engines():
            out = engine_of(engine_id).sort(tagged_table(), [("salary", True)])
            assert out.schema.get("salary").classification == CONF, engine_id

    def test_limit_keeps_the_tag(self):
        for engine_id in available_engines():
            out = engine_of(engine_id).limit(tagged_table(), 2)
            assert out.schema.get("salary").classification == CONF, engine_id

    def test_an_aggregate_inherits_its_argument(self):
        """`SUM(salary)` is at least as sensitive as what it summed."""
        aggs = {"total": Agg("SUM", Col("salary"), "total")}
        for engine_id in available_engines():
            out = engine_of(engine_id).group_by(tagged_table(), ["region"],
                                                aggs)
            assert out.schema.get("total").classification == CONF, engine_id

    def test_count_star_stays_unclassified(self):
        """The one aggregate that cannot disclose a value stays clean."""
        aggs = {"n": Agg("COUNT", None, "n")}
        for engine_id in available_engines():
            out = engine_of(engine_id).group_by(tagged_table(), ["region"],
                                                aggs)
            assert not out.schema.get("n").classification, engine_id

    def test_a_join_merges_both_sides(self):
        """Either input can contribute to a joined row.

        This test used to build a tagged table and assert nothing, so the
        claim in its own docstring was never checked. A join that dropped one
        side's tags would leak: the column would keep its value and lose the
        CONFIDENTIAL label that governs whether it may leave.
        """
        from aar.engines import create_engine
        from aar.ir import JoinType

        left = tagged_table()
        right = tagged_table(rows={"region": ["NA", "EU", "NA"],
                                  "salary": [1, 2, 3]})
        engine = create_engine("arrow")
        joined = engine.join(left, right, ["region"], JoinType.INNER)
        # The joined `salary` is a function of a row from *both* sides, so it
        # must inherit the union of their tags rather than either one's.
        assert CONF <= set(joined.schema.get("salary").classification)
        # 3 rows each side; joining on `region` pairs NA with NA twice
        # (2 x 2 = 4 rows) and EU with EU once, so 5. Written out because
        # "however many rows came out" is not a useful assertion.
        assert joined.num_rows == 5


class TestEngineParity:
    """Two engines answering differently is a defect, not a variation.

    A plan may change engine as the cost model sees fit. It must not change
    the answer, and it must certainly not change the privacy label.
    """

    def _cases(self):
        return {
            "filter": lambda e, t: e.filter(
                t, BinOp(Col("salary"), ">", Lit(100))),
            "sort": lambda e, t: e.sort(t, [("salary", False)]),
            "limit": lambda e, t: e.limit(t, 2),
            "group_by": lambda e, t: e.group_by(
                t, ["region"], {"total": Agg("SUM", Col("salary"), "total")}),
        }

    def test_every_engine_agrees_with_arrow_on_the_values(self):
        """Compared as sets, because group order is not the contract.

        See `test_group_row_order_is_not_part_of_the_contract`: Arrow emits
        groups in first-appearance order and DuckDB in hash order. Anything
        that genuinely needs a particular order asks for a sort.
        """
        reference = engine_of("arrow")
        for name, run in self._cases().items():
            want = _sorted_rows(run(reference, tagged_table()))
            for engine_id in available_engines():
                got = _sorted_rows(run(engine_of(engine_id), tagged_table()))
                assert got == want, \
                    f"{engine_id} disagrees with arrow on {name}"

    def test_every_engine_agrees_with_arrow_on_the_labels(self):
        reference = engine_of("arrow")
        for name, run in self._cases().items():
            want = {f.name: f.classification
                    for f in run(reference, tagged_table()).schema.fields}
            for engine_id in available_engines():
                got = {f.name: f.classification for f in
                       run(engine_of(engine_id), tagged_table()).schema.fields}
                assert got == want, \
                    f"{engine_id} labels {name} differently from arrow"


class TestArrowSort:
    """The reference engine's sort was an AttributeError waiting to happen.

    `sort_indices` is a `pyarrow.compute` kernel, not a method on AAR's
    `Table` wrapper, and the call also had to translate the IR's booleans
    into Arrow's sort-order enum. Every other engine could sort; the
    reference one could not, and no test covered it.
    """

    def test_sort_actually_sorts(self):
        out = engine_of("arrow").sort(tagged_table(), [("salary", True)])
        assert out.column("salary").to_pylist() == [100, 200, 300]

    def test_descending_sorts_the_other_way(self):
        out = engine_of("arrow").sort(tagged_table(), [("salary", False)])
        assert out.column("salary").to_pylist() == [300, 200, 100]

    def test_sorting_a_string_column_works(self):
        out = engine_of("arrow").sort(tagged_table(), [("region", True)])
        assert out.column("region").to_pylist() == ["EU", "NA", "NA"]

    def test_sorting_an_empty_table_is_a_no_op(self):
        empty = tagged_table({"region": [], "salary": []})
        out = engine_of("arrow").sort(empty, [("salary", True)])
        assert out.num_rows == 0


class TestReconcile:
    """The boundary helper, tested directly so the engines need not be."""

    def test_carries_a_column_through_from_the_source(self):
        plain = Table(tagged_table().arrow.select(["region", "salary"]))
        fixed = reconcile(plain, source=tagged_table())
        assert fixed.schema.get("salary").classification == CONF

    def test_uses_a_derived_tag_for_a_new_column(self):
        plain = Table(tagged_table().arrow)
        fixed = reconcile(plain, derived={"region": frozenset({"pii"})})
        assert fixed.schema.get("region").classification == frozenset({"pii"})

    def test_leaves_a_genuinely_new_column_unclassified(self):
        """The honest answer: a column that came from nowhere is untagged,
        rather than inheriting something it has no relationship to."""
        source = tagged_table()
        renamed = Table(source.arrow.rename_columns(["a", "b"]))
        fixed = reconcile(renamed, source=source)
        assert not fixed.schema.get("a").classification
        assert not fixed.schema.get("b").classification

    def test_does_nothing_without_a_source_or_derived_tags(self):
        plain = Table(tagged_table().arrow)
        assert reconcile(plain) is plain

    def test_preserves_types_and_nullability(self):
        source = tagged_table()
        fixed = reconcile(Table(source.arrow), source=source)
        assert fixed.schema.get("salary").type == INT64
        assert fixed.num_rows == source.num_rows


class TestDerivedMetadata:
    """Operations that create columns, rather than preserving them."""

    def test_a_join_merges_both_sides(self):
        """A joined row can draw from either input, so it is as sensitive as
        the more sensitive of the two - here, the union of both."""
        right_schema = Schema((Field("region", UTF8),
                               Field("bonus", INT64,
                                     classification=frozenset({"pii"}))))
        right = Table(pa.table({"region": ["NA", "EU"],
                                "bonus": [5, 6]}), right_schema)
        for engine_id in available_engines():
            out = engine_of(engine_id).join(tagged_table(), right,
                                            ["region"], "inner")
            assert out.schema.get("bonus").classification == \
                frozenset({"confidential", "pii"}), engine_id

    def test_a_udf_inherits_everything_it_could_read(self):
        for engine_id in available_engines():
            out = engine_of(engine_id).udf(
                tagged_table(), lambda r: r["salary"] * 2, "row")
            produced = [n for n in out.column_names
                        if n not in ("region", "salary")]
            assert produced, engine_id
            assert out.schema.get(produced[0]).classification == CONF, engine_id

    def test_group_row_order_is_not_part_of_the_contract(self):
        """Pinning this as a known, deliberate non-determinism.

        A group-by emits rows in first-appearance order on the Arrow engine
        and in hash order on DuckDB and Polars. The *set* of groups is the
        contract; the order is not, and a plan that changes engine should
        not be able to surprise an analyst by reordering their output
        without a sort. Making it deterministic is a real change with a real
        cost, so it is recorded here rather than quietly assumed.
        """
        for engine_id in available_engines():
            out = engine_of(engine_id).group_by(
                tagged_table(), ["region"],
                {"total": Agg("SUM", Col("salary"), "total")})
            groups = sorted(r["region"] for r in out.arrow.to_pylist())
            # NA appears twice, so it is one group; EU is the other.
            assert groups == ["EU", "NA"], engine_id

    def test_a_sum_over_integers_is_an_integer_everywhere(self):
        """DuckDB types SUM(INTEGER) as DECIMAL(38,0) and hands back
        `Decimal('400')`; Arrow hands back `400`.

        Both are the right *number*, which is why this is easy to miss. But
        AAR's whole premise is that the engine a plan picks must not change
        what a caller receives, and a column arriving as `Decimal` on one run
        and `int` on the next is exactly the surprise the canonical type
        system exists to remove.
        """
        for engine_id in available_engines():
            out = engine_of(engine_id).group_by(
                tagged_table(), ["region"],
                {"total": Agg("SUM", Col("salary"), "total")})
            assert out.schema.get("total").type == INT64, engine_id
            for row in out.arrow.to_pylist():
                assert type(row["total"]) is int, engine_id


class TestDistinctIsHonouredEverywhere:
    """`DISTINCT` changes the answer, so every engine must apply it.

    The bug this pins: Polars' `SUM`/`AVG`/`MIN`/`MAX` were rendered as
    plain `col.sum()` etc. and only `COUNT` looked at `agg.distinct`. So
    `SUM(DISTINCT x)` quietly returned the sum *including duplicates* - the
    right shape of answer to a different question, with nothing to tell you.

    It surfaced only because the audit harness itself got the `Agg`
    constructor wrong (`Agg(func, arg, distinct)` - the third positional is
    `distinct`, not an alias), computed `SUM(DISTINCT ...)` by accident, and
    got a 38x-wrong answer from three engines. Worth recording both halves.
    """

    def _dupes(self):
        import pyarrow as pa

        from aar.interchange import Table
        from aar.types import Field, INT64, Schema, UTF8

        schema = Schema((Field("k", UTF8), Field("v", INT64)))
        return Table(pa.table({"k": ["a", "a", "a", "b"],
                               "v": [1, 2, 7, 7]}), schema)

    def test_sum_distinct_differs_from_sum_with_duplicates(self):
        """1+1+2+2 is 6 with duplicates and 3 distinct.

        A fixture whose two answers coincide tests nothing, so the values
        are chosen to make the difference the whole point of the assertion.
        """
        import pyarrow as pa

        from aar.interchange import Table
        from aar.types import Field, INT64, Schema, UTF8

        schema = Schema((Field("k", UTF8), Field("v", INT64)))
        table = Table(pa.table({"k": ["a", "a", "a", "a"],
                                "v": [1, 1, 2, 2]}), schema)
        for engine_id in available_engines():
            engine = engine_of(engine_id)
            distinct = engine.group_by(
                table, ["k"],
                {"total": Agg("SUM", Col("v"), distinct=True)})
            with_dupes = engine.group_by(
                table, ["k"], {"total": Agg("SUM", Col("v"))})
            got = distinct.arrow.to_pylist()[0]["total"]
            other = with_dupes.arrow.to_pylist()[0]["total"]
            assert (got, other) == (3, 6), (
                f"{engine_id}: SUM(DISTINCT)={got}, SUM={other}, "
                f"expected 3 and 6")

    def test_count_distinct_counts_values_not_rows(self):
        for engine_id in available_engines():
            out = engine_of(engine_id).group_by(
                self._dupes(), ["k"],
                {"n": Agg("COUNT", Col("v"), distinct=True)})
            by_key = {r["k"]: r["n"] for r in out.arrow.to_pylist()}
            assert by_key == {"a": 3, "b": 1}, f"{engine_id} gave {by_key}"


class TestEmptyGroupByKeepsTheLabel:
    """A group-by over no rows must still classify its output.

    Found by the large-data audit, then carried as a long-open risk until
    it was actually checked. The Arrow empty-group path built its
    aggregate columns with no tags, so `SUM(confidential)` came back
    unlabelled on arrow and pandas while DuckDB and Polars labelled it
    correctly. The rows are empty either way, which is why nobody noticed
    - but a policy that masks on classification would then behave
    differently depending on which engine the planner happened to pick.
    """

    def _empty(self):
        import pyarrow as pa

        from aar.interchange import Table
        from aar.types import Field, INT64, Schema, UTF8

        schema = Schema((Field("region", UTF8),
                         Field("salary", INT64, classification=CONF)))
        return Table(pa.table({"region": pa.array([], pa.string()),
                               "salary": pa.array([], pa.int64())}), schema)

    def test_the_derived_column_keeps_its_tag_on_every_engine(self):
        for engine_id in available_engines():
            out = engine_of(engine_id).group_by(
                self._empty(), ["region"],
                {"total": Agg("SUM", Col("salary"))})
            assert out.num_rows == 0
            assert out.column_names == ("region", "total"), engine_id
            assert out.schema.get("total").classification == CONF, (
                f"{engine_id} dropped the classification on an empty "
                f"group-by")

    def test_count_star_is_still_unclassified_on_every_engine(self):
        for engine_id in available_engines():
            out = engine_of(engine_id).group_by(
                self._empty(), ["region"],
                {"n": Agg("COUNT", None)})
            assert not out.schema.get("n").classification, engine_id


class TestNullSemanticsAgreeAcrossEngines:
    """Null handling must be identical everywhere, or it is wrong somewhere.

    The bug this pins: DuckDB rewrote `x IS NOT NULL` as `x <> NULL`. SQL's
    three-valued logic makes any comparison with NULL *unknown*, and unknown
    is not true, so that filter matched **zero rows** instead of every
    non-null one. It was found by the large-data audit, not by a unit test,
    because no small fixture had a null in it - and a filter that returns
    nothing raises no error anywhere.
    """

    def _with_nulls(self):
        import pyarrow as pa

        from aar.interchange import Table
        from aar.types import Field, INT64, Schema, UTF8

        schema = Schema((Field("region", UTF8), Field("n", INT64)))
        return Table(pa.table({"region": ["a", "b", "c", "d"],
                                "n": [1, None, 3, None]}), schema)

    def test_is_not_null_keeps_the_non_null_rows(self):
        for engine_id in available_engines():
            out = engine_of(engine_id).filter(
                self._with_nulls(), BinOp(Col("n"), "IS NOT", Lit(None)))
            assert out.num_rows == 2, (
                f"{engine_id} kept {out.num_rows} of 2 non-null rows")

    def test_is_null_keeps_the_null_rows(self):
        for engine_id in available_engines():
            out = engine_of(engine_id).filter(
                self._with_nulls(), BinOp(Col("n"), "IS", Lit(None)))
            assert out.num_rows == 2, (
                f"{engine_id} kept {out.num_rows} of 2 null rows")

    def test_a_comparison_never_matches_a_null(self):
        """`n > 0` must skip nulls on every engine, not include or drop them.

        The row count is the assertion; the point is that it is the same
        everywhere, because a null handling difference between engines is
        invisible until two reports disagree.
        """
        for engine_id in available_engines():
            out = engine_of(engine_id).filter(
                self._with_nulls(), BinOp(Col("n"), ">", Lit(0)))
            assert out.num_rows == 2, f"{engine_id} kept {out.num_rows} rows"




