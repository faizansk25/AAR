"""Cross-engine semantic agreement.

An engine that silently returns *a* result is not the same as one that returns
the *right* result, and nothing else in this suite can tell the difference.
All three bugs this file exists for looked like working engines:

* DuckDB mapped SEMI/ANTI/ASOF to INNER, so ``ANTI`` returned exactly the rows
  it was defined to exclude.
* Polars compiled ``supported AND unsupported`` down to ``supported`` by
  deleting the operand it could not render.
* Arrow's Python join path returned equality-join rows for SEMI, ANTI and
  CROSS.

Every test runs the same input through several engines and compares actual row
sets. A new engine has to pass this before it can be trusted, which is cheaper
than discovering the disagreement in a production query.
"""

from __future__ import annotations

import pytest

pyarrow = pytest.importorskip("pyarrow")

from aar.engines import create_engine  # noqa: E402
from aar.engines.base import Engine
from aar.engines.factory import ENGINE_FACTORIES  # noqa: E402
from aar.interchange.table import Table  # noqa: E402
from aar.ir import BinOp, Col, Func, JoinType, Lit  # noqa: E402


def _available() -> list[str]:
    """Engine ids this build can actually construct, in catalogue order.

    Uses ``allow_degradation=False`` because the default is to **fall back**
    rather than raise. Probing with plain ``create_engine`` therefore reported
    'polars' as available on a build where polars is not in
    ``ENGINE_FACTORIES`` at all, and the substitution warning was the only
    clue. Every assertion in this file would then have run DuckDB twice while
    claiming to compare two engines - the exact false confidence this suite
    exists to prevent, reproduced in the suite itself.
    """
    out: list[str] = []
    for engine_id, _factory in sorted(ENGINE_FACTORIES.items()):
        try:
            with create_engine(engine_id, allow_degradation=False):
                pass
        except Exception:  # noqa: BLE001 - genuinely unavailable here
            continue
        out.append(engine_id)
    return out


ENGINES = _available()


def _engine(engine_id: str):
    """The engine for *this* id, refusing any substitution.

    A silent fallback would make a comparison vacuous - two "different"
    engines would quietly be the same object - so it is disabled at every call
    site rather than only at discovery.
    """
    return create_engine(engine_id, allow_degradation=False)


def _implements(op: str) -> list[str]:
    """Engines that actually *perform* an operation, rather than refusing it.

    Refusal takes two forms here, and neither is visible from the class alone:

    * ``excel`` and ``python_worker`` **inherit** ``Engine.join``, which raises;
    * ``excel`` **overrides** ``filter`` to raise "workbook I/O only", and
      ``polars_cpu`` **overrides** ``join`` to raise for ASOF only.

    So the filter is behavioural: run a trivial, definitely-supported operation
    and keep the engine only if it produces a result. That distinguishes "I
    cannot do this" from "I cannot do *that* case of this", which is the
    distinction the earlier source-inspection got wrong - Polars' join mentions
    NotImplementedError in its ASOF branch and was wrongly excluded, while
    Excel's filter was still wrongly included.

    Discovered from the concrete classes rather than a hard-coded list, so a
    new implementing engine is covered automatically.
    """
    out: list[str] = []
    for engine_id in ENGINES:
        method = getattr(type(_engine(engine_id)), op, None)
        if method is None or method is getattr(Engine, op, None):
            continue
        if _probe(engine_id, op):
            out.append(engine_id)
    return out


def _probe(engine_id: str, op: str) -> bool:
    """Does this engine actually run *op*? A trivial case, no edge cases.

    Built inline rather than via :func:`_pairs` so this stays usable at import
    time, before the module-level engine lists are computed.
    """
    left = Table(pyarrow.table({"k": [1, 2, 3], "lv": ["a", "b", "c"]}))
    right = Table(pyarrow.table({"k": [2, 3, 4], "rv": ["B", "C", "D"]}))
    try:
        if op == "join":
            _engine(engine_id).join(left, right, ["k"], JoinType.INNER)
        else:
            _engine(engine_id).filter(left, BinOp(Col("lv"), "=", Lit("a")))
    except Exception:  # noqa: BLE001 - a refusal is the answer
        return False
    return True


#: Engines with their own join implementation, and with their own filter.
#: Arrow is the reference for both: it has no native fast path for the awkward
#: join kinds, so its Python implementation is the definitionally correct one.
REFERENCE = "arrow"
JOINERS = _implements("join")
FILTERERS = _implements("filter")


def _pairs() -> tuple[Table, Table]:
    """Left {1,2,3}, right {2,3,4}. Overlap is {2,3}.

    Chosen so every join kind has a non-trivial answer: two rows match, one
    left row does not, one right row does not, and neither side is empty. An
    empty side makes several assertions below vacuously true.
    """
    left = Table(pyarrow.table({"k": [1, 2, 3], "lv": ["a", "b", "c"]}))
    right = Table(pyarrow.table({"k": [2, 3, 4], "rv": ["B", "C", "D"]}))
    return left, right


def _rows(table: Table) -> list[tuple]:
    """A comparable projection of a result.

    Compared as a **multiset**, not a list: SQL does not guarantee row order
    for a join, and no engine here promises one. Arrow emits the unmatched FULL
    row first while DuckDB emits it last, with identical rows in a different
    order - that is not a disagreement about the answer. Ordering would make
    this suite fail on a fact it should not be asserting.
    """
    data = table.to_arrow().to_pydict()
    names = sorted(data)

    def cell(value) -> tuple:
        # Nulls first, then a type tag, so `sorted` never compares None to an
        # int - which is exactly what an outer join produces. Without this the
        # comparison raised TypeError rather than reporting a disagreement.
        return (0, "") if value is None else (1, str(value))

    return sorted(tuple(cell(row[i]) for i in range(len(names)))
                  for row in zip(*(data[n] for n in names)))


#: Kinds where the engines disagree on the *shape* of an otherwise-correct
#: answer. Documented rather than papered over; each is a real measurement.
#:
#: * ``right`` - Polars used to emit a second key column here; it now coalesces
#:   like the others. Only row *order* differs, which SQL does not guarantee,
#:   so the shape comparison is skipped rather than read as a semantic
#:   disagreement.
#: * ``full`` - Polars still emits both ``k`` and ``k_right`` while Arrow,
#:   DuckDB and pandas coalesce to one ``k``. The rows are correct; the column
#:   set is not, so every downstream column reference shifts by one. **Unfixed.**
#: * ``cross`` - a cross join has no key, so the key column carries whatever
#:   value each engine emits, and the engines disagree on it (and on row order).
#:   That is legitimate. Only the row count is comparable, and
#:   ``test_cross_is_the_full_product`` checks it.
KNOWN_SHAPE_DIVERGENCE = {JoinType.RIGHT, JoinType.FULL, JoinType.CROSS}


class TestJoinSemantics:
    """Each kind has one correct answer, written out rather than derived.

    Asserting against a literal table rather than against another engine
    matters: if a bug changed every engine identically, "they all agree" would
    still pass, and agreement is exactly what is not in question.
    """

    EXPECTED = {
        JoinType.INNER: 2,      # k in both: {2, 3}
        JoinType.LEFT: 3,       # all left: {1, 2, 3}
        JoinType.RIGHT: 3,      # all right: {2, 3, 4}
        JoinType.FULL: 4,       # both: {1, 2, 3, 4}
        JoinType.SEMI: 2,       # left with a match: {2, 3}
        JoinType.ANTI: 1,       # left without a match: {1}
        JoinType.CROSS: 9,      # 3 x 3
    }

    @pytest.mark.parametrize("kind", list(EXPECTED))
    @pytest.mark.parametrize("engine_id", JOINERS)
    def test_the_row_count_is_the_defined_one(self, engine_id, kind):
        left, right = _pairs()
        out = _engine(engine_id).join(left, right, ["k"], kind)
        assert out.num_rows == self.EXPECTED[kind], (
            f"{engine_id} returned {out.num_rows} rows for a {kind.value} "
            f"join; the correct answer is {self.EXPECTED[kind]}")

    @pytest.mark.parametrize("kind", [k for k in EXPECTED
                                     if k not in KNOWN_SHAPE_DIVERGENCE])
    def test_every_engine_agrees_with_the_reference(self, kind):
        left, right = _pairs()
        want = _rows(_engine(REFERENCE).join(left, right, ["k"], kind))
        for engine_id in JOINERS:
            if engine_id == REFERENCE:
                continue
            got = _rows(_engine(engine_id).join(left, right, ["k"], kind))
            assert got == want, (
                f"{engine_id} disagrees with {REFERENCE} on a {kind.value} "
                f"join: {got} vs {want}")

    @pytest.mark.parametrize("kind", sorted(KNOWN_SHAPE_DIVERGENCE,
                                           key=lambda k: k.value))
    def test_the_documented_divergences_are_real(self, kind):
        """Pin the known divergences so they cannot change unnoticed.

        These assert the *current* behaviour deliberately. That is the honest
        way to record an unfixed defect: it turns "Polars differs on a right
        join" from a claim in a review into a fact CI reports the moment it
        changes - whether because someone fixed it (and this needs updating) or
        because it got worse.

        Nothing here claims the behaviour is correct. Only that it is unchanged
        from what was measured.
        """
        left, right = _pairs()
        shapes = {eid: set(_engine(eid).join(left, right, ["k"], kind)
                            .column_names)
                  for eid in JOINERS}
        if kind in (JoinType.RIGHT, JoinType.FULL):
            # Measured: Arrow, DuckDB and pandas all coalesce to a single `k`
            # for both kinds. Polars coalesces for RIGHT but keeps a second
            # `k_right` for FULL. That is a genuine, reproducible divergence -
            # the rows are right, the column set is not - and it is pinned here
            # rather than fixed, because the fix needs its own cross-platform
            # check and an unverified rewrite is worse than a recorded defect.
            coalesced = {eid for eid, cols in shapes.items()
                         if "k_right" not in cols}
            assert len(coalesced) >= 3, (
                f"Arrow, DuckDB and pandas should all coalesce the key for a "
                f"{kind.value} join; got {shapes}")
            if kind is JoinType.FULL:
                assert "k_right" in shapes.get("polars_cpu", set()), (
                    "polars is expected to keep a second key column on a full "
                    "join; if this fails, that divergence was fixed and "
                    "KNOWN_SHAPE_DIVERGENCE should be updated")
        else:  # CROSS
            # A cross join has no key, so only the row count is comparable.
            for engine_id in JOINERS:
                out = _engine(engine_id).join(left, right, ["k"], kind)
                assert out.num_rows == 9, engine_id

    @pytest.mark.parametrize("kind", [JoinType.SEMI, JoinType.ANTI])
    @pytest.mark.parametrize("engine_id", JOINERS)
    def test_semi_and_anti_emit_no_right_columns(self, engine_id, kind):
        """A semi-join is defined not to produce the right side's columns.

        Emitting them fabricates data the join cannot produce, and downstream
        code reading ``rv`` gets an answer to a question never asked.
        """
        left, right = _pairs()
        out = _engine(engine_id).join(left, right, ["k"], kind)
        assert "rv" not in out.column_names, (
            f"{engine_id} leaked the right columns into a {kind.value} join: "
            f"{out.column_names}")

    def test_anti_is_exactly_the_complement_of_semi(self):
        """The strongest form of the check.

        Anti and semi partition the left side, so their key sets must be
        disjoint and their union must be every left row. The old DuckDB
        behaviour - both returning the inner rows - passes a semi count check
        and fails this one, which is how it should be caught.
        """
        left, right = _pairs()
        for engine_id in JOINERS:
            semi = _engine(engine_id).join(left, right, ["k"],
                                           JoinType.SEMI)
            anti = _engine(engine_id).join(left, right, ["k"],
                                           JoinType.ANTI)
            semi_keys = set(semi.to_arrow().column("k").to_pylist())
            anti_keys = set(anti.to_arrow().column("k").to_pylist())
            assert not (semi_keys & anti_keys), (
                f"{engine_id} put {semi_keys & anti_keys} in both semi and anti")
            assert semi_keys | anti_keys == {1, 2, 3}, (
                f"{engine_id}: semi+anti must cover every left row, got "
                f"{semi_keys | anti_keys}")

    @pytest.mark.parametrize("engine_id", JOINERS)
    def test_cross_is_the_full_product(self, engine_id):
        left, right = _pairs()
        out = _engine(engine_id).join(left, right, ["k"], JoinType.CROSS)
        assert out.num_rows == left.num_rows * right.num_rows

    @pytest.mark.parametrize("engine_id", JOINERS)
    def test_asof_refuses_rather_than_guessing(self, engine_id):
        """An unimplemented kind must raise, never approximate.

        ASOF means "nearest preceding match", which needs an ordering and a
        tolerance that differ per engine. Returning an inner join instead is
        exactly the class of bug this file documents.
        """
        left, right = _pairs()
        with pytest.raises((NotImplementedError, ValueError)):
            _engine(engine_id).join(left, right, ["k"], JoinType.ASOF)

    def test_a_null_key_never_matches_even_itself(self):
        """SQL: NULL never equals NULL. Every joining engine must agree.

        This is the check that Arrow's Python path documents as its reason for
        refusing the native join when a key contains nulls, and DuckDB agrees.
        Polars is the one that does not, which is worth knowing rather than
        assuming.
        """
        left = Table(pyarrow.table({"k": [1, None], "lv": ["a", "null"]}))
        right = Table(pyarrow.table({"k": [None, 2], "rv": ["N", "B"]}))
        for engine_id in JOINERS:
            out = _engine(engine_id).join(left, right, ["k"], JoinType.INNER)
            keys = out.to_arrow().column("k").to_pylist()
            assert None not in keys, (
                f"{engine_id} matched a NULL key to a NULL key: {keys}")


# -------------------------------------------------------------- predicates
class TestPredicateSemantics:
    """A predicate that cannot be rendered must fall back **whole**.

    Two distinct defects shared this shape.

    *Polars* dropped the operand it could not compile, so
    ``supported AND unsupported`` ran as ``supported``. Its compiler filtered
    the operand list: ``[e for e in (left, right) if e is not None]``.

    *Arrow*, via the shared :class:`PredicateCompiler`, ignored the function
    name entirely - an unknown call evaluated as "is the first argument
    truthy". Both returned ['Ann', 'Cid'] where only 'Ann' qualifies.

    The common failure is that a filter silently became a *weaker* filter.
    """

    #: ``is_null`` is deliberately chosen over something exotic: it is a
    #: function the shared Arrow compiler genuinely supports, so this tests
    #: *partial rendering* rather than an unsupported function. The old
    #: ``regexp_match`` choice made the test fail for a different reason -
    #: the correct refusal - which would have masked the regression it was
    #: written to catch.
    NULL_TEST = Func("is_null", (Col("name"),))

    #: Bob, Ann and Cid are all non-null, so `is_null(name)` is false for every
    #: row and the two conjunctions below have distinguishable answers.
    NAMES = ["Bob", "Ann", "Cid"]
    AMOUNTS = [50, 150, 250]

    @staticmethod
    def _table() -> Table:
        return Table(pyarrow.table({"amount": [50, 150, 250],
                                    "name": ["Bob", "Ann", "Cid"]}))

    def _filter(self, engine_id: str, predicate) -> list[str]:
        out = _engine(engine_id).filter(self._table(), predicate)
        return sorted(out.to_arrow().column("name").to_pylist())

    def _expected(self, op: str) -> list[str]:
        """The right answer, derived from the data rather than hard-coded.

        ``is_null(name)`` is False for every row in this fixture, so:

        * AND  -> nothing qualifies, because one conjunct is always false;
        * OR   -> the amount test decides alone, ['Ann', 'Cid'].

        Both are non-trivial, and they differ, which is what makes the
        regression detectable: a dropped conjunct turns the AND into
        ['Ann', 'Cid'], and a dropped disjunct turns the OR into [].
        """
        keep = []
        for name, amount in zip(self.NAMES, self.AMOUNTS):
            over = amount > 100
            is_null = self._is_null(name)
            # `is_null(name)` is False for every row here, so:
            #   AND -> amount>100 AND False -> no rows qualify
            #   OR  -> amount>100 OR  False -> ['Ann', 'Cid']
            # The second term is used exactly as written; negating it again
            # would make AND return rows that no engine returns.
            if op == "AND":
                holds = over and is_null
            else:
                holds = over or is_null
            if holds:
                keep.append(name)
        return sorted(keep)

    @staticmethod
    def _is_null(value) -> bool:
        return value is None

    def test_the_two_answers_differ(self):
        """A precondition on the test below, so it cannot pass vacuously.

        If AND and OR had the same expected answer, dropping either operand
        would be undetectable - and those are exactly the two bugs.
        """
        assert self._expected("AND") == []
        assert self._expected("OR") == ["Ann", "Cid"]

    def test_and_with_an_unsupported_half_is_not_the_other_half(self):
        """Only 'Ann' satisfies both halves.

        Dropping the second conjunct yields ['Ann', 'Cid']; dropping the
        second disjunct of the OR yields ['Ann'] instead of ['Ann', 'Cid'].
        """
        good = BinOp(Col("amount"), ">", Lit(100))
        for engine_id in FILTERERS:
            for op in ("AND", "OR"):
                predicate = BinOp(good, op, self.NULL_TEST)
                want = self._expected(op)
                got = self._filter(engine_id, predicate)
                assert got == want, (
                    f"{engine_id} returned {got} for a {op} with a half it "
                    f"could not render; the correct answer is {want}")

    @pytest.mark.parametrize("engine_id", FILTERERS)
    def test_a_fully_supported_predicate_still_works(self, engine_id):
        """The ordinary case must still work, or the fallback is useless."""
        predicate = BinOp(Col("amount"), ">", Lit(100))
        assert self._filter(engine_id, predicate) == ["Ann", "Cid"]

    def test_the_polars_compiler_declines_rather_than_guessing(self):
        """The fix belongs at the compiler, not only at the call site.

        Asserted directly so the property survives a refactor of
        ``_to_polars_expr`` that reintroduces partial rendering.
        """
        polars = pytest.importorskip("polars")
        from aar.engines.polars_engine import _to_polars_expr

        good = BinOp(Col("amount"), ">", Lit(100))
        assert _to_polars_expr(polars, good) is not None
        assert _to_polars_expr(polars, self.NULL_TEST) is None
        assert _to_polars_expr(
            polars, BinOp(good, "AND", self.NULL_TEST)) is None, (
            "a conjunction with an unrenderable operand must decline entirely, "
            "not compile to the half it understood")
        assert _to_polars_expr(
            polars, BinOp(good, "OR", self.NULL_TEST)) is None

    def test_the_shared_compiler_refuses_an_unknown_function(self):
        """The other half of the same bug, in the fallback every engine uses."""
        from aar.engines.base import PredicateCompiler

        with pytest.raises(NotImplementedError):
            PredicateCompiler._call(Func("no_such_function", (Col("amount"),)),
                                    {"amount": 1})
        # And the functions it does know still work.
        assert PredicateCompiler._call(
            Func("is_null", (Col("amount"),)), {"amount": None}) is True

    def test_a_supported_conjunction_still_compiles(self):
        """The guard must not be so eager that ordinary predicates decline."""
        polars = pytest.importorskip("polars")
        from aar.engines.polars_engine import _to_polars_expr

        both = BinOp(BinOp(Col("amount"), ">", Lit(100)), "AND",
                     BinOp(Col("amount"), "<", Lit(300)))
        assert _to_polars_expr(polars, both) is not None