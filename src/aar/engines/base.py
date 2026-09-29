"""The execution-engine contract.

Every engine implements the same operations over the same Arrow-in /
Arrow-out signature. The executor never special-cases an engine: it calls
``engine.filter(table, predicate)`` and gets a :class:`Table` back, whichever
engine that is.

Two rules make this work and both are load-bearing:

* **Arrow in, Arrow out.** No engine returns a pandas frame or a Polars
  DataFrame. Converting at the engine boundary is exactly what the
  specification forbids, and letting one engine leak its native type would
  reintroduce the corruption the interchange layer exists to prevent.
* **Declare, do not discover.** An engine that cannot do an operation says
  so up front via :class:`EngineCapabilities`, so the planner never has to
  discover it by catching an exception mid-run.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..capability import Device
from ..interchange import Table
from ..ir import Expr, Node

__all__ = ["Engine", "EngineCapabilities", "PredicateCompiler"]


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    """What an engine will actually do, beyond what it claims to support.

    Claimed capability comes from the static catalogue and answers "could
    this engine in principle run this operation". This answers "will it, on
    this build, with this data". The two are separate because a Polars build
    without the right feature is a real and confusing failure mode.
    """

    #: Operations this engine declines at run time, with a reason.
    unsupported: dict[str, str] = field(default_factory=dict)
    #: Engine-specific knobs, e.g. a thread count.
    options: dict[str, Any] = field(default_factory=dict)

    def can(self, op: str) -> bool:
        return op not in self.unsupported

    def reason(self, op: str) -> str:
        return self.unsupported.get(op, "")


class Engine(abc.ABC):
    """Base class for execution engines.

    Subclasses implement what they support and leave the rest raising
    :class:`NotImplementedError`, which the executor converts into a
    *recorded degradation and a fallback* - never a silent wrong answer.
    """

    #: Matches the catalogue id, e.g. ``"duckdb"``.
    id: str = "engine"
    device: Device = Device.CPU

    def __init__(self, **options: Any) -> None:
        self._options = dict(options)
        self._capabilities = EngineCapabilities(options=dict(options))

    @property
    def capabilities(self) -> EngineCapabilities:
        return self._capabilities

    @property
    def options(self) -> dict[str, Any]:
        return dict(self._options)

    def _decline(self, op: str, reason: str) -> None:
        """Record that an operation is unavailable on this build."""
        self._capabilities = EngineCapabilities(
            unsupported={**self._capabilities.unsupported, op: reason},
            options=self._capabilities.options)

    def supports(self, op: str) -> bool:
        return self._capabilities.can(op)

    def close(self) -> None:  # noqa: B027 - deliberately not abstract
        """Release long-lived resources. Must be idempotent.

        Intentionally an empty concrete method rather than an abstract one:
        most engines have nothing to release (Arrow and pandas hold no
        connection), and making it abstract would force eight no-op
        overrides to express "nothing to do". An engine that *does* hold a
        resource - DuckDB's connection - overrides it.
        """

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.id}>"

    @abc.abstractmethod
    def read_scan(self, node: Node) -> Table:
        """Read a source described by ``node``."""

    @abc.abstractmethod
    def filter(self, table: Table, predicate: Expr) -> Table:
        """Keep rows matching ``predicate``."""

    @abc.abstractmethod
    def project(self, table: Table, columns: Sequence[str]) -> Table:
        """Keep and reorder columns."""

    @abc.abstractmethod
    def group_by(self, table: Table, keys: Sequence[str],
                 aggs: dict[str, Any]) -> Table:
        """Group by ``keys`` and aggregate."""

    @abc.abstractmethod
    def sort(self, table: Table, keys: Sequence[tuple[str, bool]]) -> Table:
        """Order by ``keys``."""

    @abc.abstractmethod
    def limit(self, table: Table, n: int) -> Table:
        """Keep the first ``n`` rows."""

    def write(self, table: Table, node: Node) -> int:
        """Write to the target in ``node``. Returns rows written."""
        raise NotImplementedError(f"{self.id} cannot write")

    def join(self, left: Table, right: Table, keys: Sequence[str],
             how: str) -> Table:
        raise NotImplementedError(f"{self.id} cannot join")

    def udf(self, table: Table, fn: Any, mode: str = "row") -> Table:
        raise NotImplementedError(f"{self.id} cannot run a Python UDF")


def _is_null(value: Any) -> bool:
    """Whether a value is SQL NULL, counting NaN as null.

    AAR reaches values by three different routes - Arrow, a Polars frame and
    a pandas DataFrame - and they do not agree on what a missing value looks
    like. Arrow gives ``None``; pandas widens an integer column with nulls to
    float and gives ``NaN``, which is *not* equal to ``None`` under ``!=``.
    Treating them as the same thing is what makes a filter mean the same
    thing on every engine.
    """
    if value is None:
        return True
    # `math.isnan` rather than the `value != value` idiom. Both are the same
    # test; one of them explains itself to the next reader and does not look
    # like a typo. Guarded by the isinstance check, so it cannot raise.
    return isinstance(value, float) and math.isnan(value)


def _is_null_literal(expr: Any) -> bool:
    """Whether an expression is the literal ``NULL``."""
    from ..ir import Lit

    return isinstance(expr, Lit) and expr.value is None


class PredicateCompiler:
    """Turns an IR predicate into a callable over an Arrow row.

    The default implementation evaluates row-wise in Python. It is correct on
    every engine and fast enough for small inputs; engines that can compile
    predicates natively override it. Correctness first is deliberate - a
    hand-written vectorised path for every operator and every type is exactly
    where subtle null-handling bugs live, and a wrong filter returns wrong
    numbers that nothing downstream can detect.
    """

    @staticmethod
    def evaluate(expr: Expr, row: dict[str, Any]) -> bool:
        """SQL-ish three-valued logic, reduced to a Python ``bool``."""
        from ..ir import BinOp, Col, Func, Lit, UnaryOp

        if isinstance(expr, Col):
            return bool(row.get(expr.name))
        if isinstance(expr, Lit):
            return bool(expr.value)
        if isinstance(expr, UnaryOp):
            return not PredicateCompiler.evaluate(expr.operand, row)
        if isinstance(expr, Func):
            return PredicateCompiler._call(expr, row)
        if isinstance(expr, BinOp):
            return PredicateCompiler._binary(expr, row)
        raise NotImplementedError(f"cannot evaluate {type(expr).__name__}")

    @staticmethod
    def _binary(expr: Any, row: dict[str, Any]) -> bool:

        op = expr.op
        if op == "AND":
            return (PredicateCompiler.evaluate(expr.left, row)
                    and PredicateCompiler.evaluate(expr.right, row))
        if op == "OR":
            return (PredicateCompiler.evaluate(expr.left, row)
                    or PredicateCompiler.evaluate(expr.right, row))

        # A null test must be a null test on every engine. Previously `IS NOT`
        # was `a != b`, which is right for a value and wrong for NULL: in
        # pandas an integer column with nulls arrives as float, so a null is
        # NaN, and `NaN != None` is True - the filter kept every null row.
        # `IS` was not handled at all and matched nothing. Both were silent.
        if op in ("IS", "IS NOT"):
            if _is_null_literal(expr.left):
                target = expr.right
            elif _is_null_literal(expr.right):
                target = expr.left
            else:
                raise NotImplementedError(
                    f"{op} is only defined against NULL, not against a value")
            value = PredicateCompiler._scalar(target, row)
            return ((not _is_null(value)) if op == "IS NOT"
                    else _is_null(value))

        a = PredicateCompiler._scalar(expr.left, row)
        b = PredicateCompiler._scalar(expr.right, row)

        if op in ("=", "<>", "!="):
            if _is_null(a) or _is_null(b):
                return False
            return a == b if op == "=" else a != b
        if _is_null(a) or _is_null(b):
            # SQL: a comparison with NULL is unknown, and unknown is not true.
            return False
        if op in (">", ">=", "<", "<="):
            try:
                return {">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b}[op]
            except TypeError:
                # Mixed types compare as text rather than raising partway
                # through a filter, which would lose the rows already taken.
                x, y = str(a), str(b)
                return {">": x > y, ">=": x >= y, "<": x < y, "<=": x <= y}[op]
        raise NotImplementedError(f"unsupported operator {op!r}")


    @staticmethod
    def _scalar(expr: Any, row: dict[str, Any]) -> Any:
        from ..ir import BinOp, Col, Lit

        if isinstance(expr, Col):
            return row.get(expr.name)
        if isinstance(expr, Lit):
            return expr.value
        if isinstance(expr, BinOp) and expr.op in (
                "+", "-", "*", "/", "AND", "OR", "IS", "IS NOT"):
            return PredicateCompiler.evaluate(expr, row)
        return PredicateCompiler.evaluate(expr, row)

    @staticmethod
    def _call(expr: Any, row: dict[str, Any]) -> bool:
        name = expr.name.lower()
        args = [PredicateCompiler._scalar(a, row) for a in expr.args]
        if name in ("isnull", "is_null"):
            return _is_null(args[0])
        if name in ("isnotnull", "is_not_null"):
            return not _is_null(args[0])
        if name == "coalesce":
            return any(not _is_null(a) for a in args)
        return bool(args[0]) if args else True

    @classmethod
    def compile(cls, expr: Expr) -> Any:
        """Return a callable taking a row dict and returning a bool."""
        def predicate(row: dict[str, Any]) -> bool:
            return cls.evaluate(expr, row)
        return predicate
