"""Execution history keyed by logical operation and target identity.

The cost model originally recorded one scalar input size under ``Node.id``.
That shape is insufficient for persistent learning for two independent reasons:

* ``Node.id`` is a random per-build identifier, so the same logical operation
  parsed twice never meets its previous measurements;
* a multi-input operator such as a join was reduced to its first input, so a
  2 GB + 8 GB join was remembered as a 2 GB operation.

This module keeps the history API deliberately small while fixing both facts.
The caller supplies the semantic operation-key function; identity generation is
kept separate from storage so changing the canonical identity format does not
silently rewrite the statistical model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ..ir.nodes import Node

__all__ = ["ExecutionHistory", "ExecutionRecord"]


VectorLike = int | Iterable[int]
OperationKeyFn = Callable[[Node], str]


def _vector(value: VectorLike | None, *, name: str) -> tuple[int, ...]:
    """Canonicalise one scalar or an iterable into a non-negative tuple."""
    if value is None:
        return ()
    if isinstance(value, int):
        out = (int(value),)
    else:
        out = tuple(int(v) for v in value)
    if any(v < 0 for v in out):
        raise ValueError(f"{name} cannot contain negative values: {out!r}")
    return out


def _estimated_input_shape(node: Node, fallback: VectorLike) -> tuple[int, ...]:
    """Input-byte vector visible to the planner for one node.

    ``CostModel.compute_s`` still calls history with its historical scalar
    ``nbytes`` argument. For a join that scalar is normally the node/output
    estimate, while execution records the two *input* buffers. Regressing one
    against the other would train a model whose X axis changes meaning between
    planning and execution.

    Prefer each parent's measured profile, then its declared byte estimate.
    If any input is unknown, fall back to the scalar the caller supplied rather
    than inventing a partial vector that looks complete.
    """
    if not node.inputs:
        return _vector(fallback, name="input_bytes")
    sizes: list[int] = []
    for parent in node.inputs:
        profile = getattr(parent, "aar_profile", None)
        measured = int(getattr(profile, "nbytes", 0) or 0)
        declared = int(getattr(parent, "estimated_bytes", 0) or 0)
        size = measured or declared
        if size <= 0:
            return _vector(fallback, name="input_bytes")
        sizes.append(size)
    return tuple(sizes)


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """One observed execution, retaining every input independently.

    ``operation_key`` is a stable semantic identity supplied by the caller;
    ``target_key`` identifies the stable target machine/environment. Dynamic
    resource budgets belong beside a later persisted record as a snapshot,
    not inside either semantic key.

    ``None`` means a metric was not measured. Zero remains a real observation
    (an empty output, zero transfer, or genuinely zero measured peak), so using
    zero as the old sentinel would corrupt calibration data.
    """

    operation_key: str
    target_key: str
    engine: str
    input_bytes: tuple[int, ...]
    input_rows: tuple[int, ...]
    elapsed_ms: float
    output_bytes: int | None = None
    output_rows: int | None = None
    peak_memory: int | None = None
    bytes_transferred: int | None = None
    success: bool = True

    @property
    def total_input_bytes(self) -> int:
        return sum(self.input_bytes)

    @property
    def total_input_rows(self) -> int:
        return sum(self.input_rows)

    # Backward-compatible read aliases. Existing callers used these names for
    # the one scalar input; they now mean the total over the preserved vector.
    @property
    def operator_hash(self) -> str:
        return self.operation_key

    @property
    def nbytes(self) -> int:
        return self.total_input_bytes

    @property
    def rows(self) -> int:
        return self.total_input_rows


class ExecutionHistory:
    """Learn elapsed cost from semantic operations on one target.

    The history remains in-memory for now. Persistence is intentionally a
    separate step: first prove that record identity and observation shape are
    stable, then make those records durable.

    ``operation_key_fn`` is optional only for backward compatibility. Without
    it, a :class:`Node` lookup falls back to its transient ``node.id`` exactly
    as the previous implementation did. Supplying a semantic resolver makes
    independently-created nodes with the same meaning share history.
    """

    __slots__ = (
        "_records", "_ewma_alpha", "_min_samples_for_regression",
        "_operation_key_fn", "_target_key",
    )

    def __init__(
        self,
        ewma_alpha: float = 0.3,
        min_samples_for_regression: int = 4,
        *,
        operation_key_fn: OperationKeyFn | None = None,
        target_key: str = "",
    ) -> None:
        self._records: list[ExecutionRecord] = []
        self._ewma_alpha = float(ewma_alpha)
        self._min_samples_for_regression = int(min_samples_for_regression)
        self._operation_key_fn = operation_key_fn
        self._target_key = str(target_key)

    @property
    def target_key(self) -> str:
        return self._target_key

    def operation_key(self, node_or_key: Node | str) -> str:
        """Return the semantic key, or the legacy transient id when unwired."""
        if isinstance(node_or_key, str):
            key = node_or_key
        elif self._operation_key_fn is not None:
            key = self._operation_key_fn(node_or_key)
        else:
            key = node_or_key.id
        key = str(key).strip()
        if not key:
            raise ValueError("execution-history operation key cannot be empty")
        return key

    def record(
        self,
        operation_key: str,
        engine: str,
        input_bytes: VectorLike,
        input_rows: VectorLike,
        elapsed_ms: float,
        peak_memory: int | None = None,
        bytes_transferred: int | None = None,
        success: bool = True,
        *,
        target_key: str | None = None,
        output_bytes: int | None = None,
        output_rows: int | None = None,
    ) -> ExecutionRecord:
        """Record one observation without collapsing multi-input operators.

        Scalars remain accepted as a compatibility convenience and are stored
        as one-element tuples. New code should pass vectors explicitly.
        """
        ibytes = _vector(input_bytes, name="input_bytes")
        irows = _vector(input_rows, name="input_rows")
        if len(ibytes) != len(irows):
            raise ValueError(
                "input_bytes and input_rows must describe the same inputs: "
                f"{len(ibytes)} != {len(irows)}")
        key = self.operation_key(operation_key)
        engine = str(engine).strip()
        if not engine:
            raise ValueError("execution-history engine cannot be empty")
        target = self._target_key if target_key is None else str(target_key)
        for name, value in (
            ("output_bytes", output_bytes), ("output_rows", output_rows),
            ("peak_memory", peak_memory),
            ("bytes_transferred", bytes_transferred),
        ):
            if value is not None and int(value) < 0:
                raise ValueError(f"{name} cannot be negative: {value}")

        rec = ExecutionRecord(
            operation_key=key,
            target_key=target,
            engine=engine,
            input_bytes=ibytes,
            input_rows=irows,
            elapsed_ms=float(elapsed_ms),
            output_bytes=None if output_bytes is None else int(output_bytes),
            output_rows=None if output_rows is None else int(output_rows),
            peak_memory=None if peak_memory is None else int(peak_memory),
            bytes_transferred=(None if bytes_transferred is None
                               else int(bytes_transferred)),
            success=bool(success),
        )
        self._records.append(rec)
        return rec

    def __len__(self) -> int:
        return len(self._records)

    def records(
        self,
        operation_key: str,
        engine: str,
        *,
        target_key: str | None = None,
        successful_only: bool = True,
    ) -> list[ExecutionRecord]:
        """Records for one semantic operation, engine and target."""
        key = self.operation_key(operation_key)
        target = self._target_key if target_key is None else str(target_key)
        return [
            r for r in self._records
            if r.operation_key == key and r.engine == engine
            and r.target_key == target and (r.success or not successful_only)
        ]

    def predict(
        self,
        node_or_key: Node | str,
        engine_id: str,
        input_bytes: VectorLike,
        *,
        target_key: str | None = None,
    ) -> float | None:
        """Predicted seconds for this semantic operation on this target.

        The current estimator still regresses on *total* input bytes, preserving
        the explainable one-dimensional model. Crucially, the original vector
        remains stored in every record, so a later join-aware/multivariate
        estimator can be introduced without throwing the evidence away.

        When called from the existing :class:`CostModel`, derive the input
        vector from the node's parents. That keeps the regression's X axis the
        same at planning time and execution time even though ``compute_s``
        still passes its historical scalar size argument.
        """
        key = self.operation_key(node_or_key)
        shape = (_estimated_input_shape(node_or_key, input_bytes)
                 if isinstance(node_or_key, Node)
                 else _vector(input_bytes, name="input_bytes"))
        obs = self.records(key, engine_id, target_key=target_key)
        if not obs:
            return None
        if len(obs) == 1:
            return obs[-1].elapsed_ms / 1e3
        if len(obs) < self._min_samples_for_regression:
            return self._ewma(obs) / 1e3
        line = self._regress(obs)
        if line is not None:
            return max(0.0, line(sum(shape))) / 1e3
        return self._ewma(obs) / 1e3

    def _ewma(self, obs: list[ExecutionRecord]) -> float:
        value = obs[0].elapsed_ms
        for rec in obs[1:]:
            value = (self._ewma_alpha * rec.elapsed_ms
                     + (1 - self._ewma_alpha) * value)
        return value

    def _regress(self, obs: list[ExecutionRecord]) -> Callable[[int], float] | None:
        """Least-squares ``t = a + b*n`` over total input bytes."""
        pts = [(r.total_input_bytes, r.elapsed_ms) for r in obs]
        n = len(pts)
        sx = sum(p[0] for p in pts)
        sy = sum(p[1] for p in pts)
        sxx = sum(p[0] * p[0] for p in pts)
        sxy = sum(p[0] * p[1] for p in pts)
        det = n * sxx - sx * sx
        if det == 0:
            return None
        b = (n * sxy - sx * sy) / det
        a = (sy - b * sx) / n
        if b < 0:
            worst = max(p[1] for p in pts)
            return lambda _n: worst
        return lambda size: a + b * size

    def render(self) -> str:
        if not self._records:
            return "No execution history on this target."
        ok = [r for r in self._records if r.success]
        failed = len(self._records) - len(ok)
        lines = [f"Execution history: {len(ok)} successful, {failed} failed"]
        for engine in sorted({r.engine for r in self._records}):
            rs = [r for r in ok if r.engine == engine]
            if rs:
                total = sum(r.elapsed_ms for r in rs)
                lines.append(
                    f"  {engine:<14} {len(rs):>4} runs, {total:>9.1f} ms total")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """A diagnostic snapshot; persistence gets a versioned schema later."""
        return {
            "target_key": self._target_key,
            "records": [
                {
                    "operation_key": r.operation_key,
                    "target_key": r.target_key,
                    "engine": r.engine,
                    "input_bytes": list(r.input_bytes),
                    "input_rows": list(r.input_rows),
                    "output_bytes": r.output_bytes,
                    "output_rows": r.output_rows,
                    "elapsed_ms": r.elapsed_ms,
                    "peak_memory": r.peak_memory,
                    "bytes_transferred": r.bytes_transferred,
                    "success": r.success,
                }
                for r in self._records
            ],
        }
