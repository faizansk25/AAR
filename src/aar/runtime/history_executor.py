"""Vector-aware executor facade for execution-history observations.

The engine dispatch itself remains in :mod:`aar.runtime.executor`. This class
only changes the observation surface: every input is retained independently
and semantic/target keys are handed to the new execution-history API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .executor import ExecutionResult, Executor as _BaseExecutor

__all__ = ["ExecutionResult", "Executor", "NodeOutcome"]


@dataclass(slots=True)
class NodeOutcome:
    """What happened to one node, including its complete input shape."""

    node_id: str
    node_type: str
    engine_requested: str
    engine_used: str
    input_rows: tuple[int, ...] = ()
    input_bytes: tuple[int, ...] = ()
    rows_in: int = 0
    rows_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    elapsed_ms: float = 0.0
    error: str | None = None
    degraded: bool = False

    def render(self) -> str:
        flag = "  DEGRADED" if self.degraded else ""
        return (f"{self.node_type:<12} {self.engine_used:<14} "
                f"{self.rows_in:>9,} -> {self.rows_out:>9,} rows  "
                f"{self.elapsed_ms:>8.1f} ms{flag}")


class Executor(_BaseExecutor):
    """The existing executor with lossless measurement observations."""

    __slots__ = ()

    def execute(self, plan: Any) -> ExecutionResult:
        """Run every node, retaining all input row/byte counts per outcome."""
        result = ExecutionResult(table=None, started_at=time.time(),
                                 history=self._history)
        values: dict[int, Any] = {}

        # Imported here to keep this thin adapter layered on the executor
        # rather than re-exporting IR implementation details.
        from ..ir import NodeType, topological_order

        for node in topological_order(plan.root):
            started = time.perf_counter()
            inputs = [values[id(i)] for i in node.inputs if id(i) in values]
            requested = node.assigned_engine or "arrow"
            input_rows = tuple(int(t.num_rows) for t in inputs)
            input_bytes = tuple(int(t.nbytes) for t in inputs)
            outcome = NodeOutcome(
                node_id=node.id,
                node_type=str(node.type),
                engine_requested=requested,
                engine_used=("executor" if self._runs_in_executor(node.type)
                             else requested),
                input_rows=input_rows,
                input_bytes=input_bytes,
                rows_in=sum(input_rows),
                bytes_in=sum(input_bytes),
            )
            try:
                engine, degraded = self._engine(requested, result.ledger, node)
                outcome.engine_used = (
                    "executor" if self._runs_in_executor(node.type)
                    else engine.id)
                outcome.degraded = degraded
                table = self._dispatch(node, engine, inputs)
                if table is None:
                    raise RuntimeError(
                        f"engine {engine.id!r} returned None for "
                        f"{node.type.value} node {node.id}; an engine must "
                        f"return a Table, even an empty one")
            except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
                outcome.error = f"{type(exc).__name__}: {exc}"
                outcome.elapsed_ms = (time.perf_counter() - started) * 1e3
                result.outcomes.append(outcome)
                self._record_failure(result.ledger, node, exc)
                self._observe(result.history, outcome, node, success=False)
                result.finished_at = time.time()
                self._last_result = result
                raise

            outcome.rows_out = int(table.num_rows)
            outcome.bytes_out = int(table.nbytes)
            outcome.elapsed_ms = (time.perf_counter() - started) * 1e3
            values[id(node)] = table
            result.outcomes.append(outcome)
            self._observe(result.history, outcome, node, success=True)
            if node.type is NodeType.WRITE and node.target:
                result.written.append((node.target, table.num_rows))

        result.table = values.get(id(plan.root))
        result.finished_at = time.time()
        self._last_result = result
        return result

    @staticmethod
    def _observe(history: Any, outcome: NodeOutcome, node: Any,
                 success: bool) -> None:
        """Record semantic identity plus the complete input vector.

        Old custom history objects keep working through the legacy scalar call.
        The built-in :class:`aar.cost.ExecutionHistory` exposes
        ``operation_key`` and therefore receives the modern vector record.
        """
        if history is None:
            return
        record = getattr(history, "record", None)
        if record is None:
            return
        try:
            operation_key_fn = getattr(history, "operation_key", None)
            if callable(operation_key_fn):
                operation_key = operation_key_fn(node)
                record(
                    operation_key,
                    outcome.engine_used,
                    outcome.input_bytes,
                    outcome.input_rows,
                    outcome.elapsed_ms,
                    target_key=getattr(history, "target_key", ""),
                    output_bytes=(outcome.bytes_out if success else None),
                    output_rows=(outcome.rows_out if success else None),
                    success=success,
                )
            else:
                # Compatibility for user-provided history sinks written for
                # the old five-positional-argument protocol.
                record(node.id, outcome.engine_used, outcome.bytes_in,
                       outcome.rows_in, outcome.elapsed_ms, success=success)
        except Exception:  # noqa: BLE001
            # Learning must never break the run it is learning from.
            return
