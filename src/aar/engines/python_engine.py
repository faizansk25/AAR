"""The Python UDF worker.

Arbitrary Python is a correctness boundary, not an optimisation choice: it
runs on the CPU, always, because that is the only place arbitrary Python
*can* run. The specification's insistence that this is a hard capability
limit rather than a preference is why the planner never offers a GPU for a
``PythonUDF`` node.

Two behaviours matter for trust:

* A failing UDF is reported with its name and the original exception, and
  the failure is recorded. It is never swallowed into a null column, which
  would turn a bug into wrong numbers.
* The whole batch is evaluated before the result is returned, so a UDF that
  raises halfway cannot leave a half-written output.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from ..capability import Device
from ..failures import UDFExecutionError
from ..interchange import Table
from ..ir import Expr, Node
from .base import Engine

__all__ = ["PythonWorkerEngine"]


class PythonWorkerEngine(Engine):
    """Executes Python UDFs in the current interpreter."""

    id = "python_worker"
    device = Device.CPU

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        #: Names of UDFs that raised, for the run's degradation report.
        self.failures: list[tuple[str, str]] = []

    def _unsupported(self, op: str) -> None:
        raise NotImplementedError(
            f"the Python worker cannot {op}; it exists only to run UDFs")

    def read_scan(self, node: Node) -> Table:
        self._unsupported("read data")

    def filter(self, table: Table, predicate: Expr) -> Table:
        self._unsupported("filter")

    def project(self, table: Table, columns: Sequence[str]) -> Table:
        self._unsupported("project")

    def group_by(self, table: Table, keys: Sequence[str],
                 aggs: dict[str, Any]) -> Table:
        self._unsupported("group")

    def sort(self, table: Table, keys: Sequence[tuple[str, bool]]) -> Table:
        self._unsupported("sort")

    def limit(self, table: Table, n: int) -> Table:
        self._unsupported("limit")

    def udf(self, table: Table, fn: Any, mode: str = "row") -> Table:
        """Run ``fn`` over the batch.

        The result is built completely before it is returned, so a failure
        cannot leave a partially-updated table behind.
        """
        name = getattr(fn, "__name__", "udf")
        started = time.perf_counter()
        try:
            from .arrow_engine import ArrowEngine
            result = ArrowEngine().udf(table, fn, mode)
        except Exception as exc:  # noqa: BLE001
            self.failures.append((name, f"{type(exc).__name__}: {exc}"))
            raise UDFExecutionError(
                f"UDF {name!r} failed on {table.num_rows} row(s): "
                f"{type(exc).__name__}: {exc}",
                udf=name, rows=table.num_rows) from exc
        self.last_ms = (time.perf_counter() - started) * 1e3
        return result

    def write(self, table: Table, node: Node) -> int:
        self._unsupported("write")
