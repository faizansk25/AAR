"""The Excel engine.

Excel is catalogued as an engine because the planner reasons about it like
one - it is a real local-I/O node with its own costs. Giving it a real
implementation matters more than it looks: without one, the planner
legitimately chooses ``excel``, the factory has to degrade, and every Excel
pipeline opens with a recorded substitution it never asked for. The plan
should describe what actually ran.

It delegates compute to the Arrow engine. Excel reading and writing are
genuinely slow for what they do; a workbook is a ZIP of XML, not a columnar
format. The engine exists to make that cost honest, not to pretend otherwise.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..capability import Device
from ..interchange import Table
from ..ir import Expr, Node, NodeType
from .arrow_engine import ArrowEngine
from .base import Engine

__all__ = ["ExcelEngine"]


class ExcelEngine(Engine):
    """Reads and writes workbooks; delegates compute to Arrow."""

    id = "excel"
    device = Device.LOCAL_IO

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self._compute = ArrowEngine()

    # ------------------------------------------------------------------ read
    def read_scan(self, node: Node) -> Table:
        from ..connectors.excel import read_excel

        spec = node.scan
        if spec is None or spec.kind != "excel":
            return self._compute.read_scan(node)
        return read_excel(spec)

    # ---------------------------------------------------------------- write
    def write(self, table: Table, node: Node) -> int:
        from ..connectors.excel import write_excel

        target = node.target
        if not target:
            raise ValueError("write node has no target")
        if (node.write_format or "").lower() != "excel":
            return self._compute.write(table, node)
        sheet = node.scan.sheet if node.scan else None
        mode = node.write_mode or "overwrite"
        return write_excel(table, target, sheet=sheet, mode=mode)

    # ---------------------------------------------------------------- others
    def _delegate(self, op: str) -> None:
        raise NotImplementedError(
            f"the Excel engine cannot {op}; it handles only workbook I/O. "
            f"The planner should not assign it a compute node.")

    def filter(self, table: Table, predicate: Expr) -> Table:
        self._delegate("filter")

    def project(self, table: Table, columns: Sequence[str]) -> Table:
        self._delegate("project")

    def group_by(self, table: Table, keys: Sequence[str],
                 aggs: dict[str, Any]) -> Table:
        self._delegate("group")

    def sort(self, table: Table, keys: Sequence[tuple[str, bool]]) -> Table:
        self._delegate("sort")

    def limit(self, table: Table, n: int) -> Table:
        self._delegate("limit")
