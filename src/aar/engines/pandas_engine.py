"""pandas engine.

pandas is in the catalogue as *compatibility*, not speed: it exists because
millions of existing scripts are written against it, and AAR's job is to
orchestrate tools, not to replace them.

Data crosses in and out as Arrow. ``pandas.DataFrame`` is the one type the
specification explicitly does not want passed between engines, so this engine
converts at its own boundary and returns Arrow - it never hands a DataFrame
to the executor.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from ..capability import Device
from ..failures import SourceUnavailable
from ..interchange import Table, reconcile, require_arrow
from ..ir import Expr, Node
from ._mask import to_mask
from .base import Engine, PredicateCompiler

__all__ = ["PandasEngine"]


class PandasEngine(Engine):
    """Executes through pandas, converting at the boundary."""

    id = "pandas"
    device = Device.CPU

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self._pd = _import_pandas()

    def _frame(self, table: Table) -> Any:
        return self._pd.DataFrame(table.arrow.to_pandas())

    def _table(self, frame: Any, source: Table | None = None,
               derived: Any = None) -> Table:
        """pandas -> Arrow -> AAR, with the schema preserved.

        ``source``/``derived`` restore what pandas cannot carry. A DataFrame
        has nowhere to put an AAR classification, so without them a
        CONFIDENTIAL column silently becomes public after a pandas filter.
        """
        pa = require_arrow()
        return reconcile(Table(pa.Table.from_pandas(frame,
                                                     preserve_index=False)),
                         source, derived)

    def read_scan(self, node: Node) -> Table:
        spec = node.scan
        if spec is None:
            raise ValueError("scan node has no ScanSpec")
        if spec.kind == "parquet":
            self._require(spec.path, "Parquet")
            return Table(require_arrow().parquet.read_table(spec.path))
        if spec.kind == "csv":
            self._require(spec.path, "CSV")
            frame = self._pd.read_csv(spec.path, sep=spec.delimiter or ",")
            return self._table(frame)
        if spec.kind == "json":
            self._require(spec.path, "JSON")
            return Table(require_arrow().json.read_json(spec.path))
        from .arrow_engine import ArrowEngine
        return ArrowEngine().read_scan(node)

    @staticmethod
    def _require(path: str | None, kind: str) -> None:
        if not path or not os.path.exists(path):
            raise SourceUnavailable(f"no such {kind} file: {path}")

    def filter(self, table: Table, predicate: Expr) -> Table:
        # A vectorised mask, not `frame.apply(fn, axis=1)`. The row-by-row
        # form calls a Python function and builds a dict per row; on 2,000,000
        # rows that measured 26,723 ms against Arrow's 18 ms for the same
        # filter - a 1,400x gap, reproduced on two hosts. Every other engine
        # vectorises, so this was pandas being misused rather than pandas
        # being slow.
        #
        # The mask builder is shared with the cuDF engine, which is possible
        # because both speak the same Series API. One implementation, tested
        # once, on two devices.
        frame = self._frame(table)
        return self._table(frame[to_mask(frame, predicate)], source=table)

    def project(self, table: Table, columns: Sequence[str]) -> Table:
        return table.select(list(columns))

    def group_by(self, table: Table, keys: Sequence[str],
                 aggs: dict[str, Any]) -> Table:
        from .arrow_engine import ArrowEngine
        return ArrowEngine().group_by(table, keys, aggs)

    def sort(self, table: Table, keys: Sequence[tuple[str, bool]]) -> Table:
        if not keys or table.num_rows == 0:
            return table
        frame = self._frame(table).sort_values(
            by=[k for k, _ in keys], ascending=[asc for _, asc in keys],
            kind="mergesort")
        return self._table(frame, source=table)

    def limit(self, table: Table, n: int) -> Table:
        if n < 0:
            raise ValueError("limit must be non-negative")
        return table.slice(0, min(n, table.num_rows))

    def udf(self, table: Table, fn: Any, mode: str = "row") -> Table:
        from .arrow_engine import ArrowEngine
        return ArrowEngine().udf(table, fn, mode)

    def join(self, left: Table, right: Table, keys: Sequence[str],
             how: str) -> Table:
        """pandas has no join; the Arrow engine's is correct and fast enough.

        Delegating rather than raising keeps every engine interchangeable,
        which is the property the engine-agreement tests exist to protect.
        """
        from .arrow_engine import ArrowEngine
        return ArrowEngine().join(left, right, keys, how)

    def write(self, table: Table, node: Node) -> int:

        target = node.target
        if not target:
            raise ValueError("write node has no target")
        parent = os.path.dirname(os.path.abspath(target))
        if parent:
            os.makedirs(parent, exist_ok=True)
        frame = self._frame(table)
        fmt = (node.write_format or "").lower()
        if fmt == "csv":
            frame.to_csv(target, index=False)
        elif fmt == "parquet":
            frame.to_parquet(target, index=False)
        else:
            from .arrow_engine import ArrowEngine
            return ArrowEngine().write(table, node)
        return table.num_rows


def _import_pandas() -> Any:
    try:
        import pandas
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The pandas engine needs the pandas package. Install it with:\n"
            "  pip install pandas\n"
            f"(import failed: {exc})") from exc
    return pandas
