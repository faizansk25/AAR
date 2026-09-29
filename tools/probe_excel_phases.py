"""Where does the Excel write actually spend its time?

The rewrite assumed `ws.cell()` per cell was the cost. The benchmark says
otherwise: `append` per row is exactly as slow, 36 seconds either way. So
the hypothesis was wrong and the phase breakdown below replaces it with a
measurement.

openpyxl keeps a `Cell` object per cell in `ws._cells` no matter which API
wrote it, so `wb.save()` has the same work to do either way. The hypothesis
that survives is that the *save* dominates, and that only `write_only=True`
avoids it - a normal workbook must hold every cell in memory to serialise
it, while a write-only sheet streams each row out as it arrives.

Run: python tools/probe_excel_phases.py [rows] [cols]
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import pyarrow as pa

from aar.interchange import Table
from aar.connectors.excel import _excel_value, _openpyxl


def build(rows: int, cols: int) -> Table:
    return Table(pa.table({f"c{i}": pa.array(list(range(rows)), type=pa.int64())
                           for i in range(cols)}))


def old_form(t: Table, out: str) -> tuple[float, float, float]:
    """to_pylist + one ws.cell per cell, then save."""
    wb = _openpyxl().Workbook()
    ws = wb.active
    names = t.column_names
    t0 = time.perf_counter()
    for c, name in enumerate(names, start=1):
        ws.cell(row=1, column=c, value=name)
    cursor = 2
    for record in t.arrow.to_pylist():
        for c, name in enumerate(names, start=1):
            ws.cell(row=cursor, column=c, value=record.get(name))
        cursor += 1
    t1 = time.perf_counter()
    wb.save(out)
    t2 = time.perf_counter()
    return (t1 - t0) * 1000, (t2 - t1) * 1000, (t2 - t0) * 1000


def append_form(t: Table, out: str) -> tuple[float, float, float]:
    """One ws.append per row, then save."""
    wb = _openpyxl().Workbook()
    ws = wb.active
    t0 = time.perf_counter()
    ws.append(list(t.column_names))
    cols = t.arrow.to_pydict()
    for row in zip(*(cols.get(n, []) for n in t.column_names)):
        ws.append([_excel_value(v) for v in row])
    t1 = time.perf_counter()
    wb.save(out)
    t2 = time.perf_counter()
    return (t1 - t0) * 1000, (t2 - t1) * 1000, (t2 - t0) * 1000


def write_only_form(t: Table, out: str) -> tuple[float, float, float]:
    """write_only=True: the sheet streams rows to disk, holding no cells."""
    wb = _openpyxl().Workbook(write_only=True)
    ws = wb.create_sheet()
    t0 = time.perf_counter()
    ws.append(list(t.column_names))
    cols = t.arrow.to_pydict()
    for row in zip(*(cols.get(n, []) for n in t.column_names)):
        ws.append([_excel_value(v) for v in row])
    t1 = time.perf_counter()
    wb.save(out)
    t2 = time.perf_counter()
    return (t1 - t0) * 1000, (t2 - t1) * 1000, (t2 - t0) * 1000


if __name__ == "__main__":
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    cols = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    import tempfile
    tmp = tempfile.mkdtemp()
    t = build(rows, cols)
    print(f"{rows:,} rows x {cols} cols = {rows * cols:,} cells\n")
    print(f"{'form':<12}{'write ms':>12}{'save ms':>12}{'total ms':>12}")
    for label, fn in (("per-cell", old_form), ("append", append_form),
                      ("write_only", write_only_form)):
        out = os.path.join(tmp, f"{label}.xlsx")
        try:
            w, s, tot = fn(t, out)
            print(f"{label:<12}{w:>12.0f}{s:>12.0f}{tot:>12.0f}")
        except Exception as exc:
            print(f"{label:<12}  FAILED: {type(exc).__name__}: {exc}")
