"""Measure the Excel write path, old form against new, one process each.

Same isolation rule as `bench_engines.py`: each measurement runs in a fresh
interpreter, because the previous round of this programme produced a 213x
"defect" that was entirely allocator contamination from a prior join.

Two modes:
  append   - the current implementation, one `ws.append` per row
  percell  - the previous implementation, one `ws.cell` per cell

Run:  python tools/bench_excel.py            # 200k rows
      python tools/bench_excel.py --real     # 1M rows of NYC taxi Parquet
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

WORKER = r'''
import os, sys, time, tempfile
sys.path.insert(0, {src!r})
import pyarrow as pa
from aar.interchange import Table
from aar.connectors.excel import write_excel, _openpyxl

mode = sys.argv[1]
rows = int(sys.argv[2])
cols = int(sys.argv[3])

n = 1_000_000 if mode == "real" else rows
if mode == "real":
    import pyarrow.parquet as pq
    src = pq.read_table(os.path.join({root!r}, "data", "real", "taxi.parquet"))
    src = src.slice(0, rows)
else:
    src = pa.table({{"c%d" % i: pa.array(list(range(rows)), type=pa.int64())
                     for i in range(cols)}})

t = Table(src)
out = os.path.join(tempfile.mkdtemp(), "bench.xlsx")
t0 = time.perf_counter()
if mode == "percell":
    # The previous implementation, verbatim.
    openpyxl = _openpyxl()
    wb = openpyxl.Workbook()
    ws = wb.active
    for c, name in enumerate(t.column_names, start=1):
        ws.cell(row=1, column=c, value=name)
    row_cursor = 2
    for record in t.arrow.to_pylist():
        for c, name in enumerate(t.column_names, start=1):
            ws.cell(row=row_cursor, column=c, value=record.get(name))
        row_cursor += 1
    wb.save(out)
else:
    write_excel(t, out)
print((time.perf_counter() - t0) * 1000.0)
'''


def _measure(mode: str, rows: int, cols: int) -> float:
    code = WORKER.format(src=os.path.join(ROOT, "src"), root=ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", code, mode, str(rows), str(cols)],
        capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    return float(proc.stdout.strip().splitlines()[-1])


def bench(rows: int, cols: int, repeats: int = 3) -> dict:
    result: dict = {"rows": rows, "cols": cols, "cells": rows * cols}
    for mode in ("percell", "append"):
        samples = [_measure(mode, rows, cols) for _ in range(repeats)]
        result[mode] = {
            "median_ms": round(statistics.median(samples), 2),
            "spread_pct": round(
                (max(samples) - min(samples)) / statistics.median(samples) * 100, 2),
        }
    before = result["percell"]["median_ms"]
    after = result["append"]["median_ms"]
    result["speedup"] = round(before / after, 1)
    return result


if __name__ == "__main__":
    real = "--real" in sys.argv
    rows = 1_000_000
    for i, a in enumerate(sys.argv):
        if a == "--rows" and i + 1 < len(sys.argv):
            rows = int(sys.argv[i + 1])
    out = bench(rows, 19 if real else 12)
    out["dataset"] = (f"{rows:,} rows of real NYC taxi Parquet"
                      if real else f"synthetic, {rows:,} rows")
    out["note"] = (
        "Each mode runs in a fresh interpreter. The per-cell form is the "
        "implementation this replaced: to_pylist() plus ws.cell() per cell. "
        "It holds one openpyxl Cell object per cell in memory, so a "
        "million-row write is not a slow write, it is an out-of-memory one.")
    print(json.dumps(out, indent=2))
    suffix = "real" if real else "synth"
    path = os.path.join(ROOT, "data", "audit", f"bench_excel_{suffix}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("written:", path)
