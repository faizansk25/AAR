"""Reproduce the two Excel write failures exactly.

The `append` probe says a fresh sheet behaves correctly, so the assumption
that was wrong is not the one I thought. This runs the real failing cases
and prints the file that comes out, rather than reasoning about it.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import openpyxl
import pyarrow as pa

from aar.connectors.excel import read_excel, write_excel
from aar.interchange import Table

tmp = tempfile.mkdtemp()

# --- Case 1: the round-trip fixture -----------------------------------
t = Table(pa.table({"a": pa.array([1, 2, 3], type=pa.int64()),
                    "b": pa.array(["x", "y", "z"], type=pa.string())}))
print("table.column_names =", t.column_names)
print("to_pydict keys     =", list(t.arrow.to_pydict().keys()))

target1 = os.path.join(tmp, "out1.xlsx")
write_excel(t, target1, sheet="Result")
wb = openpyxl.load_workbook(target1)
ws = wb["Result"]
print("case1 max_row/max_col:", ws.max_row, ws.max_column)
for r in ws.iter_rows(values_only=True):
    print("   case1 row:", r)
wb.close()

# --- Case 2: the orders-style fixture with a nested value ---------------
t2 = Table(pa.table({"a": pa.array([1, 2], type=pa.int64()),
                     "b": pa.array(["x", "y"], type=pa.string())}))
target2 = os.path.join(tmp, "out2.xlsx")
write_excel(t2, target2, sheet="Data")
back = read_excel(__import__("aar.ir", fromlist=["ScanSpec"]).ScanSpec(
    kind="excel", path=target2, sheet="Data"))
print("case2 round-trip schema:", back.arrow.schema)
