"""Does the native Arrow group-by path actually run?

`ruff` reported `F821 Undefined name 'Schema'` at
`engines/arrow_engine.py:821`, which is *outside* the `try/except` that ends
at line 813. If `Schema` and `Field` are genuinely absent from the module
namespace, then `_arrow_group_by` raises NameError on every call where the
kernel succeeds - the fast path is dead on arrival and everything silently
falls back to the Python implementation.

This checks the namespace and then calls it, rather than trusting either the
linter or the reader.

Run: python tools/probe_arrow_groupby.py
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import pyarrow as pa

from aar.interchange import Table, arrow_to_canonical
from aar.ir import Agg, Col
from aar.engines import arrow_engine as A

print("1. module namespace")
for name in ("Schema", "Field"):
    print(f"   arrow_engine.{name:8} present={hasattr(A, name)}")

print("\n2. does the kernel say it works?")
print(f"   _group_by_kernel_works() = {A._group_by_kernel_works()}")

print("\n3. call _arrow_group_by directly")
from aar.types import Field, Schema
from aar.types import INT64

schema = Schema((Field("k", INT64), Field("v", INT64)))
tbl = Table(pa.table({"k": pa.array([1, 1, 2], type=pa.int64()),
                      "v": pa.array([10, 20, 30], type=pa.int64())}),
            schema)
try:
    out = A._arrow_group_by(tbl, ["k"], {"total": Agg("sum", Col("v"))})
    print(f"   returned {type(out).__name__}")
    if out is not None:
        print(f"   rows={out.num_rows} names={list(out.column_names)}")
except Exception:
    print("   RAISED:")
    for line in traceback.format_exc().strip().splitlines()[-4:]:
        print("     ", line)

print("\n3b. force the fast path, bypassing the capability guard")
print("     (the guard is what hid the NameError; bypass it to prove the")
print("      body itself is sound)")
try:
    out = A._arrow_group_by.__wrapped__ if False else None
except Exception:
    pass
orig = A._group_by_kernel_works
A._group_by_kernel_works = lambda: True
try:
    out = A._arrow_group_by(tbl, ["k"], {"total": Agg("sum", Col("v"))})
    print(f"   returned {type(out).__name__}")
    if out is not None:
        print(f"   rows={out.num_rows} names={list(out.column_names)}")
        print(f"   arrow rows={out.arrow.to_pydict()}")
    else:
        print("   declined (returned None) - kernel genuinely cannot do it")
except Exception:
    print("   RAISED:")
    for line in traceback.format_exc().strip().splitlines()[-4:]:
        print("     ", line)
finally:
    A._group_by_kernel_works = orig

print("\n4. what does the engine's public group_by do?")
try:
    from aar.ir import Node, NodeType

    out = A.ArrowEngine().group_by(tbl, ["k"], {"total": Agg("sum", Col("v"))})
    print(f"   returned {type(out).__name__}, rows={out.num_rows}")
except Exception:
    print("   RAISED:")
    for line in traceback.format_exc().strip().splitlines()[-4:]:
        print("     ", line)
