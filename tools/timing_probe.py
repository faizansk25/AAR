"""Time each engine operation on the real dataset, one at a time.

The audit as a whole appeared to hang, so this measures the pieces and
prints as it goes. Diagnostic only.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import pyarrow.parquet as pq

from aar.engines.factory import create_engine
from aar.interchange import Table, arrow_to_canonical
from aar.ir import Agg, BinOp, Col, Lit
from aar.types import Field, Schema

PATH = os.path.join(HERE, "..", "data", "nyc_taxi_2023_01.parquet")


def step(label, fn):
    started = time.time()
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001
        print(f"  {label:28s} ERROR {type(exc).__name__}: "
              f"{str(exc)[:60]}", flush=True)
        return None
    print(f"  {label:28s} {value!s:22.22s} in {time.time() - started:6.1f}s",
          flush=True)
    return value


print(f"rows in file: {pq.ParquetFile(PATH).metadata.num_rows:,}", flush=True)
arrow_table = step("pyarrow read_table", lambda: pq.read_table(PATH))
schema = Schema(tuple(Field(f.name, arrow_to_canonical(f.type))
                      for f in arrow_table.schema))
table = step("wrap in AAR Table", lambda: Table(arrow_table, schema))
print(f"  columns: {len(table.column_names)}", flush=True)

for engine_id in ("arrow", "duckdb", "polars_cpu", "pandas"):
    try:
        engine = create_engine(engine_id)
    except Exception as exc:  # noqa: BLE001
        print(f"{engine_id}: unavailable ({exc})", flush=True)
        continue
    print(f"\n{engine_id}", flush=True)
    step("filter (all rows)",
         lambda e=engine: e.filter(
             table, BinOp(Col("passenger_count"), ">", Lit(0))).num_rows)
    step("project (3 cols)",
         lambda e=engine: e.project(
             table, list(table.column_names)[:3]).num_columns)
    step("limit 10", lambda e=engine: e.limit(table, 10).num_rows)
    step("sort desc",
         lambda e=engine: e.sort(table, [("trip_distance", False)]).num_rows)
    step("group_by PULocationID",
         lambda e=engine: e.group_by(
             table, ["PULocationID"],
             {"total": Agg("SUM", Col("trip_distance"), "total")}).num_rows)
