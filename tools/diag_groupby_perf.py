"""Is Arrow's 213x group-by gap the kernel, or AAR's wrapper?

Measured on real NYC taxi data: grouping `lpep_pickup_datetime` (1,000,000
rows into ~40,000 groups) takes polars 14 ms and arrow 2,975 ms. The Arrow
path is not the Python fallback - `_group_by_kernel_works()` passes, so
this is Arrow's own C++ hash aggregation.

The question is therefore narrow: is the cost inside
`pyarrow.TableGroupBy.aggregate`, and does threading help or hurt at this
cardinality? This varies exactly that and nothing else, on the real file,
so the answer is a measurement rather than a guess.

Run: ``python tools/diag_groupby_perf.py [parquet] [rows]``
"""
import sys
import time

import pyarrow.parquet as pq

PATH = sys.argv[1] if len(sys.argv) > 1 else r"data\nyc_taxi_2022_03.parquet"
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
MONEY = "fare_amount"
KEYS = ["lpep_pickup_datetime", "PULocationID", "RatecodeID"]

table = pq.read_table(PATH).slice(0, ROWS)
print(f"{table.num_rows:,} rows, {table.num_columns} columns")
for key in KEYS:
    if key in table.column_names:
        distinct = len(set(table.column(key).to_pylist()))
        print(f"  {key:24s} {distinct:>9,} distinct groups")

try:
    import polars as pl
except ImportError:
    pl = None

for key in KEYS:
    if key not in table.column_names:
        continue
    print(f"\n--- group by {key} ---")
    sub = table.select([key, MONEY])
    for threaded in (True, False):
        started = time.perf_counter()
        out = sub.group_by([key]).aggregate(
            [("total", ("sum", MONEY))], use_threads=threaded)
        elapsed = (time.perf_counter() - started) * 1000
        print(f"  arrow use_threads={str(threaded):5s} {elapsed:9.1f} ms  "
              f"{out.num_rows:,} groups")
    if pl is not None:
        started = time.perf_counter()
        frame = (pl.from_arrow(sub).group_by(key)
                 .agg(pl.col(MONEY).sum()))
        elapsed = (time.perf_counter() - started) * 1000
        print(f"  polars{'':19s} {elapsed:9.1f} ms  {frame.height:,} groups")
