"""Which engine disagrees at 500k rows? Python is the ground truth."""
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import pyarrow.parquet as pq

from aar.engines.factory import create_engine
from aar.interchange import Table, arrow_to_canonical
from aar.ir import Agg, Col
from aar.types import Field, Schema

PATH = os.path.join(HERE, "..", "data", "nyc_taxi_2023_01.parquet")
N = 500_000
full = pq.read_table(PATH).slice(0, N)
schema = Schema(tuple(Field(f.name, arrow_to_canonical(f.type))
                      for f in full.schema))
table = Table(full, schema)
print(f"rows {table.num_rows:,}", flush=True)

# Ground truth, in plain Python, no engine involved.
keys = full.column("PULocationID").to_pylist()
values = full.column("fare_amount").to_pylist()
truth_sum = defaultdict(float)
truth_cnt = defaultdict(int)
for k, v in zip(keys, values):
    if v is not None:
        truth_sum[k] += v
        truth_cnt[k] += 1
print(f"truth: {len(truth_sum)} groups, group 132 total={truth_sum[132]!r} "
      f"n={truth_cnt[132]}", flush=True)

aggs = {"total": Agg("SUM", Col("fare_amount")),
        "n": Agg("COUNT", Col("fare_amount"))}
for engine_id in ("arrow", "duckdb", "polars_cpu", "pandas"):
    out = create_engine(engine_id).group_by(table, ["PULocationID"], aggs)
    rows = out.arrow.to_pylist()
    got = {r["PULocationID"]: (r["total"], r["n"]) for r in rows}
    mismatched = 0
    for key, (total, n) in got.items():
        want_total = truth_sum.get(key)
        want_n = truth_cnt.get(key)
        if n != want_n or (want_total is not None and total is not None
                           and abs(total - want_total) > 1e-6):
            mismatched += 1
    g = got.get(132)
    print(f"{engine_id:11s} {len(got):4d} groups, {mismatched:3d} disagree "
          f"with python truth, group132={g}", flush=True)


