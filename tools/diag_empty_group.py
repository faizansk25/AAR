"""Is the empty-table group-by metadata correct on every engine?

Listed as an open risk for some time and never actually checked. Diagnostic.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import pyarrow as pa

from aar.engines.factory import create_engine
from aar.interchange import Table
from aar.ir import Agg, Col
from aar.types import Field, INT64, Schema, UTF8

CONF = frozenset({"confidential"})
schema = Schema((Field("region", UTF8),
                 Field("salary", INT64, classification=CONF)))
empty = Table(pa.table({"region": pa.array([], pa.string()),
                        "salary": pa.array([], pa.int64())}), schema)
print("input: 0 rows, 2 cols, salary=confidential", flush=True)

for engine_id in ("arrow", "duckdb", "polars_cpu", "pandas"):
    try:
        engine = create_engine(engine_id)
    except Exception as exc:  # noqa: BLE001
        print(f"{engine_id:11s} unavailable: {exc}", flush=True)
        continue
    try:
        out = engine.group_by(empty, ["region"],
                              {"total": Agg("SUM", Col("salary"))})
        names = list(out.column_names)
        tag = (out.schema.get("total").classification
               if out.schema.has("total") else "MISSING")
        keytag = (out.schema.get("region").classification
                  if out.schema.has("region") else "MISSING")
        print(f"{engine_id:11s} rows={out.num_rows} cols={names} "
              f"total_tag={sorted(tag)} key_tag={sorted(keytag)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"{engine_id:11s} ERROR {type(exc).__name__}: {exc}",
              flush=True)
