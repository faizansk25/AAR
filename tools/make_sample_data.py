r"""Generate ``FY26-orders.xlsx`` so the example pipeline can really run.

``aar explain`` never needs data, which is why it worked for months with no
workbook present. ``aar run`` does, so this writes a small, deterministic
workbook matching the schema in ``pipelines/example_orders.py``.

Run with:  .venv\Scripts\python tools\make_sample_data.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aar.connectors.excel import write_excel  # noqa: E402
from aar.interchange import Table  # noqa: E402

REGIONS = ["NA", "EU", "APAC", "LATAM", "APAC", "EU", "NA"]

COLUMNS = ("id", "order_date", "region", "amount", "quantity", "customer")


def build_table(rows: int = 240) -> Table:
    import pyarrow as pa

    data = {
        "id": list(range(1, rows + 1)),
        "region": [REGIONS[i % len(REGIONS)] for i in range(rows)],
        # A wide spread so the risk banding actually produces all three
        # bands; a narrow range would make the example look correct while
        # never exercising the interesting case.
        "amount": [float(25 * ((i * 37) % 199)) for i in range(rows)],
        "quantity": [(i % 9) + 1 for i in range(rows)],
    }
    return Table(pa.table({
        "id": pa.array(data["id"], type=pa.int64()),
        "order_date": pa.array([f"2026-{(i % 12) + 1:02d}-"
                                f"{(i % 28) + 1:02d}" for i in range(rows)],
                               type=pa.string()),
        "region": pa.array(data["region"], type=pa.string()),
        "amount": pa.array(data["amount"], type=pa.float64()),
        "quantity": pa.array(data["quantity"], type=pa.int64()),
        "customer": pa.array([f"CUST-{1000 + (i % 40)}" for i in range(rows)],
                             type=pa.string()),
    }))


def main() -> int:
    import openpyxl  # noqa: F401 - fail early with a clear message

    target = os.path.join(os.path.dirname(__file__), "..",
                          "FY26-orders.xlsx")
    target = os.path.abspath(target)
    table = build_table()
    written = write_excel(table, target, sheet="Orders")
    print(f"wrote {target} ({written} rows, {len(table.column_names)} "
          f"columns: {', '.join(table.column_names)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
