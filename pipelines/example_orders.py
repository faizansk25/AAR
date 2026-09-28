"""Example pipeline: the shape the specification uses as its worked example.

    ScanExcel -> Filter -> GroupBy -> PythonUDF -> Sort -> WriteExcel

Run it with:

    python tools/make_sample_data.py          # writes FY26-orders.xlsx
    aar explain pipelines/example_orders.py   # plan only, no data needed
    aar run pipelines/example_orders.py       # reads, computes, writes

`aar explain` never touches the workbook, which is why it works with no data
present. `aar run` does need it, so generate it first.
"""

from aar.sdk import (col, excel, filter_, group_by, limit, sort, sum_, udf,
                     write_excel)


def risk_band(row: dict) -> str:
    """An arbitrary Python rule - the thing a GPU cannot accelerate.

    Row mode: the default. A UDF declared this way receives the whole record,
    so the name of its parameter never has to encode a convention. See
    ``aar.sdk.udf`` for the column-mode alternative.
    """
    total = row.get("total") or 0
    if total >= 1000:
        return "high"
    if total >= 100:
        return "medium"
    return "low"


def build():
    orders = excel(
        "FY26-orders.xlsx",
        sheet="Orders",
        header_row=1,
        estimated_bytes=2_000_000_000,
    )
    paid = filter_(orders, col("amount") > 100)
    by_region = group_by(
        paid,
        "region",
        aggs={"total": sum_("amount"), "n": sum_("quantity")},
    )
    banded = udf(by_region, risk_band, name="risk_band")
    ranked = sort(banded, "total desc")
    return write_excel(limit(ranked, 100), "FY26-report.xlsx", sheet="Summary")

