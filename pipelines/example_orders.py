"""Example pipeline: the shape the specification uses as its worked example.

    ScanExcel -> Filter -> GroupBy -> PythonUDF -> Sort -> WriteExcel

Run it with:

    aar explain pipelines/example_orders.py

This file is executed by AAR, so it only builds a plan - it does not read the
workbook. Execution is the next layer; `aar explain` proves the planner can
cost and schedule the graph and justify every engine it picks.
"""

from aar.sdk import (excel, filter_, group_by, gt, col, limit, sort,
                     sum_, udf, write_excel)


def risk_band(amount: float) -> str:
    """An arbitrary Python rule - the thing a GPU cannot accelerate."""
    if amount >= 1000:
        return "high"
    if amount >= 100:
        return "medium"
    return "low"


def build():
    orders = excel(
        "FY26-orders.xlsx",
        sheet="Orders",
        header_row=1,
        estimated_bytes=2_000_000_000,
    )
    paid = filter_(orders, gt(col("amount"), 100))
    by_region = group_by(
        paid,
        "region",
        aggs={"total": sum_("amount"), "n": sum_("quantity")},
    )
    banded = udf(by_region, risk_band, name="risk_band")
    ranked = sort(banded, "total desc")
    return write_excel(limit(ranked, 100), "FY26-report.xlsx", sheet="Summary")
