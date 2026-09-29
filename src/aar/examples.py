"""Runnable example pipelines, embedded rather than shipped as loose files.

`aar run` and `aar explain` both take a path to a Python pipeline file, and
until now nothing told a new user what such a file looks like or handed them
one. The repository had `pipelines/example_orders.py`, but that path is not
inside the package, so it does not exist in an installed wheel - and AAR's
first promise is that it works on an air-gapped machine where there is no
repository to look in.

Embedding the source as strings solves that with no packaging change at all:
no ``package-data`` entry, no MANIFEST, nothing that can be forgotten at
build time. The cost is that these cannot be imported and type-checked as
modules, so `tests/test_cli.py` compiles every template and runs the
shortest one end to end - an example that does not compile is a bug, not
documentation.
"""

from __future__ import annotations

__all__ = ["EXAMPLES", "names", "source", "describe"]

_TAXI = '''\
"""Revenue by payment type, from a CSV.

    aar examples --write revenue revenue.py
    aar explain revenue.py
    aar run revenue.py

Everything below is ordinary Python. A pipeline file is a module with a
`build()` that returns the root IR node; `load_pipeline` imports it and
reads that back. The builders are functions, not methods - each takes the
node it operates on as its first argument, so a graph can be assembled in
any order and any step can be named.
"""
from aar.sdk import classify, col, csv, filter_, group_by, sort, sum_, write_csv


def build():
    trips = csv("trips.csv", estimated_bytes=2_000_000_000)
    paid = filter_(trips, col("fare_amount") > 0)
    by_payment = group_by(
        paid,
        "payment_type",
        aggs={"revenue": sum_("fare_amount")},
    )
    ranked = sort(by_payment, "revenue desc")
    tagged = classify(ranked, {"fare_amount": "INTERNAL"})
    return write_csv(tagged, "revenue_by_payment.csv")
'''

_ORDERS = '''\
"""Read an Excel workbook, aggregate, write a summary workbook.

This is the shape AAR is built around: Excel in, Excel out.

    aar examples --write orders orders.py
    aar run orders.py --explain
"""
from aar.sdk import col, excel, filter_, group_by, sum_, write_excel


def build():
    orders = excel("orders.xlsx", sheet="Orders", estimated_bytes=200_000_000)
    shipped = filter_(orders, col("status") == "shipped")
    by_region = group_by(
        shipped,
        "region",
        aggs={"revenue": sum_("amount")},
    )
    return write_excel(by_region, "regional_summary.xlsx", sheet="Summary")
'''

_HELLO = '''\
"""The smallest pipeline that still shows the shape of the thing.

    aar examples --write hello hello.py
    aar run hello.py
"""
from aar.sdk import count, excel, group_by, write_csv


def build():
    orders = excel("orders.xlsx", sheet="Orders")
    per_region = group_by(orders, "region", aggs={"orders": count()})
    return write_csv(per_region, "orders_per_region.csv")
'''

#: name -> (one-line summary, source)
EXAMPLES: dict[str, tuple[str, str]] = {
    "hello": ("smallest runnable pipeline", _HELLO),
    "orders": ("Excel in, aggregate, Excel out", _ORDERS),
    "revenue": ("CSV in, aggregate, CSV out", _TAXI),
}


def names() -> list[str]:
    """Example names, shortest first: `hello` is the one to start with."""
    return sorted(EXAMPLES, key=lambda n: (len(n), n))


def describe(name: str) -> str:
    return EXAMPLES[name][0]


def source(name: str) -> str:
    return EXAMPLES[name][1]
