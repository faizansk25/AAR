"""Analyst-facing pipeline construction and loading.

A pipeline file is ordinary Python that returns an IR node. See
:mod:`aar.sdk.pipeline` for the builder vocabulary.
"""

from .pipeline import (  # noqa: F401
    Context, aggregate, and_, build_pipeline, classify, col, count, csv, eq,
    excel, filter_, ge, group_by, gt, join, le, limit, lit, load_pipeline, lt,
    max_, mean, min_, mongo, ne, not_, or_, parquet, project, sort, sql, sum_,
    udf, write_csv, write_excel, write_parquet,
)

__all__ = [
    "Context", "aggregate", "and_", "build_pipeline", "classify", "col",
    "count", "csv", "eq", "excel", "filter_", "ge", "group_by", "gt", "join",
    "le", "limit", "lit", "load_pipeline", "lt", "max_", "mean", "min_",
    "mongo", "ne", "not_", "or_", "parquet", "project", "sort", "sql", "sum_",
    "udf", "write_csv", "write_excel", "write_parquet",
]

