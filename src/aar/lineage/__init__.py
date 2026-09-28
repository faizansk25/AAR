"""Lineage: how a classification travels from a source column to a result."""

from .taint import (  # noqa: F401
    AGGREGATE_RULE, COUNT_STAR_RULE, JOIN_RULE, UDF_RULE, LineageEvent,
    declassify, derive_from, describe, for_aggregate, inherit_all,
    is_derived_from, merge_schemas,
)

__all__ = [
    "AGGREGATE_RULE", "COUNT_STAR_RULE", "JOIN_RULE", "UDF_RULE",
    "LineageEvent", "declassify", "derive_from", "describe", "for_aggregate",
    "inherit_all", "is_derived_from", "merge_schemas",
]
