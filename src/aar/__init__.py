"""AAR - the Adaptive Analytics Runtime.

An orchestration layer for analytical work across Excel, SQL, NoSQL, Python,
local files and large-scale engines. It does not replace those tools; it
plans, routes, and explains.

The package is importable with **no third-party dependencies installed**.
Every engine (``duckdb``, ``polars``, ``pandas``, ``cudf``, ...) is optional,
probed at runtime, and its absence is reported through the capability
registry and logged - never silently swallowed.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
