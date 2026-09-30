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

# 0.0.1 is deliberate and is not a "pre-alpha is coming" placeholder. The
# specification is not finished: GPU and distributed execution are unverified
# from the repository, the PostgreSQL/MySQL/MongoDB connectors have never
# spoken to a live server, there is no data profiler, and the scheduler does
# no real resource management. Nine of nineteen layers are built. A higher
# number would invite someone to depend on that, so the version says what the
# work actually is until the rest of the system exists.
__version__ = "0.0.1"
__all__ = ["__version__"]
