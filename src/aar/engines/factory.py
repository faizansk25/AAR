"""Engine construction.

Engines are built by catalogue id and cached for the life of an execution,
so a segment assigned to DuckDB reuses one connection rather than paying
connection setup per node. If a requested engine is not installed, the
factory falls back to the Arrow engine *and says so* - the caller records it
as a degradation, which is what keeps "it ran on something else" visible.
"""

from __future__ import annotations

from typing import Any, Callable

from ..capability import Device
from ..failures import DegradationLedger, FailureKind
from .base import Engine, EngineCapabilities

__all__ = ["create_engine", "ENGINE_FACTORIES", "FALLBACK_ORDER"]


def _arrow_engine(**options: Any) -> Engine:
    from .arrow_engine import ArrowEngine
    return ArrowEngine(**options)


def _duckdb_engine(**options: Any) -> Engine:
    from .duckdb_engine import DuckDBEngine
    return DuckDBEngine(**options)


def _polars_engine(**options: Any) -> Engine:
    from .polars_engine import PolarsEngine
    return PolarsEngine(**options)


def _pandas_engine(**options: Any) -> Engine:
    from .pandas_engine import PandasEngine
    return PandasEngine(**options)


def _python_engine(**options: Any) -> Engine:
    from .python_engine import PythonWorkerEngine
    return PythonWorkerEngine(**options)


def _excel_engine(**options: Any) -> Engine:
    from .excel_engine import ExcelEngine
    return ExcelEngine(**options)


#: id -> constructor. Only engines with a real implementation appear here.
ENGINE_FACTORIES: dict[str, Callable[..., Engine]] = {
    "arrow": _arrow_engine,
    "duckdb": _duckdb_engine,
    "polars_cpu": _polars_engine,
    "pandas": _pandas_engine,
    "python_worker": _python_engine,
    "excel": _excel_engine,
}

#: Tried in order when a requested engine is unavailable. The Arrow engine is
#: last because it is correct everywhere and fast nowhere; DuckDB and Polars
#: are tried first because when they *are* present they are genuinely faster.
#: Excel is not a general fallback - it only handles workbooks - so it is
#: deliberately absent from this list.
FALLBACK_ORDER: tuple[str, ...] = ("duckdb", "polars_cpu", "pandas", "arrow")



def create_engine(
    engine_id: str,
    ledger: Any = None,
    node: Any = None,
    require_capability: str | None = None,
    **options: Any,
) -> Engine:
    """Build an engine, degrading to a working one if necessary.

    The degradation is the point. A run that quietly substituted a slower
    engine would still produce correct results but its plan would be a lie,
    so the substitution is recorded against the node that asked for it.
    """
    factory = ENGINE_FACTORIES.get(engine_id)
    if factory is not None:
        try:
            return factory(**options)
        except Exception as exc:  # noqa: BLE001 - includes "not installed"
            reason = f"{engine_id} unavailable ({exc})"
    else:
        reason = f"{engine_id} has no implementation in this build"

    for candidate in FALLBACK_ORDER:
        if candidate == engine_id and factory is not None:
            continue
        try:
            engine = ENGINE_FACTORIES[candidate](**options)
        except Exception:  # noqa: BLE001
            continue
        if require_capability and not engine.supports(require_capability):
            engine.close()
            continue
        if ledger is not None:
            # One degradation, not two: "asked for X, ran on Y" is a single
            # event, and recording it twice would overstate the damage.
            ledger.record(
                FailureKind.ENGINE_ABSENT, "engine_factory",
                f"{reason}; executed on {candidate} instead",
                node=node, from_engine=engine_id, to_engine=candidate)
        return engine



    raise RuntimeError(
        f"no execution engine is available: {engine_id} was requested and "
        f"every fallback in {FALLBACK_ORDER} failed to load. AAR cannot run "
        f"anything without at least pyarrow.")
