"""Engine construction.

Engines are built by catalogue id and cached for the life of an execution,
so a segment assigned to DuckDB reuses one connection rather than paying
connection setup per node. If a requested engine is not installed, the
factory falls back to the Arrow engine *and says so* - the caller records it
as a degradation, which is what keeps "it ran on something else" visible.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

from ..failures import (
    CapabilityError, FailureKind, process_ledger,
)
from .base import Engine

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


def _cudf_engine(**options: Any) -> Engine:
    from .cudf_engine import CudfEngine
    return CudfEngine(**options)


def _polars_gpu_engine(**options: Any) -> Engine:
    from .polars_gpu_engine import PolarsGPUEngine
    return PolarsGPUEngine(**options)


#: id -> constructor. Only engines with a real implementation appear here.
#: An id here is a promise that the class exists, not merely that the
#: package can be installed: `create_engine` falls back for anything absent,
#: so listing an engine that has no class behind it is how a GPU result
#: gets filed under a CPU engine's name. `TestDeclaredIsNotImplemented`
#: keeps this dictionary honest against the capability registry.
ENGINE_FACTORIES: dict[str, Callable[..., Engine]] = {
    "arrow": _arrow_engine,
    "duckdb": _duckdb_engine,
    "polars_cpu": _polars_engine,
    "pandas": _pandas_engine,
    "python_worker": _python_engine,
    "excel": _excel_engine,
    "cudf": _cudf_engine,
    "polars_gpu": _polars_gpu_engine,
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
    allow_degradation: bool = True,
    **options: Any,
) -> Engine:
    """Build an engine, degrading to a working one if necessary.

    The degradation is the point. A run that quietly substituted a slower
    engine would still produce correct results but its plan would be a lie,
    so the substitution is recorded against the node that asked for it.

    **A substitution is never silent.** Three guarantees, because this
    function used to be quietly dishonest:

    1. With no ``ledger``, the event goes to the process-wide
       :func:`aar.failures.process_ledger` *and* raises a
       ``RuntimeWarning``. Omitting the ledger is not permission to lose
       the record - it used to be exactly that, and that is how a benchmark
       came to report Arrow's timings under the label "cudf".
    2. With a ``ledger``, the event is recorded and no warning is raised:
       the caller supplied somewhere to put it, so they are listening.
    3. ``allow_degradation=False`` raises instead of substituting. Use it
       whenever the *specific* engine is the point of the call - a GPU
       benchmark, or a plan that named an engine deliberately. Silently
       getting a different engine is the wrong answer there, however
       graceful it is everywhere else.
    """
    factory = ENGINE_FACTORIES.get(engine_id)
    if factory is not None:
        try:
            engine = factory(**options)
        except Exception as exc:  # noqa: BLE001 - includes "not installed"
            reason = f"{engine_id} unavailable ({exc})"
        else:
            # An engine can construct and still be unable to do the job: the
            # cudf engine builds happily on a machine with no GPU and then
            # declines to execute anything. Returning it would hand the
            # caller a landmine - every operation would fail one layer up,
            # with no record of the substitution here. Declining at
            # construction means "absent", and absent is what this function
            # knows how to handle.
            if engine.supports("execute"):
                return engine
            reason = (f"{engine_id} cannot execute on this machine "
                      f"({engine.capabilities.reason('execute')})")
    else:
        reason = f"{engine_id} has no implementation in this build"

    if not allow_degradation:
        # Refused before any fallback is attempted, so this cannot be
        # mistaken for "everything failed" - only the request was refused.
        raise CapabilityError(
            f"engine {engine_id!r} was required but is not available: "
            f"{reason}. Pass allow_degradation=True to fall back to "
            f"{', '.join(FALLBACK_ORDER)} instead.",
            requested=engine_id, reason=reason)

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
        _note_substitution(reason, engine_id, candidate, ledger, node)
        return engine

    raise RuntimeError(
        f"no execution engine is available: {engine_id} was requested and "
        f"every fallback in {FALLBACK_ORDER} failed to load. AAR cannot run "
        f"anything without at least pyarrow.")


def _note_substitution(reason: str, requested: str, actual: str,
                       ledger: Any, node: Any = None) -> None:
    """Record - and, when nobody is listening, announce - a substitution.

    One event, not two: "asked for X, ran on Y" is a single fact, and
    recording it twice would overstate the damage.
    """
    target = ledger if ledger is not None else process_ledger()
    target.record(
        FailureKind.ENGINE_ABSENT, "engine_factory",
        f"{reason}; executed on {actual} instead",
        node=node, from_engine=requested, to_engine=actual)
    if ledger is None:
        # No ledger means no observer. The process ledger keeps the record,
        # but a record nobody reads is indistinguishable from silence -
        # and silence here is how a CPU result gets filed as a GPU one.
        warnings.warn(
            f"requested engine {requested!r} is unavailable and was replaced "
            f"by {actual!r} ({reason}). No ledger was supplied, so this was "
            f"recorded in aar.failures.process_ledger() only. Pass "
            f"ledger=... to record it yourself, or allow_degradation=False "
            f"to refuse the substitution.",
            RuntimeWarning, stacklevel=3)
