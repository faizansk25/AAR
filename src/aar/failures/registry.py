"""Failure modes, detection, handling, and the never-silent-failure contract.

AAR's twentieth design principle is *graceful degradation over catastrophic
failure*, but degradation without a record is a lie. So this module makes the
record structural rather than aspirational:

* Every mode in the specification's matrix has a :class:`FailureMode` entry
  with a detector, a handler, a fallback and a log template.
* Every runtime error raised anywhere in AAR carries a ``failure_mode`` tag,
  so a bare exception is already a bug.
* :class:`DegradationLedger` records every fallback that was taken. The
  runtime asserts at exit that the ledger was flushed - a swallowed
  degradation is a failed run, not a successful one.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "FailureKind", "Severity", "FailureMode", "FailureRegistry",
    "AARError", "CapabilityError", "SchemaDriftError", "TypeMismatchError",
    "PrivacyViolation", "QualityCheckFailed", "ResourceExhausted",
    "SourceUnavailable", "PolicyDenied", "UDFExecutionError",
    "Degradation", "DegradationLedger", "register_modes", "MODES",
]


class Severity(str, enum.Enum):
    """How much a degraded state should worry the operator."""

    INFO = "info"            # noticed, fully handled
    DEGRADED = "degraded"    # worked, but slower or lower fidelity
    BLOCKING = "blocking"    # stopped; a human decision is required


class FailureKind(str, enum.Enum):
    """The specification's failure taxonomy, stable identifiers.

    The ``code`` is what appears in logs, metrics and the UI, so it is a
    stable public contract and must not be renumbered.
    """

    GPU_UNAVAILABLE = "GPU_UNAVAILABLE"
    GPU_OOM_VRAM = "GPU_OOM_VRAM"
    GPU_OOM_UVM = "GPU_OOM_UVM"
    GPU_OP_UNSUPPORTED = "GPU_OP_UNSUPPORTED"
    SCHEMA_DRIFT = "SCHEMA_DRIFT"
    TYPE_MISMATCH = "TYPE_MISMATCH"
    NETWORK_PARTITION = "NETWORK_PARTITION"
    LICENSE_EXHAUSTED = "LICENSE_EXHAUSTED"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    CARDINALITY_ERROR = "CARDINALITY_ERROR"
    EXCEL_FORMAT_CHANGE = "EXCEL_FORMAT_CHANGE"
    EXCEL_FORMULA_ERROR = "EXCEL_FORMULA_ERROR"
    UDF_FAILURE = "UDF_FAILURE"
    ARROW_IPC_FAILURE = "ARROW_IPC_FAILURE"
    WORKER_FAILURE = "WORKER_FAILURE"
    MEMORY_SPILL = "MEMORY_SPILL"
    CACHE_INVALIDATED = "CACHE_INVALIDATED"
    CONCURRENT_ACCESS = "CONCURRENT_ACCESS"
    QUALITY_FAILURE = "QUALITY_FAILURE"
    PRIVACY_VIOLATION = "PRIVACY_VIOLATION"
    ENGINE_ABSENT = "ENGINE_ABSENT"
    NO_FEASIBLE_PLAN = "NO_FEASIBLE_PLAN"


class AARError(Exception):
    """Base class for every error AAR raises.

    Carrying ``failure_mode`` on the exception means any handler - including
    one written three layers away by an analyst - can tell what went wrong
    and what the sanctioned fallback is, without parsing a message string.
    """

    failure_mode: FailureKind = FailureKind.SOURCE_UNAVAILABLE
    severity: Severity = Severity.BLOCKING

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_mode": self.failure_mode.value,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
        }

    def __str__(self) -> str:
        if self.context:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(self.context.items()))
            return f"[{self.failure_mode.value}] {self.message} ({detail})"
        return f"[{self.failure_mode.value}] {self.message}"


class CapabilityError(AARError):
    failure_mode = FailureKind.GPU_OP_UNSUPPORTED
    severity = Severity.DEGRADED


class SchemaDriftError(AARError):
    failure_mode = FailureKind.SCHEMA_DRIFT
    severity = Severity.BLOCKING


class TypeMismatchError(AARError):
    failure_mode = FailureKind.TYPE_MISMATCH
    severity = Severity.BLOCKING


class PrivacyViolation(AARError):
    failure_mode = FailureKind.PRIVACY_VIOLATION
    severity = Severity.BLOCKING


class QualityCheckFailed(AARError):
    failure_mode = FailureKind.QUALITY_FAILURE
    severity = Severity.BLOCKING


class ResourceExhausted(AARError):
    failure_mode = FailureKind.GPU_OOM_VRAM
    severity = Severity.DEGRADED


class SourceUnavailable(AARError):
    failure_mode = FailureKind.SOURCE_UNAVAILABLE


class PolicyDenied(AARError):
    failure_mode = FailureKind.PRIVACY_VIOLATION
    severity = Severity.BLOCKING


class UDFExecutionError(AARError):
    failure_mode = FailureKind.UDF_FAILURE
    severity = Severity.DEGRADED


class PlanInfeasible(AARError):
    failure_mode = FailureKind.NO_FEASIBLE_PLAN
    severity = Severity.BLOCKING


class ConcurrencyError(AARError):
    failure_mode = FailureKind.CONCURRENT_ACCESS
    severity = Severity.DEGRADED


# ------------------------------------------------------------------- matrix
@dataclass(frozen=True, slots=True)
class FailureMode:
    """One row of the specification's failure matrix."""

    kind: FailureKind
    severity: Severity
    detect: str
    handle: str
    fallback: str
    log: str
    recoverable: bool = True

    def render(self) -> str:
        return (
            f"{self.kind.value}\n"
            f"    detect:   {self.detect}\n"
            f"    handle:   {self.handle}\n"
            f"    fallback: {self.fallback}\n"
            f"    log:      {self.log}\n"
            f"    severity: {self.severity.value}"
        )


_MODES: dict[FailureKind, FailureMode] = {}


def _mode(kind: FailureKind, severity: Severity, detect: str, handle: str,
          fallback: str, log: str, recoverable: bool = True) -> FailureMode:
    m = FailureMode(kind, severity, detect, handle, fallback, log, recoverable)
    _MODES[kind] = m
    return m


def register_modes() -> dict[FailureKind, FailureMode]:
    """Populate the failure matrix. Idempotent."""
    if _MODES:
        return _MODES
    _mode(FailureKind.GPU_UNAVAILABLE, Severity.DEGRADED,
          "NVML / CUDA runtime probe returns no device",
          "skip all GPU paths for the run",
          "CPU engine (duckdb, polars_cpu, pandas)",
          "GPU not detected; selected {fallback}")
    _mode(FailureKind.GPU_OOM_VRAM, Severity.DEGRADED,
          "cuDF allocator raises OutOfMemoryError",
          "enable UVM, then streaming, then chunking",
          "reduce batch, then CPU engine",
          "VRAM exceeded ({requested}); chunk size {chunk}")
    _mode(FailureKind.GPU_OOM_UVM, Severity.DEGRADED,
          "persistent OOM under UVM after spill threshold",
          "chunked processing with explicit batch sizing",
          "distributed engine, then CPU",
          "UVM OOM; chunk size {chunk}")
    _mode(FailureKind.GPU_OP_UNSUPPORTED, Severity.DEGRADED,
          "capability registry says op is absent for the engine",
          "split the node and re-plan the affected segment",
          "CPU engine for that operation only",
          "op {op} unsupported on {engine}")
    _mode(FailureKind.SCHEMA_DRIFT, Severity.BLOCKING,
          "SchemaDiff between expected and observed columns",
          "pause, report added/removed/retyped columns",
          "offer repair; never auto-coerce",
          "schema change in source {source}: {diff}")
    _mode(FailureKind.TYPE_MISMATCH, Severity.BLOCKING,
          "lossy() returns a reason at an engine boundary",
          "normalize per policy; widen where allowed",
          "flag for review under strict policy",
          "type mismatch: {column} {source} -> {target}")
    _mode(FailureKind.NETWORK_PARTITION, Severity.DEGRADED,
          "connection timeout or reset on a remote source",
          "queue, retry with capped exponential backoff",
          "serve from cache if within TTL, else fail loudly",
          "source {source} unreachable ({attempts} attempts)")
    _mode(FailureKind.LICENSE_EXHAUSTED, Severity.DEGRADED,
          "engine or accelerator pool reports no free seats",
          "enqueue the segment; hold on the barrier",
          "CPU engine while waiting",
          "{resource} license limit reached; queued")
    _mode(FailureKind.SOURCE_UNAVAILABLE, Severity.BLOCKING,
          "connection check or first read fails",
          "retry with backoff up to the threshold",
          "alert after threshold; do not emit partial data",
          "source {source} unavailable")
    _mode(FailureKind.CARDINALITY_ERROR, Severity.DEGRADED,
          "actual vs estimated row ratio beyond the error bound",
          "re-plan the remaining segments with observed statistics",
          "spill to disk if the new estimate exceeds memory",
          "cardinality off by {ratio}x at {node}")
    _mode(FailureKind.EXCEL_FORMAT_CHANGE, Severity.DEGRADED,
          "header row text, delimiter, or encoding differs from profile",
          "re-detect and report the changed region",
          "explicit repair; never silently re-map columns",
          "Excel format changed at {sheet}: {detail}")
    _mode(FailureKind.EXCEL_FORMULA_ERROR, Severity.DEGRADED,
          "cached formula result missing or error-valued",
          "use the cached value, then the evaluated value",
          "flag the cell range for review",
          "formula error in {range}: {detail}")
    _mode(FailureKind.UDF_FAILURE, Severity.DEGRADED,
          "exception escaping the user function",
          "retry once in an isolated worker, then skip the batch",
          "emit null for failed rows and continue, loudly",
          "UDF {name} failed: {error}")
    _mode(FailureKind.ARROW_IPC_FAILURE, Severity.DEGRADED,
          "serialization error on the interchange boundary",
          "retry the batch; verify schema identity",
          "fall back to a lossless in-process handoff",
          "Arrow IPC failed between {a} and {b}")
    _mode(FailureKind.WORKER_FAILURE, Severity.DEGRADED,
          "Ray/Dask task raises or the actor dies",
          "retry the task on another worker",
          "local single-node execution",
          "worker {id} failed; reassigned to {replacement}")
    _mode(FailureKind.MEMORY_SPILL, Severity.DEGRADED,
          "resident set crosses the spill threshold",
          "spill intermediate partitions to local disk",
          "reduce parallelism; alert if excessive",
          "spilled {bytes} to disk for {node}")
    _mode(FailureKind.CACHE_INVALIDATED, Severity.INFO,
          "source mtime/size or query text changed",
          "invalidate dependent cache entries",
          "re-execute the affected segment",
          "cache invalidated for {key}")
    _mode(FailureKind.CONCURRENT_ACCESS, Severity.DEGRADED,
          "lock or exclusive-open conflict on a target",
          "queue with backoff up to the timeout",
          "write to a versioned path and notify",
          "resource locked by {holder}")
    _mode(FailureKind.QUALITY_FAILURE, Severity.BLOCKING,
          "a declared quality rule evaluates false",
          "pause before the write so nothing lands",
          "offer repair; report offending rows",
          "quality check failed: {rule} on {column}")
    _mode(FailureKind.PRIVACY_VIOLATION, Severity.BLOCKING,
          "policy engine denies a classification/engine/destination pair",
          "refuse the edge and record the denial",
          "route to an approved destination if one exists",
          "privacy violation: {detail}")
    _mode(FailureKind.ENGINE_ABSENT, Severity.INFO,
          "import probe for an optional engine fails",
          "remove the engine from the feasible set",
          "next-best engine, with the absence logged",
          "engine {engine} not installed; removed from plan")
    _mode(FailureKind.NO_FEASIBLE_PLAN, Severity.BLOCKING,
          "capability x privacy x hardware leaves no candidate",
          "report the binding constraint per node",
          "none - requires a human decision",
          "no legal plan: {constraint}")
    return _MODES


class FailureRegistry:
    """Lookup over the failure matrix."""

    @staticmethod
    def get(kind: FailureKind) -> FailureMode:
        return register_modes()[kind]

    @staticmethod
    def all() -> tuple[FailureMode, ...]:
        return tuple(register_modes().values())

    @staticmethod
    def render_matrix() -> str:
        header = (
            f"{'CODE':<24}{'SEVERITY':<11}{'RECOVERABLE':<13}FALLBACK\n"
            f"{'-' * 24}{'-' * 11}{'-' * 13}{'-' * 40}\n"
        )
        rows = "".join(
            f"{m.kind.value:<24}{m.severity.value:<11}"
            f"{('yes' if m.recoverable else 'no'):<13}{m.fallback}\n"
            for m in FailureRegistry.all()
        )
        return header + rows


# --------------------------------------------------------------- degradation
@dataclass(slots=True)
class Degradation:
    """One fallback that was actually taken during a run."""

    kind: FailureKind
    node_id: str | None
    node_type: str | None
    where: str
    from_engine: str | None
    to_engine: str | None
    detail: str
    severity: Severity
    at: float = field(default_factory=time.time)
    exception: str | None = None

    def render(self) -> str:
        bits = [f"{self.kind.value}"]
        if self.node_id:
            bits.append(f"at {self.node_type or ''} {self.node_id}")
        bits.append(f"in {self.where}")
        if self.from_engine or self.to_engine:
            bits.append(f"{self.from_engine or '-'} -> {self.to_engine or '-'}")
        line = ": ".join(bits[:1]) + " | " + " | ".join(bits[1:])
        if self.exception:
            line += f"\n    cause: {self.exception}"
        return line

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_mode": self.kind.value,
            "severity": self.severity.value,
            "node_id": self.node_id,
            "node_type": self.node_type,
            "where": self.where,
            "from_engine": self.from_engine,
            "to_engine": self.to_engine,
            "detail": self.detail,
            "exception": self.exception,
            "at": self.at,
        }


class DegradationLedger:
    """Append-only record of every fallback taken during an execution.

    The runtime holds one ledger per run and refuses to report success while
    any BLOCKING-severity entry is unresolved. This is the mechanism that
    makes "the system never silently fails" a property of the code rather
    than a hope about developer discipline.
    """

    __slots__ = ("_entries", "_sinks")

    def __init__(self, *sinks: Callable[[Degradation], None]) -> None:
        self._entries: list[Degradation] = []
        self._sinks = sinks

    def record(
        self, kind: FailureKind, where: str, detail: str,
        node: Any = None, from_engine: str | None = None,
        to_engine: str | None = None, exception: BaseException | None = None,
    ) -> Degradation:
        mode = register_modes()[kind]
        d = Degradation(
            kind=kind,
            node_id=getattr(node, "id", None),
            node_type=str(getattr(node, "type", "")) or None,
            where=where, from_engine=from_engine, to_engine=to_engine,
            detail=detail, severity=mode.severity,
            exception=_fmt_exception(exception) if exception else None,
        )
        self._entries.append(d)
        for sink in self._sinks:
            try:
                sink(d)
            except Exception:  # noqa: BLE001 - a broken sink must not mask the event
                pass
        return d

    @property
    def entries(self) -> tuple[Degradation, ...]:
        return tuple(self._entries)

    @property
    def blocking(self) -> tuple[Degradation, ...]:
        return tuple(e for e in self._entries if e.severity is Severity.BLOCKING)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self._entries:
            out[e.kind.value] = out.get(e.kind.value, 0) + 1
        return out

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def render(self) -> str:
        if not self._entries:
            return "No degradations. Full-fidelity execution."
        lines = [f"{len(self._entries)} degradation(s) recorded:"]
        lines += [f"  - {e.render()}" for e in self._entries]
        return "\n".join(lines)

    def assert_clean(self) -> None:
        """Raise if any BLOCKING degradation went unresolved.

        Called by the runtime at the end of a run. A run that degraded in a
        way requiring a human must not return a success result object.
        """
        if self.blocking:
            summary = "; ".join(
                f"{e.kind.value} at {e.where}" for e in self.blocking)
            raise AARError(
                f"execution completed with unresolved blocking degradation(s): {summary}",
                degradations=len(self.blocking),
            )


def _fmt_exception(exc: BaseException) -> str:
    """Render an exception with its message but without a wall of traceback."""
    return f"{type(exc).__name__}: {exc}"


#: Where degradations go when nobody supplied a ledger. Module-level on
#: purpose: the whole point is that a caller who forgot to pass a ledger
#: still leaves a trace behind. ``create_engine`` additionally warns, so this
#: is a backstop, not the primary signal.
_PROCESS_LEDGER = DegradationLedger()


def process_ledger() -> DegradationLedger:
    """The process-wide ledger for degradations nobody asked to be recorded.

    Exists because the alternative is worse. Before this, a caller who
    omitted ``ledger=`` lost the degradation entirely, and a verification
    script wrote a CPU engine's timings to disk under a GPU engine's name
    without anything anywhere recording that it had happened.
    """
    return _PROCESS_LEDGER


def capture(fn, *args, ledger: DegradationLedger | None = None,
            where: str = "", kind: FailureKind = FailureKind.UDF_FAILURE,
            on_error: Callable[[BaseException], Any] | None = None,
            **kwargs) -> Any:
    """Run ``fn``; on failure record a degradation and apply the fallback.

    This is the standard shape for AAR's error handling. It exists so that
    "try it, degrade, and *record it*" is one line at each call site instead
    of a bespoke try/except that might forget the recording.
    """
    try:
        return fn(*args, **kwargs)
    except AARError:
        raise
    except Exception as exc:  # noqa: BLE001
        if ledger is not None:
            ledger.record(kind, where or fn.__name__, "operation failed",
                           exception=exc)
        if on_error is None:
            raise
        return on_error(exc)

