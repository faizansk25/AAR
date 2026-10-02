"""Execution history: what AAR measured, and how sure it is allowed to be.

This module exists because the cost model once asked "how long did this take
last time?" and acted on the answer, and that was premature. The history is
now **observe-only**: it records, it reports, and it refuses to steer a plan
until its units and provenance are trustworthy. :meth:`ExecutionHistory.predict`
is no longer consulted by :meth:`~aar.cost.CostModel.compute_s`.

Three defects motivated the split, each measured rather than argued:

1. **Wall time was being read as kernel time.** The executor starts its clock
   before engine acquisition, so ``elapsed_ms`` contains a fixed startup cost.
   Feeding that into ``compute_s`` and then adding startup, read and transfer
   on top double-charges. A 95 ms cold scan was reported as 95 ms of compute.
2. **One observation answered for every size.** ``predict`` on a single record
   returned that record's time regardless of the requested size, so a
   ``1 MB -> 4 ms`` measurement implied ``40 GB -> 4 ms``. Reproduced before
   this change; ``test_a_single_observation_never_extrapolates_to_another_size``
   pins the fix.
3. **Evidence pooled under no target.** A record stored without a target id
   answered for every machine, including machines it was never run on.

The record therefore carries provenance, and :meth:`ExecutionHistory.predict`
answers only when the evidence is specific enough to be worth acting on:

* **size** - refuse to extrapolate past the observed range without enough
  distinct sizes to have measured a slope;
* **target** - require a matching ``target_id`` rather than accepting any;
* **identity** - key on the real ``semantic_operation_id``, so two parses of one
  pipeline meet, and two different queries never do.

Storage stays in memory. Nothing here is persisted yet: an in-memory record
with wrong units is a bug that disappears when the process ends, and the same
record written to SQLite is a bug that outlives it. Correct units and
provenance first, durability second.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = [
    "Observation", "ExecutionRecord", "ExecutionHistory", "Confidence",
    "HISTORY_SCHEMA_VERSION", "DEFAULT_MIN_SIZES_FOR_EXTRAPOLATION",
    "DEFAULT_EXTRAPOLATION_LIMIT",
]

#: Bumped when the meaning of a stored field changes. A record written under
#: an older version describes different quantities, so mixing them would be
#: worse than having no history at all.
HISTORY_SCHEMA_VERSION = "aar-history-v2"

#: How many *distinct input sizes* are needed before a slope is believed. One
#: point is not a trend; two are the minimum for a line, and the default
#: insists on more before trusting that line away from its anchors.
DEFAULT_MIN_SIZES_FOR_EXTRAPOLATION = 3

#: How far outside the observed size range a prediction may reach, as a
#: multiple. Inside this band the model interpolates; beyond it, it refuses
#: rather than drawing a line into a region with no evidence.
DEFAULT_EXTRAPOLATION_LIMIT = 4.0


@dataclass(frozen=True, slots=True)
class Confidence:
    """How much a prediction may be trusted, and why.

    Returned alongside every prediction so a caller can *see* the reason
    rather than infer it from a number. ``usable`` is false whenever the
    evidence does not cover the question being asked; a caller that wants to
    be conservative tests that flag rather than the number.
    """

    #: A prediction in seconds, or None when there is nothing defensible.
    seconds: float | None
    #: Why that is the answer: "lookup", "ewma", "regression", or "refused".
    basis: str
    #: False when the evidence does not cover this question.
    usable: bool
    #: Successful observations behind it, and the distinct sizes among them.
    samples: int = 0
    distinct_sizes: int = 0
    #: Human-readable explanation, shown in diagnostics.
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """One observed execution: the raw material for measurement.

    **Input shape is a vector, not a scalar.** ``inputs[0]`` is harmless for a
    unary node and quietly wrong for everything else - a join of a 2 GB left
    side against an 8 GB right side recorded 2 GB and lost the other 8.

    **Unmeasured is ``None``, not ``0``.** A peak-memory reading of zero is a
    measurement; "never measured" is a different fact, and a store that
    conflates them eventually averages absent data into a confident number.

    **Wall time and compute time are different fields.**
    ``actual_elapsed_total_ms`` is everything the executor spent on the node,
    including acquiring the engine. ``actual_compute_ms`` is the dispatch
    alone, and is ``None`` whenever the two could not be separated. Reading
    the first as the second is what let a 95 ms cold scan be reported as 95 ms
    of kernel time, with startup then charged on top of it again.
    """

    #: Stable semantic identity of the operation (``aar-op-v1:...``). The
    #: *record* schema is versioned separately by ``schema_version``; the
    #: operation-id prefix is owned by :mod:`aar.ir.identity` and is not
    #: redefined here.
    operation_id: str
    #: Identity of this *position* in the graph; the same operation appearing
    #: twice in one pipeline has two of these.
    graph_node_id: str | None
    #: The machine this was measured on. Required: a record with no target
    #: cannot be attributed to any machine, so it cannot be used.
    target_id: str
    #: Resource budget of that machine at the time, so a measurement taken
    #: under a 4 GB budget is not silently compared with one taken under 64 GB.
    resource_snapshot: str
    engine: str
    #: Version of the engine implementation. A DuckDB upgrade can change cost
    #: substantially, so an unversioned engine id is not enough to reuse.
    engine_version: str = ""

    #: Every input, in declared order.
    input_rows: tuple[int, ...] = ()
    input_bytes: tuple[int, ...] = ()

    output_rows: int | None = None
    output_bytes: int | None = None

    #: Wall time for the whole node, engine acquisition included.
    actual_elapsed_total_ms: float = 0.0
    #: Dispatch time alone, or None when it could not be separated.
    actual_compute_ms: float | None = None
    #: Time spent acquiring the engine, or None when it was reused.
    acquire_ms: float | None = None
    #: None until actually measured - never 0 as a stand-in.
    actual_transfer_bytes: int | None = None
    actual_peak_memory: int | None = None

    #: Which cost model produced the prediction this observation would be
    #: compared against. Mixing models in one series makes the error
    #: meaningless.
    cost_model_version: str = ""
    schema_version: str = HISTORY_SCHEMA_VERSION
    success: bool = True
    failure_kind: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def total_input_bytes(self) -> int:
        return sum(self.input_bytes)

    @property
    def arity(self) -> int:
        return len(self.input_bytes)

    def describe(self) -> str:
        compute = (self.actual_compute_ms
                   if self.actual_compute_ms is not None else float("nan"))
        return (f"{self.operation_id[:24]} {self.engine:<12} "
                f"in={self.input_bytes} out={self.output_rows} "
                f"total={self.actual_elapsed_total_ms:.1f}ms "
                f"compute={compute:.1f}ms")


#: Retained under its historical name so existing callers keep working.
Observation = ExecutionRecord


def _node_input_bytes(node: Any) -> tuple[int, ...]:
    """The sizes of a node's inputs, in declared order.

    The predictor axis has to be the quantity the executor *observed*. A join's
    cost is driven by what it reads, not by what it emits, and its output can
    be orders of magnitude smaller than its inputs - training on output and
    predicting from inputs would compare two different quantities and call the
    result a cost.
    """
    out: list[int] = []
    for parent in getattr(node, "inputs", ()) or ():
        size = getattr(parent, "estimated_bytes", None)
        out.append(int(size) if size else 0)
    return tuple(out)


class ExecutionHistory:
    """Observed executions, kept for diagnosis rather than for planning.

    The API is intentionally small: :meth:`record` accepts an observation,
    :meth:`records` retrieves evidence, and :meth:`predict` answers - or
    explains why it cannot. Nothing here is consulted by the planner while
    ``observe_only`` holds, which is the default.
    """

    __slots__ = ("_records", "_ewma_alpha", "_min_sizes",
                 "_extrapolation_limit", "observe_only", "_target_id",
                 "_resource_snapshot", "_cost_model_version")

    def __init__(self, ewma_alpha: float = 0.3,
                 min_sizes_for_regression: int = DEFAULT_MIN_SIZES_FOR_EXTRAPOLATION,
                 extrapolation_limit: float = DEFAULT_EXTRAPOLATION_LIMIT,
                 target_id: str = "",
                 resource_snapshot: str = "",
                 cost_model_version: str = "",
                 observe_only: bool = True) -> None:
        self._records: list[ExecutionRecord] = []
        self._ewma_alpha = ewma_alpha
        self._min_sizes = max(2, int(min_sizes_for_regression))
        self._extrapolation_limit = float(extrapolation_limit)
        #: True means "collect and report, but do not steer planning". The
        #: default, and not a temporary setting: it is the correct state until
        #: the units below are demonstrated good.
        self.observe_only = observe_only
        #: What this history may attribute measurements to.
        self._target_id = target_id
        self._resource_snapshot = resource_snapshot
        self._cost_model_version = cost_model_version

    # ------------------------------------------------------------- recording
    @property
    def target_id(self) -> str:
        return self._target_id

    def record(self, observation: ExecutionRecord) -> ExecutionRecord:
        """Store one observation and return it unchanged."""
        self._records.append(observation)
        return observation

    def record_node(self, node: Any, outcome: Any, engine_version: str = "",
                    success: bool = True, failure_kind: str = "",
                    target_id: str | None = None,
                    resource_snapshot: str | None = None
                    ) -> ExecutionRecord | None:
        """Build and store an observation from a node and its outcome.

        Returns ``None`` when the node has no stable identity. That is not a
        failure: a node whose UDF cannot be hashed is simply not persistable,
        and skipping it beats storing it under a key that would later collide.
        """
        operation_id = getattr(outcome, "operation_id", None)
        if not operation_id:
            return None
        record = ExecutionRecord(
            operation_id=operation_id,
            graph_node_id=getattr(outcome, "graph_node_id", None),
            target_id=(target_id if target_id is not None else self._target_id),
            resource_snapshot=(resource_snapshot if resource_snapshot is not None
                               else self._resource_snapshot),
            engine=getattr(outcome, "engine_used", ""),
            engine_version=engine_version,
            input_rows=tuple(getattr(outcome, "input_rows", ()) or ()),
            input_bytes=tuple(getattr(outcome, "input_bytes", ()) or ()),
            output_rows=(getattr(outcome, "rows_out", None) if success else None),
            output_bytes=(getattr(outcome, "bytes_out", None) if success else None),
            actual_elapsed_total_ms=float(getattr(outcome, "elapsed_ms", 0.0)),
            # Split only if the executor separated them; otherwise None rather
            # than a guess, because a guessed compute time is a wrong one.
            actual_compute_ms=getattr(outcome, "compute_ms", None),
            acquire_ms=getattr(outcome, "acquire_ms", None),
            actual_transfer_bytes=getattr(outcome, "transfer_bytes", None),
            actual_peak_memory=getattr(outcome, "peak_memory", None),
            cost_model_version=self._cost_model_version,
            success=success,
            failure_kind=failure_kind,
        )
        return self.record(record)

    # ------------------------------------------------------------ retrieval
    def __len__(self) -> int:
        return len(self._records)

    def all(self) -> list[ExecutionRecord]:
        return list(self._records)

    def records(self, operation_id: str, engine: str | None = None,
                target_id: str | None = None,
                successful_only: bool = True) -> list[ExecutionRecord]:
        """Evidence for one operation, narrowed by engine and machine.

        ``successful_only`` defaults to True because a crashed operation's
        duration is not a cost. Failed records are still stored, and still
        reachable, because "how often does this fail?" is a question worth
        being able to ask.
        """
        out = []
        for record in self._records:
            if record.operation_id != operation_id:
                continue
            if successful_only and not record.success:
                continue
            if engine is not None and record.engine != engine:
                continue
            if target_id is not None and record.target_id != target_id:
                continue
            out.append(record)
        return out

    def predictions_disabled(self) -> bool:
        """Whether history is currently barred from influencing planning."""
        return self.observe_only

    # ------------------------------------------------------------ prediction
    def predict_for_node(self, node: Any, engine: str,
                         nbytes: int = 0) -> Confidence:
        """Predict for a *node*, using its real semantic identity.

        This is the only entry point that should reach the planner, and the
        reason is the whole point of the identity work: keying on
        ``node.id`` means a freshly parsed pipeline can never find the last
        run's measurement, because ``node.id`` is a fresh UUID every time. The
        test suite once proved that with a lambda returning
        ``"aar-op-v1:" + node.type``, which is stable for the *wrong* reason
        - it ignores the predicate, the join keys and the scan spec, so a
        filter on ``amount > 100`` and one on ``amount > 999`` would share
        every measurement.

        A node whose identity cannot be computed - an unhashable UDF, say -
        gets a refusal rather than a guess.
        """
        try:
            from ..ir.identity import semantic_operation_id

            operation_id = semantic_operation_id(node)
        except Exception:  # noqa: BLE001 - no stable identity, so no evidence
            return Confidence(None, "refused", False, reason=(
                "this operation has no stable semantic identity, so previous "
                "measurements of it cannot be located"))
        return self.predict(operation_id, engine, _node_input_bytes(node))

    def predict(self, operation_id: str, engine: str,
                input_bytes: Sequence[int],
                target_id: str | None = None) -> Confidence:
        """Predict a duration, or explain why no prediction is defensible.

        Returns a :class:`Confidence` rather than a float, because the useful
        answer is often "no, and here is what is missing".
        """
        target = target_id if target_id is not None else self._target_id
        if not target:
            return Confidence(None, "refused", False, reason=(
                "no target id, so this observation could not be attributed to "
                "any machine"))
        shape = tuple(int(b) for b in input_bytes)
        evidence = self.records(operation_id, engine, target)
        if not evidence:
            return Confidence(None, "refused", False, reason=(
                f"no successful observation of {operation_id[:24]} on "
                f"{engine}/{target}"))

        # Records with a separated compute time describe kernel work; records
        # carrying only wall time remain usable as a fallback but cannot support
        # a slope, because their fixed acquisition cost is not a per-byte term.
        split = [r for r in evidence if r.actual_compute_ms is not None]
        pool = split or evidence
        separable = bool(split)
        sizes = sorted({r.total_input_bytes for r in pool})
        wanted = sum(shape)
        samples, distinct = len(pool), len(sizes)

        # Only same-arity records describe the same shape of work. Mixing a
        # unary record into a join's series compares a 1 GB total against a
        # 10 GB total as though they were the same measurement.
        same_shape = [r for r in pool if len(r.input_bytes) == len(shape)]

        if len(sizes) < 2:
            only = sizes[0] if sizes else 0
            if wanted and only and not self._within_band(wanted, only):
                return Confidence(
                    None, "refused", False, samples=samples,
                    distinct_sizes=distinct,
                    reason=(f"asked about {wanted:,} bytes on evidence from a "
                            f"single size ({only:,}); one point is not a trend"))
            return Confidence(self._centre(pool), "lookup", True,
                              samples=samples, distinct_sizes=distinct,
                              reason=f"{samples} observation(s) at {only:,} bytes")

        if not self._within_band(wanted, sizes[0], sizes[-1]):
            # Outside the observed range. Allowed only with enough distinct
            # sizes to have measured a slope - and only when the times were
            # separable, since an acquisition cost would dominate the fit.
            if len(sizes) < self._min_sizes or not separable:
                return Confidence(
                    None, "refused", False, samples=samples,
                    distinct_sizes=distinct,
                    reason=(f"asked about {wanted:,} bytes, outside the observed "
                            f"range {sizes[0]:,}-{sizes[-1]:,} with {distinct} "
                            f"distinct sizes; refusing to extrapolate"))
            return Confidence(
                self._regress(same_shape or pool, wanted), "regression", True,
                samples=samples, distinct_sizes=distinct,
                reason=f"extrapolated beyond {sizes[-1]:,} bytes on "
                       f"{distinct} distinct sizes")

        if len(sizes) < self._min_sizes or not separable:
            return Confidence(self._centre(same_shape or pool), "ewma", True,
                              samples=samples, distinct_sizes=distinct,
                              reason=(f"{distinct} distinct size(s), below the "
                                      f"{self._min_sizes} needed for a slope; "
                                      f"using a weighted average"))

        return Confidence(self._regress(same_shape or pool, wanted),
                          "regression", True, samples=samples,
                          distinct_sizes=distinct,
                          reason=f"least squares over {samples} observations")

    # ------------------------------------------------------------- internals
    def _within_band(self, wanted: int, *observed: int) -> bool:
        """Whether ``wanted`` is close enough to the evidence to interpolate.

        With one observed size the band is measured outward from it; with
        several it spans the range, widened by the extrapolation limit at the
        top end.
        """
        points = [o for o in observed if o]
        if not points:
            return True
        lo, hi = min(points), max(points)
        return lo <= wanted <= hi * self._extrapolation_limit

    @staticmethod
    def _times(records: list[ExecutionRecord]) -> list[float]:
        return [r.actual_compute_ms if r.actual_compute_ms is not None
                else r.actual_elapsed_total_ms for r in records]

    def _centre(self, records: list[ExecutionRecord]) -> float:
        """An exponentially weighted mean, in seconds.

        Weighted toward recent observations so a machine that drifts - a noisy
        neighbour, a thermally throttled laptop - is tracked rather than
        averaged into a fiction.
        """
        times = self._times(records)
        if not times:
            return 0.0
        value = times[0]
        for item in times[1:]:
            value = self._ewma_alpha * item + (1 - self._ewma_alpha) * value
        return max(0.0, value) / 1e3

    def _regress(self, records: list[ExecutionRecord], wanted: int) -> float:
        """Least squares ``t = a + b*size`` over the observed series."""
        points = [(r.total_input_bytes,
                   r.actual_compute_ms if r.actual_compute_ms is not None
                   else r.actual_elapsed_total_ms) for r in records]
        n = len(points)
        if n < 2:
            return self._centre(records)
        sx = sum(p[0] for p in points)
        sy = sum(p[1] for p in points)
        sxx = sum(p[0] * p[0] for p in points)
        sxy = sum(p[0] * p[1] for p in points)
        det = n * sxx - sx * sx
        if det == 0:
            return self._centre(records)
        slope = (n * sxy - sx * sy) / det
        intercept = (sy - slope * sx) / n
        if slope < 0:
            # Non-monotone evidence. Predicting that bigger is faster would be
            # reading noise as signal, so hold at the worst observation.
            worst = max(p[1] for p in points)
            return max(0.0, worst) / 1e3
        return max(0.0, intercept + slope * wanted) / 1e3

    # ----------------------------------------------------------- diagnostics
    def render(self) -> str:
        """A report of what was measured and how far it may be trusted."""
        if not self._records:
            return ("No execution history: nothing has been observed on this "
                    "machine yet.")
        ok = [r for r in self._records if r.success]
        failed = len(self._records) - len(ok)
        mode = ("observe-only: these observations do not influence planning"
                if self.observe_only else "ACTIVE: influencing planning")
        lines = [f"Execution history ({len(ok)} successful, {failed} failed)",
                 f"  mode: {mode}",
                 f"  schema: {HISTORY_SCHEMA_VERSION}"]
        if not self._target_id:
            lines.append("  target: NONE - observations cannot be attributed "
                         "to a machine, so they will not be used")
        else:
            lines.append(f"  target: {self._target_id}")
        for engine in sorted({r.engine for r in self._records}):
            rows = [r for r in ok if r.engine == engine]
            if not rows:
                continue
            sizes = sorted({r.total_input_bytes for r in rows})
            lines.append(f"  {engine:<14} {len(rows):>4} runs, "
                         f"{sum(self._times(rows)):>9.1f} ms, "
                         f"{len(sizes)} distinct size(s)")
        unseparated = [r for r in ok if r.actual_compute_ms is None]
        if unseparated:
            lines.append(f"  {len(unseparated)} record(s) carry wall time only; "
                         f"they cannot support a slope and are excluded from it")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """A serialisable summary, for the Workbench and for tests."""
        ok = [r for r in self._records if r.success]
        return {
            "schema_version": HISTORY_SCHEMA_VERSION,
            "observe_only": self.observe_only,
            "target_id": self._target_id,
            "observations": len(self._records),
            "successful": len(ok),
            "failed": len(self._records) - len(ok),
            "engines": sorted({r.engine for r in self._records}),
            "operations": sorted({r.operation_id for r in self._records}),
        }
    #: False when the evidence does not cover this question.
    usable: bool
    #: Successful observations behind it, and the distinct sizes among them.
    samples: int = 0
    distinct_sizes: int = 0
    #: Human-readable explanation, shown in diagnostics.
    reason: str = ""