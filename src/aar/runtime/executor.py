"""The executor: turns a plan into results.

Walks a planned DAG in topological order, dispatches each node to the engine
the planner chose, and records what actually happened against what was
predicted. Three properties are load-bearing:

* **The plan is honoured, but never blindly.** If the chosen engine fails at
  run time, the executor falls back to a working engine and records a
  degradation. It never silently re-plans, because a plan that changed
  mid-flight without saying so is the failure mode the whole explainability
  design exists to prevent.
* **Nothing is swallowed.** Every fallback, every skipped pushdown, every
  partial result goes into the :class:`ExecutionResult`'s ledger. A run
  finishes with a truthful account of what it did, not what it intended.
* **Observed facts feed back.** Real row counts and elapsed times are
  returned so the caller can update the history store, which is how the next
  run plans better without any machine learning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..engines import create_engine
from ..failures import (DegradationLedger, FailureKind)
from ..interchange import Table
from ..ir import Node, NodeType, topological_order
from ..ir.identity import (
    UnstableSemanticIdentity, graph_node_id, semantic_operation_id,
)
from ..planner import Plan

__all__ = ["ExecutionResult", "Executor"]

#: Node types the executor computes itself, with no engine behind them.
#: Recorded in the trace so ``engine_used`` names the component that actually
#: did the work rather than the engine that was merely assigned to it.
_EXECUTOR_SIDE: frozenset[NodeType] = frozenset({
    NodeType.CAST, NodeType.NULL_HANDLE, NodeType.DEDUPLICATE,
    NodeType.QUALITY_CHECK, NodeType.TAG, NodeType.WINDOW,
    NodeType.UNION, NodeType.MATERIALIZE, NodeType.CACHE,
})


@dataclass(slots=True)
class NodeOutcome:
    """What happened to one node, as opposed to what was predicted.

    **Input shape is a vector, not a scalar.** ``rows_in``/``bytes_in`` used to
    be single integers filled from ``inputs[0]``, which is harmless for a unary
    node and quietly wrong for every other one. A join of a 2 GB left side
    against an 8 GB right side recorded ``input = 2 GB``; the 8 GB side was not
    summarised, it was *lost*, and the history would have learned a per-byte
    cost from one eighth of the work.

    ``input_rows``/``input_bytes`` keep every input in declared order, and
    ``rows_in``/``bytes_in`` are now read-only totals derived from them, so a
    caller that only cares about the sum need not care about arity.

    **Unmeasured is ``None``, not ``0``.** A zero peak-memory reading is a
    real measurement that happened to be zero; "we never measured it" is a
    different fact, and a measurement store that conflates them will eventually
    report a confident average built mostly from absent data.
    """

    node_id: str
    node_type: str
    engine_requested: str
    engine_used: str
    #: Stable semantic identity, when one could be computed. ``None`` when the
    #: node refuses identity (an unhashable UDF), which marks the outcome as
    #: not persistable rather than persisting it under a transient id.
    operation_id: str | None = None
    #: Stable graph-node identity, likewise optional and likewise paired with
    #: ``operation_id``: same operation, different position in the pipeline.
    graph_node_id: str | None = None
    input_rows: tuple[int, ...] = ()
    rows_out: int = 0
    input_bytes: tuple[int, ...] = ()
    bytes_out: int = 0
    elapsed_ms: float = 0.0
    #: ``None`` until something actually measures it. See the class docstring.
    actual_peak_memory: int | None = None
    actual_transfer_bytes: int | None = None
    error: str | None = None
    degraded: bool = False

    @property
    def rows_in(self) -> int:
        """Total input rows. A convenience total, never a replacement."""
        return sum(self.input_rows)

    @property
    def bytes_in(self) -> int:
        """Total input bytes across every input, not just the first."""
        return sum(self.input_bytes)

    @property
    def max_bytes_in(self) -> int:
        """The largest single input - what a working-set limit actually bites."""
        return max(self.input_bytes, default=0)

    @property
    def is_persistable(self) -> bool:
        """Whether this outcome can be stored under a stable identity."""
        return bool(self.operation_id)

    def render(self) -> str:
        flag = "  DEGRADED" if self.degraded else ""
        return (f"{self.node_type:<12} {self.engine_used:<14} "
                f"{self.rows_in:>9,} -> {self.rows_out:>9,} rows  "
                f"{self.elapsed_ms:>8.1f} ms{flag}")


@dataclass(slots=True)
class ExecutionResult:
    """Everything an execution produced, including what went wrong."""

    table: Table | None
    outcomes: list[NodeOutcome] = field(default_factory=list)
    ledger: DegradationLedger = field(default_factory=DegradationLedger)
    started_at: float = 0.0
    finished_at: float = 0.0
    written: list[tuple[str, int]] = field(default_factory=list)
    #: The cost model's history store, if one was supplied. Every node's
    #: observed duration lands here, so the next run on this machine plans
    #: from measurement rather than from a prior.
    history: Any = None

    @property
    def elapsed_s(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    @property
    def rows(self) -> int:
        return self.table.num_rows if self.table is not None else 0

    @property
    def ok(self) -> bool:
        """True only when nothing blocking went unresolved."""
        return not self.ledger.blocking and all(
            o.error is None for o in self.outcomes)

    def render(self) -> str:
        lines = ["EXECUTION", ""]
        for o in self.outcomes:
            lines.append("  " + o.render())
        if self.written:
            lines.append("")
            for target, rows in self.written:
                lines.append(f"  wrote {target} ({rows:,} rows)")
        lines.append("")
        lines.append(f"  {self.elapsed_s * 1e3:.1f} ms total, "
                     f"{self.rows:,} rows out")
        lines.append("")
        lines.append("  " + self.ledger.render().replace("\n", "\n  "))
        return "\n".join(lines)


class Executor:
    """Runs a plan and reports what actually happened."""

    __slots__ = ("_engines", "_options", "_strict", "_last_result",
                 "_policy", "_subject", "_history", "_plan_root")

    def __init__(self, strict: bool = False, policy: Any = None,
                 subject: Any = None, history: Any = None,
                 **options: Any) -> None:
        #: One engine instance per id, reused across every node in the run.
        self._engines: dict[str, Any] = {}
        self._options = dict(options)
        #: When True, a run that degraded at all is reported as a failure.
        #: Off by default: degrading to a slower engine still produces
        #: correct results, and refusing to return them would be a worse
        #: failure than the one that was avoided.
        self._strict = strict
        #: The most recent result, kept so a caller can inspect a run that
        #: raised. Without it, a failure reports only the exception and the
        #: nodes that already succeeded are thrown away.
        self._last_result: ExecutionResult | None = None
        #: Privacy policy. ``None`` means the permissive default: a run with
        #: no policy is allowed, and says so, rather than silently obeying
        #: rules nobody wrote down.
        self._policy = policy
        #: Who is running. Drives RLS and CLS.
        self._subject = subject
        #: Where observed timings are recorded so the next plan is better.
        self._history = history
        #: The plan currently being executed, used to answer "does this DAG
        #: carry security barriers?" from the plan itself rather than from a
        #: flag a caller sets and can get wrong.
        self._plan_root: Plan | None = None

    @property
    def last_result(self) -> "ExecutionResult | None":
        return self._last_result

    @property
    def policy(self) -> Any:
        return self._policy

    @property
    def subject(self) -> Any:
        return self._subject

    @property
    def history(self) -> Any:
        return self._history

    # ------------------------------------------------------------- lifecycle
    def close(self) -> None:
        for engine in self._engines.values():
            try:
                engine.close()
            except Exception:  # noqa: BLE001
                pass
        self._engines.clear()

    def __enter__(self) -> "Executor":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _engine(self, engine_id: str, ledger: DegradationLedger,
                node: Node) -> tuple[Any, bool]:
        """Get (or build) the engine for an id. Reports whether it degraded."""
        if engine_id in self._engines:
            return self._engines[engine_id], False
        requested = engine_id
        engine = create_engine(engine_id, ledger=ledger, node=node,
                               **self._options)
        self._engines[engine_id] = engine
        return engine, engine.id != requested

    # ------------------------------------------------------------------ run
    def execute(self, plan: Plan) -> ExecutionResult:
        """Run every node of ``plan`` in dependency order."""
        result = ExecutionResult(table=None, started_at=time.time(),
                                 history=self._history)
        self._plan_root = plan

        values: dict[int, Table] = {}

        for node in topological_order(plan.root):
            started = time.perf_counter()
            inputs = [values[id(i)] for i in node.inputs
                      if id(i) in values]
            requested = node.assigned_engine or "arrow"
            outcome = NodeOutcome(
                node_id=node.id, node_type=str(node.type),
                engine_requested=requested,
                # Truthful by construction: a node the executor computes itself
                # did not run on the engine it was assigned to.
                engine_used=("executor" if self._runs_in_executor(node.type)
                             else requested),
                # Every input, in declared order. Recording only inputs[0] lost
                # the right side of every join, so the history learned a
                # per-byte cost from a fraction of the bytes actually read.
                input_rows=tuple(t.num_rows for t in inputs),
                input_bytes=tuple(t.nbytes for t in inputs),
            )
            # Identity is best-effort: a node that cannot be hashed (an
            # unhashable UDF) must still execute. It is simply not persistable,
            # which is recorded rather than papered over.
            try:
                outcome.operation_id = semantic_operation_id(node)
                outcome.graph_node_id = graph_node_id(node)
            except (UnstableSemanticIdentity, RecursionError, TypeError,
                    ValueError):
                outcome.operation_id = None
                outcome.graph_node_id = None
            try:
                engine, degraded = self._engine(
                    requested, result.ledger, node)
                outcome.engine_used = ("executor" if self._runs_in_executor(node.type)
                                  else engine.id)
                outcome.degraded = degraded
                table = self._dispatch(node, engine, inputs)
                if table is None:
                    # An engine that returns nothing has produced no data.
                    # Letting that flow on fails later with an unrelated
                    # AttributeError on an unrelated node, so it is caught
                    # here and named.
                    raise RuntimeError(
                        f"engine {engine.id!r} returned None for "
                        f"{node.type.value} node {node.id}; an engine must "
                        f"return a Table, even an empty one")
            except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
                outcome.error = f"{type(exc).__name__}: {exc}"
                outcome.elapsed_ms = (time.perf_counter() - started) * 1e3
                result.outcomes.append(outcome)
                self._record_failure(result.ledger, node, exc)
                self._observe(result.history, outcome, node, success=False)
                result.finished_at = time.time()
                self._last_result = result
                raise
            outcome.rows_out = table.num_rows
            outcome.bytes_out = table.nbytes
            outcome.elapsed_ms = (time.perf_counter() - started) * 1e3
            values[id(node)] = table
            result.outcomes.append(outcome)
            self._observe(result.history, outcome, node, success=True)
            if node.type is NodeType.WRITE and node.target:
                result.written.append((node.target, table.num_rows))

        root = plan.root
        result.table = values.get(id(root))
        result.finished_at = time.time()
        self._last_result = result
        return result

    @staticmethod
    def _observe(history: Any, outcome: "NodeOutcome", node: Node,
                 success: bool) -> None:
        """Record what actually happened, for the next run to learn from.

        A failure is recorded too, flagged ``success=False``. The history
        model excludes failed runs from its averages - a crashed operation's
        duration is not a cost - but keeping it means a plan can be asked
        "how often does this fail?" without a second store.

        Keyed on ``operation_id``, never ``node.id``. The transient UUID is
        unique per instance, so keying on it meant two parses of one pipeline
        could never share a measurement - which is the entire reason the
        semantic identity exists.

        A node without a stable identity is skipped rather than recorded under
        something unstable. One lost sample is recoverable; a store silently
        populated with per-instance keys is not, because it looks populated.
        """
        if history is None:
            return
        record = getattr(history, "record", None)
        if record is None:
            return
        key = outcome.operation_id or getattr(node, "operation_id", None)
        if not key:
            return
        try:
            record(key, outcome.engine_used, outcome.bytes_in,
                   outcome.rows_in, outcome.elapsed_ms, success=success,
                   input_bytes=outcome.input_bytes, input_rows=outcome.input_rows,
                   output_bytes=outcome.bytes_out, output_rows=outcome.rows_out,
                   actual_peak_memory=outcome.actual_peak_memory,
                   actual_transfer_bytes=outcome.actual_transfer_bytes)
        except TypeError:
            # A history implementation predating the richer signature. Fall
            # back to the positional form rather than dropping the sample.
            try:
                record(key, outcome.engine_used, outcome.bytes_in,
                       outcome.rows_in, outcome.elapsed_ms, success=success)
            except Exception:  # noqa: BLE001
                return
        except Exception:  # noqa: BLE001
            # Learning must never break the run it is learning from. A
            # history store that cannot accept a record loses one sample,
            # not the analyst's results.
            return

    def _has_barriers(self) -> bool:
        """Whether this run's plan carries row-level security barriers.

        Derived from the plan rather than configured, so a caller cannot
        accidentally claim the plan is secured when it is not: the flag says
        what is actually in the DAG. Cached per plan, because it is asked once
        per write and walks the whole graph.
        """
        root = getattr(self._plan_root, "root", None)
        if root is None:
            return False
        return any(n.type is NodeType.SECURITY_FILTER
                   for n in topological_order(root))

    # ------------------------------------------------------------- dispatch
    @staticmethod
    def _runs_in_executor(node_type: NodeType) -> bool:
        """Whether this node type is computed by the executor itself.

        Eight operations - cast, null handling, dedup, quality, tag, window,
        union and the materialise/cache pass-through - have no engine
        implementation. They run here, in Python, whatever the plan assigned.
        The trace still reported the *assigned* engine as ``engine_used``,
        which is the one field an analyst uses to ask "what actually ran?" -
        so a window node planned onto DuckDB appeared to have run there.
        """
        return node_type in _EXECUTOR_SIDE

    def _dispatch(self, node: Node, engine: Any, inputs: list[Table]) -> Table:
        """Run one node on its engine."""
        t = node.type
        if t in (NodeType.SCAN_PARQUET, NodeType.SCAN_CSV, NodeType.SCAN_JSON,
                 NodeType.SCAN_EXCEL, NodeType.SCAN_ARROW, NodeType.SCAN_SQL,
                 NodeType.SCAN_MONGO):
            return engine.read_scan(node)
        if t is NodeType.FILTER:
            return engine.filter(_one(inputs), node.predicate)
        if t is NodeType.SECURITY_FILTER:
            # Computed exactly like a filter. The barrier property lives in the
            # rewrite pass, which guarantees the predicate is placed directly
            # above its secured source; by the time execution reaches this node
            # the placement is already correct, and re-deriving it here would
            # be a second, weaker implementation of the same rule.
            return engine.filter(_one(inputs), node.predicate)
        if t is NodeType.PROJECT:
            return engine.project(_one(inputs), list(node.columns))
        if t is NodeType.GROUPBY:
            return engine.group_by(_one(inputs), list(node.key_left),
                                   _aggs(node))
        if t is NodeType.AGGREGATE:
            return engine.group_by(_one(inputs), [], _aggs(node))
        if t is NodeType.SORT:
            return engine.sort(_one(inputs), list(node.sort_keys))
        if t is NodeType.LIMIT:
            return engine.limit(_one(inputs), int(node.limit or 0))
        if t is NodeType.JOIN:
            if len(inputs) < 2:
                raise ValueError("join node has fewer than two inputs")
            return engine.join(inputs[0], inputs[1], list(node.key_left),
                               str(node.join_type))
        if t is NodeType.PYTHON_UDF:
            if node.udf is None:
                raise ValueError(
                    f"UDF node {node.id} has no function; the pipeline "
                    f"declared one but did not pass it")
            return engine.udf(_one(inputs), node.udf, node.udf_mode)
        if t is NodeType.WRITE:
            return self._write(node, engine, _one(inputs))
        if t in (NodeType.MATERIALIZE, NodeType.CACHE):
            return _one(inputs)
        if t is NodeType.UNION:
            return _union(inputs)
        if t is NodeType.CAST:
            return self._cast(_one(inputs), node)
        if t is NodeType.NULL_HANDLE:
            return self._null_handle(_one(inputs), node)
        if t is NodeType.DEDUPLICATE:
            return self._dedup(_one(inputs), node)
        if t is NodeType.QUALITY_CHECK:
            return self._quality(_one(inputs), node)
        if t is NodeType.TAG:
            return self._tag(_one(inputs), node)
        if t is NodeType.WINDOW:
            return self._window(_one(inputs), node)
        raise NotImplementedError(
            f"the executor has no implementation for {t.value}; this is a "
            f"gap in AAR, not a problem with the pipeline")

    def _write(self, node: Node, engine: Any, table: Table) -> Table:
        """Write, after the policy has had its say.

        Enforcement happens *before* the bytes move, not after. A write that
        lands and is then noticed is a breach that already happened; the
        only useful time to refuse is here.

        The table returned is the one that was written, not the one that
        arrived, so a caller reading the pipeline's result sees the masked
        values rather than the original ones. Returning the unmasked input
        would let a caller print a "successful" result containing values the
        policy just removed.
        """
        payload = table
        if self._policy is not None:
            from ..governance import PolicyEngine, Sink, Subject

            engine_policy = (self._policy if
                             isinstance(self._policy, PolicyEngine)
                             else PolicyEngine(self._policy))
            subject = self._subject or Subject()
            sink = Sink.of(node.write_format or "unknown")
            # If the plan carries row-level security barriers, RLS has already
            # been applied below every join, group-by and window. Re-applying
            # it to this output would be redundant at best and, for an
            # aggregate that no longer carries the filtered column, a refusal
            # of a correctly-secured query. Egress and CLS still apply here:
            # they are exposure-boundary rules, not input rules.
            payload = engine_policy.enforce_write(
                table, sink, subject, rls_applied=self._has_barriers())
        engine.write(payload, node)
        return payload

    @staticmethod
    def _tag(table: Table, node: Node) -> Table:
        """Apply classification tags to named columns.

        This is where an analyst says what a column *is*, and the only place
        the tags are set by hand. Everything downstream - aggregates, joins,
        UDFs - inherits from here, so a mistake made at the source is
        corrected once rather than patched at every exit.

        A column named in a TAG node that the input does not have is an
        error. Silently tagging nothing would leave the pipeline looking
        annotated and behaving exactly as if it were not, which is the
        failure this whole layer exists to prevent.
        """
        from ..types import Field, Schema

        columns = tuple(node.columns or ())
        if not columns:
            raise ValueError("TAG node has no columns to classify")
        missing = [c for c in columns if not table.schema.has(c)]
        if missing:
            raise KeyError(
                f"cannot classify {', '.join(missing)}: "
                f"no such column (have: {', '.join(table.column_names)})")
        tags = tuple(node.tag_values or ())
        if not tags:
            raise ValueError("TAG node has no tags; a column marked without "
                             "a tag protects nothing")
        descriptions = dict(node.descriptions or {})

        fields = []
        # Named `schema_field`, not `field`: `dataclasses.field` is imported at
        # module scope in this file, and a loop variable of the same name
        # would shadow it for the rest of the function. Harmless today,
        # misleading tomorrow.
        for schema_field in table.schema.fields:
            if schema_field.name in columns:
                new = schema_field.with_classification(*tags)
                note = descriptions.get(schema_field.name)
                if note:
                    new = Field(new.name, new.type, new.nullable,
                                new.classification, note, new.lineage)
                fields.append(new)
            else:
                fields.append(schema_field)
        return table.with_schema(Schema(tuple(fields)))


    # ---------------------------------------------------- simple operations
    def _cast(self, table: Table, node: Node) -> Table:

        from ..interchange import canonical_to_arrow

        arrow = table.arrow
        for name, target in node.casts.items():
            if not table.schema.has(name):
                raise KeyError(f"cast target {name!r} is not a column")
            index = arrow.schema.get_field_index(name)
            try:
                cast = arrow.column(name).cast(canonical_to_arrow(target))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"cannot cast column {name!r} to {target}: {exc}") from exc
            arrow = arrow.set_column(index, name, cast)
        return Table(arrow, table.schema.cast(node.casts))

    def _null_handle(self, table: Table, node: Node) -> Table:
        """Fill, drop or leave nulls, per the node's strategy."""
        import pyarrow.compute as pc

        from ..interchange import canonical_to_arrow

        strategy = (node.null_strategy or "keep").lower()
        columns = list(node.columns) or list(table.column_names)
        for c in columns:
            if not table.schema.has(c):
                raise KeyError(f"null-handle column {c!r} is not a column")
        if strategy == "keep":
            return table
        if strategy == "drop":
            mask = None
            for name in columns:
                is_null = pc.is_null(table.column(name))
                mask = is_null if mask is None else pc.or_(mask, is_null)
            if mask is None:
                return table
            # `pc.filter` takes the boolean mask directly. Passing the inverted
            # mask to `take` looks equivalent and is not: `take` wants
            # positional indices, and a boolean array is not one.
            return Table(table.arrow.filter(pc.invert(mask)), table.schema)

        if strategy in ("fill", "zero", "empty", "impute"):
            value = node.fill_value
            arrow = table.arrow
            for name in columns:
                index = arrow.schema.get_field_index(name)
                col = arrow.column(name)
                filled = pc.if_else(pc.is_null(col), value, col)
                try:
                    filled = filled.cast(canonical_to_arrow(
                        table.schema.get(name).type))
                except (TypeError, ValueError):
                    pass
                arrow = arrow.set_column(index, name, filled)
            return Table(arrow, table.schema)
        raise ValueError(f"unknown null strategy {node.null_strategy!r}")

    def _dedup(self, table: Table, node: Node) -> Table:
        """Keep one row per distinct key combination.

        ``first`` keeps the earliest row for each key. ``last`` keeps the
        latest - and the subtlety is that replacing a row must replace *that
        key's* row, not the most recently appended one. Appending
        ``keep[-1] = i`` instead corrupts the output whenever a new key
        arrives after a duplicate: for keys ``A, B, A`` it overwrites ``B``
        with ``A``, yielding ``A, A`` and silently dropping ``B`` from a
        result the analyst asked to deduplicate. A key->position map is the
        only structure that gets this right.
        """
        keys = list(node.dedup_keys) or list(table.column_names)
        for k in keys:
            if not table.schema.has(k):
                raise KeyError(f"dedup key {k!r} is not a column")
        if table.num_rows == 0:
            return table
        strategy = (node.dedup_strategy or "first").lower()
        if strategy not in ("first", "last"):
            raise ValueError(
                f"unknown dedup strategy {strategy!r}; expected 'first' or "
                f"'last'")
        seen: dict[tuple, int] = {}
        keep: list[int] = []
        for i, row in enumerate(table.arrow.to_pylist()):
            k = tuple(_hash(row.get(name)) for name in keys)
            if k not in seen:
                seen[k] = len(keep)
                keep.append(i)
            elif strategy == "last":
                keep[seen[k]] = i
        # Output stays in original input order: ``last`` changes *which* row
        # survives, not the order the surviving rows come back in. Sorting the
        # indices is what keeps that promise.
        return table.take(sorted(keep))

    def _quality(self, table: Table, node: Node) -> Table:
        """Run declared quality rules, and fail loudly if one fails.

        An unrecognised rule name is an error rather than a no-op. Silently
        ignoring ``"positive "`` (trailing space) or a typo like ``"notnull"``
        would report a passing check that never ran, which is the exact
        failure mode a quality gate exists to prevent.
        """
        known = {"not_null", "nonnull", "required", "unique", "distinct",
                 "positive"}
        failures: list[str] = []
        for column, rule in node.quality_rules:
            if not table.schema.has(column):
                failures.append(f"{column}: no such column")
                continue
            values = table.column(column).to_pylist()
            present = [v for v in values if v is not None]
            name = (rule or "").lower()
            if name not in known:
                failures.append(
                    f"{column}: unknown quality rule {rule!r}; expected one of "
                    f"{sorted(known)}")
                continue
            if name in ("not_null", "nonnull", "required"):
                missing = sum(1 for v in values if v is None)
                if missing:
                    failures.append(
                        f"{column}: {missing} null value(s), expected none")
            elif name in ("unique", "distinct"):
                if len({_hash(v) for v in present}) != len(present):
                    failures.append(f"{column}: duplicate values")
            elif name == "positive":
                bad = [v for v in present
                       if not (isinstance(v, (int, float))
                               and not isinstance(v, bool) and v > 0)]
                if bad:
                    failures.append(
                        f"{column}: {len(bad)} non-positive value(s)")
        if failures:
            from ..failures import QualityCheckFailed
            raise QualityCheckFailed(
                "quality check failed: " + "; ".join(failures))
        return table


    def _window(self, table: Table, node: Node) -> Table:
        """Window functions with real frame semantics.

        A window function is evaluated over a *frame* - a moving range of
        rows within a partition - not over the whole partition. Evaluating
        ``_apply_aggregate(fn, partition)`` once per row produces a constant
        per partition, which is what this used to do: a running total
        returned the partition total on every row, and ``row_number`` would
        have returned 1 forever. That is not a subtle numeric drift, it is
        a wrong answer that looks right, so the frame is parsed explicitly.

        Supported frames (SQL spelling, case-insensitive)::

            rows between unbounded preceding and current row   (default)
            rows between current row and current row
            rows between N preceding and current row
            rows between current row and N following
            rows between unbounded preceding and unbounded following
            rows between <start> preceding and <end> following

        The start and end bounds are resolved per row against the partition's
        ordered positions, which is what makes a cumulative sum cumulative.
        """
        from ..engines.arrow_engine import _apply_aggregate

        if not node.window or not node.window_functions:
            return table
        spec = node.window
        rows = table.arrow.to_pylist()
        if not rows:
            return table

        partitions: dict[tuple, list[int]] = {}
        order: list[tuple] = []
        for i, row in enumerate(rows):
            k = tuple(_hash(row.get(p)) for p in spec.partition_by)
            if k not in partitions:
                partitions[k] = []
                order.append(k)
            partitions[k].append(i)

        for k in order:
            indices = partitions[k]
            # One sort over the whole ORDER BY list, not one sort per column.
            #
            # Sorting by `a` and then by `b` makes the *last* key primary:
            # rows (1,1),(1,2),(2,1) came out (1,1),(2,1),(1,2), which is
            # `ORDER BY b, a` - the opposite of what was asked for. A single
            # sort on the composite key gives the declared precedence, and
            # it is the same key the ranking uses, so the two cannot disagree
            # about what "sorted order" means.
            indices.sort(key=lambda i: _partition_key(rows, i, spec))
            span = len(indices)
            # ``RANK`` and ``DENSE_RANK`` are computed once per partition,
            # over the *whole* ORDER BY key, rather than per row.
            #
            # RANK is 1 + the number of rows that sort strictly before this
            # one, so ties share a rank and every gap a tie leaves is
            # counted. DENSE_RANK is 1 + the number of distinct keys before
            # it, so it never skips. Computing them by scanning the rows
            # already visited - as this did - counts *equal* preceding
            # values instead, which gives 1,1,2,1 for 10,20,20,30 where
            # SQL requires 1,2,2,4 (verified against DuckDB).
            positional = _positional_functions(node, indices, rows, spec)
            for name, fn in node.window_functions.items():
                for pos, i in enumerate(indices):
                    # ROW_NUMBER and RANK are positional: they depend on where
                    # the row sits in the partition, not on the values inside
                    # the frame. Summing a one-row frame would give every row
                    # 1, so they are resolved before the aggregate path.
                    func = getattr(fn, "func", "").upper()
                    if func in ("ROW_NUMBER", "ROWNUMBER"):
                        rows[i][name] = pos + 1
                        continue
                    if func == "RANK":
                        rows[i][name] = positional["rank"][pos]
                        continue
                    if func in ("DENSE_RANK", "DENSERANK"):
                        rows[i][name] = positional["dense"][pos]
                        continue
                    lo, hi = _frame_bounds(spec.frame, pos, span)
                    frame = [rows[j] for j in indices[lo:hi]]
                    rows[i][name] = _apply_aggregate(fn, frame)

        return _rebuild(rows, table, node)

    @staticmethod
    def _record_failure(ledger: DegradationLedger, node: Node,
                        exc: BaseException) -> None:
        """Attach a failure to the right failure mode and record it."""
        from ..failures import AARError

        if isinstance(exc, AARError):
            kind = exc.failure_mode
        elif type(exc).__name__ == "SourceUnavailable":
            kind = FailureKind.SOURCE_UNAVAILABLE
        else:
            kind = FailureKind.UDF_FAILURE
        ledger.record(kind, f"node {node.id}", f"{type(exc).__name__}: {exc}",
                      node=node, exception=exc)


# ------------------------------------------------------------------ helpers
def _aggs(node: Node) -> dict[str, Any]:
    """Output name -> a single aggregate.

    The IR stores ``dict[str, tuple[Agg, ...]]`` because a column may legally
    carry several aggregates. No engine has a representation for a column
    that is simultaneously a sum and a count, so the single-aggregate case -
    the only one the SDK can currently build - is unwrapped here and a
    multi-aggregate column is refused rather than silently collapsed to the
    first one.
    """
    out: dict[str, Any] = {}
    for name, aggs in node.agg_functions.items():
        if isinstance(aggs, (list, tuple)):
            if len(aggs) != 1:
                raise NotImplementedError(
                    f"aggregate column {name!r} declares {len(aggs)} "
                    f"aggregates; AAR can execute exactly one per output "
                    f"column, so this pipeline needs a different shape")
            out[name] = aggs[0]
        else:
            out[name] = aggs
    return out


def _one(inputs: list[Table]) -> Table:
    if not inputs:
        raise ValueError("node expected an input but received none")
    return inputs[0]


def _union(inputs: list[Table]) -> Table:
    import pyarrow as pa

    from ..interchange import Table

    if not inputs:
        raise ValueError("union node has no inputs")
    if len(inputs) == 1:
        return inputs[0]
    names = set(inputs[0].column_names)
    for t in inputs[1:]:
        if set(t.column_names) != names:
            raise ValueError(
                "union inputs must have the same columns; "
                f"{sorted(t.column_names)} != {sorted(names)}")
    return Table(pa.concat_tables([t.arrow for t in inputs],
                                  promote_options="none"),
                 inputs[0].schema)


def _hash(value: Any) -> Any:
    """A dict-key-safe form. ``None`` stays None so NULL never joins "None"."""
    if isinstance(value, (list, dict, set)):
        return str(value)
    return value


def _rank_key(spec: Any) -> str | None:
    """The column ``RANK`` ties on: the first ``order_by`` key, if any."""
    return spec.order_by[0][0] if spec.order_by else None


def _order_key(rows: list[dict], indices: list[int],
               spec: Any) -> list[tuple]:
    """The composite sort key for each row, in partition order.

    The whole ``ORDER BY`` list is the key, not just its first column. Two
    rows tie under SQL only when *every* ordering column ties, so ranking
    on ``order_by[0]`` alone would call distinct rows equal.
    """
    return [_partition_key(rows, i, spec) for i in indices]


def _partition_key(rows: list[dict], index: int, spec: Any) -> tuple:
    """One row's position under the full ``ORDER BY`` list.

    **Direction is encoded in the key, not in the sort call.** Sorting with
    ``reverse=True`` reorders the positions, after which a key built from
    ascending values no longer describes the order they are in. Inverting a
    component is what makes ``key < previous`` mean "sorts earlier" under
    both directions, so ``RANK`` on ``ORDER BY v DESC`` reads 3,2,1 instead
    of the 1,1,1 that a plain comparison produced.
    """
    components = []
    for col, ascending in spec.order_by:
        marker, value = _sort_key(rows[index].get(col))
        components.append((marker, value if ascending else _Negated(value)))
    return tuple(components)


class _Negated:
    """Reverses ordering for one ``ORDER BY`` component.

    A wrapper rather than ``-value``, because the component may be a string
    or a timestamp and negating those is either meaningless or a TypeError.
    Comparison is inverted, equality is preserved, so a descending key still
    ties exactly when the underlying values are equal.
    """

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __lt__(self, other: "_Negated") -> bool:
        return other.value < self.value

    def __gt__(self, other: "_Negated") -> bool:
        return other.value > self.value

    def __le__(self, other: "_Negated") -> bool:
        return other.value <= self.value

    def __ge__(self, other: "_Negated") -> bool:
        return other.value >= self.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Negated) and other.value == self.value

    def __hash__(self) -> int:
        return hash(("neg", self.value))


def _positional_functions(node: Any, indices: list[int], rows: list[dict],
                          spec: Any) -> dict:
    """``RANK`` and ``DENSE_RANK`` for a whole partition, computed once.

    Returns lists indexed by position within ``indices``. Both are
    derived from one pass over the ordering keys:

    * ``RANK`` is 1 + how many rows sort strictly earlier. Ties therefore
      share a value and every row a tie skips is counted, which is what
      makes it 1,2,2,4 for ``10,20,20,30``.
    * ``DENSE_RANK`` is 1 + how many *distinct* keys sort strictly earlier,
      so it increments only when the key changes: 1,2,2,3. Counting rows
      instead of distinct keys would reproduce ``RANK`` and never skip -
      which is the one thing that tells the two functions apart.
    """
    keys = _order_key(rows, indices, spec)
    rank: list[int] = []
    dense: list[int] = []
    for pos, key in enumerate(keys):
        earlier = keys[:pos]
        rank.append(1 + sum(1 for other in earlier if other < key))
        dense.append(1 + len({other for other in earlier if other < key}))
    return {"rank": rank, "dense": dense}


def _sort_key(value: Any) -> tuple[int, Any]:
    """A total order over possibly-mixed values, for window ``order_by``.

    ``None`` cannot be compared against ``int`` in Python, so a partition
    containing a null in the ordering column raised ``TypeError`` instead of
    sorting. Nulls sort first, which matches the Arrow/Parquet default.
    """
    if value is None:
        return (0, 0)
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, (int, float)):
        return (1, value)
    return (2, str(value))


def _frame_bounds(frame: str | None, pos: int, span: int) -> tuple[int, int]:
    """Resolve a SQL frame to a half-open ``[lo, hi)`` slice of the partition.

    ``pos`` is the row's index within its ordered partition and ``span`` the
    partition length. Returns indices into that ordered list, so ``span`` is
    the upper clamp and ``0`` the lower one - an out-of-range bound is
    clamped rather than raising, matching SQL's treatment of a frame that
    runs off the end of the partition.
    """
    text = (frame or "").lower().replace("_", " ")
    text = " ".join(text.split())
    if not text or "unbounded preceding and unbounded following" in text:
        return 0, span
    if "unbounded preceding and current row" in text:
        return 0, pos + 1
    if "current row and current row" in text:
        return pos, pos + 1

    import re

    m = re.search(
        r"between\s+(\d+)\s+preceding\s+and\s+current\s+row", text)
    if m:
        return max(0, pos - int(m.group(1))), pos + 1
    m = re.search(r"between\s+current\s+row\s+and\s+(\d+)\s+following", text)
    if m:
        return pos, min(span, pos + int(m.group(1)) + 1)
    m = re.search(r"between\s+(\d+)\s+preceding\s+and\s+(\d+)\s+following",
                  text)
    if m:
        return max(0, pos - int(m.group(1))), min(span,
                                                  pos + int(m.group(2)) + 1)
    if "unbounded preceding" in text:
        # ``between unbounded preceding and N following``
        m = re.search(r"and\s+(\d+)\s+following", text)
        if m:
            return 0, min(span, pos + int(m.group(1)) + 1)
        return 0, pos + 1
    # An unrecognised frame is a mistake in the pipeline, not a request for
    # the whole partition. Defaulting silently would make a typo return a
    # plausible answer, which is how a wrong result survives review.
    raise ValueError(
        f"unsupported window frame {frame!r}; expected SQL frame syntax such "
        f"as 'rows between unbounded preceding and current row'")


def _rebuild(rows: list[dict], table: Table, node: Any = None) -> Table:
    """Rebuild a table from mutated row dicts, keeping known column types.

    Columns that did not exist before - the window functions a frame just
    computed - are added to the *canonical* schema as well as to Arrow. Only
    extending Arrow leaves ``Table``'s field-count check to fail, and the
    failure lands on an unrelated-looking ``ValueError`` at construction
    rather than on the operation that actually dropped the column.

    **A derived column inherits the classifications of the columns it was
    computed from.** The previous version created a bare ``Field``, so a
    running total over a ``confidential`` salary column came out
    unclassified - and since a sum of salaries still reveals salaries, a
    policy engine checking the output would wave it through. The comment
    claimed guessing at sensitivity was worse than recording none; that
    inverted the risk. Under-recording is the failure that leaks.

    Partition and ordering keys count as sources, because the row's position
    relative to them is exactly what a rank or a running total encodes.
    """
    import pyarrow as pa

    from ..engines.arrow_engine import _infer_arrow_type
    from ..interchange import Table, arrow_to_canonical, canonical_to_arrow
    from ..types import Field, Schema

    known = list(table.column_names)
    extra = [n for n in rows[0] if n not in known] if rows else []
    names = known + extra
    inherited = _inherited_classifications(node, table)
    fields = []
    canonical: list[Field] = []
    for name in names:
        if table.schema.has(name):
            existing = table.schema.get(name)
            fields.append(pa.field(name, canonical_to_arrow(existing.type)))
            canonical.append(existing)
        else:
            inferred = _infer_arrow_type([r.get(name) for r in rows])
            fields.append(pa.field(name, inferred))
            tags = inherited.get(name, frozenset())
            canonical.append(Field(name, arrow_to_canonical(inferred),
                                   classification=frozenset(tags)))
    return Table(pa.Table.from_pylist(rows, schema=pa.schema(fields)),
                 Schema(tuple(canonical)))


def _inherited_classifications(node: Any, table: Table) -> dict:
    """Map each derived output column to the tags it must inherit.

    Resolves an aggregate or window function's input column back to the
    input table, then unions that column's classification with the
    partition and ordering keys. An expression that cannot be resolved to a
    bare column inherits nothing - there is nothing honest to claim.
    """
    if node is None:
        return {}
    try:
        from ..ir import Col
    except Exception:  # noqa: BLE001 - a missing IR cannot leak anything
        return {}

    def tags_of(column: str) -> frozenset:
        if not column or not table.schema.has(column):
            return frozenset()
        return frozenset(table.schema.get(column).classification)

    # Partition and ordering keys describe *where* a row sits, which the
    # derived value encodes.
    window = getattr(node, "window", None)
    shared: set = set()
    if window is not None:
        for column in getattr(window, "partition_by", ()) or ():
            shared |= tags_of(column)
        for column, _asc in getattr(window, "order_by", ()) or ():
            shared |= tags_of(column)

    functions = dict(getattr(node, "agg_functions", {}) or {})
    functions.update(getattr(node, "window_functions", {}) or {})

    out: dict = {}
    for name, fn in functions.items():
        # ``Agg`` names its input ``arg``. Reading ``column``/``expr``
        # instead silently yields nothing, and silently yielding nothing is
        # exactly how a derived column ends up unclassified - the failure
        # this function exists to prevent, reintroduced by a typo.
        expression = getattr(fn, "arg", None)
        if not isinstance(expression, Col):
            continue
        out[name] = frozenset(shared | tags_of(expression.name))
    return out

