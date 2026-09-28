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
from ..failures import (DegradationLedger, FailureKind, UDFExecutionError)
from ..interchange import Table
from ..ir import Node, NodeType, topological_order
from ..planner import Plan

__all__ = ["ExecutionResult", "Executor"]


@dataclass(slots=True)
class NodeOutcome:
    """What happened to one node, as opposed to what was predicted."""

    node_id: str
    node_type: str
    engine_requested: str
    engine_used: str
    rows_in: int = 0
    rows_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    elapsed_ms: float = 0.0
    error: str | None = None
    degraded: bool = False

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
                 "_policy", "_subject", "_history")

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

        values: dict[int, Table] = {}

        for node in topological_order(plan.root):
            started = time.perf_counter()
            inputs = [values[id(i)] for i in node.inputs
                      if id(i) in values]
            requested = node.assigned_engine or "arrow"
            outcome = NodeOutcome(
                node_id=node.id, node_type=str(node.type),
                engine_requested=requested, engine_used=requested,
                rows_in=inputs[0].num_rows if inputs else 0,
                bytes_in=inputs[0].nbytes if inputs else 0,
            )
            try:
                engine, degraded = self._engine(
                    requested, result.ledger, node)
                outcome.engine_used = engine.id
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
        """
        if history is None:
            return
        record = getattr(history, "record", None)
        if record is None:
            return
        try:
            record(node.id, outcome.engine_used, outcome.bytes_in,
                   outcome.rows_in, outcome.elapsed_ms, success=success)
        except Exception:  # noqa: BLE001
            # Learning must never break the run it is learning from. A
            # history store that cannot accept a record loses one sample,
            # not the analyst's results.
            return

    # ------------------------------------------------------------- dispatch

    def _dispatch(self, node: Node, engine: Any, inputs: list[Table]) -> Table:
        """Run one node on its engine."""
        t = node.type
        if t in (NodeType.SCAN_PARQUET, NodeType.SCAN_CSV, NodeType.SCAN_JSON,
                 NodeType.SCAN_EXCEL, NodeType.SCAN_ARROW):
            return engine.read_scan(node)
        if t is NodeType.FILTER:
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
            payload = engine_policy.enforce_write(table, sink, subject)
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
        for field in table.schema.fields:
            if field.name in columns:
                new = field.with_classification(*tags)
                note = descriptions.get(field.name)
                if note:
                    new = Field(new.name, new.type, new.nullable,
                                new.classification, note, new.lineage)
                fields.append(new)
            else:
                fields.append(field)
        return table.with_schema(Schema(tuple(fields)))


    # ---------------------------------------------------- simple operations
    def _cast(self, table: Table, node: Node) -> Table:
        import pyarrow.compute as pc

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
        """Keep one row per distinct key combination."""
        keys = list(node.dedup_keys) or list(table.column_names)
        for k in keys:
            if not table.schema.has(k):
                raise KeyError(f"dedup key {k!r} is not a column")
        if table.num_rows == 0:
            return table
        strategy = (node.dedup_strategy or "first").lower()
        seen: set[tuple] = set()
        keep: list[int] = []
        for i, row in enumerate(table.arrow.to_pylist()):
            k = tuple(_hash(row.get(name)) for name in keys)
            if k not in seen:
                seen.add(k)
                keep.append(i)
            elif strategy == "last":
                keep[-1] = i
        return table.take(keep)

    def _quality(self, table: Table, node: Node) -> Table:
        """Run declared quality rules, and fail loudly if one fails."""
        failures: list[str] = []
        for column, rule in node.quality_rules:
            if not table.schema.has(column):
                failures.append(f"{column}: no such column")
                continue
            values = table.column(column).to_pylist()
            present = [v for v in values if v is not None]
            name = (rule or "").lower()
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
        """A window function, via the row-wise path.

        Window semantics - framing, ordering, partitioning - are the easiest
        thing in an analytical engine to get subtly wrong, so this is explicit
        rather than clever. An engine with native window support should
        override it.
        """
        from ..engines.arrow_engine import _apply_aggregate

        if not node.window:
            return table
        spec = node.window
        rows = table.arrow.to_pylist()
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
            for col, ascending in spec.order_by:
                indices.sort(key=lambda i: rows[i].get(col),
                             reverse=not ascending)
            for name, fn in node.window_functions.items():
                for i in indices:
                    rows[i][name] = _apply_aggregate(
                        fn, [rows[i] for i in indices])
        return _rebuild(rows, table)

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


def _rebuild(rows: list[dict], table: Table) -> Table:
    """Rebuild a table from mutated row dicts, keeping known column types."""
    import pyarrow as pa

    from ..engines.arrow_engine import _infer_arrow_type
    from ..interchange import Table, canonical_to_arrow

    known = list(table.column_names)
    extra = [n for n in rows[0] if n not in known] if rows else []
    names = known + extra
    fields = []
    for name in names:
        if table.schema.has(name):
            fields.append(pa.field(name, canonical_to_arrow(
                table.schema.get(name).type)))
        else:
            fields.append(pa.field(name, _infer_arrow_type(
                [r.get(name) for r in rows])))
    return Table(pa.Table.from_pylist(rows, schema=pa.schema(fields)),
                 table.schema)

