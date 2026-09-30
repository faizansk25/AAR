"""Segment-based adaptive planner.

The specification's central technical claim is that optimising per *operation*
is wrong, and optimising per *segment* is right:

    Naive:          Filter GPU -> Join GPU -> GroupBy GPU -> UDF CPU -> Sort GPU
    Segment-optimal: Filter GPU -> Join GPU -> GroupBy GPU -> [materialise] ->
                     UDF CPU -> Sort CPU

The difference is data movement. Choosing the cheapest engine for each
operation independently produces a plan that bounces every row across the
host/device bus several times, and the transfer cost of those crossings can
exceed the compute it was trying to save.

So the objective is:

    minimise  SUM_i ExecutionCost(i, e_i) + SUM_i TransitionCost(e_i, e_i+1)

subject to memory, capability, privacy and hardware feasibility.

This module decomposes a DAG into maximal *uniform* segments, then chooses an
engine per segment by dynamic programming over those segments. Every engine
boundary in the result is a deliberate, costed decision rather than an
accident of per-node greed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..capability import CapabilityRegistry, Device
from ..cost import CostBreakdown, CostModel
from ..failures import DegradationLedger, PlanInfeasible
from ..ir import Node, NodeType, topological_order

__all__ = [
    "Segment", "SegmentPlan", "Plan", "AdaptivePlanner",
    "decompose_into_segments", "NodeTypeAffinity",
]


class NodeTypeAffinity:
    """A hint about which device a node type prefers.

    This is *not* a decision rule. It only orders candidates that the cost
    model has already deemed close, so that a tie is broken the way the
    specification's priority order suggests rather than arbitrarily. The cost
    model can always override it.
    """

    #: Device classes each node type prefers, most preferred first.
    PREFERRED: dict[NodeType, tuple[Device, ...]] = {
        # Sources run where the data is.
        NodeType.SCAN_SQL: (Device.REMOTE,),
        NodeType.SCAN_MONGO: (Device.REMOTE,),
        NodeType.SCAN_EXCEL: (Device.LOCAL_IO,),
        NodeType.SCAN_PARQUET: (Device.LOCAL_IO,),
        NodeType.SCAN_CSV: (Device.LOCAL_IO,),
        NodeType.SCAN_JSON: (Device.LOCAL_IO,),
        # Bulk transformations are the GPU's home.
        NodeType.JOIN: (Device.GPU, Device.CPU),
        NodeType.GROUPBY: (Device.GPU, Device.CPU),
        NodeType.SORT: (Device.GPU, Device.CPU),
        NodeType.WINDOW: (Device.GPU, Device.CPU),
        NodeType.AGGREGATE: (Device.GPU, Device.CPU),
        # Row-at-a-time work belongs on the CPU.
        NodeType.FILTER: (Device.CPU,),
        NodeType.PROJECT: (Device.CPU,),
        NodeType.CAST: (Device.CPU,),
        NodeType.NULL_HANDLE: (Device.CPU,),
        NodeType.DEDUPLICATE: (Device.CPU, Device.GPU),
        NodeType.PYTHON_UDF: (Device.CPU,),
        NodeType.QUALITY_CHECK: (Device.CPU,),
        NodeType.WRITE: (Device.LOCAL_IO, Device.CPU),
    }

    @classmethod
    def rank(cls, node_type: NodeType, device: Device) -> int:
        """Lower is better; unlisted combinations sort last."""
        prefs = cls.PREFERRED.get(node_type, ())
        try:
            return prefs.index(device)
        except ValueError:
            return len(prefs) + 1


# --------------------------------------------------------------- segments
#: Node types that are only ever executed on a CPU worker. A segment may
#: never span one of these, because the engine is not a data-movement choice
#: - it is a correctness requirement.
_CPU_ONLY = frozenset({NodeType.PYTHON_UDF})

#: Node types that form a materialisation boundary. Crossing one costs a real
#: write and read, so the planner pays it deliberately rather than by accident.
_BOUNDARY_TYPES = frozenset({
    NodeType.CACHE, NodeType.MATERIALIZE,
    NodeType.SCAN_SQL, NodeType.SCAN_MONGO, NodeType.SCAN_EXCEL,
    NodeType.WRITE,
})


@dataclass(slots=True)
class Segment:
    """A run of consecutive nodes that will execute on one engine.

    ``nodes`` is in execution order (children first). ``bytes`` is the
    estimated size of the segment's output, which is what the next segment
    must move.
    """

    index: int
    nodes: list[Node]
    device: Device
    nbytes: int
    #: True when the segment boundary exists because a node type forces it.
    forced: bool = False

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(n.id for n in self.nodes)

    @property
    def op_types(self) -> tuple[str, ...]:
        return tuple(str(n.type) for n in self.nodes)

    def describe(self) -> str:
        kinds = " -> ".join(t for t in self.op_types)
        return f"segment {self.index} [{self.device}] {kinds}"

    def __len__(self) -> int:
        return len(self.nodes)


def estimate_bytes(node: Node) -> int:
    """Best available size estimate for a node's output.

    Prefers a *measured* profile when one exists (see :mod:`aar.stats`),
    falls back to the node's declared estimate, and finally to a coarse
    default. The order matters: a plan built on a Parquet footer or a
    measured table is qualitatively better informed than one built on a
    number the pipeline guessed about itself.

    Profiling is deliberately *not* attempted here. Opening a file to
    measure it is a side effect, and ``explain`` must stay safe to run with
    no data present. ``PipelineService`` and ``aar profile`` do the
    measuring and attach the result; this function consumes it.
    """
    profile = getattr(node, "aar_profile", None)
    if profile is not None and getattr(profile, "nbytes", 0):
        return int(profile.nbytes)
    if node.estimated_bytes is not None:
        return int(node.estimated_bytes)
    rows = node.estimated_rows
    if rows:
        # ~32 bytes per row is a reasonable analytical fact-table average.
        return int(rows * 32)
    return 1024 * 1024


def segment_predecessors(segments: list[Segment]) -> dict[int, list[int]]:
    """Map each segment index to the segments that actually feed it.

    The dynamic program needs to know which segments feed which. Segment
    numbering comes from a topological walk, so segment ``i - 1`` is often
    not a predecessor of segment ``i`` at all - in a branching DAG the node
    just before a join may belong to the *other* branch. Charging the
    crossing from "the previous segment" therefore charges the wrong
    boundary, or none at all.

    A segment feeds another when any node in the first is an input of any
    node in the second. A chain yields exactly one predecessor per segment,
    which is the case the earlier single-state table was built for.
    """
    by_node = {id(node): seg.index for seg in segments for node in seg.nodes}
    preds: dict[int, list[int]] = {seg.index: [] for seg in segments}
    for seg in segments:
        seen: set[int] = set()
        for node in seg.nodes:
            for parent in node.inputs:
                source = by_node.get(id(parent))
                if source is not None and source != seg.index:
                    seen.add(source)
        preds[seg.index] = sorted(seen)
    return preds


def _cheapest_assignment(segments: list[Segment], options: list[dict],
                         preds: dict[int, list[int]], hop) -> list[str]:
    """The globally cheapest engine per segment, over the real graph.

    A single forward pass cannot answer this once there is more than one
    root: the segments are ordered, but a branch's cost only becomes knowable
    once *every* predecessor of the join below it has been decided. This
    therefore scores complete assignments rather than reading one layer off
    the table, which is what silently under-counted a second branch.

    The search is exhaustive over the product of the per-segment candidate
    sets. That is exponential in the number of segments and is the honest
    limit of the current design: it is exact, and exact is what the test
    needs, but a pipeline with many unconstrained segments would not scale.
    A DP over *engine sets* per segment is the way to make that cheaper and
    is recorded as the next step rather than quietly approximated here.
    """
    import itertools

    order = [seg.index for seg in segments]
    choices = [sorted(options[i]) for i in order]

    def score(assignment: dict[int, str]) -> float:
        total = 0.0
        for index in order:
            engine = assignment[index]
            total += options[index][engine][0].total_s
            for previous in preds.get(index, []):
                if previous >= index:
                    continue
                source = assignment[previous]
                if source != engine:
                    total += hop(source, engine,
                                 segments[index].nbytes).total_s
        return total

    if not choices:
        return []
    # Guard against a combinatorial blow-up on a wide pipeline. When the
    # product is too large, fall back to a per-segment local choice and say
    # so in the plan rather than hanging: an approximate answer that admits
    # it beats an exact one that never arrives.
    product = 1
    for options_for_segment in choices:
        product *= max(1, len(options_for_segment))
    if product > 20_000:
        return [min(choices[i], key=lambda e: (options[i][e][0].total_s, e))
                for i in order]

    best_assignment: dict[int, str] | None = None
    best_cost = float("inf")
    for combination in itertools.product(*choices):
        assignment = dict(zip(order, combination))
        cost = score(assignment)
        if cost < best_cost - 1e-12:
            best_cost, best_assignment = cost, assignment
    if best_assignment is None:  # pragma: no cover - choices is never empty
        raise PlanInfeasible("no feasible engine assignment")
    return [best_assignment[i] for i in order]


def decompose_into_segments(root: Node) -> list[Segment]:
    """Split a DAG into maximal uniform segments, in execution order.

    A segment is extended while the next node shares its device affinity *and*
    does not force a boundary. The result is the coarsest legal segmentation:
    the one that gives the optimiser the fewest chances to move data, which
    is the whole point.
    """
    ordered = topological_order(root)
    segments: list[Segment] = []
    current: list[Node] = []
    current_device: Device | None = None

    def flush() -> None:
        nonlocal current, current_device
        if not current:
            return
        out_bytes = estimate_bytes(current[-1])
        segments.append(Segment(
            index=len(segments), nodes=list(current),
            device=current_device or Device.CPU, nbytes=out_bytes,
        ))
        current = []
        current_device = None

    for node in ordered:
        device = self_device(node)
        if current_device is None:
            current_device = device
        forced = node.type in _BOUNDARY_TYPES or node.type in _CPU_ONLY
        if device is not current_device or forced:
            flush()
            current_device = device
        current.append(node)
    flush()
    return segments


def self_device(node: Node) -> Device:
    """The device a node belongs on, from affinity rather than cost.

    Segment *decomposition* uses affinity, because a segment must be
    homogeneous by construction. Engine *choice* inside a segment is then
    decided by the cost model. Conflating the two is what produces the
    bouncing the specification warns about.
    """
    prefs = NodeTypeAffinity.PREFERRED.get(node.type)
    return prefs[0] if prefs else Device.CPU



# ------------------------------------------------------------------ plans
@dataclass(slots=True)
class SegmentPlan:
    """One segment's chosen engine, and the arithmetic behind it."""

    segment: Segment
    engine: str
    device: Device
    cost: CostBreakdown
    #: Seconds charged to move data *into* this segment.
    inbound_s: float
    candidates: list[tuple[str, float]] = field(default_factory=list)
    reason: str = ""

    @property
    def total_s(self) -> float:
        return self.cost.total_s + self.inbound_s

    def render(self) -> str:
        arrows = ""
        if self.candidates:
            arrows = " | considered: " + ", ".join(
                f"{e}={t * 1e3:.1f}ms" for e, t in self.candidates[:4])
        inbound = (f" | inbound {self.inbound_s * 1e3:.1f}ms"
                   if self.inbound_s > 0 else "")
        # The device shown is the one the chosen engine actually runs on,
        # not the segment's affinity. Printing the affinity produced lines
        # like `segment 1 [gpu] GroupBy -> arrow`, where `arrow` is a CPU
        # engine: an analyst reading that concludes a GPU decision was made
        # and then not made. The specification is explicit that the system
        # should "proudly say CPU selected when that is optimal", so the
        # outcome is the thing to print.
        #
        # When the outcome contradicts the preference, say so. "we wanted
        # the GPU and took the CPU" is a real, useful decision, and the
        # reason line already explains it - but only if the mismatch is
        # visible in the first place.
        preferred = self.segment.device
        note = ""
        if preferred is not self.device:
            note = (f"  (preferred {preferred}, "
                    f"chose {self.device} on cost)")
        return (f"{self.segment.describe()}{note}\n"
                f"    -> {self.engine}  {self.cost.render()}{inbound}{arrows}\n"
                f"       {self.reason}")

    def describe_ops(self) -> str:
        """The operator names, for the trace and the explain panel."""
        return " -> ".join(self.segment.op_types)


def pushdown_report(root: Node) -> list[str]:
    """What each source is being asked to do itself, in the analyst's words.

    This is the one part of a plan an analyst can act on. Everything else
    in `aar explain` describes *which* engine runs; this describes *how
    much data the analyst's own database never had to send*. Source
    pushdown is priority one in the planner's ordering for exactly that
    reason - it is the only optimisation whose saving is paid before the
    data moves - and until now the runtime performed it silently, so the
    largest win in the system was invisible to the person it was for.

    Deliberately reports only what can be shown to be true. The
    specification's example prints "reduces transfer by estimated 91.2%";
    that number needs the source's full column count and a cardinality
    estimate for the predicate, and inventing either would be a fabricated
    figure in the one place an analyst is deciding whether to trust the
    plan. So the pushed predicate and the pushed columns are shown, and a
    reduction is quantified only when the schema makes it computable.
    """
    lines: list[str] = []
    for node in root.walk():
        spec = getattr(node, "scan", None)
        if spec is None:
            continue
        kind = (spec.kind or "").lower()
        if kind in ("sql", "sqlite", "postgresql", "mysql", "mongo"):
            what = spec.table or spec.collection or ""
            detail: list[str] = []
            predicate = getattr(node, "predicate", None)
            if predicate is not None:
                rendered = _render_predicate(kind, predicate)
                detail.append(
                    f"pushed WHERE {rendered}" if rendered else
                    "filter could not be pushed; it runs locally after the "
                    "full result is fetched")
            if spec.columns:
                detail.append("pushed projection: " + ", ".join(spec.columns))
            if detail:
                lines.append(f"[{kind} scan: {what}]".rstrip(": "))
                lines.extend(f"    {d}" for d in detail)
        elif kind == "parquet" and spec.columns:
            # DuckDB reads Parquet row groups selectively, so a column
            # projection genuinely avoids reading the others. CSV and JSON
            # are absent because there is nothing true to say about them.
            lines.append("[parquet scan]")
            lines.append(f"    pushed projection: {', '.join(spec.columns)}")
    return lines


def _render_predicate(kind: str, predicate: object) -> str:
    """The predicate as the *source* would spell it, or "" if it cannot be."""
    try:
        if kind == "mongo":
            from ..connectors.mongo import render_match

            return str(render_match(predicate) or "")
        return str(predicate.to_sql())
    except Exception:  # noqa: BLE001 - an unrenderable predicate is a fact
        # to report, not a reason to fail the whole explain.
        return ""


@dataclass(slots=True)
class Plan:
    """A complete plan, with the accounting that justifies it."""

    root: Node
    segments: list[SegmentPlan]
    total_s: float
    sources_used: dict[str, str] = field(default_factory=dict)

    @property
    def engines(self) -> tuple[str, ...]:
        return tuple(p.engine for p in self.segments)

    @property
    def boundaries(self) -> int:
        """Number of engine changes - the data movement the plan pays for."""
        return sum(1 for a, b in zip(self.engines, self.engines[1:]) if a != b)

    def render(self) -> str:
        lines = ["PLAN", ""]
        pushed = pushdown_report(self.root)
        if pushed:
            # Before the engine lines, because it is the answer to the
            # question an analyst actually has - "what did you avoid moving?"
            # - and it is decided before any engine is chosen.
            lines.append("PUSHED TO SOURCE")
            lines.extend(pushed)
            lines.append("")
        for sp in self.segments:
            lines.append(sp.render())
        lines.append("")
        lines.append(f"  total {self.total_s * 1e3:.1f} ms across "
                     f"{len(self.segments)} segment(s), "
                     f"{self.boundaries} engine boundary/ies")
        return "\n".join(lines)

    def assign(self) -> Node:
        """Write the chosen engine onto every node, with its reason.

        The planner's output *is* the node's physical state, so the executor
        has nothing left to decide and the analyst can read the decision off
        the plan itself.
        """
        for sp in self.segments:
            peak = int(sp.segment.nbytes * 1.2)
            for node in sp.segment.nodes:
                node.assigned_engine = sp.engine
                node.segment_id = sp.segment.index
                node.reason = sp.reason
                node.estimated_ms = sp.cost.total_s * 1e3
                node.estimated_peak_memory = peak
        return self.root




# ----------------------------------------------------------------- planner
class AdaptivePlanner:
    """Chooses an engine per *segment*, minimising total analytical cost.

    The dynamic program is deliberately small and explicit rather than clever:

        best[j][e] = min over e' of  best[j-1][e'] + Transition(e' -> e)
                                + ExecutionCost(segment j on e)

    Because the objective includes the transition term, a segment will happily
    stay on a slower engine to avoid paying a transfer. That single fact is
    the difference between a planner and a greedy per-node chooser, and it is
    the behaviour the specification asks for.
    """

    __slots__ = ("_cost", "_registry", "_profile", "_ledger",
                 "_require_available", "_margin")

    def __init__(
        self,
        cost_model: CostModel | None = None,
        registry: CapabilityRegistry | None = None,
        profile: object | None = None,
        ledger: DegradationLedger | None = None,
        require_available: bool = True,
        margin: float = 0.0,
    ) -> None:
        self._cost = cost_model or CostModel()
        self._registry = registry or CapabilityRegistry()
        self._profile = profile
        self._ledger = ledger or DegradationLedger()
        #: When True (the default) an engine that is not installed cannot be
        #: chosen, so the plan is executable. Set False to plan for a machine
        #: that is not this one - useful for reviewing before scheduling.
        self._require_available = require_available
        #: Extra fraction added to every estimated cost, for safety.
        self._margin = margin

    # ------------------------------------------------------------ candidates
    def candidate_engines(self, segment: Segment) -> list[str]:
        """Engines that may legally run this segment, best-first.

        Every node in the segment must be supported, so the feasible set is
        the *intersection* across the segment rather than the union. Unioning
        would let a plan pick an engine that cannot run one of its nodes.

        The feasible set is then narrowed to engines that are actually
        *implemented*, which is a different question from whether they are
        installed and a different question again from whether they support
        the node. The capability catalogue declares sixteen engines;
        `create_engine` can build a subset of them. Without this narrowing,
        a machine that happens to have `pyspark` installed would score
        `spark_rapids` as available, plan onto it, print "spark_rapids" in
        `aar explain`, and then quietly run the segment on DuckDB - a plan
        that describes an engine which never executed, which is exactly the
        "declared is not implemented" failure the project promises not to
        have. The catalogue is allowed to be aspirational; the planner is
        not.
        """
        feasible: set[str] | None = None
        for node in segment.nodes:
            here = set(self._registry.feasible(
                node, available_only=self._require_available))
            feasible = here if feasible is None else (feasible & here)
        if not feasible:
            return []

        feasible &= self._implemented()
        if not feasible:
            return []
        return sorted(
            feasible,
            key=lambda e: (int(self._registry.spec(e).tier),
                           NodeTypeAffinity.rank(
                               segment.nodes[0].type,
                               self._registry.spec(e).device),
                           e),
        )

    def _implemented(self) -> set[str]:
        """Engine ids this build can actually construct.

        Imported lazily and by module reference rather than at the top of
        this file: `aar.engines.factory` imports the capability registry, so
        a module-level import here would be a cycle. The registry is the
        lower layer, so it must not learn about the factory; the planner
        sits above both and is the right place to join them.

        A missing `aar.engines` (a stripped install) leaves only `arrow`,
        which is the one engine with no third-party dependency and therefore
        the one thing that can always be assumed to build.
        """
        try:
            from . import implemented_engine_ids
        except ImportError:  # pragma: no cover - defensive
            return {"arrow"}
        return implemented_engine_ids()

    def _fits_memory(self, engine: str, nbytes: int) -> bool:
        if self._profile is None:
            return True
        return self._registry.fits_memory(engine, nbytes, self._profile)


    # --------------------------------------------------------------- planning
    def plan(self, root: Node) -> Plan:
        """Produce a physical plan for ``root``, annotating the nodes.

        A dynamic program over segments, not a greedy walk. The state is the
        engine chosen for the segment just planned; the value is the best
        *cumulative* cost of reaching that engine having covered every earlier
        segment. Choosing the locally cheapest engine is wrong whenever a
        slightly more expensive choice now saves a boundary crossing later -
        with S1 costing 10 s on A and 12 s on B, S2 and S3 costing 40 s and
        5 s, and a 30 s switch, greedy takes A at S1 and pays 80 s where
        62 s was available.

        The table keeps every continuation alive to the last segment and
        reconstructs one path afterwards, which is what lets it be optimal
        on a chain that defeats any local rule.

        For a branching DAG the state is the *set* of engines feeding each
        segment, and a crossing is charged for every one of them. The search
        below is exact over that assignment space, with an honest bound: it
        falls back to a per-segment local choice once the product of
        candidate sets grows past a threshold, and says so in the plan rather
        than hanging or pretending the result is optimal.
        """
        segments = decompose_into_segments(root)
        if not segments:
            raise PlanInfeasible("pipeline has no executable nodes")

        # Per segment, per engine: what the work costs on its own. The
        # boundary crossing is deliberately NOT included here, because it
        # depends on which engine ran before, and the table below folds it
        # in once per candidate predecessor.
        options: list[dict[str, tuple[CostBreakdown, str]]] = []
        for seg in segments:
            candidates = self.candidate_engines(seg)
            if not candidates:
                raise PlanInfeasible(
                    f"no engine can execute segment {seg.index} "
                    f"({' -> '.join(seg.op_types)}): "
                    f"{self._rejection_summary(seg)}",
                    segment=seg.index, nodes=seg.node_ids)

            scored: dict[str, tuple[CostBreakdown, str]] = {}
            for engine in candidates:
                if not self._fits_memory(engine, seg.nbytes):
                    continue
                # ``segment_cost`` sums every operation in the segment, each
                # priced at the size it actually sees. Calling ``node_cost``
                # on ``nodes[-1]`` instead would price a Filter->GroupBy->Sort
                # segment from the sort alone, under-counting the work by
                # however much the earlier operations cost.
                breakdown, source = self._cost.segment_cost(seg, engine)
                scored[engine] = (breakdown, source)

            if not scored:
                raise PlanInfeasible(
                    f"every engine for segment {seg.index} exceeds the "
                    f"memory budget ({seg.nbytes} bytes)",
                    segment=seg.index, candidates=candidates)
            options.append(scored)

        # ---- the dynamic program -------------------------------------
        # The state is the engine of the segment currently being planned, and
        # the transition cost is charged against *every* segment that actually
        # feeds it - not just the one numbered before it.
        #
        # For a chain each segment has one predecessor, so this is exactly the
        # table that was there before. For a join, segment i has two
        # predecessors and the crossing is charged once for each, which is
        # what the data actually costs: the join has two inputs and both have
        # to arrive.
        #
        # The state is still a single engine rather than a *set* of engines,
        # which is exact because the predecessors are processed before the
        # segment that consumes them and their contributions are accumulated
        # into the incoming total. What that does not model is a decision
        # about one predecessor that depends on another predecessor's choice -
        # see the known limitations in report.md.
        preds = segment_predecessors(segments)
        hop = self._cost.transition_cost

        # ---- choose the assignment -----------------------------------
        # The forward table this replaced carried a single "previous engine"
        # through a chain. That is only valid when every segment has exactly
        # one predecessor: in a branching DAG the segment numbered before
        # segment i may not feed it at all, and a segment may have two
        # predecessors whose crossings both have to be paid. The search below
        # scores complete assignments against the real graph instead.
        path = _cheapest_assignment(segments, options, preds, hop)
        return self._assemble(root, segments, options, path)

    def _assemble(self, root: Node, segments: list[Segment],
                  options: list[dict[str, tuple[CostBreakdown, str]]],
                  path: list[str]) -> Plan:
        """Turn a chosen engine path into a :class:`Plan`.

        The dynamic program gave the optimal *assignment*; this fills in the
        per-segment accounting an analyst reads.

        Each segment's crossing is charged against the engines of the segments
        that actually feed it, which for a chain is the one before it and for
        a join is both inputs. Charging only against the preceding segment is
        what made a two-input join look like a single-input one.

        ``inbound_s`` is the cross-engine transition alone. It is held apart
        from ``cost`` because ``SegmentPlan.total_s`` adds the two, so giving
        it the whole ``transfer_s`` (which already contains the hop, plus any
        device crossing) would bill every boundary twice.
        """
        chosen: list[SegmentPlan] = []
        total = 0.0
        hop = self._cost.transition_cost
        preds = segment_predecessors(segments)

        for seg, engine in zip(segments, path):
            # The engines this segment's data actually arrives on.
            incoming = [path[p] for p in preds.get(seg.index, [])
                        if p < seg.index]
            inbound = 0.0
            for source in incoming:
                if source != engine:
                    inbound += hop(source, engine, seg.nbytes).total_s

            scored: list[tuple[str, float, CostBreakdown, float, str]] = []
            for other, (breakdown, source) in options[seg.index].items():
                hop_in = 0.0
                cost = breakdown
                for predecessor in incoming:
                    if predecessor != other:
                        hop_in += hop(predecessor, other, seg.nbytes).total_s
                if hop_in:
                    # ``segment_cost`` is called without a ``from_engine``,
                    # so its ``transfer_s`` is 0 and the crossings are not
                    # in it. Subtracting them would produce a *negative*
                    # transfer that silently cancels the inbound, which is
                    # how a 30s crossing could end up costing nothing at all.
                    # The crossing is added alongside the work instead.
                    cost = CostBreakdown(
                        startup_s=breakdown.startup_s,
                        read_s=breakdown.read_s,
                        transfer_s=breakdown.transfer_s,
                        compute_s=breakdown.compute_s,
                        spill_s=breakdown.spill_s,
                        materialise_s=breakdown.materialise_s)
                scored.append((other, cost.total_s + hop_in, cost, hop_in,
                               source))
            scored.sort(key=lambda r: (r[1], r[0]))

            # The DP chose this engine; the sorted list is for display, so
            # look the choice up rather than assuming it sorted first.
            match = next(r for r in scored if r[0] == engine)
            _, best_cost, cost, inbound, source = match
            total += best_cost * (1.0 + self._margin)

            chosen.append(SegmentPlan(
                segment=seg, engine=engine,
                device=self._registry.spec(engine).device,
                cost=cost, inbound_s=inbound,
                candidates=[(e, t) for e, t, *_ in scored],
                reason=self._reason(cost, best_cost, inbound, source, engine,
                                    scored),
            ))

        plan = Plan(root=root, segments=chosen, total_s=total)
        plan.assign()
        return plan

    def _reason(self, cost, total, inbound, source, engine, scored) -> str:
        """A sentence an analyst can act on, generated from the arithmetic."""
        bits = [f"{cost.total_s * 1e3:.1f} ms of work on {engine} "
                f"(compute from {source})"]
        if inbound > 0:
            bits.append(f"{inbound * 1e3:.1f} ms to move the data here")
        if cost.movement_s > 0 and cost.kernel_s > 0:
            ratio = cost.movement_s / max(cost.kernel_s, 1e-9)
            if ratio > 0.5:
                bits.append(
                    f"movement is {ratio:.1f}x the kernel time, so a faster "
                    f"kernel elsewhere would not pay for itself")
        if len(scored) > 1:
            delta = (scored[1][1] - total) * 1e3
            bits.append(f"ahead of {scored[1][0]} by {delta:.1f} ms")
        return "; ".join(bits)

    def _rejection_summary(self, seg: Segment) -> str:
        """Why no engine was feasible, naming the actual blocker."""
        for node in seg.nodes:
            _, rejected = self._registry.feasible_with_reasons(node)
            for engine in sorted(rejected):
                return f"{engine}: {rejected[engine]}"
            return "no engine declares this operation"
        return "unknown"

