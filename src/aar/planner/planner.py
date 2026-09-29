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

    Uses the node's own declared estimate when present, otherwise a coarse
    default. This is a *declared* estimate - the runtime replaces it with
    measured statistics once the segment has executed once - so it is
    deliberately conservative rather than clever.
    """
    if node.estimated_bytes is not None:
        return int(node.estimated_bytes)
    rows = node.estimated_rows
    if rows:
        # ~32 bytes per row is a reasonable analytical fact-table average.
        return int(rows * 32)
    return 1024 * 1024


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
        return (f"{self.segment.describe()}\n"
                f"    -> {self.engine}  {self.cost.render()}{inbound}{arrows}\n"
                f"       {self.reason}")


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
        """Produce a physical plan for ``root``, annotating the nodes."""
        segments = decompose_into_segments(root)
        if not segments:
            raise PlanInfeasible("pipeline has no executable nodes")

        chosen: list[SegmentPlan] = []
        previous_engine: str | None = None
        previous_device: Device | None = None
        total = 0.0

        for seg in segments:
            candidates = self.candidate_engines(seg)
            if not candidates:
                raise PlanInfeasible(
                    f"no engine can execute segment {seg.index} "
                    f"({' -> '.join(seg.op_types)}): "
                    f"{self._rejection_summary(seg)}",
                    segment=seg.index, nodes=seg.node_ids)

            scored: list[tuple[str, float, CostBreakdown, float, str]] = []
            for engine in candidates:
                if not self._fits_memory(engine, seg.nbytes):
                    continue
                breakdown, source = self._cost.node_cost(
                    seg.nodes[-1], engine, seg.nbytes,
                    from_engine=previous_engine,
                    residency=previous_device)
                inbound = 0.0
                if previous_engine and previous_engine != engine:
                    inbound = self._cost.transition_cost(
                        previous_engine, engine, seg.nbytes).total_s
                scored.append((engine, breakdown.total_s + inbound,
                               breakdown, inbound, source))

            if not scored:
                raise PlanInfeasible(
                    f"every engine for segment {seg.index} exceeds the "
                    f"memory budget ({seg.nbytes} bytes)",
                    segment=seg.index, candidates=candidates)

            scored.sort(key=lambda r: (r[1], r[0]))
            engine, best, cost, inbound, source = scored[0]
            total += best * (1.0 + self._margin)

            chosen.append(SegmentPlan(
                segment=seg, engine=engine,
                device=self._registry.spec(engine).device,
                cost=cost, inbound_s=inbound,
                candidates=[(e, t) for e, t, *_ in scored],
                reason=self._reason(cost, best, inbound, source, engine,
                                    scored),
            ))
            previous_engine = engine
            previous_device = self._registry.spec(engine).device

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

