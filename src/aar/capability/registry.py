"""Engine capability registry.

The planner cannot choose an engine without knowing which engines can execute
which operations on which datatypes, and which of those are even installed.
This module is the authority for that question, and it answers it in two
stages:

1. **Declared capability** - what an engine *could* do, from the matrix below.
   Static, versioned with AAR, and the reason a plan can be generated on a
   machine with nothing installed (so ``aar explain plan`` works on a laptop
   that cannot run the pipeline).
2. **Runtime availability** - what is *actually* importable right now.

The intersection is the feasible set. Anything removed by stage 2 is reported
as ``ENGINE_ABSENT`` in the degradation ledger, because "polars_gpu is not
installed" is a decision the analyst should be able to see, not a silent
absence that changes the plan.

Two design commitments:

* **No guessing.** An operation absent from an engine's declared set is
  unsupported, full stop. The registry never infers capability from a name.
* **Absence is explained.** Every engine reports *why* it is unavailable
  (missing module, missing GPU, unmet version) and that reason travels into
  the plan's rationale.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from importlib import metadata
from typing import Iterable, Mapping

from ..failures import DegradationLedger, FailureKind
from ..ir.nodes import Node, NodeType
from ..types import DataType, TypeKind

__all__ = [
    "Device", "Tier", "EngineSpec", "Capability", "CapabilityRegistry",
    "default_registry", "ENGINE_IDS",
]


class Device(str, enum.Enum):
    """Where an engine's work physically happens.

    This drives the cost model: a CPU device has no transfer term, a GPU
    device pays ``T_H2D + T_D2H`` on every boundary crossing, and a
    distributed device pays a network term instead.
    """

    CPU = "cpu"
    GPU = "gpu"
    REMOTE = "remote"       # the source itself, via pushdown
    ACCEL_REMOTE = "accel_remote"  # a distributed accelerator
    LOCAL_IO = "local_io"   # file readers/writers

    def __str__(self) -> str:  # pragma: no cover - display
        return self.value


class Tier(int, enum.Enum):
    """The specification's execution priority order, lowest number wins.

    Encoded as an ordering constraint rather than a hard preference: the
    planner starts from ``TIER_SOURCE_PUSHDOWN`` and only moves down when a
    tier cannot legally execute the work. Ties are broken by measured cost.
    """

    SOURCE_PUSHDOWN = 1
    LOCAL_EMBEDDED = 2
    GPU_ACCELERATION = 3
    DISTRIBUTED = 4
    COMPATIBILITY = 5
    ISOLATED_WORKER = 6

    def __str__(self) -> str:  # pragma: no cover - display
        return self.name.lower()


# ------------------------------------------------------------- capabilities
#: Operations every SQL-speaking source can push down. A source that supports
#: none of these still appears in the registry as a scan-only engine.
_PUSHDOWN_OPS = frozenset({
    NodeType.SCAN_SQL, NodeType.FILTER, NodeType.PROJECT,
    NodeType.GROUPBY, NodeType.AGGREGATE, NodeType.SORT,
    NodeType.DEDUPLICATE, NodeType.LIMIT, NodeType.JOIN, NodeType.WINDOW,
})

#: Everything a general-purpose columnar engine can do in memory.
_COLUMNAR_OPS = frozenset({
    NodeType.SCAN_PARQUET, NodeType.SCAN_CSV, NodeType.SCAN_JSON,
    NodeType.SCAN_ARROW, NodeType.FILTER, NodeType.PROJECT, NodeType.JOIN,
    NodeType.GROUPBY, NodeType.AGGREGATE, NodeType.SORT, NodeType.WINDOW,
    NodeType.DEDUPLICATE, NodeType.CAST, NodeType.NULL_HANDLE,
    NodeType.LIMIT, NodeType.UNION, NodeType.CACHE, NodeType.MATERIALIZE,
})


@dataclass(frozen=True, slots=True)
class EngineSpec:
    """The declared identity and capability of one engine.

    Deliberately static. This is what AAR *knows*, independent of what
    happens to be installed - which is what makes plan generation possible on
    a machine that cannot execute the plan.
    """

    id: str
    label: str
    device: Device
    tier: Tier
    #: Import name probed for availability, or None if always present.
    probe: str | None
    #: Distribution name for version lookup, or None if not applicable.
    distribution: str | None = None
    #: Node types this engine can execute.
    ops: frozenset[NodeType] = frozenset()
    #: Canonical type kinds this engine cannot represent. Absence means
    #: "supports everything the IR can express".
    unsupported_types: frozenset[TypeKind] = frozenset()
    #: Whether data must be moved to reach this engine (drives transfer cost).
    remote: bool = False
    #: Minimum GPU compute capability, or None if no GPU required.
    min_compute_capability: tuple[int, int] | None = None
    #: True when the engine is available with nothing installed, because it is
    #: the running interpreter itself. A Python UDF worker qualifies; a
    #: PostgreSQL connector does not, because it needs a driver.
    #:
    #: Declared explicitly rather than inferred from `probe is None`, because
    #: "no probe" means two very different things and conflating them produced
    #: a registry that claimed PostgreSQL was available on a machine without
    #: `psycopg`.
    intrinsic: bool = False
    notes: str = ""


    def supports(self, node_type: NodeType) -> bool:
        return node_type in self.ops

    def supports_type(self, dt: DataType) -> bool:
        return not any(k in self.unsupported_types for k in _kind_chain(dt))

    def __str__(self) -> str:  # pragma: no cover - display
        return self.id


def _kind_chain(dt: DataType) -> Iterable[TypeKind]:
    """A type and all of its nested element kinds."""
    yield dt.kind
    for child in dt.child_types():
        yield from _kind_chain(child)



# --------------------------------------------------------------- catalogue
#: Operations every SQL-speaking source can push down. A source that supports
#: none of these still appears in the registry as a scan-only engine.
_PUSHDOWN_OPS = frozenset({
    NodeType.SCAN_SQL, NodeType.FILTER, NodeType.PROJECT,
    NodeType.GROUPBY, NodeType.AGGREGATE, NodeType.SORT,
    NodeType.DEDUPLICATE, NodeType.LIMIT, NodeType.JOIN, NodeType.WINDOW,
})

#: Everything a general-purpose columnar engine can do in memory.
_COLUMNAR_OPS = frozenset({
    NodeType.SCAN_PARQUET, NodeType.SCAN_CSV, NodeType.SCAN_JSON,
    NodeType.SCAN_ARROW, NodeType.SCAN_CONST,
    NodeType.FILTER, NodeType.PROJECT, NodeType.JOIN,
    NodeType.GROUPBY, NodeType.AGGREGATE, NodeType.SORT, NodeType.WINDOW,
    NodeType.DEDUPLICATE, NodeType.CAST, NodeType.NULL_HANDLE,
    NodeType.LIMIT, NodeType.UNION, NodeType.CACHE, NodeType.MATERIALIZE,
    NodeType.WRITE, NodeType.QUALITY_CHECK, NodeType.TAG,
})


#: The engine catalogue, in the specification's priority order.
#:
#: Notes on specific entries, because the spec's evidence drives them:
#:  - DuckDB is LOCAL_EMBEDDED and also the reference pushdown engine for
#:    Parquet (projection/filter pushdown verified at 3.35x).
#:  - pandas is COMPATIBILITY: it exists for the installed base of existing
#:    scripts, not because it is fast.
#:  - Python UDFs are ISOLATED_WORKER and never GPU-capable; that is a hard
#:    capability fact, not a preference.
ENGINES: tuple[EngineSpec, ...] = (
    # --- tier 1: source pushdown
    EngineSpec(
        id="postgresql", label="PostgreSQL", device=Device.REMOTE,
        tier=Tier.SOURCE_PUSHDOWN, probe="psycopg", distribution="psycopg",
        ops=_PUSHDOWN_OPS, remote=True,
        notes="filter/projection/aggregate/join/window/regex pushdown verified",
    ),
    EngineSpec(
        id="mysql", label="MySQL/MariaDB", device=Device.REMOTE,
        tier=Tier.SOURCE_PUSHDOWN, probe="pymysql", distribution="pymysql",
        ops=_PUSHDOWN_OPS, remote=True,
        notes="Arrow Flight SQL partial",
    ),
    EngineSpec(
        id="sqlite", label="SQLite", device=Device.LOCAL_IO,
        tier=Tier.SOURCE_PUSHDOWN, probe="sqlite3", distribution=None,
        ops=_PUSHDOWN_OPS,
        notes="window and regex only partial",
    ),
    EngineSpec(
        id="mongodb", label="MongoDB", device=Device.REMOTE,
        tier=Tier.SOURCE_PUSHDOWN, probe="pymongo", distribution="pymongo",
        ops=frozenset({NodeType.SCAN_MONGO, NodeType.FILTER, NodeType.PROJECT,
                       NodeType.GROUPBY, NodeType.AGGREGATE, NodeType.SORT,
                       NodeType.LIMIT, NodeType.WRITE}),
        remote=True,
        notes="$match/$project/$group/$sort/$limit; complex aggs fall back",
    ),
    EngineSpec(
        id="trino", label="Trino", device=Device.REMOTE,
        tier=Tier.SOURCE_PUSHDOWN, probe="trino", distribution="trino",
        ops=_PUSHDOWN_OPS, remote=True,
        notes="federated; Arrow Flight SQL capable",
    ),


    # --- tier 2: local embedded compute
    EngineSpec(
        id="duckdb", label="DuckDB", device=Device.CPU,
        tier=Tier.LOCAL_EMBEDDED, probe="duckdb", distribution="duckdb",
        ops=_COLUMNAR_OPS,
        notes="projection/filter pushdown on Parquet verified (3.35x)",
    ),
    EngineSpec(
        id="polars_cpu", label="Polars (CPU)", device=Device.CPU,
        tier=Tier.LOCAL_EMBEDDED, probe="polars", distribution="polars",
        ops=_COLUMNAR_OPS,
        notes="lazy, parallel, streaming; strong local default",
    ),
    EngineSpec(
        id="arrow", label="Arrow (in-process)", device=Device.CPU,
        tier=Tier.LOCAL_EMBEDDED, probe="pyarrow", distribution="pyarrow",
        ops=frozenset(op for op in _COLUMNAR_OPS
                      if op is not NodeType.WRITE),
        notes="zero-copy within process; always paired with another engine",
    ),

    # --- tier 3: GPU acceleration
    EngineSpec(
        id="polars_gpu", label="Polars GPU (RAPIDS)", device=Device.GPU,
        tier=Tier.GPU_ACCELERATION, probe="cudf", distribution="cudf-cu12",
        ops=frozenset({NodeType.SCAN_PARQUET, NodeType.SCAN_CSV,
                       NodeType.SCAN_ARROW, NodeType.FILTER, NodeType.PROJECT,
                       NodeType.JOIN, NodeType.GROUPBY, NodeType.AGGREGATE,
                       NodeType.SORT, NodeType.DEDUPLICATE, NodeType.CAST,
                       NodeType.NULL_HANDLE, NodeType.LIMIT, NodeType.UNION,
                       NodeType.CACHE, NodeType.MATERIALIZE, NodeType.WRITE}),
        min_compute_capability=(6, 0),
        notes="3.2x SF1K 1 GPU, 23.2x SF3K 8 GPUs; UVM required above VRAM",
    ),
    EngineSpec(
        id="cudf", label="cuDF (cudf.pandas)", device=Device.GPU,
        tier=Tier.GPU_ACCELERATION, probe="cudf", distribution="cudf-cu12",
        ops=frozenset({NodeType.SCAN_PARQUET, NodeType.SCAN_CSV,
                       NodeType.SCAN_ARROW, NodeType.FILTER, NodeType.PROJECT,
                       NodeType.JOIN, NodeType.GROUPBY, NodeType.AGGREGATE,
                       NodeType.SORT, NodeType.DEDUPLICATE, NodeType.CAST,
                       NodeType.NULL_HANDLE, NodeType.LIMIT, NodeType.UNION,
                       NodeType.CACHE, NodeType.MATERIALIZE, NodeType.WRITE}),
        min_compute_capability=(6, 0),
        notes="5GB advanced groupby ~5min -> ~1.5s; drop-in pandas",
    ),


    # --- tier 4: distributed
    EngineSpec(
        id="ray", label="Ray", device=Device.ACCEL_REMOTE,
        tier=Tier.DISTRIBUTED, probe="ray", distribution="ray",
        ops=frozenset({NodeType.SCAN_PARQUET, NodeType.SCAN_CSV,
                       NodeType.SCAN_ARROW, NodeType.FILTER, NodeType.PROJECT,
                       NodeType.JOIN, NodeType.GROUPBY, NodeType.AGGREGATE,
                       NodeType.SORT, NodeType.DEDUPLICATE, NodeType.CAST,
                       NodeType.NULL_HANDLE, NodeType.LIMIT, NodeType.UNION,
                       NodeType.CACHE, NodeType.MATERIALIZE, NodeType.WRITE,
                       NodeType.PYTHON_UDF}),
        remote=True,
        notes="Ray logical CPU is admission info, not physical isolation",
    ),
    EngineSpec(
        id="dask", label="Dask", device=Device.ACCEL_REMOTE,
        tier=Tier.DISTRIBUTED, probe="dask.dataframe", distribution="dask",
        ops=frozenset({NodeType.SCAN_PARQUET, NodeType.SCAN_CSV,
                       NodeType.SCAN_ARROW, NodeType.FILTER, NodeType.PROJECT,
                       NodeType.JOIN, NodeType.GROUPBY, NodeType.AGGREGATE,
                       NodeType.SORT, NodeType.DEDUPLICATE, NodeType.CAST,
                       NodeType.NULL_HANDLE, NodeType.LIMIT, NodeType.UNION,
                       NodeType.CACHE, NodeType.MATERIALIZE, NodeType.WRITE,
                       NodeType.PYTHON_UDF}),
        remote=True, notes="DataFrame-native; natural for pandas/NumPy",
    ),
    EngineSpec(
        id="spark_rapids", label="Spark RAPIDS", device=Device.ACCEL_REMOTE,
        tier=Tier.DISTRIBUTED, probe="pyspark", distribution="pyspark",
        ops=frozenset({NodeType.SCAN_PARQUET, NodeType.SCAN_CSV,
                       NodeType.SCAN_ARROW, NodeType.FILTER, NodeType.PROJECT,
                       NodeType.JOIN, NodeType.GROUPBY, NodeType.AGGREGATE,
                       NodeType.SORT, NodeType.DEDUPLICATE, NodeType.CAST,
                       NodeType.NULL_HANDLE, NodeType.LIMIT, NodeType.UNION,
                       NodeType.CACHE, NodeType.MATERIALIZE, NodeType.WRITE,
                       NodeType.PYTHON_UDF}),
        remote=True, notes="100 TB estate scale; not always faster than local",
    ),

    # --- tier 5: compatibility
    EngineSpec(
        id="pandas", label="pandas", device=Device.CPU,
        tier=Tier.COMPATIBILITY, probe="pandas", distribution="pandas",
        ops=frozenset({NodeType.SCAN_PARQUET, NodeType.SCAN_CSV,
                       NodeType.SCAN_JSON, NodeType.SCAN_ARROW,
                       NodeType.FILTER, NodeType.PROJECT, NodeType.JOIN,
                       NodeType.GROUPBY, NodeType.AGGREGATE, NodeType.SORT,
                       NodeType.WINDOW, NodeType.DEDUPLICATE, NodeType.CAST,
                       NodeType.NULL_HANDLE, NodeType.LIMIT, NodeType.UNION,
                       NodeType.WRITE}),
        notes="existing scripts; correctness and reach, not speed",
    ),

    # --- tier 6: isolated worker
    EngineSpec(
        id="python_worker", label="Python UDF worker", device=Device.CPU,
        tier=Tier.ISOLATED_WORKER, probe=None, distribution=None,
        ops=frozenset({NodeType.PYTHON_UDF}), intrinsic=True,
        notes="arbitrary Python is unsupported on GPU by construction",
    ),

    # --- local IO
    EngineSpec(
        id="excel", label="Excel", device=Device.LOCAL_IO,
        tier=Tier.SOURCE_PUSHDOWN, probe="openpyxl", distribution="openpyxl",
        ops=frozenset({NodeType.SCAN_EXCEL, NodeType.WRITE}),
        notes="first-class source and target; no compute",
    ),
)

ENGINE_IDS: tuple[str, ...] = tuple(e.id for e in ENGINES)



# ------------------------------------------------------------ availability
@dataclass(frozen=True, slots=True)
class Capability:
    """Runtime availability of one engine, with the reason either way.

    ``available`` is never asserted without a ``reason``. An engine that is
    absent produces a sentence an analyst can act on, because "polars_gpu is
    not available" is only useful if it also says why and what would fix it.
    """

    engine: str
    available: bool
    reason: str
    version: str | None = None
    #: Set when the engine is present but blocked by hardware rather than
    #: by a missing package.
    blocked_by: str | None = None

    def render(self) -> str:
        """Operator-facing one-liner.

        The version appears in ``reason`` when known, so it is not appended
        again here - printing it twice looks like a bug in the probe, and it
        is one that an operator would reasonably report.
        """
        mark = "yes" if self.available else "no "
        return f"  [{mark}] {self.engine:<14} {self.reason}"



def _module_importable(name: str) -> bool:
    """True if ``name`` can be imported, without importing it.

    Uses ``importlib.util.find_spec`` where possible: importing cuDF or Spark
    costs seconds and allocates device memory, which is not an acceptable
    price for asking "is it there?".
    """
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def probe_engine(spec: EngineSpec, gpu: Any = None) -> Capability:
    """Decide whether ``spec`` can actually run on this machine right now."""
    if spec.probe and not _module_importable(spec.probe):
        return Capability(
            spec.id, False,
            f"{spec.probe} is not installed",
            blocked_by="missing_package",
        )

    if spec.device is Device.GPU:
        if gpu is None:
            from ..hardware import HardwareProfile

            gpu = HardwareProfile().gpu
        if not gpu.available:
            return Capability(spec.id, False,
                              f"no GPU ({gpu.reason})",
                              blocked_by="no_gpu")
        if spec.min_compute_capability and gpu.compute_capability:
            need = spec.min_compute_capability
            if gpu.compute_capability < need:
                return Capability(
                    spec.id, False,
                    f"GPU compute capability "
                    f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]} "
                    f"< required {need[0]}.{need[1]}",
                    blocked_by="old_gpu")

    version = None
    if spec.distribution:
        try:
            version = metadata.version(spec.distribution)
        except Exception:  # noqa: BLE001
            version = None

    detail = f"{spec.device} engine available"
    if version:
        detail += f" ({version})"
    return Capability(spec.id, True, detail, version=version)



class CapabilityRegistry:
    """Declared capability intersected with runtime availability.

    Construct once per process. The availability probe is memoised because
    the planner asks about every node on every plan, and an import probe that
    runs repeatedly is pure latency.
    """

    __slots__ = ("_specs", "_by_id", "_capabilities", "_probed", "_gpu")

    def __init__(self, specs: Iterable[EngineSpec] = ENGINES) -> None:
        self._specs: tuple[EngineSpec, ...] = tuple(specs)
        self._by_id: dict[str, EngineSpec] = {s.id: s for s in self._specs}
        if len(self._by_id) != len(self._specs):
            raise ValueError("duplicate engine id in capability registry")
        self._capabilities: dict[str, Capability] = {}
        self._probed = False
        self._gpu: Any = None

    # ------------------------------------------------------------- catalogue
    @property
    def engines(self) -> tuple[EngineSpec, ...]:
        return self._specs

    def spec(self, engine_id: str) -> EngineSpec:
        try:
            return self._by_id[engine_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown engine {engine_id!r}; "
                f"known: {', '.join(sorted(self._by_id))}") from exc

    def by_device(self, device: Device) -> tuple[EngineSpec, ...]:
        return tuple(s for s in self._specs if s.device is device)

    def by_tier(self, tier: Tier) -> tuple[EngineSpec, ...]:
        return tuple(s for s in self._specs if s.tier is tier)

    def declared_engines_for(self, node_type: NodeType) -> tuple[str, ...]:
        """Every engine that *could* execute this operation, ignoring install state."""
        return tuple(s.id for s in self._specs if s.supports(node_type))

    # -------------------------------------------------------------- probing
    def probe(self, force: bool = False) -> dict[str, Capability]:
        """Determine what is actually available. Memoised."""
        if self._probed and not force:
            return self._capabilities
        if self._gpu is None:
            from ..hardware import HardwareProfile

            self._gpu = HardwareProfile().gpu
        self._capabilities = {
            s.id: probe_engine(s, self._gpu) for s in self._specs
        }
        self._probed = True
        return self._capabilities

    def capability(self, engine_id: str) -> Capability:
        caps = self.probe()
        try:
            return caps[engine_id]
        except KeyError as exc:
            raise KeyError(f"unknown engine {engine_id!r}") from exc

    def is_available(self, engine_id: str) -> bool:
        return self.capability(engine_id).available

    def unavailable_reasons(self) -> dict[str, str]:
        """Engine -> why it cannot be used, for absent engines only."""
        return {eid: c.reason for eid, c in self.probe().items()
                if not c.available}


    # ----------------------------------------------------------- feasible set
    def feasible(
        self,
        node: Node,
        available_only: bool = True,
    ) -> tuple[str, ...]:
        """Engines that may legally execute ``node``, in tier order.

        Three filters, applied in order, each of which can be the binding
        constraint and each of which is reportable:

        1. what the node itself declares (``supported_engines``),
        2. what the catalogue says the engine can do,
        3. what is actually installed and hardware-compatible.
        """
        candidates = set(self.declared_engines_for(node.type))
        if node.supported_engines:
            candidates &= set(node.supported_engines)
        if available_only:
            caps = self.probe()
            candidates = {e for e in candidates
                          if caps.get(e) and caps[e].available}
        # Tier order is the specification's priority order; within a tier the
        # catalogue order is stable, which keeps plans reproducible.
        return tuple(s.id for s in self._specs if s.id in candidates)

    def feasible_with_reasons(
        self, node: Node
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        """The feasible set plus a per-rejected-engine explanation.

        This is the method the explain panel uses. A rejection the analyst
        cannot see is a rejection they will not trust.
        """
        feasible = set(self.feasible(node))
        rejected: dict[str, str] = {}

        for spec in self._specs:
            if spec.id in feasible:
                continue
            if node.supported_engines and spec.id not in node.supported_engines:
                rejected[spec.id] = "not declared for this operation"
            elif not spec.supports(node.type):
                rejected[spec.id] = f"does not support {node.type.value}"
            else:
                cap = self.capability(spec.id)
                rejected[spec.id] = (
                    cap.reason if not cap.available
                    else "excluded by node constraints")

        if node.type is NodeType.PYTHON_UDF:
            for spec in self._specs:
                if spec.device is Device.GPU and spec.id in rejected:
                    rejected[spec.id] = (
                        "arbitrary Python UDFs are unsupported on GPU "
                        "(hard capability, not a preference)")
        return self.feasible(node), rejected

    def fits_memory(self, engine_id: str, nbytes: int, profile: Any) -> bool:
        """Whether ``nbytes`` fits this engine's memory budget.

        A GPU engine is bounded by VRAM, not RAM, and the specification is
        explicit that a non-UVM chunked reader hits OOM before SF100 - so
        exceeding VRAM is a hard rejection, not a warning.
        """
        spec = self.spec(engine_id)
        if spec.device is Device.GPU:
            return nbytes <= profile.vram_budget_bytes
        return nbytes <= profile.memory_budget_bytes

    def record_absent(self, node: Node, ledger: DegradationLedger) -> None:
        """Log every engine dropped from ``node``'s feasible set.

        Called by the planner. The whole point is that a plan shaped by a
        missing package says so, rather than looking like a considered
        choice.

        Only engines the *node* explicitly declared are considered. An engine
        that was never a candidate has not been "excluded" - it was never
        asked. Reporting every absent engine a node *could* have used would
        bury the real constraint under a wall of "Spark isn't installed" on
        every single node of every plan.
        """
        if not node.supported_engines:
            return
        feasible = set(self.feasible(node))
        for engine_id in sorted(set(node.supported_engines) - feasible):
            if engine_id not in self._by_id:
                # Declared by the caller but not in the catalogue: the caller
                # knows something the registry does not. Say so plainly.
                reason = "declared by the plan but not in the engine catalogue"
            else:
                reason = self.capability(engine_id).reason
            ledger.record(
                FailureKind.ENGINE_ABSENT, "planner",
                f"{engine_id} excluded from {node.type.value}: {reason}",
                node=node, from_engine=engine_id,
            )



    def render(self) -> str:
        """Operator-facing capability table; the body of ``aar capabilities``."""
        caps = self.probe()
        lines = ["ENGINE CAPABILITY", ""]
        current: Tier | None = None
        for spec in self._specs:
            if spec.tier is not current:
                current = spec.tier
                lines.append(f"  tier {int(current)} - {current}")
            lines.append(caps[spec.id].render())
            if spec.notes:
                lines.append(f"        {spec.notes}")
        return "\n".join(lines)


_DEFAULT: CapabilityRegistry | None = None


def default_registry(refresh: bool = False) -> CapabilityRegistry:
    """The process-wide registry, built once.

    A single instance matters: the probe runs an import check per engine, and
    re-running it for every node in a large plan is exactly the kind of
    repeated work that makes planning feel slow.
    """
    global _DEFAULT
    if _DEFAULT is None or refresh:
        _DEFAULT = CapabilityRegistry()
    return _DEFAULT
