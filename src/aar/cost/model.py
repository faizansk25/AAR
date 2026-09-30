"""Cost model.

    Cost(O, E) = T_startup + T_read + T_transfer + T_compute + T_spill + T_materialize
    T_GPU      = T_H2D + T_kernel + T_D2H + T_startup

The specification's central GPU insight lives in the second line: a kernel can
be several times faster and still lose overall once the transfers are counted.
In its worked example the GPU kernel is 80 ms against 400 ms of CPU work, but
240 ms of H2D plus 160 ms of D2H puts the GPU at 480 ms. The CPU wins.

That outcome must *fall out of arithmetic*, not be asserted. It does here
because the transfer terms are real, calibrated quantities and the planner
adds them up. A model that special-cased "GPU is fast for groupby" would get
that example right and everything else wrong.

Three further commitments:

* **Uncalibrated is not free.** When a curve is missing, :class:`CostModel`
  falls back to a documented conservative prior and says so, so a plan built
  on priors is visibly distinct from one built on measurements.
* **Transitions are first class.** Moving data between engines costs time and
  money, and that is what makes segment optimisation worthwhile at all.
* **Everything is explainable.** Every number can be decomposed into the terms
  that produced it, and the decomposition is retained for the explain panel.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..capability import CapabilityRegistry, Device
from ..hardware.calibrate import CalibrationStore
from ..ir.nodes import Node, NodeType

__all__ = [
    "CostBreakdown", "TransferProfile", "CostModel", "Priors",
    "node_operation", "default_cost_model",
]

#: Node type -> the calibration operation that measures its compute cost.
_OP_FOR_NODE: dict[NodeType, str] = {
    NodeType.SCAN_PARQUET: "parquet_decode",
    NodeType.SCAN_CSV: "csv_decode",
    NodeType.SCAN_JSON: "csv_decode",
    NodeType.SCAN_ARROW: "scan",
    NodeType.SCAN_CONST: "scan",
    NodeType.SCAN_EXCEL: "csv_decode",      # openpyxl is a row parser
    NodeType.SCAN_SQL: "scan",
    NodeType.SCAN_MONGO: "scan",
    NodeType.FILTER: "filter",
    NodeType.PROJECT: "filter",
    NodeType.JOIN: "hash_join",
    NodeType.GROUPBY: "groupby",
    NodeType.AGGREGATE: "groupby",
    NodeType.SORT: "sort",
    NodeType.WINDOW: "window",
    NodeType.DEDUPLICATE: "groupby",
    NodeType.CAST: "filter",
    NodeType.NULL_HANDLE: "filter",
    NodeType.LIMIT: "scan",
    NodeType.UNION: "scan",
    NodeType.PYTHON_UDF: "scan",
    NodeType.QUALITY_CHECK: "scan",
    NodeType.WRITE: "parquet_decode",
    NodeType.CACHE: "arrow_ipc",
    NodeType.MATERIALIZE: "arrow_ipc",
}


def node_operation(node: Node) -> str:
    """The calibration operation that measures this node's compute cost.

    A node with no explicit entry falls back to a whole-column traversal,
    which is the cheapest honest estimate available for an unclassified
    operation.
    """
    return _OP_FOR_NODE.get(node.type, "scan")


# ------------------------------------------------------------------ priors
def _load_saved_calibration() -> CalibrationStore:
    """Load this machine's saved profile, or an empty store.

    ``CalibrationStore()`` with no arguments constructs an *empty* store. So
    a default ``CostModel`` ignored every profile the user had produced with
    ``aar calibrate``, and the machine measured itself only when a caller
    happened to pass the store in explicitly - which the default CLI path did
    not do. Calibrating and then not using the result is worse than not
    calibrating, because the plan claims to be measured and is not.

    Failures here are not fatal: an unreadable or foreign profile degrades to
    priors, and the planner reports ``prior`` as the source, so a broken file
    cannot make a plan claim evidence it does not have.
    """
    try:
        from ..hardware.calibrate import PROFILE_PATH, CalibrationStore

        return CalibrationStore.load(PROFILE_PATH)
    except Exception:  # noqa: BLE001 - priors are always a safe fallback
        return CalibrationStore()


@dataclass(frozen=True, slots=True)
class Priors:
    """Conservative defaults used where the machine has not been measured.

    Every value is a *pessimistic* estimate, and the planner marks any plan
    that relies on them. Pessimism is the correct bias: over-estimating the
    cost of an engine makes the planner more likely to keep work local and
    avoid transfers, which is the safe direction. Optimistic priors would
    send work to a GPU that turns out to be slower.

    The transfer rates are the notable ones. PCIe 4.0 x16 is ~24 GB/s
    effective, well below the ~200+ GB/s of device memory - which is exactly
    why transfers dominate for many workloads.
    """

    #: Bytes per second for host -> device. Measured or assumed PCIe.
    h2d_bytes_per_s: float = 12.0e9
    #: Device -> host. Typically faster than the inbound direction.
    d2h_bytes_per_s: float = 16.0e9
    #: One-off context/session creation on an accelerator, in seconds.
    gpu_context_s: float = 0.85
    #: Per-engine initialisation, in seconds.
    startup_s: float = 0.010
    gpu_startup_s: float = 0.90
    remote_startup_s: float = 0.050
    #: Fixed per-call overhead in the interchange layer, in seconds.
    handoff_s: float = 0.0005
    #: Serialisation cost crossing a process boundary, in seconds *per byte*.
    #: Per byte, not per call: the cost is genuinely proportional to the
    #: volume crossing the boundary, and expressing it per byte keeps it in
    #: the same units as the bandwidth terms beside it. The default is
    #: ~50 MB/s, a slow but real serialisation rate. It was documented and
    #: named as a flat number of *seconds* while being multiplied by nbytes,
    #: so 1 MB cost 20 s - a penalty large enough to make the planner reject
    #: correct plans on the strength of a unit error.
    serialise_s_per_byte: float = 2.0e-8
    #: Cost of writing and re-reading a materialised intermediate, per byte.
    materialise_bytes_per_s: float = 1.0e9
    #: Spill bandwidth, per byte. A rotational disk; conservative.
    spill_bytes_per_s: float = 150e6
    #: Multiplier applied to a low-confidence calibration fit.
    low_confidence_penalty: float = 1.5

    def render(self) -> str:
        return (f"priors: h2d={self.h2d_bytes_per_s / 1e9:.1f} GB/s "
                f"d2h={self.d2h_bytes_per_s / 1e9:.1f} GB/s "
                f"gpu_ctx={self.gpu_context_s * 1e3:.0f} ms "
                f"materialise={self.materialise_bytes_per_s / 1e9:.1f} GB/s")


DEFAULT_PRIORS = Priors()


@dataclass(frozen=True, slots=True)
class TransferProfile:
    """Per-engine and per-pair movement characteristics.

    Held separately from the cost model so a cluster with RDMA, or a host with
    measured PCIe bandwidth, can supply real numbers instead of priors.
    """

    h2d_bytes_per_s: float = DEFAULT_PRIORS.h2d_bytes_per_s
    d2h_bytes_per_s: float = DEFAULT_PRIORS.d2h_bytes_per_s
    network_bytes_per_s: float = 1.25e8   # 1 Gbit/s
    serialise_s_per_byte: float = DEFAULT_PRIORS.serialise_s_per_byte
    in_process: bool = True
    rdma: bool = False

    def to_device(self, nbytes: int) -> float:
        return nbytes / max(1.0, self.h2d_bytes_per_s)

    def to_host(self, nbytes: int) -> float:
        return nbytes / max(1.0, self.d2h_bytes_per_s)

    def over_network(self, nbytes: int) -> float:
        return nbytes / max(1.0, self.network_bytes_per_s)



# ---------------------------------------------------------------- breakdown
@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """A cost, decomposed into the terms that produced it.

    Retained rather than summed away so the explain panel can answer "why is
    the GPU slower here?" with arithmetic instead of an assertion.
    """

    startup_s: float = 0.0
    read_s: float = 0.0
    transfer_s: float = 0.0
    compute_s: float = 0.0
    spill_s: float = 0.0
    materialise_s: float = 0.0

    @property
    def total_s(self) -> float:
        return (self.startup_s + self.read_s + self.transfer_s
                + self.compute_s + self.spill_s + self.materialise_s)

    @property
    def kernel_s(self) -> float:
        """Time in the kernel only, excluding all movement."""
        return self.compute_s

    @property
    def movement_s(self) -> float:
        return self.transfer_s + self.materialise_s

    @property
    def overhead_fraction(self) -> float:
        """Share of the total that is not kernel time.

        A high value is the signal that an accelerator is not worth it: the
        data has to cross the bus more often than it is computed on.
        """
        t = self.total_s
        return 0.0 if t <= 0 else max(0.0, 1.0 - self.kernel_s / t)

    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        return CostBreakdown(
            self.startup_s + other.startup_s,
            self.read_s + other.read_s,
            self.transfer_s + other.transfer_s,
            self.compute_s + other.compute_s,
            self.spill_s + other.spill_s,
            self.materialise_s + other.materialise_s,
        )

    def scaled(self, factor: float) -> "CostBreakdown":
        return CostBreakdown(
            self.startup_s * factor, self.read_s * factor,
            self.transfer_s * factor, self.compute_s * factor,
            self.spill_s * factor, self.materialise_s * factor,
        )

    def render(self, unit: str = "ms") -> float:
        k = 1e3 if unit == "ms" else 1.0
        parts = [
            f"startup {self.startup_s * k:.1f}{unit}",
            f"read {self.read_s * k:.1f}{unit}",
            f"transfer {self.transfer_s * k:.1f}{unit}",
            f"compute {self.compute_s * k:.1f}{unit}",
        ]
        if self.spill_s:
            parts.append(f"spill {self.spill_s * k:.1f}{unit}")
        if self.materialise_s:
            parts.append(f"materialise {self.materialise_s * k:.1f}{unit}")
        return f"{self.total_s * k:.1f}{unit}  (" + ", ".join(parts) + ")"

    def to_dict(self) -> dict[str, float]:
        return {
            "startup_s": self.startup_s, "read_s": self.read_s,
            "transfer_s": self.transfer_s, "compute_s": self.compute_s,
            "spill_s": self.spill_s, "materialise_s": self.materialise_s,
            "total_s": self.total_s,
        }



class CostModel:
    """Estimates what a node will cost on a given engine.

    Combines three sources, in descending order of trust:

    1. a **measured** calibration curve for (operation, device),
    2. the **history** of real executions, when the planner has it,
    3. a **documented pessimistic prior**, when neither exists.

    Whichever source was used is recorded in the breakdown, so a plan built
    on priors can be told apart from a plan built on measurements. That
    distinction is the difference between "the system knows" and "the system
    guessed", and the analyst is entitled to it.
    """

    __slots__ = ("_calibration", "_registry", "_priors", "_transfer",
                 "_history")

    def __init__(
        self,
        calibration: CalibrationStore | None = None,
        registry: CapabilityRegistry | None = None,
        priors: Priors = DEFAULT_PRIORS,
        transfer: TransferProfile | None = None,
        history: "ExecutionHistory | None" = None,
    ) -> None:
        self._calibration = (calibration if calibration is not None
                            else _load_saved_calibration())
        self._registry = registry if registry is not None else CapabilityRegistry()
        self._priors = priors
        self._transfer = transfer or TransferProfile()
        self._history = history

    # ------------------------------------------------------------ properties
    @property
    def calibration(self) -> CalibrationStore:
        return self._calibration

    @property
    def is_calibrated(self) -> bool:
        return self._calibration.is_calibrated

    # ------------------------------------------------------------- computing
    def device_for(self, engine_id: str) -> str:
        """The calibration device key for an engine."""
        return self._registry.spec(engine_id).device.value

    def compute_s(self, node: Node, engine_id: str, nbytes: int) -> tuple[float, str]:
        """Kernel time and where the number came from.

        Returns ``(seconds, source)`` where source is one of
        ``"history"``, ``"calibration"``, ``"calibration(penalised)"`` or
        ``"prior"``.
        """
        op = node_operation(node)
        device = self.device_for(engine_id)

        if self._history is not None:
            observed = self._history.predict(node, engine_id, nbytes)
            if observed is not None:
                return observed, "history"

        curve = self._calibration.get(op, device)
        if curve is not None:
            value = curve.predict(nbytes)
            if curve.low_confidence:
                return (value * self._priors.low_confidence_penalty,
                        "calibration(penalised)")
            return value, "calibration"

        if device == "gpu":
            # No GPU curve exists because no GPU was measured. Falling back to
            # a CPU curve would be an *optimistic* error, so apply a
            # penalty and mark it.
            cpu = self._calibration.get(op, "cpu")
            if cpu is not None:
                return (cpu.predict(nbytes) * self._priors.low_confidence_penalty,
                        "calibration(penalised)")
        return self._prior_s(op, nbytes), "prior"

    def _prior_s(self, op: str, nbytes: int) -> float:
        """A pessimistic estimate used when nothing has been measured."""
        # Deliberately slow: ~100 MB/s effective for an unknown operation.
        per_byte = 1e-8
        multipliers = {
            "filter": 0.3, "scan": 0.6, "project": 0.3, "cast": 0.2,
            "sort": 4.0, "hash_join": 3.0, "groupby": 2.0, "window": 3.0,
            "parquet_decode": 2.0, "csv_decode": 5.0, "arrow_ipc": 0.5,
            "string_ops": 1.0,
        }
        return self._priors.startup_s + per_byte * nbytes * multipliers.get(op, 1.0)

    def startup_s(self, engine_id: str) -> float:
        spec = self._registry.spec(engine_id)
        if spec.device is Device.GPU:
            return self._priors.gpu_startup_s
        if spec.remote:
            return self._priors.remote_startup_s
        return self._priors.startup_s

    def read_s(self, node: Node, nbytes: int) -> float:
        """Reading from storage or a remote source."""
        if node.type in (NodeType.SCAN_PARQUET, NodeType.SCAN_CSV,
                         NodeType.SCAN_JSON, NodeType.SCAN_EXCEL):
            disk = self._calibration.get("sequential_read", "disk")
            if disk is not None:
                return disk.predict(nbytes)
            return nbytes / self._priors.spill_bytes_per_s
        if node.type in (NodeType.SCAN_SQL, NodeType.SCAN_MONGO):
            net = self._calibration.get("sequential_read", "disk")
            if net is not None and net.slope > 0:
                return nbytes * net.slope
            return self._transfer.over_network(nbytes)
        return 0.0


    def node_cost(
        self,
        node: Node,
        engine_id: str,
        nbytes: int,
        from_engine: str | None = None,
        residency: Device | None = None,
    ) -> tuple[CostBreakdown, str]:
        """Full cost of one node on one engine, plus the source of its compute.

        ``residency`` says where the input data already is, which is what
        decides whether a transfer is paid at all. Two consecutive GPU nodes
        pay no transfer; a GPU node fed from a CPU scan pays both directions.
        Getting this wrong is precisely how a plan ends up bouncing data
        between CPU and GPU on every step.
        """
        compute, source = self.compute_s(node, engine_id, nbytes)
        spec = self._registry.spec(engine_id)

        transfer = 0.0
        if spec.device is Device.GPU:
            if residency is not Device.GPU:
                transfer += self._transfer.to_device(nbytes)
            if not node.is_sink:
                # The result has to come back to be consumed by whatever
                # runs next; only a sink keeps it resident.
                transfer += self._transfer.to_host(nbytes)
        elif residency is Device.GPU:
            # Coming off the device onto the host.
            transfer += self._transfer.to_host(nbytes)

        if from_engine and from_engine != engine_id:
            transfer += self.transition_cost(from_engine, engine_id, nbytes).total_s

        breakdown = CostBreakdown(
            startup_s=self.startup_s(engine_id),
            read_s=self.read_s(node, nbytes),
            transfer_s=transfer,
            compute_s=compute,
            spill_s=self._spill_s(node, engine_id, nbytes),
            materialise_s=self._materialise_s(node, nbytes),
        )
        return breakdown, source

    def _spill_s(self, node: Node, engine_id: str, nbytes: int) -> float:
        """Cost of spilling when the working set exceeds the engine's memory.

        Only charged when the caller says the working set actually exceeds
        the budget. Charging unconditionally would mean every GPU plan paid
        a spill cost for work that comfortably fits in VRAM, which biases the
        optimiser toward CPU for reasons that have nothing to do with reality.
        """
        if not node.spillable or not nbytes:
            return 0.0
        spec = self._registry.spec(engine_id)
        if spec.device is not Device.GPU:
            return 0.0
        try:
            from ..hardware import HardwareProfile

            vram = HardwareProfile().vram_budget_bytes
        except Exception:  # noqa: BLE001
            vram = 0
        if vram and nbytes <= vram:
            return 0.0
        return nbytes / self._priors.spill_bytes_per_s


    def _materialise_s(self, node: Node, nbytes: int) -> float:
        if node.type is not NodeType.MATERIALIZE:
            return 0.0
        return nbytes / self._priors.materialise_bytes_per_s

    # ------------------------------------------------------------ transition
    def transition_cost(
        self, from_engine: str, to_engine: str, nbytes: int
    ) -> CostBreakdown:
        """Cost of moving data between two engines.

        Same device and same process: free (zero-copy via the C Data
        Interface). Same device, different process: serialisation only.
        Across the CPU/GPU boundary: full transfer each way. Across a network
        boundary: network bandwidth, and RDMA when available.
        """
        if from_engine == to_engine:
            return CostBreakdown()

        a = self._registry.spec(from_engine)
        b = self._registry.spec(to_engine)

        if a.device is Device.GPU and b.device is Device.GPU:
            # Device to device is a peer copy, not two host transfers.
            return CostBreakdown(
                transfer_s=nbytes / self._transfer.h2d_bytes_per_s,
                materialise_s=nbytes * self._transfer.serialise_s_per_byte)

        if {a.device, b.device} == {Device.GPU, Device.ACCEL_REMOTE}:
            return CostBreakdown(
                transfer_s=self._transfer.over_network(nbytes),
                materialise_s=nbytes * self._transfer.serialise_s_per_byte)

        crosses_bus = Device.GPU in (a.device, b.device)
        crosses_net = a.remote or b.remote

        transfer = 0.0
        serialise = 0.0
        if crosses_bus:
            transfer += nbytes / self._transfer.h2d_bytes_per_s
            transfer += nbytes / self._transfer.d2h_bytes_per_s
            serialise = nbytes * self._transfer.serialise_s_per_byte
        elif crosses_net:
            transfer += self._transfer.over_network(nbytes)
            serialise = nbytes * self._transfer.serialise_s_per_byte
        elif not self._transfer.in_process:
            # Different process: an IPC hop, and the bytes must be serialised.
            transfer += nbytes / self._transfer.network_bytes_per_s
            serialise = nbytes * self._transfer.serialise_s_per_byte
        # In-process CPU to CPU is genuinely free: the C Data Interface
        # hands over the same buffers with no copy and no serialisation.
        # Charging for it would make a segment optimiser avoid perfectly
        # good local handoffs, which is the opposite of the intent.

        return CostBreakdown(transfer_s=transfer, materialise_s=serialise)


    def peak_memory_b(self, node: Node, engine_id: str, nbytes: int) -> int:
        """Estimated peak memory for this node on this engine.

        Rough by design: a join holds both sides plus output, a group-by
        holds a hash table, and a filter holds output only. The planner uses
        this for feasibility, not for reporting.
        """
        spec = self._registry.spec(engine_id)
        multiplier = {
            NodeType.JOIN: 3.0, NodeType.GROUPBY: 2.5, NodeType.AGGREGATE: 2.0,
            NodeType.SORT: 2.0, NodeType.WINDOW: 2.0, NodeType.UNION: 2.0,
            NodeType.DEDUPLICATE: 1.8,
        }.get(node.type, 1.2)
        total = int(nbytes * multiplier)
        if spec.device is Device.GPU:
            return int(total * 1.15)   # device allocations fragment
        return total


    def explain(self, node: Node, engine_id: str, nbytes: int) -> str:
        """One-paragraph justification, for the explain panel."""
        breakdown, source = self.node_cost(node, engine_id, nbytes)
        spec = self._registry.spec(engine_id)
        bits = [
            f"{spec.label} on {spec.device}: {breakdown.render()}",
            f"compute from {source}",
        ]
        if breakdown.overhead_fraction > 0.5:
            bits.append(
                f"{breakdown.overhead_fraction * 100:.0f}% of the time is "
                f"movement, not computation")
        return " | ".join(bits)


# ----------------------------------------------------------------- history
@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """One observed execution, the raw material for learning."""

    operator_hash: str
    engine: str
    nbytes: int
    rows: int
    elapsed_ms: float
    peak_memory: int = 0
    bytes_transferred: int = 0
    success: bool = True


class ExecutionHistory:
    """Learns cost from what actually happened, locally and deterministically.

    No LLM, no ML model, no network. Three estimators, chosen by how much
    evidence exists:

    * **one observation** - a lookup, the plainest possible thing,
    * **several at similar sizes** - an exponentially weighted moving average,
      which tracks a machine that is drifting,
    * **several across sizes** - least-squares on size, which recovers the
      fixed/per-byte split the calibration layer estimates but from *this*
      workload on *this* machine.

    Failures are never averaged in. A record with ``success=False`` is kept
    for diagnosis and excluded from the model, because a crashed run's
    duration is not a cost.
    """

    __slots__ = ("_records", "_ewma_alpha", "_min_samples_for_regression")

    def __init__(self, ewma_alpha: float = 0.3,
                 min_samples_for_regression: int = 4) -> None:
        self._records: list[ExecutionRecord] = []
        self._ewma_alpha = ewma_alpha
        self._min_samples_for_regression = min_samples_for_regression

    def record(
        self, operator_hash: str, engine: str, nbytes: int, rows: int,
        elapsed_ms: float, peak_memory: int = 0, bytes_transferred: int = 0,
        success: bool = True,
    ) -> ExecutionRecord:
        rec = ExecutionRecord(operator_hash, engine, nbytes, rows,
                              elapsed_ms, peak_memory, bytes_transferred,
                              success)
        self._records.append(rec)
        return rec

    def __len__(self) -> int:
        return len(self._records)

    def records(self, operator_hash: str, engine: str) -> list[ExecutionRecord]:
        return [r for r in self._records
                if r.success and r.operator_hash == operator_hash
                and r.engine == engine]

    def predict(self, node: Node, engine_id: str, nbytes: int) -> float | None:
        """Predicted seconds, or ``None`` when there is not enough evidence."""
        obs = self.records(node.id, engine_id)
        if not obs:
            return None
        if len(obs) == 1:
            return obs[-1].elapsed_ms / 1e3
        if len(obs) < self._min_samples_for_regression:
            return self._ewma(obs) / 1e3
        line = self._regress(obs)
        if line is not None:
            return max(0.0, line(nbytes)) / 1e3
        return self._ewma(obs) / 1e3

    def _ewma(self, obs: list[ExecutionRecord]) -> float:
        value = obs[0].elapsed_ms
        for rec in obs[1:]:
            value = (self._ewma_alpha * rec.elapsed_ms
                     + (1 - self._ewma_alpha) * value)
        return value

    def _regress(self, obs: list[ExecutionRecord]):
        """Least-squares ``t = a + b*n`` over recent observations."""
        pts = [(r.nbytes, r.elapsed_ms) for r in obs]
        n = len(pts)
        sx = sum(p[0] for p in pts)
        sy = sum(p[1] for p in pts)
        sxx = sum(p[0] * p[0] for p in pts)
        sxy = sum(p[0] * p[1] for p in pts)
        det = n * sxx - sx * sx
        if det == 0:
            return None
        b = (n * sxy - sx * sy) / det
        a = (sy - b * sx) / n
        if b < 0:
            # Non-monotone history: fall back to a constant at the worst
            # observation rather than predicting that bigger is faster.
            worst = max(p[1] for p in pts)
            return lambda _n: worst
        return lambda size: a + b * size

    def render(self) -> str:
        if not self._records:
            return "No execution history on this machine."
        ok = [r for r in self._records if r.success]
        failed = len(self._records) - len(ok)
        lines = [f"Execution history: {len(ok)} successful, {failed} failed"]
        for engine in sorted({r.engine for r in self._records}):
            rs = [r for r in ok if r.engine == engine]
            if rs:
                total = sum(r.elapsed_ms for r in rs)
                lines.append(f"  {engine:<14} {len(rs):>4} runs, "
                             f"{total:>9.1f} ms total")
        return "\n".join(lines)


_DEFAULT_MODEL: CostModel | None = None


def default_cost_model(refresh: bool = False) -> CostModel:
    """A process-wide cost model wired to this machine's calibration."""
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None or refresh:
        from ..hardware import HardwareProfile

        profile = HardwareProfile()
        store = CalibrationStore.load(fingerprint=profile.fingerprint())
        _DEFAULT_MODEL = CostModel(calibration=store)
    return _DEFAULT_MODEL

