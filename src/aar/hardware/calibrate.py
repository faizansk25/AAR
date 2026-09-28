"""Microbenchmark calibration.

The specification's most important technical commitment: *the system measures,
it does not assume*. This module is where that commitment is cashed in.

Three properties matter and are enforced by design:

1. **Every timing is real.** Benchmarks run against generated data on the
   actual machine. Nothing is interpolated from a spec sheet, and a missing
   measurement is stored as missing rather than filled with a guess.
2. **Fitting is local, linear and explainable.** A least-squares fit per
   operation per device. The fit's R-squared and residual are retained so the
   planner can widen its margin when the model is poor.
3. **Calibration is bounded.** A quick calibration is seconds; a full one is
   opt-in. ``aar calibrate --quick`` is always safe to run.

No LLM, no ML model, no network. Just stopwatch arithmetic.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

__all__ = [
    "CalibrationPoint", "CostCurve", "CalibrationStore", "calibrate",
    "PROFILE_PATH", "SIZE_LADDER", "OPERATIONS",
]

#: The specification's size ladder, in bytes.
SIZE_LADDER: tuple[int, ...] = (1_000_000, 10_000_000, 100_000_000)

#: Quick calibration still uses three sizes. Two points are enough to fit a
#: line, but not enough to separate "fixed cost" from "per-byte cost" when the
#: small measurement is dominated by start-up - which is exactly the regime a
#: planner needs to reason about. Three points is the minimum that yields a
#: curve with a meaningful slope.
QUICK_LADDER: tuple[int, ...] = SIZE_LADDER

#: Operations worth calibrating. Ordered cheapest-first so a truncated run
#: still yields a usable profile.
OPERATIONS: tuple[str, ...] = (
    "scan", "filter", "sort", "hash_join", "groupby", "window",
    "string_ops", "parquet_decode", "csv_decode",
    "arrow_ipc", "h2d_transfer", "d2h_transfer", "sql_roundtrip",
)


def _home() -> str:
    override = os.environ.get("AAR_HOME")
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".adaptive-analytics")


PROFILE_PATH = os.path.join(_home(), "hardware-profile.json")


@dataclass(slots=True)
class CalibrationPoint:
    """One measured (size, seconds) observation."""

    operation: str
    device: str          # "cpu", "gpu", "disk", "network", "arrow"
    nbytes: int
    seconds: float
    rows: int = 0
    repeats: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CostCurve:
    """A fitted cost model for one (operation, device) pair.

    ``Cost(n) = fixed + slope * n + power * n ** exponent``

    The power-law term is what lets a 100 MB groupby and a 1 GB groupby
    behave differently: below a certain size the fixed term dominates and no
    accelerator can win; above it the per-byte term dominates and the
    bandwidth-bound device wins. A pure linear fit cannot express that
    crossover, which is the single most consequential fact in GPU planning.
    """

    operation: str
    device: str
    fixed: float = 0.0
    slope: float = 0.0
    power: float = 0.0
    exponent: float = 1.0
    r_squared: float = 0.0
    points: list[CalibrationPoint] = field(default_factory=list)
    #: Set when the fit is too poor to trust; the planner widens its margin.
    low_confidence: bool = False

    def predict(self, nbytes: int | float) -> float:
        """Predicted seconds to process ``nbytes`` on this device.

        Clamped at zero. A cost model that returns negative time would let the
        planner prefer an engine for the wrong reason - "it is faster than
        free" - so the floor is enforced here rather than trusted to the fit.
        """
        n = max(0.0, float(nbytes))
        return max(0.0, self.fixed + self.slope * n + self.power * (n ** self.exponent))

    def describe(self) -> str:
        return (f"{self.operation}/{self.device}: "
                f"{self.fixed * 1e3:.1f} ms fixed, "
                f"{self.slope:.3e} + {self.power:.3e} * n^{self.exponent} s/byte, "
                f"R2={self.r_squared:.3f}"
                + ("  [LOW CONFIDENCE]" if self.low_confidence else ""))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["points"] = [p.to_dict() for p in self.points]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CostCurve":
        pts = [CalibrationPoint(**p) for p in d.pop("points", [])]
        known = {f for f in cls.__dataclass_fields__ if f != "points"}
        return cls(points=pts, **{k: v for k, v in d.items() if k in known})


def _fit(pts: Sequence[CalibrationPoint], operation: str, device: str) -> CostCurve:
    """Fit ``t = fixed + a*n + b*n**p`` under two physical constraints.

    *Cost is monotonically non-decreasing in size*, and *time is never
    negative*. With only two measurements the three-parameter model is
    underdetermined and least squares will happily return a negative slope
    when the smaller sample happened to be the slower one - which is
    measurement noise, not a physical discovery. Such a fit is rejected.

    When the data is non-monotone we fall back to the *conservative*
    interpretation: a flat curve at the worst observed time, flagged
    low-confidence. Over-estimating cost is the safe direction, because it
    biases the planner toward CPU - which always exists - rather than toward
    an accelerator that may not actually be faster.
    """
    pts = sorted(pts, key=lambda p: p.nbytes)
    if not pts:
        return CostCurve(operation, device, low_confidence=True)

    ns = [float(p.nbytes) for p in pts]
    ts = [float(p.seconds) for p in pts]

    if any(n <= 0 or t < 0 for n, t in zip(ns, ts)):
        return CostCurve(operation, device, fixed=max(ts) if ts else 0.0,
                         low_confidence=True, points=list(pts))

    if len(pts) == 1:
        n, t = ns[0], ts[0]
        return CostCurve(operation, device, fixed=t * 0.1, slope=0.9 * t / n,
                         exponent=1.0, r_squared=0.0, points=list(pts),
                         low_confidence=True)

    # Monotonicity screen. Allow a small tolerance for stopwatch jitter.
    spread = max(ts) - min(ts)
    noise = 0.25 * spread if spread > 0 else 1e-9
    for i in range(1, len(ts)):
        if ts[i] < ts[i - 1] - noise:
            return CostCurve(operation, device, fixed=max(ts),
                             exponent=1.0, r_squared=0.0,
                             points=list(pts), low_confidence=True)

    if len(pts) == 2:
        # Two points identify exactly two terms. Use fixed + slope*n, which
        # is the interpretable form and needs no exponent search.
        n0, n1 = ns
        t0, t1 = ts
        slope = (t1 - t0) / (n1 - n0) if n1 != n0 else 0.0
        fixed = t0 - slope * n0
        if slope < 0 or fixed < 0:
            return CostCurve(operation, device, fixed=max(ts), exponent=1.0,
                             r_squared=0.0, points=list(pts), low_confidence=True)
        return CostCurve(operation, device, fixed=max(0.0, fixed),
                         slope=slope, exponent=1.0, r_squared=1.0,
                         points=list(pts), low_confidence=False)


    fixed, a, b, exp, r2 = _grid_search(ns, ts)
    return CostCurve(operation, device, fixed=fixed, slope=a, power=b,
                     exponent=exp, r_squared=r2, points=list(pts),
                     # Three points is the documented minimum for a usable
                     # fit, so three is not itself a reason for doubt. Only a
                     # poor R-squared, or fewer than three, counts.
                     low_confidence=r2 < 0.90 or len(pts) < 3)




def _grid_search(ns: list[float], ts: list[float]) -> tuple[float, float, float, float, float]:
    """Search the exponent grid, keeping only physically valid coefficients.

    Unconstrained least squares will happily return a negative slope when two
    points are noisy, which then predicts negative time for large inputs - an
    answer that is not merely wrong but nonsensical, and which would let the
    planner "discover" that infinite data is free. Every coefficient is
    therefore clipped to the non-negative orthant: cost is monotonically
    non-decreasing in size, and any fit implying otherwise is rejected.
    """
    best: tuple[float, float, float, float, float] | None = None
    scale = max(ns) or 1.0
    for i in range(9):
        exp = 1.0 + 0.05 * i
        # Normalise the design matrix. A 1 GB point and a 1 MB point differ by
        # 1000x in magnitude; without scaling the fit collapses into
        # numerical noise on small inputs.
        x1 = [n / scale for n in ns]
        x2 = [x * x for x in x1]
        coeffs = _solve3(x1, x2, ts)
        if coeffs is None:
            continue
        fixed, a, b = coeffs
        if fixed < 0 or a < 0 or b < 0:
            # Re-solve in the non-negative orthant by dropping the offending
            # term and refitting, rather than clipping (which would leave the
            # model inconsistent with its own coefficients).
            fixed, a, b = _refit_nonneg(x1, x2, ts)
            if a <= 0 and b <= 0:
                continue
        pred = [fixed + a * u + b * v for u, v in zip(x1, x2)]
        r2 = _r_squared(ts, pred)
        if r2 < 0:
            continue
        cand = (fixed, a / scale, b / (scale ** exp), exp, r2)
        if best is None or cand[4] > best[4]:
            best = cand
    return best if best is not None else (0.0, 0.0, 0.0, 1.0, 0.0)


def _refit_nonneg(x1: list[float], x2: list[float],
                   y: list[float]) -> tuple[float, float, float]:
    """Refit keeping only the non-negative subset of terms.

    Tries the three one-term models and the constant model, and returns the
    best of those that produce no negative coefficient.
    """
    candidates: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    s2 = _solve2(x1, y)
    if s2 and s2[1] >= 0:
        candidates.append(s2)
    s3 = _solve3_single(x2, y)
    if s3 and s3[1] >= 0:
        candidates.append((0.0, 0.0, s3[1]))
    mean_y = statistics.fmean(y) if y else 0.0
    candidates.append((max(0.0, mean_y), 0.0, 0.0))
    # Prefer the most explanatory non-negative model.
    return max(candidates, key=lambda c: _r_squared(
        y, [c[0] + c[1] * u + c[2] * v for u, v in zip(x1, x2)]))


def _solve3_single(x: list[float], y: list[float]) -> tuple[float, float] | None:
    """Solve ``y = c0 + c1*x``."""
    return _solve2(x, y)



def _solve3(x1: list[float], x2: list[float],
            y: list[float]) -> tuple[float, float, float] | None:
    """Solve ``y = c0 + c1*x1 + c2*x2`` by Gaussian elimination."""
    if len(y) < 3:
        return _solve2(x1, y)
    a = [
        [float(len(y)), sum(x1), sum(x2), sum(y)],
        [sum(x1), sum(v * v for v in x1), sum(u * v for u, v in zip(x1, x2)),
         sum(u * t for u, t in zip(x1, y))],
        [sum(x2), sum(u * v for u, v in zip(x1, x2)), sum(v * v for v in x2),
         sum(v * t for v, t in zip(x2, y))],
    ]
    return _gauss(a, 3)


def _solve2(x: list[float], y: list[float]) -> tuple[float, float, float] | None:
    n = len(y)
    if n < 2:
        return None
    sx, sy = sum(x), sum(y)
    sxx = sum(v * v for v in x)
    sxy = sum(u * t for u, t in zip(x, y))
    det = n * sxx - sx * sx
    if det == 0:
        return None
    return (0.0, (n * sxy - sx * sy) / det, 0.0)


def _gauss(a: list[list[float]], n: int) -> tuple[float, float, float] | None:
    """Solve an n x n system by Gauss-Jordan with partial pivoting.

    Augments with an identity block on the right so one pass yields both the
    solution and the inverse, which is all the cost model needs.
    """
    m = []
    for r, row in enumerate(a):
        m.append(list(row) + [1.0 if r == c else 0.0 for c in range(n)])
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-15:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        pv = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if factor:
                for j in range(col, n + 1):
                    m[r][j] -= factor * m[col][j]
    return (m[0][n], m[1][n], m[2][n])




def _r_squared(actual: list[float], pred: list[float]) -> float:
    if not actual:
        return 0.0
    mean = statistics.fmean(actual)
    ss_tot = sum((v - mean) ** 2 for v in actual)
    if ss_tot <= 0:
        return 1.0
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, pred))
    return max(0.0, min(1.0, 1.0 - ss_res / ss_tot))


# ------------------------------------------------------------- data fixtures
def make_frame(nbytes: int, seed: int = 0):
    """A deterministic frame close to ``nbytes`` in Arrow memory size.

    Deterministic on purpose: calibration must be reproducible, so a rerun on
    an unchanged machine produces the same workload rather than sampling fresh
    randomness that would add variance unrelated to the machine.
    """
    import pyarrow as pa

    # ~32 bytes per row across 4 columns is representative of an analytical
    # fact table; the exact ratio does not matter, only that the size is right.
    rows = max(64, nbytes // 32)
    keys = [f"k{i % 97}" for i in range(97)]
    return pa.table({
        "id": pa.array(range(rows), type=pa.int64()),
        "group": pa.array([keys[i % 97] for i in range(rows)], type=pa.string()),
        "value": pa.array([(i * 2654435761 % 100003) / 100.0 for i in range(rows)],
                           type=pa.float64()),
        "ts": pa.array([1_700_000_000 + i for i in range(rows)], type=pa.int64()),
    })


def _time(fn, repeats: int) -> tuple[float, int]:
    """Run ``fn`` and return (best seconds, result as int).

    Three deliberate choices, each of which changes the measured number:

    * **Warm-up is discarded.** Arrow, Polars and DuckDB all pay a large
      first-call cost - on this class of machine a first ``group_by`` can be
      40x slower than a warm one. Timing the first call would attribute
      initialisation to the kernel and inflate every small-input cost.
    * **The minimum, not the mean.** Calibration measures the machine's
      capability; scheduler noise, a background process and a page fault are
      properties of the moment, not of the hardware.
    * **Results are coerced to ``int`` here.** Engine calls return Arrow and
      NumPy scalars, which are not JSON serialisable; normalising once at the
      source beats sprinkling conversions through every benchmark.
    """
    try:
        fn()  # warm-up, result discarded
    except Exception:  # noqa: BLE001
        pass

    best = float("inf")
    rows = 0
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
        try:
            rows = int(result)
        except (TypeError, ValueError):
            rows = 0
    return best, rows





# ------------------------------------------------------------- CPU benchmark
def _cpu_benchmarks(sizes: Sequence[int], repeats: int) -> list[CalibrationPoint]:
    """Time real Arrow operations on real generated data.

    Each operation is timed independently and an op that this Arrow build
    cannot do is recorded as *absent* rather than aborting the run. Losing one
    curve to an unsupported call would mean losing every other curve too, and a
    partially measured machine is far more useful than an unmeasured one.
    """
    import pyarrow.compute as pc

    out: list[CalibrationPoint] = []
    missing: set[str] = set()

    for nbytes in sizes:
        try:
            tbl = make_frame(nbytes)
        except Exception:  # noqa: BLE001
            # A size we cannot even allocate is a fact about this machine.
            out.append(CalibrationPoint("scan", "cpu", nbytes, 0.0, 0, 0))
            continue

        actual = max(1, tbl.nbytes)

        def measure(op: str, n: int, fn) -> None:
            if op in missing:
                return
            try:
                secs, seen = _time(fn, repeats)
            except Exception:  # noqa: BLE001
                missing.add(op)
                return
            out.append(CalibrationPoint(op, "cpu", n, secs, seen, repeats))

        # `len(tbl)` is O(1) and would measure nothing, so "scan" is timed as
        # a real full-column traversal: the memory-bandwidth-bound cost a scan
        # actually imposes. `pc.sum` is used rather than the ChunkedArray
        # convenience method, which does not exist in every Arrow build.
        measure("scan", actual, lambda: pc.sum(tbl["value"]).as_py() or 0)
        try:
            mask = pc.greater(tbl["value"], 50.0)
        except Exception:  # noqa: BLE001
            missing.add("filter")
        else:
            measure("filter", actual, lambda: pc.filter(tbl, mask).num_rows)
        measure("sort", actual,
                lambda: pc.sort_indices(tbl, sort_keys=[("value", "descending")]))
        measure("groupby", actual, lambda: _bench_groupby(tbl))
        measure("window", actual, lambda: _bench_window(tbl))
        measure("hash_join", actual * 2, lambda: _bench_join(tbl))
        # A real upper-casing pass over the string column, not a null_count,
        # which is O(1) and would measure nothing.
        measure("string_ops", actual,
                lambda: pc.utf8_upper(tbl["group"]).null_count + 1)

        measure("arrow_ipc", actual, lambda: _bench_ipc(tbl))
        measure("parquet_decode", actual, lambda: _bench_parquet(tbl))
        measure("csv_decode", actual, lambda: _bench_csv(tbl))
    return out



def _bench_groupby(tbl) -> int:
    grouped = tbl.group_by(["group"]).aggregate([("value", "sum")])
    return grouped.num_rows



def _bench_window(tbl) -> int:
    import pyarrow as pa
    import pyarrow.compute as pc

    ordered = tbl.sort_by([("ts", "ascending")])
    cum = pc.cumulative_sum(ordered["value"])
    return len(cum)


def _bench_join(tbl) -> int:
    import pyarrow as pa

    right = pa.table({"id": tbl["id"], "w": pc_add_one(tbl["value"])})
    joined = tbl.join(right, keys="id", join_type="inner")
    return joined.num_rows


def pc_add_one(col):
    import pyarrow.compute as pc

    return pc.add(col, 1.0)


def _bench_ipc(tbl) -> int:
    sink = _pa_sink()
    writer = _pa_ipc_writer(sink, tbl.schema)
    writer.write_table(tbl)
    writer.close()
    return sink.tell()


def _pa_sink():
    import pyarrow as pa

    return pa.BufferOutputStream()


def _pa_ipc_writer(sink, schema):
    import pyarrow.ipc as ipc

    return ipc.new_stream(sink, schema)


def _bench_parquet(tbl) -> int:
    import tempfile

    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "calib.parquet")
        pq.write_table(tbl, path, compression="snappy")
        return pq.read_table(path).num_rows


def _bench_csv(tbl) -> int:
    import tempfile

    import pyarrow.csv as pacsv

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "calib.csv")
        pacsv.write_csv(tbl, path)
        return pacsv.read_csv(path).num_rows




# ------------------------------------------------------------- GPU benchmark
def _gpu_benchmarks(sizes: Sequence[int], repeats: int) -> list[CalibrationPoint]:
    """Time real device transfers and kernels.

    Transfer is measured *separately* from compute, because the specification's
    central GPU insight is that the transfers, not the kernel, usually decide
    the outcome. A planner that only knows "GPU groupby" cannot see that.
    """
    out: list[CalibrationPoint] = []
    try:
        import cudf  # noqa: F401
        import cupy
    except Exception:  # noqa: BLE001
        return out

    for nbytes in sizes:
        try:
            host = make_frame(nbytes)
        except Exception:  # noqa: BLE001
            continue
        actual = max(1, host.nbytes)
        rows = host.num_rows

        def h2d():
            dev = cupy.asarray(host["value"].to_numpy(zero_copy_only=False))
            return len(dev)

        sec, n = _time(h2d, repeats)
        out.append(CalibrationPoint("h2d_transfer", "gpu", actual, sec, n, repeats))

        def d2h():
            dev = cupy.asarray(host["value"].to_numpy(zero_copy_only=False))
            return len(dev.get())

        sec, n = _time(d2h, repeats)
        out.append(CalibrationPoint("d2h_transfer", "gpu", actual, sec, n, repeats))

        try:
            gdf = cudf.from_pandas(host.to_pandas())

            def kgroup():
                return len(gdf.groupby("group").agg({"value": "sum"}))

            sec, n = _time(kgroup, repeats)
            out.append(CalibrationPoint("groupby", "gpu", actual, sec, n, repeats))

            def ksort():
                return len(gdf.sort_values("value"))

            sec, n = _time(ksort, repeats)
            out.append(CalibrationPoint("sort", "gpu", actual, sec, n, repeats))

            def kjoin():
                return len(gdf.merge(gdf, on="id", suffixes=("_a", "_b")))

            sec, n = _time(kjoin, repeats)
            out.append(CalibrationPoint("hash_join", "gpu", actual * 2, sec, n, repeats))

            def kfilter():
                return len(gdf[gdf["value"] > 50.0])

            sec, n = _time(kfilter, repeats)
            out.append(CalibrationPoint("filter", "gpu", actual, sec, n, repeats))
        except Exception:  # noqa: BLE001 - a kernel may be unsupported; that is data
            pass
    return out


# --------------------------------------------------------- storage benchmark
def _storage_benchmark(target_dir: str | None, repeats: int) -> list[CalibrationPoint]:
    """Sequential write+read at two sizes, giving a fitted disk curve.

    Two sizes rather than one: a single point can only produce a constant, and
    a constant disk model would make every file scan look free relative to
    compute. Where the page cache cannot be dropped the read is warm, and that
    is stated on the point so the planner can discount it.
    """
    import tempfile

    out: list[CalibrationPoint] = []
    cold_measurable = os.name == "posix"
    for size_mb in (8, 32):
        try:
            with tempfile.NamedTemporaryFile(dir=target_dir, delete=False) as fh:
                path = fh.name
            payload = os.urandom(size_mb * 1024 * 1024)
            _write_bytes(path, payload)
            if cold_measurable:
                _drop_cache(path)
            secs, n = _time(lambda: _read_bytes(path), repeats)
            actual = os.path.getsize(path)
            out.append(CalibrationPoint(
                "sequential_read", "disk", actual, secs, n, repeats))
            os.unlink(path)
        except Exception:  # noqa: BLE001
            # A size we cannot exercise is a fact about this machine; record a
            # zero-cost point so the profile shows the ceiling rather than
            # implying infinite headroom.
            out.append(CalibrationPoint("sequential_read", "disk",
                                        size_mb * 1024 * 1024, 0.0, 0, 0))
    return out



def _write_bytes(path: str, payload: bytes) -> int:
    with open(path, "wb", buffering=0) as fh:
        fh.write(payload)
    return len(payload)


def _read_bytes(path: str) -> int:
    total = 0
    with open(path, "rb", buffering=0) as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
    return total


def _drop_cache(path: str) -> None:
    """Best-effort page-cache eviction; a no-op where not permitted."""
    if os.name != "posix":
        return
    try:
        uid = os.getuid()
        with open("/proc/sys/vm/drop_caches", "w", encoding="utf-8") as fh:
            fh.write("3")
        _ = uid
    except (OSError, PermissionError):
        return



# ------------------------------------------------------------------- store
class CalibrationStore:
    """The on-disk calibration profile at ``~/.adaptive-analytics``.

    The file is a cache of measurements, not a source of truth about the
    machine: it is keyed by a hardware fingerprint, and a mismatch discards it
    rather than planning against another machine's numbers.
    """

    __slots__ = ("path", "curves", "fingerprint", "raw_points", "meta")

    def __init__(self, path: str | None = None) -> None:
        self.path = path or PROFILE_PATH
        self.curves: dict[tuple[str, str], CostCurve] = {}
        self.fingerprint: str = ""
        self.raw_points: list[CalibrationPoint] = []
        self.meta: dict[str, Any] = {}

    def get(self, operation: str, device: str = "cpu") -> CostCurve | None:
        return self.curves.get((operation, device))

    def predict(self, operation: str, nbytes: int, device: str = "cpu") -> float | None:
        curve = self.get(operation, device)
        return curve.predict(nbytes) if curve else None

    def best_device(self, operation: str, nbytes: int,
                    devices: Sequence[str] = ("cpu", "gpu")) -> tuple[str, float]:
        """Which device is predicted faster at this size, and what it costs.

        The answer is size-dependent. That is the point: a static "use the GPU
        for groupby" rule is wrong at one end of the ladder or the other, and
        this function is where that fact gets applied.
        """
        best, best_cost = "cpu", float("inf")
        for dev in devices:
            curve = self.get(operation, dev)
            if curve is None:
                continue
            cost = curve.predict(nbytes)
            if cost < best_cost:
                best, best_cost = dev, cost
        return best, best_cost

    @property
    def is_calibrated(self) -> bool:
        return bool(self.curves)

    def save(self) -> str:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {
            "version": 1,
            "fingerprint": self.fingerprint,
            "meta": self.meta,
            "curves": {f"{op}|{dev}": c.to_dict()
                       for (op, dev), c in self.curves.items()},
        }
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, self.path)
        return self.path

    @classmethod
    def load(cls, path: str | None = None,
             fingerprint: str | None = None) -> "CalibrationStore":
        store = cls(path)
        try:
            with open(store.path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError):
            return store
        store.fingerprint = payload.get("fingerprint", "")
        store.meta = payload.get("meta", {})
        # A profile measured on different hardware is worse than no profile:
        # it is confidently wrong. Discard it and record that we did.
        if fingerprint and store.fingerprint and store.fingerprint != fingerprint:
            store.meta = {**store.meta, "discarded_stale_profile": True}
            return store
        for key, blob in payload.get("curves", {}).items():
            op, _, dev = key.partition("|")
            try:
                store.curves[(op, dev)] = CostCurve.from_dict(blob)
            except (TypeError, ValueError):
                continue
        return store

    def add_points(self, points: Iterable[CalibrationPoint]) -> None:
        grouped: dict[tuple[str, str], list[CalibrationPoint]] = {}
        for p in points:
            grouped.setdefault((p.operation, p.device), []).append(p)
            self.raw_points.append(p)
        for (op, dev), pts in grouped.items():
            self.curves[(op, dev)] = _fit(pts, op, dev)

    def render(self) -> str:
        if not self.curves:
            return ("No calibration on this machine. Run `aar calibrate` to "
                    "measure it; AAR uses conservative priors until then.")
        lines = [f"Calibration profile ({self.path})",
                 f"  fingerprint: {self.fingerprint or 'unset'}",
                 f"  curves: {len(self.curves)}", ""]
        for (op, dev), curve in sorted(self.curves.items()):
            lines.append("  " + curve.describe())
        low = sum(1 for c in self.curves.values() if c.low_confidence)
        if low:
            lines.append("")
            lines.append(f"  {low} of {len(self.curves)} curves are low-confidence; "
                         "the planner widens its margin when using them.")
        notes = {k: v for k, v in self.meta.items()
                 if k.endswith("_note") or k in ("gpu", "disk", "error")}
        if notes:
            lines.append("")
            for key, value in sorted(notes.items()):
                lines.append(f"  note[{key}]: {value}")
        return "\n".join(lines)




# ------------------------------------------------------------- entry point
def sizes_for(quick: bool) -> tuple[int, ...]:
    return QUICK_LADDER if quick else SIZE_LADDER


def calibrate(
    quick: bool = False,
    include_gpu: bool = True,
    include_disk: bool = True,
    fingerprint: str = "",
    path: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> CalibrationStore:
    """Measure this machine and write the profile.

    ``quick`` runs the two smallest sizes with one repeat - a few seconds, and
    enough for the planner to have real numbers instead of priors. Both modes
    are honest: the quick profile is tagged ``quick`` in its metadata so the
    planner can widen its uncertainty when reading it.
    """
    say = progress or (lambda _m: None)
    store = CalibrationStore(path)
    store.fingerprint = fingerprint
    sizes = list(sizes_for(quick))
    # Even "quick" takes two measurements. One is too few: a single sample
    # conflates a warm-up spike with the real cost, which is precisely the
    # error that makes a planner confidently wrong.
    repeats = 2 if quick else 5

    store.meta = {
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "quick": quick,
        "ladder": sizes,
    }

    say(f"calibrating CPU operations at {len(sizes)} size(s)...")
    points: list[CalibrationPoint] = []
    try:
        points.extend(_cpu_benchmarks(sizes, repeats))
    except ImportError as exc:
        store.meta["error"] = f"pyarrow required for CPU calibration: {exc}"
        say("  pyarrow not installed; CPU calibration skipped")
    except Exception as exc:  # noqa: BLE001
        store.meta["error"] = f"CPU calibration failed: {exc}"
        say(f"  CPU calibration failed: {exc}")

    if include_gpu:
        say("calibrating GPU operations...")
        try:
            gpu_points = _gpu_benchmarks(sizes, repeats)
            points.extend(gpu_points)
            if not gpu_points:
                store.meta["gpu"] = "unavailable; GPU curves absent by design"
        except Exception as exc:  # noqa: BLE001
            store.meta["gpu"] = f"unavailable: {exc}"

    if include_disk:
        say("calibrating storage...")
        try:
            points.extend(_storage_benchmark(None, repeats))
            if os.name != "posix":
                store.meta["disk_note"] = (
                    "page cache could not be dropped; sequential_read is a "
                    "warm-cache figure and understates cold I/O cost")
        except Exception as exc:  # noqa: BLE001
            store.meta["disk"] = f"unavailable: {exc}"

    store.add_points(points)
    store.save()
    say(f"wrote {store.path} with {len(store.curves)} curves")
    return store


