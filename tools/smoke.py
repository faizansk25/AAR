"""Dev utility: end-to-end smoke run of everything built so far.

Prints real output from the type system, the IR, hardware detection and live
microbenchmark calibration on this machine. This is the manual counterpart to
the test suite: if this prints sane values, the layers are genuinely working.
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, "src")

# Unbuffered so progress survives a redirected or piped run, and so a killed
# run still shows how far it got.
sys.stdout.reconfigure(line_buffering=True)

from aar import types as T            # noqa: E402
from aar.failures import (DegradationLedger, FailureKind,   # noqa: E402
                          FailureRegistry, Severity)
from aar.hardware import HardwareProfile               # noqa: E402
from aar.hardware.calibrate import calibrate          # noqa: E402
from aar.ir import (BinOp, Col, JoinType, Lit, Node,   # noqa: E402
                    NodeType, ScanSpec, topological_order)



def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    rule("1. CANONICAL TYPE SYSTEM")
    print("postgres int4        ->", T.from_source("int4", "postgresql"))
    print("postgres timestamptz ->", T.from_source("timestamptz", "postgresql"))
    print("mongo date           ->", T.from_source("date", "mongodb"))
    print("excel number         ->", T.from_source("number", "excel"))
    print("int8 -> int32        ->", T.lossy(T.INT8, T.INT32) or "lossless")
    print("int64 -> int32       ->", T.lossy(T.INT64, T.INT32))
    print("Timestamp(ns)->us    ->", T.lossy(T.TIMESTAMP("ns"), T.TIMESTAMP("us")))
    log = T.ConversionLog()
    log.record("amount", "excel", "number", T.INT32)
    print("lossy conversions    ->", len(log.lossy_entries), "of", len(log))

    rule("2. INTERNAL IR")
    scan = Node(NodeType.SCAN_PARQUET, scan=ScanSpec(kind="parquet", path="orders.parquet"),
                output_schema=T.Schema((T.Field("id", T.INT64), T.Field("amt", T.FLOAT64))))
    filt = Node(NodeType.FILTER, inputs=[scan],
                predicate=BinOp(Col("amt"), ">", Lit(100.0)))
    grp = Node(NodeType.GROUPBY, inputs=[filt], key_left=("id",))
    out = Node(NodeType.WRITE, inputs=[grp], target="summary.parquet")
    for n in topological_order(out):
        print(f"  {n.type.value:12s} {n.describe()}")

    rule("3. FAILURE REGISTRY")
    print(FailureRegistry.render_matrix())
    ledger = DegradationLedger()
    ledger.record(FailureKind.GPU_UNAVAILABLE, "join", "no CUDA device",
                  from_engine="polars_gpu", to_engine="polars_cpu")
    print(ledger.render())

    rule("4. HARDWARE PROFILE (live)")
    t0 = time.perf_counter()
    profile = HardwareProfile()
    print(profile.render())
    print(f"\n  full probe took {(time.perf_counter() - t0) * 1000:.0f} ms")

    rule("5. LIVE MICROBENCHMARK CALIBRATION")
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        store = calibrate(quick=True, include_gpu=True, include_disk=True,
                          path=f"{tmp}/hardware-profile.json",
                          fingerprint=profile.fingerprint(),
                          progress=lambda m: print(f"  > {m}"))
        print(f"\n  calibration took {time.perf_counter() - t0:.1f} s")
        print()
        print(store.render())
        rule("6. CROSSOVER: which device wins, by size?")
        for op in ("groupby", "filter", "sort", "hash_join"):
            row = []
            for size in (1_000_000, 10_000_000, 100_000_000, 1_000_000_000):
                dev, cost = store.best_device(op, size)
                row.append(f"{size // 1_000_000}MB->{dev}({cost * 1e3:.1f}ms)")
            print(f"  {op:11s} " + "  ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
