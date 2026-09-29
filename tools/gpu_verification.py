"""Verify the AAR GPU path on a real GPU (designed for Colab T4).

Run this on a machine that actually has a CUDA device. It answers the
question AAR could not answer for itself on a CPU-only host: does the GPU
code path work, and would the planner have chosen the GPU?

The result is written as JSON to be copied back into this repository. That
is deliberate - a GPU measurement that lives only in a browser session is a
rumour, not evidence.

Honest notes on what a T4 can and cannot establish:

* A Tesla T4 is compute capability 7.5: **no bfloat16**, and FP64 at 1/64
  of FP32. Work that wins on a T4 may lose on an A100. This establishes
  that the path *works*, not that a kernel is a good idea everywhere.
* Colab sessions are ephemeral; mount Drive if results must survive.
* RAPIDS installs on Colab are version-sensitive. If `cudf` will not
  import, this still runs and reports the GPU path as unavailable rather
  than crashing - AAR's own design principle, applied to itself.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time

RESULT_PATH = os.environ.get("AAR_GPU_RESULT", "aar_gpu_result.json")
ROWS = int(os.environ.get("AAR_GPU_ROWS", "2000000"))


def section(title: str) -> None:
    print("\n" + "=" * 66)
    print(f"  {title}")
    print("=" * 66, flush=True)


def probe_machine(result: dict) -> None:
    section("1. What machine is this?")
    try:
        from aar.hardware.detect import probe_gpu

        gpu = probe_gpu()
        result["cuda"] = {
            "available": bool(gpu.available), "vendor": gpu.vendor,
            "model": gpu.model, "device_count": gpu.device_count,
            "vram_gb": round(gpu.vram_bytes / 2**30, 2) or None,
            "driver": gpu.driver_version, "cuda": gpu.cuda_version,
            "compute_capability": (".".join(map(str, gpu.compute_capability))
                                   if gpu.compute_capability else None),
            "uvm": gpu.uvm_available, "reason": gpu.reason,
        }
        print(json.dumps(result["cuda"], indent=2), flush=True)
        if not gpu.available:
            result["errors"].append(f"no usable GPU: {gpu.reason}")
            print(f"No usable GPU ({gpu.reason}). AAR still runs, on CPU - "
                  f"which is the graceful-degradation path working.",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"hardware probe failed: {exc}")
        print(f"hardware probe failed: {exc}", flush=True)


def probe_engines(result: dict) -> None:
    section("2. Which engines can AAR actually use here?")
    try:
        from aar.capability import default_registry

        registry = default_registry()
        for engine_id, cap in registry.probe().items():
            result["engines"][engine_id] = {
                "available": bool(cap.available), "reason": cap.reason,
                "version": cap.version,
            }
        print(registry.render(), flush=True)
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"registry failed: {exc}")


def install_gpu_stack(result: dict) -> None:
    section("3. Install the GPU stack (optional, but the point)")
    for module, package in (("cudf", "cudf-cu12"), ("pynvml", "pynvml")):
        try:
            __import__(module)
            print(f"{module}: already present", flush=True)
            continue
        except ImportError:
            print(f"{module}: installing {package}...", flush=True)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", package],
                capture_output=True, text=True, timeout=900)
            print(f"  pip exit={proc.returncode} "
                  f"{(proc.stderr or '')[-160:]}", flush=True)
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(f"pip {package} failed: {exc}")


def check_gpu_engines(result: dict) -> None:
    section("4. Can AAR actually construct a GPU engine here?")
    try:
        from aar.engines.factory import create_engine

        for engine_id in ("cudf", "polars_gpu"):
            entry = result["engines"].setdefault(engine_id, {})
            # allow_degradation=False is the whole point: "cudf" must mean
            # cudf here. Anything that quietly returns an Arrow engine would
            # file CPU timings under a GPU label, which is the exact lie this
            # script exists to prevent.
            try:
                engine = create_engine(engine_id, allow_degradation=False)
            except Exception as exc:  # noqa: BLE001
                entry["available"] = False
                entry["reason"] = str(exc)[:200]
                print(f"{engine_id}: UNAVAILABLE - {str(exc)[:90]}",
                      flush=True)
                continue
            entry["available"] = True
            entry["engine_class"] = type(engine).__name__
            # Which GPU collect spelling this Polars build accepts. The
            # first T4 run failed on `collect(engine="cudf")` being invalid
            # in Polars 1.35, so the answer is recorded rather than assumed.
            api = getattr(engine, "collect_api", None)
            if api:
                entry["collect_api"] = api
            print(f"{engine_id}: constructed OK "
                  f"({type(engine).__name__})"
                  + (f", collect via {api}" if api else ""), flush=True)
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"engine construction failed: {exc}")


def build_table():
    """Deterministic synthetic data, so engines are compared like for like."""
    import pyarrow as pa

    from aar.interchange import Table, arrow_to_canonical
    from aar.types import Field, Schema

    n = ROWS
    arrow = pa.table({
        "g": pa.array([i % 512 for i in range(n)], pa.int32()),
        "v": pa.array([float(i % 1000) for i in range(n)], pa.float64()),
    })
    schema = Schema(tuple(Field(f.name, arrow_to_canonical(f.type))
                          for f in arrow.schema))
    return Table(arrow, schema)


def benchmark(result: dict) -> None:
    section("5. The same work on every engine that can do it")
    try:
        from aar.engines.factory import create_engine
        from aar.ir import Agg, BinOp, Col, Lit
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"cannot import the engines: {exc}")
        return

    try:
        table = build_table()
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"could not build the test table: {exc}")
        return
    print(f"table: {table.num_rows:,} rows x {table.num_columns} columns",
          flush=True)

    # Agg is (func, arg, distinct, custom). The third positional is
    # `distinct`, so passing the output name there made every aggregate
    # arrive flagged DISTINCT - which is exactly what the first T4 run
    # reported for cudf, and it was this script's bug, not the engine's.
    aggs = {"total": Agg("SUM", Col("v")), "n": Agg("COUNT", Col("v"))}
    baseline = None
    for engine_id in ("arrow", "duckdb", "polars_cpu", "pandas",
                      "cudf", "polars_gpu"):
        entry: dict = {}
        # Same rule here as in section 4: a row filed under "cudf" must have
        # been produced by cudf. Anything less and the benchmark lies.
        try:
            engine = create_engine(engine_id, allow_degradation=False)
        except Exception as exc:  # noqa: BLE001
            result["benchmark"][engine_id] = {
                "skipped": "engine unavailable",
                "reason": f"{type(exc).__name__}: {exc}"[:200]}
            print(f"  {engine_id:11s} skipped - unavailable", flush=True)
            continue
        entry["engine_class"] = type(engine).__name__
        try:
            started = time.perf_counter()
            out = engine.group_by(table, ["g"], aggs)
            entry["group_by_ms"] = round(
                (time.perf_counter() - started) * 1000, 1)
            started = time.perf_counter()
            engine.filter(table, BinOp(Col("v"), ">", Lit(500.0)))
            entry["filter_ms"] = round(
                (time.perf_counter() - started) * 1000, 1)
            rows = sorted(out.arrow.to_pylist(), key=lambda r: r["g"])
            entry["groups"] = len(rows)
            entry["checksum"] = round(sum(r["total"] for r in rows), 3)
            if baseline is None:
                baseline = entry["checksum"]
            # Cross-engine agreement, not merely "it produced a number".
            entry["agrees"] = entry["checksum"] == baseline
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"[:200]
        result["benchmark"][engine_id] = entry
        print(f"  {engine_id:11s} {entry}", flush=True)


def decisions(result: dict) -> None:
    section("6. Would AAR have chosen the GPU?")
    try:
        from aar.capability import Device
        from aar.cost import default_cost_model
        from aar.ir import Node, NodeType
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"cost model import: {exc}")
        print(f"cost model import failed: {exc}", flush=True)
        return

    # The real API is `node_cost(node, engine_id, nbytes, residency=...)`,
    # and `residency` is the parameter that decides whether a transfer is
    # paid at all. The first T4 run called a `node_cost_for` that never
    # existed, so this section produced no rows at all.
    model = default_cost_model()
    # A sink keeps its result on the device, so nothing has to come back.
    node = Node(NodeType.GROUPBY, estimated_bytes=1_000_000_000)
    for nbytes in (10_000_000, 100_000_000, 1_000_000_000):
        node.estimated_bytes = nbytes
        try:
            cpu, cpu_src = model.node_cost(node, "duckdb", nbytes)
            gpu, gpu_src = model.node_cost(node, "cudf", nbytes)
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(f"cost model: {exc}")
            print(f"cost model: {exc}", flush=True)
            return
        row = {
            "bytes": nbytes,
            "cpu_ms": round(cpu.total_ms, 1),
            "gpu_ms": round(gpu.total_ms, 1),
            "cpu_kernel_ms": round(cpu.kernel_ms, 1),
            "gpu_kernel_ms": round(gpu.kernel_ms, 1),
            "cpu_transfer_ms": round(cpu.transfer_ms, 1),
            "gpu_transfer_ms": round(gpu.transfer_ms, 1),
            "cpu_source": cpu_src,
            "gpu_source": gpu_src,
            # A device-resident feed removes the inbound transfer, which is
            # the whole point of keeping data on the GPU across segments.
            "gpu_resident_ms": round(
                model.node_cost(node, "cudf", nbytes,
                                residency=Device.GPU)[0].total_ms, 1),
            "picked": "gpu" if gpu.total_ms < cpu.total_ms else "cpu",
        }
        result["decisions"].append(row)
        print(f"  {nbytes:>13,}B  cpu={row['cpu_ms']:>9.1f}ms "
              f"(kern {row['cpu_kernel_ms']:>8.1f} + xfer "
              f"{row['cpu_transfer_ms']:>8.1f})  "
              f"gpu={row['gpu_ms']:>9.1f}ms "
              f"(kern {row['gpu_kernel_ms']:>8.1f} + xfer "
              f"{row['gpu_transfer_ms']:>8.1f})  "
              f"gpu-resident={row['gpu_resident_ms']:>9.1f}ms  "
              f"-> {row['picked']}", flush=True)
    print("\n  Note: these are estimates from the cost model on THIS host, "
          "not measurements.\n  The measured numbers are in section 5.",
          flush=True)


def main() -> int:
    result: dict = {
        "python": platform.python_version(), "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rows": ROWS, "cuda": {}, "engines": {}, "benchmark": {},
        "decisions": [], "errors": [],
    }
    try:
        from aar import __version__

        result["aar_version"] = __version__
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"AAR not importable: {exc}")
        print("AAR is not importable. Either "
              "`pip install aar-analytics`", flush=True)
        print("or run this from a checkout with src/ on PYTHONPATH.",
              flush=True)
        with open(RESULT_PATH, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        return 1

    probe_machine(result)
    probe_engines(result)
    install_gpu_stack(result)
    check_gpu_engines(result)
    benchmark(result)
    decisions(result)

    section("7. Write the result for the repository")
    with open(RESULT_PATH, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)
    print(f"wrote {RESULT_PATH}", flush=True)
    print("Copy it to data/gpu/aar_gpu_result.json and the findings "
          "become a report entry rather than a claim.", flush=True)
    if result["errors"]:
        print("\ncompleted with notes:", flush=True)
        for err in result["errors"]:
            print(f"  - {err}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

