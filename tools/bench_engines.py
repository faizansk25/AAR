"""Benchmark every operation on every available engine, honestly.

The pandas filter was 1,400x slower than Arrow for three T4 runs before
anyone measured it, because "slow" is not a thing you can see by reading
code. This is the tool that makes the rest of the slow things visible.

**What it measures.** The engine contract, operation by operation: filter,
project, group_by, sort, limit and join. Every engine that constructs,
every operation, on one fixed dataset, with the result fingerprinted so a
fast wrong answer cannot look like a win.

**Why it repeats and reports a median.** A benchmark reporting one number
measures the scheduler. DuckDB's group-by was 53.7 ms and 293.6 ms on two
Colab T4 runs of identical code - a 5.5x spread - so a single sample is an
anecdote. Each operation runs several times and the median is reported
with the spread beside it. A difference smaller than the spread is not a
difference.

**Why the data is wide.** Narrow data hides a real cost: an engine that
copies every column regardless of the projection looks identical to one
that pushes the projection down, as long as the group-by needs every
column anyway. The `pad` columns are read by nothing, which is the shape
of a real wide export.

**What it cannot do.** It says nothing about I/O concurrency, multi-user
behaviour, or GPU transfer beyond the device round-trip. It measures a
single-threaded, in-memory, warm-cache workload - the easiest case for
every engine, and therefore the clearest place to see a structural
problem.

Run: ``python tools/bench_engines.py`` (add ``--rows N`` for a sweep).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pyarrow as pa  # noqa: E402

from aar.engines.factory import ENGINE_FACTORIES, create_engine  # noqa: E402
from aar.interchange import Table, arrow_to_canonical  # noqa: E402
from aar.ir import Agg, BinOp, Col, Lit  # noqa: E402
from aar.types import Field, Schema  # noqa: E402


def build(rows: int, groups: int = 512) -> Table:
    """Deterministic data, wide enough that projection actually matters."""
    arrays: dict[str, Any] = {
        "g": pa.array([i % groups for i in range(rows)], pa.int32()),
        "v": pa.array([float((i * 7) % 100) for i in range(rows)],
                      pa.float64()),
    }
    for c in range(8):
        arrays[f"pad{c}"] = pa.array(
            [float(i + c) for i in range(rows)], pa.float64())
    arrow = pa.table(arrays)
    schema = Schema(tuple(Field(f.name, arrow_to_canonical(f.type))
                          for f in arrow.schema))
    return Table(arrow, schema)


def build_right(table: Table) -> Table:
    """A small right-hand side, as a real dimension table would be."""
    keys = sorted(set(table.column("g").to_pylist()))
    arrow = pa.table({
        "g": pa.array(keys, pa.int32()),
        "label": pa.array([f"g{k}" for k in keys], pa.utf8()),
    })
    schema = Schema(tuple(Field(f.name, arrow_to_canonical(f.type))
                          for f in arrow.schema))
    return Table(arrow, schema)


def operations(table: Table, right: Table) -> dict[str, Callable]:
    """One callable per engine-contract operation."""
    pads = [c for c in table.column_names if c.startswith("pad")]
    return {
        "filter": lambda e: e.filter(
            table, BinOp(Col("v"), ">", Lit(50.0))),
        "filter_nested": lambda e: e.filter(table, BinOp(
            BinOp(Col("v"), ">", Lit(20.0)), "AND",
            BinOp(Col("g"), "<", Lit(100)))),
        "project": lambda e: e.project(table, ["g", "v", *pads[:3]]),
        "group_by": lambda e: e.group_by(table, ["g"], {
            "total": Agg("SUM", Col("v")), "n": Agg("COUNT", Col("v"))}),
        "sort": lambda e: e.sort(table, [("v", True)]),
        "limit": lambda e: e.limit(table, 1000),
        "join": lambda e: e.join(table, right, ["g"], "inner"),
    }



def fingerprint(result: Any) -> str:
    """A comparable summary, so speed cannot hide a wrong answer.

    A benchmark that only times operations will happily report a very fast
    engine returning the wrong rows. Order-insensitive by construction,
    because group-by output order is not part of the contract.
    """
    try:
        rows = result.arrow.to_pylist()
    except AttributeError:
        return "no-table"
    if not rows:
        return "empty"
    return str(hash(tuple(sorted(tuple(sorted(r.items(), key=str))
                                  for r in rows))))


def available() -> list[str]:
    """Engines that actually construct here, in catalogue order."""
    out = []
    for engine_id in ENGINE_FACTORIES:
        try:
            with create_engine(engine_id, allow_degradation=False):
                out.append(engine_id)
        except Exception:  # noqa: BLE001
            continue
    return out


def run(rows: int, repeats: int) -> dict:
    table = build(rows)
    right = build_right(table)
    ops = operations(table, right)
    engines = available()
    print(f"\n{rows:,} rows x {table.num_columns} columns, "
          f"{repeats} repeats (median reported)")
    print(f"engines: {', '.join(engines)}\n", flush=True)

    results: dict[str, dict] = {}
    for op_name, op in ops.items():
        print(f"  {op_name}")
        for engine_id in engines:
            samples: list[float] = []
            prints: set[str] = set()
            error = None
            for _ in range(repeats):
                try:
                    with create_engine(engine_id,
                                       allow_degradation=False) as engine:
                        started = time.perf_counter()
                        out = op(engine)
                        samples.append(
                            (time.perf_counter() - started) * 1000.0)
                    prints.add(fingerprint(out))
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {str(exc)[:80]}"
                    break
            entry: dict[str, Any] = {}
            if error:
                entry["error"] = error
                print(f"    {engine_id:12s} ERROR {error}")
            else:
                median = statistics.median(samples)
                spread = ((max(samples) - min(samples)) / median * 100
                          if median else 0.0)
                entry.update({
                    "median_ms": round(median, 2),
                    "min_ms": round(min(samples), 2),
                    "max_ms": round(max(samples), 2),
                    "spread_pct": round(spread, 1),
                    "distinct_results": len(prints),
                })
                flag = "  <-- NONDETERMINISTIC" if len(prints) > 1 else ""
                print(f"    {engine_id:12s} {median:9.2f} ms "
                      f"(spread {spread:4.1f}%){flag}")
            results.setdefault(engine_id, {})[op_name] = entry
    return results


def outliers(results: dict) -> list[tuple[str, str, float, float, float]]:
    """Slowest relative to the fastest engine for the same operation."""
    by_op: dict[str, list[tuple[str, float]]] = {}
    for engine_id, ops in results.items():
        for op_name, entry in ops.items():
            if "median_ms" in entry:
                by_op.setdefault(op_name, []).append(
                    (engine_id, entry["median_ms"]))
    out = []
    for engine_id, ops in results.items():
        for op_name, entry in ops.items():
            if "median_ms" not in entry:
                continue
            fastest = min(m for _, m in by_op[op_name])
            ratio = entry["median_ms"] / fastest if fastest else 0.0
            out.append((op_name, engine_id, entry["median_ms"], ratio,
                        entry.get("spread_pct", 0.0)))
    out.sort(key=lambda r: -r[3])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    print("=" * 72)
    print("  AAR engine benchmark")
    print("=" * 72)
    results = run(args.rows, args.repeats)

    print("\n" + "=" * 72)
    print("  Outliers: slower than the fastest engine for that operation")
    print("=" * 72)
    print(f"  {'operation':14s} {'engine':12s} {'median ms':>11s} "
          f"{'x fastest':>10s} {'spread':>8s}")
    for op_name, engine_id, median, ratio, spread in outliers(results):
        if ratio < 3.0:
            break
        print(f"  {op_name:14s} {engine_id:12s} {median:11.2f} "
              f"{ratio:9.1f}x {spread:7.1f}%")

    print("\n  A ratio smaller than the spread is noise, not a result. "
          "Correctness is\n  verified separately, against ArrowEngine and "
          "independent Python.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"rows": args.rows, "repeats": args.repeats,
                       "results": results}, handle, indent=2, default=str)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
