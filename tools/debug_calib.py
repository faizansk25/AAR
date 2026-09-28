"""Dev utility: run a single calibration benchmark and report errors.

The calibration harness deliberately swallows per-operation exceptions so one
unsupported op cannot abort a run. That is right in production and wrong when
debugging, so this surfaces what was swallowed.
"""
import sys
import traceback

sys.path.insert(0, "src")

import pyarrow.compute as pc  # noqa: E402

from aar.hardware import calibrate as C  # noqa: E402


def main() -> int:
    sizes = [1_000_000, 10_000_000, 100_000_000]
    tbl = C.make_frame(sizes[0])
    print(f"frame rows={tbl.num_rows} bytes={tbl.nbytes}")
    checks = {
        "scan": lambda: pc.sum(tbl["value"]).as_py() or 0,
        "filter": lambda: tbl["value"].null_count,
        "sort": lambda: tbl.num_rows,
        "groupby": lambda: C._bench_groupby(tbl),
        "window": lambda: C._bench_window(tbl),
        "hash_join": lambda: C._bench_join(tbl),
        "string_ops": lambda: tbl["group"].null_count,
        "arrow_ipc": lambda: C._bench_ipc(tbl),
        "parquet_decode": lambda: C._bench_parquet(tbl),
        "csv_decode": lambda: C._bench_csv(tbl),
    }
    failures = 0
    for name, fn in checks.items():
        try:
            secs, rows = C._time(fn, 1)
            print(f"  OK   {name:16s} {secs*1e3:8.2f} ms  rows={rows}")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"  FAIL {name:16s} {traceback.format_exc(limit=3)}")
    print(f"\n{failures} benchmark(s) failed")

    # Per-size monotonicity: report the raw points so a non-monotonic fit can
    # be distinguished from a genuine one.
    print("\n  raw groupby timings by size (repeats=3, best-of):")
    pts = []
    for size in sizes:
        t = C.make_frame(size)
        secs, _ = C._time(lambda: C._bench_groupby(t), 3)
        pts.append((t.nbytes, secs))
        print(f"    {t.nbytes/1e6:8.1f} MB  {secs*1e3:9.2f} ms")
    curve = C._fit([C.CalibrationPoint("groupby", "cpu", int(n), s)
                    for n, s in pts], "groupby", "cpu")
    print("   ", curve.describe())
    return 1 if failures else 0



if __name__ == "__main__":
    raise SystemExit(main())
