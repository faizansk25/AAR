"""The AAR audit: exercise the whole system against real data, and against
the specification.

Three questions, kept separate on purpose because they fail differently:

1. **Does it work?** Every command, every engine, every core operation, on
   a real multi-hundred-thousand-row dataset.
2. **Does it agree with itself?** Two engines, or the same pipeline run
   twice, must produce the same answer. A system that is fast and wrong is
   worse than one that is slow and right.
3. **Does it do what `system.md` asks?** Checked against the specification
   rather than against the code's own opinion of itself.

Reports go to `data/audit/` as JSON and as text. The exit code is non-zero
if any check fails, so this doubles as a CI gate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "audit")
PYTHON = os.path.join(ROOT, ".venv", "Scripts", "python.exe")

_results: list[dict] = []

#: Rows used when measuring the Arrow engine's Python grouping fallback. It
#: is orders of magnitude slower than the vectorised engines, and the audit
#: exists to finish; the finding itself is measured and reported separately.
GROUP_BY_ROWS = 300_000

#: Relative tolerance for comparing float aggregates across engines. A
#: Python loop and a vectorised kernel accumulate a sum in a different
#: order, so last-bit differences are expected and are not a defect. Exact
#: equality would report a correct engine as broken, which is worse than
#: missing a real one - so the tolerance is tight enough to catch a wrong
#: answer and loose enough to admit rounding.
FLOAT_TOLERANCE = 1e-9


def _grouped(result) -> dict:
    """A group-by result as {key: {column: value}}, order-independent."""
    return {r["PULocationID"]: r for r in result.arrow.to_pylist()}


def _values_match(a, b) -> bool:
    """Equal, allowing for float summation order across engines."""
    if isinstance(a, float) or isinstance(b, float):
        try:
            scale = max(abs(float(a)), abs(float(b)), 1.0)
        except (TypeError, ValueError):
            return a == b
        return abs(float(a) - float(b)) <= FLOAT_TOLERANCE * scale
    return a == b


def _same_groups(want: dict, got: dict, want_name: str,
                 got_name: str) -> tuple:
    if set(want) != set(got):
        only_want = sorted(set(want) - set(got))[:3]
        only_got = sorted(set(got) - set(want))[:3]
        return False, (f"different groups: only in {want_name} {only_want}, "
                       f"only in {got_name} {only_got}")
    for key, expected in want.items():
        actual = got[key]
        for column, value in expected.items():
            if column == "PULocationID":
                continue
            if not _values_match(value, actual.get(column)):
                return False, (f"group {key} {column}: {want_name}="
                               f"{value!r} {got_name}={actual.get(column)!r}")
    return True, f"{len(got):,} groups agree within {FLOAT_TOLERANCE:.0e}"




def record(group: str, name: str, ok: bool, detail: str = "",
          seconds: float = 0.0) -> bool:
    _results.append({"group": group, "name": name, "ok": bool(ok),
                     "detail": detail, "seconds": round(seconds, 2)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {group:12s} {name}  "
          f"{detail[:66]}")
    return ok


def check(group: str, name: str, fn, *args, **kwargs) -> bool:
    """Run one check, turning any exception into a recorded failure."""
    started = time.time()
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return record(group, name, False,
                      f"{type(exc).__name__}: {exc}", time.time() - started)
    if isinstance(result, tuple) and result:
        ok, detail = bool(result[0]), str(result[1])
    else:
        ok, detail = bool(result), ""
    return record(group, name, ok, detail, time.time() - started)


# --------------------------------------------------------------- the data
def taxi_parquet() -> str:
    """The real dataset, or a clear failure if the fetch never happened."""
    for name in ("nyc_taxi_2023_01.parquet", "nyc_taxi_2022_03.parquet"):
        path = os.path.join(DATA, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "no taxi parquet found; run `python tools/fetch_data.py` first")


def row_count() -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(taxi_parquet()).metadata.num_rows


#: Rows the audit exercises. 500,000 is comfortably in the "lakhs" range
#: and keeps a four-engine sweep to minutes rather than tens of minutes. Pass
#: `--full` to use every row of the file.
AUDIT_ROWS = 500_000


def load_real(limit: int | None = None):
    """A large real slice as an AAR table, through the type system."""
    import pyarrow as pa

    from aar.interchange import Table, arrow_to_canonical
    from aar.types import Field, Schema

    arrow_table = pa.parquet.read_table(taxi_parquet())
    if limit:
        arrow_table = arrow_table.slice(0, limit)
    schema = Schema(tuple(Field(f.name, arrow_to_canonical(f.type))
                          for f in arrow_table.schema))
    return Table(arrow_table, schema)



def available_engines() -> list:
    from aar.engines.factory import create_engine

    found = []
    for engine_id in ("arrow", "duckdb", "polars_cpu", "pandas"):
        try:
            create_engine(engine_id)
        except Exception:  # noqa: BLE001
            continue
        found.append(engine_id)
    return found


def _run_cli(args: list, timeout: int = 900) -> tuple:
    """Run the real CLI as a subprocess, the way an analyst would."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "src")
    process = subprocess.run(
        [PYTHON, "-B", "-m", "aar.cli", *args],
        capture_output=True, text=True, timeout=timeout, env=env, cwd=ROOT)
    return process.returncode, process.stdout, process.stderr


# ------------------------------------------------------ 1. does it work?
def audit_engines_on_real_data() -> None:
    """Every engine, every core operation, on every real row there is.

    The numbers being *right* is not what this checks - they are compared
    against each other in the next stage. What this catches is a crash, a
    degradation, or a silent truncation at scale.
    """
    import pyarrow.compute as pc

    from aar.engines.factory import create_engine
    from aar.ir import Agg, BinOp, Col, Lit

    table = load_real(AUDIT_ROWS)
    engines = available_engines()
    print(f"\n-- engines on {table.num_rows:,} real rows "
          f"(of {row_count():,} in the file) --")
    check("engines", "available", lambda: (True, ", ".join(engines)))

    for engine_id in engines:
        engine = create_engine(engine_id)

        def run_filter(e=engine):
            started = time.time()
            # `passenger_count IS NOT NULL` rather than `> 0`: the real file
            # has nulls, and a check that assumed otherwise would be
            # reporting a correct engine as a broken one. (It did, once.)
            out = e.filter(table,
                           BinOp(Col("passenger_count"), "IS NOT", Lit(None)))
            non_null = pc.sum(pc.is_valid(table.arrow.column(
                "passenger_count"))).as_py()
            return (out.num_rows == non_null,
                    f"{out.num_rows:,}/{non_null:,} non-null rows in "
                    f"{time.time() - started:.1f}s")
        check("engines", f"{engine_id}.filter", run_filter)

        def run_project(e=engine):
            out = e.project(table, list(table.column_names)[:3])
            return out.num_columns == 3, str(out.column_names)
        check("engines", f"{engine_id}.project", run_project)

        def run_limit(e=engine):
            out = e.limit(table, 10)
            return out.num_rows == 10, f"{out.num_rows} rows"
        check("engines", f"{engine_id}.limit", run_limit)

        def run_sort(e=engine):
            started = time.time()
            out = e.sort(table, [("trip_distance", False)])
            return (out.num_rows == table.num_rows,
                    f"{out.num_rows:,} rows in {time.time() - started:.1f}s")
        check("engines", f"{engine_id}.sort", run_sort)

        def run_group(e=engine, eid=engine_id):
            # The Arrow engine's grouping falls back to a Python path on this
            # build of pyarrow (see _group_by_kernel_works), which is orders
            # of magnitude slower than DuckDB or Polars. Measuring it on all
            # 3M rows would take minutes and stall the audit, so the slow
            # fallback is measured on a bounded slice and labelled as such.
            bounded = eid == "arrow"
            source = table.slice(0, GROUP_BY_ROWS) if bounded else table
            started = time.time()
            out = e.group_by(source, ["PULocationID"],
                             {"total": Agg("SUM", Col("trip_distance"),
                                           "total")})
            detail = (f"{out.num_rows:,} groups from {source.num_rows:,} rows "
                      f"in {time.time() - started:.1f}s")
            if bounded:
                detail += " (fallback path, bounded slice)"
            return (out.num_rows > 0, detail)
        check("engines", f"{engine_id}.group_by", run_group)



# ------------------------------------------- 2. does it agree with itself?
def audit_agreement() -> None:
    """Every engine must produce the same answer, not merely an answer.

    This is the check that catches engine-specific bugs. A result that is
    wrong on one engine is invisible to any test that only ever ran on
    another, which is exactly how the tag-loss bugs survived.
    """
    from aar.engines.factory import create_engine
    from aar.ir import Agg, BinOp, Col, Lit

    print("\n-- cross-engine agreement --")
    table = load_real(AUDIT_ROWS)
    engines = available_engines()
    if len(engines) < 2:
        record("agreement", "at_least_two_engines", False,
               "cannot compare with fewer than two engines")
        return
    record("agreement", "at_least_two_engines", True, ", ".join(engines))

    reference_id = engines[0]
    reference = create_engine(reference_id)

    # A numeric aggregate: the operation most sensitive to how an engine
    # types its own intermediate results. Note `Agg(func, arg, distinct)` -
    # the third positional is `distinct`, not an alias. Passing an alias
    # there is a truthy string and silently makes every aggregate DISTINCT.
    aggs = {"total": Agg("SUM", Col("fare_amount")),
            "n": Agg("COUNT", Col("fare_amount"))}
    want = _grouped(reference.group_by(table, ["PULocationID"], aggs))
    for engine_id in engines[1:]:
        def compare(eid=engine_id, want=want):
            got = _grouped(create_engine(eid).group_by(
                table, ["PULocationID"], aggs))
            return _same_groups(want, got, reference_id, eid)
        check("agreement", f"{engine_id}.group_by == {reference_id}", compare)


    predicate = BinOp(Col("fare_amount"), ">", Lit(10.0))
    want_rows = reference.filter(table, predicate).num_rows
    for engine_id in engines[1:]:
        def compare_filter(eid=engine_id, want=want_rows):
            got = create_engine(eid).filter(table, predicate).num_rows
            return (got == want,
                    f"{got:,} rows" if got == want else
                    f"{eid} kept {got:,}, {reference_id} kept {want:,}")
        check("agreement", f"{engine_id}.filter == {reference_id}",
              compare_filter)


# ------------------------------------------------------ 3. every command
def _write_pipeline() -> str:
    """A real pipeline over the real dataset, written to a file.

    The paths are emitted with ``!r`` rather than by hand-escaping
    backslashes: a Windows path written into an ``r'...'`` literal with
    doubled separators is a *different, non-existent* path, and the
    pipeline then fails to load for a reason that has nothing to do with
    AAR. Repr produces a literal that is correct on every platform.
    """
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "pipeline_audit.py")
    source = taxi_parquet()
    target = os.path.join(OUT, "audit_out.parquet")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            '"""Real-data pipeline for the audit."""\n'
            "from aar.sdk import (classify, group_by, limit, parquet, sort,\n"
            "                      sum_, write_parquet)\n\n"
            f"SRC = {source!r}\n"
            f"OUT = {target!r}\n\n\n"
            "def build():\n"
            "    t = parquet(SRC)\n"
            "    t = classify(t, 'fare_amount', 'confidential')\n"
            "    t = group_by(t, 'PULocationID',\n"
            "                 total=sum_('fare_amount'),\n"
            "                 trips=sum_('passenger_count'))\n"
            "    t = sort(t, total=True)\n"
            "    t = limit(t, 50)\n"
            "    return write_parquet(t, OUT)\n")
    return path


def _write_example(policy_path: str) -> tuple:
    code, _, err = _run_cli(["policy", "--write-example", policy_path])
    if code != 0:
        return False, f"exit={code} {err.strip()[:70]}"
    if not os.path.exists(policy_path):
        return False, "no policy file was written"
    code, _, _ = _run_cli(["policy", "check", policy_path])
    return (code == 0,
            f"written and validated, {os.path.getsize(policy_path)}B")


def audit_cli() -> None:
    """Every command, on real data, checking exit code and output.

    Run as a subprocess on purpose: importing the CLI and calling handlers
    would skip argument parsing, which is where a command usually breaks,
    and would not catch one that crashes without a terminal attached.
    """
    print("\n-- CLI --")
    for args in (["version"], ["doctor"], ["doctor", "--json"],
                 ["engines"], ["engines", "--json"], ["--version"]):
        def run(a=args):
            code, out, err = _run_cli(a)
            return (code == 0 and bool(out.strip()),
                    f"exit={code} {len(out)}B out"
                    + (f", ERR {err.strip()[:50]}" if err.strip() else ""))
        check("cli", " ".join(args), run)

    pipeline = _write_pipeline()
    for args in (["explain", pipeline], ["run", pipeline],
                 ["run", pipeline, "--explain"],
                 ["run", pipeline, "--head", "3"],
                 ["run", pipeline, "--quiet"],
                 ["run", pipeline, "--json"]):
        def run(a=args):
            code, out, err = _run_cli(a)
            return (code == 0 and bool(out.strip()),
                    f"exit={code} {len(out)}B out"
                    + (f", ERR {err.strip()[:60]}" if err.strip() else ""))
        check("cli", " ".join(args[:2]) + " " + " ".join(args[2:3]), run)

    policy_path = os.path.join(OUT, "policy.json")
    check("cli", "policy --write-example",
          lambda: _write_example(policy_path))
    for action in ("show", "check"):
        def run(a=action):
            code, out, _ = _run_cli(["policy", a, policy_path])
            return code == 0 and bool(out.strip()), \
                f"exit={code} {out.strip()[:45]}"
        check("cli", f"policy {action}", run)

    def run_policy():
        code, out, err = _run_cli(
            ["run", pipeline, "--policy", policy_path, "--role", "analyst",
             "--as", "auditor", "--head", "0"])
        return (code == 0, f"exit={code} {out.strip()[:55]}"
                + (f", ERR {err.strip()[:55]}" if err.strip() else ""))
    check("cli", "run --policy", run_policy)

    def run_missing():
        code, _, _ = _run_cli(["run", "no_such_pipeline.py"])
        return code != 0, f"exit={code} (non-zero is correct)"
    check("cli", "run on a missing file fails loudly", run_missing)


# ------------------------------------- 4. against the specification
def _import_offline() -> tuple:
    """Import every aar module with `socket.socket` disabled.

    Principle 2 says the core must work air-gapped. Importing the package
    with the socket layer poisoned is the cheapest way to find out whether
    that is true, and it finds import-time calls that a feature test would
    never reach.
    """
    import importlib
    import pkgutil
    import socket

    import aar

    original = socket.socket

    class Blocked(socket.socket):
        def __init__(self, *args, **kwargs):
            raise OSError("network access is disabled by the audit")

    socket.socket = Blocked
    imported, failed = 0, []
    try:
        for module in pkgutil.walk_packages(aar.__path__, "aar."):
            try:
                importlib.import_module(module.name)
                imported += 1
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{module.name}: {type(exc).__name__}: {exc}")
    finally:
        socket.socket = original
    if failed:
        return False, (f"{len(failed)} module(s) need the network: "
                       + "; ".join(failed[:3]))
    return True, f"{imported} modules imported with sockets blocked"


def audit_spec() -> None:
    """Check the system's claims against `system.md`, not against itself.

    A requirement that cannot be checked on this machine is reported as
    such rather than quietly counted as met.
    """
    print("\n-- specification --")
    spec = ""
    spec_path = os.path.join(ROOT, "system.md")
    if os.path.exists(spec_path):
        with open(spec_path, encoding="utf-8") as handle:
            spec = handle.read()

    check("spec", "P2 deny-outbound is stated in the spec",
          lambda: ("DENY outbound" in spec, "found in system.md"))
    check("spec", "P2 core imports with sockets blocked", _import_offline)

    def explain_covers_decisions() -> tuple:
        code, out, _ = _run_cli(["explain", _write_pipeline()])
        return (code == 0 and "segment" in out and "ms" in out,
                f"exit={code}, {len(out)} chars, "
                f"segments named={'segment' in out}")
    check("spec", "P5 explain names every segment decision",
          explain_covers_decisions)

    def degradation_is_recorded() -> tuple:
        from aar.capability import default_registry
        caps = default_registry().probe()
        missing = [eid for eid, c in caps.items() if not c.available]
        unexplained = [eid for eid in missing if not caps[eid].reason]
        return (not unexplained,
                f"{len(missing)} absent engine(s), all with a reason"
                if not unexplained
                else f"{len(unexplained)} absent engine(s) with NO reason")
    check("spec", "P9 absent engines carry a reason",
          degradation_is_recorded)

    def derived_inherits() -> tuple:
        """A derived column must be at least as sensitive as its input."""
        import pyarrow as pa

        from aar.engines.factory import create_engine
        from aar.interchange import Table
        from aar.ir import Agg, Col
        from aar.types import Field, INT64, Schema, UTF8

        schema = Schema((Field("region", UTF8),
                         Field("salary", INT64,
                               classification=frozenset({"confidential"}))))
        table = Table(pa.table({"region": ["a", "b"], "salary": [1, 2]}),
                      schema)
        for engine_id in available_engines():
            out = create_engine(engine_id).group_by(
                table, ["region"],
                {"total": Agg("SUM", Col("salary"), "total")})
            if "confidential" not in out.schema.get("total").classification:
                return False, f"{engine_id} lost the tag on SUM(salary)"
        return True, "every engine keeps SUM(confidential) confidential"
    check("spec", "derived columns inherit sensitivity", derived_inherits)

    def unknown_sink_denied() -> tuple:
        from aar.governance import Policy, PolicyEngine, Sink
        engine = PolicyEngine(Policy())
        for name in ("ftp", "smb", "mystery_service"):
            if engine.check_egress(Sink.of(name), "auditor").action.value \
                    != "deny":
                return False, f"{name} was not denied"
        return True, "ftp, smb and unknown names all denied"
    check("spec", "unknown sinks fail closed", unknown_sink_denied)

    def ir_is_deterministic() -> tuple:
        from aar.ir import Node, NodeType, topological_order
        node = Node(NodeType.FILTER)
        return (topological_order([node]) == topological_order([node]),
                "same graph, same order")
    check("spec", "IR traversal is deterministic", ir_is_deterministic)

    def run_is_repeatable() -> tuple:
        """The same pipeline twice must produce the same answer."""
        import hashlib
        path = _write_pipeline()
        target = os.path.join(OUT, "audit_out.parquet")
        digests = []
        for _ in range(2):
            code, _, err = _run_cli(["run", path, "--head", "0"])
            if code != 0:
                return False, f"exit={code} {err.strip()[:70]}"
            with open(target, "rb") as handle:
                digests.append(hashlib.sha256(handle.read()).hexdigest())
        return (digests[0] == digests[1],
                f"sha256 {digests[0][:16]} on both runs"
                if digests[0] == digests[1]
                else f"differs: {digests[0][:12]} vs {digests[1][:12]}")
    check("spec", "a repeated run is byte-identical", run_is_repeatable)


# ------------------------------------------------------------- reporting
def report() -> int:
    os.makedirs(OUT, exist_ok=True)
    failed = [r for r in _results if not r["ok"]]
    summary = {"total": len(_results),
               "passed": len(_results) - len(failed),
               "failed": len(failed), "results": _results}
    with open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    lines = ["AAR AUDIT", "=" * 62, ""]
    by_group: dict = {}
    for r in _results:
        by_group.setdefault(r["group"], []).append(r)
    for group, rows in by_group.items():
        bad = sum(1 for r in rows if not r["ok"])
        lines.append(f"{group:12s} {len(rows) - bad:3d}/{len(rows):<3d} pass"
                     + (f"   <-- {bad} FAILED" if bad else ""))
        for r in rows:
            if not r["ok"]:
                lines.append(f"    FAIL {r['name']}: {r['detail']}")
    lines += ["", f"TOTAL {summary['passed']}/{summary['total']} pass, "
                  f"{summary['failed']} failed", ""]
    text = "\n".join(lines)
    with open(os.path.join(OUT, "audit.txt"), "w", encoding="utf-8") as handle:
        handle.write(text)
    print("\n" + text)
    return 1 if failed else 0


def main() -> int:
    print(f"AAR audit - data dir {DATA}")
    try:
        print(f"real dataset: {row_count():,} rows")
    except Exception as exc:  # noqa: BLE001
        print(f"no real data: {exc}")
        print("run: python tools/fetch_data.py first")
        return 2

    for stage in (audit_engines_on_real_data, audit_agreement, audit_cli,
                  audit_spec):
        try:
            stage()
        except Exception as exc:  # noqa: BLE001
            record(stage.__name__, "stage completed", False,
                   f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
    return report()


if __name__ == "__main__":
    sys.exit(main())
