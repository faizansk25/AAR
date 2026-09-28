"""Command-line interface.

Every subcommand here does real work against real modules. There are no
placeholder commands and no "coming soon" paths: a command exists because the
capability behind it is implemented and tested. The one command the
specification asks for that is *not* implemented yet - ``explain plan`` - is
deliberately absent rather than stubbed, because a command that prints
"not implemented" is worse than no command at all.

    aar doctor        this machine's hardware profile, in full
    aar engines       the engine catalogue and live availability
    aar calibrate     measure this machine and write the cost profile
    aar version       version and optional-dependency status
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Sequence

from . import __version__

__all__ = ["main", "build_parser"]

_EXIT_OK = 0


def build_parser() -> argparse.ArgumentParser:
    """The argument parser. Kept separate so tests can build it directly."""
    parser = argparse.ArgumentParser(
        prog="aar",
        description="Adaptive Analytics Runtime - hardware-aware analytics "
                    "orchestration.",
    )
    parser.add_argument("--version", action="version",
                        version=f"aar {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_doctor = sub.add_parser(
        "doctor", help="show this machine's hardware profile")
    p_doctor.add_argument("--json", action="store_true",
                          help="emit JSON instead of text")

    p_engines = sub.add_parser(
        "engines", help="show the engine catalogue and what is installed")
    p_engines.add_argument("--json", action="store_true",
                           help="emit JSON instead of text")

    p_cal = sub.add_parser(
        "calibrate", help="measure this machine and write a cost profile")
    p_cal.add_argument("--quick", action="store_true",
                       help="two sizes, minimal repeats; seconds not minutes")
    p_cal.add_argument("--no-gpu", action="store_true",
                       help="skip GPU benchmarks")
    p_cal.add_argument("--no-disk", action="store_true",
                       help="skip the storage benchmark")
    p_cal.add_argument("--out", default=None,
                       help="profile path (default: AAR_HOME or the "
                            "per-user AAR directory)")

    p_explain = sub.add_parser(
        "explain", help="plan a pipeline file and print the decision trace")
    p_explain.add_argument("pipeline", help="path to a Python pipeline file")
    p_explain.add_argument("--json", action="store_true",
                           help="emit JSON instead of text")

    p_run = sub.add_parser(
        "run", help="plan and execute a pipeline file")
    p_run.add_argument("pipeline", help="path to a Python pipeline file")
    p_run.add_argument("--explain", action="store_true",
                       help="also print the plan before running")
    p_run.add_argument("--json", action="store_true",
                       help="emit JSON instead of text")
    p_run.add_argument("--quiet", action="store_true",
                       help="print only the result summary")
    p_run.add_argument("--head", type=int, default=10,
                       help="rows to preview in the output (0 for none)")
    p_run.add_argument("--policy", default=None, metavar="PATH",
                       help="JSON policy file to enforce (see `aar policy` "
                            "for the format)")
    p_run.add_argument("--role", default=None,
                       help="role the run acts as, for row/column rules")
    p_run.add_argument("--as", dest="actor", default="cli",
                       help="subject name recorded in the policy audit")

    p_policy = sub.add_parser(
        "policy", help="inspect, validate or write a policy file")
    p_policy.add_argument("action", nargs="?", default=None,
                          choices=("show", "check"),
                          help="show: print a policy; check: validate one")
    p_policy.add_argument("path", nargs="?", default=None,
                          help="policy JSON file")
    p_policy.add_argument("--write-example", default=None, metavar="PATH",
                          help="write a documented example policy and exit")

    w = sub.add_parser("workbench",
                       help="open the Analyst Workbench in a browser")
    w.add_argument("--port", type=int, default=8765)
    w.add_argument("--host", default="127.0.0.1",
                   help="bind address (loopback by default)")
    w.add_argument("--open", action="store_true",
                   help="open a browser window immediately")

    sub.add_parser("version", help="version and optional-dependency status")
    return parser





def _cmd_doctor(args: argparse.Namespace) -> int:
    import json

    from .hardware import HardwareProfile

    profile = HardwareProfile()
    if args.json:
        print(json.dumps(profile.to_dict(), indent=2, default=str))
    else:
        print(profile.render())
    return _EXIT_OK


def _cmd_engines(args: argparse.Namespace) -> int:
    import json

    from .capability import default_registry

    registry = default_registry()
    if args.json:
        caps = registry.probe()
        print(json.dumps(
            {eid: {"available": c.available, "reason": c.reason,
                   "version": c.version, "blocked_by": c.blocked_by}
             for eid, c in caps.items()},
            indent=2))
    else:
        print(registry.render())
    return _EXIT_OK


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from .hardware import HardwareProfile
    from .hardware.calibrate import calibrate

    profile = HardwareProfile()
    store = calibrate(
        quick=args.quick,
        include_gpu=not args.no_gpu,
        include_disk=not args.no_disk,
        fingerprint=profile.fingerprint(),
        path=args.out,
        progress=lambda m: print(f"  {m}", file=sys.stderr),
    )
    print()
    print(store.render())
    if not store.is_calibrated:
        # Calibration produced nothing. That is a failure, not a success.
        print("\nCalibration produced no curves; see the error above.",
              file=sys.stderr)
        return _EXIT_USER_ERROR
    return _EXIT_OK


def _plan_for(path: str):
    """Load and plan a pipeline file. Returns the root node and the plan."""
    from .failures import PlanInfeasible
    from .planner import AdaptivePlanner
    from .sdk import load_pipeline

    if not os.path.isfile(path):
        print(f"aar: no such pipeline: {path}", file=sys.stderr)
        raise SystemExit(_EXIT_USER_ERROR)
    try:
        root = load_pipeline(path)
    except Exception as exc:  # noqa: BLE001
        print(f"aar: could not load {path}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        raise SystemExit(_EXIT_USER_ERROR) from exc
    try:
        return root, AdaptivePlanner().plan(root)
    except PlanInfeasible as exc:
        print(f"aar: {exc}", file=sys.stderr)
        raise SystemExit(_EXIT_USER_ERROR) from exc


def _load_policy(path: str | None):
    """Load a policy from JSON, or return None.

    Absent rules deny, so a policy that fails to load must not quietly
    become an empty one - that would turn a typo in a file path into a
    silently unprotected run. Failure here is a hard error.
    """
    if not path:
        return None
    if not os.path.isfile(path):
        print(f"aar: no such policy file: {path}", file=sys.stderr)
        raise SystemExit(_EXIT_USER_ERROR)
    from .governance import load_policy

    try:
        return load_policy(path)
    except Exception as exc:  # noqa: BLE001
        print(f"aar: could not load policy {path}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(_EXIT_USER_ERROR) from exc


def _cmd_run(args: argparse.Namespace) -> int:
    """Plan and execute a pipeline, reporting what actually happened."""
    import json

    from .failures import AARError
    from .runtime import Executor

    root, plan = _plan_for(args.pipeline)
    if args.explain and not args.json:
        print(plan.render())
        print()

    policy = _load_policy(getattr(args, "policy", None))
    subject = None
    if policy is not None:
        from .governance import Subject
        roles = frozenset({args.role}) if getattr(args, "role", None) \
            else frozenset()
        subject = Subject(name=getattr(args, "actor", "cli") or "cli",
                          roles=roles)

    with Executor(policy=policy, subject=subject) as executor:
        try:
            result = executor.execute(plan)
        except Exception as exc:  # noqa: BLE001
            # Report the node that failed and what had already succeeded, not
            # just the exception class. A bare "AttributeError" on stderr is
            # not something an analyst can act on.
            print(f"aar: execution failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            trace = executor.last_result
            if trace is not None:
                print("", file=sys.stderr)
                print(trace.render(), file=sys.stderr)
            if not isinstance(exc, AARError):
                traceback.print_exc()
            return 1


    if args.json:
        print(json.dumps({
            "ok": result.ok,
            "elapsed_ms": result.elapsed_s * 1e3,
            "rows": result.rows,
            "written": [{"target": t, "rows": r} for t, r in result.written],
            "nodes": [{
                "type": o.node_type,
                "engine_requested": o.engine_requested,
                "engine_used": o.engine_used,
                "rows_in": o.rows_in, "rows_out": o.rows_out,
                "elapsed_ms": o.elapsed_ms, "degraded": o.degraded,
            } for o in result.outcomes],
            "degradations": [d.to_dict() for d in result.ledger.entries],
        }, indent=2))
        return _EXIT_OK if result.ok else 1

    if not args.quiet:
        print(result.render())
        print()
    _print_preview(result, args.head)
    if not result.ok:
        print("\nCompleted with unresolved degradations; see above.",
              file=sys.stderr)
        return 1
    return _EXIT_OK


def _cmd_policy(args: argparse.Namespace) -> int:
    """Inspect, validate, or write out a policy file."""
    import json

    from .governance import EXAMPLE_POLICY, load_policy, policy_from_dict

    if args.write_example:
        with open(args.write_example, "w", encoding="utf-8") as fh:
            json.dump(EXAMPLE_POLICY, fh, indent=2)
            fh.write("\n")
        print(f"wrote {args.write_example}")
        return _EXIT_OK

    if not args.path:
        print("aar: policy needs a path, or --write-example PATH",
              file=sys.stderr)
        return _EXIT_USER_ERROR

    if not os.path.isfile(args.path):
        print(f"aar: no such policy file: {args.path}", file=sys.stderr)
        return _EXIT_USER_ERROR

    try:
        policy = load_policy(args.path)
    except Exception as exc:  # noqa: BLE001
        print(f"aar: invalid policy {args.path}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return _EXIT_USER_ERROR

    if args.action == "check":
        print(f"OK: {args.path} is a valid policy")
        return _EXIT_OK
    print(policy.render())
    return _EXIT_OK


def _print_preview(result: object, head: int) -> None:
    table = getattr(result, "table", None)
    if table is None or head <= 0 or table.num_rows == 0:
        return
    print(f"first {min(head, table.num_rows)} of {table.num_rows} rows:")
    for row in table.arrow.slice(0, head).to_pylist():
        print("  " + ", ".join(f"{k}={v!r}" for k, v in row.items()))


def _cmd_explain(args: argparse.Namespace) -> int:
    """Plan a pipeline file and print why each engine was chosen."""
    import json

    _root, plan = _plan_for(args.pipeline)

    if args.json:
        print(json.dumps({
            "total_ms": plan.total_s * 1e3,
            "segments": [{
                "index": sp.segment.index,
                "device": str(sp.device),
                "engine": sp.engine,
                "nodes": list(sp.segment.op_types),
                "total_ms": sp.total_s * 1e3,
                "inbound_ms": sp.inbound_s * 1e3,
                "reason": sp.reason,
            } for sp in plan.segments],
            "boundaries": plan.boundaries,
        }, indent=2))
    else:
        print(plan.render())
    return _EXIT_OK


def _cmd_version(_args: argparse.Namespace) -> int:
    from .hardware import probe_software

    print(f"aar {__version__}")
    print(f"python {sys.version.split()[0]}")
    sw = probe_software()
    present = {k: v for k, v in (
        ("pyarrow", sw.arrow), ("duckdb", sw.duckdb), ("polars", sw.polars),
        ("pandas", sw.pandas), ("cuDF", sw.cudf), ("pynvml", sw.nvml),
        ("openpyxl", sw.openpyxl), ("ray", sw.ray), ("dask", sw.dask),
    ) if v}
    print("installed: " + (", ".join(f"{k} {v}" for k, v in sorted(present.items()))
                           if present else "none (standard library only)"))
    return _EXIT_OK


def _cmd_workbench(args: argparse.Namespace) -> int:
    from .workbench import WorkbenchServer

    WorkbenchServer(host=args.host, port=args.port,
                    open_browser=args.open).serve_forever()
    return _EXIT_OK


_COMMANDS = {
    "doctor": _cmd_doctor,
    "engines": _cmd_engines,
    "calibrate": _cmd_calibrate,
    "explain": _cmd_explain,
    "run": _cmd_run,
    "policy": _cmd_policy,
    "workbench": _cmd_workbench,
    "version": _cmd_version,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code rather than calling ``exit``."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.command:
        parser.print_help()
        return _EXIT_USER_ERROR

    handler = _COMMANDS.get(args.command)
    if handler is None:  # pragma: no cover - argparse rejects unknown commands
        parser.error(f"unknown command: {args.command}")
        return _EXIT_USER_ERROR

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        # A CLI that dies with a traceback teaches the user nothing. Report
        # the failure plainly; the traceback is available with AAR_TRACEBACK=1.
        print(f"aar: {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("AAR_TRACEBACK"):
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

_EXIT_USER_ERROR = 2
