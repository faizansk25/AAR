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


_COMMANDS = {
    "doctor": _cmd_doctor,
    "engines": _cmd_engines,
    "calibrate": _cmd_calibrate,
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
