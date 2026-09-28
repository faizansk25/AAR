"""Manual test kit: everything a person needs to check AAR by hand.

The automated suite proves the system agrees with itself. This proves it
agrees with a human - which is a different claim, and the one that decides
whether an analyst trusts the output.

Run it with no arguments for the guided tour, or a step name to run one
step:

    python tools/manual_test.py
    python tools/manual_test.py privacy
    python tools/manual_test.py --list

Each step prints what to do, what a correct result looks like, and where to
look. Nothing here is mocked: it is the real CLI, the real engines, and
real files on your disk.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PYTHON = os.path.join(ROOT, ".venv", "Scripts", "python.exe")

RUN = r"""
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
cd D:\AAR
$env:PYTHONPATH = "D:\AAR\src"
"""


def banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def run(args: list) -> tuple:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "src")
    proc = subprocess.run([PYTHON, "-B", "-m", "aar.cli", *args],
                          capture_output=True, text=True, env=env, cwd=ROOT)
    return proc.returncode, proc.stdout, proc.stderr


def show(args: list, expect_zero: bool = True) -> None:
    code, out, err = run(args)
    print(f"\n$ aar {' '.join(args)}")
    print(f"  exit code: {code}"
          + ("  (expected 0)" if expect_zero else "  (non-zero expected)"))
    for line in (out or err).splitlines()[:24]:
        print(f"  | {line}")
    if len((out or err).splitlines()) > 24:
        print("  | ...")


# ------------------------------------------------------------------- steps
def step_orient() -> None:
    banner("STEP 1 - What is installed on this machine?")
    print("AAR profiles the hardware it is on and only claims what it")
    print("actually found. Read `doctor` and check it matches reality.\n")
    show(["engines"])
    print("\nLook for: every absent engine has a *reason* next to it")
    print("('duckdb is not installed'), not just a blank.")


def step_pipeline() -> None:
    banner("STEP 2 - Run the shipped example end to end")
    print("This is the specification's own shape: an Excel file in, a")
    print("summary out, with a Python rule in the middle that no SQL")
    print("engine can accelerate.\n")
    show(["run", os.path.join(ROOT, "pipelines", "example_orders.py"),
          "--explain", "--head", "5"])
    print("\nLook for: every node names the engine it ran on, and any")
    print("degradation is stated rather than hidden.")


def step_privacy() -> None:
    banner("STEP 3 - The privacy path, which is the point of AAR")
    print("Source formats cannot carry a classification, so `classify()` is")
    print("the explicit boundary. A derived column inherits it, and a")
    print("policy has something to act on.\n")
    work = tempfile.mkdtemp(prefix="aar_manual_")
    try:
        policy = os.path.join(work, "policy.json")
        show(["policy", "--write-example", policy])
        show(["policy", "check", policy])
        print("\nNow confirm the policy *denies* egress by default:")
        print("  $ aar policy show " + policy)
        print("  Look for allow_network: false and the note that absent")
        print("  rules deny.")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def step_failures() -> None:
    banner("STEP 4 - Does it fail loudly?")
    print("A wrong answer you do not notice is worse than an error. AAR")
    print("is specified to never fail silently.\n")
    show(["run", "no_such_file.py"], expect_zero=False)
    show(["policy", "check", "no_such_policy.json"], expect_zero=False)
    print("\nBoth should exit non-zero and say what was wrong, in a")
    print("sentence, on stderr.")


def step_explain() -> None:
    banner("STEP 5 - Read a decision trace")
    print("Principle 5: every automated decision is logged with")
    print("rationale. This is where you find out *why* an engine was")
    print("chosen rather than being told it was the only option.\n")
    show(["explain", os.path.join(ROOT, "pipelines", "example_orders.py")])
    print("\nLook for: a per-segment cost breakdown, the alternatives that")
    print("were considered, and the margin by which the winner won.")


def step_data() -> None:
    banner("STEP 6 - Large real data (optional, needs internet)")
    print("The audit runs on the NYC taxi file, 3,066,766 rows. If you")
    print("want to exercise AAR the way an analyst would:\n")
    print("  1. python tools/fetch_data.py        (~100 MB, one time)")
    print("  2. python tools/audit.py             (the full sweep)")
    print("  3. open data/audit/audit.txt\n")
    print("The audit exits non-zero on any failure, so it works as a gate.")


STEPS = [
    ("orient", step_orient),
    ("pipeline", step_pipeline),
    ("privacy", step_privacy),
    ("failures", step_failures),
    ("explain", step_explain),
    ("data", step_data),
]


def main(argv: list) -> int:
    if "--list" in argv:
        print("steps: " + ", ".join(name for name, _ in STEPS))
        return 0
    print(__doc__)
    print(RUN)
    wanted = [a for a in argv if not a.startswith("-")]
    for name, fn in STEPS:
        if wanted and name not in wanted:
            continue
        fn()
    if not wanted:
        banner("MANUAL TEST KIT - DONE")
        print("Note what surprised you, in either direction. That is the")
        print("part the automated suite cannot tell you.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
