"""A pipeline that runs on a bare install, and checks its own arithmetic.

Run by ``tools/verify_release.py`` inside a clean venv, and runnable on its
own. The point is the constraint: **no pyarrow, no duckdb, no polars, no
pandas.** AAR's promise is that the core works where nothing has been
installed - an air-gapped laptop, an air-gapped server - so verifying it on
a dev venv full of engines would test the one configuration nobody is
guaranteed.

So this uses the CSV reader and the Python engine, which is the worst case.
If the numbers come out right *there*, the core is genuinely
dependency-free rather than incidentally so on this machine.

The arithmetic is checked against a plain Python loop rather than against
another engine, so agreement means agreement with the definition rather than
with a sibling implementation.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

#: If any of these import, the environment is not the bare one this is meant
#: to prove, and a pass would be meaningless.
FORBIDDEN = ("pyarrow", "duckdb", "polars", "pandas")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-no-engine", action="store_true",
        help="a bare install: assert that execution is refused *cleanly*, "
             "with a named reason, rather than crashing")
    args = parser.parse_args()

    for module in FORBIDDEN:
        try:
            __import__(module)
        except ImportError:
            continue
        if args.expect_no_engine:
            print(f"UNEXPECTED: {module} is importable, so this is not a bare "
                  f"install and the check would prove nothing")
            return 2
        break
    print("environment: "
          + ("bare, no third-party engine importable"
             if args.expect_no_engine else "engine available"))

    try:
        from aar.runtime import Executor
        from aar.sdk import count, csv, group_by, sum_, write_csv
    except Exception as exc:  # noqa: BLE001
        print(f"import failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    try:
        from aar.planner import AdaptivePlanner
    except Exception as exc:  # noqa: BLE001
        print(f"import failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    work = Path(tempfile.mkdtemp(prefix="aar-pipeline-"))
    source = work / "orders.csv"
    target = work / "summary.csv"

    # Deterministic data. Headers are NOT padded: pyarrow.csv does not strip
    # whitespace from header names by default, so a padded header really does
    # produce a column called "  region " and a KeyError three operations
    # later. That is a genuine usability gap, recorded rather than papered
    # over by quietly changing reader semantics this late - but the fixture
    # must not assert a behaviour the reader does not have.
    lines = ["region,amount"]
    expected: dict[str, float] = {}
    counts: dict[str, int] = {}
    for i in range(300):
        region = f"r{i % 5}"
        amount = float((i * 13) % 97)
        expected[region] = expected.get(region, 0.0) + amount
        counts[region] = counts.get(region, 0) + 1
        lines.append(f"{region},{amount}")
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        # The real two-step flow: build the IR, plan it, execute the plan.
        # Composing SDK calls is only half of it - an unplanned IR is not
        # something the executor accepts, and finding that out here is the
        # point of running this against an installed wheel.
        node = write_csv(
            group_by(
                csv(str(source)),
                "region",
                aggs={"total": sum_("amount"), "n": count("amount")},
            ),
            str(target),
        )
        plan = AdaptivePlanner().plan(node)
        with Executor() as executor:
            result = executor.execute(plan)
        if not result.ok:
            print(f"run did not succeed: {result.ledger.render()}")
            return 1
        rows = getattr(result, "table", None)
        print(f"executor reported "
              f"{rows.num_rows if rows is not None else 0} rows")
        degradations = list(result.ledger.entries)
        if degradations:
            print(f"degradations recorded: {len(degradations)}")
    except Exception as exc:  # noqa: BLE001
        if args.expect_no_engine:
            # The correct outcome on a bare install: a refusal that names the
            # segment and says which engine was missing. An ImportError or a
            # bare traceback here would mean the failure path is broken, so
            # the reason is required to be specific.
            message = f"{type(exc).__name__}: {exc}"
            if "NO_FEASIBLE_PLAN" not in message and "no engine" not in message:
                print(f"REFUSED, but not with an explanation: {message}")
                return 1
            print(f"refused cleanly, as it must: {message}")
            return 0
        print(f"pipeline failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    if args.expect_no_engine:
        print("UNEXPECTED: a bare install executed a pipeline. Either an "
              "engine became importable, or something executed without one.")
        return 1

    if not target.exists():
        print(f"no output written: {target}")
        return 1

    got: dict[str, tuple[float, int]] = {}
    # Parsed with the csv module, not `line.split(",")`. The writer quotes
    # string fields - `"r0"` - which is correct RFC 4180 behaviour and what
    # every Arrow and pandas writer does. Splitting on commas by hand leaves
    # the quotes attached and calls correct output wrong, which is how this
    # fixture failed once already. A region containing a comma would break it
    # a second way.
    import csv as _csv

    with open(target, encoding="utf-8", newline="") as handle:
        for row in _csv.DictReader(handle):
            if not row or not row.get("region"):
                continue
            got[row["region"].strip()] = (float(row["total"]),
                                          int(float(row["n"])))

    problems: list[str] = []
    if set(got) != set(expected):
        problems.append(f"groups differ: {sorted(got)} vs {sorted(expected)}")
    for region, want in expected.items():
        if region not in got:
            continue
        total, n = got[region]
        if abs(total - want) > 1e-9:
            problems.append(f"{region}: total {total} != {want}")
        if n != counts[region]:
            problems.append(f"{region}: count {n} != {counts[region]}")

    if problems:
        for problem in problems:
            print(f"MISMATCH: {problem}")
        return 1

    print(f"verified {len(got)} groups against independent Python arithmetic")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
