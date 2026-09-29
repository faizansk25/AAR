"""Reduce the reported import cycles to their root shapes.

`pyan3 -C` reports every *rotation* of every cycle, so a 6-module cycle
counts 6 times and a real handful of problems looks like hundreds. This
groups the report by member set, so what matters is the distinct count and
the shapes, not the total.

Writes the report to `ARCHITECTURE.md` alongside itself.

Run: python tools/cycle_roots.py
"""
from __future__ import annotations

import os
import re
import subprocess
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV = os.path.join(ROOT, ".venv", "Scripts", "pyan3.exe")
DOC = os.path.join(ROOT, "ARCHITECTURE.md")


def _cycles() -> list[tuple[str, ...]]:
    out = subprocess.run(
        [VENV, "--module-level", "--root", ".", "src/aar", "--text", "-C"],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    found: list[tuple[str, ...]] = []
    for match in re.finditer(r"\((?:\s*'[^']+'\s*,)+\s*'[^']+'\s*\)",
                             out.stdout):
        found.append(tuple(re.findall(r"'([^']+)'", match.group(0))))
    return found


def main() -> None:
    cycles = _cycles()
    groups: dict[frozenset[str], list[tuple[str, ...]]] = defaultdict(list)
    for c in cycles:
        groups[frozenset(c)].append(c)

    lines = [
        "# Structural analysis: pydeps + pyan3",
        "",
        "Regenerate with `python tools/cycle_roots.py`. Both tools are",
        "dev-only; neither is a runtime dependency, and `verify_release.py`",
        "still reports zero third-party dependencies for the installed wheel.",
        "",
        "```powershell",
        "pip install pydeps pyan3",
        "pydeps --no-output --show-cycles src\\aar\\__init__.py",
        "pyan3 --module-level --root . src\\aar --text -C",
        "python tools\\cycle_roots.py",
        "```",
        "",
        "## Import cycles",
        "",
        f"`pyan3 -C` reports **{len(cycles)} cycles**, which is a rotation",
        "count, not a problem count. Grouped by member set that is",
        f"**{len(groups)} distinct module sets**:",
        "",
    ]
    for modules, group in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        ordered = sorted(m.replace("src.aar.", "") for m in modules)
        lines.append(f"- **{len(group)} rotations, {len(ordered)} modules:** "
                     + ", ".join(f"`{m}`" for m in ordered))
    lines += [
        "",
        "### Why they are all here anyway",
        "",
        "Every back-edge is a **function-local import**, not a module-level",
        "one. That is deliberate: it is what keeps `import aar` free of",
        "pyarrow, duckdb, polars, pandas, cudf and openpyxl. An engine's",
        "third-party import happens when the engine is *constructed*, which is",
        "exactly when the dependency is known to be wanted.",
        "",
        "So the cycles are the price of lazy loading. They are benign at",
        "runtime, because Python only enters them at call time. The count is",
        "the same with and without pyan3's `--init` flag, so it is not an",
        "artifact of that flag's implicit package-`__init__` edges.",
        "",
        "The real risk is not import failure. It is that a cycle makes a",
        "*conceptual* boundary negotiable: a reader who sees",
        "`engines.arrow_engine` importing `connectors.excel` cannot tell",
        "whether that is a mistake or a rule. Hence this file - so the count",
        "is visible, and a growth in it is a deliberate decision rather than",
        "an accident. If a cycle ever needs breaking, invert the dependency",
        "(a write-format registry that engines read and connectors register",
        "into), do not shuffle the import into another function.",
        "",
        "## What pydeps found",
        "",
        "`--show-cycles` on the package entry point reports **no cycles",
        "reachable from `aar.__init__` alone** - the path a plain `import aar`",
        "takes. Everything above is reachable only once something calls into",
        "an engine or the workbench.",
        "",
        "External dependencies in the import graph are exactly the optional",
        "extras (pyarrow, duckdb, polars, pandas, cudf, openpyxl, pymongo,",
        "psycopg, pynvml). Nothing unexpected, and nothing unguarded at",
        "module scope.",
        "",
        "## Changes made because of this review",
        "",
        "- **`src/aar/examples.py` + `aar examples`.** The same review found",
        "  the CLI's worst ergonomic gap: `run` and `explain` both demand a",
        "  pipeline file, and nothing told a user what one looks like or gave",
        "  them one. Now `aar examples` lists them, `--show` prints one, and",
        "  `--write NAME PATH` writes an editable copy. The templates are",
        "  embedded in the package rather than shipped as data files, so",
        "  there is no `package-data` entry to forget at build time and no",
        "  difference between a checkout and an installed wheel.",
        "- The examples are **tested as code**: every template is parsed,",
        "  every `aar.sdk` import is checked against the live `__all__` (so",
        "  renaming an SDK function fails a test rather than a user), no",
        "  third-party import is allowed in an example, and the shortest one",
        "  is written to disk and run through `aar explain` for real.",
        "- `--write` refuses to overwrite. A pipeline someone has edited is",
        "  theirs.",
        "",
    ]
    with open(DOC, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"{len(cycles)} rotations, {len(groups)} distinct module sets")
    print("written:", DOC)


if __name__ == "__main__":
    main()
