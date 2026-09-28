"""Install the built wheel into a clean environment and actually use it.

This is the check that matters most before publishing, and the one most
easily skipped. The development venv has AAR installed *editable*, which
means the local `src/` tree is on the path. Every packaging bug therefore
invisible at development time is invisible there too:

* a package-data file that never made it into the wheel (the Workbench's
  `static/` assets - a real bug this repository had)
* a `[project.scripts]` entry that produces no console script
* a dependency declared in the wrong extra
* a module importing something that only exists in the dev venv

So this builds an environment *outside the source tree*, installs the
wheel, and then uses the result: import it, run the console script, find
the packaged data files, and execute a real pipeline whose numbers are
checked against independent Python arithmetic.

**Zero third-party dependencies is a design claim, so it is verified.**
After the install, `pip list` is compared against the standard library. If
pyarrow, duckdb, polars or pandas leaked in transitively that is a failure,
not a warning - the point is that a bare install works air-gapped.

Run after `python -m build`, before `twine upload`.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Not third-party: these ship with CPython and are not evidence that a
#: dependency leaked in.
_STDLIB = {"pip", "setuptools", "wheel", "pkg_resources"}

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}"
          + (f" - {detail}" if detail else ""), flush=True)
    return ok


def run(args: list[str], cwd: Path, timeout: int = 300
        ) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=str(cwd), capture_output=True,
                          text=True, timeout=timeout)


def newest_wheel() -> Path | None:
    """The most recently built wheel, or None.

    Newest by mtime rather than version, because the point is "the artifact
    you are about to upload". A higher version that failed to build is worse
    than a lower one that exists, and a stale 0.1.0 sitting next to a
    rebuilt 0.1.0 would silently be the one verified.
    """
    wheels = sorted(glob.glob(str(ROOT / "dist" / "*.whl")),
                    key=os.path.getmtime)
    return Path(wheels[-1]) if wheels else None


def main() -> int:
    print("=" * 70)
    print("  AAR release verification: install the wheel, then use it")
    print("=" * 70, flush=True)

    wheel = newest_wheel()
    if wheel is None:
        print("\nNo wheel in dist/. Run `python -m build` first.", flush=True)
        return 1
    print(f"\nwheel: {wheel.name}", flush=True)

    # Deliberately in the system temp directory. Inside the repo, `import
    # aar` can resolve to src/ and every check below passes while testing the
    # source tree instead of the artifact.
    workdir = Path(tempfile.mkdtemp(prefix="aar-verify-"))
    venv = workdir / "venv"
    print(f"clean env: {venv}\n", flush=True)

    try:
        print("building the environment", flush=True)
        proc = run([sys.executable, "-m", "venv", str(venv)], workdir)
        if proc.returncode != 0:
            print(proc.stderr[-800:], flush=True)
            return 1
        py = venv / "Scripts" / "python.exe"
        check("venv created", py.exists(), str(py))

        print("\ninstalling the wheel", flush=True)
        proc = run([str(py), "-m", "pip", "install", "--quiet",
                    "--no-cache-dir", str(wheel)], workdir, timeout=600)
        if not check("wheel installed", proc.returncode == 0,
                     proc.stderr.strip()[-200:]):
            print(proc.stdout[-800:], flush=True)
            return 1

        print("\ndependencies", flush=True)
        proc = run([str(py), "-m", "pip", "list", "--format=json"], workdir)
        installed: dict = {}
        if proc.returncode == 0:
            installed = {p["name"].lower(): p["version"]
                         for p in json.loads(proc.stdout)}
        third_party = {n: v for n, v in installed.items()
                       if n not in _STDLIB and "analytics-runtime" not in n}
        # The design claim: a bare install must be pure standard library, so
        # AAR runs on an air-gapped machine with nothing preinstalled.
        check("zero third-party dependencies", not third_party,
              f"unexpected: {third_party}" if third_party
              else f"only {sorted(installed)}")

        print("\nthe package as a user gets it", flush=True)
        proc = run([str(py), "-c",
                    "import aar; print(aar.__version__)"], workdir)
        check("import aar works", proc.returncode == 0,
              proc.stdout.strip() or proc.stderr.strip()[-200:])

        script = venv / "Scripts" / "aar.exe"
        check("console script exists", script.exists(), str(script))
        if script.exists():
            proc = run([str(script), "--version"], workdir)
            check("console script runs", proc.returncode == 0,
                  (proc.stdout + proc.stderr).strip()[:120])

        proc = run([str(py), "-c", "import aar, os; "
                    "print(os.path.dirname(aar.__file__))"], workdir)
        location = proc.stdout.strip()
        check("imported from the venv, not from src/",
              "venv" in location.lower() and str(ROOT / "src") not in location,
              location)

        print("\npackaged data files", flush=True)
        proc = run([str(py), "-c",
                    "from aar.workbench.server import STATIC; "
                    "import os; print(sorted(os.listdir(STATIC)))"], workdir)
        try:
            names = eval(proc.stdout.strip()) if proc.returncode == 0 else []
        except (SyntaxError, NameError):
            names = []
        for required in ("index.html", "app.js", "app.css"):
            check(f"workbench asset {required}", required in names,
                  "" if required in names else f"found {names}")

        proc = run([str(py), "-c",
                    "import importlib.metadata as m;"
                    "d=m.distribution('adaptive-analytics-runtime');"
                    "m2=d.metadata;"
                    "print(bool(m2.get('License-Expression') or "
                    "'Apache' in str(m2.get('License'))))"], workdir)
        check("licence metadata present", proc.stdout.strip() == "True",
              proc.stdout.strip()[:80])

        print("\nphase 1: bare install (no engine available)", flush=True)
        # Executing a pipeline needs an engine, and every engine needs
        # pyarrow. So on a bare install the *expected* result is a refusal -
        # and the check is that the refusal is specific and explained. A
        # traceback here would mean the failure path itself is broken.
        proc = run([str(py), str(ROOT / "tools" / "release_pipeline.py"),
                    "--expect-no-engine"], workdir, timeout=300)
        detail = ""
        combined = (proc.stdout + proc.stderr).strip()
        if combined:
            detail = combined.splitlines()[-1][:150]
        check("bare install refuses cleanly, with a reason",
              proc.returncode == 0, detail)
        for line in proc.stdout.splitlines():
            print(f"      {line}", flush=True)

        print("\nphase 2: add an engine and really run it", flush=True)
        proc = run([str(py), "-m", "pip", "install", "--quiet",
                    "--no-cache-dir", "pyarrow"], workdir, timeout=900)
        if not check("pyarrow installed", proc.returncode == 0,
                     proc.stderr.strip()[-200:]):
            print(proc.stdout[-600:], flush=True)
        else:
            proc = run([str(py), str(ROOT / "tools" / "release_pipeline.py")],
                       workdir, timeout=300)
            detail = ""
            combined = (proc.stdout + proc.stderr).strip()
            if combined:
                detail = combined.splitlines()[-1][:150]
            check("end-to-end pipeline from the installed wheel",
                  proc.returncode == 0, detail)
            for line in proc.stdout.splitlines():
                print(f"      {line}", flush=True)
            if proc.returncode != 0:
                print(proc.stdout[-1200:], flush=True)
                print(proc.stderr[-1200:], flush=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    failed = [r for r in results if r[0] == FAIL]
    print("\n" + "=" * 70)
    print(f"  {len(results) - len(failed)} passed, {len(failed)} failed")
    print("=" * 70, flush=True)
    for _, name, detail in failed:
        print(f"  FAILED: {name} {detail}", flush=True)
    if failed:
        print("\nDo not upload. These are only visible from the installed "
              "artifact, never from the source tree.", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())



def newest_wheel() -> Path | None:
    wheels = sorted(glob.glob(str(ROOT / "dist" / "*.whl")),
                    key=os.path.getmtime)
    return Path(wheels[-1]) if wheels else None
