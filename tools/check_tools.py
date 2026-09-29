"""Check whether pydeps and pyan3 are installed or installable."""
import importlib.util as u
import subprocess
import sys

for name in ("pydeps", "pyan", "pyan3", "ruff", "mypy", "pyflakes"):
    try:
        found = bool(u.find_spec(name))
    except Exception as exc:
        found = f"error: {exc}"
    print(f"{name:10} installed={found}")

print("\n--- pip index availability ---")
for pkg in ("pydeps", "pyan3", "pyan"):
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "download", "--no-deps",
             "--dest", "NUL", pkg],
            capture_output=True, text=True, timeout=180)
        tail = (out.stderr or out.stdout).strip().splitlines()[-3:]
        print(f"\n{pkg}: rc={out.returncode}")
        for line in tail:
            print("   ", line)
    except Exception as exc:
        print(f"{pkg}: {type(exc).__name__}: {exc}")
