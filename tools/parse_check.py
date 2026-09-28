"""Parse every module under src/ and report the first syntax error per file.

Complements tools/check_syntax.py, which normalises line endings first and
so can mask an error this catches. Not part of the test suite.
"""
import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "src")
bad = 0
for path in sorted(root.rglob("*.py")):
    src = path.read_text(encoding="utf-8")
    try:
        ast.parse(src)
    except SyntaxError as exc:
        bad += 1
        print(f"{path}:{exc.lineno}: {exc.msg}")
        for n in range(max(1, exc.lineno - 2),
                       min(len(src.splitlines()), exc.lineno + 2) + 1):
            print(f"   {n:4d} {src.splitlines()[n - 1]!r}")
print(f"{bad} file(s) with syntax errors")
