"""Dev utility: normalise encodings and report syntax errors.

Writes source files as UTF-8 without a BOM and with LF newlines, then parses
every module. PowerShell redirection on Windows tends to introduce a BOM,
which is a hard syntax error for the Python parser, so this is worth having
as a single command.
"""
import ast
import glob
import sys


def normalise(path: str) -> bool:
    """Rewrite a file as UTF-8 (no BOM) with LF endings. True if changed."""
    with open(path, "rb") as fh:
        raw = fh.read()
    original = raw
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    text = raw.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if text.encode("utf-8") != original:
        with open(path, "wb") as fh:
            fh.write(text.encode("utf-8"))
        return True
    return False


targets = sorted(glob.glob("src/aar/**/*.py", recursive=True) + glob.glob("tests/*.py"))
changed = [f for f in targets if normalise(f)]
if changed:
    print("normalised (BOM/CRLF removed):")
    for f in changed:
        print("  " + f)
else:
    print("no encoding changes needed")

bad = 0
for f in targets:
    src = open(f, encoding="utf-8").read()
    try:
        ast.parse(src)
        print(f"OK   {f}")
    except SyntaxError as e:
        bad += 1
        print(f"FAIL {f}:{e.lineno} {e.msg}")
        lines = src.splitlines()
        ln = e.lineno or 1
        for i in range(max(0, ln - 4), min(len(lines), ln + 2)):
            mark = ">>" if i == ln - 1 else "  "
            print(f"   {mark} {i + 1}: {lines[i]!r}")
print(f"\n{bad} file(s) with syntax errors")
sys.exit(1 if bad else 0)

