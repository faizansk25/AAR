"""Dev utility: truncate a module to a given line, safely.

Usage: python tools/truncate.py <file> <keep-through-line>

Used to drop orphaned fragments left behind by an interrupted edit. The file
is rewritten atomically as UTF-8 (no BOM) with a trailing newline.
"""
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    path, n_raw = sys.argv[1], sys.argv[2]
    try:
        keep = int(n_raw)
    except ValueError:
        print(f"not a line number: {n_raw!r}")
        return 2

    with open(path, "rb") as fh:
        raw = fh.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    lines = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").split("\n")

    if keep < 1 or keep > len(lines):
        print(f"{path}: line {keep} out of range (file has {len(lines)})")
        return 2
    kept = "\n".join(lines[:keep]).rstrip() + "\n"
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(kept.encode("utf-8"))
    import os

    os.replace(tmp, path)
    print(f"{path}: kept {keep} of {len(lines)} lines "
          f"(dropped {len(lines) - keep})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
