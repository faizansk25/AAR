"""Trim the stray markdown tail that a bad edit left at the end of a file.

Usage: python tools/trim.py <path> <first_line_to_keep_inclusive>
Keeps lines 1..N (1-based) and discards the rest.
"""
import sys

path, n = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as fh:
    lines = fh.readlines()
with open(path, "w", encoding="utf-8") as fh:
    fh.writelines(lines[:n])
print(f"kept {n} lines of {len(lines)}")
