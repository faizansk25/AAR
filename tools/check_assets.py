"""Dev utility: validate that the repository's SVG assets are well-formed.

An SVG that is not parseable renders as a broken image on the README, and the
failure is silent - the page still loads. This makes it loud instead.
"""
import glob
import sys
import xml.etree.ElementTree as ET


def main() -> int:
    paths = sorted(glob.glob("*.svg") + glob.glob("docs/**/*.svg",
                                                 recursive=True))
    if not paths:
        print("no SVG assets found")
        return 0

    bad = 0
    for path in paths:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            bad += 1
            print(f"FAIL {path}: {exc}")
            continue
        if not root.tag.endswith("svg"):
            bad += 1
            print(f"FAIL {path}: root element is {root.tag}, not <svg>")
            continue
        # A logo that cannot adapt to a theme is a silent regression.
        text = open(path, encoding="utf-8").read()
        has_theme = "prefers-color-scheme" in text
        has_current = "currentColor" in text
        print(f"  OK {path}: viewBox={root.get('viewBox')} "
              f"theme={'media-query' if has_theme else 'currentColor only'}")
        if not (has_theme or has_current):
            bad += 1
            print(f"     FAIL {path}: no theme adaptation at all")

    print(f"\n{len(paths) - bad} of {len(paths)} SVG asset(s) valid")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
