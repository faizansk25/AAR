"""Fetch real, public datasets for the audit. Not part of the package.

AAR is specified as air-gapped, so nothing in `src/aar` may reach the
network. This script is the *harness* reaching out to obtain real data; the
distinction matters and is why the fetch lives here rather than in a
connector.

Every file records where it came from, when, and how big it is, because an
audit result is only as good as the provenance of what it measured.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data")
USER_AGENT = "AAR-audit/1.0 (local verification harness)"

#: Real, public, no-authentication datasets chosen for shape rather than
#: subject: a large fact table, a wide table with text, and a CSV fallback.
SOURCES = [
    {
        "name": "nyc_taxi_2023_01.parquet",
        "url": "https://d37ci6vzurychx.cloudfront.net/trip-data/"
               "yellow_tripdata_2023-01.parquet",
        "kind": "parquet",
        "expect_mb": 30.0,
    },
    {
        "name": "nyc_taxi_2022_03.parquet",
        "url": "https://d37ci6vzurychx.cloudfront.net/trip-data/"
               "yellow_tripdata_2022-03.parquet",
        "kind": "parquet",
        "expect_mb": 30.0,
    },
    {
        "name": "nyc_311.csv",
        "url": "https://data.cityofnewyork.us/api/views/erm2-nwe9/rows.csv"
               "?accessType=DOWNLOAD",
        "kind": "csv",
        "expect_mb": 20.0,
    },
]


def _download(url: str, destination: str, timeout: int = 300) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    written = 0
    with urllib.request.urlopen(request, timeout=timeout) as response, \
            open(destination, "wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
    return written


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    os.makedirs(DATA_DIR, exist_ok=True)
    manifest_path = os.path.join(DATA_DIR, "MANIFEST.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)

    for source in SOURCES:
        path = os.path.join(DATA_DIR, source["name"])
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
            print(f"have   {source['name']} "
                  f"({os.path.getsize(path) / 1e6:.1f} MB)")
            continue
        print(f"fetch  {source['name']} <- {source['url']}")
        started = time.time()
        try:
            size = _download(source["url"], path)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {type(exc).__name__}: {exc}")
            continue
        elapsed = time.time() - started
        print(f"  ok   {size / 1e6:.1f} MB in {elapsed:.1f}s")
        manifest[source["name"]] = {
            "url": source["url"],
            "kind": source["kind"],
            "bytes": size,
            "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sha256": _sha256(path),
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    print("\nmanifest:")
    for name, info in sorted(manifest.items()):
        print(f"  {name:32s} {info['bytes'] / 1e6:7.1f} MB  "
              f"{info['sha256'][:16]}  {info['fetched']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
