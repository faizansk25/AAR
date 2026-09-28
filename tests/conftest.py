"""Shared fixtures.

Tests run against the source tree, so a fresh checkout needs no install step.
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Keep every test run hermetic: never read or write the developer's real
# calibration profile or AAR home.
os.environ.setdefault("AAR_HOME", os.path.join(_ROOT, ".pytest-aar-home"))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long-running test")
    config.addinivalue_line("markers", "gpu: requires a CUDA GPU")


@pytest.fixture(scope="session")
def arrow():
    """pyarrow, or skip. Many AAR behaviours are Arrow-shaped by design."""
    return pytest.importorskip("pyarrow")


@pytest.fixture(scope="session")
def sample_table(arrow):
    """A small, deterministic analytical table."""
    import pyarrow as pa

    return pa.table({
        "customer_id": pa.array([1, 2, 3, 4, 5, 6, 7, 8], type=pa.int64()),
        "region": pa.array(["NA", "EU", "NA", "APAC", "EU", "NA", "APAC", "EU"],
                           type=pa.string()),
        "amount": pa.array([100.5, 250.0, 75.25, 900.0, 310.5, 42.0, 610.0, 88.0],
                           type=pa.float64()),
        "ts": pa.array([1_700_000_000 + i * 86_400 for i in range(8)],
                       type=pa.int64()),
    })


@pytest.fixture()
def tmp_profile(tmp_path):
    """An isolated calibration profile path."""
    return str(tmp_path / "profile.json")
