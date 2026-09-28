"""Hardware profiling: detection, fingerprinting and microbenchmark calibration.

The eighth design principle is *empirical, not hardcoded*. That principle is
implemented here: AAR measures the machine it is running on rather than
trusting a lookup table, because two machines with identical spec sheets
behave differently and a spec sheet is never present at runtime.
"""

from .detect import (  # noqa: F401
    CpuInfo, GpuInfo, HardwareProfile, MemoryInfo, NetworkInfo, OsInfo,
    SoftwareInfo, StorageInfo, bytes_from_human, clear_probe_cache,
    human_bytes, probe_cpu, probe_gpu, probe_memory, probe_network, probe_os,
    probe_software, probe_storage,
)

__all__ = [
    "CpuInfo", "GpuInfo", "HardwareProfile", "MemoryInfo", "NetworkInfo",
    "OsInfo", "SoftwareInfo", "StorageInfo", "bytes_from_human",
    "clear_probe_cache", "human_bytes", "probe_cpu", "probe_gpu",
    "probe_memory", "probe_network", "probe_os", "probe_software",
    "probe_storage",
]

