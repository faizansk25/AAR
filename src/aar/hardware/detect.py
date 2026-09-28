"""Hardware detection.

Pure standard library. Every probe is wrapped so that a platform quirk
degrades the *fingerprint*, never the import. A machine with no NVML, an
unreadable ``/proc``, or a locked-down container still produces a valid
fingerprint with the unknown fields explicitly ``None`` and a recorded
reason - because "we could not measure this" and "we assumed this" must be
distinguishable in the explain output.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CpuInfo", "MemoryInfo", "GpuInfo", "StorageInfo", "NetworkInfo",
    "OsInfo", "SoftwareInfo", "probe_cpu", "probe_memory", "probe_gpu",
    "probe_storage", "probe_network", "probe_os", "probe_software",
    "bytes_from_human", "human_bytes",
]

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:  # noqa: BLE001
    psutil = None  # type: ignore
    _HAS_PSUTIL = False


def bytes_from_human(text: str) -> int | None:
    """Parse ``'32 GB'`` / ``'3200 MB/s'`` / ``'8GiB'`` into bytes."""
    m = re.search(r"([\d.]+)\s*([KMGT]?i?B)", text, re.IGNORECASE)
    if not m:
        return None
    scale = {"": 1, "B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12,
             "KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40}
    key = m.group(2).upper()
    factor = scale.get(key) or scale.get(key.replace("IB", "B")) or scale.get(key[0])
    return int(float(m.group(1)) * (factor or 1))


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1000 or unit == "PB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1000.0
    return f"{n:.1f}PB"  # pragma: no cover


@dataclass(slots=True)
class CpuInfo:
    model: str = "unknown"
    physical_cores: int = 1
    logical_cores: int = 1
    avx2: bool = False
    avx512: bool = False
    avx: bool = False
    numa_nodes: int = 1
    cache_l3_bytes: int | None = None
    max_freq_mhz: float | None = None
    architecture: str = ""
    #: What we could not determine, and why. Surfaced in `aar explain`.
    unknowns: list[str] = field(default_factory=list)

    @property
    def simd_level(self) -> str:
        if self.avx512:
            return "avx512"
        if self.avx2:
            return "avx2"
        if self.avx:
            return "avx"
        return "scalar"

    def render(self) -> str:
        extras = [f"avx={self.simd_level}"]
        if self.cache_l3_bytes:
            extras.append(f"L3={human_bytes(self.cache_l3_bytes)}")
        if self.numa_nodes > 1:
            extras.append(f"numa={self.numa_nodes}")
        if self.unknowns:
            extras.append("unmeasured: " + ",".join(self.unknowns))
        return (f"{self.model} | {self.physical_cores}P/{self.logical_cores}L | "
                + " | ".join(extras))


@dataclass(slots=True)
class MemoryInfo:
    total_bytes: int = 0
    available_bytes: int = 0
    swap_total_bytes: int = 0
    #: Measured achievable bandwidth in bytes/sec, or None before calibration.
    bandwidth_bytes_s: float | None = None
    unknowns: list[str] = field(default_factory=list)

    def render(self) -> str:
        bw = (f"{self.bandwidth_bytes_s / 1e9:.1f} GB/s"
              if self.bandwidth_bytes_s else "bandwidth unmeasured")
        return (f"{human_bytes(self.total_bytes)} total, "
                f"{human_bytes(self.available_bytes)} available, {bw}")


@dataclass(slots=True)
class GpuInfo:
    available: bool = False
    vendor: str | None = None
    model: str | None = None
    device_index: int = 0
    compute_capability: tuple[int, int] | None = None
    vram_bytes: int = 0
    free_vram_bytes: int = 0
    driver_version: str | None = None
    cuda_version: str | None = None
    device_count: int = 0
    uvm_available: bool = False
    nvlink: bool = False
    #: Why we believe there is no GPU. Never empty - a missing GPU must be an
    #: explained decision, not an absence of information.
    reason: str = "not probed"
    unknowns: list[str] = field(default_factory=list)

    def render(self) -> str:
        if not self.available:
            return f"no GPU ({self.reason})"
        cap = (f" sm_{self.compute_capability[0]}{self.compute_capability[1]}"
               if self.compute_capability else "")
        link = ", NVLink" if self.nvlink else ""
        return (f"{self.vendor} {self.model}{cap} | {human_bytes(self.vram_bytes)} VRAM "
                f"({human_bytes(self.free_vram_bytes)} free){link} | "
                f"driver {self.driver_version or '?'}, CUDA {self.cuda_version or '?'}")


@dataclass(slots=True)
class StorageInfo:
    media_type: str = "unknown"
    sequential_read_b_s: float | None = None
    sequential_write_b_s: float | None = None
    random_read_iops: int | None = None
    filesystem: str = "unknown"
    free_bytes: int | None = None
    total_bytes: int | None = None
    unknowns: list[str] = field(default_factory=list)

    def render(self) -> str:
        r = (f"{self.sequential_read_b_s / 1e6:.0f} MB/s read"
             if self.sequential_read_b_s else "read bw unmeasured")
        w = (f"{self.sequential_write_b_s / 1e6:.0f} MB/s write"
             if self.sequential_write_b_s else "write bw unmeasured")
        return f"{self.media_type} ({self.filesystem}) | {r} | {w}"


@dataclass(slots=True)
class NetworkInfo:
    bandwidth_bytes_s: float | None = None
    latency_ms: float | None = None
    rdma: bool = False
    #: The one network fact AAR always has: it does not phone home.
    egress_default: str = "deny"
    unknowns: list[str] = field(default_factory=list)

    def render(self) -> str:
        bw = (f"{self.bandwidth_bytes_s / 1e9:.1f} GB/s"
              if self.bandwidth_bytes_s else "bandwidth unmeasured")
        lat = (f"{self.latency_ms:.1f} ms" if self.latency_ms
               else "latency unmeasured")
        rdma = ", RDMA" if self.rdma else ""
        return f"{bw} | RTT {lat}{rdma} | egress {self.egress_default}"


@dataclass(slots=True)
class OsInfo:
    system: str = "unknown"
    release: str = ""
    version: str = ""
    architecture: str = ""
    container: str | None = None
    cgroup_memory_limit: int | None = None
    cgroup_cpu_limit: float | None = None
    unknowns: list[str] = field(default_factory=list)

    def render(self) -> str:
        bits = [f"{self.system} {self.release}".strip()]
        if self.container:
            bits.append(f"container={self.container}")
        if self.cgroup_memory_limit:
            bits.append(f"cgroup-mem={human_bytes(self.cgroup_memory_limit)}")
        if self.cgroup_cpu_limit:
            bits.append(f"cgroup-cpu={self.cgroup_cpu_limit:g}")
        return " | ".join(bits)


@dataclass(slots=True)
class SoftwareInfo:
    python: str = ""
    arrow: str | None = None
    duckdb: str | None = None
    polars: str | None = None
    pandas: str | None = None
    cudf: str | None = None
    nvml: str | None = None
    openpyxl: str | None = None
    ray: str | None = None
    dask: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        present = [k for k, v in (
            ("pyarrow", self.arrow), ("duckdb", self.duckdb),
            ("polars", self.polars), ("pandas", self.pandas),
            ("cudf", self.cudf), ("pynvml", self.nvml),
            ("openpyxl", self.openpyxl), ("ray", self.ray), ("dask", self.dask),
        ) if v]
        return f"python {self.python} | installed: {', '.join(present) or 'stdlib only'}"



# ------------------------------------------------------------------ probes
def probe_os() -> OsInfo:
    """Identify the OS, architecture and any container/cgroup limits.

    Container limits matter: inside a 4-core, 8 GiB cgroup the *host* CPU
    count is a lie, and planning against it produces OOM.
    """
    info = OsInfo(
        system=platform.system() or "unknown",
        release=platform.release(),
        version=platform.version(),
        architecture=platform.machine(),
    )
    if os.path.exists("/.dockerenv"):
        info.container = "docker"
    elif os.environ.get("KUBERNETES_SERVICE_HOST"):
        info.container = "kubernetes"
    else:
        try:
            with open("/proc/1/cgroup", encoding="utf-8") as fh:
                body = fh.read()
            for kind in ("docker", "containerd", "kubepods", "lxc"):
                if kind in body:
                    info.container = kind
                    break
        except OSError:
            pass

    # cgroup v2 then v1 memory limit
    for path, parser in (
        ("/sys/fs/cgroup/memory.max", _cgroup_v2_limit),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", _cgroup_v1_limit),
    ):
        try:
            with open(path, encoding="utf-8") as fh:
                raw = fh.read().strip()
            if raw and raw != "max":
                val = parser(raw)
                if val and 0 < val < (1 << 62):
                    info.cgroup_memory_limit = val
                    break
        except OSError:
            continue
    else:
        if _HAS_PSUTIL:
            try:
                lim = psutil.virtual_memory().total  # type: ignore[union-attr]
                info.cgroup_memory_limit = int(lim)
            except Exception:  # noqa: BLE001
                pass

    for path in ("/sys/fs/cgroup/cpu.max",
                 "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            with open(path, encoding="utf-8") as fh:
                raw = fh.read().split()
            if path.endswith("cpu.max") and len(raw) == 2:
                info.cgroup_cpu_limit = min(float(raw[0]) / float(raw[1]), 1e6)
            elif len(raw) == 1:
                info.cgroup_cpu_limit = float(raw[0]) / 1e6
            break
        except (OSError, ValueError):
            continue
    return info


def _cgroup_v2_limit(raw: str) -> int | None:
    try:
        return int(raw)
    except ValueError:
        return None


def _cgroup_v1_limit(raw: str) -> int | None:
    try:
        val = int(raw)
    except ValueError:
        return None
    # v1 uses a huge sentinel to mean "unlimited".
    return None if val >= (1 << 62) else val


def probe_cpu() -> CpuInfo:
    """Detect CPU model, core topology and SIMD capability.

    SIMD is measured where the platform exposes it, and inferred (and labelled
    as inferred) where it does not. The same model number ships with and
    without AVX-512 depending on the SKU, and the disabled case is exactly
    where a hardcoded "x86 has AVX2" rule produces wrong kernel timings.
    """
    system = platform.system()
    info = CpuInfo(
        logical_cores=os.cpu_count() or 1,
        architecture=platform.machine(),
    )
    model = platform.processor()
    flags: list[str] = []

    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                body = fh.read()
            m = re.search(r"^model name\s*:\s*(.+)$", body, re.MULTILINE)
            if m:
                model = m.group(1).strip()
            fm = re.search(r"^flags\s*:\s*(.+)$", body, re.MULTILINE)
            if fm:
                flags = fm.group(1).split()
            cache = re.search(r"^cache size\s*:\s*(.+)$", body, re.MULTILINE)
            if cache:
                info.cache_l3_bytes = bytes_from_human(cache.group(1))
            freq = re.search(r"^cpu MHz\s*:\s*([\d.]+)", body, re.MULTILINE)
            if freq:
                info.max_freq_mhz = float(freq.group(1))
        except OSError as exc:
            info.unknowns.append(f"cpuinfo: {exc}")
    elif platform.system() == "Windows":
        model = _win_facts().get("cpu_name") or model
    elif platform.system() == "Darwin":
        model = _sysctl_str("machdep.cpu.brand_string") or model
        info.cache_l3_bytes = bytes_from_human(
            _sysctl_str("machdep.cpu.l3cachesize") or "")
        info.avx2 = True  # every Apple Silicon and recent Intel Mac has it
        info.avx = True
        info.unknowns.append("avx512: not applicable on this architecture")

    if not model or (model == platform.processor()
                     and platform.system() == "Windows"):
        model = _wmic_cpu() or model or "unknown"
    info.model = str(model).strip() or "unknown"

    if system == "Windows":
        facts = _win_facts()
        if facts.get("arch"):
            fset = {str(facts["arch"]).lower()}
            # Get-CimInstance never reports the feature set, so SIMD is an
            # inference. It is labelled as such in `unknowns` rather than
            # presented as a measurement - an assumed AVX2 on a SKU without
            # it is exactly the silent-wrongness calibration exists to prevent.
            info.avx = True
            info.avx2 = True
            info.unknowns.append(
                "SIMD level inferred from architecture, not measured")
        if facts.get("max_mhz"):
            try:
                info.max_freq_mhz = float(facts["max_mhz"])
            except (TypeError, ValueError):
                pass
        if facts.get("l3"):
            try:
                kb = float(facts["l3"])
                info.cache_l3_bytes = int(kb * 1024) if kb < 1_000_000 else int(kb)
            except (TypeError, ValueError):
                pass
    elif flags:
        fset = {f.lower() for f in flags}
        info.avx512 = any(f.startswith("avx512") for f in fset)
        info.avx2 = "avx2" in fset
        info.avx = "avx" in fset
    elif platform.system() == "Windows":
        info.unknowns.append("cpu flags not readable; SIMD level assumed scalar")
    else:
        info.unknowns.append("cpu flags not exposed by this platform")

    info.physical_cores = _physical_cores()
    info.logical_cores = os.cpu_count() or 1
    info.numa_nodes = _numa_nodes()

    if info.max_freq_mhz is None and _HAS_PSUTIL:
        try:
            freq = psutil.cpu_freq()  # type: ignore[union-attr]
            if freq and freq.max:
                info.max_freq_mhz = float(freq.max)
        except Exception:  # noqa: BLE001
            pass
    return info



def _sysctl_str(key: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True,
                             text=True, timeout=3, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _win_facts() -> dict[str, Any]:
    """Collect every Windows hardware fact in one PowerShell round trip.

    Each ``powershell.exe`` launch costs 1-3 seconds. Querying CPU, memory
    and disk separately turned every probe into a multi-second stall, which
    is unacceptable for a UI that redraws a resource panel. One command with
    a JSON payload replaces four.
    """
    script = (
        "$ErrorActionPreference='SilentlyContinue';"
        "$cpu=Get-CimInstance Win32_Processor|Select-Object -First 1;"
        "$cs=Get-CimInstance Win32_ComputerSystem;"
        "$os=Get-CimInstance Win32_OperatingSystem;"
        "$vol=Get-Volume|Where-Object{$_.DriveLetter}|Select-Object -First 1;"
        "$pd=Get-PhysicalDisk|Select-Object -First 1;"
        "$d=[ordered]@{"
        "cpu_name=$cpu.Name;cores=$cpu.NumberOfCores;logical=$cpu.NumberOfLogicalProcessors;"
        "arch=$cpu.Architecture;max_mhz=$cpu.MaxClockSpeed;l2=$cpu.L2CacheSize;"
        "l3=$cpu.L3CacheSize;"
        "ram=$cs.TotalPhysicalMemory;avail_mb=$os.FreePhysicalMemory*1KB;"
        "vol_size=$vol.Size;vol_free=$vol.SizeRemaining;"
        "media=$pd.MediaType;bus=$pd.BusType}"
        "$d|ConvertTo-Json -Compress"
    )
    raw = _run_hidden(["powershell", "-NoProfile", "-NonInteractive",
                       "-Command", script], timeout=25)
    if not raw:
        return {}
    try:
        import json

        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}


def _physical_cores() -> int:
    system = platform.system()
    try:
        if system == "Windows":
            cores = _win_facts().get("cores")
            if cores:
                return int(cores)
        elif system == "Linux":
            ids = _linux_core_ids()
            if ids:
                return len(ids)
    except Exception:  # noqa: BLE001
        pass
    return os.cpu_count() or 1



def glob_paths(pattern: str) -> list[str]:
    import glob

    return glob.glob(pattern)


def _linux_core_ids() -> set[str]:
    out: set[str] = set()
    for entry in glob_paths("/sys/devices/system/cpu/cpu[0-9]*"):
        try:
            with open(os.path.join(entry, "topology/core_id"), encoding="utf-8") as fh:
                out.add(fh.read().strip())
        except OSError:
            continue
    return out


def _numa_nodes() -> int:
    try:
        nodes = glob_paths("/sys/devices/system/node/node[0-9]*")
        if nodes:
            return len(nodes)
    except Exception:  # noqa: BLE001
        pass
    return 1


def _run_hidden(cmd: list[str], timeout: int = 8) -> str | None:
    """Run a helper process with no window flash. Never raises.

    Results are memoised per command: spawning PowerShell costs 1-3 seconds
    on Windows, and the probes are called repeatedly (once per plan, per
    explain, per UI frame). Caching turns a multi-second stall into a dict
    lookup. ``clear_probe_cache()`` exists for tests and for the rare case
    where hardware state legitimately changes mid-process.
    """
    key = tuple(cmd)
    if key in _RUN_CACHE:
        return _RUN_CACHE[key]
    try:
        kwargs: dict[str, Any] = {
            "capture_output": True, "text": True, "timeout": timeout,
            "check": False,
        }
        if os.name == "nt":
            si = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
            kwargs["startupinfo"] = si
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.run(cmd, **kwargs)
        result = out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        result = None
    _RUN_CACHE[key] = result
    return result


#: Memoised helper-process output, keyed by command.
_RUN_CACHE: dict[tuple[str, ...], str | None] = {}


def clear_probe_cache() -> None:
    """Forget memoised helper-process output."""
    _RUN_CACHE.clear()



def _wmic_cpu() -> str | None:
    raw = _run_hidden(["wmic", "cpu", "get", "name"])
    if not raw:
        return None
    for line in raw.splitlines()[1:]:
        if line.strip():
            return line.strip()
    return None


def probe_memory() -> MemoryInfo:
    """Total, available and swap. Available is what the planner plans against."""
    info = MemoryInfo()
    if _HAS_PSUTIL:
        try:
            vm = psutil.virtual_memory()  # type: ignore[union-attr]
            info.total_bytes = int(vm.total)
            info.available_bytes = int(vm.available)
            sw = psutil.swap_memory()  # type: ignore[union-attr]
            info.swap_total_bytes = int(sw.total)
            return info
        except Exception:  # noqa: BLE001
            info.unknowns.append("psutil failed; falling back to os")
    system = platform.system()
    try:
        if system == "Linux":
            mem = {}
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    k, _, v = line.partition(":")
                    mem[k.strip()] = v.strip()
            info.total_bytes = int(mem.get("MemTotal", "0 kB").split()[0]) * 1024
            avail = mem.get("MemAvailable")
            if avail is None:
                free = int(mem.get("MemFree", "0 kB").split()[0])
                cached = int(mem.get("Cached", "0 kB").split()[0])
                avail = f"{free + cached} kB"
            info.available_bytes = int(avail.split()[0]) * 1024
            info.swap_total_bytes = int(
                mem.get("SwapTotal", "0 kB").split()[0]) * 1024
        elif system == "Windows":
            facts = _win_facts()
            if facts.get("ram"):
                try:
                    info.total_bytes = int(facts["ram"])
                except (TypeError, ValueError):
                    pass
            if facts.get("avail_mb"):
                try:
                    info.available_bytes = int(float(facts["avail_mb"]))
                except (TypeError, ValueError):
                    pass
        elif system == "Darwin":
            raw = _sysctl_str("hw.memsize")
            if raw and raw.isdigit():
                info.total_bytes = int(raw)
            raw = _run_vm_stat_free()
            if raw:
                page = 4096
                info.available_bytes = int(raw) * page
    except Exception as exc:  # noqa: BLE001
        info.unknowns.append(f"memory probe: {exc}")
    if not info.total_bytes:
        info.total_bytes = _fallback_total_memory()
    if not info.available_bytes:
        info.available_bytes = int(info.total_bytes * 0.6)
        info.unknowns.append("available memory not measurable; assumed 60% of total")
    return info


def _run_vm_stat_free() -> int | None:
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=3, check=False)
        m = re.search(r"Pages free:\s*(\d+)", out.stdout)
        return int(m.group(1)) if m else None
    except (OSError, subprocess.SubprocessError):
        return None


def _fallback_total_memory() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 8 * 1024**3


def probe_gpu() -> GpuInfo:
    """Detect accelerators, with a documented reason on failure.

    Order of evidence, strongest first:
      1. ``pynvml`` - the real NVIDIA management interface.
      2. ``nvidia-smi`` - present without the Python binding.
      3. Apple unified memory / ROCm / a CUDA runtime (``cudf``/``torch``).

    A failure records *why*, so the planner's "no GPU" decision is explainable
    rather than a bare default.
    """
    info = GpuInfo()
    if _probe_gpu_nvml(info):
        return info
    if _probe_gpu_smi(info):
        return info
    if _probe_gpu_apple(info):
        return info
    if _probe_gpu_vendor_runtime(info):
        return info
    info.available = False
    info.reason = _no_gpu_reason()
    return info


def _probe_gpu_nvml(info: GpuInfo) -> bool:
    try:
        import pynvml  # type: ignore
    except Exception:  # noqa: BLE001
        info.unknowns.append("pynvml not installed")
        return False
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001
        info.unknowns.append(f"nvmlInit failed: {exc}")
        return False
    try:
        count = int(pynvml.nvmlDeviceGetCount())
        info.device_count = count
        if count == 0:
            info.reason = "NVML reports zero devices"
            return False
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        info.model = name.decode() if isinstance(name, bytes) else str(name)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        info.vram_bytes = int(mem.total)
        info.free_vram_bytes = int(mem.free)
        ver = pynvml.nvmlSystemGetDriverVersion()
        info.driver_version = ver.decode() if isinstance(ver, bytes) else str(ver)
        try:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
            info.compute_capability = (int(major), int(minor))
        except Exception:  # noqa: BLE001
            info.unknowns.append("compute capability unavailable")
        try:
            info.cuda_version = _cuda_version_string(pynvml)
        except Exception:  # noqa: BLE001
            info.unknowns.append("CUDA driver version unavailable")
        info.uvm_available = True
        info.vendor = "NVIDIA"
        info.available = True
        info.reason = "detected via NVML"
        if count > 1:
            info.reason += f" ({count} devices; planning against device 0)"
        return True
    except Exception as exc:  # noqa: BLE001
        info.unknowns.append(f"NVML query failed: {exc}")
        return False
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def _cuda_version_string(pynvml: Any) -> str:
    raw = pynvml.nvmlSystemGetCudaDriverVersion()
    if isinstance(raw, tuple):
        major, minor = raw
    elif isinstance(raw, int):
        major, minor = raw // 1000, (raw % 1000) // 10
    else:  # pragma: no cover - driver dependent
        return str(raw)
    return f"{major}.{minor}"


def _probe_gpu_smi(info: GpuInfo) -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        info.unknowns.append("nvidia-smi not on PATH")
        return False
    raw = _run_hidden([exe, "--query-gpu=name,memory.total,driver_version,"
                              "compute_cap", "--format=csv,noheader,nounits"], 10)
    if not raw or "No devices" in raw:
        info.unknowns.append("nvidia-smi reported no devices")
        return False
    parts = [p.strip() for p in raw.splitlines()[0].split(",")]
    info.available = True
    info.vendor = "NVIDIA"
    info.reason = "detected via nvidia-smi (NVML binding absent)"
    if parts:
        info.model = parts[0]
    if len(parts) > 1:
        try:
            info.vram_bytes = int(float(parts[1]) * 1024**2)
            info.free_vram_bytes = int(info.vram_bytes * 0.8)
        except ValueError:
            pass
    if len(parts) > 2:
        info.driver_version = parts[2]
    if len(parts) > 3 and "." in parts[3]:
        try:
            maj, minr = parts[3].split(".")
            info.compute_capability = (int(maj), int(minr))
        except ValueError:
            pass
    listing = _run_hidden([exe, "-L"], 10) or ""
    info.device_count = max(1, len([ln for ln in listing.splitlines()
                                    if ln.startswith("GPU ")]))
    return True


def _probe_gpu_apple(info: GpuInfo) -> bool:
    if platform.system() != "Darwin":
        return False
    brand = _sysctl_str("machdep.cpu.brand_string") or ""
    if "Apple" not in brand:
        return False
    mem = probe_memory()
    # Apple Silicon exposes a single unified pool. Reporting a fraction as
    # "VRAM" is the closest honest analogue of what an accelerator can address.
    info.available = True
    info.vendor = "Apple"
    info.model = brand
    info.vram_bytes = int(mem.total_bytes * 0.75)
    info.free_vram_bytes = int(mem.available_bytes * 0.75)
    info.device_count = 1
    info.reason = "Apple unified memory; VRAM modelled as a 75% share of the pool"
    info.unknowns.append("no CUDA; GPU paths require a Metal-capable engine")
    return True


def _probe_gpu_vendor_runtime(info: GpuInfo) -> bool:
    import importlib

    for module, label in (("cudf", "cuDF"), ("torch", "PyTorch"), ("cupy", "CuPy")):
        try:
            mod = importlib.import_module(module)
        except Exception:  # noqa: BLE001
            continue
        try:
            count = mod.cuda.device_count()
        except Exception:  # noqa: BLE001
            continue
        if not count:
            continue
        info.available = True
        info.vendor = "NVIDIA"
        info.model = f"detected via {label}"
        info.device_count = int(count)
        props = mod.cuda.get_device_properties(0)
        info.vram_bytes = int(getattr(props, "total_memory", 0))
        info.free_vram_bytes = int(info.vram_bytes * 0.8)
        major = getattr(props, "major", None)
        if major is not None:
            info.compute_capability = (int(major), int(getattr(props, "minor", 0)))
        info.reason = f"detected via {label} runtime"
        return True
    if shutil.which("rocm-smi"):
        info.available = True
        info.vendor = "AMD"
        info.model = "AMD GPU (ROCm)"
        info.reason = "rocm-smi present; AAR ships no ROCm engine yet"
        return True
    return False


def _no_gpu_reason() -> str:
    """Best available explanation for GPU absence, in operator language."""
    for probe, label in (("nvidia-smi", "nvidia-smi"), ("rocm-smi", "rocm-smi")):
        if not shutil.which(probe):
            continue
        return f"{label} present but no usable device"
    if platform.system() == "Linux":
        if not glob_paths("/dev/nvidia*"):
            return "no /dev/nvidia* device nodes and no vendor tooling"
        return "nvidia device nodes present but no driver interface available"
    if platform.system() == "Darwin":
        return "no CUDA device on this platform"
    return "no accelerator driver or runtime detected"


def probe_storage(path: str | None = None) -> StorageInfo:
    """Media type and filesystem capacity. Bandwidth is measured, not guessed.

    Media type comes from the platform (rotational flag, NVMe name, bus
    type). Throughput is deliberately left ``None`` here and filled by
    :mod:`aar.hardware.calibrate`, because a value copied from a spec sheet
    is exactly the kind of assumption that makes a planner confidently wrong.
    """
    info = StorageInfo()
    target = os.path.abspath(path or os.getcwd())
    probe_path = target
    while probe_path and not os.path.exists(probe_path):
        parent = os.path.dirname(probe_path)
        if parent == probe_path:
            break
        probe_path = parent
    if not probe_path:
        probe_path = os.getcwd()

    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/mounts", encoding="utf-8") as fh:
                best, dev, fstype = "", "", ""
                for line in fh:
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    mount = parts[1]
                    if target.startswith(mount) and len(mount) >= len(best):
                        best, dev, fstype = mount, parts[0], parts[2]
                if best:
                    info.filesystem = fstype
                    name = dev.split("/")[-1]
                    if name.startswith("nvme"):
                        info.media_type = "NVMe"
                    else:
                        rot = os.path.join("/sys/block", name, "queue/rotational")
                        if os.path.exists(rot):
                            with open(rot, encoding="utf-8") as rfh:
                                info.media_type = (
                                    "HDD" if rfh.read().strip() == "1" else "SSD")
                        else:
                            info.media_type = "unknown"
        elif system == "Windows":
            facts = _win_facts()
            info.filesystem = "NTFS"
            media = str(facts.get("media", "")).lower()
            bus = str(facts.get("bus", "")).lower()
            if "nvme" in bus:
                info.media_type = "NVMe"
            elif "hdd" in media or "raid" in media:
                info.media_type = "HDD"
            elif "ssd" in media or "solid" in media:
                info.media_type = "SSD"
            else:
                info.media_type = "SSD (assumed; not queryable)"
                info.unknowns.append("media type not reported by this host")
            for key, attr in (("vol_size", "total_bytes"),
                              ("vol_free", "free_bytes")):
                if facts.get(key):
                    try:
                        setattr(info, attr, int(facts[key]))
                    except (TypeError, ValueError):
                        pass
        elif system == "Darwin":
            out = subprocess.run(["df", "-k", probe_path], capture_output=True,
                                 text=True, timeout=5, check=False)
            lines = out.stdout.strip().splitlines()
            if len(lines) >= 2:
                parts = lines[-1].split()
                info.total_bytes = int(parts[1]) * 1024
                info.free_bytes = int(parts[3]) * 1024
                info.media_type = "SSD (assumed)"
        if _HAS_PSUTIL and info.total_bytes is None:
            du = psutil.disk_usage(probe_path)  # type: ignore[union-attr]
            info.total_bytes, info.free_bytes = int(du.total), int(du.free)
    except Exception as exc:  # noqa: BLE001
        info.unknowns.append(f"storage probe: {exc}")
    return info


def _parse_json_media(raw: str) -> str:
    import json

    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return "unknown"
    media = str(data.get("MediaType", "")).lower()
    bus = str(data.get("BusType", "")).lower()
    if "nvme" in bus:
        return "NVMe"
    if "hdd" in media or "raid" in media:
        return "HDD"
    if "ssd" in media or "solid" in media:
        return "SSD"
    return "unknown"


def probe_network() -> NetworkInfo:
    """Network characteristics plus AAR's egress posture.

    ``egress_default`` is ``"deny"`` by construction. AAR never resolves or
    contacts an external service on its own initiative; the only hosts it will
    contact are ones an administrator has added to the allow-list. The probe
    reports the configured posture rather than testing connectivity, because
    testing connectivity would itself be an outbound action.
    """
    info = NetworkInfo()
    if platform.system() == "Linux":
        for nic in glob_paths("/sys/class/net/*"):
            try:
                with open(os.path.join(nic, "speed"), encoding="utf-8") as fh:
                    mbps = int(fh.read().strip())
                if mbps > 0:
                    info.bandwidth_bytes_s = mbps * 10**6 / 8
                    break
            except (OSError, ValueError):
                continue
    elif platform.system() == "Windows":
        raw = _run_hidden([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_NetworkAdapter | Where-Object {$_.Speed} | "
            "Select-Object -First 1).Speed"], 12)
        if raw and raw.strip().isdigit():
            mbps = int(raw.strip())
            if 0 < mbps < 10**7:
                info.bandwidth_bytes_s = mbps * 10**6 / 8
    if info.bandwidth_bytes_s is None:
        info.unknowns.append("link bandwidth not queryable; assumed 1 GB/s")
        info.bandwidth_bytes_s = 1.25e8
    return info


def probe_software() -> SoftwareInfo:
    """Versions of every optional engine, without importing the heavy ones."""
    import importlib
    from importlib import metadata

    info = SoftwareInfo(python=platform.python_version())
    for attr, dist in (
        ("arrow", "pyarrow"), ("duckdb", "duckdb"), ("polars", "polars"),
        ("pandas", "pandas"), ("cudf", "cudf-cu12"), ("nvml", "pynvml"),
        ("openpyxl", "openpyxl"), ("ray", "ray"), ("dask", "dask"),
    ):
        try:
            setattr(info, attr, metadata.version(dist))
        except Exception:  # noqa: BLE001
            try:
                mod = importlib.import_module(
                    {"arrow": "pyarrow", "nvml": "pynvml"}.get(attr, attr))
                setattr(info, attr, getattr(mod, "__version__", "present"))
            except Exception:  # noqa: BLE001
                setattr(info, attr, None)
    return info


# ------------------------------------------------------------------ facade
class HardwareProfile:
    """One coherent snapshot of the machine, computed once.

    The planner needs all six probes together and asks for them repeatedly -
    once per plan, per explain render, per workbench frame. Probing lazily
    and memoising here means the first call pays the cost (at most one
    PowerShell launch on Windows) and every later call is free.
    """

    __slots__ = ("_os", "_cpu", "_memory", "_gpu", "_storage", "_network",
                 "_software")

    def __init__(self) -> None:
        self._os: OsInfo | None = None
        self._cpu: CpuInfo | None = None
        self._memory: MemoryInfo | None = None
        self._gpu: GpuInfo | None = None
        self._storage: StorageInfo | None = None
        self._network: NetworkInfo | None = None
        self._software: SoftwareInfo | None = None

    @property
    def os(self) -> OsInfo:
        if self._os is None:
            self._os = probe_os()
        return self._os

    @property
    def cpu(self) -> CpuInfo:
        if self._cpu is None:
            self._cpu = probe_cpu()
        return self._cpu

    @property
    def memory(self) -> MemoryInfo:
        if self._memory is None:
            self._memory = probe_memory()
        return self._memory

    @property
    def gpu(self) -> GpuInfo:
        if self._gpu is None:
            self._gpu = probe_gpu()
        return self._gpu

    @property
    def storage(self) -> StorageInfo:
        if self._storage is None:
            self._storage = probe_storage()
        return self._storage

    @property
    def network(self) -> NetworkInfo:
        if self._network is None:
            self._network = probe_network()
        return self._network

    @property
    def software(self) -> SoftwareInfo:
        if self._software is None:
            self._software = probe_software()
        return self._software

    @property
    def has_gpu(self) -> bool:
        return self.gpu.available

    @property
    def memory_budget_bytes(self) -> int:
        """What the planner may plan against.

        Available memory, not total, and further reduced inside a container
        where the host's total is meaningless. Overcommitting here is how a
        plan becomes an OOM at run time.
        """
        mem = self.memory.available_bytes
        limit = self.os.cgroup_memory_limit
        if limit:
            mem = min(mem, limit)
        # Leave headroom for the Python process itself and the OS page cache.
        return int(mem * 0.8)

    @property
    def vram_budget_bytes(self) -> int:
        gpu = self.gpu
        if not gpu.available or not gpu.vram_bytes:
            return 0
        return int(min(gpu.free_vram_bytes or gpu.vram_bytes,
                       gpu.vram_bytes) * 0.85)

    def fingerprint(self) -> str:
        """Stable identity for this machine's measurements.

        Deliberately excludes free/available quantities, which change minute to
        minute; a profile keyed on them would be discarded on every reload.
        It includes free VRAM because a driver upgrade genuinely changes
        what the GPU can do.
        """
        parts = [
            self.os.system, self.os.release, self.os.architecture,
            self.cpu.model, str(self.cpu.physical_cores),
            str(self.cpu.logical_cores), self.cpu.simd_level,
            self.software.python or "",
            self.software.arrow or "", self.software.duckdb or "",
            self.software.polars or "", self.software.cudf or "",
            f"{self.gpu.vendor}:{self.gpu.model}:{self.gpu.vram_bytes}",
        ]
        blob = "|".join(parts)
        import hashlib

        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def render(self) -> str:
        """Operator-facing summary; the body of ``aar doctor``."""
        lines = [
            "HARDWARE PROFILE",
            f"  os        {self.os.render()}",
            f"  cpu       {self.cpu.render()}",
            f"  memory    {self.memory.render()}",
            f"  gpu       {self.gpu.render()}",
            f"  storage   {self.storage.render()}",
            f"  network   {self.network.render()}",
            f"  software  {self.software.render()}",
            f"  budgets   memory={human_bytes(self.memory_budget_bytes)}"
            f"  vram={human_bytes(self.vram_budget_bytes)}",
            f"  id        {self.fingerprint()}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint(),
            "os": self.os.__dict__ if hasattr(self.os, "__dict__") else {},
            "cpu": self.cpu.render(),
            "memory_gb": round(self.memory.total_bytes / 1e9, 2),
            "memory_available_gb": round(self.memory.available_bytes / 1e9, 2),
            "gpu": self.gpu.render(),
            "storage": self.storage.render(),
            "network": self.network.render(),
            "software": self.software.render(),
        }


