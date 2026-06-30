"""System & hardware telemetry for the dashboard.

Reads CPU / memory / disk via stdlib (with psutil as an optional accelerator)
and queries multiple GPU sources so the dashboard makes sense on every host:

* NVIDIA on any platform via `nvidia-smi` (the original path).
* Windows AMD / Intel precise metrics via optional `nwinfo --gpu`.
* Windows AMD / Intel / fallback GPU identity via CIM (`Win32_VideoController`).
* macOS Apple Silicon / Intel GPUs via `system_profiler -json SPDisplaysDataType`.
* AMD on Linux via `rocm-smi` / bundled `amdgpu_top`, falling back to DRM/sysfs.
* Linux AMD / Intel iGPUs via `/sys/class/drm` vendor IDs and `hwmon` sensors.

All probes are best-effort: a missing tool, a parse error or a subprocess
timeout downgrades a field rather than failing the snapshot, so the API
endpoint always has *something* to render.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from lorahub.api.system_stats_types import (
    BatteryStats,
    CpuStats,
    DiskIoDevice,
    DiskIoStats,
    DiskUsage,
    GpuProcessInfo,
    GpuStats,
    HostInfo,
    InterfaceAddress,
    MemoryStats,
    NetworkInterfaceStats,
    NetworkStats,
    ProcessInfo,
    PublicIpInfo,
    SystemSnapshot,
    TcpConnectionStats,
    network_to_dict,
)

try:  # optional - gives us per-cpu utilization, load, swap, freq, temps, battery.
    import psutil

    _HAS_PSUTIL = True
except Exception:  # noqa: BLE001
    psutil = None
    _HAS_PSUTIL = False


# Cache external-tool lookups - spawning these on Windows is ~80ms each.
_NVIDIA_SMI: str | None = None
_NVIDIA_SMI_PROBED = False
_SYSTEM_PROFILER: str | None = None
_SYSTEM_PROFILER_PROBED = False
_ROCM_SMI: str | None = None
_ROCM_SMI_PROBED = False
_NWINFO: str | None = None
_NWINFO_PROBED = False
_AMDGPU_TOP: str | None = None
_AMDGPU_TOP_PROBED = False


def _run_hidden(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    if platform.system() == "Windows":
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)  # type: ignore[attr-defined]
    return subprocess.run(cmd, **kwargs)


def _find_nvidia_smi() -> str | None:
    global _NVIDIA_SMI, _NVIDIA_SMI_PROBED
    if _NVIDIA_SMI_PROBED:
        return _NVIDIA_SMI
    _NVIDIA_SMI_PROBED = True
    candidate = shutil.which("nvidia-smi")
    if candidate:
        _NVIDIA_SMI = candidate
        return candidate
    # On Windows, nvidia-smi often lives in System32 even when not on PATH.
    if platform.system() == "Windows":
        sysroot = os.environ.get("SYSTEMROOT") or r"C:\Windows"
        fallback = Path(sysroot) / "System32" / "nvidia-smi.exe"
        if fallback.is_file():
            _NVIDIA_SMI = str(fallback)
            return _NVIDIA_SMI
    return None


def _find_system_profiler() -> str | None:
    global _SYSTEM_PROFILER, _SYSTEM_PROFILER_PROBED
    if _SYSTEM_PROFILER_PROBED:
        return _SYSTEM_PROFILER
    _SYSTEM_PROFILER_PROBED = True
    if platform.system() != "Darwin":
        return None
    candidate = shutil.which("system_profiler") or "/usr/sbin/system_profiler"
    if Path(candidate).is_file():
        _SYSTEM_PROFILER = candidate
        return candidate
    return None


def _find_rocm_smi() -> str | None:
    global _ROCM_SMI, _ROCM_SMI_PROBED
    if _ROCM_SMI_PROBED:
        return _ROCM_SMI
    _ROCM_SMI_PROBED = True
    if platform.system() != "Linux":
        return None
    candidate = shutil.which("rocm-smi") or "/opt/rocm/bin/rocm-smi"
    if Path(candidate).is_file():
        _ROCM_SMI = candidate
        return candidate
    return None


def _find_nwinfo() -> str | None:
    global _NWINFO, _NWINFO_PROBED
    if _NWINFO_PROBED:
        return _NWINFO
    _NWINFO_PROBED = True
    if platform.system() != "Windows":
        return None
    candidate = shutil.which("nwinfo") or shutil.which("nwinfo.exe")
    if candidate:
        _NWINFO = candidate
        return candidate
    root = Path(__file__).resolve().parents[2]
    for path in (
        root / ".lorahub" / "nwinfo" / "nwinfo.exe",
        root / "tools" / "nwinfo" / "nwinfo.exe",
        root / "bin" / "nwinfo.exe",
    ):
        if path.is_file():
            _NWINFO = str(path)
            return _NWINFO
    return None


def _find_amdgpu_top() -> str | None:
    global _AMDGPU_TOP, _AMDGPU_TOP_PROBED
    if _AMDGPU_TOP_PROBED:
        return _AMDGPU_TOP
    _AMDGPU_TOP_PROBED = True
    if platform.system() != "Linux":
        return None
    candidate = shutil.which("amdgpu_top")
    if candidate:
        _AMDGPU_TOP = candidate
        return candidate
    root = Path(__file__).resolve().parents[2]
    for path in (
        root / ".lorahub" / "amdgpu_top" / "amdgpu_top",
        root / "tools" / "amdgpu_top" / "amdgpu_top",
        root / "bin" / "amdgpu_top",
    ):
        if path.is_file():
            _AMDGPU_TOP = str(path)
            return _AMDGPU_TOP
    return None


# --------------------------------------------------------------------------- #
# Collectors                                                                  #
# --------------------------------------------------------------------------- #


def _collect_cpu_frequency() -> tuple[float | None, float | None, float | None, list[float]]:
    """Return (current_mean, min, max, per_core_currents).

    Per-core values come from `psutil.cpu_freq(percpu=True)` and we average
    them so the headline number is more representative than a single-shot
    `cpu_freq().current` (which often returns the base clock inside containers).
    Hosts where per-core readings are unavailable (commonly macOS) fall back
    to the single-value `cpu_freq()` reading and an empty per-core list.
    """
    if not _HAS_PSUTIL:
        return None, None, None, []

    per_core: list[float] = []
    try:
        per = psutil.cpu_freq(percpu=True)
    except (OSError, NotImplementedError, AttributeError):
        per = None
    if per:
        for entry in per:
            current = getattr(entry, "current", None)
            try:
                per_core.append(float(current))
            except (TypeError, ValueError):
                continue

    try:
        agg = psutil.cpu_freq()
    except (OSError, NotImplementedError, AttributeError):
        agg = None

    def _opt_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        # psutil reports 0.0 when min/max are not exposed - treat that as unknown.
        if f == 0.0:
            return None
        return f

    freq_min = _opt_float(getattr(agg, "min", None)) if agg is not None else None
    freq_max = _opt_float(getattr(agg, "max", None)) if agg is not None else None

    current_mean: float | None
    if per_core:
        current_mean = sum(per_core) / len(per_core)
    elif agg is not None:
        current_mean = _opt_float(getattr(agg, "current", None))
    else:
        current_mean = None

    return current_mean, freq_min, freq_max, per_core


_CPUINFO_MODEL_RE = re.compile(r"^\s*model name\s*:\s*(.+?)\s*$", re.MULTILINE)


def _collect_cpu_model() -> str:
    """Best-effort CPU brand string. Returns empty string on failure.

    Order: Linux ``/proc/cpuinfo`` -> macOS ``sysctl machdep.cpu.brand_string``
    -> ``platform.processor()`` (Windows / fallback).
    """
    system = platform.system()
    if system == "Linux":
        try:
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        if text:
            m = _CPUINFO_MODEL_RE.search(text)
            if m:
                return m.group(1).strip()
    if system == "Darwin":
        try:
            proc = _run_hidden(  # noqa: S603, S607
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=False,
                capture_output=True,
                text=True,
                timeout=1,
            )
        except (OSError, subprocess.TimeoutExpired):
            proc = None
        if proc is not None and proc.returncode == 0:
            value = (proc.stdout or "").strip()
            if value:
                return value
    # Windows + final fallback. platform.processor() is sometimes empty in venvs.
    fallback = platform.processor() or ""
    return fallback.strip()


def _collect_cpu_temperature() -> float | None:
    """Read CPU package temperature via psutil sensors (Linux only in practice)."""
    if not _HAS_PSUTIL:
        return None
    fn = getattr(psutil, "sensors_temperatures", None)
    if fn is None:
        return None
    try:
        sensors = fn()
    except (OSError, NotImplementedError, AttributeError):
        return None
    if not sensors:
        return None
    # Common chip names that expose CPU package temp on Linux.
    preferred = ("coretemp", "k10temp", "k8temp", "zenpower", "cpu_thermal", "acpitz")
    for key in preferred:
        entries = sensors.get(key)
        if not entries:
            continue
        for entry in entries:
            current = getattr(entry, "current", None)
            if current is not None:
                try:
                    return float(current)
                except (TypeError, ValueError):
                    continue
    # Fallback: pick the first sensor that has a numeric current value.
    for entries in sensors.values():
        for entry in entries:
            current = getattr(entry, "current", None)
            if current is None:
                continue
            try:
                return float(current)
            except (TypeError, ValueError):
                continue
    return None


def _collect_cpu() -> CpuStats:
    freq_mean, freq_min, freq_max, freq_per_core = _collect_cpu_frequency()
    model = _collect_cpu_model()
    if _HAS_PSUTIL:
        per = psutil.cpu_percent(interval=None, percpu=True)
        load: list[float] | None = None
        getloadavg = getattr(os, "getloadavg", None)
        if getloadavg is not None:
            try:
                load = list(getloadavg())  # 1/5/15 min - Linux + recent macOS.
            except OSError:
                load = None
        return CpuStats(
            cores_logical=psutil.cpu_count(logical=True) or os.cpu_count() or 0,
            cores_physical=psutil.cpu_count(logical=False),
            usage_percent=psutil.cpu_percent(interval=None),
            per_core_percent=[float(p) for p in per],
            load_average=load,
            arch=platform.machine(),
            model=model,
            frequency_mhz=freq_mean,
            frequency_min_mhz=freq_min,
            frequency_max_mhz=freq_max,
            frequency_per_core_mhz=freq_per_core,
            cpu_temperature_c=_collect_cpu_temperature(),
        )
    return CpuStats(
        cores_logical=os.cpu_count() or 0,
        cores_physical=None,
        usage_percent=None,
        per_core_percent=[],
        load_average=None,
        arch=platform.machine(),
        model=model,
        frequency_mhz=freq_mean,
        frequency_min_mhz=freq_min,
        frequency_max_mhz=freq_max,
        frequency_per_core_mhz=freq_per_core,
        cpu_temperature_c=None,
    )


def _collect_memory() -> MemoryStats:
    if _HAS_PSUTIL:
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        return MemoryStats(
            total_bytes=int(vm.total),
            used_bytes=int(vm.used),
            available_bytes=int(vm.available),
            percent=float(vm.percent),
            swap_total_bytes=int(sw.total),
            swap_used_bytes=int(sw.used),
        )
    # stdlib fallback - Linux /proc/meminfo only; on Windows return zeros.
    if platform.system() == "Linux":
        try:
            info: dict[str, int] = {}
            with Path("/proc/meminfo").open(encoding="utf-8") as fh:
                for line in fh:
                    key, _, rest = line.partition(":")
                    parts = rest.strip().split()
                    if parts and parts[-1].lower() == "kb":
                        info[key.strip()] = int(parts[0]) * 1024
            total = info.get("MemTotal", 0)
            available = info.get("MemAvailable", 0)
            used = max(total - available, 0)
            percent = (used / total * 100.0) if total else 0.0
            return MemoryStats(
                total_bytes=total,
                used_bytes=used,
                available_bytes=available,
                percent=percent,
            )
        except OSError:
            pass
    return MemoryStats(total_bytes=0, used_bytes=0, available_bytes=0, percent=0.0)


# Pseudo / virtual filesystems we never want to report as a "disk" - they
# are not backed by real storage so their used/free numbers are misleading.
_VIRTUAL_FSTYPES = frozenset(
    {
        "tmpfs",
        "devtmpfs",
        "overlay",
        "overlay2",
        "squashfs",
        "proc",
        "sysfs",
        "cgroup",
        "cgroup2",
        "autofs",
        "fusectl",
        "pstore",
        "efivarfs",
        "mqueue",
        "devpts",
        "binfmt_misc",
        "tracefs",
        "debugfs",
        "configfs",
        "hugetlbfs",
    }
)


def _iter_real_mounts() -> list[tuple[str, Path]]:
    """Return `(label, mount_point)` for every real (non-virtual) partition.

    Uses ``psutil.disk_partitions(all=False)`` so the kernel-level virtual
    filesystems are mostly skipped already; we still defensively filter the
    fstype against :data:`_VIRTUAL_FSTYPES` for cases where the host reports
    overlays / tmpfs as "physical" (Docker, WSL, snap mounts).
    """
    if not _HAS_PSUTIL:
        return []
    try:
        partitions = psutil.disk_partitions(all=False)
    except (OSError, RuntimeError):
        return []
    out: list[tuple[str, Path]] = []
    for part in partitions:
        fstype = (getattr(part, "fstype", "") or "").lower()
        if fstype in _VIRTUAL_FSTYPES:
            continue
        mount_raw = getattr(part, "mountpoint", None)
        if not mount_raw:
            continue
        try:
            mount = Path(mount_raw)
        except (TypeError, ValueError):
            continue
        out.append((mount.as_posix(), mount))
    return out


def _collect_disks(extra_paths: list[Path] | None = None) -> list[DiskUsage]:
    """Report cwd / home / extras *plus* every real mount point on the host."""
    seen: set[str] = set()
    out: list[DiskUsage] = []
    targets: list[tuple[str, Path]] = []
    targets.append(("当前工作目录", Path.cwd()))
    home = Path.home()
    if home != Path.cwd():
        targets.append(("用户目录", home))
    if extra_paths:
        for p in extra_paths:
            try:
                if p.exists():
                    targets.append((p.as_posix(), p))
            except OSError:
                continue
    # All real mount points come last so cwd / home keep their friendly labels.
    targets.extend(_iter_real_mounts())

    for label, path in targets:
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        # De-duplicate by mount point: same drive, same numbers.
        key = f"{usage.total}-{usage.free}"
        if key in seen:
            continue
        seen.add(key)
        percent = (usage.used / usage.total * 100.0) if usage.total else 0.0
        out.append(
            DiskUsage(
                path=str(path),
                label=label,
                total_bytes=int(usage.total),
                used_bytes=int(usage.used),
                free_bytes=int(usage.free),
                percent=round(percent, 2),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# GPU collectors - one per source, then a dispatcher                          #
# --------------------------------------------------------------------------- #


_GPU_QUERY = (
    "index,name,driver_version,memory.total,memory.used,memory.free,"
    "utilization.gpu,temperature.gpu,power.draw,power.limit,fan.speed,"
    "pcie.link.gen.current,pcie.link.width.current,"
    "pcie.link.gen.max,pcie.link.width.max,"
    "clocks.current.sm,clocks.current.memory,"
    "clocks.max.sm,clocks.max.memory,"
    "compute_cap"
)


def _collect_nvidia_gpus(start_index: int = 0) -> list[GpuStats]:
    smi = _find_nvidia_smi()
    if smi is None:
        return []
    try:
        proc = _run_hidden(  # noqa: S603
            [smi, f"--query-gpu={_GPU_QUERY}", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=4,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []

    out: list[GpuStats] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 11:
            continue

        def _f(s: str) -> float | None:
            if not s or s.lower().startswith(("[n/a]", "n/a", "not")):
                return None
            try:
                return float(s)
            except ValueError:
                return None

        def _i(s: str) -> int | None:
            v = _f(s)
            return int(v) if v is not None else None

        # Helper to safely fetch optional fields - older driver releases don't
        # ship every column we ask for.
        def _at(idx: int) -> str:
            return parts[idx] if idx < len(parts) else ""

        mem_total = _f(parts[3])
        mem_used = _f(parts[4])
        mem_free = _f(parts[5])
        out.append(
            GpuStats(
                index=start_index + len(out),
                name=parts[1],
                driver=parts[2] or None,
                memory_total_bytes=int(mem_total * 1024 * 1024) if mem_total is not None else None,
                memory_used_bytes=int(mem_used * 1024 * 1024) if mem_used is not None else None,
                memory_free_bytes=int(mem_free * 1024 * 1024) if mem_free is not None else None,
                utilization_percent=_f(parts[6]),
                temperature_c=_f(parts[7]),
                power_w=_f(parts[8]),
                power_limit_w=_f(parts[9]),
                fan_percent=_f(parts[10]),
                vendor="nvidia",
                pcie_gen_current=_i(_at(11)),
                pcie_width_current=_i(_at(12)),
                pcie_gen_max=_i(_at(13)),
                pcie_width_max=_i(_at(14)),
                sm_clock_mhz=_i(_at(15)),
                mem_clock_mhz=_i(_at(16)),
                sm_clock_max_mhz=_i(_at(17)),
                mem_clock_max_mhz=_i(_at(18)),
                compute_capability=(_at(19) or None),
            )
        )
    return out


def _collect_gpu_processes() -> list[GpuProcessInfo]:
    """Per-process GPU memory map via two ``nvidia-smi`` queries.

    First call resolves UUID -> index because ``--query-compute-apps`` reports
    the GPU UUID (no index column available in current nvidia-smi). The
    second pulls the actual compute apps. Both go through best-effort error
    handling: a missing tool, a non-zero exit, or a parse failure simply
    yields an empty list so the snapshot stays usable.
    """
    smi = _find_nvidia_smi()
    if smi is None:
        return []
    try:
        idx_proc = _run_hidden(  # noqa: S603
            [smi, "--query-gpu=index,uuid", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=4,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if idx_proc.returncode != 0:
        return []
    uuid_to_index: dict[str, int] = {}
    for line in idx_proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            uuid_to_index[parts[1]] = int(parts[0])
        except ValueError:
            continue
    if not uuid_to_index:
        return []

    try:
        apps_proc = _run_hidden(  # noqa: S603
            [
                smi,
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=4,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if apps_proc.returncode != 0:
        return []

    out: list[GpuProcessInfo] = []
    for line in apps_proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpu_uuid, pid_str, name, mem_str = parts[0], parts[1], parts[2], parts[3]
        gpu_index = uuid_to_index.get(gpu_uuid)
        if gpu_index is None:
            continue
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        try:
            mem_mib = int(float(mem_str)) if mem_str and not mem_str.lower().startswith(("[n/a]", "n/a")) else 0
        except ValueError:
            mem_mib = 0
        out.append(
            GpuProcessInfo(
                gpu_index=gpu_index,
                pid=pid,
                process_name=name,
                used_memory_mib=mem_mib,
                type="C",  # --query-compute-apps only enumerates compute (CUDA) clients
            )
        )
    return out


def _collect_macos_gpus(start_index: int = 0) -> list[GpuStats]:
    """Parse `system_profiler -json SPDisplaysDataType`. Apple Silicon has no
    utilization/temperature/VRAM accessible to userland, so leave those None.
    """
    profiler = _find_system_profiler()
    if profiler is None:
        return []
    try:
        proc = _run_hidden(  # noqa: S603
            [profiler, "-json", "SPDisplaysDataType"],
            check=False,
            capture_output=True,
            text=True,
            timeout=4,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []

    entries = payload.get("SPDisplaysDataType")
    if not isinstance(entries, list):
        return []

    out: list[GpuStats] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = (
            entry.get("sppci_model")
            or entry.get("_name")
            or entry.get("sppci_chip_model")
            or "Apple GPU"
        )
        # Apple Silicon also exposes core count under sppci_cores.
        cores_str = entry.get("sppci_cores")
        if cores_str:
            name = f"{name} ({cores_str} cores)"
        vendor_field = (entry.get("spdisplays_vendor") or "").lower()
        if "apple" in vendor_field or name.lower().startswith(("apple ", "apple m")):
            vendor = "apple"
        elif "amd" in vendor_field or "ati" in vendor_field:
            vendor = "amd"
        elif "intel" in vendor_field:
            vendor = "intel"
        elif "nvidia" in vendor_field:
            vendor = "nvidia"
        else:
            vendor = "unknown"

        # Discrete GPUs sometimes report VRAM as e.g. "8 GB".
        mem_total: int | None = None
        for vram_key in ("spdisplays_vram_shared", "spdisplays_vram", "_spdisplays_vram"):
            raw = entry.get(vram_key)
            if isinstance(raw, str):
                mem_total = _parse_size_string(raw)
                if mem_total is not None:
                    break

        driver = entry.get("spdisplays_metalfamily") or entry.get("spdisplays_driver_version")

        out.append(
            GpuStats(
                index=start_index + len(out),
                name=str(name),
                driver=str(driver) if driver else None,
                memory_total_bytes=mem_total,
                memory_used_bytes=None,
                memory_free_bytes=None,
                utilization_percent=None,
                temperature_c=None,
                power_w=None,
                power_limit_w=None,
                fan_percent=None,
                vendor=vendor,
            )
        )
    return out


def _collect_windows_video_gpus(start_index: int = 0) -> list[GpuStats]:
    powershell = shutil.which("powershell") or shutil.which("powershell.exe")
    if powershell is None:
        return []
    try:
        proc = _run_hidden(  # noqa: S603
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name,AdapterCompatibility,AdapterRAM,DriverVersion | "
                "ConvertTo-Json -Compress",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    entries = payload if isinstance(payload, list) else [payload]

    out: list[GpuStats] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("Name") or "").strip()
        vendor_raw = str(entry.get("AdapterCompatibility") or name).lower()
        vendor = _gpu_vendor_from_text(vendor_raw)
        if vendor == "nvidia" and _find_nvidia_smi() is not None:
            continue
        if vendor == "unknown" and not name:
            continue
        mem_total = _coerce_int(entry.get("AdapterRAM"))
        out.append(
            GpuStats(
                index=start_index + len(out),
                name=name or "GPU",
                driver=str(entry.get("DriverVersion") or "") or None,
                memory_total_bytes=mem_total if mem_total and mem_total > 0 else None,
                memory_used_bytes=None,
                memory_free_bytes=None,
                utilization_percent=None,
                temperature_c=None,
                power_w=None,
                power_limit_w=None,
                fan_percent=None,
                vendor=vendor,
            )
        )
    return out


def _collect_windows_nwinfo_gpus(start_index: int = 0) -> list[GpuStats]:
    nwinfo = _find_nwinfo()
    if nwinfo is None:
        return []
    try:
        proc = _run_hidden(  # noqa: S603
            [nwinfo, "--format=json", "--cp=UTF8", "--gpu"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []

    out: list[GpuStats] = []
    seen: set[str] = set()
    for entry in _nwinfo_gpu_entries(payload):
        flat = _nwinfo_flatten(entry)
        name = _flat_pick_str(flat, ("name", "devicename", "adaptername", "model", "gpu"))
        vendor = _gpu_vendor_from_text(
            " ".join(
                v
                for v in (
                    _flat_pick_str(flat, ("vendor", "manufacturer", "adaptercompatibility", "api")),
                    name,
                )
                if v
            )
        )
        if vendor == "nvidia" and _find_nvidia_smi() is not None:
            continue
        if not name or vendor == "unknown":
            continue
        key = f"{vendor}:{name.lower()}"
        if key in seen:
            continue
        seen.add(key)

        total = _flat_pick_bytes(flat, ("vramtotal", "memorytotal", "dedicatedmemory", "dedicatedvideomemory", "totalmemory"))
        used = _flat_pick_bytes(flat, ("vramused", "memoryused", "usedmemory", "dedicatedmemoryused"))
        free = _flat_pick_bytes(flat, ("vramfree", "memoryfree", "freememory"))
        if free is None and total is not None and used is not None:
            free = max(total - used, 0)
        out.append(
            GpuStats(
                index=start_index + len(out),
                name=name,
                driver=_flat_pick_str(flat, ("driver", "driverversion")),
                memory_total_bytes=total,
                memory_used_bytes=used,
                memory_free_bytes=free,
                utilization_percent=_flat_pick_float(flat, ("gpuusage", "gpuutilization", "utilization", "usage", "load")),
                temperature_c=_flat_pick_float(flat, ("gputemperature", "temperature", "temperaturec")),
                power_w=_flat_pick_float(flat, ("powerdraw", "boardpower", "gpupower", "power")),
                power_limit_w=_flat_pick_float(flat, ("powerlimit", "powerlimitw")),
                fan_percent=_flat_pick_float(flat, ("fanspeed", "fan")),
                vendor=vendor,
                sm_clock_mhz=_flat_pick_int(flat, ("coreclock", "gpuclock", "frequency")),
                mem_clock_mhz=_flat_pick_int(flat, ("memoryclock", "memclock")),
            )
        )
    return out


def _nwinfo_gpu_entries(value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return
        flat = _nwinfo_flatten(node)
        has_name = _flat_pick_str(flat, ("name", "devicename", "adaptername", "model", "gpu")) is not None
        has_metric = any(k in flat for k in ("gpuusage", "gpuutilization", "vramtotal", "memorytotal", "temperature", "powerdraw"))
        if has_name and has_metric:
            out.append(node)
            return
        for child in node.values():
            visit(child)

    visit(value)
    return out


def _nwinfo_flatten(value: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}

    def add(key: str, item: Any) -> None:
        if item is None or isinstance(item, (dict, list)):
            return
        norm = re.sub(r"[^a-z0-9]+", "", key.lower())
        if norm and norm not in out:
            out[norm] = str(item)

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                text_key = str(key)
                add(text_key, item)
                walk(f"{prefix} {text_key}".strip(), item)
        elif isinstance(node, list):
            for item in node:
                walk(prefix, item)
        else:
            add(prefix, node)

    walk("", value)
    return out


def _flat_pick_str(flat: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = flat.get(key)
        if value and value.lower() not in {"n/a", "na", "none", "not supported"}:
            return value.strip()
    return None


def _flat_pick_float(flat: dict[str, str], keys: tuple[str, ...]) -> float | None:
    text = _flat_pick_str(flat, keys)
    if text is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _flat_pick_int(flat: dict[str, str], keys: tuple[str, ...]) -> int | None:
    value = _flat_pick_float(flat, keys)
    return int(value) if value is not None else None


def _flat_pick_bytes(flat: dict[str, str], keys: tuple[str, ...]) -> int | None:
    text = _flat_pick_str(flat, keys)
    if text is None:
        return None
    parsed = _parse_size_string(text)
    if parsed is not None:
        return parsed
    value = _flat_pick_float(flat, keys)
    return int(value) if value is not None else None


def _gpu_vendor_from_text(text: str) -> str:
    lower = text.lower()
    if "amd" in lower or "advanced micro devices" in lower or "ati" in lower or "radeon" in lower:
        return "amd"
    if "nvidia" in lower:
        return "nvidia"
    if "intel" in lower:
        return "intel"
    if "apple" in lower:
        return "apple"
    return "unknown"


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _collect_amd_rocm_gpus(start_index: int = 0) -> list[GpuStats]:
    smi = _find_rocm_smi()
    if smi is None:
        return []
    try:
        proc = _run_hidden(  # noqa: S603
            [
                smi,
                "--showproductname",
                "--showmeminfo",
                "vram",
                "--showuse",
                "--showtemp",
                "--showpower",
                "--showfan",
                "--showdriverversion",
                "--json",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []

    cards = sorted(
        ((k, v) for k, v in payload.items() if isinstance(v, dict)),
        key=lambda item: _rocm_card_index(item[0]),
    )
    out: list[GpuStats] = []
    for key, card in cards:
        name = _rocm_pick_str(card, ("Card series", "Card model", "GPU ID", "Device Name"))
        total = _rocm_pick_bytes(card, ("VRAM Total Memory (B)", "VRAM Total Memory", "vram Total Memory (B)"))
        used = _rocm_pick_bytes(card, ("VRAM Total Used Memory (B)", "VRAM Total Used Memory", "vram Total Used Memory (B)"))
        free = max(total - used, 0) if total is not None and used is not None else None
        out.append(
            GpuStats(
                index=start_index + len(out),
                name=name or key,
                driver=_rocm_pick_str(card, ("Driver version", "Driver Version")),
                memory_total_bytes=total,
                memory_used_bytes=used,
                memory_free_bytes=free,
                utilization_percent=_rocm_pick_float(card, ("GPU use (%)", "GPU use")),
                temperature_c=_rocm_pick_float(card, ("Temperature (Sensor edge) (C)", "Temperature (Sensor junction) (C)", "Temperature")),
                power_w=_rocm_pick_float(card, ("Average Graphics Package Power (W)", "Current Socket Graphics Package Power (W)", "Power (W)")),
                power_limit_w=None,
                fan_percent=_rocm_pick_float(card, ("Fan Speed (%)", "Fan Level (%)")),
                vendor="amd",
            )
        )
    return out


def _collect_amd_amdgpu_top_gpus(start_index: int = 0) -> list[GpuStats]:
    tool = _find_amdgpu_top()
    if tool is None:
        return []
    try:
        proc = _run_hidden(  # noqa: S603
            [tool, "--json", "-n", "1"],
            check=False,
            capture_output=True,
            text=True,
            timeout=4,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    line = next((ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")), "")
    if not line:
        return []
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return []
    devices = payload.get("devices") if isinstance(payload, dict) else None
    if not isinstance(devices, list):
        return []

    out: list[GpuStats] = []
    for device in devices:
        if not isinstance(device, dict):
            continue
        info = device.get("Info") if isinstance(device.get("Info"), dict) else {}
        name = str(info.get("DeviceName") or info.get("MarketingName") or info.get("Name") or "AMD GPU")
        total = _amdgpu_top_bytes(device, ("VRAM", "vram"), ("Total", "total"))
        used = _amdgpu_top_bytes(device, ("VRAM", "vram"), ("Usage", "Used", "used", "usage"))
        free = max(total - used, 0) if total is not None and used is not None else None
        out.append(
            GpuStats(
                index=start_index + len(out),
                name=name,
                driver=None,
                memory_total_bytes=total,
                memory_used_bytes=used,
                memory_free_bytes=free,
                utilization_percent=_amdgpu_top_metric(device, ("gpu_activity", "GPU Activity"), ("GFX", "gfx", "GPU")),
                temperature_c=_amdgpu_top_metric(device, ("Sensors", "sensors", "gpu_metrics"), ("Temperature", "Edge Temperature", "edge_temperature")),
                power_w=_amdgpu_top_metric(device, ("Sensors", "sensors", "gpu_metrics"), ("Power", "Average Power", "average_power")),
                power_limit_w=None,
                fan_percent=_amdgpu_top_metric(device, ("Sensors", "sensors"), ("Fan", "Fan Speed", "fan_speed")),
                vendor="amd",
            )
        )
    return out


def _amdgpu_top_metric(device: dict[str, Any], groups: tuple[str, ...], keys: tuple[str, ...]) -> float | None:
    for group_key in groups:
        group = device.get(group_key)
        if not isinstance(group, dict):
            continue
        for key in keys:
            value = group.get(key)
            if isinstance(value, dict):
                value = value.get("value")
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _amdgpu_top_bytes(device: dict[str, Any], groups: tuple[str, ...], keys: tuple[str, ...]) -> int | None:
    for group_key in groups:
        group = device.get(group_key)
        if not isinstance(group, dict):
            continue
        for key in keys:
            value = group.get(key)
            unit = ""
            if isinstance(value, dict):
                unit = str(value.get("unit") or "")
                value = value.get("value")
            if value is None:
                continue
            if isinstance(value, str):
                parsed = _parse_size_string(value)
                if parsed is not None:
                    return parsed
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            factor = 1
            lower = unit.lower()
            if lower in {"kb", "kib"}:
                factor = 1024
            elif lower in {"mb", "mib"}:
                factor = 1024**2
            elif lower in {"gb", "gib"}:
                factor = 1024**3
            return int(number * factor)
    return None


def _rocm_card_index(name: str) -> int:
    m = re.search(r"\d+", name)
    return int(m.group(0)) if m else 0


def _rocm_pick_str(card: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = card.get(key)
        if value not in (None, "", "N/A", "Not supported"):
            return str(value)
    return None


def _rocm_pick_float(card: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    text = _rocm_pick_str(card, keys)
    if text is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _rocm_pick_bytes(card: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    text = _rocm_pick_str(card, keys)
    if text is None:
        return None
    value = _rocm_pick_float(card, keys)
    if value is None:
        return None
    lower = text.lower()
    if "tib" in lower or "tb" in lower:
        return int(value * 1024**4)
    if "gib" in lower or "gb" in lower:
        return int(value * 1024**3)
    if "mib" in lower or "mb" in lower:
        return int(value * 1024**2)
    if "kib" in lower or "kb" in lower:
        return int(value * 1024)
    return int(value)


_SIZE_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(B|KB|KIB|MB|MIB|GB|GIB|TB|TIB)", re.IGNORECASE)


def _parse_size_string(text: str) -> int | None:
    m = _SIZE_RE.search(text)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    unit = m.group(2).upper()
    factor = {"B": 1, "KB": 1024, "KIB": 1024, "MB": 1024**2, "MIB": 1024**2, "GB": 1024**3, "GIB": 1024**3, "TB": 1024**4, "TIB": 1024**4}[unit]
    return int(value * factor)


# Linux DRM vendor IDs we care about.
_DRM_VENDORS = {
    "0x1002": ("amd", "AMD GPU"),
    "0x10de": ("nvidia", "NVIDIA GPU"),
    "0x8086": ("intel", "Intel GPU"),
    "0x1234": ("qemu", "QEMU GPU"),
}


def _read_text_file(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except (OSError, UnicodeError):
        return None


def _read_int_file(path: Path) -> int | None:
    raw = _read_text_file(path)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _amdgpu_temperature_c(device_dir: Path) -> float | None:
    """Walk hwmon under the device for amdgpu/k10temp temp1_input (millidegC)."""
    hwmon_root = device_dir / "hwmon"
    try:
        candidates = list(hwmon_root.iterdir())
    except OSError:
        return None
    for hwmon_dir in candidates:
        try:
            temp_raw = _read_int_file(hwmon_dir / "temp1_input")
        except OSError:
            continue
        if temp_raw is None:
            continue
        return temp_raw / 1000.0
    return None


def _gpu_name_from_uevent(device_dir: Path) -> str | None:
    raw = _read_text_file(device_dir / "uevent")
    if raw is None:
        return None
    driver: str | None = None
    pci_id: str | None = None
    for line in raw.splitlines():
        if line.startswith("DRIVER="):
            driver = line.partition("=")[2]
        elif line.startswith("PCI_ID="):
            pci_id = line.partition("=")[2]
    if driver and pci_id:
        return f"{driver} {pci_id}"
    return driver or pci_id


def _collect_linux_drm_gpus(start_index: int = 0) -> list[GpuStats]:
    drm_root = Path("/sys/class/drm")
    try:
        cards = sorted(p for p in drm_root.iterdir() if re.fullmatch(r"card\d+", p.name))
    except OSError:
        return []

    out: list[GpuStats] = []
    seen_devices: set[str] = set()
    for card in cards:
        device_link = card / "device"
        try:
            device_dir = device_link.resolve()
        except OSError:
            continue
        device_key = str(device_dir)
        if device_key in seen_devices:
            continue
        seen_devices.add(device_key)

        vendor_raw = _read_text_file(device_dir / "vendor")
        if vendor_raw is None:
            continue
        vendor_key = vendor_raw.lower()
        # Vendor-specific tools report richer metrics; DRM is only fallback.
        if vendor_key == "0x10de" and _find_nvidia_smi() is not None:
            continue
        if vendor_key == "0x1002" and _find_rocm_smi() is not None:
            continue
        vendor_label, default_name = _DRM_VENDORS.get(vendor_key, ("unknown", "GPU"))
        if vendor_label in {"unknown", "qemu"}:
            continue

        name = _gpu_name_from_uevent(device_dir) or default_name

        # VRAM (amdgpu exposes this; intel iGPU usually doesn't).
        mem_total = _read_int_file(device_dir / "mem_info_vram_total")
        mem_used = _read_int_file(device_dir / "mem_info_vram_used")
        mem_free: int | None = None
        if mem_total is not None and mem_used is not None:
            mem_free = max(mem_total - mem_used, 0)

        temp_c = _amdgpu_temperature_c(device_dir)

        out.append(
            GpuStats(
                index=start_index + len(out),
                name=name,
                driver=None,
                memory_total_bytes=mem_total,
                memory_used_bytes=mem_used,
                memory_free_bytes=mem_free,
                utilization_percent=None,
                temperature_c=temp_c,
                power_w=None,
                power_limit_w=None,
                fan_percent=None,
                vendor=vendor_label,
            )
        )
    return out


def _collect_gpus() -> list[GpuStats]:
    """Multi-source GPU discovery: NVIDIA first, then platform-specific paths."""
    gpus: list[GpuStats] = []
    # never let a single probe kill the snapshot - swallow any unexpected error
    with contextlib.suppress(Exception):
        gpus.extend(_collect_nvidia_gpus(start_index=len(gpus)))

    system = platform.system()
    if system == "Windows":
        with contextlib.suppress(Exception):
            gpus.extend(_collect_windows_nwinfo_gpus(start_index=len(gpus)))
        if not gpus:
            with contextlib.suppress(Exception):
                gpus.extend(_collect_windows_video_gpus(start_index=len(gpus)))
    elif system == "Darwin":
        with contextlib.suppress(Exception):
            gpus.extend(_collect_macos_gpus(start_index=len(gpus)))
    elif system == "Linux":
        with contextlib.suppress(Exception):
            gpus.extend(_collect_amd_rocm_gpus(start_index=len(gpus)))
        if not any(g.vendor == "amd" for g in gpus):
            with contextlib.suppress(Exception):
                gpus.extend(_collect_amd_amdgpu_top_gpus(start_index=len(gpus)))
        with contextlib.suppress(Exception):
            gpus.extend(_collect_linux_drm_gpus(start_index=len(gpus)))
    return gpus


# --------------------------------------------------------------------------- #
# Battery                                                                     #
# --------------------------------------------------------------------------- #


def _collect_battery() -> BatteryStats | None:
    if not _HAS_PSUTIL:
        return None
    fn = getattr(psutil, "sensors_battery", None)
    if fn is None:
        return None
    try:
        info = fn()
    except (OSError, NotImplementedError, AttributeError):
        return None
    if info is None:
        return None
    secs_left: int | None
    raw_secs = getattr(info, "secsleft", None)
    unlimited = getattr(psutil, "POWER_TIME_UNLIMITED", -1)
    unknown = getattr(psutil, "POWER_TIME_UNKNOWN", -2)
    if raw_secs is None or raw_secs in (unlimited, unknown):
        secs_left = None
    else:
        try:
            secs_left = int(raw_secs)
            if secs_left < 0:
                secs_left = None
        except (TypeError, ValueError):
            secs_left = None
    plugged = getattr(info, "power_plugged", None)
    try:
        percent = float(info.percent)
    except (TypeError, ValueError, AttributeError):
        return None
    return BatteryStats(
        percent=percent,
        plugged=bool(plugged) if plugged is not None else None,
        secs_left=secs_left,
    )


def _collect_host() -> HostInfo:
    return HostInfo(
        hostname=platform.node() or "",
        system=platform.system(),
        release=platform.release(),
        python=platform.python_version(),
    )


# Lightweight rolling state for network throughput. We snapshot the
# psutil counters on every call and diff against the previous reading to
# get a per-second rate. First call always returns a 0 rate (no prior).
_last_net_sample: tuple[float, int, int] | None = None

# Per-NIC rolling state, keyed by interface name. Same shape: (t, sent, recv).
_last_pernic_sample: dict[str, tuple[float, int, int]] = {}

# Public-IP cache. We hit the network at most once every 5 minutes; failures
# are cached too so we don't hammer external endpoints when offline.
_PUBLIC_IP_TTL_SECONDS = 300.0
_public_ip_cache: PublicIpInfo | None = None
_public_ip_cache_monotonic: float | None = None


_VIRTUAL_PREFIXES = ("docker", "veth", "br-", "virbr", "tun", "tap", "wg")
_WIRELESS_PREFIXES = ("wlan", "wifi", "wlp")


def _classify_interface(name: str) -> str:
    """Heuristic kind for an interface name.

    'lo*'                       -> 'loopback'
    docker / veth / br- / virbr / tun / tap / wg prefixes -> 'virtual'
    wlan / wifi / wlp prefixes  -> 'wireless'
    everything else             -> 'physical'
    """
    lname = name.lower()
    if lname == "lo" or lname.startswith("lo"):
        # Be careful: 'lo' alone, 'lo0' on macOS, 'Loopback Pseudo-Interface' on
        # Windows all count. Real NICs rarely start with 'lo' - 'localhost'-ish
        # adapter names on Windows include the word 'Loopback' too.
        return "loopback"
    if "loopback" in lname:
        return "loopback"
    if lname.startswith(_VIRTUAL_PREFIXES):
        return "virtual"
    if lname.startswith(_WIRELESS_PREFIXES):
        return "wireless"
    return "physical"


def _address_family_label(family: Any) -> str:
    if family == socket.AF_INET:
        return "IPv4"
    if family == socket.AF_INET6:
        return "IPv6"
    af_link = getattr(psutil, "AF_LINK", None) if _HAS_PSUTIL else None
    if af_link is not None and family == af_link:
        return "MAC"
    return str(family)


def _collect_network_interfaces() -> list[NetworkInterfaceStats]:
    """Per-NIC counters + addresses + link state + rolling per-NIC rate."""
    if not _HAS_PSUTIL:
        return []
    try:
        per_io = psutil.net_io_counters(pernic=True)
    except Exception:  # noqa: BLE001
        return []
    try:
        per_if_stats = psutil.net_if_stats()
    except Exception:  # noqa: BLE001
        per_if_stats = {}
    try:
        per_if_addrs = psutil.net_if_addrs()
    except Exception:  # noqa: BLE001
        per_if_addrs = {}

    now = time.monotonic()
    out: list[NetworkInterfaceStats] = []
    seen_names: set[str] = set()
    for name, counters in per_io.items():
        seen_names.add(name)
        sent = int(getattr(counters, "bytes_sent", 0) or 0)
        recv = int(getattr(counters, "bytes_recv", 0) or 0)

        rate_sent = 0.0
        rate_recv = 0.0
        prev = _last_pernic_sample.get(name)
        if prev is not None:
            prev_t, prev_sent, prev_recv = prev
            dt = now - prev_t
            if dt > 0:
                rate_sent = max(0.0, (sent - prev_sent) / dt)
                rate_recv = max(0.0, (recv - prev_recv) / dt)
        _last_pernic_sample[name] = (now, sent, recv)

        if_stats = per_if_stats.get(name)
        is_up = bool(getattr(if_stats, "isup", False)) if if_stats is not None else False
        speed_raw = getattr(if_stats, "speed", None) if if_stats is not None else None
        try:
            speed_mbps = int(speed_raw) if speed_raw is not None else None
        except (TypeError, ValueError):
            speed_mbps = None
        # psutil reports 0 when speed is unknown - normalise that to None.
        if speed_mbps == 0:
            speed_mbps = None
        mtu_raw = getattr(if_stats, "mtu", None) if if_stats is not None else None
        try:
            mtu = int(mtu_raw) if mtu_raw is not None else None
        except (TypeError, ValueError):
            mtu = None

        addresses: list[InterfaceAddress] = []
        for addr in per_if_addrs.get(name, []):
            family = getattr(addr, "family", None)
            address = getattr(addr, "address", None)
            if not address:
                continue
            addresses.append(
                InterfaceAddress(
                    family=_address_family_label(family),
                    address=str(address),
                    netmask=getattr(addr, "netmask", None) or None,
                    broadcast=getattr(addr, "broadcast", None) or None,
                )
            )

        out.append(
            NetworkInterfaceStats(
                name=name,
                is_up=is_up,
                speed_mbps=speed_mbps,
                mtu=mtu,
                addresses=addresses,
                bytes_sent_total=sent,
                bytes_recv_total=recv,
                bytes_sent_per_sec=round(rate_sent, 2),
                bytes_recv_per_sec=round(rate_recv, 2),
                packets_sent_total=int(getattr(counters, "packets_sent", 0) or 0),
                packets_recv_total=int(getattr(counters, "packets_recv", 0) or 0),
                errors_in=int(getattr(counters, "errin", 0) or 0),
                errors_out=int(getattr(counters, "errout", 0) or 0),
                drops_in=int(getattr(counters, "dropin", 0) or 0),
                drops_out=int(getattr(counters, "dropout", 0) or 0),
                kind=_classify_interface(name),
            )
        )

    # Drop disappeared NICs from the rolling state so the dict can't grow
    # forever on long-lived processes.
    stale = [k for k in _last_pernic_sample if k not in seen_names]
    for k in stale:
        _last_pernic_sample.pop(k, None)
    return out


def _collect_tcp_connections() -> TcpConnectionStats | None:
    """Aggregate TCP connection states.

    On Linux/macOS, non-root processes only see their own connections via
    psutil.net_connections(); we still return what's visible rather than
    erroring out. PermissionError / OSError are swallowed and become None.
    """
    if not _HAS_PSUTIL:
        return None
    try:
        conns = psutil.net_connections(kind="tcp")
    except (PermissionError, OSError, RuntimeError, NotImplementedError):
        return None
    except Exception:  # noqa: BLE001
        return None

    established = 0
    listen = 0
    time_wait = 0
    close_wait = 0
    other = 0
    total = 0
    for c in conns:
        total += 1
        status = (getattr(c, "status", "") or "").upper()
        if status == "ESTABLISHED":
            established += 1
        elif status == "LISTEN":
            listen += 1
        elif status == "TIME_WAIT":
            time_wait += 1
        elif status == "CLOSE_WAIT":
            close_wait += 1
        else:
            other += 1
    return TcpConnectionStats(
        total=total,
        established=established,
        listen=listen,
        time_wait=time_wait,
        close_wait=close_wait,
        other=other,
    )


_PUBLIC_IP_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("ip.sb", "https://api.ip.sb/ip"),
    ("ipinfo.io", "https://ipinfo.io/ip"),
)


def _fetch_public_ip_once(url: str) -> str | None:
    """Hit a single endpoint with a short timeout and return one IP line."""
    try:
        req = urllib.request.Request(  # noqa: S310 - https only, see endpoints tuple
            url,
            headers={"User-Agent": "lorahub-system-stats/1.0"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:  # noqa: S310
            raw = resp.read()
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return None
    except Exception:  # noqa: BLE001
        return None
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not line:
        return None
    # Loose sanity: strip anything that obviously isn't an IP. We keep it
    # forgiving because IPv6 contains colons and ipinfo plain endpoint is
    # text/plain anyway.
    if len(line) > 64:
        return None
    return line


def _collect_public_ip() -> PublicIpInfo | None:
    """Resolve the host's public IP, cached for 5 minutes (success or failure).

    Returns None only if psutil is missing - actually that's irrelevant here,
    public IP is independent. We keep returning None as a sentinel so callers
    can drop the field when caching has never run.
    """
    global _public_ip_cache, _public_ip_cache_monotonic
    now_mono = time.monotonic()
    if (
        _public_ip_cache is not None
        and _public_ip_cache_monotonic is not None
        and (now_mono - _public_ip_cache_monotonic) < _PUBLIC_IP_TTL_SECONDS
    ):
        # Return a copy so callers can't mutate our cache.
        cached = _public_ip_cache
        return PublicIpInfo(
            ip=cached.ip,
            fetched_at=cached.fetched_at,
            source="cached",
        )

    fetched_at = time.time()
    for source, url in _PUBLIC_IP_ENDPOINTS:
        ip = _fetch_public_ip_once(url)
        if ip:
            info = PublicIpInfo(ip=ip, fetched_at=fetched_at, source=source)
            _public_ip_cache = info
            _public_ip_cache_monotonic = now_mono
            return info

    info = PublicIpInfo(ip=None, fetched_at=fetched_at, source="unreachable")
    _public_ip_cache = info
    _public_ip_cache_monotonic = now_mono
    return info


# Same idea for disk IO: aggregate sample is (t, read_bytes, write_bytes,
# read_count, write_count); per-device is keyed by device name with the same
# tuple shape minus the timestamp (we reuse the aggregate's timestamp).
_last_disk_sample: tuple[float, int, int, int, int] | None = None
_last_perdisk_sample: dict[str, tuple[int, int, int, int]] = {}


def _collect_network() -> NetworkStats | None:
    if not _HAS_PSUTIL:
        return None
    try:
        counters = psutil.net_io_counters()
    except Exception:  # noqa: BLE001
        return None

    global _last_net_sample
    now = time.monotonic()
    sent = int(counters.bytes_sent)
    recv = int(counters.bytes_recv)

    rate_sent = 0.0
    rate_recv = 0.0
    if _last_net_sample is not None:
        prev_t, prev_sent, prev_recv = _last_net_sample
        dt = now - prev_t
        if dt > 0:
            rate_sent = max(0.0, (sent - prev_sent) / dt)
            rate_recv = max(0.0, (recv - prev_recv) / dt)
    _last_net_sample = (now, sent, recv)

    interfaces = _collect_network_interfaces()
    tcp = _collect_tcp_connections()
    public_ip = _collect_public_ip()

    return NetworkStats(
        bytes_sent_total=sent,
        bytes_recv_total=recv,
        bytes_sent_per_sec=round(rate_sent, 2),
        bytes_recv_per_sec=round(rate_recv, 2),
        interfaces=interfaces,
        tcp_connections=tcp,
        public_ip=public_ip,
    )


def _network_to_dict(net: NetworkStats | None) -> dict[str, Any] | None:
    return network_to_dict(net)


# Cache the most recent cpu_percent reading per pid so that the very first
# `process_iter` call (which always emits 0.0 because psutil needs two samples
# to compute a delta) can fall back to whatever we measured last time.
_last_process_cpu: dict[int, float] = {}


def _is_kernel_thread_name(name: str) -> bool:
    # Linux kernel threads expose their comm wrapped in square brackets
    # (e.g. "[kthreadd]", "[ksoftirqd/0]"). They have no meaningful RSS so
    # we skip them - they only clutter a top-N list.
    return name.startswith("[") and name.endswith("]")


def _collect_top_processes(n: int = 5) -> list[ProcessInfo]:
    """Return the top ``n`` processes ranked by RSS (descending).

    psutil quirks we paper over here:
    * The first ``cpu_percent()`` reading per process is always ``0.0``
      because psutil needs two samples - we cache the previous reading and
      reuse it when the new one is zero but a prior one exists.
    * Processes can disappear mid-iteration; ``NoSuchProcess`` /
      ``AccessDenied`` simply skip that pid.
    """
    if not _HAS_PSUTIL:
        return []

    NoSuchProcess = getattr(psutil, "NoSuchProcess", Exception)
    AccessDenied = getattr(psutil, "AccessDenied", Exception)
    ZombieProcess = getattr(psutil, "ZombieProcess", Exception)

    rows: list[ProcessInfo] = []
    new_cache: dict[int, float] = {}
    try:
        iterator = psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_percent", "memory_info"]
        )
    except Exception:  # noqa: BLE001
        return []

    for proc in iterator:
        try:
            info = proc.info
            pid = int(info.get("pid") or 0)
            if pid <= 1:
                continue
            name = info.get("name") or ""
            if _is_kernel_thread_name(name):
                continue
            mem_info = info.get("memory_info")
            rss = int(getattr(mem_info, "rss", 0) or 0)
            mem_pct_raw = info.get("memory_percent")
            try:
                mem_pct = float(mem_pct_raw) if mem_pct_raw is not None else 0.0
            except (TypeError, ValueError):
                mem_pct = 0.0
            cpu_pct_raw = info.get("cpu_percent")
            try:
                cpu_pct = float(cpu_pct_raw) if cpu_pct_raw is not None else 0.0
            except (TypeError, ValueError):
                cpu_pct = 0.0
            # First sample per pid is always 0.0; reuse the previous reading
            # so the dashboard does not look universally idle.
            if cpu_pct == 0.0 and pid in _last_process_cpu:
                cpu_pct = _last_process_cpu[pid]
            new_cache[pid] = cpu_pct
        except (NoSuchProcess, AccessDenied, ZombieProcess):
            continue
        except Exception:  # noqa: BLE001
            # Any other oddity from a single process must not break the snapshot.
            continue
        rows.append(
            ProcessInfo(
                pid=pid,
                name=name,
                cpu_percent=cpu_pct,
                memory_rss_bytes=rss,
                memory_percent=mem_pct,
            )
        )

    _last_process_cpu.clear()
    _last_process_cpu.update(new_cache)

    rows.sort(key=lambda p: p.memory_rss_bytes, reverse=True)
    if n <= 0:
        return []
    return rows[:n]


def _collect_disk_io() -> DiskIoStats | None:
    """Aggregate + per-device disk IO with rolling rate calculation.

    Returns None if psutil is missing or the kernel does not expose IO
    counters (some containers strip /sys/block, ``disk_io_counters`` then
    raises or returns None).
    """
    if not _HAS_PSUTIL:
        return None
    try:
        agg = psutil.disk_io_counters(perdisk=False)
    except Exception:  # noqa: BLE001 - container kernels can raise here
        return None
    if agg is None:
        return None
    try:
        perdisk = psutil.disk_io_counters(perdisk=True) or {}
    except Exception:  # noqa: BLE001
        perdisk = {}

    global _last_disk_sample
    now = time.monotonic()
    read_bytes = int(agg.read_bytes)
    write_bytes = int(agg.write_bytes)
    read_count = int(agg.read_count)
    write_count = int(agg.write_count)

    # Capture the previous aggregate timestamp BEFORE we overwrite the global,
    # so the per-device loop below can reuse the same dt window.
    prev_sample = _last_disk_sample
    rate_read_b = 0.0
    rate_write_b = 0.0
    rate_read_ops = 0.0
    rate_write_ops = 0.0
    if prev_sample is not None:
        prev_t, prev_rb, prev_wb, prev_rc, prev_wc = prev_sample
        dt = now - prev_t
        if dt > 0:
            rate_read_b = max(0.0, (read_bytes - prev_rb) / dt)
            rate_write_b = max(0.0, (write_bytes - prev_wb) / dt)
            rate_read_ops = max(0.0, (read_count - prev_rc) / dt)
            rate_write_ops = max(0.0, (write_count - prev_wc) / dt)
    _last_disk_sample = (now, read_bytes, write_bytes, read_count, write_count)

    per_device: list[DiskIoDevice] = []
    new_perdisk: dict[str, tuple[int, int, int, int]] = {}
    prev_t = prev_sample[0] if prev_sample is not None else None
    dt = (now - prev_t) if (prev_t is not None and now > prev_t) else 0.0
    for device, counters in perdisk.items():
        rb = int(getattr(counters, "read_bytes", 0))
        wb = int(getattr(counters, "write_bytes", 0))
        rc = int(getattr(counters, "read_count", 0))
        wc = int(getattr(counters, "write_count", 0))
        new_perdisk[device] = (rb, wb, rc, wc)

        prev = _last_perdisk_sample.get(device)
        if prev is None or dt <= 0:
            d_rb = d_wb = d_rc = d_wc = 0.0
        else:
            p_rb, p_wb, p_rc, p_wc = prev
            d_rb = max(0.0, (rb - p_rb) / dt)
            d_wb = max(0.0, (wb - p_wb) / dt)
            d_rc = max(0.0, (rc - p_rc) / dt)
            d_wc = max(0.0, (wc - p_wc) / dt)
        per_device.append(
            DiskIoDevice(
                device=device,
                read_bytes_per_sec=round(d_rb, 2),
                write_bytes_per_sec=round(d_wb, 2),
                read_ops_per_sec=round(d_rc, 2),
                write_ops_per_sec=round(d_wc, 2),
            )
        )
    _last_perdisk_sample.clear()
    _last_perdisk_sample.update(new_perdisk)

    return DiskIoStats(
        read_bytes_total=read_bytes,
        write_bytes_total=write_bytes,
        read_bytes_per_sec=round(rate_read_b, 2),
        write_bytes_per_sec=round(rate_write_b, 2),
        read_ops_per_sec=round(rate_read_ops, 2),
        write_ops_per_sec=round(rate_write_ops, 2),
        per_device=per_device,
    )


def collect_snapshot(
    extra_disk_paths: list[Path] | None = None,
    top_processes_n: int = 5,
) -> SystemSnapshot:
    """Read every probe once and pack the result. Cheap (~10-100ms with GPU)."""
    return SystemSnapshot(
        timestamp=time.time(),
        host=_collect_host(),
        cpu=_collect_cpu(),
        memory=_collect_memory(),
        disks=_collect_disks(extra_disk_paths),
        gpus=_collect_gpus(),
        has_psutil=_HAS_PSUTIL,
        has_nvidia_smi=_find_nvidia_smi() is not None,
        battery=_collect_battery(),
        network=_collect_network(),
        processes=_collect_top_processes(top_processes_n),
        disk_io=_collect_disk_io(),
        gpu_processes=_collect_gpu_processes(),
    )


__all__ = [
    "BatteryStats",
    "CpuStats",
    "DiskIoDevice",
    "DiskIoStats",
    "GpuProcessInfo",
    "GpuStats",
    "InterfaceAddress",
    "NetworkInterfaceStats",
    "NetworkStats",
    "ProcessInfo",
    "PublicIpInfo",
    "SystemSnapshot",
    "TcpConnectionStats",
    "ALL_ATTENTION_BACKENDS",
    "attention_backends_for_gpu",
    "collect_snapshot",
]


# Canonical superset of attention backends LoraHub knows how to translate
# at the config level (see schema.AttentionConfig.training). The frontend
# disables anything missing from `attention_backends_for_gpu(...)` so the
# user can't pick a kernel their GPU can't run.
ALL_ATTENTION_BACKENDS: tuple[str, ...] = (
    "auto",
    "torch",
    "sdpa",
    "flex",
    "xformers",
    "flash",
    "flash3",
    "flash4",
)


def attention_backends_for_gpu(cap: str | None) -> list[str]:
    """Return the attention backends usable on a GPU with compute cap ``cap``.

    The compute-capability gating is conservative on purpose — we'd rather
    grey out a kernel that *might* work than let the trainer crash inside
    sd-scripts. References:

    * FlashAttention 2 needs sm_80+ (Ampere); RTX 30/40, A/H100 are fine.
    * FlashAttention 3 is Hopper-only (sm_90).
    * FlashAttention 4 (beta) supports Hopper and the early Blackwell
      sm_10x/sm_12x silicon; the FA3 Hopper-only kernels don't run on
      Blackwell directly.
    * xformers ships official wheels up through Hopper; Blackwell support
      is still trickling into nightly builds, so we mark it unsupported on
      sm_10+ rather than promise wheels that don't yet exist.

    A non-NVIDIA GPU (or no GPU detected) gets the safe set: PyTorch-native
    kernels only.
    """
    safe = ["auto", "torch", "sdpa", "flex"]
    if not cap:
        return safe

    raw = cap.strip()
    try:
        major_str, _, _ = raw.partition(".")
        major = int(major_str)
    except (TypeError, ValueError):
        return safe

    if major < 8:
        # Volta (sm_70/72): no FlashAttention, but xformers ships a fallback.
        return [*safe, "xformers"]
    if major == 8:
        # Ampere (sm_80/86) + Ada Lovelace (sm_89): FA2 yes, FA3/FA4 no.
        return [*safe, "xformers", "flash"]
    if major == 9:
        # Hopper (sm_90): the only family that runs FA3.
        return [*safe, "xformers", "flash", "flash3", "flash4"]
    if major >= 10:
        # Blackwell (sm_100/120): FA2 + FA4 land but xformers wheels and
        # FA3 are not generally available yet.
        return [*safe, "flash", "flash4"]
    return safe
