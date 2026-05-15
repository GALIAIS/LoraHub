"""System & hardware telemetry for the dashboard.

Reads CPU / memory / disk via stdlib (with psutil as an optional accelerator)
and queries `nvidia-smi` for GPU info. All probes are best-effort: a missing
nvidia-smi or psutil downgrades a field rather than failing the request, so
the dashboard always has *something* to render.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # optional — gives us per-cpu utilization, load, swap.
    import psutil  # type: ignore[import-not-found]

    _HAS_PSUTIL = True
except Exception:  # noqa: BLE001
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False


# Cache the GPU device list — nvidia-smi spawn is ~80ms on Windows. We'll still
# probe live values every call; only the static device discovery is memoized.
_NVIDIA_SMI: str | None = None
_NVIDIA_SMI_PROBED = False


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
        fallback = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        if fallback.is_file():
            _NVIDIA_SMI = str(fallback)
            return _NVIDIA_SMI
    return None


# --------------------------------------------------------------------------- #
# Snapshot dataclasses                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class CpuStats:
    cores_logical: int
    cores_physical: int | None
    usage_percent: float | None
    per_core_percent: list[float] = field(default_factory=list)
    load_average: list[float] | None = None
    arch: str = ""


@dataclass
class MemoryStats:
    total_bytes: int
    used_bytes: int
    available_bytes: int
    percent: float
    swap_total_bytes: int | None = None
    swap_used_bytes: int | None = None


@dataclass
class DiskUsage:
    path: str
    label: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    percent: float


@dataclass
class GpuStats:
    index: int
    name: str
    driver: str | None
    memory_total_bytes: int | None
    memory_used_bytes: int | None
    memory_free_bytes: int | None
    utilization_percent: float | None
    temperature_c: float | None
    power_w: float | None
    power_limit_w: float | None
    fan_percent: float | None


@dataclass
class HostInfo:
    hostname: str
    system: str
    release: str
    python: str


@dataclass
class SystemSnapshot:
    timestamp: float
    host: HostInfo
    cpu: CpuStats
    memory: MemoryStats
    disks: list[DiskUsage]
    gpus: list[GpuStats]
    has_psutil: bool
    has_nvidia_smi: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "has_psutil": self.has_psutil,
            "has_nvidia_smi": self.has_nvidia_smi,
            "host": self.host.__dict__,
            "cpu": self.cpu.__dict__,
            "memory": self.memory.__dict__,
            "disks": [d.__dict__ for d in self.disks],
            "gpus": [g.__dict__ for g in self.gpus],
        }


# --------------------------------------------------------------------------- #
# Collectors                                                                  #
# --------------------------------------------------------------------------- #


def _collect_cpu() -> CpuStats:
    if _HAS_PSUTIL:
        per = psutil.cpu_percent(interval=None, percpu=True)
        load: list[float] | None = None
        try:
            load = list(os.getloadavg())  # 1/5/15 min — Linux + recent macOS.
        except (AttributeError, OSError):
            load = None
        return CpuStats(
            cores_logical=psutil.cpu_count(logical=True) or os.cpu_count() or 0,
            cores_physical=psutil.cpu_count(logical=False),
            usage_percent=psutil.cpu_percent(interval=None),
            per_core_percent=[float(p) for p in per],
            load_average=load,
            arch=platform.machine(),
        )
    return CpuStats(
        cores_logical=os.cpu_count() or 0,
        cores_physical=None,
        usage_percent=None,
        per_core_percent=[],
        load_average=None,
        arch=platform.machine(),
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
    # stdlib fallback — Linux /proc/meminfo only; on Windows return zeros.
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


def _collect_disks(extra_paths: list[Path] | None = None) -> list[DiskUsage]:
    """Report the workspace's disk plus any extras the caller cares about."""
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


_GPU_QUERY = (
    "index,name,driver_version,memory.total,memory.used,memory.free,"
    "utilization.gpu,temperature.gpu,power.draw,power.limit,fan.speed"
)


def _collect_gpus() -> list[GpuStats]:
    smi = _find_nvidia_smi()
    if smi is None:
        return []
    try:
        proc = subprocess.run(  # noqa: S603
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
        try:
            idx = int(parts[0])
        except ValueError:
            continue

        def _f(s: str) -> float | None:
            if not s or s.lower().startswith(("[n/a]", "n/a", "not")):
                return None
            try:
                return float(s)
            except ValueError:
                return None

        mem_total = _f(parts[3])
        mem_used = _f(parts[4])
        mem_free = _f(parts[5])
        out.append(
            GpuStats(
                index=idx,
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
            )
        )
    return out


def _collect_host() -> HostInfo:
    return HostInfo(
        hostname=platform.node() or "",
        system=platform.system(),
        release=platform.release(),
        python=platform.python_version(),
    )


def collect_snapshot(extra_disk_paths: list[Path] | None = None) -> SystemSnapshot:
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
    )


__all__ = ["SystemSnapshot", "collect_snapshot"]
