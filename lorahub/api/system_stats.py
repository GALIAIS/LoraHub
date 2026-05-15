"""System & hardware telemetry for the dashboard.

Reads CPU / memory / disk via stdlib (with psutil as an optional accelerator)
and queries multiple GPU sources so the dashboard makes sense on every host:

* NVIDIA on any platform via `nvidia-smi` (the original path).
* macOS Apple Silicon / Intel GPUs via `system_profiler -json SPDisplaysDataType`.
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
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # optional - gives us per-cpu utilization, load, swap, freq, temps, battery.
    import psutil  # type: ignore[import-not-found]

    _HAS_PSUTIL = True
except Exception:  # noqa: BLE001
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False


# Cache external-tool lookups - spawning these on Windows is ~80ms each.
_NVIDIA_SMI: str | None = None
_NVIDIA_SMI_PROBED = False
_SYSTEM_PROFILER: str | None = None
_SYSTEM_PROFILER_PROBED = False


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
    frequency_mhz: float | None = None
    cpu_temperature_c: float | None = None


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
    vendor: str = "unknown"  # "nvidia" | "amd" | "intel" | "apple" | "unknown"


@dataclass
class BatteryStats:
    percent: float
    plugged: bool | None
    secs_left: int | None


@dataclass
class NetworkStats:
    """Per-snapshot network throughput.

    `bytes_*_total` is monotonically increasing across the host's lifetime;
    `bytes_*_per_sec` is computed against the previous snapshot we emitted
    so the dashboard can show rolling rate without keeping its own history.
    """

    bytes_sent_total: int
    bytes_recv_total: int
    bytes_sent_per_sec: float
    bytes_recv_per_sec: float


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
    battery: BatteryStats | None = None
    network: NetworkStats | None = None

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
            "battery": self.battery.__dict__ if self.battery is not None else None,
            "network": self.network.__dict__ if self.network is not None else None,
        }


# --------------------------------------------------------------------------- #
# Collectors                                                                  #
# --------------------------------------------------------------------------- #


def _collect_cpu_frequency() -> float | None:
    if not _HAS_PSUTIL:
        return None
    try:
        freq = psutil.cpu_freq()
    except (OSError, NotImplementedError, AttributeError):
        return None
    if freq is None:
        return None
    current = getattr(freq, "current", None)
    if current is None:
        return None
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


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
    if _HAS_PSUTIL:
        per = psutil.cpu_percent(interval=None, percpu=True)
        load: list[float] | None = None
        try:
            load = list(os.getloadavg())  # 1/5/15 min - Linux + recent macOS.
        except (AttributeError, OSError):
            load = None
        return CpuStats(
            cores_logical=psutil.cpu_count(logical=True) or os.cpu_count() or 0,
            cores_physical=psutil.cpu_count(logical=False),
            usage_percent=psutil.cpu_percent(interval=None),
            per_core_percent=[float(p) for p in per],
            load_average=load,
            arch=platform.machine(),
            frequency_mhz=_collect_cpu_frequency(),
            cpu_temperature_c=_collect_cpu_temperature(),
        )
    return CpuStats(
        cores_logical=os.cpu_count() or 0,
        cores_physical=None,
        usage_percent=None,
        per_core_percent=[],
        load_average=None,
        arch=platform.machine(),
        frequency_mhz=None,
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


# --------------------------------------------------------------------------- #
# GPU collectors - one per source, then a dispatcher                          #
# --------------------------------------------------------------------------- #


_GPU_QUERY = (
    "index,name,driver_version,memory.total,memory.used,memory.free,"
    "utilization.gpu,temperature.gpu,power.draw,power.limit,fan.speed"
)


def _collect_nvidia_gpus(start_index: int = 0) -> list[GpuStats]:
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
        proc = subprocess.run(  # noqa: S603
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


_SIZE_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(KB|MB|GB|TB)", re.IGNORECASE)


def _parse_size_string(text: str) -> int | None:
    m = _SIZE_RE.search(text)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    unit = m.group(2).upper()
    factor = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}[unit]
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
        # NVIDIA on Linux is already covered by nvidia-smi when available; we
        # still report it via DRM only if nvidia-smi is missing.
        if vendor_key == "0x10de" and _find_nvidia_smi() is not None:
            continue
        vendor_label, default_name = _DRM_VENDORS.get(vendor_key, ("unknown", "GPU"))

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
    if system == "Darwin":
        with contextlib.suppress(Exception):
            gpus.extend(_collect_macos_gpus(start_index=len(gpus)))
    elif system == "Linux":
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

    return NetworkStats(
        bytes_sent_total=sent,
        bytes_recv_total=recv,
        bytes_sent_per_sec=round(rate_sent, 2),
        bytes_recv_per_sec=round(rate_recv, 2),
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
        battery=_collect_battery(),
        network=_collect_network(),
    )


__all__ = [
    "BatteryStats",
    "CpuStats",
    "GpuStats",
    "SystemSnapshot",
    "collect_snapshot",
]
