"""Typed payloads emitted by system telemetry collectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CpuStats:
    cores_logical: int
    cores_physical: int | None
    usage_percent: float | None
    per_core_percent: list[float] = field(default_factory=list)
    load_average: list[float] | None = None
    arch: str = ""
    model: str = ""
    frequency_mhz: float | None = None
    frequency_min_mhz: float | None = None
    frequency_max_mhz: float | None = None
    frequency_per_core_mhz: list[float] = field(default_factory=list)
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
    pcie_gen_current: int | None = None
    pcie_width_current: int | None = None
    pcie_gen_max: int | None = None
    pcie_width_max: int | None = None
    sm_clock_mhz: int | None = None
    mem_clock_mhz: int | None = None
    sm_clock_max_mhz: int | None = None
    mem_clock_max_mhz: int | None = None
    # CUDA compute capability ("8.6" for Ampere, "8.9" for Ada Lovelace,
    # "9.0" for Hopper, "10.0"/"12.0" for Blackwell). Used by the
    # frontend to gate attention-backend choices: FlashAttention 3
    # requires 9.x, FlashAttention 4 needs 9.x or 10+.
    compute_capability: str | None = None


@dataclass
class GpuProcessInfo:
    """Per-process GPU memory occupancy from ``nvidia-smi --query-compute-apps``."""

    gpu_index: int
    pid: int
    process_name: str
    used_memory_mib: int
    type: str  # "C" | "G" | "C+G" - compute / graphics / both


@dataclass
class BatteryStats:
    percent: float
    plugged: bool | None
    secs_left: int | None


@dataclass
class InterfaceAddress:
    family: str  # 'IPv4' | 'IPv6' | 'MAC' | other
    address: str
    netmask: str | None = None
    broadcast: str | None = None


@dataclass
class NetworkInterfaceStats:
    name: str
    is_up: bool
    speed_mbps: int | None
    mtu: int | None
    addresses: list[InterfaceAddress]
    bytes_sent_total: int
    bytes_recv_total: int
    bytes_sent_per_sec: float
    bytes_recv_per_sec: float
    packets_sent_total: int
    packets_recv_total: int
    errors_in: int
    errors_out: int
    drops_in: int
    drops_out: int
    # 'physical' | 'loopback' | 'virtual' | 'wireless'
    kind: str = "physical"


@dataclass
class TcpConnectionStats:
    total: int
    established: int
    listen: int
    time_wait: int
    close_wait: int
    other: int


@dataclass
class PublicIpInfo:
    ip: str | None
    fetched_at: float
    source: str


@dataclass
class DiskIoDevice:
    device: str
    read_bytes_per_sec: float
    write_bytes_per_sec: float
    read_ops_per_sec: float
    write_ops_per_sec: float


@dataclass
class DiskIoStats:
    """Aggregate disk IO with per-device breakdown."""

    read_bytes_total: int
    write_bytes_total: int
    read_bytes_per_sec: float
    write_bytes_per_sec: float
    read_ops_per_sec: float
    write_ops_per_sec: float
    per_device: list[DiskIoDevice] = field(default_factory=list)


@dataclass
class NetworkStats:
    """Per-snapshot network throughput."""

    bytes_sent_total: int
    bytes_recv_total: int
    bytes_sent_per_sec: float
    bytes_recv_per_sec: float
    interfaces: list[NetworkInterfaceStats] = field(default_factory=list)
    tcp_connections: TcpConnectionStats | None = None
    public_ip: PublicIpInfo | None = None


@dataclass
class HostInfo:
    hostname: str
    system: str
    release: str
    python: str


@dataclass
class ProcessInfo:
    pid: int
    name: str
    cpu_percent: float
    memory_rss_bytes: int
    memory_percent: float


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
    processes: list[ProcessInfo] = field(default_factory=list)
    disk_io: DiskIoStats | None = None
    gpu_processes: list[GpuProcessInfo] = field(default_factory=list)

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
            "network": network_to_dict(self.network),
            "processes": [p.__dict__ for p in self.processes],
            "disk_io": (
                {
                    **{
                        k: v
                        for k, v in self.disk_io.__dict__.items()
                        if k != "per_device"
                    },
                    "per_device": [d.__dict__ for d in self.disk_io.per_device],
                }
                if self.disk_io is not None
                else None
            ),
            "gpu_processes": [p.__dict__ for p in self.gpu_processes],
        }


def network_to_dict(net: NetworkStats | None) -> dict[str, Any] | None:
    if net is None:
        return None
    return {
        "bytes_sent_total": net.bytes_sent_total,
        "bytes_recv_total": net.bytes_recv_total,
        "bytes_sent_per_sec": net.bytes_sent_per_sec,
        "bytes_recv_per_sec": net.bytes_recv_per_sec,
        "interfaces": [
            {
                **{k: v for k, v in iface.__dict__.items() if k != "addresses"},
                "addresses": [a.__dict__ for a in iface.addresses],
            }
            for iface in net.interfaces
        ],
        "tcp_connections": (
            net.tcp_connections.__dict__ if net.tcp_connections is not None else None
        ),
        "public_ip": net.public_ip.__dict__ if net.public_ip is not None else None,
    }
