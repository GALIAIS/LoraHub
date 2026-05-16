"""Tests for the new disk IO + GPU telemetry helpers in `system_stats`.

These cover the unit-level helpers (mount filtering, IO rate diffing,
nvidia-smi parsing) so we don't depend on the host's actual hardware.
"""

from __future__ import annotations

from collections import namedtuple
from types import SimpleNamespace
from typing import Any

import pytest

from lorahub.api import system_stats


# --------------------------------------------------------------------------- #
# A. Mount enumeration                                                        #
# --------------------------------------------------------------------------- #


_FakePart = namedtuple("_FakePart", ["device", "mountpoint", "fstype", "opts"])


def test_iter_real_mounts_filters_virtual_fs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Virtual filesystems (tmpfs, overlay, proc, ...) must not be reported."""
    fake_partitions = [
        _FakePart("/dev/sda1", "/", "ext4", "rw"),
        _FakePart("/dev/nvme0n1p1", "/data", "xfs", "rw"),
        _FakePart("tmpfs", "/run", "tmpfs", "rw"),
        _FakePart("overlay", "/var/lib/docker/overlay", "overlay", "rw"),
        _FakePart("proc", "/proc", "proc", "rw"),
        _FakePart("cgroup2", "/sys/fs/cgroup", "cgroup2", "rw"),
        _FakePart("/dev/sdb1", "/home", "ext4", "rw"),
    ]
    monkeypatch.setattr(system_stats, "_HAS_PSUTIL", True)
    monkeypatch.setattr(
        system_stats,
        "psutil",
        SimpleNamespace(disk_partitions=lambda all=False: fake_partitions),
    )

    mounts = system_stats._iter_real_mounts()
    mount_points = [m.as_posix() for _, m in mounts]
    assert mount_points == ["/", "/data", "/home"]
    # labels should mirror the mount points so the dashboard renders friendly names
    assert all(label == mp for (label, _), mp in zip(mounts, mount_points, strict=True))


# --------------------------------------------------------------------------- #
# B. Disk IO rate                                                             #
# --------------------------------------------------------------------------- #


class _IoCounters:
    def __init__(self, rb: int, wb: int, rc: int, wc: int) -> None:
        self.read_bytes = rb
        self.write_bytes = wb
        self.read_count = rc
        self.write_count = wc


@pytest.fixture
def reset_disk_state() -> Any:
    system_stats._last_disk_sample = None
    system_stats._last_perdisk_sample.clear()
    yield
    system_stats._last_disk_sample = None
    system_stats._last_perdisk_sample.clear()


def test_disk_io_rate_is_diff_based(
    monkeypatch: pytest.MonkeyPatch, reset_disk_state: Any
) -> None:
    """Two samples 2s apart with +20MB read / +10MB write -> 10MB/s + 5MB/s."""
    counters_seq = [
        _IoCounters(rb=1_000_000, wb=2_000_000, rc=10, wc=5),
        _IoCounters(rb=21_000_000, wb=12_000_000, rc=30, wc=15),
    ]
    times = iter([100.0, 102.0])

    def fake_disk_io(perdisk: bool = False) -> Any:
        if perdisk:
            return {}
        return counters_seq.pop(0)

    monkeypatch.setattr(system_stats, "_HAS_PSUTIL", True)
    monkeypatch.setattr(
        system_stats,
        "psutil",
        SimpleNamespace(disk_io_counters=fake_disk_io),
    )
    monkeypatch.setattr(system_stats.time, "monotonic", lambda: next(times))

    first = system_stats._collect_disk_io()
    assert first is not None
    # First call: no prior sample, rates are zero.
    assert first.read_bytes_per_sec == 0.0
    assert first.write_bytes_per_sec == 0.0

    second = system_stats._collect_disk_io()
    assert second is not None
    assert second.read_bytes_total == 21_000_000
    assert second.write_bytes_total == 12_000_000
    assert second.read_bytes_per_sec == pytest.approx(10_000_000.0)
    assert second.write_bytes_per_sec == pytest.approx(5_000_000.0)
    assert second.read_ops_per_sec == pytest.approx(10.0)
    assert second.write_ops_per_sec == pytest.approx(5.0)


def test_disk_io_per_device(
    monkeypatch: pytest.MonkeyPatch, reset_disk_state: Any
) -> None:
    """per_device entries diff against their previous per-device sample."""
    agg_seq = [
        _IoCounters(rb=0, wb=0, rc=0, wc=0),
        _IoCounters(rb=30_000_000, wb=0, rc=0, wc=0),
    ]
    perdisk_seq = [
        {
            "sda": _IoCounters(rb=10_000_000, wb=0, rc=0, wc=0),
            "nvme0n1": _IoCounters(rb=0, wb=0, rc=0, wc=0),
        },
        {
            "sda": _IoCounters(rb=10_000_000, wb=0, rc=0, wc=0),
            "nvme0n1": _IoCounters(rb=20_000_000, wb=0, rc=0, wc=0),
        },
    ]
    times = iter([0.0, 4.0])

    def fake_disk_io(perdisk: bool = False) -> Any:
        if perdisk:
            return perdisk_seq.pop(0)
        return agg_seq.pop(0)

    monkeypatch.setattr(system_stats, "_HAS_PSUTIL", True)
    monkeypatch.setattr(
        system_stats,
        "psutil",
        SimpleNamespace(disk_io_counters=fake_disk_io),
    )
    monkeypatch.setattr(system_stats.time, "monotonic", lambda: next(times))

    system_stats._collect_disk_io()  # prime
    second = system_stats._collect_disk_io()
    assert second is not None
    by_device = {d.device: d for d in second.per_device}
    assert by_device["sda"].read_bytes_per_sec == 0.0
    # 20MB across 4s window -> 5MB/s
    assert by_device["nvme0n1"].read_bytes_per_sec == pytest.approx(5_000_000.0)


# --------------------------------------------------------------------------- #
# C. GPU PCIe + clock fields                                                  #
# --------------------------------------------------------------------------- #


def test_gpu_pcie_fields_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    """nvidia-smi --query-gpu with PCIe + clocks columns must populate GpuStats."""
    line = (
        "0, NVIDIA GeForce RTX 4090, 555.42, 24576, 1024, 23552, "
        "37, 56, 102.34, 450.0, 30, "
        "4, 16, 4, 16, "
        "1980, 10501, 2520, 10501"
    )
    fake_completed = SimpleNamespace(returncode=0, stdout=line + "\n", stderr="")

    monkeypatch.setattr(system_stats, "_find_nvidia_smi", lambda: "nvidia-smi")
    monkeypatch.setattr(
        system_stats.subprocess,
        "run",
        lambda *a, **kw: fake_completed,
    )

    gpus = system_stats._collect_nvidia_gpus()
    assert len(gpus) == 1
    g = gpus[0]
    assert g.name == "NVIDIA GeForce RTX 4090"
    assert g.pcie_gen_current == 4
    assert g.pcie_width_current == 16
    assert g.pcie_gen_max == 4
    assert g.pcie_width_max == 16
    assert g.sm_clock_mhz == 1980
    assert g.mem_clock_mhz == 10501
    assert g.sm_clock_max_mhz == 2520
    assert g.mem_clock_max_mhz == 10501


def test_gpu_pcie_fields_handle_n_a(monkeypatch: pytest.MonkeyPatch) -> None:
    """Older drivers report `[N/A]` for unsupported columns; we fold to None."""
    line = (
        "0, NVIDIA Tesla T4, 470.57, 15360, 0, 15360, "
        "0, 30, 25.0, 70.0, [N/A], "
        "[N/A], [N/A], [N/A], [N/A], "
        "[N/A], [N/A], [N/A], [N/A]"
    )
    fake_completed = SimpleNamespace(returncode=0, stdout=line + "\n", stderr="")

    monkeypatch.setattr(system_stats, "_find_nvidia_smi", lambda: "nvidia-smi")
    monkeypatch.setattr(
        system_stats.subprocess,
        "run",
        lambda *a, **kw: fake_completed,
    )

    g = system_stats._collect_nvidia_gpus()[0]
    assert g.pcie_gen_current is None
    assert g.sm_clock_mhz is None
    assert g.mem_clock_max_mhz is None


# --------------------------------------------------------------------------- #
# D. GPU per-process VRAM                                                     #
# --------------------------------------------------------------------------- #


def test_gpu_processes_maps_uuid_to_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """First nvidia-smi call returns the index/uuid map; second the compute apps."""
    uuid_a = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    uuid_b = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    map_stdout = f"0, {uuid_a}\n1, {uuid_b}\n"
    apps_stdout = (
        f"{uuid_a}, 12345, python train.py, 4096\n"
        f"{uuid_b}, 67890, ComfyUI, 8192\n"
        # Unknown UUID must be silently dropped.
        "GPU-zzzzzzzz, 99999, ghost, 1024\n"
    )

    seq = iter(
        [
            SimpleNamespace(returncode=0, stdout=map_stdout, stderr=""),
            SimpleNamespace(returncode=0, stdout=apps_stdout, stderr=""),
        ]
    )

    monkeypatch.setattr(system_stats, "_find_nvidia_smi", lambda: "nvidia-smi")
    monkeypatch.setattr(
        system_stats.subprocess,
        "run",
        lambda *a, **kw: next(seq),
    )

    procs = system_stats._collect_gpu_processes()
    assert len(procs) == 2
    by_pid = {p.pid: p for p in procs}
    assert by_pid[12345].gpu_index == 0
    assert by_pid[12345].used_memory_mib == 4096
    assert by_pid[12345].process_name == "python train.py"
    assert by_pid[67890].gpu_index == 1
    assert by_pid[67890].used_memory_mib == 8192
    assert all(p.type == "C" for p in procs)


def test_gpu_processes_empty_when_smi_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system_stats, "_find_nvidia_smi", lambda: None)
    assert system_stats._collect_gpu_processes() == []
