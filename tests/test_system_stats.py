"""Tests for lorahub.api.system_stats — combined CPU + network + disk/GPU suite.

These tests monkeypatch psutil, urllib, /proc/cpuinfo, nvidia-smi and disk
counters so they run deterministically on hosts without those resources.
Covers all three enhancement waves:
  * CPU: model string / per-core frequency range / top-N processes
  * Network: per-NIC counters + kind heuristic / TCP stats / public IP cache
  * Disk + GPU: real-mount filter / IO rate diffs / PCIe + clocks /
    per-process VRAM
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lorahub.api import system_stats
from lorahub.api.system_stats_types import SystemSnapshot



# === net_tests.py ===


# --------------------------------------------------------------------------- #
# Fixtures and helpers                                                        #
# --------------------------------------------------------------------------- #


def _reset_module_state() -> None:
    """Wipe rolling caches between tests so each one starts from a clean slate."""
    system_stats._last_net_sample = None
    system_stats._last_pernic_sample.clear()
    system_stats._public_ip_cache = None
    system_stats._public_ip_cache_monotonic = None
    system_stats.invalidate_snapshot_cache()


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    _reset_module_state()
    yield
    _reset_module_state()


def _fake_io_counters(values: dict[str, tuple[int, int]]) -> dict[str, SimpleNamespace]:
    """Build a fake psutil.net_io_counters(pernic=True) result.

    `values` maps interface name -> (bytes_sent, bytes_recv). We fill in the
    remaining counters with zeros so the dataclass population does not raise.
    """
    return {
        name: SimpleNamespace(
            bytes_sent=sent,
            bytes_recv=recv,
            packets_sent=0,
            packets_recv=0,
            errin=0,
            errout=0,
            dropin=0,
            dropout=0,
        )
        for name, (sent, recv) in values.items()
    }


def test_system_snapshot_types_serialize_nested_stats() -> None:
    from lorahub.api.system_stats_types import (
        CpuStats,
        DiskIoDevice,
        DiskIoStats,
        HostInfo,
        InterfaceAddress,
        MemoryStats,
        NetworkInterfaceStats,
        NetworkStats,
        SystemSnapshot,
    )

    snapshot = SystemSnapshot(
        timestamp=1.0,
        host=HostInfo(hostname="host", system="Linux", release="6", python="3"),
        cpu=CpuStats(cores_logical=2, cores_physical=1, usage_percent=12.5),
        memory=MemoryStats(
            total_bytes=100,
            used_bytes=40,
            available_bytes=60,
            percent=40.0,
        ),
        disks=[],
        gpus=[],
        has_psutil=True,
        has_nvidia_smi=False,
        network=NetworkStats(
            bytes_sent_total=10,
            bytes_recv_total=20,
            bytes_sent_per_sec=1.5,
            bytes_recv_per_sec=2.5,
            interfaces=[
                NetworkInterfaceStats(
                    name="eth0",
                    is_up=True,
                    speed_mbps=1000,
                    mtu=1500,
                    addresses=[InterfaceAddress(family="IPv4", address="10.0.0.2")],
                    bytes_sent_total=10,
                    bytes_recv_total=20,
                    bytes_sent_per_sec=1.5,
                    bytes_recv_per_sec=2.5,
                    packets_sent_total=1,
                    packets_recv_total=2,
                    errors_in=0,
                    errors_out=0,
                    drops_in=0,
                    drops_out=0,
                ),
            ],
        ),
        disk_io=DiskIoStats(
            read_bytes_total=1,
            write_bytes_total=2,
            read_bytes_per_sec=3.0,
            write_bytes_per_sec=4.0,
            read_ops_per_sec=5.0,
            write_ops_per_sec=6.0,
            per_device=[
                DiskIoDevice(
                    device="sda",
                    read_bytes_per_sec=3.0,
                    write_bytes_per_sec=4.0,
                    read_ops_per_sec=5.0,
                    write_ops_per_sec=6.0,
                ),
            ],
        ),
    )

    data = snapshot.to_dict()

    assert data["network"]["interfaces"][0]["addresses"][0]["address"] == "10.0.0.2"
    assert data["disk_io"]["per_device"][0]["device"] == "sda"


# --------------------------------------------------------------------------- #
# Per-NIC categorisation                                                      #
# --------------------------------------------------------------------------- #


def test_collect_network_interfaces_categorizes_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system_stats, "_HAS_PSUTIL", True)
    fake_io = _fake_io_counters(
        {
            "lo": (10, 20),
            "docker0": (1, 2),
            "veth1234": (3, 4),
            "wlan0": (5, 6),
            "eth0": (7, 8),
            "wlp3s0": (9, 10),
        }
    )
    fake_stats = {
        name: SimpleNamespace(isup=True, speed=1000, mtu=1500)
        for name in fake_io
    }
    fake_addrs = {
        "eth0": [
            SimpleNamespace(
                family=socket.AF_INET,
                address="10.0.0.5",
                netmask="255.255.255.0",
                broadcast="10.0.0.255",
            ),
        ],
    }

    fake_psutil = SimpleNamespace(
        net_io_counters=lambda pernic=False: fake_io,
        net_if_stats=lambda: fake_stats,
        net_if_addrs=lambda: fake_addrs,
        AF_LINK=-1,
    )
    monkeypatch.setattr(system_stats, "psutil", fake_psutil)

    interfaces = system_stats._collect_network_interfaces()
    by_name = {i.name: i for i in interfaces}

    assert by_name["lo"].kind == "loopback"
    assert by_name["docker0"].kind == "virtual"
    assert by_name["veth1234"].kind == "virtual"
    assert by_name["wlan0"].kind == "wireless"
    assert by_name["wlp3s0"].kind == "wireless"
    assert by_name["eth0"].kind == "physical"

    eth0 = by_name["eth0"]
    assert eth0.is_up is True
    assert eth0.speed_mbps == 1000
    assert eth0.mtu == 1500
    assert any(a.family == "IPv4" and a.address == "10.0.0.5" for a in eth0.addresses)


# --------------------------------------------------------------------------- #
# Per-NIC rolling rate                                                        #
# --------------------------------------------------------------------------- #


def test_per_nic_rate_is_diff_based(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system_stats, "_HAS_PSUTIL", True)

    counters_sequence: list[dict[str, SimpleNamespace]] = [
        _fake_io_counters({"eth0": (1_000, 2_000)}),
        _fake_io_counters({"eth0": (3_000, 6_000)}),  # +2000 sent, +4000 recv
    ]
    aggregate_sequence = [
        SimpleNamespace(bytes_sent=1_000, bytes_recv=2_000),
        SimpleNamespace(bytes_sent=3_000, bytes_recv=6_000),
    ]
    state = {"call": 0}

    def fake_net_io_counters(pernic: bool = False):
        idx = state["call"]
        if pernic:
            return counters_sequence[idx]
        return aggregate_sequence[idx]

    fake_stats = {"eth0": SimpleNamespace(isup=True, speed=1000, mtu=1500)}
    fake_psutil = SimpleNamespace(
        net_io_counters=fake_net_io_counters,
        net_if_stats=lambda: fake_stats,
        net_if_addrs=lambda: {},
        net_connections=lambda kind="tcp": [],
        AF_LINK=-1,
    )
    monkeypatch.setattr(system_stats, "psutil", fake_psutil)

    monotonic_values = iter([100.0, 100.0, 102.0, 102.0])  # +2s between snapshots

    def fake_monotonic() -> float:
        return next(monotonic_values)

    monkeypatch.setattr(system_stats.time, "monotonic", fake_monotonic)
    # Disable public IP fetch entirely so it can't perturb timing.
    monkeypatch.setattr(system_stats, "_collect_public_ip", lambda: None)

    snap1 = system_stats._collect_network()
    assert snap1 is not None
    assert snap1.interfaces[0].bytes_sent_per_sec == 0.0
    assert snap1.interfaces[0].bytes_recv_per_sec == 0.0

    state["call"] = 1
    snap2 = system_stats._collect_network()
    assert snap2 is not None
    iface = snap2.interfaces[0]
    # 2000 bytes / 2s = 1000 b/s; 4000 / 2 = 2000 b/s
    assert iface.bytes_sent_per_sec == 1000.0
    assert iface.bytes_recv_per_sec == 2000.0
    # Aggregate counters (top-level) match the per-NIC numbers here.
    assert snap2.bytes_sent_per_sec == 1000.0
    assert snap2.bytes_recv_per_sec == 2000.0


# --------------------------------------------------------------------------- #
# TCP connection aggregation                                                  #
# --------------------------------------------------------------------------- #


def test_tcp_connections_categorizes_states(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system_stats, "_HAS_PSUTIL", True)
    fake_conns = [
        SimpleNamespace(status="ESTABLISHED"),
        SimpleNamespace(status="ESTABLISHED"),
        SimpleNamespace(status="LISTEN"),
        SimpleNamespace(status="TIME_WAIT"),
        SimpleNamespace(status="TIME_WAIT"),
        SimpleNamespace(status="TIME_WAIT"),
        SimpleNamespace(status="CLOSE_WAIT"),
        SimpleNamespace(status="SYN_SENT"),
    ]
    fake_psutil = SimpleNamespace(
        net_connections=lambda kind="tcp": fake_conns,
        AF_LINK=-1,
    )
    monkeypatch.setattr(system_stats, "psutil", fake_psutil)

    stats = system_stats._collect_tcp_connections()
    assert stats is not None
    assert stats.total == 8
    assert stats.established == 2
    assert stats.listen == 1
    assert stats.time_wait == 3
    assert stats.close_wait == 1
    assert stats.other == 1  # SYN_SENT


def test_tcp_connections_handles_permission_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system_stats, "_HAS_PSUTIL", True)

    def boom(kind: str = "tcp"):
        raise PermissionError("non-root cannot list system-wide connections")

    fake_psutil = SimpleNamespace(net_connections=boom, AF_LINK=-1)
    monkeypatch.setattr(system_stats, "psutil", fake_psutil)

    assert system_stats._collect_tcp_connections() is None


# --------------------------------------------------------------------------- #
# Public IP caching                                                           #
# --------------------------------------------------------------------------- #


def test_public_ip_caches_for_five_minutes(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = {"n": 0}

    def fake_fetch(url: str) -> str | None:
        call_count["n"] += 1
        return "203.0.113.42"

    monkeypatch.setattr(system_stats, "_fetch_public_ip_once", fake_fetch)

    monotonic_values = iter([1000.0, 1001.0, 1100.0])  # all within 5-min TTL

    def fake_monotonic() -> float:
        return next(monotonic_values)

    monkeypatch.setattr(system_stats.time, "monotonic", fake_monotonic)

    a = system_stats._collect_public_ip()
    b = system_stats._collect_public_ip()
    c = system_stats._collect_public_ip()

    assert call_count["n"] == 1
    assert a is not None and a.ip == "203.0.113.42" and a.source == "ip.sb"
    assert b is not None and b.ip == "203.0.113.42" and b.source == "cached"
    assert c is not None and c.ip == "203.0.113.42" and c.source == "cached"
    # Cache returns the original fetched_at, not the time of the cache hit.
    assert a.fetched_at == b.fetched_at == c.fetched_at

# === cpu_tests.py ===


# --------------------------------------------------------------------------- #
# A. CPU model string                                                         #
# --------------------------------------------------------------------------- #


def test_collect_cpu_model_from_proc_cpuinfo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_cpuinfo = tmp_path / "cpuinfo"
    fake_cpuinfo.write_text(
        "processor\t: 0\n"
        "vendor_id\t: GenuineIntel\n"
        "model name\t: Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz\n"
        "cache size\t: 36864 KB\n"
        "\n"
        "processor\t: 1\n"
        "model name\t: Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz\n",
        encoding="utf-8",
    )

    # Pretend we're on Linux and the canonical /proc/cpuinfo lives in tmp_path.
    monkeypatch.setattr(system_stats.platform, "system", lambda: "Linux")

    real_path = system_stats.Path

    def _path_factory(arg: str = "") -> Path:
        if arg == "/proc/cpuinfo":
            return fake_cpuinfo
        return real_path(arg)

    monkeypatch.setattr(system_stats, "Path", _path_factory)

    assert (
        system_stats._collect_cpu_model()
        == "Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz"
    )


def test_collect_cpu_model_falls_back_when_proc_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(system_stats.platform, "system", lambda: "Linux")
    missing = tmp_path / "does-not-exist"

    real_path = system_stats.Path

    def _path_factory(arg: str = "") -> Path:
        if arg == "/proc/cpuinfo":
            return missing
        return real_path(arg)

    monkeypatch.setattr(system_stats, "Path", _path_factory)
    monkeypatch.setattr(system_stats.platform, "processor", lambda: "")

    assert system_stats._collect_cpu_model() == ""


# --------------------------------------------------------------------------- #
# B. CPU frequency range + per-core list                                      #
# --------------------------------------------------------------------------- #


def test_collect_cpu_frequency_per_core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system_stats, "_HAS_PSUTIL", True)

    per_core_samples = [
        SimpleNamespace(current=2400.0, min=800.0, max=3600.0),
        SimpleNamespace(current=2800.0, min=800.0, max=3600.0),
        SimpleNamespace(current=3200.0, min=800.0, max=3600.0),
        SimpleNamespace(current=3600.0, min=800.0, max=3600.0),
    ]
    aggregate = SimpleNamespace(current=3000.0, min=800.0, max=3600.0)

    def fake_cpu_freq(percpu: bool = False) -> Any:
        return per_core_samples if percpu else aggregate

    monkeypatch.setattr(
        system_stats,
        "psutil",
        SimpleNamespace(cpu_freq=fake_cpu_freq),
    )

    mean, lo, hi, per = system_stats._collect_cpu_frequency()
    assert per == [2400.0, 2800.0, 3200.0, 3600.0]
    assert lo == 800.0
    assert hi == 3600.0
    assert mean == pytest.approx(3000.0)


def test_collect_cpu_frequency_falls_back_when_no_per_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(system_stats, "_HAS_PSUTIL", True)
    aggregate = SimpleNamespace(current=2200.0, min=0.0, max=2200.0)

    def fake_cpu_freq(percpu: bool = False) -> Any:
        return [] if percpu else aggregate

    monkeypatch.setattr(
        system_stats,
        "psutil",
        SimpleNamespace(cpu_freq=fake_cpu_freq),
    )

    mean, lo, hi, per = system_stats._collect_cpu_frequency()
    assert per == []
    # min of 0.0 is reported by psutil as "unknown" -> we coerce to None;
    # max=2200 is genuine, so it should pass through.
    assert lo is None
    assert hi == 2200.0
    assert mean == pytest.approx(2200.0)


# --------------------------------------------------------------------------- #
# C. Top N process list                                                       #
# --------------------------------------------------------------------------- #


class _FakeProc:
    def __init__(self, info: dict[str, Any]) -> None:
        self.info = info


class _DyingProc:
    """Raises NoSuchProcess as soon as anyone touches its info."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    @property
    def info(self) -> dict[str, Any]:
        raise self._exc


def _make_psutil_stub(procs: list[Any]) -> SimpleNamespace:
    class _NoSuchProcess(Exception):
        pass

    class _AccessDenied(Exception):
        pass

    class _ZombieProcess(Exception):
        pass

    return SimpleNamespace(
        process_iter=lambda fields: list(procs),
        NoSuchProcess=_NoSuchProcess,
        AccessDenied=_AccessDenied,
        ZombieProcess=_ZombieProcess,
    )


def test_collect_top_processes_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system_stats, "_HAS_PSUTIL", True)
    system_stats._last_process_cpu.clear()

    procs = [
        _FakeProc(
            {
                "pid": 1010,
                "name": "python3",
                "cpu_percent": 12.5,
                "memory_percent": 7.5,
                "memory_info": SimpleNamespace(rss=300 * 1024 * 1024),
            }
        ),
        _FakeProc(
            {
                "pid": 1020,
                "name": "node",
                "cpu_percent": 5.0,
                "memory_percent": 2.0,
                "memory_info": SimpleNamespace(rss=80 * 1024 * 1024),
            }
        ),
        _FakeProc(
            {
                "pid": 1030,
                "name": "uvicorn",
                "cpu_percent": 30.0,
                "memory_percent": 4.0,
                "memory_info": SimpleNamespace(rss=160 * 1024 * 1024),
            }
        ),
        _FakeProc(
            {
                "pid": 1040,
                "name": "redis-server",
                "cpu_percent": 1.0,
                "memory_percent": 0.5,
                "memory_info": SimpleNamespace(rss=20 * 1024 * 1024),
            }
        ),
        _FakeProc(
            {
                "pid": 1050,
                "name": "postgres",
                "cpu_percent": 8.0,
                "memory_percent": 6.0,
                "memory_info": SimpleNamespace(rss=240 * 1024 * 1024),
            }
        ),
        _FakeProc(
            {
                "pid": 1060,
                "name": "vscode",
                "cpu_percent": 4.0,
                "memory_percent": 9.0,
                "memory_info": SimpleNamespace(rss=400 * 1024 * 1024),
            }
        ),
        # Kernel thread - should be filtered out.
        _FakeProc(
            {
                "pid": 7,
                "name": "[ksoftirqd/0]",
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "memory_info": SimpleNamespace(rss=0),
            }
        ),
        # PID 1 / init - skipped.
        _FakeProc(
            {
                "pid": 1,
                "name": "init",
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "memory_info": SimpleNamespace(rss=0),
            }
        ),
    ]
    monkeypatch.setattr(system_stats, "psutil", _make_psutil_stub(procs))

    top = system_stats._collect_top_processes(5)
    assert len(top) == 5
    assert [p.name for p in top] == [
        "vscode",  # 400 MB
        "python3",  # 300 MB
        "postgres",  # 240 MB
        "uvicorn",  # 160 MB
        "node",  # 80 MB
    ]
    # Field types are correct (frontend assumes ints / floats).
    head = top[0]
    assert isinstance(head.pid, int)
    assert isinstance(head.memory_rss_bytes, int)
    assert isinstance(head.memory_percent, float)
    assert isinstance(head.cpu_percent, float)


def test_collect_top_processes_handles_dead_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(system_stats, "_HAS_PSUTIL", True)
    system_stats._last_process_cpu.clear()

    class _NoSuchProcess(Exception):
        pass

    procs = [
        _DyingProc(_NoSuchProcess("gone")),
        _FakeProc(
            {
                "pid": 2020,
                "name": "python3",
                "cpu_percent": 1.0,
                "memory_percent": 1.0,
                "memory_info": SimpleNamespace(rss=10 * 1024 * 1024),
            }
        ),
    ]
    stub = _make_psutil_stub(procs)
    # The dying proc raises this exact class; rebind it on the stub so the
    # exception handler in _collect_top_processes catches it.
    stub.NoSuchProcess = _NoSuchProcess
    monkeypatch.setattr(system_stats, "psutil", stub)

    top = system_stats._collect_top_processes(5)
    assert len(top) == 1
    assert top[0].name == "python3"


# --------------------------------------------------------------------------- #
# Snapshot integration                                                        #
# --------------------------------------------------------------------------- #


def test_snapshot_has_processes_field() -> None:
    snap = system_stats.collect_snapshot(top_processes_n=3).to_dict()
    assert "processes" in snap
    assert isinstance(snap["processes"], list)
    # New CPU fields must appear too.
    cpu = snap["cpu"]
    for key in (
        "model",
        "frequency_min_mhz",
        "frequency_max_mhz",
        "frequency_per_core_mhz",
    ):
        assert key in cpu


# --------------------------------------------------------------------------- #
# Shared snapshot cache                                                        #
# --------------------------------------------------------------------------- #


def _stub_snapshot(ts: float = 0.0) -> SystemSnapshot:
    """A minimal SystemSnapshot for cache tests.

    Cache tests only exercise the hit/miss/invalidate logic — they never
    read snapshot fields — so the required nested dataclasses are left
    ``None`` to avoid constructing the full probe payload. Using a stub
    (instead of calling the real ``collect_snapshot``) keeps the tests
    fast and offline: the real probe spawns ``nvidia-smi`` and hits the
    public-IP network endpoints, which would both slow CI and risk flaky
    timeouts (and pollute the rolling ``_last_perdisk_sample`` /
    ``_last_process_cpu`` caches for later tests).
    """
    return SystemSnapshot(
        timestamp=ts,
        host=None,  # type: ignore[arg-type]
        cpu=None,  # type: ignore[arg-type]
        memory=None,  # type: ignore[arg-type]
        disks=[],
        gpus=[],
        has_psutil=False,
        has_nvidia_smi=False,
    )


def _patch_snapshot_with_counter(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Replace ``collect_snapshot`` with a stub-returning counter.

    Returns the ``calls`` dict so each test can assert on the probe count.
    """
    calls = {"n": 0}

    def _count(*args: Any, **kwargs: Any) -> SystemSnapshot:
        calls["n"] += 1
        return _stub_snapshot(float(calls["n"]))

    monkeypatch.setattr(system_stats, "collect_snapshot", _count)
    return calls


def test_collect_snapshot_shared_caches_within_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two reads inside the TTL return the same object and probe only once."""
    calls = _patch_snapshot_with_counter(monkeypatch)

    first = system_stats.collect_snapshot_shared(ttl_seconds=1.0)
    second = system_stats.collect_snapshot_shared(ttl_seconds=1.0)

    assert first is second
    assert calls["n"] == 1


def test_collect_snapshot_shared_reprobes_after_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expired cache triggers a fresh probe."""
    calls = _patch_snapshot_with_counter(monkeypatch)
    # Patching ``system_stats.time.monotonic`` patches the global ``time``
    # module, but no real probe runs here (the stub returns instantly with
    # no subprocess), so there is no ``subprocess.run`` timeout to disable.
    monkeypatch.setattr(system_stats.time, "monotonic", lambda: 0.0)

    system_stats.collect_snapshot_shared(ttl_seconds=1.0)
    # Advance past the TTL.
    monkeypatch.setattr(system_stats.time, "monotonic", lambda: 2.0)
    system_stats.collect_snapshot_shared(ttl_seconds=1.0)

    assert calls["n"] == 2


def test_collect_snapshot_shared_invalidate_drops_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """invalidate_snapshot_cache forces the next read to re-probe."""
    calls = _patch_snapshot_with_counter(monkeypatch)

    system_stats.collect_snapshot_shared(ttl_seconds=1.0)
    system_stats.invalidate_snapshot_cache()
    system_stats.collect_snapshot_shared(ttl_seconds=1.0)

    assert calls["n"] == 2


def test_collect_snapshot_shared_nonblocking_seeds_then_refreshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(system_stats, "collect_summary_snapshot", lambda: _stub_snapshot(0.0))

    def fake_collect() -> SystemSnapshot:
        started.set()
        assert release.wait(2)
        return _stub_snapshot(1.0)

    monkeypatch.setattr(system_stats, "collect_snapshot", fake_collect)

    first = system_stats.collect_snapshot_shared(block_on_miss=False)

    assert first.timestamp == 0.0
    assert started.wait(1)
    release.set()

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        refreshed = system_stats.collect_snapshot_shared(block_on_miss=False)
        if refreshed.timestamp == 1.0:
            break
        time.sleep(0.01)
    else:
        pytest.fail("background refresh did not publish the full snapshot")


def test_collect_snapshot_shared_refresh_does_not_block_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    monkeypatch.setattr(system_stats, "collect_summary_snapshot", lambda: _stub_snapshot(0.0))

    def fake_collect() -> SystemSnapshot:
        started.set()
        assert release.wait(2)
        return _stub_snapshot(1.0)

    monkeypatch.setattr(system_stats, "collect_snapshot", fake_collect)
    worker = threading.Thread(
        target=lambda: system_stats.collect_snapshot_shared(block_on_miss=True),
        daemon=True,
    )
    worker.start()
    assert started.wait(1)

    t0 = time.perf_counter()
    summary = system_stats.collect_snapshot_shared(block_on_miss=False)
    elapsed = time.perf_counter() - t0
    release.set()
    worker.join(2)

    assert summary.timestamp == 0.0
    assert elapsed < 0.1


# === disk_tests.py ===


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


def test_external_gpu_probe_hides_windows_console(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(system_stats.platform, "system", lambda: "Windows")
    monkeypatch.setattr(system_stats.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)

    def fake_run(*_args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(system_stats.subprocess, "run", fake_run)

    system_stats._run_hidden(["nvidia-smi"], check=False)

    assert calls[0]["creationflags"] == 0x08000000


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


def test_rocm_smi_parses_amd_gpu_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "card0": {
            "Card series": "AMD Radeon RX 7900 XTX",
            "Driver version": "6.1.1",
            "VRAM Total Memory (B)": "25757220864",
            "VRAM Total Used Memory (B)": "2147483648",
            "GPU use (%)": "73",
            "Temperature (Sensor edge) (C)": "62.0",
            "Average Graphics Package Power (W)": "285.5",
            "Fan Speed (%)": "44",
        }
    }
    monkeypatch.setattr(system_stats, "_find_rocm_smi", lambda: "rocm-smi")
    monkeypatch.setattr(
        system_stats.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr=""),
    )

    gpu = system_stats._collect_amd_rocm_gpus()[0]

    assert gpu.vendor == "amd"
    assert gpu.name == "AMD Radeon RX 7900 XTX"
    assert gpu.driver == "6.1.1"
    assert gpu.memory_total_bytes == 25757220864
    assert gpu.memory_used_bytes == 2147483648
    assert gpu.utilization_percent == 73
    assert gpu.temperature_c == 62.0
    assert gpu.power_w == 285.5
    assert gpu.fan_percent == 44


def test_amdgpu_top_parses_linux_amd_gpu_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "devices": [
            {
                "Info": {"DeviceName": "AMD Radeon RX 6600"},
                "VRAM": {
                    "Total": {"value": 8, "unit": "GB"},
                    "Usage": {"value": 1536, "unit": "MB"},
                },
                "gpu_activity": {"GFX": {"value": 51, "unit": "%"}},
                "Sensors": {
                    "Temperature": {"value": 63, "unit": "C"},
                    "Power": {"value": 87.5, "unit": "W"},
                    "Fan": {"value": 35, "unit": "%"},
                },
            }
        ]
    }
    monkeypatch.setattr(system_stats, "_find_amdgpu_top", lambda: "amdgpu_top")
    monkeypatch.setattr(
        system_stats.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(payload) + "\n", stderr=""),
    )

    gpu = system_stats._collect_amd_amdgpu_top_gpus()[0]

    assert gpu.vendor == "amd"
    assert gpu.name == "AMD Radeon RX 6600"
    assert gpu.memory_total_bytes == 8 * 1024**3
    assert gpu.memory_used_bytes == 1536 * 1024**2
    assert gpu.utilization_percent == 51
    assert gpu.temperature_c == 63
    assert gpu.power_w == 87.5
    assert gpu.fan_percent == 35


def test_windows_cim_parses_amd_gpu_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "Name": "AMD Radeon RX 7900 XTX",
        "AdapterCompatibility": "Advanced Micro Devices, Inc.",
        "AdapterRAM": 4293918720,
        "DriverVersion": "31.0.24027.1012",
    }
    monkeypatch.setattr(system_stats.shutil, "which", lambda name: "powershell.exe")
    monkeypatch.setattr(system_stats, "_find_nvidia_smi", lambda: None)
    monkeypatch.setattr(
        system_stats.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr=""),
    )

    gpu = system_stats._collect_windows_video_gpus()[0]

    assert gpu.vendor == "amd"
    assert gpu.name == "AMD Radeon RX 7900 XTX"
    assert gpu.driver == "31.0.24027.1012"
    assert gpu.memory_total_bytes == 4293918720
    assert gpu.utilization_percent is None


def test_windows_nwinfo_parses_amd_gpu_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "GPU": [
            {
                "Name": "AMD Radeon RX 7900 XTX",
                "Vendor": "AMD",
                "Driver Version": "32.0.12027.9001",
                "VRAM Total": "24 GB",
                "VRAM Used": "3.5 GB",
                "GPU Usage": "67 %",
                "Temperature": "58 C",
                "Power Draw": "212 W",
                "Fan Speed": "41 %",
                "Core Clock": "2450 MHz",
                "Memory Clock": "2500 MHz",
            }
        ]
    }
    monkeypatch.setattr(system_stats, "_find_nwinfo", lambda: "nwinfo.exe")
    monkeypatch.setattr(system_stats, "_find_nvidia_smi", lambda: None)
    monkeypatch.setattr(
        system_stats.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr=""),
    )

    gpu = system_stats._collect_windows_nwinfo_gpus()[0]

    assert gpu.vendor == "amd"
    assert gpu.name == "AMD Radeon RX 7900 XTX"
    assert gpu.driver == "32.0.12027.9001"
    assert gpu.memory_total_bytes == 24 * 1024**3
    assert gpu.memory_used_bytes == int(3.5 * 1024**3)
    assert gpu.utilization_percent == 67
    assert gpu.temperature_c == 58
    assert gpu.power_w == 212
    assert gpu.fan_percent == 41
    assert gpu.sm_clock_mhz == 2450


def test_windows_gpu_collection_falls_back_to_cim_without_nwinfo(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "Name": "AMD Radeon RX 7800 XT",
        "AdapterCompatibility": "AMD",
        "AdapterRAM": 17171480576,
        "DriverVersion": "31.0.24027.1012",
    }
    monkeypatch.setattr(system_stats.platform, "system", lambda: "Windows")
    monkeypatch.setattr(system_stats, "_collect_nvidia_gpus", lambda start_index=0: [])
    monkeypatch.setattr(system_stats, "_find_nwinfo", lambda: None)
    monkeypatch.setattr(system_stats.shutil, "which", lambda name: "powershell.exe")
    monkeypatch.setattr(system_stats, "_find_nvidia_smi", lambda: None)
    monkeypatch.setattr(
        system_stats.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr=""),
    )

    gpu = system_stats._collect_gpus()[0]

    assert gpu.vendor == "amd"
    assert gpu.name == "AMD Radeon RX 7800 XT"


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


def test_linux_drm_skips_bmc_display_adapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cloud hosts expose ASPEED BMC display cards; they are not training GPUs."""
    drm = tmp_path / "sys" / "class" / "drm"
    card = drm / "card0"
    device = card / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1a03", encoding="utf-8")
    (device / "uevent").write_text("DRIVER=ast\nPCI_ID=1A03:2000\n", encoding="utf-8")

    real_path = system_stats.Path

    def path_factory(value: str) -> Path:
        if value == "/sys/class/drm":
            return drm
        return real_path(value)

    monkeypatch.setattr(system_stats, "Path", path_factory)
    monkeypatch.setattr(system_stats, "_find_nvidia_smi", lambda: None)

    assert system_stats._collect_linux_drm_gpus() == []


def test_linux_drm_skips_amd_when_rocm_smi_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drm = tmp_path / "sys" / "class" / "drm"
    card = drm / "card0"
    device = card / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002", encoding="utf-8")
    (device / "uevent").write_text("DRIVER=amdgpu\nPCI_ID=1002:744C\n", encoding="utf-8")

    real_path = system_stats.Path
    monkeypatch.setattr(system_stats, "Path", lambda value: drm if value == "/sys/class/drm" else real_path(value))
    monkeypatch.setattr(system_stats, "_find_rocm_smi", lambda: "rocm-smi")

    assert system_stats._collect_linux_drm_gpus() == []
