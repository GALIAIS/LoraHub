"""Tests for lorahub.api.system_stats 鈥?combined CPU + network test suite.

These tests monkeypatch psutil, /proc/cpuinfo and urllib so they run
deterministically on hosts without network access, special interfaces, or
matching kernel layouts. Both the CPU collector enhancements (model
string / frequency range / top processes) and the network enhancements
(per-NIC counters / TCP stats / public IP cache) are covered here.
"""

from __future__ import annotations

import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lorahub.api import system_stats


# --------------------------------------------------------------------------- #
# Fixtures and helpers                                                        #
# --------------------------------------------------------------------------- #


def _reset_module_state() -> None:
    """Wipe rolling caches between tests so each one starts from a clean slate."""
    system_stats._last_net_sample = None
    system_stats._last_pernic_sample.clear()
    system_stats._public_ip_cache = None
    system_stats._public_ip_cache_monotonic = None


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
