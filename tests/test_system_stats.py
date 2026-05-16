"""Tests for the network-side enhancements in lorahub.api.system_stats.

These tests monkeypatch psutil and urllib so they run deterministically on
hosts without network access or real interfaces.
"""

from __future__ import annotations

import socket
from types import SimpleNamespace

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
