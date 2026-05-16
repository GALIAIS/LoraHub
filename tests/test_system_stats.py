"""Unit tests for the CPU / process collectors in lorahub.api.system_stats.

These tests use monkeypatch + tmp_path to drive the platform-specific code
paths deterministically (Windows hosts can verify the Linux /proc/cpuinfo
parser, etc.) so the suite stays meaningful in CI.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lorahub.api import system_stats


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
