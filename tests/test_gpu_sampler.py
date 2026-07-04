from __future__ import annotations

from typing import Any

from lorahub.api.jobs_helpers.preview import _gpu_sampler_loop
from lorahub.api.system_stats_types import GpuStats, SystemSnapshot
from lorahub.core.events import EventType, TrainingEvent


class _OneTickStop:
    def __init__(self) -> None:
        self.calls = 0

    def wait(self, _timeout: float) -> bool:
        self.calls += 1
        return self.calls > 1


def _gpu(index: int, util: float) -> GpuStats:
    return GpuStats(
        index=index,
        name=f"GPU {index}",
        driver="555",
        memory_total_bytes=24 * 1024 * 1024,
        memory_used_bytes=(index + 1) * 1024 * 1024,
        memory_free_bytes=None,
        utilization_percent=util,
        temperature_c=50 + index,
        power_w=None,
        power_limit_w=None,
        fan_percent=None,
        vendor="nvidia",
    )


def _snapshot(gpus: list[GpuStats]) -> SystemSnapshot:
    """Minimal SystemSnapshot — only ``gpus`` is read by the sampler loop."""
    return SystemSnapshot(
        timestamp=0.0,
        host=None,  # type: ignore[arg-type]
        cpu=None,  # type: ignore[arg-type]
        memory=None,  # type: ignore[arg-type]
        disks=[],
        gpus=gpus,
        has_psutil=False,
        has_nvidia_smi=True,
    )


def test_gpu_sampler_emits_all_assigned_slots(monkeypatch: Any) -> None:
    from lorahub.api import system_stats

    # The sampler now reads GPUs from the shared snapshot cache rather than
    # calling _collect_nvidia_gpus directly, so patch the cache entry point.
    monkeypatch.setattr(
        system_stats,
        "collect_snapshot_shared",
        lambda **_: _snapshot([_gpu(0, 12.0), _gpu(1, 87.0)]),
    )
    events: list[TrainingEvent] = []

    _gpu_sampler_loop(
        "job-1",
        [0, 1],
        events.append,
        _OneTickStop(),  # type: ignore[arg-type]
    )

    assert [event.type for event in events] == [
        EventType.gpu_sample,
        EventType.gpu_sample,
    ]
    assert [event.payload["gpu_index"] for event in events] == [0, 1]
    assert [event.payload["util_percent"] for event in events] == [12.0, 87.0]
