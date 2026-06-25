from __future__ import annotations

from typing import Any

from lorahub.api.jobs_helpers.preview import _gpu_sampler_loop
from lorahub.api.system_stats_types import GpuStats
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


def test_gpu_sampler_emits_all_assigned_slots(monkeypatch: Any) -> None:
    from lorahub.api import system_stats

    monkeypatch.setattr(
        system_stats,
        "_collect_nvidia_gpus",
        lambda: [_gpu(0, 12.0), _gpu(1, 87.0)],
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
