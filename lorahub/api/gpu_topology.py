"""GPU grouping helpers for scheduler dispatch."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GpuSlotInfo:
    index: int
    name: str
    memory_total_bytes: int | None
    compute_capability: str | None


def nvidia_slots() -> list[GpuSlotInfo]:
    try:
        from lorahub.api.system_stats import _collect_nvidia_gpus  # noqa: PLC0415

        gpus = _collect_nvidia_gpus()
    except Exception:  # noqa: BLE001
        return []
    return [
        GpuSlotInfo(
            index=g.index,
            name=g.name.strip(),
            memory_total_bytes=g.memory_total_bytes,
            compute_capability=g.compute_capability,
        )
        for g in gpus
    ]


def homogeneous_slot_groups(slots: list[GpuSlotInfo] | None = None) -> list[list[int]]:
    slots = nvidia_slots() if slots is None else slots
    groups: dict[tuple[str, int | None, str | None], list[int]] = {}
    for gpu in slots:
        key = (gpu.name, gpu.memory_total_bytes, gpu.compute_capability)
        groups.setdefault(key, []).append(gpu.index)
    return [sorted(v) for v in groups.values()]


__all__ = ["GpuSlotInfo", "homogeneous_slot_groups", "nvidia_slots"]
