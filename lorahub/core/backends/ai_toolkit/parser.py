"""Parse ai-toolkit logs into LoraHub training events."""

from __future__ import annotations

import re

from lorahub.core.events import EventType, TrainingEvent

_STEP_RE = re.compile(
    r"(?:(?P<step>\d+)\s*/\s*(?P<total>\d+)|step[=: ]+(?P<step2>\d+))"
    r".*?\bloss[:= ]+(?P<loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
_SAVE_RE = re.compile(r"\bSaved checkpoint to\s+(?P<path>.+?)\s*$", re.IGNORECASE)


def parse_line(line: str, *, job_id: str | None = None) -> TrainingEvent | None:
    stripped = line.rstrip("\r\n")
    if not stripped.strip():
        return None
    if (m := _SAVE_RE.search(stripped)) is not None:
        return TrainingEvent(
            type=EventType.checkpoint_saved,
            payload={"path": m.group("path")},
            job_id=job_id,
        )
    if (m := _STEP_RE.search(stripped)) is not None:
        payload: dict[str, object] = {
            "step": int(m.group("step") or m.group("step2")),
            "loss": float(m.group("loss")),
        }
        if m.group("total"):
            payload["total_steps"] = int(m.group("total"))
        return TrainingEvent(type=EventType.step, payload=payload, job_id=job_id)

    level = "error" if _looks_like_error(stripped) else "info"
    return TrainingEvent(
        type=EventType.log,
        payload={"level": level, "message": stripped},
        job_id=job_id,
    )


def _looks_like_error(line: str) -> bool:
    lowered = line.lower()
    return "error" in lowered or "traceback" in lowered or "out of memory" in lowered


__all__ = ["parse_line"]
