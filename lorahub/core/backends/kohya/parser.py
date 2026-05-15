"""Parse kohya_ss/sd-scripts stdout lines into structured TrainingEvent objects.

Kohya's stdout mixes tqdm progress bars, plain log lines, and ad-hoc status
messages. We pattern-match the lines we recognize and emit `log` events for
everything else so nothing is silently dropped.
"""

from __future__ import annotations

import re

from lorahub.core.events import EventType, TrainingEvent

_STEP_RE = re.compile(
    r"steps:\s*\d+%\|[^|]*\|\s*(?P<cur>\d+)/(?P<total>\d+)"
    r"(?:.*?avr_loss=(?P<loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?",
)

_EPOCH_RE = re.compile(r"^epoch\s+(?P<cur>\d+)\s*/\s*(?P<total>\d+)\s*$", re.IGNORECASE)

_SAVE_RE = re.compile(
    r"\b(?:saving|saved)\b[^:\n]*?"
    r"(?::\s*|\sat\s+|\sas\s+|\sto\s+)"
    r"(?P<path>[^\n]+?\.safetensors)\b",
    re.IGNORECASE,
)

_SAMPLE_RE = re.compile(
    r"\bsample\b[^:\n]*?"
    r"(?::\s*|\sat\s+|\sas\s+|\sto\s+)"
    r"(?P<path>[^\n]+?\.(?:png|jpg|jpeg|webp))\b",
    re.IGNORECASE,
)


def parse_line(line: str, *, job_id: str | None = None) -> TrainingEvent | None:
    """Return a `TrainingEvent` for `line`, or `None` to drop it.

    `None` is reserved for empty / whitespace-only lines so callers can
    cheaply skip them without allocating an event.
    """
    stripped = line.rstrip("\r\n")
    if not stripped.strip():
        return None

    if (m := _STEP_RE.search(stripped)) is not None:
        payload: dict[str, object] = {
            "step": int(m.group("cur")),
            "total_steps": int(m.group("total")),
        }
        if (loss := m.group("loss")) is not None:
            payload["loss"] = float(loss)
        return TrainingEvent(type=EventType.step, payload=payload, job_id=job_id)

    if (m := _EPOCH_RE.match(stripped)) is not None:
        return TrainingEvent(
            type=EventType.epoch_end,
            payload={"epoch": int(m.group("cur")), "total_epochs": int(m.group("total"))},
            job_id=job_id,
        )

    if (m := _SAVE_RE.search(stripped)) is not None:
        return TrainingEvent(
            type=EventType.checkpoint_saved,
            payload={"path": m.group("path")},
            job_id=job_id,
        )

    if (m := _SAMPLE_RE.search(stripped)) is not None:
        return TrainingEvent(
            type=EventType.sample_ready,
            payload={"path": m.group("path")},
            job_id=job_id,
        )

    level = "error" if _looks_like_error(stripped) else "info"
    return TrainingEvent(
        type=EventType.log,
        payload={"level": level, "message": stripped},
        job_id=job_id,
    )


def _looks_like_error(line: str) -> bool:
    lowered = line.lower()
    return (
        "error" in lowered
        or "traceback" in lowered
        or "out of memory" in lowered
        or "cuda error" in lowered
    )
