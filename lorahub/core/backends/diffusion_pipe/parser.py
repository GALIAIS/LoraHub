"""Parse diffusion-pipe stdout/stderr lines into structured TrainingEvent objects.

diffusion-pipe drives training through DeepSpeed, so its terminal output is a
mix of:
  * DeepSpeed's own stepwise stats lines: ``[INFO] [.../engine.py:...] [Rank 0]
    step=42, skipped=0, lr=[1e-4], mom=[(0.9, 0.99)]``
  * Print statements from `train.py` / `utils/saver.py` such as
    "Started new epoch: 3" and "Saving model to directory epoch5"
  * Plain log noise (tqdm bars during caching, warnings, etc.)

We pattern-match the lines we recognise and emit ``log`` events for the rest
so nothing is silently dropped. The kohya parser does the same job for kohya.
"""

from __future__ import annotations

import re

from lorahub.core.events import EventType, TrainingEvent

# DeepSpeed's per-step engine line. The only mandatory bit is `step=N`; the
# trailing fields (skipped, lr, mom, ...) drift between releases so we keep
# the regex anchored on `step=` and parse what's there.
_STEP_RE = re.compile(
    r"step=(?P<step>\d+)"
    r"(?:.*?lr=\[(?P<lr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?",
)

# `train.py`/`utils/saver.py` use these exact phrasings.
_EPOCH_RE = re.compile(
    r"^Started new epoch:\s*(?P<epoch>\d+)\s*$",
    re.IGNORECASE,
)
_SAVE_RE = re.compile(
    r"^Saving model to directory\s+(?P<path>.+?)\s*$",
    re.IGNORECASE,
)
# Loss is reported on a separate `train/loss <value>` line by some
# wrappers; we accept either ``train/loss <num>`` or ``loss=<num>``.
_LOSS_RE = re.compile(
    r"\b(?:train/loss|loss)\s*[=:]\s*(?P<loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b"
)


def parse_line(line: str, *, job_id: str | None = None) -> TrainingEvent | None:
    """Return a `TrainingEvent` for `line`, or `None` to drop it.

    `None` is reserved for empty / whitespace-only lines so callers can
    cheaply skip them without allocating an event.
    """
    stripped = line.rstrip("\r\n")
    if not stripped.strip():
        return None

    if (m := _EPOCH_RE.match(stripped.strip())) is not None:
        return TrainingEvent(
            type=EventType.epoch_end,
            payload={"epoch": int(m.group("epoch"))},
            job_id=job_id,
        )

    if (m := _SAVE_RE.match(stripped.strip())) is not None:
        return TrainingEvent(
            type=EventType.checkpoint_saved,
            payload={"path": m.group("path")},
            job_id=job_id,
        )

    if (m := _STEP_RE.search(stripped)) is not None:
        payload: dict[str, object] = {"step": int(m.group("step"))}
        if (lr := m.group("lr")) is not None:
            payload["lr"] = float(lr)
        if (loss_match := _LOSS_RE.search(stripped)) is not None:
            payload["loss"] = float(loss_match.group("loss"))
        return TrainingEvent(type=EventType.step, payload=payload, job_id=job_id)

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


__all__ = ["parse_line"]
