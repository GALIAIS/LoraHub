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

# diffusion-pipe (newer releases) prints its own per-step summary line
# separately from the deepspeed engine line: `steps: 30 loss: 0.1808
# iter time (s): 3.662 samples/sec: 1.092`. This is the only line that
# carries the actual training loss, so without recognising it the
# metrics endpoint would never see a loss series. We allow the trailing
# `iter time (s): T` and `samples/sec: R` to be optional so format
# drift doesn't silently break recognition.
_STEPS_LOSS_RE = re.compile(
    r"^\s*steps:\s*(?P<step>\d+)"
    r"\s+loss:\s*(?P<loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"(?:\s+iter time\s*\(s\):\s*(?P<iter>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?"
    r"(?:\s+samples/sec:\s*(?P<sps>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?"
    r"\s*$"
)

# `train.py`/`utils/saver.py` use these exact phrasings.
_EPOCH_RE = re.compile(
    r"^Started new epoch:\s*(?P<epoch>\d+)\s*$",
    re.IGNORECASE,
)
# dp emits two phrasings for checkpoint saves: ``Saving model to directory <p>``
# (utils/saver.py before write) and ``Saved model to <p>`` (after write). We
# recognise both so the caller sees a checkpoint_saved event regardless of
# which one the upstream version logs.
_SAVE_RE = re.compile(
    r"^Sav(?:ing model to directory|ed model to)\s+(?P<path>.+?)\s*$",
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

    # diffusion-pipe per-step summary (the only line carrying loss).
    # Try this BEFORE _STEP_RE so a `steps: ...` line never falls
    # through to the deepspeed-style branch (which can't see loss).
    if (m := _STEPS_LOSS_RE.match(stripped)) is not None:
        payload: dict[str, object] = {
            "step": int(m.group("step")),
            "loss": float(m.group("loss")),
        }
        if (it := m.group("iter")) is not None:
            payload["iter_time_s"] = float(it)
        if (sps := m.group("sps")) is not None:
            payload["samples_per_sec"] = float(sps)
        return TrainingEvent(type=EventType.step, payload=payload, job_id=job_id)

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
