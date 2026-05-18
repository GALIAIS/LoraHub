"""Parse anima_lora stdout/stderr lines into structured TrainingEvent objects.

anima_lora is the third backend; its training output is a mix of:

* tqdm progress bars from `library/training/loop.py:224`:
  ``steps: 17%|##2 | 51/300 [00:30<02:30, 1.67it/s, avr_loss=0.243, lr=5e-05]``
  Updated every step via ``progress_bar.set_postfix(refresh=False, ...)``.
  This is the only line carrying loss + step.
* ``library/datasets/base.py:212``:
  ``epoch is incremented. current_epoch: 1, epoch: 2``
* ``library/training/checkpoints.py``:
  ``saving checkpoint: <path>`` / ``saving model: <dir>``
  (one is INFO logger output, the other is `accelerator.print` —
  same regex catches both prefixes).
* Stack-formatted python logging from `library.log.setup_logging`
  (we treat them as ``log`` events).
* Plain print noise (cache progress, model loads, warnings).

We pattern-match what we need and emit ``log`` events for the rest so
nothing is silently dropped — same posture as the kohya / dp parsers.
"""

from __future__ import annotations

import re

from lorahub.core.events import EventType, TrainingEvent

# tqdm step bar — primary loss + step source.
# tqdm carriage-returns ``\r`` are stripped before this regex sees the
# line. The regex anchors on ``steps: <pct>%|<bar> | <step>/<total>``;
# step + total are mandatory, the suffixed ``avr_loss=`` and ``lr=``
# kwargs are extracted by separate regex passes so a stray formatting
# tweak only loses one signal, not the whole step event.
_TQDM_STEPS_RE = re.compile(
    r"steps:\s*\d+%\|[^|]*\|\s*(?P<step>\d+)/(?P<total>\d+)",
)
_TQDM_LOSS_RE = re.compile(
    r"avr_loss=(?P<loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
_TQDM_LR_RE = re.compile(
    r"\blr=(?P<lr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

# Epoch transition: ``library/datasets/base.py:212`` logger.info.
_EPOCH_RE = re.compile(
    r"epoch is incremented\.\s*current_epoch:\s*\d+,\s*epoch:\s*(?P<epoch>\d+)",
    re.IGNORECASE,
)

# Checkpoint saved: matches both
#   ``saving checkpoint: /abs/path/foo.safetensors``
#   ``saving model: /abs/path/dirname``
# Allowed prefix INFO/WARNING/etc. so logging-formatter banners don't
# block the match.
_SAVE_RE = re.compile(
    r"saving (?:checkpoint|model)(?:\s+state)?(?:\s+to)?:\s*(?P<path>\S.*)$",
    re.IGNORECASE,
)

# Validation loss reported by the val dataloader sweep — ``loop.py``
# constructs ``logs = {"avr_loss": ...}`` for the val pass too, so the
# tqdm regex catches both. The ``val_loss`` keyed event is emitted by
# the val_loss_recorder line further down. We support either phrasing
# because upstream's exact wording drifted across releases.
_VAL_LOSS_RE = re.compile(
    r"\bval(?:idation)?[\s_/]?loss\s*[=:]\s*(?P<loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"(?:.*?\bepoch\s*[=:]\s*(?P<epoch>\d+))?"
    r"(?:.*?\bstep\s*[=:]\s*(?P<step>\d+))?",
    re.IGNORECASE,
)


def parse_line(line: str, *, job_id: str | None = None) -> TrainingEvent | None:
    """Return a `TrainingEvent` for `line`, or `None` to drop it.

    `None` is reserved for empty / whitespace-only lines so callers can
    cheaply skip them without allocating an event.
    """
    stripped = line.rstrip("\r\n").strip()
    if not stripped:
        return None

    # Order matters: validation-loss lines often co-occur with a step
    # number, but they're a different event type (``validation`` vs
    # ``step``). Match them first so a val pass doesn't get filed as a
    # training step.
    if (m := _VAL_LOSS_RE.search(stripped)) is not None:
        payload: dict[str, object] = {"val_loss": float(m.group("loss"))}
        if (epoch := m.group("epoch")) is not None:
            payload["epoch"] = int(epoch)
        if (step := m.group("step")) is not None:
            payload["step"] = int(step)
        return TrainingEvent(
            type=EventType.validation, payload=payload, job_id=job_id
        )

    if (m := _EPOCH_RE.search(stripped)) is not None:
        return TrainingEvent(
            type=EventType.epoch_end,
            payload={"epoch": int(m.group("epoch"))},
            job_id=job_id,
        )

    if (m := _SAVE_RE.search(stripped)) is not None:
        return TrainingEvent(
            type=EventType.checkpoint_saved,
            payload={"path": m.group("path").strip()},
            job_id=job_id,
        )

    if (m := _TQDM_STEPS_RE.search(stripped)) is not None:
        payload = {
            "step": int(m.group("step")),
            "total_steps": int(m.group("total")),
        }
        if (lm := _TQDM_LOSS_RE.search(stripped)) is not None:
            payload["loss"] = float(lm.group("loss"))
        if (rm := _TQDM_LR_RE.search(stripped)) is not None:
            payload["lr"] = float(rm.group("lr"))
        return TrainingEvent(type=EventType.step, payload=payload, job_id=job_id)

    level = "error" if _looks_like_error(stripped) else "info"
    return TrainingEvent(
        type=EventType.log,
        payload={"level": level, "message": stripped},
        job_id=job_id,
    )


# Patterns that look like errors but really aren't — clean shutdowns,
# benign warnings. Same posture as dp parser; keep the list short.
_CANCEL_HINTS = (
    "keyboardinterrupt",
    "process group received signal",
    "exits with return code = -2",
    "exits with return code = -9",
    "exits with return code = -15",
)


def _looks_like_error(line: str) -> bool:
    lowered = line.lower()
    if any(h in lowered for h in _CANCEL_HINTS):
        return False
    return (
        "error" in lowered
        or "out of memory" in lowered
        or "cuda error" in lowered
        or "traceback" in lowered
    )


__all__ = ["parse_line"]
