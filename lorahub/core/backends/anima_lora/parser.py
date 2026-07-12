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

import math
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
_FLOAT_RE = r"[-+]?(?:\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf(?:inity)?)"
_TQDM_LOSS_RE = re.compile(rf"avr_loss=(?P<loss>{_FLOAT_RE})", re.IGNORECASE)
_TQDM_LR_RE = re.compile(
    r"\blr=(?P<lr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
_NAN_LOSS_RE = re.compile(
    r"\b(?:non-finite\s+loss|loss\s+became\s+nan|nan_guard\s+recovery)\b",
    re.IGNORECASE,
)

_SIMPLE_LOG_RE = re.compile(
    r"^(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)\t"
    r"(?P<message>.*?)\t(?P<location>[^\t]+\.py:\d+)$",
    re.DOTALL,
)

# The training loop prints this once from the main process. DataLoader workers
# separately emit the second form below, once per worker, so those internal
# duplicate notices are discarded rather than exposed as fake epoch-end rows.
_EPOCH_START_RE = re.compile(
    r"^epoch\s+(?P<epoch>\d+)\s*/\s*(?P<total>\d+)\s*$",
    re.IGNORECASE,
)
_WORKER_EPOCH_RE = re.compile(
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

# Validation loss reported by the val/eval dataloader sweep — ``loop.py``
# constructs ``logs = {"avr_loss": ...}`` for the val pass too, so the
# tqdm regex catches both. The ``val_loss`` keyed event is emitted by
# the val_loss_recorder line further down. Some builds print
# ``eval_loss`` instead. We support all phrasings because upstream's
# exact wording drifted across releases.
_VAL_LOSS_RE = re.compile(
    rf"\b(?:val(?:idation)?|eval)[\s_/]?loss\s*[=:]\s*(?P<loss>{_FLOAT_RE})"
    r"(?:.*?\bepoch\s*[=:]\s*(?P<epoch>\d+))?"
    r"(?:.*?\bstep\s*[=:]\s*(?P<step>\d+))?",
    re.IGNORECASE,
)


def parse_line(line: str, *, job_id: str | None = None) -> TrainingEvent | None:
    """Return a `TrainingEvent` for `line`, or `None` to drop it.

    `None` drops empty lines and internal worker chatter that would duplicate
    a main-process lifecycle event.
    """
    stripped = line.rstrip("\r\n").strip()
    if not stripped:
        return None

    structured = _SIMPLE_LOG_RE.match(stripped)
    structured_level = structured.group("level").lower() if structured else None
    structured_location = structured.group("location") if structured else None
    if structured is not None:
        stripped = structured.group("message").strip()

    # Order matters: validation-loss lines often co-occur with a step
    # number, but they're a different event type (``validation`` vs
    # ``step``). Match them first so a val pass doesn't get filed as a
    # training step.
    if (m := _VAL_LOSS_RE.search(stripped)) is not None:
        payload: dict[str, object] = {}
        val_loss = float(m.group("loss"))
        if math.isfinite(val_loss):
            payload["val_loss"] = val_loss
        if (epoch := m.group("epoch")) is not None:
            payload["epoch"] = int(epoch)
        if (step := m.group("step")) is not None:
            payload["step"] = int(step)
        return TrainingEvent(
            type=EventType.validation, payload=payload, job_id=job_id
        )

    if (m := _EPOCH_START_RE.match(stripped)) is not None:
        return TrainingEvent(
            type=EventType.epoch_start,
            payload={
                "epoch": int(m.group("epoch")),
                "total_epochs": int(m.group("total")),
            },
            job_id=job_id,
        )
    if _WORKER_EPOCH_RE.search(stripped) is not None:
        return None

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
            loss = float(lm.group("loss"))
            if math.isfinite(loss):
                payload["loss"] = loss
        if (rm := _TQDM_LR_RE.search(stripped)) is not None:
            payload["lr"] = float(rm.group("lr"))
        return TrainingEvent(type=EventType.step, payload=payload, job_id=job_id)

    if _NAN_LOSS_RE.search(stripped):
        return TrainingEvent(
            type=EventType.diagnostic_warning,
            payload={
                "category": "nan_loss",
                "severity": "error",
                "message": "Loss became NaN — training is numerically unstable.",
                "remediation": "Lower learning rate or disable fp16-risky options, then restart from a clean checkpoint.",
                "evidence": stripped,
            },
            job_id=job_id,
        )

    level = structured_level or ("error" if _looks_like_error(stripped) else "info")
    payload: dict[str, object] = {"level": level, "message": stripped}
    if structured_location is not None:
        payload["location"] = structured_location
    return TrainingEvent(
        type=EventType.log,
        payload=payload,
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

# Substrings that contain "error" but are benign informational output.
_ERROR_FALSE_POSITIVES = (
    "brokenpipeerror: [errno 32] broken pipe",
    "mean ar error",
    "forrtl: error (200): program aborting due to control-break event",
)


def _looks_like_error(line: str) -> bool:
    lowered = line.lower()
    if any(h in lowered for h in _CANCEL_HINTS):
        return False
    if any(fp in lowered for fp in _ERROR_FALSE_POSITIVES):
        return False
    return (
        "error" in lowered
        or "out of memory" in lowered
        or "cuda error" in lowered
        or "traceback" in lowered
    )


__all__ = ["parse_line"]
