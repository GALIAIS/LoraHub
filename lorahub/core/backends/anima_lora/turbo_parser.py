"""Parse `scripts/distill_turbo.py` stdout into TrainingEvents.

distill_turbo.py drives a bespoke DMD2 distillation loop. The tqdm
desc is ``"turbo"`` (not ``"steps"``) and the postfix uses RMS metric
shorthands instead of ``avr_loss``:

    turbo:  17%|##2 | 51/300 [00:30<02:30, 1.67it/s, g=1.23e-03, dca=4.56e-04, \
        ddm=7.89e-05, xp=0.421, vs=0.317, fake=2.10e-05]

Where:
* ``g`` = student grad-norm RMS (most useful single signal — track this
  as ``loss`` so the LoraHub UI gets a curve)
* ``dca`` = decoupled-CA branch loss
* ``ddm`` = decoupled-DM branch loss
* ``xp`` = predicted-x RMS
* ``vs`` = student velocity RMS
* ``fake`` = fake LoRA loss

There's no ``avr_loss`` here — ``g`` (grad-norm) is the canonical
"is training healthy" signal and the closest thing to what the rest
of LoraHub treats as ``loss``.

Save events: distill_turbo.py doesn't log ``saving checkpoint:`` —
``turbo.save_student`` writes silently. The LoraHub workspace mtime
watcher already handles checkpoint detection downstream, so this
parser doesn't try to invent a save line.

Other signals: errors, warnings, KeyboardInterrupt — same triage as
the regular `parser.py`.
"""

from __future__ import annotations

import re

from lorahub.core.events import EventType, TrainingEvent

# tqdm bar with desc=``turbo``. Step + total mandatory; RMS-shorthand
# postfix kwargs extracted by separate regex so a stray formatting
# tweak only loses one signal, not the whole step event.
_TURBO_STEPS_RE = re.compile(
    r"turbo:\s*\d+%\|[^|]*\|\s*(?P<step>\d+)/(?P<total>\d+)",
)
# Each postfix kwarg is its own regex — distill_turbo emits them in a
# fixed order today, but we don't depend on that ordering.
_TURBO_GRAD_RE = re.compile(
    r"\bg=(?P<g>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
_TURBO_DCA_RE = re.compile(
    r"\bdca=(?P<dca>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
_TURBO_DDM_RE = re.compile(
    r"\bddm=(?P<ddm>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
_TURBO_XPRED_RE = re.compile(
    r"\bxp=(?P<xp>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
_TURBO_VSTUDENT_RE = re.compile(
    r"\bvs=(?P<vs>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
_TURBO_FAKE_RE = re.compile(
    r"\bfake=(?P<fake>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def parse_line(line: str, *, job_id: str | None = None) -> TrainingEvent | None:
    """Return a `TrainingEvent` for `line`, or `None` to drop empties."""
    stripped = line.rstrip("\r\n").strip()
    if not stripped:
        return None

    if (m := _TURBO_STEPS_RE.search(stripped)) is not None:
        payload: dict[str, object] = {
            "step": int(m.group("step")),
            "total_steps": int(m.group("total")),
        }
        # Treat student grad-norm RMS as the primary "loss" signal so
        # the LoraHub progress chart stays useful for turbo runs. The
        # other RMS values land in the payload too — analytics tab can
        # surface them once UI catches up.
        if (gm := _TURBO_GRAD_RE.search(stripped)) is not None:
            payload["loss"] = float(gm.group("g"))
            payload["grad_rms"] = float(gm.group("g"))
        if (dm := _TURBO_DCA_RE.search(stripped)) is not None:
            payload["dca_rms"] = float(dm.group("dca"))
        if (dm := _TURBO_DDM_RE.search(stripped)) is not None:
            payload["ddm_rms"] = float(dm.group("ddm"))
        if (xm := _TURBO_XPRED_RE.search(stripped)) is not None:
            payload["xpred_rms"] = float(xm.group("xp"))
        if (vm := _TURBO_VSTUDENT_RE.search(stripped)) is not None:
            payload["vstudent_rms"] = float(vm.group("vs"))
        if (fm := _TURBO_FAKE_RE.search(stripped)) is not None:
            payload["fake_loss"] = float(fm.group("fake"))
        return TrainingEvent(type=EventType.step, payload=payload, job_id=job_id)

    level = "error" if _looks_like_error(stripped) else "info"
    return TrainingEvent(
        type=EventType.log,
        payload={"level": level, "message": stripped},
        job_id=job_id,
    )


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
