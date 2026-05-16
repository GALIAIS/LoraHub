"""Parse kohya_ss/sd-scripts stdout lines into structured TrainingEvent objects.

Kohya's stdout mixes tqdm progress bars, plain log lines, multi-line Python
tracebacks, and ad-hoc status messages. We pattern-match the lines we
recognize and emit `log` events for everything else so nothing is silently
dropped.

The parser is stateful (traceback aggregation across lines + cache-progress
throttling) so it lives on `KohyaLineParser`. A module-level `parse_line`
wrapper is kept for callers that don't care about the multi-line signals —
each invocation uses a fresh instance, so traceback / cache features only
fire when callers hold on to a `KohyaLineParser` instance themselves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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

# sd-scripts prints validation loss in two shapes depending on version:
#   "validation loss: 0.5237"             (newer logger formatter)
#   "... val_loss=0.5237 ..."              (tqdm postfix on the eval bar)
# Both forms surface the same number, so we accept either via alternation
# and emit a single `validation` event. An optional epoch hint (printed as
# `epoch N` somewhere on the line) is captured opportunistically.
_VAL_LOSS_RE = re.compile(
    r"(?:validation\s*loss[:\s=]+|val_loss\s*=\s*)"
    r"(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
_VAL_EPOCH_HINT_RE = re.compile(r"epoch\s+(?P<epoch>\d+)", re.IGNORECASE)
_VAL_STEP_HINT_RE = re.compile(r"step\s+(?P<step>\d+)", re.IGNORECASE)

# CUDA OOM. `torch.cuda.OutOfMemoryError` is the modern PyTorch class name;
# the legacy `RuntimeError: CUDA out of memory` form still shows up on
# pinned-version sd-scripts checkouts so we match both.
_OOM_RE = re.compile(
    r"(?:RuntimeError:\s*CUDA out of memory"
    r"|torch\.cuda\.OutOfMemoryError)",
)

# Traceback boundaries. We accept the standard `Traceback (most recent call
# last):` opener and close on the first non-indented line that looks like
# an exception summary (matches "ExceptionType: message").
_TRACEBACK_OPEN_RE = re.compile(r"^Traceback \(most recent call last\):")
_EXCEPTION_LINE_RE = re.compile(r"^[A-Z]\w*(?:Error|Exception|Warning):")

# Cache-latents / cache-text-encoder progress lines (tqdm). We accept both
# phases and capture done/total so listeners can show a real percentage.
_CACHE_RE = re.compile(
    r"caching\s+(?P<phase>latents|text\s+encoder\s+outputs)\s*:"
    r".+?\|\s*(?P<done>\d+)\s*/\s*(?P<total>\d+)",
    re.IGNORECASE,
)

# Tunables for the throttling/buffering features. Exposed as module-level
# constants so tests can monkeypatch and so the values are easy to spot.
# Cache thresholds: emit when either the absolute count moves by 10 or the
# percentage moves by 10 since the last emission. Small totals (30 items)
# emit start/~33%/~66%/end; large totals (10k latents) hit roughly every
# 10% — enough for a smooth UI without spamming hundreds of updates.
# Tunables for the throttling/buffering features. Exposed as module-level
# constants so tests can monkeypatch and so the values are easy to spot.
# Cache thresholds: emit when BOTH the absolute count moved by 10 AND the
# percentage moved by 10 since the last emission. Going AND lets large
# totals (10k latents) still pace at ~10% milestones while small totals
# (30 items) collapse to ~4 events instead of one-per-tick.
TRACEBACK_MAX_LINES = 50
CACHE_MIN_PERCENT_DELTA = 10.0
CACHE_MIN_DONE_DELTA = 10


@dataclass(frozen=True, slots=True)
class _CacheOutcome:
    """Sentinel wrapper letting the cache helper distinguish "matched but
    throttled" from "didn't match" without overloading None semantics."""

    event: TrainingEvent | None


class KohyaLineParser:
    """Stateful kohya stdout parser.

    Holds the running traceback buffer and per-phase cache-progress
    bookkeeping. One instance per training subprocess.
    """

    def __init__(self) -> None:
        self._traceback_lines: list[str] = []
        self._in_traceback: bool = False
        # phase -> (last emitted `done`, last emitted percentage)
        self._cache_last: dict[str, tuple[int, float]] = {}

    def parse_line(
        self, line: str, *, job_id: str | None = None
    ) -> TrainingEvent | None:
        """Return a `TrainingEvent` for `line`, or `None` to drop it.

        Multi-line traceback aggregation buffers intermediate lines and
        emits a single `error` event when the traceback closes. Cache
        progress throttling drops noise so listeners only see meaningful
        deltas. All other patterns are stateless.
        """
        stripped = line.rstrip("\r\n")
        if not stripped.strip():
            return None

        # Traceback handling has to run first so the body of a traceback
        # doesn't get classified as random log lines.
        if self._in_traceback:
            return self._consume_traceback_line(stripped, job_id=job_id)
        if _TRACEBACK_OPEN_RE.match(stripped):
            self._in_traceback = True
            self._traceback_lines = [stripped]
            return None

        # OOM is also distinct from generic errors; emit before the regex
        # cascade so OOM doesn't accidentally fall through as `log`.
        if _OOM_RE.search(stripped):
            return TrainingEvent(
                type=EventType.oom,
                payload={"message": stripped},
                job_id=job_id,
            )

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
                payload={
                    "epoch": int(m.group("cur")),
                    "total_epochs": int(m.group("total")),
                },
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

        # Cache progress: returns either a (possibly throttled-suppressed)
        # outcome or None when the line isn't a cache progress bar at all.
        # We need to distinguish "matched but throttled" (drop entirely) from
        # "didn't match" (fall through to log) — `_maybe_cache_event` returns
        # a sentinel tuple to make that explicit.
        cache_outcome = self._maybe_cache_event(stripped, job_id=job_id)
        if cache_outcome is not None:
            return cache_outcome.event

        # Validation loss matching runs after the step regex so the train-loss
        # `avr_loss=` postfix on the progress bar wins for the common case.
        if (m := _VAL_LOSS_RE.search(stripped)) is not None:
            payload: dict[str, object] = {"val_loss": float(m.group("val"))}
            if (em := _VAL_EPOCH_HINT_RE.search(stripped)) is not None:
                payload["epoch"] = int(em.group("epoch"))
            if (sm := _VAL_STEP_HINT_RE.search(stripped)) is not None:
                payload["step"] = int(sm.group("step"))
            return TrainingEvent(
                type=EventType.validation, payload=payload, job_id=job_id
            )

        level = "error" if _looks_like_error(stripped) else "info"
        return TrainingEvent(
            type=EventType.log,
            payload={"level": level, "message": stripped},
            job_id=job_id,
        )

    def _consume_traceback_line(
        self, stripped: str, *, job_id: str | None
    ) -> TrainingEvent | None:
        """Buffer this line as part of an active traceback, possibly close.

        Closing happens either on a recognisable exception-summary line or
        when the buffer hits `TRACEBACK_MAX_LINES` (defensive cap so an
        adversarial child can't blow up our memory).
        """
        self._traceback_lines.append(stripped)

        if _EXCEPTION_LINE_RE.match(stripped):
            return self._flush_traceback(summary=stripped, job_id=job_id)

        if len(self._traceback_lines) >= TRACEBACK_MAX_LINES:
            # Force-flush truncated; mark via a sentinel so consumers know.
            return self._flush_traceback(
                summary=self._traceback_lines[-1],
                job_id=job_id,
                truncated=True,
            )
        return None

    def _flush_traceback(
        self,
        *,
        summary: str,
        job_id: str | None,
        truncated: bool = False,
    ) -> TrainingEvent:
        traceback_text = "\n".join(self._traceback_lines)
        self._traceback_lines = []
        self._in_traceback = False
        payload: dict[str, object] = {
            "traceback": traceback_text,
            "summary": summary,
        }
        if truncated:
            payload["truncated"] = True
        return TrainingEvent(
            type=EventType.error,
            payload=payload,
            job_id=job_id,
        )

    def _maybe_cache_event(
        self, stripped: str, *, job_id: str | None
    ) -> _CacheOutcome | None:
        """Match a cache-progress tqdm line.

        Returns:
            * `None` when `stripped` is not a cache progress bar at all (let
              the caller continue the regex cascade).
            * `_CacheOutcome(event=ev)` when we want to emit `ev`.
            * `_CacheOutcome(event=None)` when the line *is* cache progress
              but throttling says "drop entirely" — caller must NOT fall
              through to the log handler in this case.
        """
        m = _CACHE_RE.search(stripped)
        if m is None:
            return None
        raw_phase = m.group("phase").lower()
        phase = "text_encoder" if raw_phase.startswith("text") else "latents"
        done = int(m.group("done"))
        total = int(m.group("total"))
        if total <= 0:
            # Malformed but still a cache line; suppress so it doesn't leak
            # to the log channel either.
            return _CacheOutcome(event=None)
        percent = (done / total) * 100.0

        last = self._cache_last.get(phase)
        is_terminal = done >= total
        if last is None:
            should_emit = True
        elif is_terminal and last[0] < total:
            should_emit = True
        else:
            last_done, last_percent = last
            should_emit = (
                done - last_done >= CACHE_MIN_DONE_DELTA
                and percent - last_percent >= CACHE_MIN_PERCENT_DELTA
            )

        if not should_emit:
            return _CacheOutcome(event=None)

        self._cache_last[phase] = (done, percent)
        return _CacheOutcome(
            event=TrainingEvent(
                type=EventType.cache_progress,
                payload={"phase": phase, "done": done, "total": total},
                job_id=job_id,
            )
        )


def parse_line(line: str, *, job_id: str | None = None) -> TrainingEvent | None:
    """Stateless single-line parsing, kept for backward compatibility.

    A fresh `KohyaLineParser` is used per call, so multi-line features
    (traceback aggregation, cache throttling) only fire when callers hold
    on to a `KohyaLineParser` instance themselves. The module-level entry
    is mostly useful for ad-hoc inspection and the legacy single-line
    tests.
    """
    return KohyaLineParser().parse_line(line, job_id=job_id)


def _looks_like_error(line: str) -> bool:
    lowered = line.lower()
    return (
        "error" in lowered
        or "traceback" in lowered
        or "out of memory" in lowered
        or "cuda error" in lowered
    )
