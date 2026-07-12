"""Structured training events.

A backend reports progress to the rest of the application through a stream
of `TrainingEvent` objects. Sinks (CLI, API/WebSocket, JSONL persistence)
subscribe to the bus and react. The schema here is the only contract between
training backends and the upper layers, so it must stay backend-agnostic.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Any, Self


class EventType(StrEnum):
    step = "step"
    epoch_start = "epoch_start"
    epoch_end = "epoch_end"
    sample_ready = "sample_ready"
    checkpoint_saved = "checkpoint_saved"
    # Emitted whenever the backend reports a validation-set loss (sd-scripts'
    # `--validation_split_percentage` / `--validate_every_n_epochs`). Payload
    # keys: `val_loss` (float, required), `epoch` and `step` (optional ints).
    # Older consumers without explicit handling fall through harmlessly.
    validation = "validation"
    # Caching latents / text-encoder outputs progress. The parser throttles
    # tqdm spam so listeners only see meaningful jumps. Payload keys:
    # `phase` ("latents" or "text_encoder"), `done` (int), `total` (int).
    cache_progress = "cache_progress"
    # CUDA out-of-memory. Distinct from generic `error` so the UI can render
    # a tailored toast with VRAM-trimming suggestions. Payload key: `message`.
    oom = "oom"
    # Periodic resource sample emitted while a job is running. Payload:
    # `gpu_index` (int), `util_percent` (float|null), `vram_used_mib`
    # (int|null), `vram_total_mib` (int|null), `temperature_c` (float|null).
    # Sampled by the API host every few seconds so we can replay an
    # accurate hardware-usage trend after the run finishes.
    gpu_sample = "gpu_sample"
    log = "log"
    error = "error"
    done = "done"
    # Emitted by the preview worker setup path when no inference backend
    # in the registry can serve ``cfg.base_model.arch`` (or all of them
    # bow out — diffusers wheel missing, anima sd-scripts paths absent,
    # ...). Lets the UI surface "previews disabled because no backend
    # supports {arch}" instead of silently falling back to the placeholder
    # PNG. Payload keys: ``arch`` (str), ``available_backends`` (list[str]),
    # ``reason`` (str, human-readable).
    preview_unavailable = "preview_unavailable"
    # Diagnostic match against a known failure-mode pattern, emitted in
    # real time by the runner's stderr/stdout pump. Mirrors the shape of
    # ``DiagnosisFinding`` so the UI can render the same toast / panel
    # row whether the hit happens mid-run (this event) or post-mortem
    # (``diagnose_failure`` reply). Payload keys:
    #   - ``category`` (str): stable rule slug
    #   - ``severity`` (info / warn / error)
    #   - ``message`` (str): user-facing summary
    #   - ``remediation`` (str): actionable next step
    #   - ``evidence`` (str): line that triggered the match
    #   - ``source`` (stdout / stderr): which pipe carried the line
    diagnostic_warning = "diagnostic_warning"
    # Singular value decomposition summary of a LoRA adapter checkpoint.
    # Computed on the API host (not the trainer) right after a
    # ``checkpoint_saved`` event lands, so the live training loop is
    # never blocked by the SVD. Payload keys:
    #   - ``checkpoint`` (str): absolute path to the .safetensors file
    #   - ``step`` (int): training step the checkpoint was saved at
    #   - ``layers`` (int): number of LoRA matrices analysed
    #   - ``effective_rank`` (float): geometric mean of per-layer
    #     effective ranks (Σσ)² / Σσ²
    #   - ``top1_energy`` (float): fraction of energy in the largest
    #     singular value, averaged across layers (0..1)
    #   - ``fro_norm`` (float): mean Frobenius norm of ΔW = α·B·A
    #   - ``per_layer`` (list[dict] | None): up to 16 representative
    #     layers' raw stats — surfaces in the AI analysis prompt.
    lora_spectrum = "lora_spectrum"
    # Catastrophic-forgetting probe result. Optional: only emitted when
    # ``cfg.sampling.forgetting_probe.enable=True`` and the backend
    # synthesised the comparison images. Payload keys:
    #   - ``checkpoint`` (str), ``step`` (int)
    #   - ``preserved`` (float): mean similarity vs the pristine base
    #     model on a held-out neutral prompt set, in [0..1]
    #   - ``samples`` (int): number of probe prompts used
    #   - ``image_path`` (str | None): grid image laying out probe
    #     prompts side-by-side for the UI lightbox
    forgetting_probe = "forgetting_probe"


@dataclass(frozen=True, slots=True)
class TrainingEvent:
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            type=EventType(data["type"]),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            job_id=data.get("job_id"),
        )


EventListener = Callable[[TrainingEvent], None]

_MAX_EVENT_STRING = 16_000


def normalize_event(event: TrainingEvent, *, job_id: str | None = None) -> TrainingEvent:
    """Return a JSON-safe event with bounded payload strings."""
    payload = _normalize_payload(event.payload)
    normalized_job_id = event.job_id or job_id
    if payload is event.payload and normalized_job_id == event.job_id:
        return event
    return TrainingEvent(
        type=event.type,
        payload=payload,
        timestamp=event.timestamp,
        job_id=normalized_job_id,
    )


def _normalize_payload(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        if len(value) <= _MAX_EVENT_STRING:
            return value
        return value[:_MAX_EVENT_STRING] + "\n...[truncated]"
    if isinstance(value, dict):
        return {str(k): _normalize_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_payload(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize_payload(v) for v in value]
    return value


class EventBus:
    """Thread-safe in-memory pub/sub for `TrainingEvent`.

    Listeners are invoked synchronously in registration order. Exceptions
    raised by listeners are swallowed and surfaced as an `error` event so a
    misbehaving sink cannot kill the training process.
    """

    def __init__(self) -> None:
        self._listeners: list[EventListener] = []
        self._lock = RLock()

    def subscribe(self, listener: EventListener) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(listener)

        def _unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return _unsubscribe

    def publish(self, event: TrainingEvent) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception as exc:  # noqa: BLE001
                err = TrainingEvent(
                    type=EventType.error,
                    payload={"source": "event_bus", "error": repr(exc)},
                    job_id=event.job_id,
                )
                with self._lock:
                    fallbacks = [other for other in self._listeners if other is not listener]
                for fallback in fallbacks:
                    with contextlib.suppress(Exception):
                        fallback(err)


class JsonlEventSink:
    """Append-only JSONL persistence for events, one event per line.

    Use as a context manager so the file handle is closed deterministically:

        with JsonlEventSink(path) as sink:
            bus.subscribe(sink)
            ...
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh: Any = None
        self._lock = RLock()

    def __enter__(self) -> Self:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8", buffering=1)
        return self

    def __exit__(self, *_exc: object) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __call__(self, event: TrainingEvent) -> None:
        with self._lock:
            if self._fh is None:
                raise RuntimeError("JsonlEventSink used outside its context")
            self._fh.write(event.to_json() + "\n")

    @staticmethod
    def replay(path: Path) -> Iterator[TrainingEvent]:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield TrainingEvent.from_dict(json.loads(line))
