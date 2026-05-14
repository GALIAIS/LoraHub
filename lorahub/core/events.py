"""Structured training events.

A backend reports progress to the rest of the application through a stream
of `TrainingEvent` objects. Sinks (CLI, API/WebSocket, JSONL persistence)
subscribe to the bus and react. The schema here is the only contract between
training backends and the upper layers, so it must stay backend-agnostic.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Any, Self


class EventType(StrEnum):
    step = "step"
    epoch_end = "epoch_end"
    sample_ready = "sample_ready"
    checkpoint_saved = "checkpoint_saved"
    log = "log"
    error = "error"
    done = "done"


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
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            type=EventType(data["type"]),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            job_id=data.get("job_id"),
        )


EventListener = Callable[[TrainingEvent], None]


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
