"""Tests for `lorahub.core.events`."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from lorahub.core.events import (
    EventBus,
    EventType,
    JsonlEventSink,
    TrainingEvent,
    normalize_event,
)


def test_event_round_trips_through_dict_and_json() -> None:
    ev = TrainingEvent(
        type=EventType.step,
        payload={"step": 42, "loss": 0.123},
        timestamp=1700000000.0,
        job_id="job-x",
    )
    restored = TrainingEvent.from_dict(json.loads(ev.to_json()))
    assert restored == ev


def test_event_json_rejects_non_finite_numbers() -> None:
    ev = TrainingEvent(type=EventType.step, payload={"loss": math.nan})
    with pytest.raises(ValueError):
        ev.to_json()


def test_normalize_event_makes_payload_json_safe_and_bounded() -> None:
    ev = normalize_event(
        TrainingEvent(
            type=EventType.log,
            payload={"loss": math.nan, "message": "x" * 20_000},
        ),
        job_id="job-1",
    )

    assert ev.job_id == "job-1"
    assert ev.payload["loss"] is None
    assert len(ev.payload["message"]) < 17_000
    assert ev.payload["message"].endswith("...[truncated]")
    json.loads(ev.to_json())


def test_bus_delivers_to_all_listeners_in_order() -> None:
    bus = EventBus()
    received: list[tuple[str, EventType]] = []
    bus.subscribe(lambda e: received.append(("a", e.type)))
    bus.subscribe(lambda e: received.append(("b", e.type)))

    bus.publish(TrainingEvent(type=EventType.log, payload={"msg": "hi"}))

    assert received == [("a", EventType.log), ("b", EventType.log)]


def test_bus_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    received: list[EventType] = []
    unsub = bus.subscribe(lambda e: received.append(e.type))

    bus.publish(TrainingEvent(type=EventType.log))
    unsub()
    bus.publish(TrainingEvent(type=EventType.done))

    assert received == [EventType.log]


def test_bus_isolates_listener_failures() -> None:
    bus = EventBus()
    seen_errors: list[TrainingEvent] = []

    def bad(_e: TrainingEvent) -> None:
        raise RuntimeError("boom")

    def good(e: TrainingEvent) -> None:
        if e.type is EventType.error:
            seen_errors.append(e)

    bus.subscribe(bad)
    bus.subscribe(good)

    bus.publish(TrainingEvent(type=EventType.log, job_id="j1"))

    assert len(seen_errors) == 1
    assert seen_errors[0].payload["source"] == "event_bus"
    assert seen_errors[0].job_id == "j1"


def test_jsonl_sink_persists_and_replays(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    events = [
        TrainingEvent(type=EventType.step, payload={"step": i}, timestamp=float(i))
        for i in range(3)
    ]

    with JsonlEventSink(log) as sink:
        for e in events:
            sink(e)

    replayed = list(JsonlEventSink.replay(log))
    assert replayed == events


def test_jsonl_sink_rejects_use_outside_context(tmp_path: Path) -> None:
    sink = JsonlEventSink(tmp_path / "x.jsonl")
    with pytest.raises(RuntimeError):
        sink(TrainingEvent(type=EventType.log))
