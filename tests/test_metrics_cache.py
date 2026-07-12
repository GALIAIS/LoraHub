from __future__ import annotations

import json
from pathlib import Path

from lorahub.api.jobs_helpers import metrics
from lorahub.core.events import EventType, TrainingEvent


def _step(step: int) -> str:
    return json.dumps(
        {
            "type": "step",
            "timestamp": float(step),
            "payload": {"step": step, "loss": 1.0 / (step + 1)},
        }
    )


def test_metrics_cache_replaces_older_snapshot_for_same_log(tmp_path: Path) -> None:
    metrics._METRICS_CACHE.clear()
    log = tmp_path / "events.jsonl"
    log.write_text(_step(1) + "\n", encoding="utf-8")

    metrics._read_metrics(tmp_path)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(_step(2) + "\n")
    result = metrics._read_metrics(tmp_path)

    keys = [key for key in metrics._METRICS_CACHE if key[0] == str(log.resolve())]
    assert len(keys) == 1
    assert result["last_step"] == 2


def test_metrics_downsample_starts_at_response_cap(
    tmp_path: Path, monkeypatch,
) -> None:
    metrics._METRICS_CACHE.clear()
    monkeypatch.setattr(metrics, "_METRICS_MAX_POINTS", 20)
    monkeypatch.setattr(metrics, "_METRICS_DOWNSAMPLE_THRESHOLD", 20)
    log = tmp_path / "events.jsonl"
    log.write_text("\n".join(_step(i) for i in range(100)) + "\n", encoding="utf-8")

    result = metrics._read_metrics(tmp_path)

    assert len(result["loss"]) <= 21
    assert result["loss"][0]["step"] == 0
    assert result["loss"][-1]["step"] == 99


def test_metrics_preserves_backend_timing_epoch_and_diagnostics(tmp_path: Path) -> None:
    metrics._METRICS_CACHE.clear()
    events = [
        TrainingEvent(
            type=EventType.epoch_start,
            payload={"epoch": 3, "total_epochs": 8},
            timestamp=10.0,
        ),
        TrainingEvent(
            type=EventType.cache_progress,
            payload={
                "phase": "latents",
                "done": 70,
                "total": 280,
                "percent": 25,
                "rate": "3.31it/s",
                "eta": "01:03",
            },
            timestamp=11.0,
        ),
        TrainingEvent(
            type=EventType.step,
            payload={
                "step": 27,
                "total_steps": 100,
                "loss": 0.1964,
                "rate": "6.83s/it",
                "eta": "08:18",
                "snr": 2.5,
                "grad_norm": 0.75,
            },
            timestamp=12.0,
        ),
        TrainingEvent(
            type=EventType.diagnostic_warning,
            payload={
                "category": "sample_failure",
                "severity": "error",
                "message": "sample failed",
            },
            timestamp=13.0,
        ),
    ]
    (tmp_path / "events.jsonl").write_text(
        "\n".join(event.to_json() for event in events) + "\n",
        encoding="utf-8",
    )

    result = metrics._read_metrics(tmp_path)

    point = result["loss"][0]
    assert point["epoch"] == 3
    assert point["iter_time_s"] == 6.83
    assert point["samples_per_sec"] == 1 / 6.83
    assert point["eta_s"] == 8 * 60 + 18
    assert point["snr"] == 2.5
    assert point["grad_norm"] == 0.75
    assert result["cache_progress"][0]["eta_s"] == 63
    assert result["diagnostics"][0]["category"] == "sample_failure"
