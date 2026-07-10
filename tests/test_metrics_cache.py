from __future__ import annotations

import json
from pathlib import Path

from lorahub.api.jobs_helpers import metrics


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
