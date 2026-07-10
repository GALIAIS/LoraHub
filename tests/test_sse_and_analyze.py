"""Coverage for the three SSE endpoints + the /analyze route.

These were the main "0 coverage" gaps flagged by the v0.3 audit (B3).
The SSE generator is async, but we lean on the FastAPI TestClient
collecting the streamed body in one shot — when the job is in a
terminal state the generator returns immediately after replay, so the
whole exchange is bounded.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lorahub.api import app as app_mod
from lorahub.api import state
from lorahub.api.ai_store import AIRoute
from lorahub.api.app import (
    _replay_has_current_done,
    _resume_index_from_header,
    _sse_format,
)
from lorahub.core.events import EventType, JsonlEventSink, TrainingEvent


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_sse_format_basic_event_has_id_and_data() -> None:
    out = _sse_format(event_id="3", data='{"step":3}')
    # Spec: id line, data line, blank terminator.
    assert "id: 3" in out
    assert 'data: {"step":3}' in out
    assert out.endswith("\n\n") or out.endswith("\n")


def test_sse_format_multiline_data_emits_one_data_line_per_line() -> None:
    out = _sse_format(event_id="0", data="line1\nline2\nline3")
    lines = [line for line in out.splitlines() if line.startswith("data:")]
    assert lines == ["data: line1", "data: line2", "data: line3"]


def test_sse_format_comment_only_uses_colon_prefix() -> None:
    out = _sse_format(comment="ping")
    # Comments per the SSE spec are `: <text>`.
    assert ": ping" in out


def test_sse_format_retry_emits_retry_field() -> None:
    out = _sse_format(retry_ms=2000, comment="hint")
    assert "retry: 2000" in out


# --------------------------------------------------------------------------- #
# _resume_index_from_header
# --------------------------------------------------------------------------- #


class _FakeRequest:
    """Tiny stand-in for a request carrying SSE resume metadata."""

    def __init__(
        self,
        last_event_id: str | None = None,
        query_event_id: str | None = None,
    ) -> None:
        self.headers = (
            {"last-event-id": last_event_id} if last_event_id is not None else {}
        )
        self.query_params = (
            {"lastEventId": query_event_id} if query_event_id is not None else {}
        )


def test_resume_index_returns_zero_when_header_missing() -> None:
    assert _resume_index_from_header(_FakeRequest()) == 0  # type: ignore[arg-type]


def test_resume_index_returns_n_plus_one() -> None:
    assert (
        _resume_index_from_header(_FakeRequest("3"))  # type: ignore[arg-type]
        == 4
    )


def test_resume_index_accepts_manual_reconnect_query() -> None:
    assert (
        _resume_index_from_header(  # type: ignore[arg-type]
            _FakeRequest(query_event_id="7")
        )
        == 8
    )


def test_resume_index_clamps_negative_to_zero() -> None:
    # Browsers won't send negative ids, but a corrupt cookie shouldn't
    # be able to rewind the stream past the start.
    assert (
        _resume_index_from_header(_FakeRequest("-7"))  # type: ignore[arg-type]
        == 0
    )


def test_resume_index_falls_back_on_garbage() -> None:
    assert (
        _resume_index_from_header(_FakeRequest("not-a-number"))  # type: ignore[arg-type]
        == 0
    )


def test_resume_history_done_does_not_close_active_run(tmp_path: Path) -> None:
    job = state.registry.create(workspace=tmp_path / "resume", config_snapshot={})
    job.state = state.JobState.running
    job.started_at = datetime.fromtimestamp(200, tz=UTC)
    old_done = TrainingEvent(
        type=EventType.done,
        payload={"returncode": 0},
        timestamp=100,
    )

    assert not _replay_has_current_done(job, [old_done])


def test_current_run_done_closes_active_stream(tmp_path: Path) -> None:
    job = state.registry.create(workspace=tmp_path / "current", config_snapshot={})
    job.state = state.JobState.running
    job.started_at = datetime.fromtimestamp(200, tz=UTC)
    current_done = TrainingEvent(
        type=EventType.done,
        payload={"returncode": 0},
        timestamp=201,
    )

    assert _replay_has_current_done(job, [current_done])


# --------------------------------------------------------------------------- #
# /api/jobs/{id}/sse end-to-end (terminal job — generator returns immediately
# after replay so the test client can collect the whole body)
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[state.JobRegistry]:
    fresh = state.JobRegistry()
    monkeypatch.setattr(state, "registry", fresh)
    yield fresh


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    return TestClient(app_mod.app)


def _seed_terminal_job(workspace: Path) -> str:
    """Create a job whose events.jsonl already contains a `done`
    event so the SSE generator returns straight after replay."""
    workspace.mkdir(parents=True, exist_ok=True)
    log = workspace / "events.jsonl"
    base_ts = 1_700_000_000.0
    lines: list[str] = []
    for step in range(1, 4):
        lines.append(
            TrainingEvent(
                type=EventType.step,
                payload={"step": step, "loss": 0.5 / step, "total_steps": 100},
                timestamp=base_ts + step,
            ).to_json(),
        )
    lines.append(
        TrainingEvent(
            type=EventType.done,
            payload={"returncode": 0, "duration_s": 4.0},
            timestamp=base_ts + 5,
        ).to_json(),
    )
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    job = state.registry.create(workspace=workspace, config_snapshot={})
    job.state = state.JobState.succeeded
    state.registry.update(job)
    return job.id


def _parse_sse_body(text: str) -> list[dict[str, str]]:
    """Group the SSE event stream into a list of frames keyed by field.

    Each frame looks like::
        id: 3\\ndata: {"step":3}\\n\\n
    Comments (`: ...`) are dropped.
    """
    frames: list[dict[str, str]] = []
    cur: dict[str, str] = {}
    for raw in text.splitlines():
        if not raw:
            if cur:
                frames.append(cur)
                cur = {}
            continue
        if raw.startswith(":"):
            continue  # SSE comment / keepalive
        key, _, value = raw.partition(":")
        if not key:
            continue
        cur[key.strip()] = value.lstrip()
    if cur:
        frames.append(cur)
    return frames


def test_jobs_sse_replays_full_history_for_terminal_job(
    client: TestClient, tmp_path: Path
) -> None:
    job_id = _seed_terminal_job(tmp_path / "ws")

    with client.stream("GET", f"/api/jobs/{job_id}/sse") as r:
        assert r.status_code == 200
        body = r.read().decode("utf-8")

    frames = _parse_sse_body(body)
    # Three step events + one done event = four data frames; we don't
    # assert on the leading `retry:` hint frame because it's not a
    # data frame.
    data_frames = [f for f in frames if "data" in f]
    assert len(data_frames) == 4
    # Ids are monotone starting at 0.
    assert [f["id"] for f in data_frames] == ["0", "1", "2", "3"]
    # Final frame is the `done` event.
    assert '"type":"done"' in data_frames[-1]["data"]


def test_jobs_sse_resumes_from_last_event_id(
    client: TestClient, tmp_path: Path
) -> None:
    """Last-Event-ID: 1 means "I've seen 0 and 1, send me 2 onward"."""
    job_id = _seed_terminal_job(tmp_path / "ws")

    with client.stream(
        "GET",
        f"/api/jobs/{job_id}/sse",
        headers={"Last-Event-ID": "1"},
    ) as r:
        body = r.read().decode("utf-8")
    data_frames = [f for f in _parse_sse_body(body) if "data" in f]
    assert [f["id"] for f in data_frames] == ["2", "3"]


def test_jobs_sse_returns_404_for_unknown_job(client: TestClient) -> None:
    r = client.get("/api/jobs/nonexistent/sse")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# /api/system/sse — intentionally not exercised here.
#
# The endpoint's generator does `await asyncio.sleep(1.0)` between
# snapshots, and FastAPI's TestClient drains the StreamingResponse
# fully on context exit. That means even a "smoke" test that only
# wants the first frame ends up paying the 1s sleep on teardown,
# which makes a tiny test 1+ seconds long. Not worth the noise — the
# route is structurally identical to /jobs/{id}/sse and
# /backend/bootstrap/sse, so the format / resume-header coverage
# above guards the same shape.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# /api/jobs/{id}/analyze — only the 404 path. The success path needs
# a real AI provider configured (the route returns 503 without one,
# which is the right behaviour but not a useful assertion). The
# tighter integration test belongs in the AI router suite, not here.
# --------------------------------------------------------------------------- #


def test_jobs_analyze_returns_404_for_unknown_job(client: TestClient) -> None:
    r = client.post("/api/jobs/missing/analyze")
    assert r.status_code == 404


def test_jobs_analyze_uses_training_diagnose_route_and_real_config_fields(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "events.jsonl").write_text(
        "\n".join(
            [
                TrainingEvent(
                    type=EventType.step,
                    payload={"step": 1, "loss": 0.8, "lr": 7e-5, "total_steps": 20},
                    timestamp=1.0,
                ).to_json(),
                TrainingEvent(
                    type=EventType.step,
                    payload={"step": 2, "loss": 0.6, "lr": 7e-5, "total_steps": 20},
                    timestamp=2.0,
                ).to_json(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    job = state.registry.create(
        workspace=workspace,
        config_snapshot={
            "baseModel": {"arch": "anima"},
            "dataset": {
                "numRepeats": 3,
                "caption": {"dropRate": 0.18, "keepTokens": 1},
            },
            "schedule": {"batchSize": 2, "gradAccum": 4},
            "optimizer": {"schedule": "cosine"},
            "backend": {
                "type": "anima_lora",
                "animaLora": {
                    "outputName": "style_anima_32gb",
                    "networkDim": 16,
                    "networkAlpha": 16,
                    "learningRate": 7e-5,
                    "lrScheduler": "cosine",
                    "maxTrainEpochs": 10,
                    "captionDropoutRate": 0,
                    "keepTokens": 0,
                    "validationSplitNum": 0,
                    "lora": {"algorithm": "loha"},
                },
            },
        },
    )
    job.state = state.JobState.succeeded
    state.registry.update(job)

    seen_routes: list[str] = []

    class FakeStore:
        def get_route(self, task_id: str):  # type: ignore[no-untyped-def]
            seen_routes.append(task_id)
            if task_id == "training.diagnose":
                return AIRoute(task_id=task_id, provider_id="p", model_id="m")
            return None

    class Result:
        content = "结论：训练正常。"
        provider_name = "Provider"
        model_id = "model"

    captured: dict[str, object] = {}

    def fake_invoke(store, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr(app_mod, "_ai_store", FakeStore())
    monkeypatch.setattr("lorahub.core.ai.client.invoke", fake_invoke)

    r = client.post(f"/api/jobs/{job.id}/analyze")
    assert r.status_code == 200, r.text
    assert seen_routes == ["training.diagnose"]
    analysis = r.json()["analysis"]
    cfg = analysis["summary_payload"]["config"]
    assert cfg["backend"] == "anima_lora"
    assert cfg["rank"] == 16
    assert cfg["alpha"] == 16
    assert cfg["algorithm"] == "loha"
    assert cfg["lr"] == 7e-5
    assert cfg["lr_scheduler"] == "cosine"
    assert cfg["epochs"] == 10
    assert cfg["num_repeats"] == 3
    assert cfg["caption_dropout_rate"] == 0
    assert cfg["keep_tokens"] == 0
    assert cfg["validation_split_num"] == 0
    assert captured["provider_id"] == "p"
    assert captured["model_id"] == "m"


def test_jobs_analyze_falls_back_to_global_default_route(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "events.jsonl").write_text(
        TrainingEvent(
            type=EventType.step,
            payload={"step": 1, "loss": 0.8, "total_steps": 1},
            timestamp=1.0,
        ).to_json()
        + "\n",
        encoding="utf-8",
    )
    job = state.registry.create(
        workspace=workspace,
        config_snapshot={
            "base_model": {"arch": "sdxl"},
            "network": {"rank": 8, "alpha": 4},
            "optimizer": {"lr": 1e-4, "schedule": "constant"},
            "schedule": {"epochs": 1},
        },
    )
    job.state = state.JobState.succeeded
    state.registry.update(job)

    seen_routes: list[str] = []

    class FakeStore:
        def get_route(self, task_id: str):  # type: ignore[no-untyped-def]
            seen_routes.append(task_id)
            if task_id == "global.default":
                return AIRoute(task_id=task_id, provider_id="fallback", model_id="m")
            return None

    class Result:
        content = "结论：数据不足。"
        provider_name = "Provider"
        model_id = "model"

    captured: dict[str, object] = {}

    def fake_invoke(store, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr(app_mod, "_ai_store", FakeStore())
    monkeypatch.setattr("lorahub.core.ai.client.invoke", fake_invoke)

    r = client.post(f"/api/jobs/{job.id}/analyze")
    assert r.status_code == 200, r.text
    assert seen_routes == ["training.diagnose", "global.default"]
    assert captured["provider_id"] == "fallback"
    cfg = r.json()["analysis"]["summary_payload"]["config"]
    assert cfg["arch"] == "sdxl"
    assert cfg["rank"] == 8
    assert cfg["lr"] == 1e-4


# --------------------------------------------------------------------------- #
# gpu_sample event type — round-trip through events.jsonl + /metrics
# --------------------------------------------------------------------------- #


def test_gpu_sample_events_surface_in_metrics_payload(
    client: TestClient, tmp_path: Path
) -> None:
    """The metrics endpoint extracts gpu_samples; the event type was
    added in the v0.3 cycle but never had a test."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log = workspace / "events.jsonl"

    base_ts = 1_700_000_500.0
    lines = [
        TrainingEvent(
            type=EventType.step,
            payload={"step": 1, "loss": 0.4, "total_steps": 10},
            timestamp=base_ts,
        ).to_json(),
        TrainingEvent(
            type=EventType.gpu_sample,
            payload={
                "gpu_index": 0,
                "util_percent": 92.5,
                "vram_used_mib": 14_322,
                "vram_total_mib": 24_576,
                "temperature_c": 67.0,
            },
            timestamp=base_ts + 1.0,
        ).to_json(),
        TrainingEvent(
            type=EventType.gpu_sample,
            payload={
                "gpu_index": 0,
                "util_percent": 95.0,
                "vram_used_mib": 14_400,
                "vram_total_mib": 24_576,
                "temperature_c": 69.5,
            },
            timestamp=base_ts + 2.0,
        ).to_json(),
    ]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    job = state.registry.create(workspace=workspace, config_snapshot={})
    job.state = state.JobState.succeeded
    state.registry.update(job)

    r = client.get(f"/api/jobs/{job.id}/metrics")
    assert r.status_code == 200
    body = r.json()
    samples = body.get("gpu_samples")
    assert samples is not None
    assert len(samples) == 2
    # Field names follow the wire shape established in the metric grid.
    assert samples[0]["util_percent"] == 92.5
    assert samples[0]["vram_used_mib"] == 14_322
    assert samples[0]["vram_total_mib"] == 24_576
    assert samples[1]["temperature_c"] == 69.5


# Quiet the unused-import warning when pytest collects this file standalone.
_ = JsonlEventSink
