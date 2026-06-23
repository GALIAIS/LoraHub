from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient

from lorahub.api import state


@pytest.fixture(autouse=True)
def fresh_registry() -> Iterator[state.JobRegistry]:
    original = state.registry
    fresh = state.JobRegistry()
    state.registry = fresh
    try:
        yield fresh
    finally:
        state.registry = original


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from lorahub.api import app as app_mod
    from lorahub.api.task_sessions import TaskSessionStore

    monkeypatch.setattr(app_mod, "_task_session_store", TaskSessionStore(tmp_path / "tasks.sqlite3"))
    monkeypatch.chdir(tmp_path)
    return TestClient(app_mod.app)


def _anima_snapshot(tmp_path: Path) -> dict:
    ckpt = tmp_path / "base.safetensors"
    ae = tmp_path / "vae.safetensors"
    qwen3 = tmp_path / "qwen3"
    data = tmp_path / "data"
    for path in (ckpt, ae):
        path.write_bytes(b"x")
    qwen3.mkdir()
    data.mkdir()
    (data / "0.png").write_bytes(b"x")
    return {
        "baseModel": {
            "arch": "anima",
            "checkpoint": str(ckpt),
            "archPaths": {"ae": str(ae), "qwen3": str(qwen3)},
        },
        "dataset": {"source": str(data)},
        "backend": {"type": "anima_lora"},
        "output": {"name": "style"},
    }


def test_lora_test_models_lists_checkpoint_artifacts(
    client: TestClient,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    output = workspace / "output"
    output.mkdir(parents=True)
    (output / "style.safetensors").write_bytes(b"lora")
    job = state.registry.create(workspace=workspace, config_snapshot=_anima_snapshot(tmp_path))
    job.state = state.JobState.succeeded
    state.registry.update(job)

    r = client.get("/api/lora-test/models")
    assert r.status_code == 200
    jobs = r.json()["jobs"]
    assert jobs[0]["job_id"] == job.id
    assert jobs[0]["backend"] == "anima_lora"
    assert jobs[0]["output_name"] == "style"
    assert jobs[0]["checkpoints"][0]["path"] == "output/style.safetensors"


def test_lora_test_generate_rejects_workspace_escape(
    client: TestClient,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    output = workspace / "output"
    output.mkdir(parents=True)
    (output / "style.safetensors").write_bytes(b"lora")
    job = state.registry.create(workspace=workspace, config_snapshot=_anima_snapshot(tmp_path))

    r = client.post(
        "/api/lora-test/generate",
        json={
            "job_id": job.id,
            "checkpoint_path": "../style.safetensors",
            "prompt": "1girl",
        },
    )
    assert r.status_code == 400


def test_lora_test_generate_session_completes_with_fake_inference(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lorahub.api.routers import lora_test

    workspace = tmp_path / "ws"
    output = workspace / "output"
    output.mkdir(parents=True)
    (output / "style.safetensors").write_bytes(b"lora")
    job = state.registry.create(workspace=workspace, config_snapshot=_anima_snapshot(tmp_path))
    job.state = state.JobState.succeeded
    state.registry.update(job)

    def fake_inference(resolved, req, out_path, seed, cancel_evt):  # type: ignore[no-untyped-def]
        out_path.write_bytes(b"png")

    monkeypatch.setattr(lora_test, "_run_anima_inference", fake_inference)
    r = client.post(
        "/api/lora-test/generate",
        json={
            "job_id": job.id,
            "checkpoint_path": "output/style.safetensors",
            "prompt": "1girl",
            "batch_count": 2,
            "seed": 42,
        },
    )
    assert r.status_code == 200, r.text
    session_id = r.json()["session_id"]

    for _ in range(20):
        status = client.get(f"/api/lora-test/sessions/{session_id}").json()
        if status["status"] == "succeeded":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("session did not complete")
    assert len(status["result"]["images"]) == 2
    assert status["result"]["images"][0]["seed"] == 42
    assert status["result"]["images"][1]["seed"] == 43
