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


def test_lora_test_generate_rejects_non_32_multiple_size(
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
            "checkpoint_path": "output/style.safetensors",
            "prompt": "1girl",
            "width": 912,
            "height": 1632,
        },
    )

    assert r.status_code == 422
    assert "divisible by 32" in r.text


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

    def fake_inference(resolved, req, case, out_path, cancel_evt):  # type: ignore[no-untyped-def]
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


def test_lora_test_dedupes_loras_and_collects_anima_directory_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from lorahub.api.routers import lora_test

    workspace = tmp_path / "ws"
    output = workspace / "output"
    output.mkdir(parents=True)
    (output / "style.safetensors").write_bytes(b"lora")
    job = state.registry.create(workspace=workspace, config_snapshot=_anima_snapshot(tmp_path))
    job.state = state.JobState.succeeded
    state.registry.update(job)
    resolved = lora_test._resolve_model(job.id, "output/style.safetensors")
    req = lora_test.GenerateRequest(
        job_id=job.id,
        checkpoint_path="output/style.safetensors",
        prompt="1girl",
        loras=[
            lora_test.LoraInput(job_id=job.id, checkpoint_path="output/style.safetensors", weight=0.7),
            lora_test.LoraInput(job_id=job.id, checkpoint_path="output/style.safetensors", weight=1.2),
        ],
    )
    loras, weights = lora_test._resolve_loras(req, resolved)
    cases = lora_test._build_cases(req, resolved, loras, weights)

    assert len(cases[0].loras) == 1
    assert cases[0].multipliers == [0.7]

    calls: list[list[str]] = []

    class FakeProcess:
        returncode = 0

        def __init__(self, argv, **_kwargs):  # type: ignore[no-untyped-def]
            calls.append([str(item) for item in argv])
            save_dir = Path(argv[argv.index("--save_path") + 1])
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "sample.png").write_bytes(b"png")

        def poll(self):  # type: ignore[no-untyped-def]
            return 0

    monkeypatch.setattr(
        lora_test.anima_bootstrap,
        "resolve",
        lambda **_kwargs: SimpleNamespace(
            python_executable=tmp_path / "python.exe",
            repo_path=tmp_path,
            script=lambda name: tmp_path / name,
        ),
    )
    monkeypatch.setattr(lora_test.subprocess, "Popen", FakeProcess)

    out_path = tmp_path / "result.png"
    lora_test._run_anima_inference(resolved, req, cases[0], out_path, cancel_evt=SimpleNamespace(is_set=lambda: False))

    assert out_path.read_bytes() == b"png"
    assert not out_path.with_suffix("").exists()
    assert calls[0].count(str(output / "style.safetensors")) == 1
    assert calls[0][calls[0].index("--save_path") + 1] == str(out_path.with_suffix(""))


def test_lora_test_xy_grid_with_fake_inference(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image
    from lorahub.api.routers import lora_test

    workspace = tmp_path / "ws"
    output = workspace / "output"
    output.mkdir(parents=True)
    (output / "style.safetensors").write_bytes(b"lora")
    job = state.registry.create(workspace=workspace, config_snapshot=_anima_snapshot(tmp_path))
    job.state = state.JobState.succeeded
    state.registry.update(job)

    seen: list[tuple[float, int]] = []

    def fake_inference(resolved, req, case, out_path, cancel_evt):  # type: ignore[no-untyped-def]
        seen.append((case.cfg, case.steps))
        Image.new("RGB", (8, 8), "white").save(out_path)

    monkeypatch.setattr(lora_test, "_run_anima_inference", fake_inference)
    r = client.post(
        "/api/lora-test/generate",
        json={
            "job_id": job.id,
            "checkpoint_path": "output/style.safetensors",
            "prompt": "1girl",
            "seed": 7,
            "x_axis": {"field": "cfg", "values": ["3", "5"]},
            "y_axis": {"field": "steps", "values": ["10", "20"]},
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

    assert len(status["result"]["images"]) == 4
    assert status["result"]["grid"] == "xy_grid.png"
    assert seen == [(3.0, 10), (5.0, 10), (3.0, 20), (5.0, 20)]


def test_lora_test_base_prompt_seed_axes(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image
    from lorahub.api.routers import lora_test

    workspace = tmp_path / "ws"
    output = workspace / "output"
    output.mkdir(parents=True)
    (output / "style.safetensors").write_bytes(b"lora")
    job = state.registry.create(workspace=workspace, config_snapshot=_anima_snapshot(tmp_path))
    job.state = state.JobState.succeeded
    state.registry.update(job)

    seen: list[tuple[int, str, int]] = []

    def fake_inference(resolved, req, case, out_path, cancel_evt):  # type: ignore[no-untyped-def]
        seen.append((len(case.loras), case.prompt, case.seed))
        Image.new("RGB", (8, 8), "white").save(out_path)

    monkeypatch.setattr(lora_test, "_run_anima_inference", fake_inference)
    r = client.post(
        "/api/lora-test/generate",
        json={
            "job_id": job.id,
            "checkpoint_path": "output/style.safetensors",
            "prompt": "p0",
            "x_axis": {"field": "variant", "values": ["base", "lora"]},
            "y_axis": {"field": "prompt", "values": ["p1", "p2"]},
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

    assert [item[:2] for item in seen] == [(0, "p1"), (1, "p1"), (0, "p2"), (1, "p2")]


def test_lora_test_size_negative_and_base_weight_axes(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image
    from lorahub.api.routers import lora_test

    workspace = tmp_path / "ws"
    output = workspace / "output"
    output.mkdir(parents=True)
    (output / "style.safetensors").write_bytes(b"lora")
    job = state.registry.create(workspace=workspace, config_snapshot=_anima_snapshot(tmp_path))
    job.state = state.JobState.succeeded
    state.registry.update(job)

    seen: list[tuple[int, int, str, float, int]] = []

    def fake_inference(resolved, req, case, out_path, cancel_evt):  # type: ignore[no-untyped-def]
        seen.append(
            (
                case.width,
                case.height,
                case.negative_prompt,
                case.multipliers[0] if case.multipliers else 0.0,
                len(case.loras),
            )
        )
        Image.new("RGB", (8, 8), "white").save(out_path)

    monkeypatch.setattr(lora_test, "_run_anima_inference", fake_inference)
    r = client.post(
        "/api/lora-test/generate",
        json={
            "job_id": job.id,
            "checkpoint_path": "output/style.safetensors",
            "prompt": "p0",
            "x_axis": {"field": "variant", "values": ["base", "lora"]},
            "y_axis": {"field": "lora_weight", "values": ["0.6", "1.0"]},
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

    assert [(item[3], item[4]) for item in seen] == [
        (0.0, 0),
        (0.6, 1),
        (0.0, 0),
        (1.0, 1),
    ]

    seen.clear()
    r = client.post(
        "/api/lora-test/generate",
        json={
            "job_id": job.id,
            "checkpoint_path": "output/style.safetensors",
            "prompt": "p0",
            "x_axis": {"field": "size", "values": ["768x1344"]},
            "y_axis": {"field": "negative_prompt", "values": ["empty"]},
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

    assert seen[0][:3] == (768, 1344, "")
