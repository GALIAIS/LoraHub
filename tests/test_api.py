"""Tests for the LoraHub HTTP API."""

from __future__ import annotations

import textwrap
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from lorahub.api import state
from lorahub.core.config.schema import TrainingConfig
from lorahub.core.events import EventType, TrainingEvent


def _make_stub_sd_scripts(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stub = textwrap.dedent(
        """
        import sys
        print("epoch 1/1", flush=True)
        print("saving checkpoint: out.safetensors", flush=True)
        sys.exit(0)
        """
    ).strip() + "\n"
    for name in (
        "train_network.py",
        "sdxl_train_network.py",
        "sd3_train_network.py",
        "flux_train_network.py",
        "lumina_train_network.py",
        "hunyuan_image_train_network.py",
        "anima_train_network.py",
    ):
        (root / name).write_text(stub, encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[state.JobRegistry]:
    """Isolate registry + scheduler per-test so threads don't bleed across cases."""
    from lorahub.api import scheduler as sched_module

    fresh = state.JobRegistry()
    monkeypatch.setattr(state, "registry", fresh)
    fresh_sched = sched_module.JobScheduler(concurrency=1)
    monkeypatch.setattr(sched_module, "scheduler", fresh_sched)
    fresh_sched.start()
    try:
        yield fresh
    finally:
        fresh_sched.stop(timeout=2.0)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from lorahub.api import app as app_mod
    from lorahub.api.settings import SettingsStore

    # Isolate the settings store so tests don't read or write the real
    # user-data file. Patch on the imported `app` module 鈥?that's the symbol
    # the request handlers resolve at call time.
    monkeypatch.setattr(
        app_mod, "_settings_store", SettingsStore(tmp_path / "settings.json")
    )
    # Don't let a developer's .env (LORAHUB_KOHYA_*) leak into backend probes 鈥?    # those env vars are valid in production but confuse settings tests.
    monkeypatch.delenv("LORAHUB_KOHYA_SD_SCRIPTS", raising=False)
    monkeypatch.delenv("LORAHUB_KOHYA_PYTHON", raising=False)
    # Reset the singleton bootstrap session so tests can't leak state into
    # one another (each test starts from "idle").
    monkeypatch.setattr(app_mod, "_bootstrap_session", None)
    return TestClient(app_mod.app)


def _config_payload(tmp_path: Path) -> dict[str, Any]:
    sd = _make_stub_sd_scripts(tmp_path / "sd-scripts")
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    return {
        "base_model": {"checkpoint": str(ckpt)},
        "dataset": {"source": str(data)},
        "schedule": {"epochs": 1, "batch_size": 1},
        "sampling": {"enabled": False},
        "backend": {
            "sd_scripts_path": str(sd),
            "python_executable": __import__("sys").executable,
        },
    }


def test_health_returns_version(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "backend" in body
    assert "sd_scripts_path" in body["backend"]


def test_config_schema_is_valid_json_schema(client: TestClient) -> None:
    r = client.get("/api/configs/schema")
    assert r.status_code == 200
    schema = r.json()
    assert schema["title"] == "TrainingConfig"
    assert "base_model" in schema["$defs"] or "base_model" in str(schema)


def test_list_jobs_starts_empty(client: TestClient) -> None:
    r = client.get("/api/jobs")
    assert r.status_code == 200
    assert r.json() == {"jobs": []}


def test_get_unknown_job_returns_404(client: TestClient) -> None:
    r = client.get("/api/jobs/does-not-exist")
    assert r.status_code == 404


def test_create_and_complete_job(client: TestClient, tmp_path: Path) -> None:
    payload = {"config": _config_payload(tmp_path), "workspace": str(tmp_path / "ws")}
    r = client.post("/api/jobs", json=payload)
    assert r.status_code == 202, r.text
    summary = r.json()
    assert summary["state"] in ("queued", "running", "succeeded", "failed")

    job_id = summary["id"]

    # Wait up to ~30s for the stub kohya to finish
    import time

    deadline = time.time() + 30
    while time.time() < deadline:
        s = client.get(f"/api/jobs/{job_id}").json()
        if s["state"] in ("succeeded", "failed"):
            break
        time.sleep(0.2)

    final = client.get(f"/api/jobs/{job_id}").json()
    assert final["state"] == "succeeded", final
    assert final["returncode"] == 0


def test_recent_events_returned_after_completion(
    client: TestClient, tmp_path: Path
) -> None:
    payload = {"config": _config_payload(tmp_path), "workspace": str(tmp_path / "ws")}
    job_id = client.post("/api/jobs", json=payload).json()["id"]

    import time

    deadline = time.time() + 30
    while time.time() < deadline:
        if client.get(f"/api/jobs/{job_id}").json()["state"] in ("succeeded", "failed"):
            break
        time.sleep(0.2)

    events = client.get(f"/api/jobs/{job_id}/events").json()["events"]
    assert any(e["type"] == "epoch_end" for e in events)
    assert any(e["type"] == "checkpoint_saved" for e in events)
    assert events[-1]["type"] == "done"


def test_recent_events_replay_from_workspace_jsonl(
    client: TestClient, tmp_path: Path
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    job = state.registry.create(workspace=ws, config_snapshot={})
    job.state = state.JobState.succeeded
    job.started_at = datetime.now(UTC)
    job.finished_at = datetime.now(UTC)
    state.registry.update(job)

    expected = TrainingEvent(
        type=EventType.log,
        payload={"message": "from disk"},
        job_id=job.id,
    )
    (ws / "events.jsonl").write_text(expected.to_json() + "\n", encoding="utf-8")

    events = client.get(f"/api/jobs/{job.id}/events").json()["events"]

    assert events == [expected.to_dict()]


def test_websocket_replays_workspace_jsonl_for_rehydrated_job(
    client: TestClient, tmp_path: Path
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    job = state.registry.create(workspace=ws, config_snapshot={})
    job.state = state.JobState.succeeded
    state.registry.update(job)

    done = TrainingEvent(
        type=EventType.done,
        payload={"returncode": 0},
        job_id=job.id,
    )
    (ws / "events.jsonl").write_text(done.to_json() + "\n", encoding="utf-8")

    with client.websocket_connect(f"/api/jobs/{job.id}/stream") as websocket:
        event = websocket.receive_json()

    assert event == done.to_dict()


def test_invalid_recipe_returns_422(client: TestClient) -> None:
    r = client.post("/api/jobs", json={"config": {"missing": "everything"}})
    assert r.status_code == 422


def test_resume_unknown_job_returns_404(client: TestClient) -> None:
    r = client.post("/api/jobs/no-such-id/resume")
    assert r.status_code == 404


def test_resume_running_job_returns_409(
    client: TestClient, tmp_path: Path
) -> None:
    """Resume only makes sense for terminated runs; running jobs must 409."""
    ws = tmp_path / "ws"
    ws.mkdir()
    job = state.registry.create(workspace=ws, config_snapshot={})
    job.state = state.JobState.running
    state.registry.update(job)

    r = client.post(f"/api/jobs/{job.id}/resume")
    assert r.status_code == 409
    assert "not resumable" in r.json()["detail"]


def test_resume_without_state_returns_409(
    client: TestClient, tmp_path: Path
) -> None:
    """No `*-state*` directory => no checkpoint to resume from."""
    ws = tmp_path / "ws"
    ws.mkdir()
    job = state.registry.create(
        workspace=ws, config_snapshot=_config_payload(tmp_path)
    )
    job.state = state.JobState.failed
    state.registry.update(job)

    r = client.post(f"/api/jobs/{job.id}/resume")
    assert r.status_code == 409
    assert "state directory" in r.json()["detail"]


def test_resume_without_weights_returns_409(
    client: TestClient, tmp_path: Path
) -> None:
    """A state dir without any safetensors next to it => 409."""
    ws = tmp_path / "ws"
    (ws / "out").mkdir(parents=True)
    (ws / "out" / "lora-state").mkdir()
    job = state.registry.create(
        workspace=ws, config_snapshot=_config_payload(tmp_path)
    )
    job.state = state.JobState.interrupted
    state.registry.update(job)

    r = client.post(f"/api/jobs/{job.id}/resume")
    assert r.status_code == 409
    assert "safetensors" in r.json()["detail"]


def _dp_config_payload(tmp_path: Path) -> dict[str, Any]:
    """Minimal dp recipe snapshot — sdxl arch + dp backend type."""
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    return {
        "base_model": {"checkpoint": str(ckpt), "arch": "sdxl"},
        "dataset": {"source": str(data)},
        "schedule": {"epochs": 1, "batch_size": 1},
        "sampling": {"enabled": False},
        "backend": {"type": "diffusion-pipe"},
    }


def test_resume_dp_without_run_dir_returns_409(
    client: TestClient, tmp_path: Path
) -> None:
    """An interrupted dp job with no output_dir on disk => 409."""
    ws = tmp_path / "ws"
    ws.mkdir()
    job = state.registry.create(
        workspace=ws, config_snapshot=_dp_config_payload(tmp_path)
    )
    job.state = state.JobState.interrupted
    state.registry.update(job)

    r = client.post(f"/api/jobs/{job.id}/resume")
    assert r.status_code == 409
    assert "output_dir" in r.json()["detail"]


def test_resume_dp_run_dir_without_global_step_returns_409(
    client: TestClient, tmp_path: Path
) -> None:
    """Run dir without a `global_step*` subdir is not resumable."""
    ws = tmp_path / "ws"
    run_dir = ws / "output" / "20260101_00-00-00"
    run_dir.mkdir(parents=True)
    (run_dir / "latest").write_text("global_step100", encoding="utf-8")
    # No global_step* subdir.
    job = state.registry.create(
        workspace=ws, config_snapshot=_dp_config_payload(tmp_path)
    )
    job.state = state.JobState.interrupted
    state.registry.update(job)

    r = client.post(f"/api/jobs/{job.id}/resume")
    assert r.status_code == 409
    assert "run directory" in r.json()["detail"]


def test_resume_dp_run_dir_without_latest_file_returns_409(
    client: TestClient, tmp_path: Path
) -> None:
    """Run dir with global_step but no `latest` text file is not resumable."""
    ws = tmp_path / "ws"
    run_dir = ws / "output" / "20260101_00-00-00"
    (run_dir / "global_step100").mkdir(parents=True)
    job = state.registry.create(
        workspace=ws, config_snapshot=_dp_config_payload(tmp_path)
    )
    job.state = state.JobState.interrupted
    state.registry.update(job)

    r = client.post(f"/api/jobs/{job.id}/resume")
    assert r.status_code == 409
    assert "run directory" in r.json()["detail"]


def test_resume_dp_happy_path_relaunches_in_place_with_resume_argv(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete dp run_dir => 202 + same JobRecord re-enqueued with
    `--resume_from_checkpoint=<basename>` and `output.output_dir`
    overridden so dp picks up the same run_dir on relaunch."""
    from lorahub.api import jobs_helpers

    ws = tmp_path / "ws"
    out = ws / "output"
    run_dir = out / "20260102_03-04-05"
    (run_dir / "global_step100").mkdir(parents=True)
    (run_dir / "latest").write_text("global_step100", encoding="utf-8")

    captured: dict[str, Any] = {}

    def stub_enqueue(job, cfg, *, extra_argv=None):  # type: ignore[no-untyped-def]
        captured["job_id"] = job.id
        captured["extra_argv"] = list(extra_argv or [])
        captured["output_dir"] = (
            str(cfg.output.output_dir) if cfg.output.output_dir else None
        )
        captured["state_at_enqueue"] = job.state.value

    monkeypatch.setattr(jobs_helpers, "_enqueue_launch", stub_enqueue)

    original = state.registry.create(
        workspace=ws, config_snapshot=_dp_config_payload(tmp_path)
    )
    original.state = state.JobState.interrupted
    state.registry.update(original)

    r = client.post(f"/api/jobs/{original.id}/resume")
    assert r.status_code == 202, r.text
    summary = r.json()
    # Same id, same workspace — in-place relaunch.
    assert summary["id"] == original.id
    assert Path(summary["workspace"]) == ws
    assert summary["state"] == "queued"
    assert "last_resumed_at" in summary["metadata"]

    assert captured["job_id"] == original.id
    assert captured["state_at_enqueue"] == "queued"
    assert any(
        a == "--resume_from_checkpoint=20260102_03-04-05"
        for a in captured["extra_argv"]
    ), captured
    assert captured["output_dir"] == str(out.resolve())


def test_cancel_queued_job_short_circuits_to_canceled(
    client: TestClient, tmp_path: Path
) -> None:
    """A job pending on the worker deque must cancel without launching."""
    ws = tmp_path / "ws"
    ws.mkdir()
    job = state.registry.create(workspace=ws, config_snapshot={})
    # Default state is 'queued'.
    assert job.state is state.JobState.queued

    r = client.delete(f"/api/jobs/{job.id}")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "canceled"
    assert body["finished_at"] is not None


def test_enqueue_launch_passes_cuda_visible_devices_from_slot(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker slot -> `CUDA_VISIBLE_DEVICES` env var on backend.launch()."""
    from lorahub.api import jobs_helpers
    from lorahub.api import scheduler as sched_module
    from lorahub.core.backends.base import TrainingHandle

    captured: dict[str, Any] = {}

    class FakeBackend:
        def launch(
            self,
            cfg: Any,
            workspace: Path,
            on_event: Any,
            *,
            extra_argv: list[str] | None = None,
            env: dict[str, str] | None = None,
        ) -> TrainingHandle:
            captured["env"] = env
            captured["workspace"] = workspace
            captured["extra_argv"] = extra_argv
            return TrainingHandle(
                job_id="fake",
                pid=0,
                _stop_fn=lambda _g: None,
                _wait_fn=lambda _t: 0,
            )

    monkeypatch.setattr(jobs_helpers, "_select_backend", lambda _cfg: FakeBackend())

    # Use a 2-slot scheduler so we exercise a non-zero slot id (slot=7).
    fresh_sched = sched_module.JobScheduler(concurrency=1, available_slots=[7])
    monkeypatch.setattr(sched_module, "scheduler", fresh_sched)
    fresh_sched.start()
    try:
        payload = {"config": _config_payload(tmp_path), "workspace": str(tmp_path / "ws")}
        r = client.post("/api/jobs", json=payload)
        assert r.status_code == 202, r.text
        job_id = r.json()["id"]

        import time as _time

        deadline = _time.time() + 5.0
        while _time.time() < deadline and "env" not in captured:
            _time.sleep(0.02)
    finally:
        fresh_sched.stop(timeout=2.0)

    assert "env" in captured, "FakeBackend.launch was never invoked"
    assert captured["env"] == {"CUDA_VISIBLE_DEVICES": "7"}


# --------------------------------------------------------------------------- #
# Recipe template browsing
# --------------------------------------------------------------------------- #


@pytest.fixture
def configs_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the API at an isolated recipes directory."""
    rdir = tmp_path / "recipes"
    rdir.mkdir()
    monkeypatch.setenv("LORAHUB_configs_dir", str(rdir))
    return rdir


def _write_valid_config(rdir: Path, name: str = "demo") -> Path:
    ckpt = rdir.parent / "model.safetensors"
    ckpt.write_bytes(b"")
    data = rdir.parent / "data"
    data.mkdir(exist_ok=True)
    body = textwrap.dedent(
        f"""
        base_model:
          arch: sdxl
          checkpoint: {ckpt!s}
        dataset:
          source: {data!s}
        schedule:
          epochs: 2
          batch_size: 1
        sampling:
          enabled: false
        """
    ).strip() + "\n"
    p = rdir / f"{name}.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_list_configs_returns_valid_and_invalid(
    client: TestClient, configs_dir: Path
) -> None:
    _write_valid_config(configs_dir, "good")
    (configs_dir / "broken.yaml").write_text("base_model: {}\n", encoding="utf-8")
    (configs_dir / "ignore-me.txt").write_text("not yaml", encoding="utf-8")

    r = client.get("/api/configs")
    assert r.status_code == 200
    body = r.json()
    names = {it["name"] for it in body["configs"]}
    assert names == {"good", "broken"}

    good = next(it for it in body["configs"] if it["name"] == "good")
    assert good["valid"] is True
    assert good["arch"] == "sdxl"
    assert "epoch" in good["summary"]

    broken = next(it for it in body["configs"] if it["name"] == "broken")
    assert broken["valid"] is False
    assert broken["error"]


def test_get_config_returns_content_and_parsed(
    client: TestClient, configs_dir: Path
) -> None:
    _write_valid_config(configs_dir, "good")
    r = client.get("/api/configs/good")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "good"
    assert "base_model:" in body["content"]
    assert body["parsed"]["base_model"]["arch"] == "sdxl"
    assert body["error"] is None


def test_get_config_missing_returns_404(
    client: TestClient, configs_dir: Path
) -> None:
    r = client.get("/api/configs/nope")
    assert r.status_code == 404


def test_get_config_blocks_path_traversal(
    client: TestClient, configs_dir: Path
) -> None:
    r = client.get("/api/configs/..%2Fpasswd")
    # FastAPI normalizes %2F into /, our handler rejects bare names with slashes
    assert r.status_code in (400, 404)


def test_config_schema_still_resolves_under_recipes_prefix(
    client: TestClient, configs_dir: Path
) -> None:
    # /recipes/schema must keep working alongside /recipes/{name}
    r = client.get("/api/configs/schema")
    assert r.status_code == 200
    assert r.json()["title"] == "TrainingConfig"


# --------------------------------------------------------------------------- #
# Recipe validate + save
# --------------------------------------------------------------------------- #


def _valid_config_dict(tmp_path: Path) -> dict[str, Any]:
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    return {
        "base_model": {"arch": "sdxl", "checkpoint": str(ckpt)},
        "dataset": {"source": str(data)},
        "schedule": {"epochs": 2, "batch_size": 1},
        "sampling": {"enabled": False},
    }


def test_validate_config_returns_normalized_payload(
    client: TestClient, tmp_path: Path
) -> None:
    r = client.post("/api/configs/validate", json={"config": _valid_config_dict(tmp_path)})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert body["normalized"]["base_model"]["arch"] == "sdxl"
    # defaults should be filled in
    assert body["normalized"]["network"]["rank"] >= 1
    assert body["preflight"]["paths"]["checkpoint_exists"] is True
    assert body["preflight"]["paths"]["dataset_exists"] is True
    assert body["preflight"]["vram"]["total_mib"] > 0
    assert isinstance(body["preflight"]["issues"], list)


def test_validate_config_reports_dataset_caption_preflight(
    client: TestClient, tmp_path: Path
) -> None:
    cfg_dict = _valid_config_dict(tmp_path)
    data = Path(str(cfg_dict["dataset"]["source"]))
    (data / "sample.png").write_bytes(b"fake image bytes")

    r = client.post("/api/configs/validate", json={"config": cfg_dict})

    assert r.status_code == 200
    paths = r.json()["preflight"]["paths"]
    assert paths["image_files"] == 1
    assert paths["caption_files"] == 0
    assert paths["missing_caption_files"] == ["sample.png"]


def test_validate_config_returns_structured_errors(client: TestClient) -> None:
    r = client.post("/api/configs/validate", json={"config": {}})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is False
    assert len(body["errors"]) >= 1
    # each error should include a loc list
    assert all("loc" in e for e in body["errors"])


def test_validate_config_rejects_sd15_with_arch_variant(
    client: TestClient, tmp_path: Path
) -> None:
    """arch_variant only makes sense on the SDXL backbone."""
    cfg_dict = _valid_config_dict(tmp_path)
    cfg_dict["base_model"]["arch"] = "sd15"
    cfg_dict["base_model"]["arch_variant"] = "pony"

    r = client.post("/api/configs/validate", json={"config": cfg_dict})
    # The validate route always returns 200; the rejection surfaces via
    # `valid=false` plus a structured error mentioning arch_variant.
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is False
    assert any("arch_variant" in (e.get("msg") or "") for e in body["errors"])

    # The save route validates the same way and returns 422 outright.
    r2 = client.post(
        "/api/configs",
        json={"name": "bad-variant", "config": cfg_dict},
    )
    assert r2.status_code == 422
    assert "arch_variant" in r2.json()["detail"]


def test_save_config_writes_file_and_blocks_overwrite(
    client: TestClient, tmp_path: Path, configs_dir: Path
) -> None:
    payload = {"name": "demo", "config": _valid_config_dict(tmp_path)}
    r = client.post("/api/configs", json=payload)
    assert r.status_code == 201, r.text
    saved = r.json()
    assert saved["filename"] == "demo.yaml"
    assert (configs_dir / "demo.yaml").is_file()

    # Repeat without overwrite 鈥?should 409
    r2 = client.post("/api/configs", json=payload)
    assert r2.status_code == 409

    # With overwrite 鈥?should 201
    r3 = client.post("/api/configs", json={**payload, "overwrite": True})
    assert r3.status_code == 201


def test_save_config_rejects_invalid_name(
    client: TestClient, tmp_path: Path, configs_dir: Path
) -> None:
    r = client.post(
        "/api/configs",
        json={"name": "../etc/passwd", "config": _valid_config_dict(tmp_path)},
    )
    assert r.status_code == 400


def test_save_config_rejects_invalid_recipe(
    client: TestClient, configs_dir: Path
) -> None:
    r = client.post("/api/configs", json={"name": "bad", "config": {}})
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def test_get_settings_returns_defaults(client: TestClient) -> None:
    r = client.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert body["settings"]["sd_scripts_path"] is None
    assert body["settings"]["tagger_device"] == "auto"
    assert "sd_scripts_path" in body["backend"]
    assert body["path"].endswith("settings.json")


def test_put_settings_persists_and_reflects_in_get(
    client: TestClient, tmp_path: Path
) -> None:
    sd = tmp_path / "fake-sd-scripts"
    sd.mkdir()
    payload = {
        "sd_scripts_path": str(sd),
        "python_executable": "",  # empty -> clear
        "tagger_device": "cpu",
    }
    r = client.put("/api/settings", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["settings"]["sd_scripts_path"] == str(sd)
    assert body["settings"]["python_executable"] is None
    assert body["settings"]["tagger_device"] == "cpu"

    # GET should round-trip the same values
    r2 = client.get("/api/settings")
    assert r2.json()["settings"]["sd_scripts_path"] == str(sd)
    assert r2.json()["backend"]["sd_scripts_path"] == str(sd)


def test_put_settings_rejects_bad_tagger_device(client: TestClient) -> None:
    r = client.put("/api/settings", json={"tagger_device": "tpu"})
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# Dataset scanning
# --------------------------------------------------------------------------- #


def test_scan_dataset_summarizes_images_and_captions(
    client: TestClient, tmp_path: Path
) -> None:
    data = tmp_path / "dataset"
    data.mkdir()
    (data / "one.png").write_bytes(b"image")
    (data / "one.txt").write_text("blue hair, solo\n", encoding="utf-8")
    (data / "two.jpg").write_bytes(b"image")
    (data / "notes.txt").write_text("ignore me\n", encoding="utf-8")

    r = client.get("/api/datasets/scan", params={"path": str(data)})

    assert r.status_code == 200
    body = r.json()
    assert body["exists"] is True
    assert body["image_files"] == 2
    assert body["caption_files"] == 1
    assert body["missing_caption_files"] == ["two.jpg"]
    assert body["samples"][0]["caption"] == "blue hair, solo"


def test_scan_dataset_missing_path_returns_empty_summary(client: TestClient) -> None:
    r = client.get("/api/datasets/scan", params={"path": "Z:/definitely/missing"})

    assert r.status_code == 200
    assert r.json()["exists"] is False


# --------------------------------------------------------------------------- #
# Dataset thumbnails + caption I/O
# --------------------------------------------------------------------------- #


def _write_png(target: Path, color: tuple[int, int, int] = (200, 80, 40)) -> Path:
    """Write a real (Pillow-decodable) 4x4 PNG so thumbnail generation works."""
    from PIL import Image

    img = Image.new("RGB", (4, 4), color=color)
    target.parent.mkdir(parents=True, exist_ok=True)
    img.save(target, format="PNG")
    return target


def test_thumb_generates_and_caches_webp(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid image under an allowed root yields a WEBP thumbnail; the second
    call must hit the on-disk cache (same content, same path)."""
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    img = _write_png(tmp_path / "ds" / "one.png")

    r1 = client.get("/api/datasets/thumb", params={"path": str(img), "size": 64})
    assert r1.status_code == 200, r1.text
    assert r1.headers["content-type"] == "image/webp"
    # Cache header is present so the browser can re-use the thumbnail.
    assert "max-age" in r1.headers.get("cache-control", "")
    body1 = r1.content
    assert body1[:4] == b"RIFF"  # WEBP magic prefix

    # Cache file should now exist on disk.
    cache_dir = tmp_path / "runs" / ".thumbs"
    assert cache_dir.is_dir()
    cached = list(cache_dir.glob("*.webp"))
    assert len(cached) == 1

    # Second call returns the same bytes (served from cache).
    r2 = client.get("/api/datasets/thumb", params={"path": str(img), "size": 64})
    assert r2.status_code == 200
    assert r2.content == body1


def test_thumb_blocks_path_traversal(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path outside every allowed root must be rejected with 400, not 404."""
    (tmp_path / "ds").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(tmp_path / "ds"))
    monkeypatch.chdir(tmp_path / "ds")

    # Path-traversal attempt that resolves above the allowed root.
    bad = tmp_path / "outside" / "secret.png"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"")

    r = client.get(
        "/api/datasets/thumb",
        params={"path": str(bad), "size": 128},
    )
    assert r.status_code == 400
    assert "outside" in r.json()["detail"].lower()


def test_thumb_rejects_non_image_suffix(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even inside an allowed root, a non-image file is rejected with 400."""
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "notes.txt"
    target.write_text("hi", encoding="utf-8")

    r = client.get(
        "/api/datasets/thumb", params={"path": str(target), "size": 128}
    )
    assert r.status_code == 400


def test_thumb_size_bounds_enforced(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    img = _write_png(tmp_path / "one.png")

    too_small = client.get(
        "/api/datasets/thumb", params={"path": str(img), "size": 1}
    )
    too_big = client.get(
        "/api/datasets/thumb", params={"path": str(img), "size": 99999}
    )
    assert too_small.status_code == 400
    assert too_big.status_code == 400


def test_thumb_corrupt_image_returns_404(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pillow can't decode a `.png` of arbitrary bytes 鈥?we surface 404."""
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"definitely not a real png")

    r = client.get("/api/datasets/thumb", params={"path": str(bad), "size": 64})
    assert r.status_code == 404


def test_get_caption_returns_existing_text(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    img = _write_png(tmp_path / "shot.png")
    img.with_suffix(".txt").write_text("blue eyes, smile", encoding="utf-8")

    r = client.get("/api/datasets/caption", params={"path": str(img)})
    assert r.status_code == 200
    body = r.json()
    assert body["caption"] == "blue eyes, smile"
    assert body["path"].endswith("shot.txt")


def test_get_caption_missing_returns_null(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing companion is `caption=null`, not an error 鈥?there's nothing
    wrong with an unlabeled image, that's just the pre-tagging state."""
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    img = _write_png(tmp_path / "shot.png")

    r = client.get("/api/datasets/caption", params={"path": str(img)})
    assert r.status_code == 200
    assert r.json()["caption"] is None


def test_put_caption_writes_file_and_round_trips(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    img = _write_png(tmp_path / "shot.png")

    r = client.put(
        "/api/datasets/caption",
        json={"path": str(img), "caption": "blue hair, solo\nportrait"},
    )
    assert r.status_code == 200, r.text
    # File is written as UTF-8 with LF newlines (no BOM).
    on_disk = (tmp_path / "shot.txt").read_bytes()
    assert on_disk == b"blue hair, solo\nportrait"

    # GET reflects the write.
    g = client.get("/api/datasets/caption", params={"path": str(img)}).json()
    assert g["caption"] == "blue hair, solo\nportrait"


def test_put_caption_normalises_crlf_to_lf(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    img = _write_png(tmp_path / "shot.png")

    r = client.put(
        "/api/datasets/caption",
        json={"path": str(img), "caption": "first\r\nsecond\rthird"},
    )
    assert r.status_code == 200
    assert (tmp_path / "shot.txt").read_bytes() == b"first\nsecond\nthird"


def test_put_caption_blocks_path_traversal(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "ds").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(tmp_path / "ds"))
    monkeypatch.chdir(tmp_path / "ds")
    bad = tmp_path / "outside" / "secret.png"
    bad.parent.mkdir(parents=True, exist_ok=True)

    r = client.put(
        "/api/datasets/caption",
        json={"path": str(bad), "caption": "should not be written"},
    )
    assert r.status_code == 400
    # Make sure no .txt was created above the allowed root.
    assert not (tmp_path / "outside" / "secret.txt").exists()


def test_caption_endpoints_reject_non_image_suffix(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`path` must point at an image. The .txt sibling is derived server-side
    so the user can't supply a raw .txt path and overwrite an arbitrary file."""
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    foreign = tmp_path / "secrets.txt"
    foreign.write_text("don't overwrite me", encoding="utf-8")

    g = client.get("/api/datasets/caption", params={"path": str(foreign)})
    p = client.put(
        "/api/datasets/caption",
        json={"path": str(foreign), "caption": "pwn"},
    )
    assert g.status_code == 400
    assert p.status_code == 400
    # Original file content is untouched.
    assert foreign.read_text(encoding="utf-8") == "don't overwrite me"


# --------------------------------------------------------------------------- #
# Rerun / reveal / archive
# --------------------------------------------------------------------------- #


def _wait_terminal(client: TestClient, job_id: str, timeout: float = 30.0) -> dict[str, Any]:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        s = client.get(f"/api/jobs/{job_id}").json()
        if s["state"] in ("succeeded", "failed", "canceled", "interrupted"):
            return s
        time.sleep(0.2)
    return client.get(f"/api/jobs/{job_id}").json()


def test_rerun_relaunches_job_in_place(client: TestClient, tmp_path: Path) -> None:
    """Rerun re-uses the original JobRecord and workspace; no new id.

    A logical job stays as a single timeline regardless of how many times
    the user re-runs it — the events log appends to the same file.
    """
    payload = {"config": _config_payload(tmp_path), "workspace": str(tmp_path / "ws")}
    first = client.post("/api/jobs", json=payload).json()
    first_id = first["id"]
    final_first = _wait_terminal(client, first_id)
    assert final_first["state"] == "succeeded", final_first

    r = client.post(f"/api/jobs/{first_id}/rerun")
    assert r.status_code == 202, r.text
    fresh = r.json()
    # Same id, same workspace — in-place relaunch.
    assert fresh["id"] == first_id
    assert fresh["workspace"] == final_first["workspace"]
    assert fresh["state"] == "queued"
    # Runtime fields cleared from the previous terminal state.
    assert fresh["finished_at"] is None
    assert fresh["returncode"] is None
    assert fresh["error"] is None

    final_fresh = _wait_terminal(client, fresh["id"])
    assert final_fresh["state"] == "succeeded", final_fresh
    assert final_fresh["returncode"] == 0


def test_rerun_refuses_when_active(client: TestClient, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    job = state.registry.create(
        workspace=ws, config_snapshot=_config_payload(tmp_path)
    )
    job.state = state.JobState.running
    state.registry.update(job)

    r = client.post(f"/api/jobs/{job.id}/rerun")
    assert r.status_code == 409
    assert "cancel" in r.json()["detail"]


def test_rerun_unknown_job_404(client: TestClient) -> None:
    r = client.post("/api/jobs/does-not-exist/rerun")
    assert r.status_code == 404


def test_reveal_unknown_job_404(client: TestClient) -> None:
    r = client.post("/api/jobs/does-not-exist/reveal")
    assert r.status_code == 404


def test_reveal_existing_job_invokes_subprocess(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    job = state.registry.create(workspace=ws, config_snapshot={})
    job.state = state.JobState.succeeded
    state.registry.update(job)

    captured: dict[str, Any] = {}

    def fake_popen(argv: list[str], **kwargs: Any) -> None:
        captured["argv"] = argv
        captured["kwargs"] = kwargs

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    r = client.post(f"/api/jobs/{job.id}/reveal")
    assert r.status_code == 200, r.text
    assert r.json()["opened"] == str(ws)
    assert captured["argv"][-1] == str(ws)
    # Never use shell=True 鈥?argv form only.
    assert "shell" not in captured["kwargs"] or captured["kwargs"]["shell"] is False


def test_reveal_returns_409_when_workspace_missing(
    client: TestClient, tmp_path: Path
) -> None:
    job = state.registry.create(
        workspace=tmp_path / "gone", config_snapshot={}
    )
    job.state = state.JobState.succeeded
    state.registry.update(job)

    r = client.post(f"/api/jobs/{job.id}/reveal")
    assert r.status_code == 409


def test_archive_completed_job_moves_workspace(
    client: TestClient, tmp_path: Path
) -> None:
    payload = {"config": _config_payload(tmp_path), "workspace": str(tmp_path / "ws")}
    job_id = client.post("/api/jobs", json=payload).json()["id"]
    final = _wait_terminal(client, job_id)
    assert final["state"] == "succeeded", final
    workspace = Path(final["workspace"])
    assert workspace.exists()

    r = client.delete(f"/api/jobs/{job_id}", params={"archive": "true"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["archived"] is True
    assert body["warnings"] == []
    moved_to = body["workspace_moved_to"]
    assert moved_to is not None
    moved = Path(moved_to)
    assert moved.exists()
    assert moved.parent.name == "_archive"
    assert not workspace.exists()

    # Job is gone from both registry and store.
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_archive_refuses_when_interrupted_job_pid_still_alive(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An `interrupted` job whose process is still running (uvicorn restart
    while the deepspeed launcher kept going) must NOT be archivable: the
    workspace `mv` would yank the directory out from under tensorboard.
    """
    import os

    from lorahub.api import state as state_mod
    from lorahub.api.state import JobRecord, JobState

    # Hand-craft an interrupted job whose PID is the test process itself —
    # which is guaranteed alive for the duration of the assertion.
    workspace = tmp_path / "ws-interrupted"
    workspace.mkdir(parents=True)
    record = JobRecord(
        id="01TEST_INTERRUPTED",
        state=JobState.interrupted,
        workspace=workspace,
        config_snapshot={},
        created_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        pid=os.getpid(),
    )
    state_mod.registry._jobs[record.id] = record  # noqa: SLF001
    state_mod.registry._listeners[record.id] = []  # noqa: SLF001

    r = client.delete(f"/api/jobs/{record.id}", params={"archive": "true"})
    assert r.status_code == 409, r.text
    assert "still alive" in r.json()["detail"]
    # Workspace must still be in place — the mv was refused.
    assert workspace.exists()


def test_kill_unknown_job_returns_404(client: TestClient) -> None:
    r = client.post("/api/jobs/no-such-id/kill")
    assert r.status_code == 404


def test_kill_job_without_pid_returns_409(
    client: TestClient, tmp_path: Path
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    job = state.registry.create(workspace=ws, config_snapshot={})
    job.state = state.JobState.running
    job.pid = None
    state.registry.update(job)

    r = client.post(f"/api/jobs/{job.id}/kill")
    assert r.status_code == 409
    assert "no recorded PID" in r.json()["detail"]


def test_kill_dead_pid_still_flips_state_to_interrupted(
    client: TestClient, tmp_path: Path
) -> None:
    """A job whose PID is already gone — kill should not raise; flip state."""
    ws = tmp_path / "ws"
    ws.mkdir()
    job = state.registry.create(workspace=ws, config_snapshot={})
    job.state = state.JobState.running
    # PID -1 is guaranteed not to exist; os.kill raises ProcessLookupError.
    job.pid = 999_999_999
    state.registry.update(job)

    r = client.post(f"/api/jobs/{job.id}/kill")
    # Either 200 (with warning) or 500 — depends on whether the OS has any
    # process at this PID. Accept either, but state must be interrupted on 200.
    if r.status_code == 200:
        body = r.json()
        assert body["killed_process_group"] is False
        assert body["killed_pid_only"] is False
        refreshed = state.registry.get(job.id)
        assert refreshed is not None
        assert refreshed.state is state.JobState.interrupted


def test_archive_running_job_returns_409(client: TestClient, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    job = state.registry.create(workspace=ws, config_snapshot={})
    job.state = state.JobState.running
    state.registry.update(job)

    r = client.delete(f"/api/jobs/{job.id}", params={"archive": "true"})
    assert r.status_code == 409
    # Job is still tracked.
    assert client.get(f"/api/jobs/{job.id}").status_code == 200


def test_archive_unknown_job_404(client: TestClient) -> None:
    r = client.delete("/api/jobs/does-not-exist", params={"archive": "true"})
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Backend bootstrap (one-click kohya install)
# --------------------------------------------------------------------------- #


def test_bootstrap_status_when_idle(client: TestClient) -> None:
    r = client.get("/api/backend/bootstrap/status")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "idle"
    assert body["events"] == []


def test_bootstrap_concurrent_returns_409(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second POST while one install is running must 409 instead of racing."""
    import threading

    from lorahub.api import app as app_mod

    release = threading.Event()

    def builder(_req: object) -> Any:
        def runner(progress: Any) -> None:
            progress("clone")
            # Block here so the session stays in `running` while we issue the
            # second POST below 鈥?this is the whole point of the test.
            release.wait(timeout=5.0)

        return runner

    monkeypatch.setattr(app_mod, "_build_bootstrap_runner", builder)

    first = client.post("/api/backend/bootstrap", json={"target": str(tmp_path / "sd")})
    assert first.status_code == 202, first.text

    second = client.post("/api/backend/bootstrap", json={"target": str(tmp_path / "sd")})
    assert second.status_code == 409

    # Let the first install finish so the test doesn't hang the worker thread.
    release.set()


def test_bootstrap_succeeds_with_stub(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fast stubbed install should walk to status=succeeded and emit events."""
    import time

    from lorahub.api import app as app_mod

    def builder(_req: object) -> Any:
        def runner(progress: Any) -> None:
            progress("clone")
            progress("create venv")
            progress("install xformers")

        return runner

    monkeypatch.setattr(app_mod, "_build_bootstrap_runner", builder)

    r = client.post("/api/backend/bootstrap", json={})
    assert r.status_code == 202, r.text
    # The POST returns as soon as the worker thread is spawned; for very fast
    # stubs the install may already be done 鈥?both running and succeeded are OK.
    assert r.json()["status"] in ("running", "succeeded")

    deadline = time.time() + 5
    final: dict[str, Any] | None = None
    while time.time() < deadline:
        body = client.get("/api/backend/bootstrap/status").json()
        if body["status"] in ("succeeded", "failed"):
            final = body
            break
        time.sleep(0.05)

    assert final is not None, "bootstrap did not reach a terminal state"
    assert final["status"] == "succeeded", final
    levels = [e["level"] for e in final["events"]]
    assert levels[-1] == "done"
    assert "info" in levels  # at least one progress step was buffered


# --------------------------------------------------------------------------- #
# System telemetry
# --------------------------------------------------------------------------- #


def test_system_stats_returns_full_snapshot(client: TestClient) -> None:
    r = client.get("/api/system/stats")
    assert r.status_code == 200
    body = r.json()
    # Always-on fields, even if their probes degrade.
    for key in ("timestamp", "host", "cpu", "memory", "disks", "gpus", "has_psutil", "has_nvidia_smi"):
        assert key in body, f"missing top-level key {key}"
    assert body["host"]["python"]
    assert body["cpu"]["cores_logical"] >= 1
    assert isinstance(body["disks"], list) and len(body["disks"]) >= 1
    assert isinstance(body["gpus"], list)  # may be empty on non-NVIDIA hosts
    # Memory shape 鈥?fall back to 0s rather than missing keys.
    for key in ("total_bytes", "used_bytes", "available_bytes", "percent"):
        assert key in body["memory"]


def test_system_stats_disk_entry_has_paths_and_percentage(client: TestClient) -> None:
    body = client.get("/api/system/stats").json()
    disk = body["disks"][0]
    assert disk["path"]
    assert disk["label"]
    assert 0.0 <= disk["percent"] <= 100.0
    assert disk["total_bytes"] >= disk["used_bytes"]


def test_system_stats_includes_optional_fields(client: TestClient) -> None:
    """New cross-platform fields must be present (value may be None where the
    host does not expose them) and existing GPU shape must stay backwards
    compatible.
    """
    body = client.get("/api/system/stats").json()

    # CPU: new fields exist on the dict, may be None on Windows / minimal hosts.
    cpu = body["cpu"]
    assert "frequency_mhz" in cpu
    assert "cpu_temperature_c" in cpu
    assert cpu["frequency_mhz"] is None or isinstance(cpu["frequency_mhz"], (int, float))
    assert cpu["cpu_temperature_c"] is None or isinstance(cpu["cpu_temperature_c"], (int, float))

    # Battery is a top-level key; None on desktops/servers, dict on laptops.
    assert "battery" in body
    if body["battery"] is not None:
        for key in ("percent", "plugged", "secs_left"):
            assert key in body["battery"]

    # GPU shape: legacy callers still get index/name/driver.
    for gpu in body["gpus"]:
        for key in ("index", "name", "driver", "vendor"):
            assert key in gpu, f"missing GPU key {key}"


# ----------------------------------------------------------------------------
# Job artifacts and metrics endpoints
# ----------------------------------------------------------------------------


def _make_job_with_workspace(ws: Path) -> str:
    """Create a job record bound to `ws` and return its id.

    The fixture in this module patches `state.registry`, so callers can keep
    creating jobs without bleeding into other tests.
    """
    ws.mkdir(parents=True, exist_ok=True)
    job = state.registry.create(workspace=ws, config_snapshot={})
    return job.id


def test_job_files_lists_workspace_artifacts(
    client: TestClient, tmp_path: Path
) -> None:
    ws = tmp_path / "run-1"
    ws.mkdir()
    (ws / "model.safetensors").write_bytes(b"weights")
    (ws / "config.yaml").write_text("name: test\n", encoding="utf-8")
    (ws / "events.jsonl").write_text("", encoding="utf-8")
    out_dir = ws / "output"
    out_dir.mkdir()
    (out_dir / "sample-1.png").write_bytes(b"\x89PNG\r\n")
    # Archive subdirs should be ignored.
    archive = ws / "_archive" / "old"
    archive.mkdir(parents=True)
    (archive / "stale.safetensors").write_bytes(b"old")

    job_id = _make_job_with_workspace(ws)

    r = client.get(f"/api/jobs/{job_id}/files")
    assert r.status_code == 200
    body = r.json()

    assert body["workspace"] == str(ws)

    checkpoints = {e["path"] for e in body["checkpoints"]}
    samples = {e["path"] for e in body["samples"]}
    logs = {e["path"] for e in body["logs"]}
    other = {e["path"] for e in body["other"]}

    assert checkpoints == {"model.safetensors"}
    assert samples == {"output/sample-1.png"}
    assert logs == {"events.jsonl"}
    assert "config.yaml" in other
    # Archive contents must be filtered out entirely.
    assert all("_archive" not in e["path"] for e in body["checkpoints"])
    # Each entry carries size + mtime.
    ckpt = body["checkpoints"][0]
    assert ckpt["size_bytes"] == len(b"weights")
    assert isinstance(ckpt["modified_at"], (int, float))


def test_job_files_unknown_id_404(client: TestClient) -> None:
    r = client.get("/api/jobs/does-not-exist/files")
    assert r.status_code == 404


def test_job_files_raw_blocks_traversal(
    client: TestClient, tmp_path: Path
) -> None:
    ws = tmp_path / "run-2"
    job_id = _make_job_with_workspace(ws)

    r = client.get(
        f"/api/jobs/{job_id}/files/raw",
        params={"path": "../../../etc/passwd"},
    )
    assert r.status_code == 400


def test_job_files_raw_serves_workspace_file(
    client: TestClient, tmp_path: Path
) -> None:
    ws = tmp_path / "run-3"
    ws.mkdir()
    (ws / "events.jsonl").write_bytes(b"hello\n")
    job_id = _make_job_with_workspace(ws)

    r = client.get(
        f"/api/jobs/{job_id}/files/raw", params={"path": "events.jsonl"}
    )
    assert r.status_code == 200
    assert r.content == b"hello\n"


def test_job_metrics_parses_events_jsonl(
    client: TestClient, tmp_path: Path
) -> None:
    ws = tmp_path / "run-metrics"
    ws.mkdir()
    log = ws / "events.jsonl"
    lines = []
    base_ts = 1_700_000_000.0
    for step in range(1, 6):
        ev = TrainingEvent(
            type=EventType.step,
            payload={"step": step, "total_steps": 100, "loss": 1.0 / step},
            timestamp=base_ts + step,
        )
        lines.append(ev.to_json())
    epoch_ev = TrainingEvent(
        type=EventType.epoch_end,
        payload={"epoch": 1, "total_epochs": 1},
        timestamp=base_ts + 6,
    )
    lines.append(epoch_ev.to_json())
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    job_id = _make_job_with_workspace(ws)

    r = client.get(f"/api/jobs/{job_id}/metrics")
    assert r.status_code == 200
    body = r.json()

    assert len(body["loss"]) == 5
    assert body["loss"][0]["step"] == 1
    assert body["loss"][-1]["step"] == 5
    assert body["loss"][0]["loss"] == pytest.approx(1.0)
    assert len(body["epochs"]) == 1
    assert body["epochs"][0]["epoch"] == 1
    assert body["first_step_ts"] == pytest.approx(base_ts + 1)
    assert body["last_step_ts"] == pytest.approx(base_ts + 5)
    assert body["duration_s"] == pytest.approx(4.0)


def test_job_metrics_handles_corrupt_lines(
    client: TestClient, tmp_path: Path
) -> None:
    ws = tmp_path / "run-corrupt"
    ws.mkdir()
    log = ws / "events.jsonl"
    base_ts = 1_700_000_100.0
    good_lines = [
        TrainingEvent(
            type=EventType.step,
            payload={"step": 1, "total_steps": 10, "loss": 0.5},
            timestamp=base_ts + 1,
        ).to_json(),
        # Garbage line 鈥?must not break parsing of the rest.
        "{not json at all",
        TrainingEvent(
            type=EventType.step,
            payload={"step": 2, "total_steps": 10, "loss": 0.4},
            timestamp=base_ts + 2,
        ).to_json(),
    ]
    log.write_text("\n".join(good_lines) + "\n", encoding="utf-8")

    job_id = _make_job_with_workspace(ws)

    r = client.get(f"/api/jobs/{job_id}/metrics")
    assert r.status_code == 200
    body = r.json()
    # Both valid step lines parse; the garbage line is skipped.
    assert [p["step"] for p in body["loss"]] == [1, 2]


def test_job_metrics_missing_log_returns_empty(
    client: TestClient, tmp_path: Path
) -> None:
    ws = tmp_path / "run-empty"
    job_id = _make_job_with_workspace(ws)

    r = client.get(f"/api/jobs/{job_id}/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["loss"] == []
    assert body["epochs"] == []
    assert body["duration_s"] is None
    # New v1.0 fields stay backwards-compatible: empty list + null trend
    # when no validation events have ever been written.
    assert body["val_loss"] == []
    assert body["overfit_signal"]["trend"] is None
    assert body["overfit_signal"]["gap"] is None


def test_job_metrics_returns_val_loss_and_overfit_signal(
    client: TestClient, tmp_path: Path
) -> None:
    """Train loss flat-ish + val loss climbing -> overfitting trend."""
    ws = tmp_path / "run-overfit"
    ws.mkdir()
    log = ws / "events.jsonl"
    base_ts = 1_700_000_500.0

    lines: list[str] = []
    # Train loss decreasing 鈥?5 step events.
    train_losses = [0.50, 0.40, 0.30, 0.22, 0.18]
    for i, tl in enumerate(train_losses, start=1):
        lines.append(
            TrainingEvent(
                type=EventType.step,
                payload={"step": i, "total_steps": 100, "loss": tl},
                timestamp=base_ts + i,
            ).to_json()
        )
    # Validation loss going up 鈥?classic overfit signature.
    val_losses = [(1, 0.45), (2, 0.50), (3, 0.58)]
    for ep, vl in val_losses:
        lines.append(
            TrainingEvent(
                type=EventType.validation,
                payload={"epoch": ep, "val_loss": vl},
                timestamp=base_ts + 10 + ep,
            ).to_json()
        )
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    job_id = _make_job_with_workspace(ws)

    r = client.get(f"/api/jobs/{job_id}/metrics")
    assert r.status_code == 200
    body = r.json()

    assert len(body["loss"]) == 5
    assert [p["epoch"] for p in body["val_loss"]] == [1, 2, 3]
    assert [p["val_loss"] for p in body["val_loss"]] == [0.45, 0.50, 0.58]

    sig = body["overfit_signal"]
    assert sig["latest_train"] == pytest.approx(0.18)
    assert sig["latest_val"] == pytest.approx(0.58)
    assert sig["gap"] == pytest.approx(0.40)
    assert sig["trend"] == "overfitting"


# --------------------------------------------------------------------------- #
# Recipe duplicate / rename / delete / templates / import
# --------------------------------------------------------------------------- #


def test_duplicate_config_creates_copy(
    client: TestClient, configs_dir: Path
) -> None:
    src = _write_valid_config(configs_dir, "demo")

    r = client.post("/api/configs/demo/duplicate", json={"new_name": "demo_v2"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "demo_v2"
    assert body["filename"] == "demo_v2.yaml"
    copy = configs_dir / "demo_v2.yaml"
    assert copy.is_file()
    assert copy.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")

    # Source missing -> 404
    r_missing = client.post(
        "/api/configs/nope/duplicate", json={"new_name": "ghost"}
    )
    assert r_missing.status_code == 404

    # Destination already exists -> 409
    r_clash = client.post(
        "/api/configs/demo/duplicate", json={"new_name": "demo_v2"}
    )
    assert r_clash.status_code == 409

    # Bad new_name -> 400
    r_bad = client.post(
        "/api/configs/demo/duplicate", json={"new_name": "../etc/passwd"}
    )
    assert r_bad.status_code == 400


def test_rename_config(client: TestClient, configs_dir: Path) -> None:
    _write_valid_config(configs_dir, "demo")
    _write_valid_config(configs_dir, "other")

    r = client.post("/api/configs/demo/rename", json={"new_name": "demo_renamed"})
    assert r.status_code == 200, r.text
    assert not (configs_dir / "demo.yaml").exists()
    assert (configs_dir / "demo_renamed.yaml").is_file()

    # Renaming to a name that's already taken -> 409
    r_clash = client.post(
        "/api/configs/demo_renamed/rename", json={"new_name": "other"}
    )
    assert r_clash.status_code == 409

    # Renaming a missing recipe -> 404
    r_missing = client.post(
        "/api/configs/ghost/rename", json={"new_name": "demo_v3"}
    )
    assert r_missing.status_code == 404


def test_delete_config(client: TestClient, configs_dir: Path) -> None:
    _write_valid_config(configs_dir, "demo")

    r = client.delete("/api/configs/demo")
    assert r.status_code == 200, r.text
    assert r.json() == {"deleted": True, "name": "demo"}

    # Now it's gone
    assert client.get("/api/configs/demo").status_code == 404
    # Re-deleting a missing recipe -> 404
    assert client.delete("/api/configs/demo").status_code == 404


def test_list_templates_returns_validated_configs(client: TestClient) -> None:
    r = client.get("/api/configs/templates")
    assert r.status_code == 200, r.text
    body = r.json()

    ids = {t["id"] for t in body["templates"]}
    assert ids == {
        "sdxl_character",
        "sdxl_style",
        "sd15_character",
        "blank",
        "low_vram",
    }

    # Each template recipe must round-trip through the schema.
    for tpl in body["templates"]:
        cfg = TrainingConfig.model_validate(tpl["config"])
        assert cfg.base_model.arch in {"sdxl", "sd15", "flux", "sd3"}


def test_list_templates_skips_invalid_yaml_files(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One good YAML + one schema-invalid YAML -> only the good one survives,
    and the bad one logs a warning instead of taking the endpoint down.
    """
    from lorahub.api import config_templates as config_templates_module

    builtin_dir = tmp_path / "builtin"
    builtin_dir.mkdir()

    good = {
        "_template": {
            "name": "Good Template",
            "description": "A valid template for testing.",
            "arch": "sdxl",
        },
        "base_model": {"arch": "sdxl", "checkpoint": ""},
        "dataset": {"source": ""},
    }
    (builtin_dir / "good.yaml").write_text(
        yaml.safe_dump(good, sort_keys=False), encoding="utf-8"
    )

    # Missing the required `dataset` key -> TrainingConfig.model_validate fails.
    bad = {
        "_template": {"name": "Bad Template", "description": "x", "arch": "sdxl"},
        "base_model": {"arch": "sdxl", "checkpoint": ""},
    }
    (builtin_dir / "bad.yaml").write_text(
        yaml.safe_dump(bad, sort_keys=False), encoding="utf-8"
    )

    monkeypatch.setattr(
        config_templates_module, "_DEFAULT_BUILTIN_DIR", builtin_dir
    )

    caplog.set_level("WARNING", logger=config_templates_module.logger.name)

    r = client.get("/api/configs/templates")
    assert r.status_code == 200, r.text
    body = r.json()

    ids = [t["id"] for t in body["templates"]]
    assert ids == ["good"], body
    assert body["templates"][0]["name"] == "Good Template"
    assert body["templates"][0]["arch"] == "sdxl"

    warnings = [rec.message for rec in caplog.records if rec.levelname == "WARNING"]
    assert any("bad.yaml" in msg for msg in warnings), warnings


def test_import_config_from_yaml(
    client: TestClient, tmp_path: Path, configs_dir: Path
) -> None:
    config_dict = _valid_config_dict(tmp_path)
    yaml_bytes = yaml.safe_dump(config_dict, sort_keys=False).encode("utf-8")

    r = client.post(
        "/api/configs/import",
        files={"file": ("foo.yaml", yaml_bytes, "application/x-yaml")},
        data={"name": "imported"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "imported"
    assert body["filename"] == "imported.yaml"
    saved = configs_dir / "imported.yaml"
    assert saved.is_file()
    # The persisted file is canonical YAML emitted by dump_config; just confirm
    # it loads back to an equivalent TrainingConfig.
    assert saved.read_text(encoding="utf-8").startswith("schema_version")


# --------------------------------------------------------------------------- #
# Backend catalog + multi-backend selection
# --------------------------------------------------------------------------- #


def test_get_backends_lists_kohya_and_diffusion_pipe(client: TestClient) -> None:
    r = client.get("/api/backends")
    assert r.status_code == 200, r.text
    body = r.json()
    ids = [b["id"] for b in body["backends"]]
    assert "kohya" in ids
    assert "diffusion-pipe" in ids
    # Default is kohya until the user picks otherwise.
    assert body["default"] == "kohya"
    # Each entry exposes UI metadata + a probe payload the UI can render.
    for entry in body["backends"]:
        for key in ("name", "description", "repo_url", "default_path", "ready", "status"):
            assert key in entry, f"missing backend key {key} in {entry['id']}"


def test_settings_can_set_default_backend(client: TestClient) -> None:
    r = client.put("/api/settings", json={"default_backend": "diffusion-pipe"})
    assert r.status_code == 200, r.text
    assert r.json()["settings"]["default_backend"] == "diffusion-pipe"

    # Round-trip: GET reflects the persisted choice.
    r2 = client.get("/api/settings")
    assert r2.json()["settings"]["default_backend"] == "diffusion-pipe"
    # /api/backends advertises the same default.
    assert client.get("/api/backends").json()["default"] == "diffusion-pipe"


def test_settings_rejects_unknown_backend(client: TestClient) -> None:
    r = client.put("/api/settings", json={"default_backend": "invalid"})
    assert r.status_code == 422


def test_bootstrap_with_diffusion_pipe_backend(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issuing a bootstrap with backend='diffusion-pipe' walks to succeeded."""
    import time

    from lorahub.api import app as app_mod
    from lorahub.api.bootstrap_session import BootstrapRequest

    captured: dict[str, Any] = {}

    def builder(req: BootstrapRequest) -> Any:
        captured["backend"] = req.backend

        def runner(progress: Any) -> None:
            progress("clone diffusion-pipe")
            progress("install deepspeed")

        return runner

    monkeypatch.setattr(app_mod, "_build_bootstrap_runner", builder)

    r = client.post(
        "/api/backend/bootstrap",
        json={"backend": "diffusion-pipe"},
    )
    assert r.status_code == 202, r.text
    assert r.json()["backend"] == "diffusion-pipe"
    assert captured["backend"] == "diffusion-pipe"

    deadline = time.time() + 5
    final: dict[str, Any] | None = None
    while time.time() < deadline:
        body = client.get("/api/backend/bootstrap/status").json()
        if body["status"] in ("succeeded", "failed"):
            final = body
            break
        time.sleep(0.05)

    assert final is not None and final["status"] == "succeeded", final
    assert final["backend"] == "diffusion-pipe"
    levels = [e["level"] for e in final["events"]]
    assert levels[-1] == "done"
    # The completion message names the chosen backend.
    assert "diffusion-pipe" in final["events"][-1]["message"]


def test_config_with_diffusion_pipe_validates(client: TestClient, tmp_path: Path) -> None:
    """A recipe using backend.type='diffusion-pipe' must validate cleanly."""
    cfg_dict = _valid_config_dict(tmp_path)
    cfg_dict["backend"] = {"type": "diffusion-pipe"}

    r = client.post("/api/configs/validate", json={"config": cfg_dict})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["valid"] is True
    assert body["normalized"]["backend"]["type"] == "diffusion-pipe"


def test_diffusion_pipe_launch_writes_toml_and_starts_subprocess(tmp_path: Path) -> None:
    """launch() compiles the recipe to TOML and spawns train.py."""
    import sys
    import textwrap

    if sys.platform == "win32":
        pytest.skip("shell-script stubs only work on POSIX")

    from lorahub.core.backends.diffusion_pipe.backend import DiffusionPipeBackend
    from lorahub.core.config.schema import TrainingConfig

    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()

    repo = tmp_path / "dp"
    repo.mkdir()
    # train.py replacement that exits cleanly so the runner sees `done`.
    (repo / "train.py").write_text(
        "import sys\nprint('loaded'); sys.exit(0)\n", encoding="utf-8"
    )

    # Stub `<venv>/bin/{python,deepspeed}` next to repo so the runner has a
    # `deepspeed` launcher to call (it now uses that instead of plain python).
    bindir = repo / "venv" / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    stub = textwrap.dedent(
        f"""\
        #!/bin/sh
        exec "{sys.executable}" "$@"
        """
    )
    for name in ("python", "deepspeed"):
        p = bindir / name
        p.write_text(stub, encoding="utf-8")
        p.chmod(0o755)

    cfg = TrainingConfig.model_validate(
        {
            "base_model": {"arch": "sdxl", "checkpoint": str(ckpt)},
            "dataset": {"source": str(data)},
            "schedule": {"epochs": 1, "batch_size": 1},
            "sampling": {"enabled": False},
            "backend": {
                "type": "diffusion-pipe",
                "sd_scripts_path": str(repo),
                "python_executable": str(bindir / "python"),
            },
        }
    )

    backend = DiffusionPipeBackend()
    workspace = tmp_path / "ws"
    handle = backend.launch(cfg, workspace, on_event=lambda _ev: None)
    assert handle.pid is not None
    rc = handle.wait(timeout=30)
    assert rc == 0
    assert (workspace / "diffusion_pipe.toml").is_file()
    assert (workspace / "dataset.toml").is_file()


# --------------------------------------------------------------------------- #
# Network acceleration: github proxy + huggingface mirror + modelscope        #
# --------------------------------------------------------------------------- #


def test_apply_github_proxy_rewrites_only_github_urls() -> None:
    from lorahub.api.settings import apply_github_proxy

    proxy = "https://gh-proxy.org"
    assert apply_github_proxy(
        "https://github.com/foo/bar.git", proxy
    ) == "https://gh-proxy.org/https://github.com/foo/bar.git"
    # Trailing slash on the proxy is normalised.
    assert apply_github_proxy(
        "https://github.com/foo/bar.git", "https://gh-proxy.org/"
    ) == "https://gh-proxy.org/https://github.com/foo/bar.git"
    # Empty proxy 鈫?identity.
    assert (
        apply_github_proxy("https://github.com/foo/bar.git", None)
        == "https://github.com/foo/bar.git"
    )
    assert (
        apply_github_proxy("https://github.com/foo/bar.git", "  ")
        == "https://github.com/foo/bar.git"
    )
    # Non-github URLs untouched.
    assert (
        apply_github_proxy("https://gitlab.com/foo/bar.git", proxy)
        == "https://gitlab.com/foo/bar.git"
    )


def test_settings_persists_network_fields(client: TestClient) -> None:
    payload = {
        "github_proxy": "https://gh-proxy.org",
        "huggingface_endpoint": "https://hf-mirror.com",
        "modelscope_enabled": True,
        "modelscope_token": "secret",
    }
    r = client.put("/api/settings", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["settings"]["github_proxy"] == "https://gh-proxy.org"
    assert body["settings"]["huggingface_endpoint"] == "https://hf-mirror.com"
    assert body["settings"]["modelscope_enabled"] is True
    assert body["settings"]["modelscope_token"] == "secret"

    # GET round-trips them.
    r2 = client.get("/api/settings")
    assert r2.json()["settings"]["github_proxy"] == "https://gh-proxy.org"


def test_settings_persists_max_concurrent_jobs(client: TestClient) -> None:
    """`max_concurrent_jobs` round-trips through PUT/GET and rejects bad values."""
    r = client.put("/api/settings", json={"max_concurrent_jobs": 4})
    assert r.status_code == 200, r.text
    assert r.json()["settings"]["max_concurrent_jobs"] == 4

    # GET round-trips it.
    assert client.get("/api/settings").json()["settings"]["max_concurrent_jobs"] == 4

    # 0 / negative is rejected.
    r_bad = client.put("/api/settings", json={"max_concurrent_jobs": 0})
    assert r_bad.status_code == 422
    assert "max_concurrent_jobs" in r_bad.json()["detail"]


def test_env_overrides_injects_hf_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from lorahub.api.settings import Settings, env_overrides

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_ENDPOINT", raising=False)
    s = Settings(huggingface_endpoint="https://hf-mirror.com")
    out = env_overrides(s)
    assert out["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert out["HUGGINGFACE_HUB_ENDPOINT"] == "https://hf-mirror.com"

    # When the user already exported HF_ENDPOINT we don't overwrite it.
    monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")
    assert "HF_ENDPOINT" not in env_overrides(s)


def test_models_download_rejects_bad_repo_id(client: TestClient) -> None:
    r = client.post(
        "/api/models/download",
        json={"source": "huggingface", "repo_id": "no-slash"},
    )
    assert r.status_code == 400
    assert "owner/name" in r.json()["detail"]


def test_models_download_starts_session_and_reports_progress(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from lorahub.api.routers import models as models_router
    from lorahub.core.models.downloader import DownloadProgress, DownloadResult

    seen_threads: list[int] = []

    def fake_download(req: Any, progress: Any = None) -> DownloadResult:
        seen_threads.append(req.threads)
        target = req.target_dir or tmp_path / "model"
        target.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(DownloadProgress(message="listed files", percent=10, files_done=0, files_total=2))
            progress(
                DownloadProgress(
                    message="downloaded weights",
                    percent=55,
                    files_done=1,
                    files_total=2,
                    bytes_done=4,
                    bytes_total=8,
                )
            )
        (target / "weights.bin").write_bytes(b"weights")
        return DownloadResult(target=target, files=1, total_bytes=7)

    monkeypatch.setattr(models_router, "download", fake_download)
    r = client.post(
        "/api/models/download",
        json={
            "source": "modelscope",
            "repo_id": "owner/name",
            "target_dir": str(tmp_path / "downloaded"),
            "threads": 3,
        },
    )

    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "running"
    assert body["session_id"]

    status = {}
    deadline = time.time() + 3
    while time.time() < deadline:
        status = client.get(f"/api/models/download/{body['session_id']}").json()
        if status["status"] == "succeeded":
            break
        time.sleep(0.01)

    assert status["status"] == "succeeded"
    assert status["percent"] == 100
    assert status["result"]["files"] == 1
    assert status["events"][-1]["message"].startswith("download complete")
    assert seen_threads == [3]


def test_models_download_status_unknown_session_returns_404(client: TestClient) -> None:
    r = client.get("/api/models/download/missing")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# WD14 auto-tagging
# --------------------------------------------------------------------------- #


def test_tagging_rejects_missing_directory(client: TestClient, tmp_path: Path) -> None:
    r = client.post(
        "/api/tagging/tag",
        json={"path": str(tmp_path / "nope")},
    )
    assert r.status_code == 400
    assert "not a directory" in r.json()["detail"]


def test_tagging_runs_session_with_progress_and_writes_captions(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mock the tagger so the route's session/threading/progress plumbing is
    exercised end-to-end without touching the real ONNX model."""
    import time

    from lorahub.api.routers import tagging as tagging_router

    data = tmp_path / "dataset"
    data.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        (data / name).write_bytes(b"fake image bytes")

    captured: dict[str, Any] = {}

    class FakeTagger:
        def __init__(self) -> None:
            self.active_provider = "CPUExecutionProvider"

        def load(self) -> None:
            captured["loaded"] = True

        def tag_directory(
            self,
            directory: Path,
            *,
            recursive: bool,
            write_caption: bool,
            skip_existing: bool,
            underscores: bool,
            include_character: bool,
            on_progress: Any = None,
        ) -> list[Any]:
            captured["params"] = {
                "directory": str(directory),
                "recursive": recursive,
                "skip_existing": skip_existing,
                "underscores": underscores,
                "include_character": include_character,
            }
            from lorahub.core.tagging.wd14 import _iter_images  # noqa: PLC0415

            results: list[Any] = []
            for image in _iter_images(directory, recursive=recursive):
                if write_caption:
                    image.with_suffix(".txt").write_text("1girl, blue hair", encoding="utf-8")
                if on_progress is not None:
                    on_progress(image, object())
                results.append(object())
            return results

    monkeypatch.setattr(tagging_router, "_build_tagger", lambda _req: FakeTagger())

    r = client.post(
        "/api/tagging/tag",
        json={"path": str(data), "device": "cpu", "general": 0.5, "character": 0.9},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "running"
    sid = body["session_id"]

    deadline = time.time() + 5
    final: dict[str, Any] = {}
    while time.time() < deadline:
        final = client.get(f"/api/tagging/tag/{sid}").json()
        if final["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.02)

    assert final["status"] == "succeeded", final
    assert final["written"] == 3
    assert final["total"] == 3
    assert final["percent"] == 100
    assert final["active_provider"] == "CPUExecutionProvider"
    # Per-image events make it through (one per file plus framing messages).
    image_events = [e for e in final["events"] if e.get("image")]
    assert {Path(e["image"]).name for e in image_events} == {"a.png", "b.png", "c.png"}
    # The route translates `overwrite=False` into `skip_existing=True`.
    assert captured["params"]["skip_existing"] is True
    # Captions actually landed on disk.
    for name in ("a", "b", "c"):
        assert (data / f"{name}.txt").read_text(encoding="utf-8") == "1girl, blue hair"


def test_tagging_status_unknown_session_returns_404(client: TestClient) -> None:
    r = client.get("/api/tagging/tag/does-not-exist")
    assert r.status_code == 404


def test_tagging_dispatches_to_joytag_when_requested(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`tagger='joytag'` must reach `_build_tagger` and instantiate JoyTagger.

    The real `JoyTagger.load()` raises today, so we monkeypatch the class on
    the router module to a stub that records construction kwargs and runs
    cleanly to completion."""
    import time

    from lorahub.api.routers import tagging as tagging_router

    data = tmp_path / "ds"
    data.mkdir()
    (data / "a.png").write_bytes(b"fake")

    captured: dict[str, Any] = {}

    class FakeJoyTagger:
        def __init__(self, *, predict_threshold: float, device: str) -> None:
            captured["init"] = {"predict_threshold": predict_threshold, "device": device}
            self.active_provider = "cpu"

        def load(self) -> None:
            captured["loaded"] = True

        def tag_directory(self, directory: Path, **kwargs: Any) -> list[Any]:
            captured["tag_directory_kwargs"] = kwargs
            from lorahub.core.tagging.wd14 import _iter_images  # noqa: PLC0415

            results: list[Any] = []
            for image in _iter_images(directory, recursive=kwargs["recursive"]):
                if kwargs["write_caption"]:
                    image.with_suffix(".txt").write_text("1girl", encoding="utf-8")
                if kwargs.get("on_progress") is not None:
                    kwargs["on_progress"](image, object())
                results.append(object())
            return results

    monkeypatch.setattr(tagging_router, "JoyTagger", FakeJoyTagger)

    r = client.post(
        "/api/tagging/tag",
        json={
            "path": str(data),
            "tagger": "joytag",
            "joytag_threshold": 0.55,
            "device": "cpu",
        },
    )
    assert r.status_code == 202, r.text
    sid = r.json()["session_id"]

    deadline = time.time() + 5
    final: dict[str, Any] = {}
    while time.time() < deadline:
        final = client.get(f"/api/tagging/tag/{sid}").json()
        if final["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.02)

    assert final["status"] == "succeeded", final
    assert final["tagger"] == "joytag"
    assert final["active_provider"] == "cpu"
    assert captured["init"] == {"predict_threshold": 0.55, "device": "cpu"}
    assert captured["loaded"] is True
    assert captured["tag_directory_kwargs"]["skip_existing"] is True
    assert (data / "a.txt").read_text(encoding="utf-8") == "1girl"


def test_tagging_rejects_bad_tagger_value(client: TestClient, tmp_path: Path) -> None:
    data = tmp_path / "ds"
    data.mkdir()
    r = client.post(
        "/api/tagging/tag",
        json={"path": str(data), "tagger": "blip"},
    )
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# Caption preprocessing (/api/captions/normalize)
# --------------------------------------------------------------------------- #


def test_captions_normalize_rejects_missing_directory(
    client: TestClient, tmp_path: Path
) -> None:
    r = client.post(
        "/api/captions/normalize",
        json={"path": str(tmp_path / "nope")},
    )
    assert r.status_code == 400
    assert "not a directory" in r.json()["detail"]


def test_captions_normalize_runs_session_and_rewrites_files(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mock the pipeline so the route's session/threading/progress plumbing
    is exercised end-to-end without depending on the real CaptionPipeline."""
    import time

    from lorahub.api.routers import captions as captions_router

    data = tmp_path / "ds"
    data.mkdir()
    (data / "a.txt").write_text("Blue_Hair, NSFW", encoding="utf-8")
    (data / "b.txt").write_text("blue hair", encoding="utf-8")

    captured: dict[str, Any] = {}

    class FakePipeline:
        def __init__(self, *, blacklist: set[str], **_: Any) -> None:
            self.blacklist = blacklist

        def transform_directory(
            self,
            directory: Path,
            *,
            recursive: bool,
            overwrite: bool,
            progress: Any,
        ) -> int:
            captured["params"] = {
                "directory": str(directory),
                "recursive": recursive,
                "overwrite": overwrite,
                "blacklist": set(self.blacklist),
            }
            files = sorted(directory.glob("*.txt"))
            written = 0
            for idx, p in enumerate(files, start=1):
                old = p.read_text(encoding="utf-8")
                # Trivial fake: lowercase + drop "nsfw" tags.
                tags = [t.strip().lower() for t in old.split(",") if t.strip()]
                tags = [t.replace("_", " ") for t in tags if t not in self.blacklist]
                new = ", ".join(tags)
                if new != old:
                    p.write_text(new, encoding="utf-8")
                    written += 1
                if progress is not None:
                    progress(p, idx, len(files))
            return written

    monkeypatch.setattr(
        captions_router, "_build_pipeline", lambda req: FakePipeline(
            blacklist=set(req.blacklist),
        )
    )

    r = client.post(
        "/api/captions/normalize",
        json={
            "path": str(data),
            "blacklist": ["nsfw"],
            "recursive": False,
        },
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "running"
    sid = body["session_id"]

    deadline = time.time() + 5
    final: dict[str, Any] = {}
    while time.time() < deadline:
        final = client.get(f"/api/captions/normalize/{sid}").json()
        if final["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.02)

    assert final["status"] == "succeeded", final
    assert final["total"] == 2
    assert final["written"] == 2
    assert final["percent"] == 100
    file_events = [e for e in final["events"] if e.get("file")]
    assert {Path(e["file"]).name for e in file_events} == {"a.txt", "b.txt"}
    assert captured["params"]["blacklist"] == {"nsfw"}
    assert (data / "a.txt").read_text(encoding="utf-8") == "blue hair"


def test_captions_normalize_status_unknown_returns_404(client: TestClient) -> None:
    r = client.get("/api/captions/normalize/does-not-exist")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Cross-job sample gallery (/api/samples)
# --------------------------------------------------------------------------- #


def _png_bytes() -> bytes:
    """Return a tiny but valid-enough PNG header for tests."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def test_samples_aggregates_across_jobs(
    client: TestClient, tmp_path: Path
) -> None:
    """Two jobs each produce one sample image -> /api/samples returns both
    items with raw_url pointing at the existing per-job inline endpoint."""
    ws_a = tmp_path / "run-a"
    ws_a.mkdir()
    (ws_a / "sample-1.png").write_bytes(_png_bytes())
    job_a = state.registry.create(
        workspace=ws_a, config_snapshot={"output": {"name": "alpha"}}
    )

    ws_b = tmp_path / "run-b"
    ws_b.mkdir()
    out = ws_b / "out"
    out.mkdir()
    (out / "sample-2.jpg").write_bytes(_png_bytes())
    job_b = state.registry.create(
        workspace=ws_b, config_snapshot={"output": {"name": "beta"}}
    )

    r = client.get("/api/samples")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 2
    assert body["limit"] == 200
    assert body["offset"] == 0

    by_job = {item["job_id"]: item for item in body["items"]}
    assert set(by_job.keys()) == {job_a.id, job_b.id}

    item_a = by_job[job_a.id]
    assert item_a["path"] == "sample-1.png"
    assert item_a["config_name"] == "alpha"
    assert item_a["job_name"] == "run-a"
    assert item_a["raw_url"] == (
        f"/api/jobs/{job_a.id}/files/raw?path=sample-1.png"
    )

    item_b = by_job[job_b.id]
    assert item_b["path"] == "out/sample-2.jpg"
    # URL-encoded slash so the path stays a single query-string value.
    assert item_b["raw_url"] == (
        f"/api/jobs/{job_b.id}/files/raw?path=out%2Fsample-2.jpg"
    )

    # The raw_url should resolve to an actual byte stream from the existing
    # per-job endpoint 鈥?that is the reuse story we promised.
    raw = client.get(
        f"/api/jobs/{job_b.id}/files/raw", params={"path": "out/sample-2.jpg"}
    )
    assert raw.status_code == 200
    assert raw.content == _png_bytes()


def test_samples_filter_by_job_ids(client: TestClient, tmp_path: Path) -> None:
    ws_a = tmp_path / "run-a"
    ws_a.mkdir()
    (ws_a / "a.png").write_bytes(_png_bytes())
    job_a = state.registry.create(workspace=ws_a, config_snapshot={})

    ws_b = tmp_path / "run-b"
    ws_b.mkdir()
    (ws_b / "b.png").write_bytes(_png_bytes())
    state.registry.create(workspace=ws_b, config_snapshot={})

    r = client.get("/api/samples", params={"job_ids": job_a.id})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["job_id"] == job_a.id

    # Unknown job id -> 404 so the UI can flag stale filter chips.
    r_missing = client.get("/api/samples", params={"job_ids": "ghost-id"})
    assert r_missing.status_code == 404


def test_samples_empty_when_no_jobs(client: TestClient) -> None:
    r = client.get("/api/samples")
    assert r.status_code == 200
    assert r.json() == {"items": [], "total": 0, "limit": 200, "offset": 0}


def test_samples_sorted_newest_first(client: TestClient, tmp_path: Path) -> None:
    import os

    ws = tmp_path / "run"
    ws.mkdir()
    (ws / "old.png").write_bytes(_png_bytes())
    (ws / "new.png").write_bytes(_png_bytes())
    # Force an older mtime on old.png so sorting has something to chew on.
    old_ts = (ws / "new.png").stat().st_mtime - 60
    os.utime(ws / "old.png", (old_ts, old_ts))
    state.registry.create(workspace=ws, config_snapshot={})

    r = client.get("/api/samples")
    paths = [item["path"] for item in r.json()["items"]]
    assert paths == ["new.png", "old.png"]


# --------------------------------------------------------------------------- #
# Recipe template instantiate (POST /api/recipes/templates/{id}/instantiate)
# --------------------------------------------------------------------------- #


def test_instantiate_template_substitutes_placeholders(
    client: TestClient,
    configs_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Placeholders applied by dotted-path setter, recipe validates, and the
    persisted YAML carries the substituted values."""
    from lorahub.api import config_templates as config_templates_module

    builtin_dir = tmp_path / "builtin"
    builtin_dir.mkdir()
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()

    template_yaml = {
        "_template": {
            "name": "Test SDXL",
            "description": "x",
            "arch": "sdxl",
        },
        "_placeholders": [
            {
                "key": "checkpoint",
                "label": "ckpt",
                "path_field": "base_model.checkpoint",
                "placeholder": "ckpt.safetensors",
            },
            {
                "key": "dataset",
                "label": "ds",
                "path_field": "dataset.source",
                "placeholder": "./datasets/x",
            },
            {
                "key": "name",
                "label": "name",
                "path_field": "output.name",
                "placeholder": "out_v1",
            },
        ],
        "base_model": {"arch": "sdxl", "checkpoint": ""},
        "dataset": {"source": ""},
        "schedule": {"epochs": 3, "batch_size": 1},
        "sampling": {"enabled": False},
    }
    (builtin_dir / "test.yaml").write_text(
        yaml.safe_dump(template_yaml, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setattr(
        config_templates_module, "_DEFAULT_BUILTIN_DIR", builtin_dir
    )

    # Confirm the listing exposes the new placeholders array.
    listed = client.get("/api/configs/templates").json()["templates"]
    assert len(listed) == 1
    assert [p["key"] for p in listed[0]["placeholders"]] == [
        "checkpoint",
        "dataset",
        "name",
    ]

    r = client.post(
        "/api/configs/templates/test/instantiate",
        json={
            "name": "myrun",
            "values": {
                "checkpoint": str(ckpt),
                "dataset": str(data),
                "name": "myrun_out",
            },
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "myrun"
    assert body["template_id"] == "test"

    saved = configs_dir / "myrun.yaml"
    assert saved.is_file()
    parsed = yaml.safe_load(saved.read_text(encoding="utf-8"))
    assert parsed["base_model"]["checkpoint"] == str(ckpt)
    assert parsed["dataset"]["source"] == str(data)
    assert parsed["output"]["name"] == "myrun_out"
    # Sanity: the schedule values from the template survived.
    assert parsed["schedule"]["epochs"] == 3
    # The metadata blocks are stripped from what gets persisted.
    assert "_template" not in parsed
    assert "_placeholders" not in parsed


def test_instantiate_template_unknown_id_returns_404(
    client: TestClient, configs_dir: Path
) -> None:
    r = client.post(
        "/api/configs/templates/does-not-exist/instantiate",
        json={"name": "anything", "values": {}},
    )
    assert r.status_code == 404


def test_instantiate_template_conflict_without_overwrite(
    client: TestClient,
    configs_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lorahub.api import config_templates as config_templates_module

    builtin_dir = tmp_path / "builtin"
    builtin_dir.mkdir()
    (builtin_dir / "skel.yaml").write_text(
        yaml.safe_dump(
            {
                "_template": {"name": "Skel", "arch": "sdxl"},
                "base_model": {"arch": "sdxl", "checkpoint": ""},
                "dataset": {"source": ""},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        config_templates_module, "_DEFAULT_BUILTIN_DIR", builtin_dir
    )

    first = client.post(
        "/api/configs/templates/skel/instantiate",
        json={"name": "dup", "values": {}},
    )
    assert first.status_code == 201, first.text

    clash = client.post(
        "/api/configs/templates/skel/instantiate",
        json={"name": "dup", "values": {}},
    )
    assert clash.status_code == 409

    overwrite = client.post(
        "/api/configs/templates/skel/instantiate",
        json={"name": "dup", "values": {}, "overwrite": True},
    )
    assert overwrite.status_code == 201, overwrite.text


def test_apply_placeholders_creates_intermediate_dicts() -> None:
    """The dotted-path setter is the load-bearing piece 鈥?make sure it
    creates missing intermediates and rejects non-mapping traversal."""
    from lorahub.api.config_templates import apply_placeholders

    placeholders = [
        {
            "key": "name",
            "label": "name",
            "path_field": "output.name",
            "placeholder": "",
        }
    ]
    out = apply_placeholders({}, placeholders, {"name": "v1"})
    assert out == {"output": {"name": "v1"}}

    # Empty values are no-ops so callers can submit a half-filled form.
    untouched = apply_placeholders(
        {"output": {"name": "old"}}, placeholders, {"name": ""}
    )
    assert untouched == {"output": {"name": "old"}}


# --------------------------------------------------------------------------- #
# /api/sweeps
# --------------------------------------------------------------------------- #


def test_create_sweep_enqueues_one_job_per_variant(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /api/sweeps must call _launch_job exactly once per grid variant.

    The real launcher would spawn kohya subprocesses, so we patch it on the
    sweep router to record metadata and return a fake job id without ever
    touching the scheduler. With three values across one axis we expect
    three calls; the response variants must echo the same job ids.
    """
    from lorahub.api import state as state_mod
    from lorahub.api.routers import sweeps as sweeps_router

    captured: list[dict[str, Any]] = []

    def fake_launch(cfg: TrainingConfig, workspace: Path, *, metadata: dict[str, Any]) -> dict[str, Any]:
        # Mirror what _launch_job does in production: register a JobRecord so
        # the GET endpoint has something to aggregate, then stamp metadata.
        record = state_mod.registry.create(
            workspace=workspace, config_snapshot=cfg.model_dump(mode="json")
        )
        record.metadata = metadata
        state_mod.registry.update(record)
        captured.append({"workspace": workspace, "metadata": metadata})
        return record.to_summary()

    monkeypatch.setattr(sweeps_router, "_launch_job", fake_launch)

    payload = {
        "base_config": _config_payload(tmp_path) | {"network": {"rank": 32, "alpha": 16}},
        "axes": [{"path": "network.rank", "values": [16, 32, 64]}],
        "workspace_root": str(tmp_path / "runs"),
    }
    r = client.post("/api/sweeps", json=payload)
    assert r.status_code == 202, r.text
    body = r.json()

    assert len(captured) == 3
    assert len(body["variants"]) == 3
    assert len(body["job_ids"]) == 3
    # Every variant carries the same sweep_id and a 1-based axis_values entry.
    sweep_ids = {c["metadata"]["sweep_id"] for c in captured}
    assert sweep_ids == {body["sweep_id"]}
    rank_values = [c["metadata"]["axis_values"]["network.rank"] for c in captured]
    assert rank_values == [16, 32, 64]


def test_get_sweep_aggregates_job_states(
    client: TestClient, tmp_path: Path
) -> None:
    """Three fake jobs in distinct states must aggregate into the right counters."""
    sweep_id = "01TESTSWEEPID0000000000000"
    states_by_index: list[state.JobState] = [
        state.JobState.queued,
        state.JobState.running,
        state.JobState.succeeded,
    ]
    for i, st in enumerate(states_by_index, start=1):
        ws = tmp_path / f"variant-{i}"
        ws.mkdir()
        rec = state.registry.create(workspace=ws, config_snapshot={})
        rec.state = st
        rec.metadata = {"sweep_id": sweep_id, "variant_name": f"v-{i:03d}"}
        state.registry.update(rec)

    # Drop a sibling job under a different sweep_id to confirm filtering works.
    other_ws = tmp_path / "other"
    other_ws.mkdir()
    other = state.registry.create(workspace=other_ws, config_snapshot={})
    other.metadata = {"sweep_id": "another-sweep"}
    state.registry.update(other)

    r = client.get(f"/api/sweeps/{sweep_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sweep_id"] == sweep_id
    assert body["total"] == 3
    assert body["queued"] == 1
    assert body["running"] == 1
    assert body["succeeded"] == 1
    assert body["failed"] == 0
    assert len(body["jobs"]) == 3
    # Sibling sweep is filtered out.
    assert all(j["metadata"]["sweep_id"] == sweep_id for j in body["jobs"])


def test_sweep_metadata_persists_across_restart(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job tagged with sweep metadata must remain grouped after rehydration.

    Simulates the lifespan path: write a JobRecord through a backed registry,
    drop the in-memory registry on the floor, and re-open the same SQLite
    file. The sweep aggregation endpoint must still find the job.
    """
    from lorahub.api.store import JobStore

    sweep_id = "01TESTRESTART000000000000"
    db_path = tmp_path / "restart.sqlite"

    # First "process": register a job under a backed registry and stamp metadata.
    store_a = JobStore(db_path)
    reg_a = state.JobRegistry(store=store_a)
    monkeypatch.setattr(state, "registry", reg_a)
    ws = tmp_path / "variant-1"
    ws.mkdir()
    job = reg_a.create(workspace=ws, config_snapshot={"x": 1})
    job.state = state.JobState.succeeded
    job.metadata = {"sweep_id": sweep_id, "axis_values": {"network.rank": 16}}
    reg_a.update(job)

    # Second "process": fresh store + registry pointing at the same SQLite
    # file, just like `_lifespan` does on server startup.
    store_b = JobStore(db_path)
    reg_b = state.JobRegistry(store=store_b)
    loaded = reg_b.load_persisted()
    assert loaded == 1
    monkeypatch.setattr(state, "registry", reg_b)

    r = client.get(f"/api/sweeps/{sweep_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sweep_id"] == sweep_id
    assert body["total"] == 1
    assert body["succeeded"] == 1
    assert body["jobs"][0]["metadata"] == {
        "sweep_id": sweep_id,
        "axis_values": {"network.rank": 16},
    }
def test_list_sweeps_aggregates(client: TestClient, tmp_path: Path) -> None:
    """GET /api/sweeps groups every metadata-tagged job by ``sweep_id``.

    We register two sibling sweeps with distinct prefixes plus an untagged
    job (which must be ignored). The endpoint is expected to bubble up a
    rolled-up count per sweep and a ``name_prefix`` derived from the common
    head of every variant's ``output.name``.
    """
    # Sweep A 鈥?three variants in mixed states; the shared name prefix is
    # ``alpha-`` (the trailing dash is stripped by ``_common_prefix``).
    sweep_a = "01SWEEPALPHA00000000000000"
    a_states: list[state.JobState] = [
        state.JobState.queued,
        state.JobState.running,
        state.JobState.succeeded,
    ]
    for i, st in enumerate(a_states, start=1):
        ws = tmp_path / f"alpha-{i:03d}"
        ws.mkdir()
        rec = state.registry.create(
            workspace=ws,
            config_snapshot={"output": {"name": f"alpha-{i:03d}"}},
        )
        rec.state = st
        rec.metadata = {
            "sweep_id": sweep_a,
            "variant_name": f"alpha-{i:03d}",
            "axis_values": {"network.rank": [16, 32, 64][i - 1]},
        }
        state.registry.update(rec)

    # Sweep B 鈥?two failed variants under a different prefix.
    sweep_b = "01SWEEPBRAVO00000000000000"
    for i in range(1, 3):
        ws = tmp_path / f"bravo-{i:03d}"
        ws.mkdir()
        rec = state.registry.create(
            workspace=ws,
            config_snapshot={"output": {"name": f"bravo-{i:03d}"}},
        )
        rec.state = state.JobState.failed
        rec.metadata = {"sweep_id": sweep_b}
        state.registry.update(rec)

    # Stray job without a sweep tag 鈥?must not appear in the response.
    stray = tmp_path / "stray"
    stray.mkdir()
    state.registry.create(workspace=stray, config_snapshot={})

    r = client.get("/api/sweeps")
    assert r.status_code == 200, r.text
    sweeps = r.json()["sweeps"]
    assert len(sweeps) == 2

    by_id = {s["sweep_id"]: s for s in sweeps}
    a = by_id[sweep_a]
    assert a["total"] == 3
    assert a["queued"] == 1
    assert a["running"] == 1
    assert a["succeeded"] == 1
    assert a["failed"] == 0
    assert a["name_prefix"] == "alpha"
    assert "earliest_created_at" in a
    assert "latest_modified_at" in a

    b = by_id[sweep_b]
    assert b["total"] == 2
    assert b["failed"] == 2
    assert b["name_prefix"] == "bravo"


# --------------------------------------------------------------------------- #
# Attention backend endpoints
# --------------------------------------------------------------------------- #


def test_attention_backends_endpoint_shape(client: TestClient) -> None:
    """`GET /api/system/attention-backends` returns a stable, typed shape."""
    r = client.get("/api/system/attention-backends")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body.keys()) == {"compute_capability", "supported", "all"}
    assert body["compute_capability"] is None or isinstance(body["compute_capability"], str)
    assert isinstance(body["supported"], list)
    assert isinstance(body["all"], list)
    # `all` is the canonical superset; `supported` must be a subset of it.
    assert set(body["supported"]).issubset(body["all"])
    # The PyTorch-native quartet is always available.
    assert {"auto", "torch", "sdpa", "flex"}.issubset(body["supported"])


def test_attention_backends_endpoint_uses_first_nvidia_gpu(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the host has an NVIDIA GPU, its compute_cap drives the supported set."""
    from lorahub.api import system_stats

    fake = system_stats.SystemSnapshot(
        timestamp=0.0,
        host=system_stats.HostInfo(hostname="x", system="X", release="x", python="x"),
        cpu=system_stats.CpuStats(cores_logical=1, cores_physical=1, usage_percent=0.0),
        memory=system_stats.MemoryStats(total_bytes=1, used_bytes=0, available_bytes=1, percent=0.0),
        disks=[],
        gpus=[
            system_stats.GpuStats(
                index=0,
                name="H100",
                driver="555",
                memory_total_bytes=None,
                memory_used_bytes=None,
                memory_free_bytes=None,
                utilization_percent=None,
                temperature_c=None,
                power_w=None,
                power_limit_w=None,
                fan_percent=None,
                vendor="nvidia",
                compute_capability="9.0",
            )
        ],
        has_psutil=False,
        has_nvidia_smi=True,
    )
    # Patch the symbol the router imports rather than the module under test;
    # FastAPI bound it at import time.
    from lorahub.api.routers import system as system_router

    monkeypatch.setattr(system_router, "collect_snapshot", lambda: fake)

    body = client.get("/api/system/attention-backends").json()
    assert body["compute_capability"] == "9.0"
    assert "flash3" in body["supported"]
    assert "flash4" in body["supported"]


def test_install_flash_attn_returns_501(client: TestClient) -> None:
    """The conservative path: refuse the auto-install with a doc URL."""
    r = client.post(
        "/api/backend/install-flash-attn",
        json={"backend": "kohya", "version": "3"},
    )
    assert r.status_code == 501
    detail = r.json()["detail"]
    assert detail["backend"] == "kohya"
    assert detail["version"] == "3"
    assert "install_doc_url" in detail
    assert detail["install_doc_url"].startswith("https://")


def test_install_flash_attn_validates_version(client: TestClient) -> None:
    """Body shape is validated by pydantic before the 501 fires."""
    r = client.post(
        "/api/backend/install-flash-attn",
        json={"backend": "kohya", "version": "5"},  # invalid literal
    )
    assert r.status_code == 422

