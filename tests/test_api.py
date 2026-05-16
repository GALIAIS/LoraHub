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
from lorahub.core.config.schema import RecipeConfig
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
    (root / "train_network.py").write_text(stub, encoding="utf-8")
    (root / "sdxl_train_network.py").write_text(stub, encoding="utf-8")
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
    # user-data file. Patch on the imported `app` module — that's the symbol
    # the request handlers resolve at call time.
    monkeypatch.setattr(
        app_mod, "_settings_store", SettingsStore(tmp_path / "settings.json")
    )
    # Don't let a developer's .env (LORAHUB_KOHYA_*) leak into backend probes —
    # those env vars are valid in production but confuse settings tests.
    monkeypatch.delenv("LORAHUB_KOHYA_SD_SCRIPTS", raising=False)
    monkeypatch.delenv("LORAHUB_KOHYA_PYTHON", raising=False)
    # Reset the singleton bootstrap session so tests can't leak state into
    # one another (each test starts from "idle").
    monkeypatch.setattr(app_mod, "_bootstrap_session", None)
    return TestClient(app_mod.app)


def _recipe_payload(tmp_path: Path) -> dict[str, Any]:
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


def test_recipe_schema_is_valid_json_schema(client: TestClient) -> None:
    r = client.get("/api/recipes/schema")
    assert r.status_code == 200
    schema = r.json()
    assert schema["title"] == "RecipeConfig"
    assert "base_model" in schema["$defs"] or "base_model" in str(schema)


def test_list_jobs_starts_empty(client: TestClient) -> None:
    r = client.get("/api/jobs")
    assert r.status_code == 200
    assert r.json() == {"jobs": []}


def test_get_unknown_job_returns_404(client: TestClient) -> None:
    r = client.get("/api/jobs/does-not-exist")
    assert r.status_code == 404


def test_create_and_complete_job(client: TestClient, tmp_path: Path) -> None:
    payload = {"recipe": _recipe_payload(tmp_path), "workspace": str(tmp_path / "ws")}
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
    payload = {"recipe": _recipe_payload(tmp_path), "workspace": str(tmp_path / "ws")}
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
    job = state.registry.create(workspace=ws, recipe_snapshot={})
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
    job = state.registry.create(workspace=ws, recipe_snapshot={})
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
    r = client.post("/api/jobs", json={"recipe": {"missing": "everything"}})
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
    job = state.registry.create(workspace=ws, recipe_snapshot={})
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
    job = state.registry.create(workspace=ws, recipe_snapshot={})
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
    job = state.registry.create(workspace=ws, recipe_snapshot={})
    job.state = state.JobState.interrupted
    state.registry.update(job)

    r = client.post(f"/api/jobs/{job.id}/resume")
    assert r.status_code == 409
    assert "safetensors" in r.json()["detail"]


def test_cancel_queued_job_short_circuits_to_canceled(
    client: TestClient, tmp_path: Path
) -> None:
    """A job pending on the worker deque must cancel without launching."""
    ws = tmp_path / "ws"
    ws.mkdir()
    job = state.registry.create(workspace=ws, recipe_snapshot={})
    # Default state is 'queued'.
    assert job.state is state.JobState.queued

    r = client.delete(f"/api/jobs/{job.id}")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "canceled"
    assert body["finished_at"] is not None


# --------------------------------------------------------------------------- #
# Recipe template browsing
# --------------------------------------------------------------------------- #


@pytest.fixture
def recipes_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the API at an isolated recipes directory."""
    rdir = tmp_path / "recipes"
    rdir.mkdir()
    monkeypatch.setenv("LORAHUB_RECIPES_DIR", str(rdir))
    return rdir


def _write_valid_recipe(rdir: Path, name: str = "demo") -> Path:
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


def test_list_recipes_returns_valid_and_invalid(
    client: TestClient, recipes_dir: Path
) -> None:
    _write_valid_recipe(recipes_dir, "good")
    (recipes_dir / "broken.yaml").write_text("base_model: {}\n", encoding="utf-8")
    (recipes_dir / "ignore-me.txt").write_text("not yaml", encoding="utf-8")

    r = client.get("/api/recipes")
    assert r.status_code == 200
    body = r.json()
    names = {it["name"] for it in body["recipes"]}
    assert names == {"good", "broken"}

    good = next(it for it in body["recipes"] if it["name"] == "good")
    assert good["valid"] is True
    assert good["arch"] == "sdxl"
    assert "epoch" in good["summary"]

    broken = next(it for it in body["recipes"] if it["name"] == "broken")
    assert broken["valid"] is False
    assert broken["error"]


def test_get_recipe_returns_content_and_parsed(
    client: TestClient, recipes_dir: Path
) -> None:
    _write_valid_recipe(recipes_dir, "good")
    r = client.get("/api/recipes/good")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "good"
    assert "base_model:" in body["content"]
    assert body["parsed"]["base_model"]["arch"] == "sdxl"
    assert body["error"] is None


def test_get_recipe_missing_returns_404(
    client: TestClient, recipes_dir: Path
) -> None:
    r = client.get("/api/recipes/nope")
    assert r.status_code == 404


def test_get_recipe_blocks_path_traversal(
    client: TestClient, recipes_dir: Path
) -> None:
    r = client.get("/api/recipes/..%2Fpasswd")
    # FastAPI normalizes %2F into /, our handler rejects bare names with slashes
    assert r.status_code in (400, 404)


def test_recipe_schema_still_resolves_under_recipes_prefix(
    client: TestClient, recipes_dir: Path
) -> None:
    # /recipes/schema must keep working alongside /recipes/{name}
    r = client.get("/api/recipes/schema")
    assert r.status_code == 200
    assert r.json()["title"] == "RecipeConfig"


# --------------------------------------------------------------------------- #
# Recipe validate + save
# --------------------------------------------------------------------------- #


def _valid_recipe_dict(tmp_path: Path) -> dict[str, Any]:
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


def test_validate_recipe_returns_normalized_payload(
    client: TestClient, tmp_path: Path
) -> None:
    r = client.post("/api/recipes/validate", json={"recipe": _valid_recipe_dict(tmp_path)})
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


def test_validate_recipe_reports_dataset_caption_preflight(
    client: TestClient, tmp_path: Path
) -> None:
    recipe = _valid_recipe_dict(tmp_path)
    data = Path(str(recipe["dataset"]["source"]))
    (data / "sample.png").write_bytes(b"fake image bytes")

    r = client.post("/api/recipes/validate", json={"recipe": recipe})

    assert r.status_code == 200
    paths = r.json()["preflight"]["paths"]
    assert paths["image_files"] == 1
    assert paths["caption_files"] == 0
    assert paths["missing_caption_files"] == ["sample.png"]


def test_validate_recipe_returns_structured_errors(client: TestClient) -> None:
    r = client.post("/api/recipes/validate", json={"recipe": {}})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is False
    assert len(body["errors"]) >= 1
    # each error should include a loc list
    assert all("loc" in e for e in body["errors"])


def test_save_recipe_writes_file_and_blocks_overwrite(
    client: TestClient, tmp_path: Path, recipes_dir: Path
) -> None:
    payload = {"name": "demo", "recipe": _valid_recipe_dict(tmp_path)}
    r = client.post("/api/recipes", json=payload)
    assert r.status_code == 201, r.text
    saved = r.json()
    assert saved["filename"] == "demo.yaml"
    assert (recipes_dir / "demo.yaml").is_file()

    # Repeat without overwrite — should 409
    r2 = client.post("/api/recipes", json=payload)
    assert r2.status_code == 409

    # With overwrite — should 201
    r3 = client.post("/api/recipes", json={**payload, "overwrite": True})
    assert r3.status_code == 201


def test_save_recipe_rejects_invalid_name(
    client: TestClient, tmp_path: Path, recipes_dir: Path
) -> None:
    r = client.post(
        "/api/recipes",
        json={"name": "../etc/passwd", "recipe": _valid_recipe_dict(tmp_path)},
    )
    assert r.status_code == 400


def test_save_recipe_rejects_invalid_recipe(
    client: TestClient, recipes_dir: Path
) -> None:
    r = client.post("/api/recipes", json={"name": "bad", "recipe": {}})
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
    """Pillow can't decode a `.png` of arbitrary bytes — we surface 404."""
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
    """A missing companion is `caption=null`, not an error — there's nothing
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


def test_rerun_creates_new_job(client: TestClient, tmp_path: Path) -> None:
    payload = {"recipe": _recipe_payload(tmp_path), "workspace": str(tmp_path / "ws")}
    first = client.post("/api/jobs", json=payload).json()
    first_id = first["id"]
    final_first = _wait_terminal(client, first_id)
    assert final_first["state"] == "succeeded", final_first

    r = client.post(f"/api/jobs/{first_id}/rerun")
    assert r.status_code == 202, r.text
    fresh = r.json()
    assert fresh["id"] != first_id
    # The fresh job must land in its own workspace so the two runs don't fight
    # over the same `events.jsonl`.
    assert fresh["workspace"] != final_first["workspace"]

    final_fresh = _wait_terminal(client, fresh["id"])
    assert final_fresh["state"] == "succeeded", final_fresh
    assert final_fresh["returncode"] == 0


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
    job = state.registry.create(workspace=ws, recipe_snapshot={})
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
    # Never use shell=True — argv form only.
    assert "shell" not in captured["kwargs"] or captured["kwargs"]["shell"] is False


def test_reveal_returns_409_when_workspace_missing(
    client: TestClient, tmp_path: Path
) -> None:
    job = state.registry.create(
        workspace=tmp_path / "gone", recipe_snapshot={}
    )
    job.state = state.JobState.succeeded
    state.registry.update(job)

    r = client.post(f"/api/jobs/{job.id}/reveal")
    assert r.status_code == 409


def test_archive_completed_job_moves_workspace(
    client: TestClient, tmp_path: Path
) -> None:
    payload = {"recipe": _recipe_payload(tmp_path), "workspace": str(tmp_path / "ws")}
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


def test_archive_running_job_returns_409(client: TestClient, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    job = state.registry.create(workspace=ws, recipe_snapshot={})
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
            # second POST below — this is the whole point of the test.
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
    # stubs the install may already be done — both running and succeeded are OK.
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
    # Memory shape — fall back to 0s rather than missing keys.
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
    job = state.registry.create(workspace=ws, recipe_snapshot={})
    return job.id


def test_job_files_lists_workspace_artifacts(
    client: TestClient, tmp_path: Path
) -> None:
    ws = tmp_path / "run-1"
    ws.mkdir()
    (ws / "model.safetensors").write_bytes(b"weights")
    (ws / "recipe.yaml").write_text("name: test\n", encoding="utf-8")
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
    assert "recipe.yaml" in other
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
        # Garbage line — must not break parsing of the rest.
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


# --------------------------------------------------------------------------- #
# Recipe duplicate / rename / delete / templates / import
# --------------------------------------------------------------------------- #


def test_duplicate_recipe_creates_copy(
    client: TestClient, recipes_dir: Path
) -> None:
    src = _write_valid_recipe(recipes_dir, "demo")

    r = client.post("/api/recipes/demo/duplicate", json={"new_name": "demo_v2"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "demo_v2"
    assert body["filename"] == "demo_v2.yaml"
    copy = recipes_dir / "demo_v2.yaml"
    assert copy.is_file()
    assert copy.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")

    # Source missing -> 404
    r_missing = client.post(
        "/api/recipes/nope/duplicate", json={"new_name": "ghost"}
    )
    assert r_missing.status_code == 404

    # Destination already exists -> 409
    r_clash = client.post(
        "/api/recipes/demo/duplicate", json={"new_name": "demo_v2"}
    )
    assert r_clash.status_code == 409

    # Bad new_name -> 400
    r_bad = client.post(
        "/api/recipes/demo/duplicate", json={"new_name": "../etc/passwd"}
    )
    assert r_bad.status_code == 400


def test_rename_recipe(client: TestClient, recipes_dir: Path) -> None:
    _write_valid_recipe(recipes_dir, "demo")
    _write_valid_recipe(recipes_dir, "other")

    r = client.post("/api/recipes/demo/rename", json={"new_name": "demo_renamed"})
    assert r.status_code == 200, r.text
    assert not (recipes_dir / "demo.yaml").exists()
    assert (recipes_dir / "demo_renamed.yaml").is_file()

    # Renaming to a name that's already taken -> 409
    r_clash = client.post(
        "/api/recipes/demo_renamed/rename", json={"new_name": "other"}
    )
    assert r_clash.status_code == 409

    # Renaming a missing recipe -> 404
    r_missing = client.post(
        "/api/recipes/ghost/rename", json={"new_name": "demo_v3"}
    )
    assert r_missing.status_code == 404


def test_delete_recipe(client: TestClient, recipes_dir: Path) -> None:
    _write_valid_recipe(recipes_dir, "demo")

    r = client.delete("/api/recipes/demo")
    assert r.status_code == 200, r.text
    assert r.json() == {"deleted": True, "name": "demo"}

    # Now it's gone
    assert client.get("/api/recipes/demo").status_code == 404
    # Re-deleting a missing recipe -> 404
    assert client.delete("/api/recipes/demo").status_code == 404


def test_list_templates_returns_validated_recipes(client: TestClient) -> None:
    r = client.get("/api/recipes/templates")
    assert r.status_code == 200, r.text
    body = r.json()

    ids = {t["id"] for t in body["templates"]}
    assert ids == {"sdxl_character", "sdxl_style", "sd15_character", "blank"}

    # Each template recipe must round-trip through the schema.
    for tpl in body["templates"]:
        cfg = RecipeConfig.model_validate(tpl["recipe"])
        assert cfg.base_model.arch in {"sdxl", "sd15", "flux", "sd3"}


def test_import_recipe_from_yaml(
    client: TestClient, tmp_path: Path, recipes_dir: Path
) -> None:
    recipe_dict = _valid_recipe_dict(tmp_path)
    yaml_bytes = yaml.safe_dump(recipe_dict, sort_keys=False).encode("utf-8")

    r = client.post(
        "/api/recipes/import",
        files={"file": ("foo.yaml", yaml_bytes, "application/x-yaml")},
        data={"name": "imported"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "imported"
    assert body["filename"] == "imported.yaml"
    saved = recipes_dir / "imported.yaml"
    assert saved.is_file()
    # The persisted file is canonical YAML emitted by dump_recipe; just confirm
    # it loads back to an equivalent RecipeConfig.
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


def test_recipe_with_diffusion_pipe_validates(client: TestClient, tmp_path: Path) -> None:
    """A recipe using backend.type='diffusion-pipe' must validate cleanly."""
    recipe = _valid_recipe_dict(tmp_path)
    recipe["backend"] = {"type": "diffusion-pipe"}

    r = client.post("/api/recipes/validate", json={"recipe": recipe})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["valid"] is True
    assert body["normalized"]["backend"]["type"] == "diffusion-pipe"


def test_diffusion_pipe_launch_writes_toml_and_starts_subprocess(tmp_path: Path) -> None:
    """launch() compiles the recipe to TOML and spawns train.py."""
    import sys

    from lorahub.core.backends.diffusion_pipe.backend import DiffusionPipeBackend
    from lorahub.core.config.schema import RecipeConfig

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

    cfg = RecipeConfig.model_validate(
        {
            "base_model": {"arch": "sdxl", "checkpoint": str(ckpt)},
            "dataset": {"source": str(data)},
            "schedule": {"epochs": 1, "batch_size": 1},
            "sampling": {"enabled": False},
            "backend": {
                "type": "diffusion-pipe",
                "sd_scripts_path": str(repo),
                "python_executable": sys.executable,
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
    # Empty proxy → identity.
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
