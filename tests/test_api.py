"""Tests for the LoraHub HTTP API."""

from __future__ import annotations

import textwrap
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lorahub.api import state
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
    fresh = state.JobRegistry()
    monkeypatch.setattr(state, "registry", fresh)
    yield fresh


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
