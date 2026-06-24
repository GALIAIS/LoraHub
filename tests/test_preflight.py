"""Pre-flight blocker tests for POST /api/jobs and /api/jobs/{id}/resume.

Each test sets up a *minimally-valid* config via ``_config_payload`` and
then deliberately corrupts one field. We want a 422 with a structured
``findings`` list — *not* a 202 + a job that crashes 5 seconds later in
the trainer subprocess.
"""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lorahub.api import state
from lorahub.api.preflight import PreflightFinding, run_preflight
from lorahub.core.config.schema import TrainingConfig


# -- Reuse the same fixtures the existing /api tests use. They live in
# tests/test_api.py so we duplicate the bare minimum here rather than
# import private helpers (cross-test imports break in CI when only one
# file is selected).
def _stub_sd_scripts(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    body = "import sys; sys.exit(0)\n"
    for name in (
        "train_network.py",
        "sdxl_train_network.py",
        "flux_train_network.py",
    ):
        (root / name).write_text(body, encoding="utf-8")
    return root


def _valid_payload(tmp_path: Path) -> dict[str, Any]:
    sd = _stub_sd_scripts(tmp_path / "sd-scripts")
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    (data / "0.png").write_bytes(b"")
    return {
        "base_model": {"checkpoint": str(ckpt)},
        "dataset": {"source": str(data)},
        "schedule": {"epochs": 1, "batch_size": 1},
        "sampling": {"enabled": False},
        "backend": {
            "sd_scripts_path": str(sd),
            "python_executable": sys.executable,
        },
    }


def _stub_anima_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.py").write_text("import sys; sys.exit(0)\n", encoding="utf-8")
    (root / "inference.py").write_text("import sys; sys.exit(0)\n", encoding="utf-8")
    (root / "library").mkdir()
    (root / "library" / "anima").mkdir()
    return root


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[state.JobRegistry]:
    from lorahub.api import scheduler as sched_module
    from lorahub.api import sweep_runtime

    fresh = state.JobRegistry()
    monkeypatch.setattr(state, "registry", fresh)
    fresh_sched = sched_module.JobScheduler(concurrency=1)
    monkeypatch.setattr(sched_module, "scheduler", fresh_sched)
    fresh_sched.start()
    sweep_runtime.reset_for_tests()
    try:
        yield fresh
    finally:
        fresh_sched.stop(timeout=2.0)
        sweep_runtime.reset_for_tests()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from lorahub.api import app as app_mod
    from lorahub.api.settings import SettingsStore

    monkeypatch.setattr(
        app_mod, "_settings_store", SettingsStore(tmp_path / "settings.json")
    )
    monkeypatch.delenv("LORAHUB_KOHYA_SD_SCRIPTS", raising=False)
    monkeypatch.delenv("LORAHUB_KOHYA_PYTHON", raising=False)
    monkeypatch.setattr(app_mod, "_bootstrap_session", None)
    monkeypatch.setattr(app_mod, "_sweep_store", None)
    monkeypatch.setattr(app_mod, "_session_store", None)
    monkeypatch.setattr(app_mod, "_ai_store", None)
    monkeypatch.chdir(tmp_path)
    return TestClient(app_mod.app)


# --------------------------------------------------------------------- #
# Direct unit tests for run_preflight
# --------------------------------------------------------------------- #
def _cfg(tmp_path: Path, **overrides: Any) -> TrainingConfig:
    payload = _valid_payload(tmp_path)
    for dotted, value in overrides.items():
        keys = dotted.split(".")
        cur = payload
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = value
    return TrainingConfig.model_validate(payload)


def test_preflight_clean_config_has_no_blockers(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    findings = run_preflight(cfg, tmp_path / "ws", skip=("disk_low", "path_encoding"))
    blockers = [f for f in findings if f.severity == "error"]
    assert blockers == []


def test_preflight_missing_checkpoint_blocks(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, **{"base_model.checkpoint": str(tmp_path / "ghost.safetensors")})
    findings = run_preflight(cfg, tmp_path / "ws", skip=("disk_low", "path_encoding"))
    matches = [f for f in findings if f.category == "model_missing"]
    assert any(f.field == "baseModel.checkpoint" for f in matches), findings


def test_preflight_missing_dataset_blocks(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, **{"dataset.source": str(tmp_path / "ghost-data")})
    findings = run_preflight(cfg, tmp_path / "ws", skip=("disk_low", "path_encoding"))
    cats = {f.category for f in findings if f.severity == "error"}
    assert "dataset_missing" in cats, findings


def test_preflight_empty_dataset_dir_blocks(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    cfg = _cfg(tmp_path, **{"dataset.source": str(empty_dir)})
    findings = run_preflight(cfg, tmp_path / "ws", skip=("disk_low", "path_encoding"))
    cats = {f.category for f in findings if f.severity == "error"}
    assert "dataset_empty" in cats, findings


def test_preflight_backend_repo_missing_script_blocks(tmp_path: Path) -> None:
    bare = tmp_path / "bare-repo"
    bare.mkdir()
    cfg = _cfg(tmp_path, **{"backend.repo_path": str(bare)})
    findings = run_preflight(cfg, tmp_path / "ws", skip=("disk_low", "path_encoding"))
    cats = {f.category for f in findings if f.severity == "error"}
    assert "backend_repo_missing" in cats, findings


def test_preflight_python_executable_missing_blocks(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        **{"backend.python_executable": str(tmp_path / "no-such-python.exe")},
    )
    findings = run_preflight(cfg, tmp_path / "ws", skip=("disk_low", "path_encoding"))
    cats = {f.category for f in findings if f.severity == "error"}
    assert "venv_missing" in cats, findings


def test_preflight_collects_all_blockers_in_one_pass(tmp_path: Path) -> None:
    """Sanity: we surface every blocker, not just the first one."""
    cfg = _cfg(
        tmp_path,
        **{
            "base_model.checkpoint": str(tmp_path / "ghost.safetensors"),
            "dataset.source": str(tmp_path / "ghost-data"),
        },
    )
    findings = run_preflight(cfg, tmp_path / "ws", skip=("disk_low", "path_encoding"))
    cats = {f.category for f in findings if f.severity == "error"}
    assert {"model_missing", "dataset_missing"} <= cats, findings


def test_preflight_fsdp_requires_distributed_gpu_dispatch(
    tmp_path: Path,
) -> None:
    anima = _stub_anima_repo(tmp_path / "anima_lora")
    cfg = _cfg(
        tmp_path,
        **{
            "base_model.arch": "anima",
            "backend.type": "anima_lora",
            "backend.repo_path": str(anima),
            "backend.animaLora": {},
            "backend.distributed": {"strategy": "fsdp"},
        },
    )

    findings = run_preflight(cfg, tmp_path / "ws", skip=("disk_low", "path_encoding"))

    assert any(
        f.category == "gpu_dispatch"
        and f.field == "backend.distributed.strategy"
        and f.severity == "error"
        for f in findings
    ), findings


@pytest.mark.skipif(sys.platform != "win32", reason="mbcs probe is Windows-specific")
def test_preflight_path_encoding_emits_findings_when_provoked(tmp_path: Path) -> None:
    # Force a path with a code point that no Windows ANSI page can
    # encode (a high-plane emoji). The check must fire.
    bad = tmp_path / "ckpt-\U0001F4A9.safetensors"
    bad.write_bytes(b"")
    cfg = _cfg(tmp_path, **{"base_model.checkpoint": str(bad)})
    findings = run_preflight(cfg, tmp_path / "ws", skip=("disk_low",))
    cats = {f.category for f in findings if f.severity == "error"}
    assert "path_encoding" in cats, findings


# --------------------------------------------------------------------- #
# End-to-end via TestClient — make sure the router translates findings
# into a structured 422 response.
# --------------------------------------------------------------------- #
def test_create_job_blocks_when_checkpoint_missing(
    client: TestClient, tmp_path: Path
) -> None:
    payload = _valid_payload(tmp_path)
    payload["base_model"]["checkpoint"] = str(tmp_path / "no-such-model.safetensors")
    r = client.post(
        "/api/jobs",
        json={"config": payload, "workspace": str(tmp_path / "ws")},
    )
    assert r.status_code == 422, r.text
    body = r.json()
    detail = body["detail"]
    assert isinstance(detail, dict)
    assert "preflight failed" in detail["message"]
    assert any(
        f["category"] == "model_missing" and f["field"] == "baseModel.checkpoint"
        for f in detail["findings"]
    ), detail


def test_create_job_blocks_when_dataset_empty(
    client: TestClient, tmp_path: Path
) -> None:
    payload = _valid_payload(tmp_path)
    empty_dir = tmp_path / "empty-data"
    empty_dir.mkdir()
    payload["dataset"]["source"] = str(empty_dir)
    r = client.post(
        "/api/jobs",
        json={"config": payload, "workspace": str(tmp_path / "ws")},
    )
    assert r.status_code == 422, r.text
    body = r.json()
    cats = {f["category"] for f in body["detail"]["findings"]}
    assert "dataset_empty" in cats


def test_create_job_passes_with_warnings(client: TestClient, tmp_path: Path) -> None:
    """Warning-severity findings (e.g. disk_low when intentionally small) do
    not block creation. We can't reliably trigger disk_low in CI, so we
    just confirm a clean config gets a 202."""
    payload = _valid_payload(tmp_path)
    r = client.post(
        "/api/jobs",
        json={"config": payload, "workspace": str(tmp_path / "ws")},
    )
    assert r.status_code == 202, r.text
