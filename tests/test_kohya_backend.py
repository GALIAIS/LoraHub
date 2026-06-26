"""Tests for KohyaBackend (uses a stubbed sd-scripts checkout)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from lorahub.core.backends.base import Severity
from lorahub.core.backends.kohya.backend import KohyaBackend
from lorahub.core.config.schema import TrainingConfig
from lorahub.core.events import EventType, TrainingEvent


def _make_stub_sd_scripts(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stub = textwrap.dedent(
        """
        import sys
        print("loading model", flush=True)
        print("steps:   1%|          | 1/2 [00:01<00:01,  1.00s/it, avr_loss=0.5]", flush=True)
        print("epoch 1/1", flush=True)
        print("saving checkpoint: out.safetensors", flush=True)
        sys.exit(0)
        """
    ).strip() + "\n"
    # Stub every entry script the bootstrap probe checks for. New kohya
    # arches (sd2 reuses train_network.py; lumina/hunyuan_image/anima get
    # their own scripts) need a file on disk so probe_kohya_backend reports
    # `ready=True`. The contents are identical no-op stubs.
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


def _make_config(tmp_path: Path, sd_scripts: Path) -> TrainingConfig:
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    return TrainingConfig.model_validate(
        {
            "base_model": {"checkpoint": str(ckpt)},
            "dataset": {"source": str(data)},
            "schedule": {"epochs": 1, "batch_size": 1},
            "sampling": {"enabled": False},
            "backend": {
                "sd_scripts_path": str(sd_scripts),
                "python_executable": sys.executable,
            },
        }
    )


@pytest.fixture
def backend() -> KohyaBackend:
    return KohyaBackend()


def test_supported_archs_cover_main_models(backend: KohyaBackend) -> None:
    names = {a.value for a in backend.supported_archs}
    # Eight upstream-supported families per kohya sd-scripts README.
    assert names == {
        "sd15",
        "sd2",
        "sdxl",
        "sd3",
        "flux",
        "lumina",
        "hunyuan_image",
        "anima",
    }


def test_supported_archs_excludes_dp_only_models(backend: KohyaBackend) -> None:
    """kohya does not ship trainers for these dp-only arches."""
    names = {a.value for a in backend.supported_archs}
    for dp_only in ("wan", "hunyuan_video", "chroma", "ltx_video", "flux2"):
        assert dp_only not in names


def test_validate_passes_for_good_config(tmp_path: Path, backend: KohyaBackend) -> None:
    sd = _make_stub_sd_scripts(tmp_path / "sd-scripts")
    config = _make_config(tmp_path, sd)
    issues = backend.validate(config)
    errors = [i for i in issues if i.severity is Severity.error]
    assert errors == []


def test_validate_reports_missing_sd_scripts(tmp_path: Path, backend: KohyaBackend) -> None:
    config = _make_config(tmp_path, tmp_path / "missing")
    issues = backend.validate(config)
    assert any(
        i.severity is Severity.error and "repo_path" in i.field for i in issues
    )


def test_estimate_vram_returns_sane_numbers(tmp_path: Path, backend: KohyaBackend) -> None:
    sd = _make_stub_sd_scripts(tmp_path / "sd-scripts")
    config = _make_config(tmp_path, sd)
    est = backend.estimate_vram(config)
    assert est.total_mib > 0
    assert 1.0 <= est.total_gib <= 32.0


@pytest.mark.parametrize(
    "arch",
    [
        "sd15",
        "sd2",
        "sdxl",
        "sd3",
        "flux",
        "flux2",
        "lumina",
        "anima",
        "hunyuan_image",
        "chroma",
        "hidream",
        "omnigen2",
        "auraflow",
        "qwen_image",
        "cosmos",
        "cosmos_predict2",
        "hunyuan_video",
        "hunyuan_video_15",
        "ltx_video",
        "ltx2",
        "wan",
        "z_image",
        "ernie_image",
    ],
)
def test_estimate_vram_covers_every_arch(
    tmp_path: Path, backend: KohyaBackend, arch: str
) -> None:
    """Every arch in the matrix must yield a non-crashing, positive estimate."""
    from lorahub.core.backends.base import VRAMEstimate

    sd = _make_stub_sd_scripts(tmp_path / "sd-scripts")
    config = _make_config(tmp_path, sd)
    cfg = config.model_copy(
        update={
            "base_model": config.base_model.model_copy(update={"arch": arch}),
        }
    )
    est = backend.estimate_vram(cfg)
    assert isinstance(est, VRAMEstimate)
    assert est.total_mib > 0


def test_estimate_vram_activations_scale_with_batch_size(
    tmp_path: Path, backend: KohyaBackend
) -> None:
    """Doubling batch_size doubles the activations component."""
    sd = _make_stub_sd_scripts(tmp_path / "sd-scripts")
    config = _make_config(tmp_path, sd)
    # Disable gradient_checkpointing so the //3 discount doesn't fold the
    # multiplier away through integer truncation.
    config = config.model_copy(update={"gradient_checkpointing": False})

    bs1 = backend.estimate_vram(
        config.model_copy(
            update={"schedule": config.schedule.model_copy(update={"batch_size": 1})}
        )
    )
    bs2 = backend.estimate_vram(
        config.model_copy(
            update={"schedule": config.schedule.model_copy(update={"batch_size": 2})}
        )
    )
    assert bs2.activations_mib == 2 * bs1.activations_mib


def test_launch_runs_to_completion(tmp_path: Path, backend: KohyaBackend) -> None:
    sd = _make_stub_sd_scripts(tmp_path / "sd-scripts")
    config = _make_config(tmp_path, sd)

    events: list[TrainingEvent] = []
    handle = backend.launch(config, workspace=tmp_path / "ws", on_event=events.append)
    assert handle.pid is not None
    rc = handle.wait(timeout=30)
    assert rc == 0

    types = [e.type for e in events]
    assert EventType.step in types
    assert EventType.epoch_end in types
    assert EventType.checkpoint_saved in types
    assert types[-1] is EventType.done


def test_requirement_probe_hides_windows_console(monkeypatch: pytest.MonkeyPatch) -> None:
    from lorahub.core.backends._common import bootstrap

    calls: list[dict[str, object]] = []

    monkeypatch.setattr(bootstrap.sys, "platform", "win32")
    monkeypatch.setattr(bootstrap.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)

    def fake_run(*_args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(returncode=0, stdout="torch==2.0.0\n", stderr="")

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)

    assert bootstrap._get_installed_packages(Path("python.exe")) == {"torch"}
    assert calls[0]["creationflags"] == 0x08000000


def test_msvc_probe_hides_windows_console(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from lorahub.core.backends.anima_lora import msvc

    calls: list[dict[str, object]] = []
    vswhere = tmp_path / "vswhere.exe"
    vswhere.write_text("", encoding="utf-8")

    monkeypatch.setattr(msvc.sys, "platform", "win32")
    monkeypatch.setattr(msvc.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(msvc, "_find_vswhere", lambda: vswhere)

    def fake_check_output(*_args: object, **kwargs: object) -> str:
        calls.append(kwargs)
        return ""

    monkeypatch.setattr(msvc.subprocess, "check_output", fake_check_output)

    assert msvc._msvc_root_via_vswhere() is None
    assert calls[0]["creationflags"] == 0x08000000
