from pathlib import Path
import sys

import yaml

from lorahub.api.jobs_helpers import _select_backend
from lorahub.api.settings import Settings, probe_ai_toolkit_backend
from lorahub.api.terminal_runner import resolve_backend_session
from lorahub.core.backends.ai_toolkit.backend import AIToolkitBackend
from lorahub.core.backends.ai_toolkit.compiler import compile_config
from lorahub.core.backends.ai_toolkit.parser import parse_line
from lorahub.core.config.schema import TrainingConfig
from lorahub.core.events import EventType


def _cfg() -> TrainingConfig:
    return TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": ".", "resolution": [1024, 1024]},
            "backend": {
                "type": "ai_toolkit",
                "extraArgs": {
                    "model.name_or_path": "krea/Krea-2-Turbo",
                    "model.assistant_lora_path": "adapter.safetensors",
                },
            },
        }
    )


def test_ai_toolkit_compiler_emits_krea2_yaml(tmp_path: Path) -> None:
    argv, files = compile_config(_cfg(), tmp_path)
    assert argv == [str(tmp_path / "_lorahub_ai_toolkit.yaml")]
    data = yaml.safe_load(next(iter(files.values())))
    process = data["config"]["process"][0]
    assert process["model"]["arch"] == "krea2"
    assert process["model"]["name_or_path"] == "krea/Krea-2-Turbo"
    assert process["model"]["assistant_lora_path"] == "adapter.safetensors"


def test_ai_toolkit_compiler_maps_visible_sampling_fields(tmp_path: Path) -> None:
    cfg = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": ".", "resolution": [768, 1024]},
            "backend": {"type": "ai_toolkit"},
            "sampling": {
                "enabled": True,
                "everyNSteps": 250,
                "resolution": [832, 1216],
                "seed": 123,
                "inferenceSteps": 28,
                "inferenceCfg": 4.5,
                "prompts": [{"prompt": "test prompt"}],
            },
        }
    )

    _, files = compile_config(cfg, tmp_path)
    data = yaml.safe_load(next(iter(files.values())))
    process = data["config"]["process"][0]

    assert process["train"]["disable_sampling"] is False
    assert process["sample"] == {
        "sample_every": 250,
        "width": 832,
        "height": 1216,
        "seed": 123,
        "sample_steps": 28,
        "guidance_scale": 4.5,
        "prompts": ["test prompt"],
    }


def test_ai_toolkit_compiler_honors_disabled_sampling(tmp_path: Path) -> None:
    cfg = TrainingConfig.model_validate(
        {
            "baseModel": {"arch": "krea2", "checkpoint": "krea/Krea-2-Raw"},
            "dataset": {"source": ".", "resolution": [1024, 1024]},
            "backend": {"type": "ai_toolkit"},
            "sampling": {"enabled": False},
        }
    )

    _, files = compile_config(cfg, tmp_path)
    data = yaml.safe_load(next(iter(files.values())))
    process = data["config"]["process"][0]

    assert process["train"]["disable_sampling"] is True


def test_ai_toolkit_parser_recognizes_step_and_checkpoint() -> None:
    step = parse_line("training 12/100 loss: 0.1234")
    assert step is not None
    assert step.type is EventType.step
    assert step.payload["step"] == 12
    assert step.payload["total_steps"] == 100
    assert step.payload["loss"] == 0.1234

    saved = parse_line("Saved checkpoint to C:/runs/foo.safetensors")
    assert saved is not None
    assert saved.type is EventType.checkpoint_saved
    assert saved.payload["path"] == "C:/runs/foo.safetensors"


def test_select_backend_returns_ai_toolkit_backend() -> None:
    backend = _select_backend(_cfg())
    assert isinstance(backend, AIToolkitBackend)
    assert backend.name == "ai_toolkit"


def test_probe_ai_toolkit_backend_reports_vendored_ready(tmp_path: Path) -> None:
    repo = tmp_path / "ai_toolkit"
    (repo / "toolkit").mkdir(parents=True)
    (repo / "extensions_built_in" / "sd_trainer").mkdir(parents=True)
    (repo / "run.py").write_text("", encoding="utf-8")
    (repo / "toolkit" / "job.py").write_text("", encoding="utf-8")
    (repo / "extensions_built_in" / "sd_trainer" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    (repo / "requirements.txt").write_text("", encoding="utf-8")
    py = repo / ".venv" / "Scripts" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")

    status = probe_ai_toolkit_backend(
        Settings(ai_toolkit_repo_path=str(repo), ai_toolkit_python=sys.executable)
    )

    assert status["id"] == "ai_toolkit"
    assert status["repo_ok"] is True
    assert status["python_ok"] is True
    assert status["venv_detected"] is True
    assert status["ready"] is True


def test_terminal_resolves_ai_toolkit_session(tmp_path: Path) -> None:
    repo = tmp_path / "ai_toolkit"
    repo.mkdir()
    py = repo / ".venv" / "Scripts" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")

    session = resolve_backend_session(
        "ai_toolkit",
        Settings(ai_toolkit_repo_path=str(repo), ai_toolkit_python=str(py)),
    )

    assert session.backend_id == "ai_toolkit"
    assert session.display_name == "AI Toolkit"
    assert session.repo_path == repo
    assert session.python_path == py
    assert session.ready is True
