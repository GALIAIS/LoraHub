"""Tests for DiffusionPipeBackend (uses a stubbed diffusion-pipe checkout)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from lorahub.core.backends.base import Severity
from lorahub.core.backends.diffusion_pipe.backend import DiffusionPipeBackend
from lorahub.core.config.schema import TrainingConfig
from lorahub.core.events import EventType, TrainingEvent


def _make_stub_repo(root: Path) -> Path:
    """Create a fake diffusion-pipe checkout with a no-op `train.py`."""
    root.mkdir(parents=True, exist_ok=True)
    stub = textwrap.dedent(
        """
        import sys
        # Mimic the surface of train.py just enough for the parser to fire.
        print("loaded config", flush=True)
        print("Started new epoch: 1", flush=True)
        print("Saving model to directory epoch1", flush=True)
        sys.exit(0)
        """
    ).strip() + "\n"
    (root / "train.py").write_text(stub, encoding="utf-8")
    return root


def _make_stub_python_with_deepspeed(tmp_path: Path) -> Path:
    """Create a fake `<venv>/bin/python` plus a `deepspeed` launcher next to it.

    The runner now invokes `<python>.parent / 'deepspeed'` instead of plain
    `python` (so deepspeed.init_distributed() doesn't fall back to MPI),
    so tests that exercise launch need a stub deepspeed shim that just
    forwards argv to the real interpreter.
    """
    bindir = tmp_path / "venv" / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    python = bindir / "python"
    deepspeed = bindir / "deepspeed"
    if sys.platform == "win32":
        # On Windows the runner falls back to deepspeed.exe; we keep tests
        # POSIX-only by skipping there. Backends test runs on linux CI.
        pass
    python.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/sh
            exec "{sys.executable}" "$@"
            """
        ),
        encoding="utf-8",
    )
    deepspeed.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/sh
            exec "{sys.executable}" "$@"
            """
        ),
        encoding="utf-8",
    )
    python.chmod(0o755)
    deepspeed.chmod(0o755)
    return python


def _make_config(tmp_path: Path, repo: Path, *, arch: str = "sdxl") -> TrainingConfig:
    ckpt = tmp_path / ("model.safetensors" if arch == "sdxl" else "diffusers")
    if arch == "sdxl":
        ckpt.write_bytes(b"")
    else:
        ckpt.mkdir(exist_ok=True)
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    return TrainingConfig.model_validate(
        {
            "base_model": {"arch": arch, "checkpoint": str(ckpt)},
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


@pytest.fixture
def backend() -> DiffusionPipeBackend:
    return DiffusionPipeBackend()


def test_supported_archs_excludes_sd15(backend: DiffusionPipeBackend) -> None:
    names = {a.value for a in backend.supported_archs}
    assert {"sdxl", "flux", "sd3"}.issubset(names)
    assert "sd15" not in names
    assert "sd2" not in names


def test_supported_archs_cover_dp_only_models(backend: DiffusionPipeBackend) -> None:
    """The full dp matrix (Wan, HunyuanVideo, Chroma, ...) is reachable."""
    names = {a.value for a in backend.supported_archs}
    assert {
        "wan",
        "hunyuan_video",
        "hunyuan_video_15",
        "ltx_video",
        "ltx2",
        "chroma",
        "hidream",
        "omnigen2",
        "auraflow",
        "qwen_image",
        "cosmos",
        "cosmos_predict2",
        "anima",
        "hunyuan_image",
        "lumina",
        "flux2",
        "z_image",
        "ernie_image",
    }.issubset(names)


def test_validate_passes_for_good_config(
    tmp_path: Path, backend: DiffusionPipeBackend
) -> None:
    repo = _make_stub_repo(tmp_path / "dp")
    config = _make_config(tmp_path, repo, arch="flux")
    issues = backend.validate(config)
    errors = [i for i in issues if i.severity is Severity.error]
    assert errors == []


def test_validate_reports_missing_repo(
    tmp_path: Path, backend: DiffusionPipeBackend
) -> None:
    config = _make_config(tmp_path, tmp_path / "missing")
    issues = backend.validate(config)
    assert any(
        i.severity is Severity.error and "repo_path" in i.field for i in issues
    )


def test_validate_rejects_sd15_with_pointer_to_kohya(
    tmp_path: Path, backend: DiffusionPipeBackend
) -> None:
    repo = _make_stub_repo(tmp_path / "dp")
    ckpt = tmp_path / "sd15.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "d"
    data.mkdir()
    config = TrainingConfig.model_validate(
        {
            "base_model": {"arch": "sd15", "checkpoint": str(ckpt)},
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
    issues = backend.validate(config)
    arch_errors = [
        i for i in issues if i.severity is Severity.error and i.field == "base_model.arch"
    ]
    assert len(arch_errors) == 1
    assert "kohya" in arch_errors[0].message.lower()


def test_estimate_vram_returns_sane_numbers(
    tmp_path: Path, backend: DiffusionPipeBackend
) -> None:
    repo = _make_stub_repo(tmp_path / "dp")
    config = _make_config(tmp_path, repo, arch="sdxl")
    est = backend.estimate_vram(config)
    assert est.total_mib > 0


@pytest.mark.parametrize(
    "arch",
    [
        # full 23-arch matrix; dp supports a superset for vram estimation
        # purposes even though sd15/sd2 fail validation downstream.
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
    tmp_path: Path, backend: DiffusionPipeBackend, arch: str
) -> None:
    """Every arch yields a positive estimate, even ones dp would refuse to launch."""
    from lorahub.core.backends.base import VRAMEstimate

    repo = _make_stub_repo(tmp_path / "dp")
    # `_make_config` creates either a flat .safetensors or a diffusers dir
    # depending on the arch token; reuse it so checkpoint shape matches what
    # the dp config expects.
    config = _make_config(tmp_path, repo, arch="sdxl")
    cfg = config.model_copy(
        update={
            "base_model": config.base_model.model_copy(update={"arch": arch}),
        }
    )
    est = backend.estimate_vram(cfg)
    assert isinstance(est, VRAMEstimate)
    assert est.total_mib > 0


def test_estimate_vram_activations_scale_with_batch_size(
    tmp_path: Path, backend: DiffusionPipeBackend
) -> None:
    """Doubling batch_size doubles the activations component."""
    repo = _make_stub_repo(tmp_path / "dp")
    config = _make_config(tmp_path, repo, arch="sdxl")
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


def test_launch_writes_toml_files_and_runs_subprocess(
    tmp_path: Path, backend: DiffusionPipeBackend
) -> None:
    if sys.platform == "win32":
        pytest.skip("shell-script stubs only work on POSIX")
    repo = _make_stub_repo(tmp_path / "dp")
    stub_python = _make_stub_python_with_deepspeed(tmp_path / "dp")
    config = TrainingConfig.model_validate(
        {
            "base_model": {"arch": "sdxl", "checkpoint": str(_make_config(tmp_path, repo, arch="sdxl").base_model.checkpoint)},
            "dataset": {"source": str(tmp_path / "data")},
            "schedule": {"epochs": 1, "batch_size": 1},
            "sampling": {"enabled": False},
            "backend": {
                "type": "diffusion-pipe",
                "sd_scripts_path": str(repo),
                "python_executable": str(stub_python),
            },
        }
    )
    workspace = tmp_path / "ws"

    events: list[TrainingEvent] = []
    handle = backend.launch(config, workspace=workspace, on_event=events.append)
    assert handle.pid is not None
    rc = handle.wait(timeout=30)
    assert rc == 0

    # The two TOML files were materialised before launch.
    assert (workspace / "diffusion_pipe.toml").is_file()
    assert (workspace / "dataset.toml").is_file()
    main_toml = (workspace / "diffusion_pipe.toml").read_text(encoding="utf-8")
    assert "[model]" in main_toml
    assert "[adapter]" in main_toml

    # Parser surfaced the stub's epoch + save lines, and the runner
    # always emits a terminal `done` event.
    types = [e.type for e in events]
    assert EventType.epoch_end in types
    assert EventType.checkpoint_saved in types
    assert types[-1] is EventType.done


# --------------------------------------------------------------------------- #
# B8 — Multi-node DeepSpeed launcher
# --------------------------------------------------------------------------- #


def test_dp_runner_passes_multi_node_launcher_flags(tmp_path: Path) -> None:
    """B8: when config.backend.diffusionPipe.multiNode is set,
    the runner prepends ``--hostfile`` / ``--num_nodes`` (etc.) to the
    deepspeed argv BEFORE the train.py path. DeepSpeed's launcher parses
    its own argv up to ``train.py`` so the order matters; this test
    locks both presence + ordering.
    """
    from types import SimpleNamespace
    from unittest.mock import patch

    from lorahub.core.backends.diffusion_pipe.runner import DiffusionPipeRunner

    repo = tmp_path / "dp"
    repo.mkdir()
    (repo / "train.py").write_text("# stub\n", encoding="utf-8")
    bindir = repo / "venv" / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    deepspeed_bin = bindir / "deepspeed"
    deepspeed_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    deepspeed_bin.chmod(0o755) if sys.platform != "win32" else None

    captured: list[list[str]] = []

    def fake_start(self):  # type: ignore[no-untyped-def]
        captured.append(list(self._argv))
        self._proc = SimpleNamespace(pid=4242, returncode=0)

    with patch(
        "lorahub.core.backends._common.runner.SubprocessRunner.start", fake_start
    ):
        runner = DiffusionPipeRunner(
            python=bindir / "python",
            repo=repo,
            argv=["--deepspeed", "--config", "x.toml"],
            workspace=tmp_path / "ws",
            on_event=lambda _e: None,
            launcher_args=[
                "--hostfile", "/abs/hostfile",
                "--num_nodes", "4",
                "--master_addr", "10.0.0.1",
                "--master_port", "29501",
            ],
        )
        runner.start()

    assert len(captured) == 1
    argv = captured[0]
    # deepspeed bin first
    assert argv[0].endswith("deepspeed") or argv[0].endswith("deepspeed.exe")
    # launcher args come BEFORE train.py
    train_idx = next(i for i, x in enumerate(argv) if x.endswith("train.py"))
    hostfile_idx = argv.index("--hostfile")
    num_nodes_idx = argv.index("--num_nodes")
    master_addr_idx = argv.index("--master_addr")
    master_port_idx = argv.index("--master_port")
    assert hostfile_idx < train_idx
    assert num_nodes_idx < train_idx
    assert master_addr_idx < train_idx
    assert master_port_idx < train_idx
    # values land where expected
    assert argv[hostfile_idx + 1] == "/abs/hostfile"
    assert argv[num_nodes_idx + 1] == "4"
    assert argv[master_addr_idx + 1] == "10.0.0.1"
    assert argv[master_port_idx + 1] == "29501"
    # config argv preserved AFTER train.py
    assert "--deepspeed" in argv[train_idx + 1 :]
    assert "--config" in argv[train_idx + 1 :]


def test_dp_runner_no_launcher_args_when_single_node(tmp_path: Path) -> None:
    """Single-node path: launcher_args=None → no --hostfile leaks in."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from lorahub.core.backends.diffusion_pipe.runner import DiffusionPipeRunner

    repo = tmp_path / "dp"
    repo.mkdir()
    (repo / "train.py").write_text("# stub\n", encoding="utf-8")
    bindir = repo / "venv" / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    (bindir / "deepspeed").write_text("#!/bin/sh\n", encoding="utf-8")

    captured: list[list[str]] = []

    def fake_start(self):  # type: ignore[no-untyped-def]
        captured.append(list(self._argv))
        self._proc = SimpleNamespace(pid=4242, returncode=0)

    with patch(
        "lorahub.core.backends._common.runner.SubprocessRunner.start", fake_start
    ):
        runner = DiffusionPipeRunner(
            python=bindir / "python",
            repo=repo,
            argv=["--deepspeed"],
            workspace=tmp_path / "ws",
            on_event=lambda _e: None,
        )
        runner.start()

    argv = captured[0]
    for flag in ("--hostfile", "--num_nodes", "--master_addr", "--master_port"):
        assert flag not in argv, f"single-node should not emit {flag}"


def test_multi_node_schema_requires_num_nodes_ge_2() -> None:
    """schema validates num_nodes >= 2 — single-node use should leave the
    whole multi_node block None instead of setting num_nodes=1."""
    import pydantic

    from lorahub.core.config.schema import MultiNodeConfig

    # 2+ is fine
    MultiNodeConfig(hostfile="/x", num_nodes=2)
    MultiNodeConfig(hostfile="/x", num_nodes=8)
    # 1 should fail
    with pytest.raises(pydantic.ValidationError):
        MultiNodeConfig(hostfile="/x", num_nodes=1)
