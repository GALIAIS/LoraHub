from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import lorahub.api.app  # noqa: F401
from lorahub.api import terminal_runner
from lorahub.api.routers.terminal import _enforce_command_policy
from lorahub.api.terminal_runner import TerminalDenied


@pytest.fixture
def python_path(tmp_path: Path) -> Path:
    path = tmp_path / "python"
    path.touch()
    return path


@pytest.mark.parametrize(
    "argv",
    [
        ["python", "-c", "import os; os.remove('data')"],
        ["python", "script.py"],
        ["python", "-m", "ensurepip"],
        ["/tmp/git", "status"],
        ["uv", "run", "tool.py"],
        ["git", "clean", "-fdx"],
        ["git", "reset", "--hard"],
        ["git", "diff", "--output", "report.patch"],
        ["git", "diff", "--no-index", "a", "b"],
        ["git", "cat-file", "--filters=main:README.md"],
        ["git", "cat-file", "--filters", "main:README.md"],
        ["git", "blame", "--contents", "/tmp/secret", "README.md"],
        ["git", "ls-files", "--exclude-from=/tmp/secret"],
        ["git", "show", "--output=report.txt", "HEAD"],
        ["git", "log", "-oreport.txt"],
        ["find", ".", "-delete"],
        ["find", ".", "-exec", "rm", "{}", ";"],
        ["pip", "download", "torch"],
        ["pip", "install", "torch"],
        ["pip", "uninstall", "-y", "torch"],
        ["pip", "install", "--target", "outside", "torch"],
        ["pip", "install", "-toutside", "torch"],
        ["python", "-m", "pip", "install", "torch"],
        ["uv", "pip", "install", "torch"],
        ["python", "-m", "pip", "config", "set", "global.index-url", "x"],
        ["uv", "pip", "install", "--python", "outside", "torch"],
        ["rg", "--pre", "run-me", "pattern", "."],
        ["tree", "-o", "tree.txt"],
        ["ps", "auxe"],
        ["ps", "-auxe"],
        ["lorahub", "manage", "update"],
        ["lorahub", "jobs", "kill", "job-id"],
        ["lorahub", "ref-extract", "model.safetensors", "--output", "outside"],
        ["nvidia-smi", "--gpu-reset"],
        ["nvidia-smi", "-pl", "200"],
        ["nvidia-smi", "-f", "report.xml"],
        ["nvcc", "kernel.cu", "-o", "kernel"],
    ],
)
def test_restricted_terminal_rejects_execution_and_mutation(
    argv: list[str], python_path: Path,
) -> None:
    with pytest.raises(TerminalDenied):
        _enforce_command_policy(
            argv,
            python_path=python_path,
            unrestricted=False,
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["python", "--version"],
        ["python", "-m", "pip", "list"],
        ["uv", "pip", "list"],
        ["git", "status", "--short"],
        ["git", "branch", "--show-current"],
        ["git", "remote", "-v"],
        ["lorahub", "doctor"],
        ["lorahub", "system", "gpu"],
        ["lorahub", "jobs", "show", "job-id"],
        ["nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader"],
        ["nvcc", "--version"],
    ],
)
def test_restricted_terminal_allows_supported_diagnostics(
    argv: list[str], python_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lorahub.api.routers.terminal._venv_has_pip",
        lambda _path: True,
    )
    monkeypatch.setattr("lorahub.api.routers.terminal._find_uv", lambda: "uv")
    result = _enforce_command_policy(
        argv,
        python_path=python_path,
        unrestricted=False,
    )
    assert result


def test_unrestricted_terminal_keeps_arbitrary_argv(python_path: Path) -> None:
    argv = ["git", "reset", "--hard"]
    assert _enforce_command_policy(
        argv,
        python_path=python_path,
        unrestricted=True,
    ) == argv


def test_restricted_git_disables_configured_helpers(python_path: Path) -> None:
    result = _enforce_command_policy(
        ["git", "diff", "--stat"],
        python_path=python_path,
        unrestricted=False,
    )

    assert result[:3] == ["git", "-c", "core.fsmonitor=false"]
    assert "--no-ext-diff" in result
    assert "--no-textconv" in result


def test_restricted_terminal_limits_file_reads_to_backend_directory(
    tmp_path: Path,
    python_path: Path,
) -> None:
    backend = tmp_path / "backend"
    backend.mkdir()
    inside = backend / "README.md"
    inside.write_text("safe", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    allowed = _enforce_command_policy(
        ["cat", "README.md"],
        python_path=python_path,
        unrestricted=False,
        cwd=backend,
    )
    assert allowed == ["cat", "README.md"]

    for argv in (["cat", str(outside)], ["cat", "../secret.txt"]):
        with pytest.raises(TerminalDenied):
            _enforce_command_policy(
                argv,
                python_path=python_path,
                unrestricted=False,
                cwd=backend,
            )


def test_restricted_terminal_rejects_link_escape(
    tmp_path: Path,
    python_path: Path,
) -> None:
    backend = tmp_path / "backend"
    backend.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = backend / "linked.txt"
    try:
        link.symlink_to(outside)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(TerminalDenied):
        _enforce_command_policy(
            ["cat", "linked.txt"],
            python_path=python_path,
            unrestricted=False,
            cwd=backend,
        )


def test_windows_process_tree_cleanup_falls_back_when_taskkill_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 42
        killed = False

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.killed = True

    proc = FakeProcess()
    monkeypatch.setattr(terminal_runner.os, "name", "nt")
    monkeypatch.setattr(
        terminal_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )

    terminal_runner._terminate_process_tree(proc)  # type: ignore[arg-type]

    assert proc.killed is True
