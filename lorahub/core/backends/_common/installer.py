"""Shared install-step helpers for backend bootstrappers.

Both `kohya.installer` and `diffusion_pipe.installer` clone a Git repo,
build a uv venv, install pinned torch wheels, then install the rest of the
backend's `requirements.txt` -- always with the same progress callback shape
and the same error wrapping. The functions here factor that out and take a
plan-shaped object via the `BootstrapPlanLike` protocol so each backend can
keep its own `BootstrapPlan` dataclass with its own extras (xformers vs
deepspeed, etc.).
"""

from __future__ import annotations

import collections
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from lorahub.core.backends.errors import BootstrapError
from lorahub.core.paths import project_root
from lorahub.core.redaction import redact_command_text
from lorahub.core.toolchain import uv as _uv

ProgressCallback = Callable[[str], None]

DEFAULT_TORCH = "2.6.0"
DEFAULT_TORCHVISION = "0.21.0"
DEFAULT_CUDA = "cu124"
DEFAULT_DEPTH = 1
DEFAULT_TORCH_INDEX_BASE = "https://download.pytorch.org/whl"
FALLBACK_TORCH_INDEX_BASES: tuple[str, ...] = (
    "https://mirrors.nju.edu.cn/pytorch/whl",
    "https://mirror.sjtu.edu.cn/pytorch-wheels",
    DEFAULT_TORCH_INDEX_BASE,
    "https://mirrors.aliyun.com/pytorch-wheels",
)


def _subprocess_no_window() -> int:
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


class BootstrapPlanLike(Protocol):
    """The minimum shape every backend's BootstrapPlan must expose.

    Both backends' frozen dataclasses already satisfy this implicitly; the
    protocol just documents the surface so the helpers below can stay
    backend-agnostic.
    """

    @property
    def target(self) -> Path: ...

    @property
    def cuda_version(self) -> str: ...

    @property
    def torch_version(self) -> str: ...

    @property
    def torchvision_version(self) -> str: ...

    @property
    def base_python(self) -> Path | None: ...

    @property
    def pypi_index(self) -> str | None: ...

    @property
    def torch_index_base(self) -> str | None: ...

    @property
    def venv_python(self) -> Path: ...

    @property
    def torch_index(self) -> str: ...


class ClonePlanLike(BootstrapPlanLike, Protocol):
    """Additional fields required only by remote repository clones."""

    @property
    def git_depth(self) -> int: ...

    @property
    def github_proxy(self) -> str | None: ...


def run_step(
    cmd: list[str],
    step: str,
    progress: ProgressCallback | None,
) -> None:
    """Run a non-package subprocess (typically `git clone`) with stderr capture.

    Streams the subprocess's stderr **line by line** to ``progress`` so the
    dashboard can surface git's own progress output (e.g. ``Receiving objects:
    23% (...)``) while the clone is still running, instead of waiting for the
    process to exit. Reports the step name through ``progress`` before
    launching, and on a non-zero exit code attaches the last 12 stderr lines
    to the progress stream so the UI can surface a useful error message.
    """
    if progress is not None:
        progress(step)
    proc = subprocess.Popen(  # noqa: S603 -- caller controls argv
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered so we get progress lines as they're written
        creationflags=_subprocess_no_window(),
    )
    tail: collections.deque[str] = collections.deque(maxlen=12)
    assert proc.stderr is not None  # noqa: S101 -- PIPE above guarantees this
    for raw_line in proc.stderr:
        line = redact_command_text(raw_line.rstrip())
        if not line:
            continue
        tail.append(line)
        if progress is not None:
            # Forward each line so the dashboard sees git's own progress
            # output. Indent with two spaces so multiple concurrent steps
            # stay readable in a combined log stream.
            progress(f"  {line}")
    rc = proc.wait()
    if rc != 0:
        if progress is not None and tail:
            progress(f"{step} failed (exit {rc}):\n" + "\n".join(tail))
        raise BootstrapError(step, rc)


def _is_complete_git_repo(target: Path) -> bool:
    """Return True if *target* is a usable shallow/full git checkout."""
    git_dir = target / ".git"
    if not git_dir.is_dir():
        return False
    result = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
        creationflags=_subprocess_no_window(),
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def is_complete_git_repo(target: Path) -> bool:
    """Public, side-effect-free checkout probe used by bootstrap safety gates."""
    return _is_complete_git_repo(target)


def _install_marker_path(target: Path) -> Path:
    return target.parent / f".{target.name}.lorahub-install.json"


def _write_install_marker(target: Path, repo_url: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    marker = _install_marker_path(target)
    payload = json.dumps(
        {"target": str(target.expanduser().resolve()), "repo_url": repo_url},
        ensure_ascii=True,
    )
    fd, raw = tempfile.mkstemp(
        dir=marker.parent,
        prefix=f".{marker.name}.",
        suffix=".tmp",
    )
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        tmp.replace(marker)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def is_managed_partial_install(target: Path, repo_url: str) -> bool:
    """Return whether an interrupted clone marker owns this exact target."""
    marker = _install_marker_path(target)
    if marker.is_symlink() or not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return bool(
        isinstance(payload, dict)
        and payload.get("target") == str(target.expanduser().resolve())
        and payload.get("repo_url") == repo_url
    )


def clear_install_marker(target: Path, repo_url: str) -> None:
    marker = _install_marker_path(target)
    if is_managed_partial_install(target, repo_url):
        try:
            marker.unlink(missing_ok=True)
        except OSError:
            pass


def clone_repo(
    plan: ClonePlanLike,
    *,
    repo_url: str,
    label: str,
    recurse_submodules: bool = False,
    progress: ProgressCallback | None = None,
) -> None:
    """Run ``git clone --depth ... <repo_url> <plan.target>``.

    If the target already contains a complete git checkout, the clone is
    skipped entirely (avoids re-downloading multi-GB repos on retry).
    Otherwise, refuses to clone into a non-empty directory and applies the
    optional GitHub proxy from settings to the URL before invoking git.

    When ``recurse_submodules`` is True, also runs
    ``git submodule update --init --recursive`` after cloning. Required for
    diffusion-pipe, which keeps ComfyUI and HunyuanVideo as submodules whose
    contents are imported at training time (e.g. ``import comfy``).
    """
    try:
        validate_backend_source_target(plan.target)
    except ValueError as exc:
        raise BootstrapError("validate clone target", 1) from exc
    if plan.target.exists() and not plan.target.is_dir():
        msg = f"target path is not a directory: {plan.target}"
        raise BootstrapError("clone", 1) from NotADirectoryError(msg)
    if plan.target.exists() and any(plan.target.iterdir()):
        if _is_complete_git_repo(plan.target):
            clear_install_marker(plan.target, repo_url)
            if progress is not None:
                progress(f"clone {label} -> {plan.target} (already complete, skipped)")
            if recurse_submodules:
                _ensure_submodules(plan.target, label, progress=progress)
            return
        msg = f"target directory is not empty: {plan.target}"
        raise BootstrapError("clone", 1) from FileExistsError(msg)
    plan.target.parent.mkdir(parents=True, exist_ok=True)
    from lorahub.api.settings import apply_github_proxy  # noqa: PLC0415

    proxied = apply_github_proxy(repo_url, plan.github_proxy)
    cmd = [
        "git",
        "clone",
        "--progress",
        "--depth",
        str(plan.git_depth),
    ]
    if recurse_submodules:
        cmd += ["--recurse-submodules", "--shallow-submodules"]
    cmd += [proxied, str(plan.target)]
    try:
        _write_install_marker(plan.target, repo_url)
    except OSError as exc:
        raise BootstrapError("prepare clone", 1) from exc
    run_step(cmd, f"clone {label} -> {plan.target}", progress)
    clear_install_marker(plan.target, repo_url)


def _ensure_submodules(
    target: Path,
    label: str,
    *,
    progress: ProgressCallback | None = None,
) -> None:
    """Re-init submodules on an existing checkout (idempotent)."""
    cmd = [
        "git",
        "-C",
        str(target),
        "submodule",
        "update",
        "--init",
        "--recursive",
        "--depth",
        "1",
    ]
    run_step(cmd, f"sync submodules for {label}", progress)


def create_venv(
    plan: BootstrapPlanLike,
    *,
    progress: ProgressCallback | None = None,
) -> None:
    try:
        validate_backend_source_target(plan.target)
        _uv.create_venv(plan.target, python=plan.base_python, progress=progress)
    except (RuntimeError, ValueError) as exc:
        raise BootstrapError("create venv", 1) from exc


def upgrade_pip(plan: BootstrapPlanLike, *, progress: ProgressCallback | None = None) -> None:
    """No-op under uv -- uv ships its own resolver and skips pip+wheel.

    Kept on the bootstrap plan so the per-step progress UI keeps lining up;
    we just emit a status line and move on.
    """
    if progress is not None:
        progress("upgrade pip + wheel + setuptools (skipped under uv)")
    _ = plan  # keep signature symmetric with sibling helpers


def install_torch(
    plan: BootstrapPlanLike,
    *,
    progress: ProgressCallback | None = None,
) -> None:
    args = [
        f"torch=={plan.torch_version}",
        f"torchvision=={plan.torchvision_version}",
        "--index-url",
        plan.torch_index,
    ]
    try:
        pip_install_with_torch_index_fallback(
            plan,
            args,
            step=f"install torch=={plan.torch_version} ({plan.cuda_version})",
            progress=progress,
        )
    except RuntimeError as exc:
        raise BootstrapError(f"install torch=={plan.torch_version}", 1) from exc


def torch_index_from_base(base: str | None, cuda: str) -> str:
    """Return a concrete PyTorch wheel index for *cuda*.

    User settings store a base URL such as ``https://.../pytorch/whl``.
    If someone has already entered a concrete ``.../cu124`` URL, normalize it
    by replacing that suffix with the requested CUDA suffix.
    """
    clean = (base or "").strip().rstrip("/") or DEFAULT_TORCH_INDEX_BASE
    parts = clean.rsplit("/", 1)
    if len(parts) == 2 and parts[1].startswith("cu") and parts[1][2:].isdigit():
        clean = parts[0]
    return f"{clean}/{cuda}"


def torch_index_candidates(
    base: str | None,
    cuda: str,
) -> list[str]:
    """Concrete torch indexes, user-configured source first, then built-ins."""
    indexes: list[str] = []
    bases = []
    if base and base.strip():
        bases.append(base.strip())
    bases.extend(FALLBACK_TORCH_INDEX_BASES)
    for candidate_base in bases:
        index = torch_index_from_base(candidate_base, cuda)
        if index not in indexes:
            indexes.append(index)
    return indexes


def _with_index_url(args: list[str], index_url: str) -> list[str]:
    out = list(args)
    for i, arg in enumerate(out):
        if arg == "--index-url" and i + 1 < len(out):
            out[i + 1] = index_url
            return out
        if arg.startswith("--index-url="):
            out[i] = f"--index-url={index_url}"
            return out
    return [*out, "--index-url", index_url]


def pip_install_with_torch_index_fallback(
    plan: BootstrapPlanLike,
    args: list[str],
    *,
    step: str,
    progress: ProgressCallback | None = None,
) -> None:
    """Install from PyTorch wheel indexes, trying the next source on failure."""
    indexes = torch_index_candidates(plan.torch_index_base, plan.cuda_version)
    errors: list[str] = []
    for idx, index_url in enumerate(indexes):
        current_step = step if idx == 0 else f"{step} (source {idx + 1}/{len(indexes)})"
        try:
            _uv.pip_install(
                plan.venv_python,
                _with_index_url(args, index_url),
                step=current_step,
                progress=progress,
            )
            return
        except RuntimeError as exc:
            errors.append(f"{index_url}: {exc}")
            if idx + 1 < len(indexes) and progress is not None:
                progress(f"{current_step} failed; trying {indexes[idx + 1]}")
    detail = "\n".join(errors[-4:])
    raise RuntimeError(f"{step} failed on all PyTorch indexes:\n{detail}")


def cleanup_partial(target: Path, repo_url: str) -> None:
    """Remove a half-installed checkout so the user can retry.

    Git pack files inside ``.git/objects/pack/*.idx`` are written read-only
    on Windows, so the default ``shutil.rmtree`` raises PermissionError on
    them. Hook ``onexc`` (Python 3.12+) / ``onerror`` to flip the read-only
    bit and retry, otherwise the user gets stuck in a 409 loop on every
    reinstall.
    """
    if not target.exists():
        return
    validate_destructive_cleanup_target(target, allow_managed_backend=True)
    if not is_managed_partial_install(target, repo_url):
        raise ValueError(
            f"refusing to delete backend without a matching install marker: {target}"
        )

    _rmtree_force(target)
    clear_install_marker(target, repo_url)


def _rmtree_force(target: Path) -> None:
    """Remove a verified directory, retrying read-only files on Windows."""

    def _force_writable(func: Any, path: str, _exc_info: Any) -> None:  # noqa: ANN401
        import stat as _stat  # noqa: PLC0415

        try:
            Path(path).chmod(_stat.S_IWRITE | _stat.S_IREAD)
            func(path)
        except OSError:
            pass

    if sys.version_info >= (3, 12):
        shutil.rmtree(target, onexc=_force_writable)
    else:
        shutil.rmtree(target, onerror=_force_writable)


def cleanup_managed_venvs(target: Path) -> None:
    """Remove only backend-owned venv directories, never backend source."""
    if _path_uses_link(target):
        raise ValueError(f"backend source cannot use links during cleanup: {target}")
    root = validate_backend_source_target(target)
    for name in (".venv", "venv"):
        venv = root / name
        if venv.is_symlink():
            venv.unlink(missing_ok=True)
            continue
        try:
            attrs = getattr(os.lstat(venv), "st_file_attributes", 0)
        except OSError:
            attrs = 0
        if attrs & 0x400:
            os.rmdir(venv)
            continue
        if venv.is_dir():
            _rmtree_force(venv)


def validate_destructive_cleanup_target(
    target: Path,
    *,
    allow_managed_backend: bool = False,
) -> Path:
    """Reject recursive deletion targets that overlap code or user data."""
    if _path_uses_link(target):
        raise ValueError(f"refusing to recursively delete linked path: {target}")
    resolved = target.expanduser().resolve()
    root = project_root().resolve()
    home = Path.home().resolve()
    if resolved.parent == resolved or resolved in {home, root}:
        raise ValueError(f"refusing to delete protected directory: {resolved}")
    try:
        root.relative_to(resolved)
    except ValueError:
        pass
    else:
        raise ValueError(f"refusing to delete a parent of the project: {resolved}")

    external_root = (root / "external").resolve()
    protected = {
        root / ".git",
        root / ".lorahub",
        root / ".venv",
        root / "configs",
        root / "datasets",
        root / "docker",
        root / "external",
        root / "lorahub",
        root / "models",
        root / "output",
        root / "runs",
        root / "scripts",
        root / "tests",
        root / "web",
    }
    if allow_managed_backend:
        try:
            relative_backend = resolved.relative_to(external_root)
        except ValueError:
            pass
        else:
            # Cloned backends are direct children of external/. A marker is
            # checked by cleanup_partial before any removal occurs.
            if len(relative_backend.parts) == 1:
                protected.remove(root / "external")
    for env_name in ("LORAHUB_DATASETS_ROOT", "LORAHUB_MODELS_ROOT"):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            protected.add(Path(raw).expanduser().resolve())
    for item in protected:
        item = item.resolve()
        try:
            resolved.relative_to(item)
            target_inside = True
        except ValueError:
            target_inside = False
        try:
            item.relative_to(resolved)
            contains_protected = True
        except ValueError:
            contains_protected = False
        if target_inside or contains_protected:
            raise ValueError(f"refusing to delete protected directory: {item}")
    return resolved


def validate_backend_source_target(target: Path) -> Path:
    """Ensure an installer cannot treat an application or data root as a backend."""
    if _path_uses_link(target):
        raise ValueError(f"backend target cannot use links: {target}")
    resolved = target.expanduser().resolve()
    data_root = project_root().resolve()
    code_root = Path(__file__).resolve().parents[4]
    exact_roots = {Path.home().resolve(), data_root, code_root}
    if resolved.parent == resolved or resolved in exact_roots:
        raise ValueError(f"backend target overlaps a protected root: {resolved}")

    protected_data = {
        data_root / "configs",
        data_root / "datasets",
        data_root / "models",
        data_root / "output",
        data_root / "runs",
    }
    for env_name in ("LORAHUB_DATASETS_ROOT", "LORAHUB_MODELS_ROOT"):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            protected_data.add(Path(raw).expanduser().resolve())
    for item in protected_data:
        try:
            resolved.relative_to(item.resolve())
        except ValueError:
            continue
        raise ValueError(f"backend target overlaps user data: {item.resolve()}")
    return resolved


def _is_link_path(path: Path) -> bool:
    try:
        attrs = getattr(os.lstat(path), "st_file_attributes", 0)
    except OSError:
        return False
    return path.is_symlink() or bool(
        attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _path_uses_link(path: Path) -> bool:
    current = path.expanduser().absolute()
    while True:
        if _is_link_path(current):
            return True
        if current.parent == current:
            return False
        current = current.parent


__all__ = [
    "DEFAULT_CUDA",
    "DEFAULT_DEPTH",
    "DEFAULT_TORCH_INDEX_BASE",
    "DEFAULT_TORCH",
    "DEFAULT_TORCHVISION",
    "FALLBACK_TORCH_INDEX_BASES",
    "BootstrapPlanLike",
    "ProgressCallback",
    "cleanup_partial",
    "cleanup_managed_venvs",
    "clear_install_marker",
    "clone_repo",
    "create_venv",
    "install_torch",
    "is_complete_git_repo",
    "is_managed_partial_install",
    "pip_install_with_torch_index_fallback",
    "run_step",
    "torch_index_candidates",
    "torch_index_from_base",
    "upgrade_pip",
    "validate_destructive_cleanup_target",
    "validate_backend_source_target",
]
