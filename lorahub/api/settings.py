"""User-level Settings store.

Persists workbench-wide defaults (per-backend checkout + python, default
backend choice, tagger device) to a single JSON file under the user's data
directory. Env vars (LORAHUB_*) still take precedence at config-launch
time -- Settings are the *fallback* defaults the UI lets users override
without touching their shell config.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from platformdirs import user_data_path

from lorahub.core.backends.registry import known_ids


@dataclass
class Settings:
    """User-configurable defaults applied when a config doesn't specify them."""

    # kohya backend
    sd_scripts_path: str | None = None
    python_executable: str | None = None

    # diffusion-pipe backend
    diffusion_pipe_repo_path: str | None = None
    diffusion_pipe_python: str | None = None

    # anima_lora backend (vendored under external/anima_lora — no clone
    # step, only the python interpreter is user-configurable. The repo
    # path field exists for parity / dev override only).
    anima_lora_repo_path: str | None = None
    anima_lora_python: str | None = None

    # Which backend the UI defaults to when starting a fresh config.
    default_backend: str = "kohya"

    tagger_device: str = "auto"  # "auto" | "cpu" | "cuda"
    default_tagger: str = "wd14"  # "wd14" | "joytag"

    # Maximum number of training jobs the scheduler can run concurrently.
    # Slot id == GPU index, so this maps directly onto `CUDA_VISIBLE_DEVICES`
    # for each worker. Default 1 preserves the historical single-slot queue.
    # NOTE: changing this value does NOT hot-reload the live scheduler — it
    # is read once at server startup. Restart `lorahub serve` for the new
    # value to take effect.
    max_concurrent_jobs: int = 1

    # --- Network acceleration ---
    # Optional GitHub mirror prefix (e.g. "https://gh-proxy.org") rewriting
    # `https://github.com/...` URLs at clone time. Leave empty for direct.
    github_proxy: str | None = None
    # HuggingFace endpoint mirror (set as HF_ENDPOINT env var on subprocess
    # launches). Leave empty for the official site.
    huggingface_endpoint: str | None = None
    # Optional HuggingFace API token (HF_TOKEN). Required for gated repos
    # (e.g. Black Forest Labs Flux). Stored as plain text; the JSON file
    # is in the user data directory and not synced anywhere by lorahub.
    huggingface_token: str | None = None
    # Optional Weights & Biases API key. Forwarded as WANDB_API_KEY to
    # training subprocesses so users don't need to `export` it from a
    # shell before each run.
    wandb_api_key: str | None = None
    # Optional W&B base URL for self-hosted W&B Server. Empty/None
    # targets wandb.ai (SaaS). Forwarded both as ``WANDB_BASE_URL`` to
    # subprocesses and as ``wandb.Api(overrides={"base_url": ...})`` for
    # the read-only proxy that powers "训练分析 → W&B".
    wandb_base_url: str | None = None
    # When true, downloads default to ModelScope where applicable.
    modelscope_enabled: bool = False
    # Optional access token for private ModelScope models.
    modelscope_token: str | None = None
    # Optional PyPI index URL (e.g. https://pypi.tuna.tsinghua.edu.cn/simple).
    # Used by `uv pip install` when the caller hasn't pinned its own
    # --index-url. Leaves wheel-store URLs (download.pytorch.org/...) alone.
    pypi_index_url: str | None = None

    # Optional PyTorch wheel index mirror. Leave empty to use the official
    # ``download.pytorch.org/whl/{cuda}`` (correct outside China). Inside
    # China the official index is unreachable; popular mirrors:
    #   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/whl
    #   - https://mirrors.aliyun.com/pytorch-wheels
    # When set, the value is suffixed with ``/{cuda_version}`` automatically
    # by ``BootstrapPlan.torch_index`` so the user only configures the base
    # URL. Setting this also covers xformers (kohya) since xformers wheels
    # live on the same index.
    torch_index_url: str | None = None

    # Optional SOCKS5/HTTP proxy for model downloads (HuggingFace, ModelScope).
    # Format: socks5h://user:pass@host:port or http://user:pass@host:port
    download_proxy: str | None = None

    # When true, on server startup any job that was running before the
    # process died and has a usable checkpoint on disk is automatically
    # re-launched via the resume flow. Off by default — a corrupt run
    # that flaps between kill -9 and resume could otherwise loop. The
    # per-lineage cap below bounds blast radius even when on.
    # Per-job opt-out via metadata.auto_resume = False; per-job opt-in
    # via metadata.auto_resume = True is honored even with this flag off.
    auto_resume_interrupted: bool = False
    # Maximum consecutive auto-resume attempts per lineage. Tracked via
    # metadata.auto_resume_attempts on each resumed JobRecord.
    auto_resume_max_attempts: int = 3

    # --- Terminal ---
    # In-app terminal restricts commands to a small whitelist of Python
    # package management entry points (pip / uv / python) by default so
    # a stray paste / prompt injection can't fire `rm -rf` against the
    # user's tree. Flip to True only if you actually need a free shell;
    # the UI surfaces this as a single toggle.
    terminal_unrestricted: bool = False
    # Per-command timeout (seconds). pip install for a heavy wheel can
    # take a while, so the default sits at 10 minutes; users can crank
    # it up locally if they're upgrading torch on a slow link.
    terminal_command_timeout_s: int = 600

    # --- Error report fan-out ---
    # Optional remote sink for the local error registry. Default
    # ``off`` means *nothing* leaves the box; the user has to opt in
    # explicitly from Settings → 错误上报. ``gitlab`` opens / appends
    # GitLab issues with fingerprint-based de-dupe; ``gitea`` is the
    # same contract over Gitea's v1 API (git.galiais.com defaults to
    # this); ``webhook`` POSTs the redacted payload to an arbitrary URL.
    error_upstream_channel: str = "off"  # "off" | "gitlab" | "gitea" | "webhook"
    # GitLab / Gitea share these three fields; the channel discriminator
    # picks the matching API dialect at sink-construction time. Pre-fill
    # the project-level defaults so a fresh install only needs the
    # token before "测试连通" works.
    error_upstream_gitlab_base_url: str = "https://git.galiais.com"
    error_upstream_gitlab_repo: str = "Shiro/LoraHubReport"
    # Token defaults to empty — the only acceptable place for a real
    # token is the user's local settings.json or the LORAHUB_GITEA_TOKEN
    # env var, never source code (it would leak via ``git push`` and
    # never be revocable from a third-party fork).
    error_upstream_gitlab_token: str = ""
    # Webhook fields (only meaningful when channel == "webhook")
    error_upstream_webhook_url: str = ""
    error_upstream_webhook_auth_header: str = ""
    # Auto-send threshold: ``off`` keeps every report queued for manual
    # send; ``error`` auto-pushes severity ≥ error; ``all`` pushes
    # everything (matches the user's earlier choice in the AskUser flow
    # where the default was "error and above").
    error_upstream_auto_severity: str = "error"  # "off" | "error" | "all"

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["extra"] = {k: v for k, v in data.items() if k not in known}
        return cls(**kwargs)


def default_settings_path() -> Path:
    """`<platformdirs user_data>/lorahub/lorahub/settings.json`."""
    return user_data_path("lorahub", "lorahub") / "settings.json"


class SettingsStore:
    """Tiny JSON-backed store. Atomic writes via a sibling temp file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = (path or default_settings_path()).resolve()

    def load(self) -> Settings:
        if not self.path.is_file():
            return Settings()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return Settings()
        if not isinstance(data, dict):
            return Settings()
        return Settings.from_dict(data)

    def save(self, settings: Settings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(settings.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.path)


# --------------------------------------------------------------------------- #
# Per-backend probes (pure read-only; never raise).
# --------------------------------------------------------------------------- #


def _venv_site_packages(python: Path) -> Path | None:
    """Walk to the venv's ``site-packages`` from its python interpreter.

    POSIX: ``<venv>/lib/python<X.Y>/site-packages``.
    Windows: ``<venv>/Lib/site-packages``.

    Returns None when the layout doesn't match either (rare — distro
    splits, build-from-source pythons). Callers treat None as "skip
    this check" rather than "broken venv".
    """
    venv_root = python.parent
    # POSIX: <venv>/bin/python; Windows: <venv>/Scripts/python.exe
    if venv_root.name in ("bin", "Scripts"):
        venv_root = venv_root.parent
    win_layout = venv_root / "Lib" / "site-packages"
    if win_layout.is_dir():
        return win_layout
    posix_lib = venv_root / "lib"
    if posix_lib.is_dir():
        for child in posix_lib.iterdir():
            if child.is_dir() and child.name.startswith("python"):
                cand = child / "site-packages"
                if cand.is_dir():
                    return cand
    return None


def probe_kohya_backend(settings: Settings) -> dict[str, Any]:
    """Inspect whether the configured kohya checkout looks usable."""
    from lorahub.core.backends._common.bootstrap import (  # noqa: PLC0415
        check_requirements,
    )
    from lorahub.core.backends.kohya.bootstrap import (  # noqa: PLC0415
        _ENV_PYTHON,
        _ENV_SD_SCRIPTS,
        _REQUIRED_SCRIPTS,
        _venv_python,
        default_sd_scripts_path,
    )

    sd_raw = (
        os.environ.get(_ENV_SD_SCRIPTS)
        or settings.sd_scripts_path
        or str(default_sd_scripts_path())
    )
    sd_path = Path(sd_raw).expanduser()

    sd_ok = sd_path.is_dir()
    missing = (
        [s for s in _REQUIRED_SCRIPTS if not (sd_path / s).is_file()]
        if sd_ok
        else list(_REQUIRED_SCRIPTS)
    )

    py_raw = (
        os.environ.get(_ENV_PYTHON)
        or settings.python_executable
        or (str(_venv_python(sd_path)) if _venv_python(sd_path) else None)
    )
    py_path = Path(py_raw).expanduser() if py_raw else None
    py_ok = bool(py_path and py_path.is_file())

    # Check requirements.txt completeness when repo and python are both OK.
    requirements_ok = True
    missing_requirements: list[str] = []
    if sd_ok and not missing and py_ok and py_path:
        req_file = sd_path / "requirements.txt"
        missing_requirements = check_requirements(py_path, req_file)
        requirements_ok = len(missing_requirements) == 0

    if os.environ.get(_ENV_SD_SCRIPTS):
        source = "env"
    elif settings.sd_scripts_path:
        source = "settings"
    else:
        source = "default"

    return {
        "id": "kohya",
        "sd_scripts_path": str(sd_path),
        "sd_scripts_ok": sd_ok and not missing,
        "missing_scripts": missing,
        "python": str(py_path) if py_path else None,
        "python_ok": py_ok,
        "venv_detected": _venv_python(sd_path) is not None if sd_ok else False,
        "requirements_ok": requirements_ok,
        "missing_requirements": missing_requirements,
        "ready": (sd_ok and not missing) and py_ok and requirements_ok,
        "source": source,
    }


def probe_diffusion_pipe_backend(settings: Settings) -> dict[str, Any]:
    """Inspect whether the configured diffusion-pipe checkout looks usable."""
    from lorahub.core.backends._common.bootstrap import (  # noqa: PLC0415
        check_requirements,
    )
    from lorahub.core.backends.diffusion_pipe.bootstrap import (  # noqa: PLC0415
        _ENV_PYTHON,
        _ENV_REPO,
        _REQUIRED_FILES,
        _venv_python,
        default_repo_path,
    )

    repo_raw = (
        os.environ.get(_ENV_REPO)
        or settings.diffusion_pipe_repo_path
        or str(default_repo_path())
    )
    repo_path = Path(repo_raw).expanduser()

    repo_ok = repo_path.is_dir()
    missing = (
        [f for f in _REQUIRED_FILES if not (repo_path / f).is_file()]
        if repo_ok
        else list(_REQUIRED_FILES)
    )

    py_raw = (
        os.environ.get(_ENV_PYTHON)
        or settings.diffusion_pipe_python
        or (str(_venv_python(repo_path)) if _venv_python(repo_path) else None)
    )
    py_path = Path(py_raw).expanduser() if py_raw else None
    py_ok = bool(py_path and py_path.is_file())

    # Check requirements.txt completeness when repo and python are both OK.
    requirements_ok = True
    missing_requirements: list[str] = []
    if repo_ok and not missing and py_ok and py_path:
        req_file = repo_path / "requirements.txt"
        missing_requirements = check_requirements(
            py_path, req_file, skip_patterns=("deepspeed",)
        )
        requirements_ok = len(missing_requirements) == 0

    if os.environ.get(_ENV_REPO):
        source = "env"
    elif settings.diffusion_pipe_repo_path:
        source = "settings"
    else:
        source = "default"

    return {
        "id": "diffusion-pipe",
        "repo_path": str(repo_path),
        "repo_ok": repo_ok and not missing,
        "missing_files": missing,
        "python": str(py_path) if py_path else None,
        "python_ok": py_ok,
        "venv_detected": _venv_python(repo_path) is not None if repo_ok else False,
        "requirements_ok": requirements_ok,
        "missing_requirements": missing_requirements,
        "ready": (repo_ok and not missing) and py_ok and requirements_ok,
        "source": source,
    }


def probe_anima_lora_backend(settings: Settings) -> dict[str, Any]:
    """Inspect whether the vendored anima_lora copy is usable.

    Differences from the kohya / dp probe:
      * No clone step — the source ships under ``external/anima_lora``,
        so the repo path is auto-resolved from the LoraHub source tree
        and the user normally doesn't override it.
      * No requirements.txt check — anima_lora needs torch 2.11 nightly
        + CUDA 13 in its own venv, which we don't manage. The python
        check just confirms the interpreter exists; package presence
        is the user's responsibility.
    """
    from lorahub.core.backends.anima_lora.bootstrap import (  # noqa: PLC0415
        _ENV_PYTHON,
        _ENV_REPO,
        _REQUIRED_FILES,
        _venv_python,
        default_repo_path,
    )
    from lorahub.core.backends.anima_lora.models import (  # noqa: PLC0415
        missing_files as _anima_missing_models,
    )
    from lorahub.core.backends.anima_lora import msvc as _anima_msvc  # noqa: PLC0415

    repo_raw = (
        os.environ.get(_ENV_REPO)
        or settings.anima_lora_repo_path
        or str(default_repo_path())
    )
    repo_path = Path(repo_raw).expanduser()

    repo_ok = repo_path.is_dir()
    missing = (
        [f for f in _REQUIRED_FILES if not (repo_path / f).is_file()]
        if repo_ok
        else list(_REQUIRED_FILES)
    )

    py_raw = (
        os.environ.get(_ENV_PYTHON)
        or settings.anima_lora_python
        or (str(_venv_python(repo_path)) if _venv_python(repo_path) else None)
    )
    py_path = Path(py_raw).expanduser() if py_raw else None
    # Only count as "python ok" when the resolved interpreter lives
    # inside the dedicated .venv (or the user explicitly pointed at
    # one via env / settings). If both fall back to None we deliberately
    # don't try sys.executable here — a misleading "ready=true" would
    # let the UI hide the install prompt even though anima_lora cannot
    # actually run on the LoraHub main interpreter.
    py_ok = bool(py_path and py_path.is_file())

    if os.environ.get(_ENV_REPO):
        source = "env"
    elif settings.anima_lora_repo_path:
        source = "settings"
    else:
        # ``default`` matches the kohya / diffusion-pipe probe shape so
        # the UI doesn't have to switch on backend id when rendering
        # the source label. The repo is still vendored (the path comes
        # from default_repo_path() which walks up the source tree) —
        # the field name is just signalling "no override active".
        source = "default"

    # Cheap "key packages installed?" check — sniff for the import
    # entry-point files inside the venv's site-packages. Avoids the
    # cost of spawning a subprocess to ``pip freeze`` on every probe
    # while still catching the half-finished ``uv sync`` case
    # (interpreter exists but deps weren't materialised). When the
    # venv layout is unfamiliar (rare — Linux distro python_lib hash
    # variations) we degrade to "ok=True" so a probing user isn't
    # blocked by a layout-detection bug.
    _anima_required_pkgs = ("torch", "accelerate", "diffusers", "safetensors")
    requirements_ok = True
    missing_pkgs: list[str] = []
    if py_ok and py_path is not None:
        site_pkgs = _venv_site_packages(py_path)
        if site_pkgs is not None:
            for pkg in _anima_required_pkgs:
                if not (site_pkgs / pkg).exists():
                    missing_pkgs.append(pkg)
            requirements_ok = not missing_pkgs

    # MSVC Build Tools detection — only meaningful on Windows. anima's
    # torch_compile path needs triton-windows -> cl.exe; without it
    # the trainer crashes inside Inductor codegen with a TypeError
    # that has nothing to do with the user's config. The install
    # panel uses these fields to surface a one-click installer CTA.
    msvc = _anima_msvc.detect()
    import sys as _sys  # noqa: PLC0415
    msvc_payload = {
        "platform_relevant": _sys.platform == "win32",
        "ok": msvc.installed,
        "cl_path": msvc.cl_path,
        "msvc_version": msvc.msvc_version,
        "reason": msvc.reason,
        "winget_available": _anima_msvc.winget_available(),
    }

    return {
        "id": "anima_lora",
        "repo_path": str(repo_path),
        "repo_ok": repo_ok and not missing,
        "missing_files": missing,
        "python": str(py_path) if py_path else None,
        "python_ok": py_ok,
        "venv_detected": _venv_python(repo_path) is not None if repo_ok else False,
        # Vendored: no LoraHub-managed requirements.txt to diff against.
        # We sniff site-packages for the four import-entry packages anima
        # absolutely needs (torch / accelerate / diffusers / safetensors)
        # so a half-finished ``uv sync`` (interpreter present, deps not
        # materialised) shows up as ``ready=false`` instead of a
        # misleading green tick.
        "requirements_ok": requirements_ok,
        "missing_requirements": missing_pkgs,
        # Anima base / TE / VAE checkpoints are downloaded separately
        # (multi-GB) — we surface presence here so the install panel
        # can show a "Download models" CTA after `uv sync` finishes.
        "missing_models": _anima_missing_models(),
        "models_ok": not _anima_missing_models(),
        # MSVC Build Tools — Windows-only; the install panel renders a
        # one-click ``winget install`` button when ``msvc.ok`` is False
        # on Windows (and hides the section everywhere else).
        "msvc": msvc_payload,
        # `ready` here means "we can dispatch to the backend" — repo
        # files present + python interpreter resolvable + key packages
        # installed. Whether the interpreter has the *right* torch
        # nightly version is not knowable cheaply; the runner surfaces
        # that as a launch error.
        "ready": (repo_ok and not missing) and py_ok and requirements_ok,
        "source": source,
    }


def probe_all_backends(settings: Settings) -> dict[str, dict[str, Any]]:
    """Return a probe payload for every backend in the registry."""
    return {
        "kohya": probe_kohya_backend(settings),
        "diffusion-pipe": probe_diffusion_pipe_backend(settings),
        "anima_lora": probe_anima_lora_backend(settings),
    }


def probe_backend(settings: Settings) -> dict[str, Any]:
    """Backwards-compatible probe -- returns the kohya status payload.

    Kept so legacy callers (notably `/api/health`) keep working without
    every site having to opt in to the multi-backend payload at once.
    """
    return probe_kohya_backend(settings)


VALID_BACKEND_IDS: frozenset[str] = frozenset(known_ids())


# --------------------------------------------------------------------------- #
# Network acceleration helpers (consumed by the installer + subprocess launch)
# --------------------------------------------------------------------------- #


def apply_github_proxy(url: str, proxy: str | None) -> str:
    """Rewrite a github.com URL through `proxy` (e.g. "https://gh-proxy.org").

    No-op when `proxy` is empty or when the URL doesn't point at github.com.
    Strips the trailing slash from `proxy` so callers can paste either form.
    """
    if not proxy:
        return url
    p = proxy.strip().rstrip("/")
    if not p:
        return url
    if not url.startswith(("https://github.com/", "http://github.com/")):
        return url
    return f"{p}/{url}"


def env_overrides(settings: Settings) -> dict[str, str]:
    """Environment variables to inject into subprocesses started by lorahub.

    Currently routes HuggingFace traffic through a mirror when configured.
    The shell environment still wins — these are defaults applied only when
    the user hasn't set the variable themselves.
    """
    overrides: dict[str, str] = {}
    hf = (settings.huggingface_endpoint or "").strip().rstrip("/")
    if hf and "HF_ENDPOINT" not in os.environ:
        overrides["HF_ENDPOINT"] = hf
        # huggingface_hub also reads HUGGINGFACE_HUB_ENDPOINT historically.
        overrides["HUGGINGFACE_HUB_ENDPOINT"] = hf
    if settings.modelscope_token and "MODELSCOPE_API_TOKEN" not in os.environ:
        overrides["MODELSCOPE_API_TOKEN"] = settings.modelscope_token
    if settings.huggingface_token and "HF_TOKEN" not in os.environ:
        # huggingface_hub reads HF_TOKEN preferentially; the legacy
        # HUGGING_FACE_HUB_TOKEN is honored as a fallback for older
        # clients (some upstream training scripts still look for it).
        overrides["HF_TOKEN"] = settings.huggingface_token
        overrides["HUGGING_FACE_HUB_TOKEN"] = settings.huggingface_token
    if settings.wandb_api_key and "WANDB_API_KEY" not in os.environ:
        overrides["WANDB_API_KEY"] = settings.wandb_api_key
    if settings.wandb_base_url and "WANDB_BASE_URL" not in os.environ:
        overrides["WANDB_BASE_URL"] = settings.wandb_base_url
    return overrides


__all__ = [
    "Settings",
    "SettingsStore",
    "VALID_BACKEND_IDS",
    "apply_github_proxy",
    "default_settings_path",
    "env_overrides",
    "probe_all_backends",
    "probe_backend",
    "probe_diffusion_pipe_backend",
    "probe_kohya_backend",
]
