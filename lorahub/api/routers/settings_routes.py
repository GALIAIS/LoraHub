"""Settings GET/PUT.

The route file is named `settings_routes` to avoid colliding with the
sibling `lorahub.api.settings` module that holds the `Settings` dataclass and
`SettingsStore`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lorahub.api import app as app_module
from lorahub.api.settings import (
    VALID_BACKEND_IDS,
    Settings,
    probe_all_backends,
    probe_kohya_backend,
)

router = APIRouter(prefix="/api")

# Field names that hold credentials. They get masked in the GET response
# and treated specially on PUT (None == keep prior value, "" == clear).
# Keeping this list in one place so /api/settings audits stay grep-able.
_SECRET_FIELDS: tuple[str, ...] = (
    "huggingface_token",
    "wandb_api_key",
    "modelscope_token",
)


def _mask_secret(value: str | None) -> str | None:
    """Return a UI-safe preview of a secret. Empty/None stays None."""
    if not value:
        return None
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


class SettingsResponse(BaseModel):
    settings: dict[str, Any]
    backend: dict[str, Any]
    backends: dict[str, dict[str, Any]]
    path: str


class UpdateSettingsRequest(BaseModel):
    sd_scripts_path: str | None = None
    python_executable: str | None = None
    diffusion_pipe_repo_path: str | None = None
    diffusion_pipe_python: str | None = None
    default_backend: str | None = None
    tagger_device: str | None = None
    default_tagger: str | None = None
    max_concurrent_jobs: int | None = None
    github_proxy: str | None = None
    huggingface_endpoint: str | None = None
    modelscope_enabled: bool | None = None
    modelscope_token: str | None = None
    pypi_index_url: str | None = None
    download_proxy: str | None = None
    huggingface_token: str | None = None
    wandb_api_key: str | None = None
    terminal_unrestricted: bool | None = None
    terminal_command_timeout_s: int | None = None


def _norm(v: str | None) -> str | None:
    """Treat empty / whitespace-only strings as 'clear this field'."""
    if v is None:
        return None
    v = v.strip()
    return v or None


def _to_response(s: Settings, path: str) -> SettingsResponse:
    payload = s.to_dict()
    # Replace each secret field's raw value with (a) a masked preview
    # under the same key — so existing UI bindings still render — and
    # (b) a `has_<field>` boolean for forms that drive a "set / clear"
    # toggle. The PUT side treats None as "keep prior value" so the UI
    # can echo the masked preview back without leaking it.
    for name in _SECRET_FIELDS:
        raw = payload.get(name)
        payload[name] = _mask_secret(raw) if raw else None
        payload[f"has_{name}"] = bool(raw)
    return SettingsResponse(
        settings=payload,
        backend=probe_kohya_backend(s),
        backends=probe_all_backends(s),
        path=path,
    )


def _resolve_secret(
    incoming: str | None, current_value: str | None, masked_preview: str | None
) -> str | None:
    """Decide what to persist for a secret field given the PUT payload.

    Cases:
      * ``None`` (field absent / left unset by the client): keep prior.
      * Empty string (``""`` after stripping): clear the secret.
      * Echo of the masked preview (e.g. ``hf_X...abcd``): keep prior —
        the client round-tripped the GET response without re-typing.
      * Anything else: treat as a fresh secret and persist verbatim.
    """
    if incoming is None:
        return current_value
    stripped = incoming.strip()
    if not stripped:
        return None
    if masked_preview and stripped == masked_preview:
        return current_value
    return stripped


@router.get("/settings", response_model=SettingsResponse)
def get_settings() -> SettingsResponse:
    store = app_module._settings_store
    return _to_response(store.load(), str(store.path))


@router.put("/settings", response_model=SettingsResponse)
def update_settings(req: UpdateSettingsRequest) -> SettingsResponse:
    store = app_module._settings_store
    current = store.load()

    default_backend = (
        req.default_backend.strip()
        if req.default_backend is not None
        else current.default_backend
    )
    if default_backend not in VALID_BACKEND_IDS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"default_backend must be one of "
                f"{sorted(VALID_BACKEND_IDS)}, got {default_backend!r}"
            ),
        )

    tagger_device = (req.tagger_device or current.tagger_device or "auto").strip() or "auto"
    if tagger_device not in {"auto", "cpu", "cuda"}:
        raise HTTPException(
            status_code=422,
            detail=f"tagger_device must be auto/cpu/cuda, got {tagger_device!r}",
        )

    default_tagger = (req.default_tagger or current.default_tagger or "wd14").strip() or "wd14"
    if default_tagger not in {"wd14", "joytag"}:
        raise HTTPException(
            status_code=422,
            detail=f"default_tagger must be wd14/joytag, got {default_tagger!r}",
        )

    max_concurrent_jobs = (
        req.max_concurrent_jobs
        if req.max_concurrent_jobs is not None
        else current.max_concurrent_jobs
    )
    if not isinstance(max_concurrent_jobs, int) or max_concurrent_jobs < 1:
        raise HTTPException(
            status_code=422,
            detail=(
                "max_concurrent_jobs must be a positive integer; "
                f"got {max_concurrent_jobs!r}"
            ),
        )

    new = Settings(
        sd_scripts_path=_norm(req.sd_scripts_path),
        python_executable=_norm(req.python_executable),
        diffusion_pipe_repo_path=_norm(req.diffusion_pipe_repo_path),
        diffusion_pipe_python=_norm(req.diffusion_pipe_python),
        default_backend=default_backend,
        tagger_device=tagger_device,
        default_tagger=default_tagger,
        max_concurrent_jobs=max_concurrent_jobs,
        github_proxy=_norm(req.github_proxy),
        huggingface_endpoint=_norm(req.huggingface_endpoint),
        modelscope_enabled=(
            req.modelscope_enabled
            if req.modelscope_enabled is not None
            else current.modelscope_enabled
        ),
        modelscope_token=_resolve_secret(
            req.modelscope_token,
            current.modelscope_token,
            _mask_secret(current.modelscope_token),
        ),
        pypi_index_url=_norm(req.pypi_index_url),
        download_proxy=_norm(req.download_proxy),
        huggingface_token=_resolve_secret(
            req.huggingface_token,
            current.huggingface_token,
            _mask_secret(current.huggingface_token),
        ),
        wandb_api_key=_resolve_secret(
            req.wandb_api_key,
            current.wandb_api_key,
            _mask_secret(current.wandb_api_key),
        ),
        terminal_unrestricted=(
            req.terminal_unrestricted
            if req.terminal_unrestricted is not None
            else current.terminal_unrestricted
        ),
        terminal_command_timeout_s=(
            req.terminal_command_timeout_s
            if req.terminal_command_timeout_s is not None
            else current.terminal_command_timeout_s
        ),
        extra=current.extra,
    )
    store.save(new)
    return _to_response(new, str(store.path))
