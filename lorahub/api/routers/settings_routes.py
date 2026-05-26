"""Settings GET/PUT.

The route file is named `settings_routes` to avoid colliding with the
sibling `lorahub.api.settings` module that holds the `Settings` dataclass and
`SettingsStore`.
"""

from __future__ import annotations

from dataclasses import fields, replace
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
    # GitLab / Gitea / Webhook PAT for the error report fan-out.
    # Same masking + "echo == keep prior" round-trip rules so an
    # unchanged token survives the form even though we never expose
    # it to the UI in plaintext after first save.
    "error_upstream_gitlab_token",
)

# Path-shaped fields use ``_norm`` (empty string == clear).
# Booleans / ints are written verbatim. Everything else falls through.
# Listing the path fields explicitly so adding a new ``foo_path`` setting
# can't silently get the "raw write" treatment.
_PATH_FIELDS: tuple[str, ...] = (
    "sd_scripts_path",
    "python_executable",
    "diffusion_pipe_repo_path",
    "diffusion_pipe_python",
    "anima_lora_repo_path",
    "anima_lora_python",
    "github_proxy",
    "huggingface_endpoint",
    "pypi_index_url",
    "torch_index_url",
    "download_proxy",
    "wandb_base_url",
)

# Free-form text fields that aren't path-shaped but should still
# round-trip as-is — empty string clears them, otherwise persist
# verbatim. Used by the error-upstream config block so users can
# blank a value to fall back to env vars.
_FREEFORM_TEXT_FIELDS: tuple[str, ...] = (
    "error_upstream_gitlab_base_url",
    "error_upstream_gitlab_repo",
    "error_upstream_webhook_url",
    "error_upstream_webhook_auth_header",
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
    """PUT /api/settings request body.

    Every settable Settings field appears here. Using ``exclude_unset``
    on the route side means a field omitted by the client keeps its
    prior value automatically; only explicitly-supplied keys cause a
    write. Adding a new Settings field is therefore a one-line change
    here — no parallel update of a hand-written mapping in the route
    handler.
    """

    sd_scripts_path: str | None = None
    python_executable: str | None = None
    diffusion_pipe_repo_path: str | None = None
    diffusion_pipe_python: str | None = None
    anima_lora_repo_path: str | None = None
    anima_lora_python: str | None = None
    default_backend: str | None = None
    tagger_device: str | None = None
    default_tagger: str | None = None
    max_concurrent_jobs: int | None = None
    github_proxy: str | None = None
    huggingface_endpoint: str | None = None
    modelscope_enabled: bool | None = None
    modelscope_token: str | None = None
    pypi_index_url: str | None = None
    torch_index_url: str | None = None
    download_proxy: str | None = None
    huggingface_token: str | None = None
    wandb_api_key: str | None = None
    wandb_base_url: str | None = None
    auto_resume_interrupted: bool | None = None
    auto_resume_max_attempts: int | None = None
    terminal_unrestricted: bool | None = None
    terminal_command_timeout_s: int | None = None
    # Error report fan-out — see Settings.error_upstream_*.
    # All but the channel + auto-severity are free-form strings.
    # Empty string == clear (fall back to env var for the token).
    error_upstream_channel: str | None = None
    error_upstream_gitlab_base_url: str | None = None
    error_upstream_gitlab_repo: str | None = None
    error_upstream_gitlab_token: str | None = None
    error_upstream_webhook_url: str | None = None
    error_upstream_webhook_auth_header: str | None = None
    error_upstream_auto_severity: str | None = None


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


def _validate_choice(name: str, value: str, allowed: set[str]) -> str:
    """Reject anything outside the allowed set with a structured 422."""
    if value not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"{name} must be one of {sorted(allowed)}, got {value!r}",
        )
    return value


@router.put("/settings", response_model=SettingsResponse)
def update_settings(req: UpdateSettingsRequest) -> SettingsResponse:
    store = app_module._settings_store
    current = store.load()

    # Resolve each request field into the value we'll pass to ``replace``.
    # ``model_dump(exclude_unset=True)`` only includes keys the client
    # actually sent, so omitted fields fall through to ``current.<field>``
    # automatically.
    sent = req.model_dump(exclude_unset=True)
    updates: dict[str, Any] = {}
    known_fields = {f.name for f in fields(Settings)}

    for key, raw in sent.items():
        if key not in known_fields:
            continue
        if key in _SECRET_FIELDS:
            updates[key] = _resolve_secret(
                raw, getattr(current, key), _mask_secret(getattr(current, key))
            )
            continue
        if key in _PATH_FIELDS:
            updates[key] = _norm(raw)
            continue
        if key in _FREEFORM_TEXT_FIELDS:
            # Strip whitespace; empty becomes "" so the user can
            # explicitly clear a value (vs paths where empty -> None).
            # The dataclass default is "" anyway, so writing "" is
            # equivalent to "fall back to env var / channel default".
            if raw is None:
                # Pydantic's exclude_unset already filters "key not
                # in payload" — a literal None means the user sent
                # null, which we treat as "clear".
                updates[key] = ""
            else:
                updates[key] = str(raw).strip()
            continue
        if key == "error_upstream_channel":
            updates[key] = _validate_choice(
                "error_upstream_channel",
                (raw or current.error_upstream_channel or "off").strip() or "off",
                {"off", "gitlab", "gitea", "webhook"},
            )
            continue
        if key == "error_upstream_auto_severity":
            updates[key] = _validate_choice(
                "error_upstream_auto_severity",
                (raw or current.error_upstream_auto_severity or "error").strip()
                or "error",
                {"off", "error", "all"},
            )
            continue
        # Choice fields — validated then written verbatim. None means
        # "client sent null, treat as clear/default" except for booleans
        # / ints which never carry the path-style "empty == clear" idiom.
        if key == "default_backend":
            updates[key] = _validate_choice(
                "default_backend",
                (raw or current.default_backend).strip() or current.default_backend,
                VALID_BACKEND_IDS,
            )
            continue
        if key == "tagger_device":
            updates[key] = _validate_choice(
                "tagger_device",
                (raw or current.tagger_device or "auto").strip() or "auto",
                {"auto", "cpu", "cuda"},
            )
            continue
        if key == "default_tagger":
            updates[key] = _validate_choice(
                "default_tagger",
                (raw or current.default_tagger or "wd14").strip() or "wd14",
                {"wd14", "joytag"},
            )
            continue
        if key == "max_concurrent_jobs":
            value = raw if raw is not None else current.max_concurrent_jobs
            if not isinstance(value, int) or value < 1:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "max_concurrent_jobs must be a positive integer; "
                        f"got {value!r}"
                    ),
                )
            updates[key] = value
            continue
        if key == "auto_resume_max_attempts":
            value = raw if raw is not None else current.auto_resume_max_attempts
            if not isinstance(value, int) or value < 1:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "auto_resume_max_attempts must be a positive integer; "
                        f"got {value!r}"
                    ),
                )
            updates[key] = value
            continue
        if key == "terminal_command_timeout_s":
            value = raw if raw is not None else current.terminal_command_timeout_s
            if not isinstance(value, int) or value < 1:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "terminal_command_timeout_s must be a positive integer; "
                        f"got {value!r}"
                    ),
                )
            updates[key] = value
            continue
        # Bools and the rest pass through as-is.
        updates[key] = raw

    new = replace(current, **updates)
    store.save(new)
    return _to_response(new, str(store.path))
