"""Centralized network helpers for mirror/proxy configuration.

All HuggingFace and proxy-aware network calls should go through this module
to avoid the huggingface_hub "import-time caching" pitfall and os.environ
side-effect leakage.
"""

from __future__ import annotations

import os
from typing import Any


def _hf_endpoint() -> str | None:
    """Read the configured HF endpoint from settings (lazy import to avoid cycles)."""
    try:
        from lorahub.api import app as _app  # noqa: PLC0415

        settings = _app._settings_store.load()
        return (settings.huggingface_endpoint or "").strip().rstrip("/") or None
    except Exception:  # noqa: BLE001
        return os.environ.get("HF_ENDPOINT") or None


def _download_proxy() -> str | None:
    """Read the configured download proxy from settings."""
    try:
        from lorahub.api import app as _app  # noqa: PLC0415

        settings = _app._settings_store.load()
        return (settings.download_proxy or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


def hf_api(**kwargs: Any) -> Any:
    """Create an HfApi instance with the configured endpoint."""
    from huggingface_hub import HfApi  # noqa: PLC0415

    endpoint = kwargs.pop("endpoint", None) or _hf_endpoint()
    return HfApi(endpoint=endpoint, **kwargs)


def hf_download(
    repo_id: str,
    filename: str,
    *,
    endpoint: str | None = None,
    repo_type: str | None = None,
    revision: str | None = None,
    local_dir: str | None = None,
    **kwargs: Any,
) -> str:
    """Wrapper around hf_hub_download that injects the configured endpoint."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    ep = endpoint or _hf_endpoint()
    kw: dict[str, Any] = {
        "repo_id": repo_id,
        "filename": filename,
        **kwargs,
    }
    if ep:
        kw["endpoint"] = ep
    if repo_type:
        kw["repo_type"] = repo_type
    if revision:
        kw["revision"] = revision
    if local_dir:
        kw["local_dir"] = local_dir
    return hf_hub_download(**kw)


def subprocess_env(
    *,
    include_proxy: bool = False,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build an env dict for subprocess calls with network settings injected.

    Does NOT mutate os.environ. Returns a fresh dict based on the current
    environ plus any overrides from Settings.
    """
    env = dict(os.environ)
    endpoint = _hf_endpoint()
    if endpoint:
        env["HF_ENDPOINT"] = endpoint
        env["HUGGINGFACE_HUB_ENDPOINT"] = endpoint
    if include_proxy:
        proxy = _download_proxy()
        if proxy:
            env["HTTPS_PROXY"] = proxy
            env["HTTP_PROXY"] = proxy
            env["ALL_PROXY"] = proxy
    if extra:
        env.update(extra)
    return env
