"""Centralized network helpers for mirror/proxy configuration.

All HuggingFace and proxy-aware network calls should go through this module
to avoid the huggingface_hub "import-time caching" pitfall and os.environ
side-effect leakage.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


def _clean_endpoint(value: str | None) -> str | None:
    return (value or "").strip().rstrip("/") or None


def hf_endpoint(explicit: str | None = None) -> str | None:
    """Resolve the HF endpoint from explicit input, env, then settings."""
    if ep := _clean_endpoint(explicit):
        return ep
    if ep := _clean_endpoint(os.environ.get("HF_ENDPOINT")):
        return ep
    if ep := _clean_endpoint(os.environ.get("HUGGINGFACE_HUB_ENDPOINT")):
        return ep
    try:
        from lorahub.api import app as _app  # noqa: PLC0415

        settings = _app._settings_store.load()
        if ep := _clean_endpoint(settings.huggingface_endpoint):
            return ep
    except Exception:  # noqa: BLE001
        pass
    return None


def _hf_endpoint(explicit: str | None = None) -> str | None:
    """Backward-compatible alias for older internal imports."""
    return hf_endpoint(explicit)


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

    endpoint = hf_endpoint(kwargs.pop("endpoint", None))
    return HfApi(endpoint=endpoint, **kwargs)


def hf_download(
    repo_id: str,
    filename: str,
    *,
    endpoint: str | None = None,
    repo_type: str | None = None,
    revision: str | None = None,
    local_dir: str | None = None,
    tqdm_class: Any | None = None,
    **kwargs: Any,
) -> str:
    """Wrapper around hf_hub_download that injects the configured endpoint.

    Optional ``tqdm_class`` is forwarded straight through to
    ``hf_hub_download``. Callers that want UI-visible progress (e.g.
    the WD14 tagger first-load) plug in
    ``download_status.tqdm_class_for(repo_id, filename)`` so the
    download bytes show up in ``GET /api/tagging/wd14/download-status``.
    Most callers don't need it and can pass ``None`` to keep the
    default tqdm behaviour.
    """
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    ep = hf_endpoint(endpoint)
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
    if tqdm_class is not None:
        kw["tqdm_class"] = tqdm_class
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
    endpoint = hf_endpoint()
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


@contextmanager
def proxy_env(proxy: str | None) -> Iterator[None]:
    """Temporarily expose proxy vars for libraries that only read os.environ."""
    value = (proxy or "").strip()
    if not value:
        yield
        return

    names = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = value
        yield
    finally:
        for name, old in previous.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
