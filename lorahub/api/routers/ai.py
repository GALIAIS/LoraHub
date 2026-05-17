"""HTTP API for the AI provider subsystem.

Three concerns:

1. **Catalogue** — `GET /api/ai/providers` lists every registered
   provider's descriptor (id / display name / models / docs URL /
   whether base_url is user-configurable).
2. **Credential CRUD** — get/list/upsert/delete rows in
   `runs/ai_credentials.sqlite`. The `api_key` is masked on read so a
   shoulder-surfer can't pull it from the dashboard.
3. **Live calls** — `POST /api/ai/test` does a 1-token "say hi" round
   trip to confirm the credential works; `POST /api/ai/chat` is the
   general-purpose proxy other features (vision tagging, dataset
   analysis, caption rewrite, error diagnostics) call into.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from lorahub.api import app as app_module
from lorahub.api.ai_credentials_store import (
    AICredential,
    AICredentialStore,
    default_ai_credentials_path,
)
from lorahub.core.ai import list_providers
from lorahub.core.ai.dispatch import build_provider, load_provider
from lorahub.core.ai.provider_base import (
    ChatMessage,
    ChatOptions,
    ProviderError,
)

router = APIRouter(prefix="/api")
_log = logging.getLogger(__name__)


def _store() -> AICredentialStore:
    """Reach for the lifespan-initialised store on the app module.

    Test fixtures monkeypatch ``app_module._ai_credentials_store`` to a
    per-test SQLite file before issuing requests, so routers must look
    up the symbol dynamically rather than caching at import time.
    """
    store = getattr(app_module, "_ai_credentials_store", None)
    if store is None:
        store = AICredentialStore(default_ai_credentials_path())
        app_module._ai_credentials_store = store
    return store


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #


@router.get("/ai/providers")
def get_providers() -> dict[str, Any]:
    """Static catalogue + live credential status for every provider."""
    store = _store()
    creds = {c.provider: c for c in store.list()}
    out: list[dict[str, Any]] = []
    for d in list_providers():
        cred = creds.get(d.id)
        out.append(
            {
                "id": d.id,
                "name": d.name,
                "homepage": d.homepage,
                "docs_url": d.docs_url,
                "auth_help": d.auth_help,
                "default_base_url": d.default_base_url,
                "default_model": d.default_model,
                "custom_base_url": d.custom_base_url,
                "models": [
                    {"id": m.id, "label": m.label, "vision": m.vision, "context": m.context}
                    for m in d.models
                ],
                "configured": cred is not None and bool(cred.api_key),
                "enabled": cred.enabled if cred else False,
                "current_base_url": (cred.base_url if cred else None),
                "current_default_model": (cred.default_model if cred else None),
            }
        )
    return {"providers": out}


# --------------------------------------------------------------------------- #
# Credential CRUD
# --------------------------------------------------------------------------- #


def _mask_key(key: str | None) -> str | None:
    if not key:
        return None
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def _serialize_credential(cred: AICredential, *, reveal: bool = False) -> dict[str, Any]:
    return {
        "provider": cred.provider,
        "api_key": cred.api_key if reveal else _mask_key(cred.api_key),
        "api_key_set": bool(cred.api_key),
        "base_url": cred.base_url,
        "default_model": cred.default_model,
        "enabled": cred.enabled,
        "updated_at": cred.updated_at.isoformat() if cred.updated_at else None,
    }


class UpsertCredentialRequest(BaseModel):
    provider: str
    api_key: str | None = None
    base_url: str | None = None
    default_model: str | None = None
    enabled: bool = True


@router.get("/ai/credentials")
def list_credentials() -> dict[str, Any]:
    return {"credentials": [_serialize_credential(c) for c in _store().list()]}


@router.put("/ai/credentials")
def upsert_credential(req: UpsertCredentialRequest) -> dict[str, Any]:
    valid_ids = {d.id for d in list_providers()}
    if req.provider not in valid_ids:
        raise HTTPException(
            status_code=400,
            detail=f"unknown provider {req.provider!r}; expected one of {sorted(valid_ids)}",
        )
    cred = AICredential(
        provider=req.provider,
        api_key=(req.api_key or None),
        base_url=(req.base_url or None),
        default_model=(req.default_model or None),
        enabled=req.enabled,
    )
    _store().upsert(cred)
    return {"credential": _serialize_credential(_store().get(req.provider) or cred)}


@router.delete("/ai/credentials/{provider}")
def delete_credential(provider: str) -> dict[str, Any]:
    deleted = _store().delete(provider)
    return {"deleted": deleted, "provider": provider}


# --------------------------------------------------------------------------- #
# Live calls
# --------------------------------------------------------------------------- #


class TestRequest(BaseModel):
    provider: str
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None


@router.post("/ai/test")
def test_credential(req: TestRequest) -> dict[str, Any]:
    """Confirm a key works without persisting it.

    Useful for a "Test" button next to the API-key field — we build a
    transient AICredential from the request, send a tiny prompt, and
    return either ``{ok: true, model, sample}`` or ``{ok: false, error}``.
    No state is mutated.
    """
    valid_ids = {d.id for d in list_providers()}
    if req.provider not in valid_ids:
        raise HTTPException(
            status_code=400,
            detail=f"unknown provider {req.provider!r}",
        )
    # Use the request's overrides; fall back to the saved credential when
    # the user clicked Test on an existing row.
    saved = _store().get(req.provider)
    cred = AICredential(
        provider=req.provider,
        api_key=req.api_key or (saved.api_key if saved else None),
        base_url=req.base_url or (saved.base_url if saved else None),
        default_model=req.model or (saved.default_model if saved else None),
        enabled=True,
    )
    try:
        provider = build_provider(cred)
        result = provider.chat(
            [ChatMessage(role="user", content="Reply with the single word: ok.")],
            ChatOptions(
                model=req.model or (saved.default_model if saved else None),
                max_tokens=8,
                temperature=0.0,
                timeout_s=15.0,
            ),
        )
    except ProviderError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "status_code": exc.status_code,
            "retryable": exc.retryable,
        }
    return {
        "ok": True,
        "model": result.model,
        "sample": result.text[:200],
        "usage_input_tokens": result.usage_input_tokens,
        "usage_output_tokens": result.usage_output_tokens,
    }


class ChatRequest(BaseModel):
    provider: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = None
    response_format: str = "text"  # "text" | "json"
    stream: bool = False
    timeout_s: float = 60.0
    extra: dict[str, Any] = Field(default_factory=dict)


def _coerce_messages(raw: list[dict[str, Any]]) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    for m in raw:
        role = m.get("role")
        content = m.get("content")
        if role not in {"system", "user", "assistant"}:
            raise HTTPException(
                status_code=400,
                detail=f"invalid message role {role!r}",
            )
        if not isinstance(content, str | list):
            raise HTTPException(
                status_code=400,
                detail="message.content must be a string or a list of parts",
            )
        out.append(ChatMessage(role=role, content=content))
    return out


@router.post("/ai/chat")
async def chat(req: ChatRequest):
    if req.response_format not in {"text", "json"}:
        raise HTTPException(
            status_code=400,
            detail="response_format must be 'text' or 'json'",
        )
    try:
        provider = load_provider(_store(), req.provider)
    except ProviderError as exc:
        raise HTTPException(
            status_code=exc.status_code or 500,
            detail=str(exc),
        ) from exc

    options = ChatOptions(
        model=req.model,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        response_format=req.response_format,  # type: ignore[arg-type]
        timeout_s=req.timeout_s,
        extra=dict(req.extra),
    )
    messages = _coerce_messages(req.messages)

    if not req.stream:
        try:
            result = provider.chat(messages, options)
        except ProviderError as exc:
            raise HTTPException(
                status_code=exc.status_code or 502,
                detail=str(exc),
            ) from exc
        return {
            "text": result.text,
            "model": result.model,
            "finish_reason": result.finish_reason,
            "usage_input_tokens": result.usage_input_tokens,
            "usage_output_tokens": result.usage_output_tokens,
        }

    async def gen():
        try:
            async for chunk in provider.chat_stream(messages, options):
                # SSE-style framing so the frontend can consume with a
                # tiny line-buffer parser instead of pulling in a full
                # WebSocket dependency.
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except ProviderError as exc:
            yield f"event: error\ndata: {exc}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
