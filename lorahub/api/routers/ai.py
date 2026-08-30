"""HTTP API for the ShiroManager-shaped AI subsystem.

Endpoint surface mirrors `src/lib/shiro-api.ts` 1:1:

    GET    /api/ai/providers
    GET    /api/ai/providers/{id}
    PUT    /api/ai/providers
    DELETE /api/ai/providers/{id}
    GET    /api/ai/models
    PUT    /api/ai/models
    DELETE /api/ai/models/{id}
    POST   /api/ai/providers/{id}/discover-models
    GET    /api/ai/routes
    PUT    /api/ai/routes
    POST   /api/ai/test
    POST   /api/ai/invoke

Plus a couple of LoraHub-specific extras kept out of the panel UI:
    POST   /api/ai/keys/{id}/reset-runtime   reset a single key's stats
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lorahub.api import app as app_module
from lorahub.api.ai_store import (
    AIModel,
    AIProvider,
    AIProviderKey,
    AIRoute,
    AIStore,
    default_ai_store_path,
)
from lorahub.api.dataset_files import (
    ImageInputTooLarge,
    encode_image_data_url,
    max_ai_image_bytes,
)
from lorahub.core.ai import client as ai_client

router = APIRouter(prefix="/api")


# --------------------------------------------------------------------------- #
# Lifespan singleton bootstrap
# --------------------------------------------------------------------------- #


def _resolve_image_url(img: "InvokeImageInput") -> str:
    """Convert an image input to a data URL for the OpenAI vision API."""
    if img.kind == "data_url":
        header, separator, payload = img.value.partition(",")
        allowed_headers = {
            "data:image/bmp;base64",
            "data:image/gif;base64",
            "data:image/jpeg;base64",
            "data:image/png;base64",
            "data:image/webp;base64",
        }
        if separator != "," or header.lower() not in allowed_headers:
            raise HTTPException(400, "invalid image data URL")
        try:
            decoded = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(400, "invalid base64 image data") from exc
        limit = max_ai_image_bytes()
        if len(decoded) > limit:
            raise HTTPException(
                413,
                f"image exceeds AI input limit of {limit} bytes",
            )
        return f"{header.lower()},{base64.b64encode(decoded).decode('ascii')}"
    try:
        return encode_image_data_url(Path(img.value))
    except ImageInputTooLarge as exc:
        raise HTTPException(413, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _store() -> AIStore:
    """Return the live AIStore, creating it lazily if lifespan didn't.

    Tests monkeypatch ``app_module._ai_store`` to a per-test path before
    issuing requests, so we resolve the symbol dynamically rather than
    capturing it at module import.
    """
    store = getattr(app_module, "_ai_store", None)
    if store is None:
        store = AIStore(default_ai_store_path())
        app_module._ai_store = store
    return store


# --------------------------------------------------------------------------- #
# Request / response shapes (ShiroManager-flavoured camelCase on the wire,
# snake_case in Python; pydantic handles the bridge)
# --------------------------------------------------------------------------- #


def _key_preview(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def _serialise_key(k: AIProviderKey) -> dict[str, Any]:
    return {
        "id": k.id,
        "preview": _key_preview(k.api_key),
        "createdAt": k.created_at,
        "updatedAt": k.updated_at,
        "runtime": {
            "requestCount": k.runtime.request_count,
            "successCount": k.runtime.success_count,
            "failureCount": k.runtime.failure_count,
            "consecutiveFailures": k.runtime.consecutive_failures,
            "lastUsedAt": k.runtime.last_used_at,
            "lastSucceededAt": k.runtime.last_succeeded_at,
            "lastFailedAt": k.runtime.last_failed_at,
            "lastError": k.runtime.last_error,
            "cooldownUntil": k.runtime.cooldown_until,
        },
    }


def _serialise_provider(
    store: AIStore, p: AIProvider
) -> dict[str, Any]:
    keys = store.list_keys(p.id)
    return {
        "id": p.id,
        "name": p.name,
        "kind": p.kind,
        "baseUrl": p.base_url,
        "organization": p.organization,
        "project": p.project,
        "headers": p.headers,
        "enabled": p.enabled,
        "hasApiKey": any(k.api_key for k in keys),
        "apiKeyPreview": _key_preview(keys[0].api_key) if keys else "",
        "apiKeyCount": len(keys),
        "apiKeySelectionMode": p.selection_mode,
        "apiKeys": [_serialise_key(k) for k in keys],
        "createdAt": p.created_at,
        "updatedAt": p.updated_at,
    }


def _serialise_model(m: AIModel) -> dict[str, Any]:
    return {
        "id": m.id,
        "providerId": m.provider_id,
        "modelId": m.model_id,
        "displayName": m.display_name,
        "source": m.source,
        "enabled": m.enabled,
        "raw": m.raw,
        "createdAt": m.created_at,
        "updatedAt": m.updated_at,
    }


def _serialise_route(r: AIRoute) -> dict[str, Any]:
    return {
        "taskId": r.task_id,
        "providerId": r.provider_id,
        "modelId": r.model_id,
        "systemPrompt": r.system_prompt,
        "stream": r.stream,
        "temperature": r.temperature,
        "topP": r.top_p,
        "frequencyPenalty": r.frequency_penalty,
        "presencePenalty": r.presence_penalty,
        "maxOutputTokens": r.max_output_tokens,
        "seed": r.seed,
        "reasoningEffort": r.reasoning_effort,
        "thinkingBudgetTokens": r.thinking_budget_tokens,
        "includeReasoning": r.include_reasoning,
        "stopSequences": r.stop_sequences,
        "extraBodyJson": r.extra_body_json,
        "enabled": r.enabled,
        "createdAt": r.created_at,
        "updatedAt": r.updated_at,
    }


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


class ProviderKeyDraft(BaseModel):
    id: str | None = None
    value: str | None = None  # full plaintext; absent means "keep existing"
    preview: str | None = None  # ignored on input, recomputed on save


class ProviderDraft(BaseModel):
    id: str | None = None
    name: str
    kind: str = "openai-compatible"
    baseUrl: str = ""
    organization: str = ""
    project: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    apiKeySelectionMode: str = "round_robin"
    apiKeys: list[ProviderKeyDraft] | None = None
    apiKey: str | None = None  # legacy single-key shortcut
    clearApiKey: bool = False


@router.get("/ai/providers")
def list_providers() -> dict[str, Any]:
    s = _store()
    return {"providers": [_serialise_provider(s, p) for p in s.list_providers()]}


@router.get("/ai/providers/{provider_id}")
def get_provider(provider_id: str) -> dict[str, Any]:
    s = _store()
    p = s.get_provider(provider_id)
    if p is None:
        raise HTTPException(status_code=404, detail="provider not found")
    return _serialise_provider(s, p)


def _resolve_keys(
    s: AIStore,
    provider_id: str,
    drafts: list[ProviderKeyDraft] | None,
    legacy_single: str | None,
    clear: bool,
) -> list[AIProviderKey] | None:
    """Translate the ProviderDraft key shape into AIProviderKey list, or
    None if the request didn't touch keys at all (so we don't wipe).
    """
    if clear:
        return []
    if drafts is not None:
        existing = {k.id: k for k in s.list_keys(provider_id)}
        out: list[AIProviderKey] = []
        for d in drafts:
            value = d.value
            if not value and d.id and d.id in existing:
                value = existing[d.id].api_key  # keep prior value
            if not value:
                continue
            out.append(
                AIProviderKey(
                    id=d.id or "",
                    provider_id=provider_id,
                    api_key=value,
                )
            )
        return out
    if legacy_single is not None:
        return [
            AIProviderKey(
                id="", provider_id=provider_id, api_key=legacy_single
            )
        ]
    return None


@router.put("/ai/providers")
def upsert_provider(req: ProviderDraft) -> dict[str, Any]:
    s = _store()
    provider = AIProvider(
        id=req.id or "",
        name=req.name,
        kind=req.kind or "openai-compatible",
        base_url=req.baseUrl,
        organization=req.organization,
        project=req.project,
        headers=dict(req.headers),
        enabled=req.enabled,
        selection_mode=req.apiKeySelectionMode or "round_robin",
        last_key_index=-1,
    )
    saved = s.upsert_provider(provider)
    keys = _resolve_keys(s, saved.id, req.apiKeys, req.apiKey, req.clearApiKey)
    if keys is not None:
        s.replace_keys(saved.id, keys)
    fresh = s.get_provider(saved.id)
    assert fresh is not None
    return {"provider": _serialise_provider(s, fresh)}


@router.delete("/ai/providers/{provider_id}")
def delete_provider(provider_id: str) -> dict[str, Any]:
    deleted = _store().delete_provider(provider_id)
    return {"ok": deleted, "providerId": provider_id}


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class ModelDraft(BaseModel):
    id: str | None = None
    providerId: str
    modelId: str
    displayName: str
    source: str = "manual"
    enabled: bool = True
    raw: dict[str, Any] = Field(default_factory=dict)


@router.get("/ai/models")
def list_models(provider_id: str | None = None) -> dict[str, Any]:
    return {
        "models": [
            _serialise_model(m) for m in _store().list_models(provider_id)
        ]
    }


@router.put("/ai/models")
def upsert_model(req: ModelDraft) -> dict[str, Any]:
    s = _store()
    if s.get_provider(req.providerId) is None:
        raise HTTPException(status_code=400, detail="providerId not found")
    saved = s.upsert_model(
        AIModel(
            id=req.id or "",
            provider_id=req.providerId,
            model_id=req.modelId,
            display_name=req.displayName or req.modelId,
            source=req.source,
            enabled=req.enabled,
            raw=dict(req.raw),
        )
    )
    return {"model": _serialise_model(saved)}


@router.delete("/ai/models/{model_id}")
def delete_model(model_id: str) -> dict[str, Any]:
    deleted = _store().delete_model(model_id)
    return {"ok": deleted, "modelId": model_id}


@router.post("/ai/providers/{provider_id}/discover-models")
def discover_models(provider_id: str) -> dict[str, Any]:
    try:
        models = ai_client.discover_models(_store(), provider_id)
    except ai_client.AIError as exc:
        raise HTTPException(
            status_code=exc.status_code or 502, detail=str(exc)
        ) from exc
    return {"models": [_serialise_model(m) for m in models]}


# --------------------------------------------------------------------------- #
# Routes (task -> provider+model+sampling)
# --------------------------------------------------------------------------- #


class RouteDraft(BaseModel):
    taskId: str
    providerId: str | None = None
    modelId: str | None = None
    systemPrompt: str = ""
    stream: bool | None = None
    temperature: float | None = None
    topP: float | None = None
    frequencyPenalty: float | None = None
    presencePenalty: float | None = None
    maxOutputTokens: int | None = None
    seed: int | None = None
    reasoningEffort: str | None = None
    thinkingBudgetTokens: int | None = None
    includeReasoning: bool | None = None
    stopSequences: list[str] = Field(default_factory=list)
    extraBodyJson: str = ""
    enabled: bool = True


@router.get("/ai/recommended-prompts")
def list_recommended_prompts() -> dict[str, Any]:
    """Return the bundled recommended system_prompt templates by task.

    The Settings → AI 路由 panel hits this so a "use recommended" button
    can splice the suggested prompt into the right textarea without us
    duplicating the body across the React bundle.
    """
    from lorahub.core.ai.prompts import (  # noqa: PLC0415
        ANIMA_CAPTION_DEFAULT_TASKS,
        ANIMA_CAPTION_PROMPT,
    )

    return {
        "prompts": {
            task_id: ANIMA_CAPTION_PROMPT for task_id in ANIMA_CAPTION_DEFAULT_TASKS
        }
    }


@router.get("/ai/routes")
def list_routes() -> dict[str, Any]:
    return {"routes": [_serialise_route(r) for r in _store().list_routes()]}


@router.put("/ai/routes")
def upsert_route(req: RouteDraft) -> dict[str, Any]:
    saved = _store().upsert_route(
        AIRoute(
            task_id=req.taskId,
            provider_id=req.providerId,
            model_id=req.modelId,
            system_prompt=req.systemPrompt,
            stream=req.stream,
            temperature=req.temperature,
            top_p=req.topP,
            frequency_penalty=req.frequencyPenalty,
            presence_penalty=req.presencePenalty,
            max_output_tokens=req.maxOutputTokens,
            seed=req.seed,
            reasoning_effort=req.reasoningEffort,
            thinking_budget_tokens=req.thinkingBudgetTokens,
            include_reasoning=req.includeReasoning,
            stop_sequences=list(req.stopSequences),
            extra_body_json=req.extraBodyJson,
            enabled=req.enabled,
        )
    )
    return {"route": _serialise_route(saved)}


# --------------------------------------------------------------------------- #
# Test + invoke
# --------------------------------------------------------------------------- #


class TestRequest(BaseModel):
    providerId: str
    modelId: str | None = None
    prompt: str | None = None
    systemPrompt: str | None = None
    stream: bool | None = None
    temperature: float | None = None
    topP: float | None = None
    frequencyPenalty: float | None = None
    presencePenalty: float | None = None
    maxOutputTokens: int | None = None
    seed: int | None = None
    reasoningEffort: str | None = None
    thinkingBudgetTokens: int | None = None
    includeReasoning: bool | None = None
    stopSequences: list[str] | None = None
    extraBodyJson: str | None = None


def _sampling_from_test(req: TestRequest) -> dict[str, Any]:
    return {
        "stream": req.stream,
        "temperature": req.temperature,
        "top_p": req.topP,
        "frequency_penalty": req.frequencyPenalty,
        "presence_penalty": req.presencePenalty,
        "max_output_tokens": req.maxOutputTokens,
        "seed": req.seed,
        "reasoning_effort": req.reasoningEffort,
        "thinking_budget_tokens": req.thinkingBudgetTokens,
        "include_reasoning": req.includeReasoning,
        "stop": req.stopSequences,
    }


@router.post("/ai/test")
def test_connection(req: TestRequest) -> dict[str, Any]:
    sampling = {k: v for k, v in _sampling_from_test(req).items() if v is not None}
    result = ai_client.test_connection(
        _store(),
        provider_id=req.providerId,
        model_id=req.modelId,
        prompt=req.prompt,
        system_prompt=req.systemPrompt,
        sampling=sampling,
    )
    completion = None
    if result.completion is not None:
        c = result.completion
        completion = {
            "taskId": "test",
            "providerId": c.provider_id,
            "providerName": c.provider_name,
            "modelId": c.model_id,
            "content": c.content,
            "reasoning": c.reasoning,
            "finishReason": c.finish_reason,
            "usage": {
                "promptTokens": c.usage_input_tokens,
                "completionTokens": c.usage_output_tokens,
                "totalTokens": c.usage_total_tokens,
            },
        }
    return {
        "ok": result.ok,
        "providerId": result.provider_id,
        "providerName": result.provider_name,
        "modelCount": len(result.models),
        "models": [
            {
                "id": item.get("id"),
                "object": item.get("object", "model"),
                "ownedBy": item.get("owned_by"),
            }
            for item in result.models
        ],
        "completion": completion,
        "error": result.error,
    }


class InvokeImageInput(BaseModel):
    kind: Literal["data_url", "file_path"] = "data_url"
    value: str = Field(min_length=1, max_length=40 * 1024**2)


class InvokeRequest(BaseModel):
    taskId: str = Field(min_length=1, max_length=128)
    prompt: str = Field(max_length=200_000)
    systemPrompt: str | None = Field(default=None, max_length=100_000)
    images: list[InvokeImageInput] | None = Field(default=None, max_length=8)
    stream: bool | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    topP: float | None = Field(default=None, ge=0.0, le=1.0)
    frequencyPenalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presencePenalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    maxOutputTokens: int | None = Field(default=None, ge=1, le=1_000_000)
    seed: int | None = None
    reasoningEffort: str | None = Field(default=None, max_length=32)
    thinkingBudgetTokens: int | None = Field(default=None, ge=1, le=1_000_000)
    includeReasoning: bool | None = None
    stopSequences: list[str] | None = Field(default=None, max_length=32)
    extraBodyJson: str | None = Field(default=None, max_length=1_000_000)


@router.post("/ai/invoke")
def invoke_task(req: InvokeRequest) -> dict[str, Any]:
    s = _store()
    route = s.get_route(req.taskId)
    if route is None:
        # Fall back to global.default if no per-task row exists.
        route = s.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(
            status_code=409,
            detail=f"no AI route configured for task {req.taskId!r}; "
            "set one in Settings -> AI providers.",
        )
    if not route.enabled:
        raise HTTPException(
            status_code=409,
            detail=f"AI route for task {req.taskId!r} is disabled",
        )

    overrides: dict[str, Any] = {}
    if req.systemPrompt is None:
        system_prompt = route.system_prompt
    else:
        system_prompt = req.systemPrompt
    if req.stream is not None:
        overrides["stream"] = req.stream
    if req.temperature is not None:
        overrides["temperature"] = req.temperature
    if req.topP is not None:
        overrides["top_p"] = req.topP
    if req.frequencyPenalty is not None:
        overrides["frequency_penalty"] = req.frequencyPenalty
    if req.presencePenalty is not None:
        overrides["presence_penalty"] = req.presencePenalty
    if req.maxOutputTokens is not None:
        overrides["max_output_tokens"] = req.maxOutputTokens
    if req.seed is not None:
        overrides["seed"] = req.seed
    if req.reasoningEffort is not None:
        overrides["reasoning_effort"] = req.reasoningEffort
    if req.thinkingBudgetTokens is not None:
        overrides["thinking_budget_tokens"] = req.thinkingBudgetTokens
    if req.includeReasoning is not None:
        overrides["include_reasoning"] = req.includeReasoning
    if req.stopSequences is not None:
        overrides["stop"] = req.stopSequences

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if req.images:
        content_parts: list[dict[str, Any]] = []
        if req.prompt:
            content_parts.append({"type": "text", "text": req.prompt})
        for img in req.images:
            url = _resolve_image_url(img)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": url},
            })
        messages.append({"role": "user", "content": content_parts})
    else:
        messages.append({"role": "user", "content": req.prompt})

    try:
        result = ai_client.invoke(
            s,
            provider_id=route.provider_id,
            model_id=route.model_id,
            messages=messages,
            route=route,
            overrides=overrides,
            extra_body_json=req.extraBodyJson or route.extra_body_json,
        )
    except ai_client.AIError as exc:
        raise HTTPException(
            status_code=exc.status_code or 502, detail=str(exc)
        ) from exc

    return {
        "taskId": req.taskId,
        "providerId": result.provider_id,
        "providerName": result.provider_name,
        "modelId": result.model_id,
        "content": result.content,
        "reasoning": result.reasoning,
        "finishReason": result.finish_reason,
        "usage": {
            "promptTokens": result.usage_input_tokens,
            "completionTokens": result.usage_output_tokens,
            "totalTokens": result.usage_total_tokens,
        },
    }


# --------------------------------------------------------------------------- #
# Misc maintenance
# --------------------------------------------------------------------------- #


@router.post("/ai/keys/{key_id}/reset-runtime")
def reset_key_runtime(key_id: str) -> dict[str, Any]:
    _store().reset_key_runtime(key_id)
    return {"ok": True, "keyId": key_id}
