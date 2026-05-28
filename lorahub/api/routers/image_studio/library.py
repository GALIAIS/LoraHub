"""Image Studio cross-dataset library: tags, trigger words, prompt templates.

These endpoints are *global* — entries here are not bound to any single
dataset. They cover the cross-dataset gaps the per-dataset endpoints can't:

* a curated tag dictionary (favourites + categories + aliases) the user
  builds up across projects;
* a trigger-word index that maps a trigger phrase to a character/concept
  and the datasets it has been used in;
* a prompt template library for VLM-driven captioning + auditing tools.

Persistence lives in ``ImageStudioLibrary`` (separate sqlite tables in the
same studio.sqlite file), so the per-dataset routers don't share lock
contention or schema changes with library writes.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from lorahub.api import app as app_module
from lorahub.api.image_studio_library import (
    ImageStudioLibrary,
    PromptTemplate,
    TagEntry,
    TriggerWordEntry,
)


router = APIRouter(prefix="/api/image-studio/library", tags=["image-studio"])


_CAMEL_CONFIG = ConfigDict(alias_generator=to_camel, populate_by_name=True)


def _library() -> ImageStudioLibrary:
    """Return the process-wide library handle, or 503 if not ready yet.

    Routers always look up via ``app_module._image_studio_library`` so
    tests can monkeypatch the singleton without rebuilding the test
    client.
    """
    lib = app_module._image_studio_library
    if lib is None:
        raise HTTPException(503, "image studio library not initialised")
    return lib


# --------------------------------------------------------------------------- #
# Tags
# --------------------------------------------------------------------------- #


class TagEntryIn(BaseModel):
    """Body for upsert_tag — accepts both snake and camel case via alias."""

    model_config = _CAMEL_CONFIG

    tag: str
    category: str = "other"
    aliases: list[str] = []
    color: str | None = None
    notes: str | None = None


class TagEntryOut(BaseModel):
    model_config = _CAMEL_CONFIG

    tag: str
    category: str
    aliases: list[str]
    color: str | None
    notes: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_entry(cls, e: TagEntry) -> TagEntryOut:
        return cls(
            tag=e.tag,
            category=e.category,
            aliases=e.aliases,
            color=e.color,
            notes=e.notes,
            created_at=e.created_at,
            updated_at=e.updated_at,
        )


@router.get("/tags")
def list_tags(
    category: str | None = Query(default=None),
    search: str | None = Query(default=None),
) -> dict[str, Any]:
    rows = _library().list_tags(category=category, search=search)
    return {"tags": [TagEntryOut.from_entry(r).model_dump(by_alias=True) for r in rows]}


@router.put("/tags/{tag}")
def upsert_tag(tag: str, body: TagEntryIn) -> dict[str, Any]:
    if body.tag != tag:
        raise HTTPException(400, "path tag must match body.tag")
    saved = _library().upsert_tag(
        TagEntry(
            tag=body.tag,
            category=body.category,
            aliases=body.aliases,
            color=body.color,
            notes=body.notes,
        )
    )
    return TagEntryOut.from_entry(saved).model_dump(by_alias=True)


@router.delete("/tags/{tag}")
def delete_tag(tag: str) -> dict[str, Any]:
    deleted = _library().delete_tag(tag)
    if not deleted:
        raise HTTPException(404, f"tag not found: {tag!r}")
    return {"deleted": True, "tag": tag}


# --------------------------------------------------------------------------- #
# Trigger words
# --------------------------------------------------------------------------- #


class TriggerEntryIn(BaseModel):
    model_config = _CAMEL_CONFIG

    trigger_word: str
    character_name: str | None = None
    concept: str | None = None
    datasets: list[str] = []
    prompt_hint: str | None = None


class TriggerEntryOut(BaseModel):
    model_config = _CAMEL_CONFIG

    trigger_word: str
    character_name: str | None
    concept: str | None
    datasets: list[str]
    prompt_hint: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_entry(cls, e: TriggerWordEntry) -> TriggerEntryOut:
        return cls(
            trigger_word=e.trigger_word,
            character_name=e.character_name,
            concept=e.concept,
            datasets=e.datasets,
            prompt_hint=e.prompt_hint,
            created_at=e.created_at,
            updated_at=e.updated_at,
        )


@router.get("/triggers")
def list_triggers(
    character_name: str | None = Query(default=None, alias="characterName"),
    search: str | None = Query(default=None),
) -> dict[str, Any]:
    rows = _library().list_triggers(character_name=character_name, search=search)
    return {
        "triggers": [
            TriggerEntryOut.from_entry(r).model_dump(by_alias=True) for r in rows
        ]
    }


@router.put("/triggers/{trigger_word}")
def upsert_trigger(trigger_word: str, body: TriggerEntryIn) -> dict[str, Any]:
    if body.trigger_word != trigger_word:
        raise HTTPException(400, "path trigger_word must match body.triggerWord")
    saved = _library().upsert_trigger(
        TriggerWordEntry(
            trigger_word=body.trigger_word,
            character_name=body.character_name,
            concept=body.concept,
            datasets=body.datasets,
            prompt_hint=body.prompt_hint,
        )
    )
    return TriggerEntryOut.from_entry(saved).model_dump(by_alias=True)


@router.delete("/triggers/{trigger_word}")
def delete_trigger(trigger_word: str) -> dict[str, Any]:
    deleted = _library().delete_trigger(trigger_word)
    if not deleted:
        raise HTTPException(404, f"trigger not found: {trigger_word!r}")
    return {"deleted": True, "triggerWord": trigger_word}


# --------------------------------------------------------------------------- #
# Prompt templates
# --------------------------------------------------------------------------- #


class PromptTemplateIn(BaseModel):
    """Body for upsert_prompt — id is optional on create."""

    model_config = _CAMEL_CONFIG

    id: str | None = None
    name: str
    category: str = "general"
    body: str = ""
    vars: list[str] = []
    is_default: bool = False
    notes: str | None = None


class PromptTemplateOut(BaseModel):
    model_config = _CAMEL_CONFIG

    id: str
    name: str
    category: str
    body: str
    vars: list[str]
    is_default: bool
    notes: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_entry(cls, e: PromptTemplate) -> PromptTemplateOut:
        return cls(
            id=e.id,
            name=e.name,
            category=e.category,
            body=e.body,
            vars=e.vars,
            is_default=e.is_default,
            notes=e.notes,
            created_at=e.created_at,
            updated_at=e.updated_at,
        )


@router.get("/prompts")
def list_prompts(
    category: str | None = Query(default=None),
) -> dict[str, Any]:
    rows = _library().list_prompts(category=category)
    return {
        "prompts": [
            PromptTemplateOut.from_entry(r).model_dump(by_alias=True) for r in rows
        ]
    }


@router.post("/prompts")
def create_prompt(body: PromptTemplateIn) -> dict[str, Any]:
    if body.id:
        # Reject explicit ids on POST so callers can't accidentally
        # overwrite an existing template via the create endpoint.
        # PUT /prompts/{id} is the upsert path.
        raise HTTPException(
            400,
            "POST /prompts does not accept body.id; use PUT /prompts/{id}",
        )
    if _library().get_prompt_by_name(body.name) is not None:
        raise HTTPException(409, f"prompt name already exists: {body.name!r}")
    saved = _library().upsert_prompt(
        PromptTemplate(
            id="",
            name=body.name,
            category=body.category,
            body=body.body,
            vars=body.vars,
            is_default=body.is_default,
            notes=body.notes,
        )
    )
    return PromptTemplateOut.from_entry(saved).model_dump(by_alias=True)


@router.put("/prompts/{prompt_id}")
def upsert_prompt(prompt_id: str, body: PromptTemplateIn) -> dict[str, Any]:
    saved = _library().upsert_prompt(
        PromptTemplate(
            id=prompt_id,
            name=body.name,
            category=body.category,
            body=body.body,
            vars=body.vars,
            is_default=body.is_default,
            notes=body.notes,
        )
    )
    return PromptTemplateOut.from_entry(saved).model_dump(by_alias=True)


@router.delete("/prompts/{prompt_id}")
def delete_prompt(prompt_id: str) -> dict[str, Any]:
    deleted = _library().delete_prompt(prompt_id)
    if not deleted:
        raise HTTPException(404, f"prompt not found: {prompt_id}")
    return {"deleted": True, "id": prompt_id}


__all__ = ["router"]
