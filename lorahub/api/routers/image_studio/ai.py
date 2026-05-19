"""Image Studio AI endpoints — batch caption, quality scoring, and smart caption.

Smart caption combines a local WD14 tagger with a vision LLM to produce
Anima-format captions used for LoRA training.
"""

from __future__ import annotations

import threading
import time as _time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lorahub.api.dataset_files import _resolve_under_roots
from lorahub.api.image_studio_store import ImageAnnotation, ImageStudioStore

from ._shared import _file_sha256, _scan_images, _store

if TYPE_CHECKING:
    from lorahub.core.tagging.wd14 import WD14Tagger

router = APIRouter(prefix="/api/image-studio", tags=["image-studio"])


def _ulid_safe() -> str:
    """Stand-in for ulid that's safe to call without the package.

    `ulid-py` is on the dependency list but the smart-caption sessions
    don't really need lexicographic sortability — uuid4 is plenty.
    """
    return uuid.uuid4().hex


@dataclass
class _SmartCaptionSession:
    """Live state for a background smart-caption batch.

    Mirrors the shape of `_ISTaggingSession` so the frontend can reuse
    the same polling pattern. ``stop_requested`` is honoured between
    images, so cancel arrives at most one image-render late.
    """

    session_id: str
    path: str
    total: int
    status: str = "running"  # running / succeeded / failed / canceled
    processed: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None
    last_image: str = ""
    started_at: float = field(default_factory=_time.time)
    finished_at: float | None = None
    _stop_flag: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_result(self, item: dict[str, Any], image_name: str) -> None:
        with self._lock:
            self.results.append(item)
            self.processed += 1
            self.last_image = image_name

    def add_error(self, path: str, msg: str, image_name: str) -> None:
        with self._lock:
            self.errors.append({"path": path, "error": msg})
            self.processed += 1
            self.last_image = image_name

    def set_error(self, msg: str) -> None:
        with self._lock:
            self.error = msg

    def finish(self, status: str) -> None:
        with self._lock:
            self.status = status
            self.finished_at = _time.time()

    def request_stop(self) -> None:
        with self._lock:
            self._stop_flag = True

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop_flag

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "path": self.path,
                "status": self.status,
                "processed": self.processed,
                "total": self.total,
                "percent": (
                    100.0 * self.processed / self.total
                    if self.total > 0
                    else 0.0
                ),
                "last_image": self.last_image,
                "results": list(self.results),
                "errors": list(self.errors),
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }


# Module-level session registry. Same shape as the tagging tab — the only
# state we keep across requests is "what's running right now"; finished
# sessions stick around so the frontend can pull final results once.
# Memory bound: cleared whenever the process restarts, plus best-effort
# eviction of sessions older than 1h after they finish (see snapshot).
_smart_caption_sessions: dict[str, _SmartCaptionSession] = {}
_smart_caption_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# AI batch endpoints
# --------------------------------------------------------------------------- #


class AIBatchCaptionInput(BaseModel):
    path: str
    recursive: bool = False
    task: str = "tagging.assist"
    mergeStrategy: str = "replace"


@router.post("/ai/caption")
def ai_batch_caption(body: AIBatchCaptionInput) -> dict[str, Any]:
    """Queue AI captioning for all images in a directory.

    For IS-3 this is synchronous (processes sequentially). A future
    iteration will use the session pattern for async progress.
    """
    from lorahub.api import app as app_mod  # noqa: PLC0415
    from lorahub.core.ai import client as ai_client  # noqa: PLC0415

    directory = _resolve_under_roots(body.path)
    if not directory.is_dir():
        raise HTTPException(400, "not a directory")

    ai_store = app_mod._ai_store
    if ai_store is None:
        raise HTTPException(503, "AI store not initialised")

    route = ai_store.get_route(body.task)
    if route is None or not (route.provider_id and route.model_id):
        route = ai_store.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(409, f"no AI route for task {body.task!r}")

    images = _scan_images(directory, body.recursive)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    store = _store()
    for img_path in images:
        try:
            import base64  # noqa: PLC0415
            import mimetypes  # noqa: PLC0415

            mime = mimetypes.guess_type(str(img_path))[0] or "image/png"
            data = img_path.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            data_url = f"data:{mime};base64,{b64}"

            messages: list[dict[str, Any]] = []
            if route.system_prompt:
                messages.append({"role": "system", "content": route.system_prompt})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            })

            result = ai_client.invoke(
                ai_store,
                provider_id=route.provider_id,
                model_id=route.model_id,
                messages=messages,
                route=route,
            )

            caption_path = img_path.with_suffix(".txt")
            existing = caption_path.read_text(encoding="utf-8") if caption_path.is_file() else ""

            if body.mergeStrategy == "append":
                new_caption = (existing.strip() + ", " + result.content).strip(", ")
            elif body.mergeStrategy == "rewrite":
                new_caption = result.content
            else:
                new_caption = result.content

            caption_path.write_text(new_caption, encoding="utf-8")

            ann = store.get_annotation(str(img_path))
            if ann is None:
                ann = ImageAnnotation(
                    image_path=str(img_path),
                    sha256=_file_sha256(img_path),
                )
            ann.ai_caption = result.content
            ann.ai_caption_provider = f"{result.provider_name}/{result.model_id}"
            from datetime import UTC, datetime  # noqa: PLC0415
            ann.ai_caption_at = datetime.now(UTC).isoformat()
            store.upsert_annotation(ann)

            results.append({"path": str(img_path), "caption": new_caption})
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(img_path), "error": str(exc)})

    return {"processed": len(results), "results": results, "errors": errors}


class AIBatchQualityInput(BaseModel):
    path: str
    recursive: bool = False
    task: str = "quality.score"


@router.post("/ai/quality")
def ai_batch_quality(body: AIBatchQualityInput) -> dict[str, Any]:
    """Score image quality via VLM for all images in a directory."""
    from lorahub.api import app as app_mod  # noqa: PLC0415
    from lorahub.core.ai import client as ai_client  # noqa: PLC0415

    directory = _resolve_under_roots(body.path)
    if not directory.is_dir():
        raise HTTPException(400, "not a directory")

    ai_store = app_mod._ai_store
    if ai_store is None:
        raise HTTPException(503, "AI store not initialised")

    route = ai_store.get_route(body.task)
    if route is None or not (route.provider_id and route.model_id):
        route = ai_store.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(409, f"no AI route for task {body.task!r}")

    images = _scan_images(directory, body.recursive)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    store = _store()
    for img_path in images:
        try:
            import base64  # noqa: PLC0415
            import json as json_mod  # noqa: PLC0415
            import mimetypes  # noqa: PLC0415

            mime = mimetypes.guess_type(str(img_path))[0] or "image/png"
            data = img_path.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            data_url = f"data:{mime};base64,{b64}"

            system_prompt = route.system_prompt or (
                'Rate this training image on a 0-100 scale. '
                'Return JSON: {"score": 0-100, "label": "good"|"medium"|"bad", "reason": "..."}'
            )

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ]

            result = ai_client.invoke(
                ai_store,
                provider_id=route.provider_id,
                model_id=route.model_id,
                messages=messages,
                route=route,
            )

            score: float | None = None
            label: str | None = None
            reason: str | None = None
            try:
                parsed = json_mod.loads(result.content)
                score = float(parsed.get("score", 0)) / 100.0
                label = parsed.get("label")
                reason = parsed.get("reason")
            except (json_mod.JSONDecodeError, ValueError, TypeError):
                reason = result.content

            ann = store.get_annotation(str(img_path))
            if ann is None:
                ann = ImageAnnotation(
                    image_path=str(img_path),
                    sha256=_file_sha256(img_path),
                )
            ann.ai_quality_score = score
            ann.ai_quality_label = label
            ann.ai_quality_reason = reason
            from datetime import UTC, datetime  # noqa: PLC0415
            ann.ai_quality_at = datetime.now(UTC).isoformat()
            store.upsert_annotation(ann)

            results.append({
                "path": str(img_path),
                "score": score,
                "label": label,
                "reason": reason,
            })
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(img_path), "error": str(exc)})

    return {"processed": len(results), "results": results, "errors": errors}


# --------------------------------------------------------------------------- #
# Smart caption (WD14 + Vision LLM)
# --------------------------------------------------------------------------- #

# Tags that are pure quality/medium noise (the Anima header carries quality
# explicitly, so don't double-print them in the general-tag section).
_QUALITY_NOISE_TAGS = {
    "highres", "absurdres", "best quality", "masterpiece", "high quality",
    "low quality", "worst quality", "normal quality", "lowres",
    "official art", "key visual", "promotional art", "screencap",
    "artist name", "signature", "watermark", "logo", "english text",
    "dated", "twitter username", "patreon username", "artist logo",
}

# Map WD14 rating tag -> Anima rating keyword for the header line.
_RATING_MAP = {
    "general": "safe",
    "sensitive": "sensitive",
    "questionable": "nsfw",
    "explicit": "nsfw",
}

# Process-level cache for loaded WD14 taggers, keyed by full config tuple.
# EVA02-large weights are ~1.2GB so re-loading per request kills throughput.
_TAGGER_CACHE: dict[tuple[str, float, float, str], Any] = {}
_TAGGER_LOCK = threading.Lock()


def _get_tagger(
    model_id: str,
    general_threshold: float,
    character_threshold: float,
    device: str,
) -> Any:
    """Return a WD14Tagger that's loaded once per process per config."""
    from lorahub.core.tagging.wd14 import WD14Tagger  # noqa: PLC0415

    key = (model_id, general_threshold, character_threshold, device)
    with _TAGGER_LOCK:
        cached = _TAGGER_CACHE.get(key)
        if cached is not None:
            return cached
        tagger = WD14Tagger(
            model_id=model_id,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            device=device,
        )
        tagger.load()
        _TAGGER_CACHE[key] = tagger
        return tagger

_SMART_CAPTION_PROMPT_STYLE = (
    "You are writing the natural-language sentence that will sit inside an Anima training "
    "caption for a STYLE LoRA. The reader is the text encoder; your sentence must teach it "
    "the visual style of the image.\n\n"
    "Write 2-3 sentences in plain English describing:\n"
    "  - the artistic medium and rendering (e.g. clean lineart with soft cel shading, "
    "    painterly highlights, halftone screentone, vivid saturated palette, soft pastel "
    "    palette, dynamic angle, painterly background)\n"
    "  - lighting and color mood (warm/cool/neon/golden hour/etc.)\n"
    "  - composition and framing (close-up portrait, dynamic low angle, full-body shot, etc.)\n"
    "  - the subject and pose ONLY at a high level (one girl in a dynamic pose, a group on "
    "    a ship deck), without enumerating clothing items or accessories.\n\n"
    "Do NOT begin with a trigger word, header, or label — output ONLY the sentences. "
    "Do NOT use vague praise (beautiful, stunning, gorgeous, amazing).\n\n"
    "Reference WD14 general tags (for grounding only): {tags}"
)

_SMART_CAPTION_PROMPT_CHARACTER = (
    "You are writing the natural-language sentences that will sit inside an Anima training "
    "caption for a CHARACTER LoRA. The model must learn the character's fixed identity from "
    "the latent, so your sentences must describe what VARIES across images.\n\n"
    "Write 2-3 sentences focusing on:\n"
    "  - pose, action, expression\n"
    "  - position/direction inside the frame (e.g. \"standing on the left side of the image, "
    "    looking back over her shoulder\")\n"
    "  - background and setting\n"
    "  - framing (close-up, full body, from behind, etc.)\n"
    "  - lighting/mood\n\n"
    "Do NOT describe: hair color, eye color, hair style/length, the character's signature "
    "outfit, or any other fixed identity feature. Do NOT begin with a trigger word or label "
    "— output ONLY the sentences.\n\n"
    "Reference WD14 general tags: {tags}"
)

_SMART_CAPTION_PROMPT_GENERAL = (
    "Write a 2-3 sentence natural-language description of the image for LoRA training. "
    "Cover subject, pose, clothing, background, lighting, composition. Plain English, no "
    "headers or labels.\n\nReference WD14 tags: {tags}"
)


def _drop_tags(tags: list[str], drop: set[str]) -> list[str]:
    """Case-insensitive filter — keep order, drop matches."""
    return [t for t in tags if t.lower() not in drop]


def _split_normalize_tags(raw: str) -> list[str]:
    """Split a comma list, lowercase, dedupe in-order."""
    seen: set[str] = set()
    out: list[str] = []
    for piece in raw.split(","):
        t = piece.strip().lower()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _build_anima_caption(
    *,
    rating_tag: str | None,
    general_tags: list[str],
    character_tags: list[str],
    nl_text: str,
    caption_mode: str,
    trigger_word: str | None,
) -> str:
    """Assemble an Anima-format caption.

    Layout:
      masterpiece, best quality, score_7, <safe|sensitive|nsfw>,
      [1girl/1boy/etc], [character_trigger or @artist], [series],
      <NL paragraph>,
      <remaining general tags>
    """
    rating = _RATING_MAP.get((rating_tag or "").lower(), "safe")
    header = f"masterpiece, best quality, score_7, {rating}"

    # Pick subject-count tag (1girl, 2girls, 1boy, etc.) from general — Anima
    # wants this immediately after the header.
    subject_tags: list[str] = []
    rest_general: list[str] = []
    subject_pattern = (
        "1girl", "2girls", "3girls", "4girls", "5girls", "6+girls",
        "1boy", "2boys", "3boys", "multiple_girls", "multiple_boys",
        "solo", "no humans",
    )
    for t in general_tags:
        if t in subject_pattern:
            subject_tags.append(t)
        else:
            rest_general.append(t)

    # Trigger / character / artist line.
    trigger_part: list[str] = []
    trig = (trigger_word or "").strip().lower()
    if trig:
        if caption_mode == "style":
            # Style LoRA → format as @artist-style trigger if not already.
            if not trig.startswith("@"):
                trig = f"@{trig}"
        trigger_part.append(trig)
    # WD14 character predictions go on the same line as identity hints.
    trigger_part.extend(character_tags)

    line2 = ", ".join([*subject_tags, *trigger_part])

    # Clean general tag tail: drop quality noise (header already covers it).
    tail = _drop_tags(rest_general, _QUALITY_NOISE_TAGS)
    tail_str = ", ".join(tail)

    parts = [header]
    if line2:
        parts.append(line2)
    if nl_text.strip():
        parts.append(nl_text.strip())
    if tail_str:
        parts.append(tail_str)
    return ",\n".join(parts)


class SmartCaptionBatchInput(BaseModel):
    path: str
    recursive: bool = False
    taggerModel: str = "SmilingWolf/wd-eva02-large-tagger-v3"
    visionTask: str = "tagging.assist"
    mergeStrategy: str = "replace"
    device: str = "auto"
    generalThreshold: float = 0.35
    characterThreshold: float = 0.85
    captionMode: str = "style"  # general | style | character
    triggerWord: str | None = None
    stripStyleTags: bool = True  # accepted for compat; cleanup is built-in


def _smart_caption_single_image(
    img_path: Path,
    tagger: WD14Tagger,
    ai_store: Any,
    route: Any,
    merge_strategy: str,
    store: ImageStudioStore,
    caption_mode: str = "general",
    trigger_word: str | None = None,
    strip_style_tags: bool = True,
) -> dict[str, Any]:
    """Run WD14 tagging then vision LLM on a single image. Returns result dict."""
    import base64  # noqa: PLC0415
    import mimetypes  # noqa: PLC0415
    from datetime import UTC, datetime  # noqa: PLC0415

    from lorahub.core.ai import client as ai_client  # noqa: PLC0415

    # Step 1: WD14 tagging — keep general/character/rating separately so we
    # can place them in the Anima header / line2 / tail rather than dump them
    # all into one comma list.
    tag_result = tagger.tag_image(img_path)
    general_tags_underscore = [t.name for t in tag_result.general]
    general_tags = [t.replace("_", " ").lower() for t in general_tags_underscore]
    character_tags = [
        t.name.replace("_", " ").lower() for t in tag_result.character
    ]
    rating_name = tag_result.rating.name if tag_result.rating else None
    # Tags shown to the VLM as content grounding (drop noise either way).
    tags_for_prompt = ", ".join(_drop_tags(general_tags, _QUALITY_NOISE_TAGS))

    # Step 2: Prepare image for vision LLM
    mime = mimetypes.guess_type(str(img_path))[0] or "image/png"
    data = img_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    if caption_mode == "style":
        prompt_template = _SMART_CAPTION_PROMPT_STYLE
    elif caption_mode == "character":
        prompt_template = _SMART_CAPTION_PROMPT_CHARACTER
    else:
        prompt_template = _SMART_CAPTION_PROMPT_GENERAL
    prompt_text = prompt_template.format(tags=tags_for_prompt)

    messages: list[dict[str, Any]] = []
    if route.system_prompt:
        messages.append({"role": "system", "content": route.system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt_text},
        ],
    })

    # Step 3: Invoke vision LLM
    result = ai_client.invoke(
        ai_store,
        provider_id=route.provider_id,
        model_id=route.model_id,
        messages=messages,
        route=route,
    )

    # Step 4: Assemble the Anima-format caption (header + line2 + NL + tail).
    nl_text = result.content.strip()
    new_caption = _build_anima_caption(
        rating_tag=rating_name,
        general_tags=general_tags,
        character_tags=character_tags,
        nl_text=nl_text,
        caption_mode=caption_mode,
        trigger_word=trigger_word,
    )

    caption_path = img_path.with_suffix(".txt")
    existing = caption_path.read_text(encoding="utf-8") if caption_path.is_file() else ""
    if merge_strategy == "append":
        new_caption = (existing.strip() + "\n" + new_caption).strip()
    elif merge_strategy == "prepend":
        new_caption = (new_caption + "\n" + existing.strip()).strip()
    # else replace — keep as-is.

    caption_path.write_text(new_caption, encoding="utf-8")

    # Step 5: Update annotation
    ann = store.get_annotation(str(img_path))
    if ann is None:
        ann = ImageAnnotation(
            image_path=str(img_path),
            sha256=_file_sha256(img_path),
        )
    ann.ai_caption = result.content
    ann.ai_caption_provider = f"{result.provider_name}/{result.model_id}"
    ann.ai_caption_at = datetime.now(UTC).isoformat()
    store.upsert_annotation(ann)

    return {
        "path": str(img_path),
        "wd14Tags": ", ".join(general_tags),
        "caption": new_caption,
    }


@router.post("/ai/smart-caption", status_code=202)
def ai_smart_caption_batch(body: SmartCaptionBatchInput) -> dict[str, Any]:
    """Run WD14 tagging + vision LLM captioning for all images in a directory.

    Background-task shape (unblocks uvicorn for big batches): the request
    validates inputs, returns a session_id immediately with HTTP 202, and
    runs the for-loop in a worker thread. Progress is polled via
    ``GET /api/image-studio/ai/smart-caption/status/<id>``; cancel via
    ``POST /api/image-studio/ai/smart-caption/cancel/<id>``.

    Synchronous return shape (the legacy ``processed`` / ``results`` / ``errors``
    fields) is preserved on the status endpoint when the session finishes,
    so existing callers can poll-then-pull without code changes beyond
    going through the session_id.
    """
    from lorahub.api import app as app_mod  # noqa: PLC0415

    directory = _resolve_under_roots(body.path)
    if not directory.is_dir():
        raise HTTPException(400, "not a directory")

    ai_store = app_mod._ai_store
    if ai_store is None:
        raise HTTPException(503, "AI store not initialised")

    route = ai_store.get_route(body.visionTask)
    if route is None or not (route.provider_id and route.model_id):
        route = ai_store.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(409, f"no AI route for task {body.visionTask!r}")

    # Initialize WD14 tagger (cached per-process so we don't re-load 1.2GB
    # of weights on every request). This is the only call we keep in the
    # request thread — it takes ~1s on a warm cache and the UI freezing for
    # a second is fine; running it in the background means the user has no
    # signal that the tagger failed to load.
    tagger = _get_tagger(
        body.taggerModel,
        body.generalThreshold,
        body.characterThreshold,
        body.device,
    )

    images = _scan_images(directory, body.recursive)
    store = _store()
    session = _SmartCaptionSession(
        session_id=str(_ulid_safe()),
        path=str(directory),
        total=len(images),
    )
    with _smart_caption_lock:
        _smart_caption_sessions[session.session_id] = session

    def run() -> None:
        try:
            for img_path in images:
                if session.should_stop():
                    break
                try:
                    item = _smart_caption_single_image(
                        img_path, tagger, ai_store, route, body.mergeStrategy, store,
                        caption_mode=body.captionMode,
                        trigger_word=body.triggerWord,
                        strip_style_tags=body.stripStyleTags,
                    )
                    session.add_result(item, img_path.name)
                except Exception as exc:  # noqa: BLE001
                    session.add_error(str(img_path), str(exc), img_path.name)
            session.finish("succeeded" if not session.should_stop() else "canceled")
        except Exception as exc:  # noqa: BLE001
            # Catastrophic failure (e.g. AI route token revoked mid-run).
            # Mark the session failed instead of leaking the traceback into
            # the request thread (which has long since returned 202).
            session.set_error(str(exc))
            session.finish("failed")

    threading.Thread(
        target=run,
        name=f"smart-caption-{session.session_id[:8]}",
        daemon=True,
    ).start()

    return {
        "session_id": session.session_id,
        "total": len(images),
        "status_url": (
            f"/api/image-studio/ai/smart-caption/status/{session.session_id}"
        ),
    }


@router.get("/ai/smart-caption/status/{session_id}")
def ai_smart_caption_status(session_id: str) -> dict[str, Any]:
    """Poll a smart-caption batch session's progress and final results."""
    with _smart_caption_lock:
        session = _smart_caption_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    return session.snapshot()


@router.post("/ai/smart-caption/cancel/{session_id}")
def ai_smart_caption_cancel(session_id: str) -> dict[str, Any]:
    """Request a running smart-caption batch session to stop after the current image."""
    with _smart_caption_lock:
        session = _smart_caption_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    session.request_stop()
    return {"session_id": session_id, "stop_requested": True}


class SmartCaptionSingleInput(BaseModel):
    path: str
    taggerModel: str = "SmilingWolf/wd-eva02-large-tagger-v3"
    visionTask: str = "tagging.assist"
    mergeStrategy: str = "replace"
    device: str = "auto"
    generalThreshold: float = 0.35
    characterThreshold: float = 0.85
    captionMode: str = "style"
    triggerWord: str | None = None
    stripStyleTags: bool = True


@router.post("/ai/smart-caption/single")
def ai_smart_caption_single(body: SmartCaptionSingleInput) -> dict[str, Any]:
    """Run WD14 tagging + vision LLM captioning for a single image."""
    from lorahub.api import app as app_mod  # noqa: PLC0415

    file_path = _resolve_under_roots(body.path)
    if not file_path.is_file():
        raise HTTPException(404, "image not found")

    ai_store = app_mod._ai_store
    if ai_store is None:
        raise HTTPException(503, "AI store not initialised")

    route = ai_store.get_route(body.visionTask)
    if route is None or not (route.provider_id and route.model_id):
        route = ai_store.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(409, f"no AI route for task {body.visionTask!r}")

    tagger = _get_tagger(
        body.taggerModel,
        body.generalThreshold,
        body.characterThreshold,
        body.device,
    )

    store = _store()
    try:
        item = _smart_caption_single_image(
            file_path, tagger, ai_store, route, body.mergeStrategy, store,
            caption_mode=body.captionMode,
            trigger_word=body.triggerWord,
            strip_style_tags=body.stripStyleTags,
        )
        return {"ok": True, **item}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"smart caption failed: {exc}") from exc
