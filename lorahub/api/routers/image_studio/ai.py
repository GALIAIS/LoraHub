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
    # Skip images that already have a non-empty .txt sidecar. Empty /
    # zero-byte sidecars are NOT skipped (they're usually crash-leftover
    # half-writes that should be reprocessed).
    skipAnnotated: bool = True


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
    skipped = 0
    if body.skipAnnotated:
        before = len(images)
        images = [
            p for p in images
            if not (p.with_suffix(".txt").is_file()
                    and p.with_suffix(".txt").stat().st_size > 0)
        ]
        skipped = before - len(images)
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

    return {
        "processed": len(results),
        "skipped": skipped,
        "results": results,
        "errors": errors,
    }


class AIBatchQualityInput(BaseModel):
    path: str
    recursive: bool = False
    task: str = "quality.score"
    # Skip images that already have an AI quality score in the store.
    # Quality scoring writes to the store (not to .txt), so the
    # "completed" check is different from caption batches.
    skipScored: bool = True


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
    store = _store()
    skipped = 0
    if body.skipScored:
        before = len(images)
        images = [
            p for p in images
            if not (
                (ann := store.get_annotation(str(p))) is not None
                and ann.ai_quality_label is not None
            )
        ]
        skipped = before - len(images)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

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

    return {
        "processed": len(results),
        "skipped": skipped,
        "results": results,
        "errors": errors,
    }


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


# -- Tags-only mode (no VLM, LLM composes from WD14 tags alone) --------------
#
# Used when ``captionSource == "tags"`` — the LLM never sees the image,
# only the WD14 tag list. The prompts are explicit about that constraint
# so the model doesn't hallucinate details that aren't in the tags
# (a generic VLM prompt would casually invent "soft blue lighting" out
# of nothing). Each variant mirrors the VLM-mode counterpart so the
# downstream caption assembly (`_build_anima_caption`) can stay
# agnostic to which path produced ``nl_text``.

_TAGS_ONLY_PROMPT_STYLE = (
    "You are writing the natural-language sentence that will sit inside an Anima training "
    "caption for a STYLE LoRA. You DO NOT have access to the image — only the WD14 tagger's "
    "output for it. Treat the tag list as the ground truth and do NOT invent visual details "
    "that aren't supported by at least one tag.\n\n"
    "From the tags, infer the high-level visual style: artistic medium and rendering hint "
    "(lineart, screentone, painterly, cel-shaded, monochrome, watercolor, ...), the overall "
    "lighting and color mood when the tags imply one (warm/cool/neon/golden hour/...), and "
    "the composition / framing (close-up, low angle, full body, dynamic pose, ...). When a "
    "tag like ``looking at viewer`` / ``from above`` / ``from behind`` is present, treat it "
    "as a hint about framing.\n\n"
    "Write 2-3 sentences in plain English. Mention the subject ONLY at a high level (one "
    "girl in a dynamic pose, a group on a ship deck) — do NOT enumerate clothing items, "
    "accessories, or hair details that the tag list happens to contain.\n\n"
    "If the tags are too sparse or ambiguous to support a confident sentence about a given "
    "axis (e.g. lighting), simply omit that axis. Do NOT write vague praise (beautiful, "
    "stunning, gorgeous, amazing). Do NOT begin with a trigger word, header, or label.\n\n"
    "WD14 tags: {tags}"
)

_TAGS_ONLY_PROMPT_CHARACTER = (
    "You are writing the natural-language sentences that will sit inside an Anima training "
    "caption for a CHARACTER LoRA. You DO NOT have access to the image — only the WD14 "
    "tagger's output for it. Treat the tag list as the ground truth and do NOT invent "
    "details unsupported by the tags.\n\n"
    "The model must learn the character's FIXED identity from the latent, so your sentences "
    "describe what VARIES across images. Pull these from the tag list when present:\n"
    "  - pose / action verbs (sitting, running, holding, looking back over shoulder, ...)\n"
    "  - expression (smiling, blushing, crying, sweating, ...)\n"
    "  - position / direction inside the frame (looking at viewer, from above, from behind, "
    "    profile view)\n"
    "  - background / setting tags (outdoors, indoors, classroom, beach, night, ...)\n"
    "  - framing tags (close-up, upper body, full body, cowboy shot, ...)\n"
    "  - lighting / mood tags (sunlight, moonlight, dim lighting, ...)\n\n"
    "Do NOT describe: hair color, eye color, hair style or length, signature outfit pieces, "
    "or any other fixed identity feature — even if those tags are present, the redundant "
    "ones get stripped from the final caption later.\n\n"
    "Write 2-3 sentences in plain English. If the tags are too sparse to support a given "
    "axis, omit it. Do NOT begin with a trigger word, header, or label.\n\n"
    "WD14 tags: {tags}"
)

_TAGS_ONLY_PROMPT_GENERAL = (
    "Write a 2-3 sentence natural-language description for an Anima LoRA training caption. "
    "You do NOT have the image — only the WD14 tagger's output for it. Compose the sentences "
    "strictly from what the tags support: subject, pose, framing, background, lighting, "
    "composition. If a given axis isn't supported by any tag, omit it instead of inventing. "
    "Plain English, no headers or labels.\n\nWD14 tags: {tags}"
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
    # "vlm" — multimodal LLM sees the image directly (default behaviour
    #         since the feature shipped). Best caption quality but
    #         requires a vision-capable model and burns image tokens.
    # "tags" — LLM only sees the WD14 tag list, never the image. Cheap
    #          and works against text-only models; useful when the
    #          configured VLM is rate-limited / quota-exhausted, or the
    #          user wants a faster cheaper pass.
    captionSource: str = "vlm"
    triggerWord: str | None = None
    stripStyleTags: bool = True  # accepted for compat; cleanup is built-in
    # Parallelism + reliability knobs.
    #
    # Pipeline shape:
    #   images -> [WD14 pool, taggerConcurrency workers]
    #          -> intermediate queue (tags + base64 image)
    #          -> [VLM pool, concurrency workers]
    #          -> caption written to disk
    #
    # ``concurrency`` controls the VLM stage (network-bound — we want
    # this fairly high so the API rate is the only floor). The default
    # of 8 covers most providers without 429s; cap is 64 because beyond
    # that the upstream usually starts throttling anyway.
    #
    # ``taggerConcurrency`` controls the WD14 stage. WD14 is a
    # single-GPU ONNX session — running >2-3 inferences in parallel on
    # one GPU saturates the SM scheduler with no real wall-clock gain
    # and risks OOM on smaller cards. Cap at 4.
    #
    # Per-image timeout protects the VLM stage; WD14 is fast enough we
    # don't bother timing it (a hung WD14 means the GPU is wedged and
    # the user needs to restart anyway).
    concurrency: int = 8
    taggerConcurrency: int = 2
    perImageTimeoutSec: float = 90.0
    maxRetries: int = 2
    # Skip images that already have a non-empty .txt sidecar. Useful
    # for re-running a batch that hit upstream rate-limits — the
    # second run only retries the images that failed the first time.
    # Defaults to true so the common case ("don't waste tokens
    # re-captioning already-tagged images") just works.
    skipExisting: bool = True


@dataclass
class _StageOneResult:
    """Output of the WD14 / image-prep stage handed to the VLM stage."""

    img_path: Path
    rating_name: str | None
    general_tags: list[str]
    character_tags: list[str]
    prompt_text: str
    data_url: str
    # "vlm" → stage two sends an image_url + text content list.
    # "tags" → stage two sends a single text message; the LLM never
    # sees the picture and composes the natural-language sentence
    # from ``prompt_text`` (which already has the WD14 tag list
    # baked in via the tags-only prompt template).
    caption_source: str = "vlm"


def _smart_caption_stage_one(
    img_path: Path,
    tagger: WD14Tagger,
    caption_mode: str,
    *,
    caption_source: str = "vlm",
) -> _StageOneResult:
    """Run WD14 tagging + image prep — everything that doesn't need the VLM.

    Pulled out of ``_smart_caption_single_image`` so the batch worker
    can run this on a small GPU-bound pool while the VLM stage runs on
    a much wider network-bound pool. Side-effect free: returns a plain
    dataclass, doesn't write files or touch the store.
    """
    import base64  # noqa: PLC0415
    import mimetypes  # noqa: PLC0415

    tag_result = tagger.tag_image(img_path)
    general_tags_underscore = [t.name for t in tag_result.general]
    general_tags = [t.replace("_", " ").lower() for t in general_tags_underscore]
    character_tags = [
        t.name.replace("_", " ").lower() for t in tag_result.character
    ]
    rating_name = tag_result.rating.name if tag_result.rating else None
    tags_for_prompt = ", ".join(_drop_tags(general_tags, _QUALITY_NOISE_TAGS))

    if caption_source == "tags":
        # Tags-only path — skip the base64 encode entirely; stage two
        # sends a plain text completion request.
        data_url = ""
        if caption_mode == "style":
            prompt_template = _TAGS_ONLY_PROMPT_STYLE
        elif caption_mode == "character":
            prompt_template = _TAGS_ONLY_PROMPT_CHARACTER
        else:
            prompt_template = _TAGS_ONLY_PROMPT_GENERAL
    else:
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

    return _StageOneResult(
        img_path=img_path,
        rating_name=rating_name,
        general_tags=general_tags,
        character_tags=character_tags,
        prompt_text=prompt_text,
        data_url=data_url,
        caption_source=caption_source,
    )


def _smart_caption_stage_two(
    s1: _StageOneResult,
    ai_store: Any,
    route: Any,
    merge_strategy: str,
    store: ImageStudioStore,
    caption_mode: str,
    trigger_word: str | None,
) -> dict[str, Any]:
    """Network-bound VLM (or text-only LLM) call + caption assembly + disk + store write."""
    from datetime import UTC, datetime  # noqa: PLC0415

    from lorahub.core.ai import client as ai_client  # noqa: PLC0415

    messages: list[dict[str, Any]] = []
    if route.system_prompt:
        messages.append({"role": "system", "content": route.system_prompt})
    if s1.caption_source == "tags":
        # Text-only path — many cheap / non-vision LLMs reject the
        # multimodal content list with a 400 ("invalid content type:
        # image_url") so we send a plain string. The prompt template
        # already contains the WD14 tag list inline.
        messages.append({"role": "user", "content": s1.prompt_text})
    else:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": s1.data_url}},
                {"type": "text", "text": s1.prompt_text},
            ],
        })

    result = ai_client.invoke(
        ai_store,
        provider_id=route.provider_id,
        model_id=route.model_id,
        messages=messages,
        route=route,
    )

    nl_text = result.content.strip()
    new_caption = _build_anima_caption(
        rating_tag=s1.rating_name,
        general_tags=s1.general_tags,
        character_tags=s1.character_tags,
        nl_text=nl_text,
        caption_mode=caption_mode,
        trigger_word=trigger_word,
    )

    caption_path = s1.img_path.with_suffix(".txt")
    existing = caption_path.read_text(encoding="utf-8") if caption_path.is_file() else ""
    if merge_strategy == "append":
        new_caption = (existing.strip() + "\n" + new_caption).strip()
    elif merge_strategy == "prepend":
        new_caption = (new_caption + "\n" + existing.strip()).strip()
    # else replace — keep as-is.

    caption_path.write_text(new_caption, encoding="utf-8")

    ann = store.get_annotation(str(s1.img_path))
    if ann is None:
        ann = ImageAnnotation(
            image_path=str(s1.img_path),
            sha256=_file_sha256(s1.img_path),
        )
    ann.ai_caption = result.content
    ann.ai_caption_provider = f"{result.provider_name}/{result.model_id}"
    ann.ai_caption_at = datetime.now(UTC).isoformat()
    store.upsert_annotation(ann)

    return {
        "path": str(s1.img_path),
        "wd14Tags": ", ".join(s1.general_tags),
        "caption": new_caption,
    }


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
    *,
    caption_source: str = "vlm",
) -> dict[str, Any]:
    """Single-image pipeline kept for the /single endpoint and tests.

    Composes ``_smart_caption_stage_one`` and ``_smart_caption_stage_two``.
    The batch path uses the two stages directly so it can keep them
    on separate thread pools (WD14 on a small GPU pool, VLM on a wide
    network pool).
    """
    del strip_style_tags  # kept for API compat; cleanup is built into stage1
    s1 = _smart_caption_stage_one(
        img_path, tagger, caption_mode, caption_source=caption_source,
    )
    return _smart_caption_stage_two(
        s1, ai_store, route, merge_strategy, store, caption_mode, trigger_word,
    )


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
    if body.skipExisting:
        # Drop images that already have a non-empty .txt sidecar.
        # Empty/zero-byte sidecars are NOT counted as completed —
        # they're usually the half-written remnant of a crashed
        # caption attempt and should be reprocessed.
        before = len(images)
        images = [
            p for p in images
            if not (p.with_suffix(".txt").is_file()
                    and p.with_suffix(".txt").stat().st_size > 0)
        ]
        skipped = before - len(images)
    else:
        skipped = 0
    store = _store()
    session = _SmartCaptionSession(
        session_id=str(_ulid_safe()),
        path=str(directory),
        total=len(images),
    )
    with _smart_caption_lock:
        _smart_caption_sessions[session.session_id] = session

    def run() -> None:
        # Two-stage pipeline:
        #   stage 1 (WD14 + image prep) on a small GPU-bound pool
        #   intermediate queue (bounded so we don't blow RAM with
        #     base64-encoded payloads when stage 2 falls behind)
        #   stage 2 (VLM call + write) on a wide network-bound pool
        #
        # We deliberately do NOT use one ThreadPoolExecutor for both
        # stages: that bottlenecks the VLM stage to the GPU pool's
        # worker count and was the throughput floor we hit during
        # smoke testing on the qing0ying0 dataset.
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout  # noqa: PLC0415
        import queue as _queue  # noqa: PLC0415

        vlm_workers = max(1, min(int(body.concurrency or 1), 64))
        wd14_workers = max(1, min(int(body.taggerConcurrency or 1), 4))
        timeout = float(body.perImageTimeoutSec or 90.0)
        max_retries = max(0, int(body.maxRetries or 0))

        # Bounded intermediate queue. Stage 2 is the slow stage (VLM
        # network call); buffering more than ~2x the VLM pool keeps
        # workers fed during transients without retaining hundreds of
        # base64-encoded images in RAM (each ~1-3 MiB).
        s1_queue: _queue.Queue[_StageOneResult | None] = _queue.Queue(
            maxsize=max(vlm_workers * 2, 8)
        )
        # Stage-one errors get short-circuited to the session error
        # list directly — no need to round-trip them through stage 2.
        # Tracked by a counter so the stage-two consumer knows when
        # producers are done.
        s1_done = threading.Event()

        def stage_one_worker(img_path: Path) -> None:
            if session.should_stop():
                return
            try:
                s1 = _smart_caption_stage_one(
                    img_path,
                    tagger,
                    body.captionMode,
                    caption_source=body.captionSource,
                )
            except Exception as exc:  # noqa: BLE001
                err_msg = f"WD14: {type(exc).__name__}: {exc}"
                session.add_error(str(img_path), err_msg, img_path.name)
                return
            # block-put so we honour back-pressure when stage 2 is
            # behind. Cancel checks are cheap so just retry every
            # second instead of using a queue timeout exception path.
            while not session.should_stop():
                try:
                    s1_queue.put(s1, timeout=1.0)
                    return
                except _queue.Full:
                    continue

        def stage_two_worker() -> None:
            while True:
                try:
                    s1 = s1_queue.get(timeout=1.0)
                except _queue.Empty:
                    if s1_done.is_set() and s1_queue.empty():
                        return
                    continue
                if s1 is None:
                    # Sentinel — push it back and exit so peer
                    # workers also see it. Using None as the sentinel
                    # avoids needing a separate "drained" event.
                    s1_queue.put(None)
                    return
                if session.should_stop():
                    s1_queue.task_done()
                    continue
                last_err: Exception | None = None
                for attempt in range(max_retries + 1):
                    if session.should_stop():
                        break
                    try:
                        item = _smart_caption_stage_two(
                            s1, ai_store, route, body.mergeStrategy, store,
                            body.captionMode, body.triggerWord,
                        )
                        session.add_result(item, s1.img_path.name)
                        last_err = None
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_err = exc
                        if attempt < max_retries:
                            # 429 / quota errors need much longer backoff —
                            # the upstream's window is usually minute-scale,
                            # so 2-4s isn't enough to clear the bucket. We
                            # detect "429" / "rate" / "exhausted" / "quota"
                            # in the message and step up to 30s+30s*attempt
                            # (capped at 120s). Other errors keep the fast
                            # 2-4s exponential backoff.
                            msg_l = str(exc).lower()
                            is_rate_limit = (
                                "429" in msg_l
                                or "rate" in msg_l
                                or "exhausted" in msg_l
                                or "quota" in msg_l
                            )
                            if is_rate_limit:
                                _time.sleep(min(30.0 + 30.0 * attempt, 120.0))
                            else:
                                _time.sleep(min(2.0 ** attempt, 4.0))
                            continue
                if last_err is not None:
                    err_msg = f"VLM: {type(last_err).__name__}: {last_err}"
                    session.add_error(str(s1.img_path), err_msg, s1.img_path.name)
                s1_queue.task_done()

        try:
            # Producer pool — small, GPU-bound. We use the executor as
            # a futures collector so we can apply a per-image timeout
            # against stage 1 (a hung WD14 forward shouldn't stall
            # producers indefinitely).
            with ThreadPoolExecutor(
                max_workers=wd14_workers,
                thread_name_prefix=f"sc-wd14-{session.session_id[:8]}",
            ) as wd14_pool, ThreadPoolExecutor(
                max_workers=vlm_workers,
                thread_name_prefix=f"sc-vlm-{session.session_id[:8]}",
            ) as vlm_pool:
                # Spin up consumers first so producers can start
                # back-pressuring immediately.
                vlm_futures = [
                    vlm_pool.submit(stage_two_worker)
                    for _ in range(vlm_workers)
                ]
                wd14_futures = {
                    wd14_pool.submit(stage_one_worker, p): p for p in images
                }

                # Wait for stage-one producers, applying the per-image
                # timeout against each as a stuck-WD14 safety net.
                for fut in list(wd14_futures.keys()):
                    if session.should_stop():
                        break
                    try:
                        fut.result(timeout=timeout)
                    except _Timeout:
                        p = wd14_futures[fut]
                        session.add_error(
                            str(p), f"WD14 timeout after {timeout:.0f}s", p.name,
                        )
                        fut.cancel()
                    except Exception as exc:  # noqa: BLE001
                        # Stage-one worker swallows errors; reaching
                        # here means the executor itself failed.
                        p = wd14_futures[fut]
                        session.add_error(str(p), str(exc), p.name)

                # Producers done — drop a sentinel so each consumer
                # eventually exits. We push exactly one None and rely
                # on the consumer chain (each one re-pushes it before
                # exiting) to fan it out.
                s1_done.set()
                s1_queue.put(None)

                # Wait for VLM consumers to drain. fut.result() with no
                # timeout is fine here: any stuck VLM request has its
                # own per-call timeout via stage_two_worker's retries.
                for fut in vlm_futures:
                    try:
                        fut.result(timeout=timeout * (max_retries + 2))
                    except Exception:  # noqa: BLE001
                        # A worker dying is a bug, not a per-image
                        # error; ignore so we still finish the batch.
                        pass

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
        "skipped": skipped,
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
    captionSource: str = "vlm"
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
            caption_source=body.captionSource,
        )
        return {"ok": True, **item}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"smart caption failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Trigger word suggestion
# --------------------------------------------------------------------------- #
#
# A trigger word is the rare-token-grade label LoRA training relies on to
# bind a learned concept ("blue-haired magical girl with a star wand")
# without leaking into normal prompt vocabulary. The task here is "per
# image, suggest 1-3 trigger word *candidates* that capture this image's
# distinctive identity content" — what the user would later wrap into
# the dataset's keepTokens prefix.
#
# Why per-image and not dataset-level: the user's existing inspector
# panel already renders ann.aiTriggerWords as chips next to each image,
# and the per-image signal is what makes "is this image off-distribution
# for the chosen trigger?" auditable. A dataset-level top-k can be
# computed cheaply over the per-image results (collections.Counter on
# the union of all suggestions) — this endpoint returns the per-image
# results plus that aggregation as a `dataset_top` field.

_TRIGGER_WORD_PROMPT = (
    "You are helping pick LoRA training trigger words for an image dataset. "
    "Look at this single image and suggest 1-3 short, content-distinctive "
    "phrases that uniquely identify what's in it — the character / concept / "
    "object / scene specifics that this image is *about*. "
    "\n"
    "Strict rules:\n"
    "- Phrases must be 1-3 words each, lowercase, English.\n"
    "- Prefer concrete identity ('crimson robe', 'lop ears', 'glass dome city') "
    "over generic descriptors ('cute', 'high quality', 'detailed').\n"
    "- Skip art-style words ('anime', 'illustration', 'masterpiece') — they're "
    "not trigger material.\n"
    "- Skip rating tags (safe / nsfw / etc).\n"
    "- If the image has a clear named character or franchise, lead with that.\n"
    "\n"
    "Output JSON only, no surrounding prose: "
    '{"triggers": ["phrase one", "phrase two"]}'
)


class TriggerWordsBatchInput(BaseModel):
    path: str
    recursive: bool = False
    task: str = "trigger.words"
    # Skip images that already have a trigger word suggestion stored.
    skipAnalyzed: bool = True


def _parse_trigger_words(raw: str) -> list[str]:
    """Best-effort parse of the VLM response into a clean trigger list.

    Accepts either the JSON-only output the prompt asks for or a fallback
    comma-separated string the model might emit when it ignores the JSON
    instruction. Always returns at most 3 entries, deduped, lowercased.
    """
    import json as _json  # noqa: PLC0415
    import re as _re  # noqa: PLC0415

    text = raw.strip()
    triggers: list[str] = []
    # Most VLMs honour the JSON-only request, sometimes wrapping in ```json
    # code fences. Strip those before parsing.
    fenced = _re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = _json.loads(text)
        candidate = data.get("triggers") if isinstance(data, dict) else None
        if isinstance(candidate, list):
            triggers = [str(t) for t in candidate]
    except (_json.JSONDecodeError, AttributeError, TypeError):
        # Fallback: comma / newline split.
        parts = [p.strip().strip("\"'") for p in _re.split(r"[,\n]", text)]
        triggers = [p for p in parts if p]

    seen: set[str] = set()
    cleaned: list[str] = []
    for t in triggers:
        norm = t.strip().lower()
        # Drop punctuation-only or empty tokens, cap at 3 words, skip dups.
        if not norm or not _re.search(r"[a-z]", norm):
            continue
        words = norm.split()
        if len(words) > 3:
            norm = " ".join(words[:3])
        if norm in seen:
            continue
        seen.add(norm)
        cleaned.append(norm)
        if len(cleaned) >= 3:
            break
    return cleaned


@router.post("/ai/trigger-words")
def ai_batch_trigger_words(body: TriggerWordsBatchInput) -> dict[str, Any]:
    """Suggest 1-3 LoRA-trigger-word candidates per image, then aggregate."""
    from collections import Counter  # noqa: PLC0415
    from datetime import UTC, datetime  # noqa: PLC0415

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
    store = _store()
    skipped = 0
    if body.skipAnalyzed:
        before = len(images)
        images = [
            p for p in images
            if not (
                (ann := store.get_annotation(str(p))) is not None
                and ann.ai_trigger_words is not None
                and len(ann.ai_trigger_words) > 0
            )
        ]
        skipped = before - len(images)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    counter: Counter[str] = Counter()

    # Pre-seed the counter with already-analysed images so the dataset_top
    # aggregation reflects the whole dataset, not just this batch.
    for p in _scan_images(directory, body.recursive):
        ann_existing = store.get_annotation(str(p))
        if ann_existing and ann_existing.ai_trigger_words:
            counter.update(ann_existing.ai_trigger_words)

    for img_path in images:
        try:
            import base64  # noqa: PLC0415
            import mimetypes  # noqa: PLC0415

            mime = mimetypes.guess_type(str(img_path))[0] or "image/png"
            data = img_path.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            data_url = f"data:{mime};base64,{b64}"

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": route.system_prompt or _TRIGGER_WORD_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ]
            # If the configured route override didn't mention triggers in
            # the system prompt, re-state the JSON contract on the user
            # turn so we still get parseable output.
            if route.system_prompt and "trigger" not in route.system_prompt.lower():
                messages[1]["content"].append({"type": "text", "text": _TRIGGER_WORD_PROMPT})

            result = ai_client.invoke(
                ai_store,
                provider_id=route.provider_id,
                model_id=route.model_id,
                messages=messages,
                route=route,
            )

            triggers = _parse_trigger_words(result.content)
            if not triggers:
                # Don't store an empty list — that would mark the image
                # "analyzed but produced nothing", which the next run's
                # skipAnalyzed would then skip forever. Treat empty as
                # an error so the user can retry.
                errors.append({
                    "path": str(img_path),
                    "error": "model returned no parseable triggers",
                })
                continue

            counter.update(triggers)

            ann = store.get_annotation(str(img_path))
            if ann is None:
                ann = ImageAnnotation(
                    image_path=str(img_path),
                    sha256=_file_sha256(img_path),
                )
            ann.ai_trigger_words = triggers
            ann.ai_trigger_words_at = datetime.now(UTC).isoformat()
            store.upsert_annotation(ann)

            results.append({"path": str(img_path), "triggers": triggers})
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(img_path), "error": str(exc)})

    # Top-N most common across the dataset. 8 is a sensible upper bound
    # for a "pick your trigger word" picker — beyond that the tail is
    # just noise.
    dataset_top = [
        {"trigger": t, "count": c}
        for t, c in counter.most_common(8)
    ]

    return {
        "processed": len(results),
        "skipped": skipped,
        "results": results,
        "errors": errors,
        "dataset_top": dataset_top,
    }

