"""WD v1.4 / v3 tagger — convert images into Danbooru-style tag strings.

Loads an ONNX checkpoint from Hugging Face on first use and runs sigmoid
multi-label classification. Default thresholds match the official Space
(`general=0.35`, `character=0.85`). Inputs go through the WD-specific
preprocessing pipeline: square-pad with white, resize to 448, RGB->BGR,
float32 in [0, 255] without mean/std normalization.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from lorahub.core.models.downloader import modelscope_download_file
from lorahub.core.net import DownloadPreferences, download_preferences, hf_download
from lorahub.core.redaction import redact_command_text

if TYPE_CHECKING:
    import onnxruntime as ort

DEFAULT_MODEL = "SmilingWolf/wd-eva02-large-tagger-v3"

# Curated catalogue of WD tagger checkpoints under the
# ``SmilingWolf`` HF org. Order = recommended-first: v3-eva02 leads
# because it's the highest-quality model in the family at the time of
# writing; v3 vit / v3 swinv2 are the typical "good enough + fast"
# alternatives; v2 entries are kept for compatibility with older
# captioning workflows that pinned to specific tag distributions.
#
# Each entry mirrors the HF repo layout we depend on:
#
#     <repo_id>/model.onnx + selected_tags.csv
#
# Adding a new model id here is enough for the UI dropdown / CLI
# autocomplete; nothing else is plumbed off-list. Use full
# ``<owner>/<name>`` form so non-SmilingWolf forks (e.g. user-trained
# checkpoints) can be listed alongside in the future without a code
# change at every call site.
WD14_MODEL_CATALOG: tuple[tuple[str, str], ...] = (
    ("SmilingWolf/wd-eva02-large-tagger-v3", "v3 · EvaCLIP-Large(推荐)"),
    ("SmilingWolf/wd-vit-large-tagger-v3", "v3 · ViT-Large"),
    ("SmilingWolf/wd-swinv2-tagger-v3", "v3 · SwinV2(下载量最高)"),
    ("SmilingWolf/wd-vit-tagger-v3", "v3 · ViT"),
    ("SmilingWolf/wd-convnext-tagger-v3", "v3 · ConvNeXt"),
    ("SmilingWolf/wd-v1-4-moat-tagger-v2", "v2 · MOAT"),
    ("SmilingWolf/wd-v1-4-swinv2-tagger-v2", "v2 · SwinV2"),
    ("SmilingWolf/wd-v1-4-convnextv2-tagger-v2", "v2 · ConvNeXtV2"),
    ("SmilingWolf/wd-v1-4-convnext-tagger-v2", "v2 · ConvNeXt"),
    ("SmilingWolf/wd-v1-4-vit-tagger-v2", "v2 · ViT"),
    ("SmilingWolf/wd-v1-4-convnext-tagger", "v1 · ConvNeXt"),
    ("SmilingWolf/wd-v1-4-vit-tagger", "v1 · ViT"),
)
WD14_MODEL_IDS: tuple[str, ...] = tuple(repo for repo, _ in WD14_MODEL_CATALOG)
WD14_MODELSCOPE_REPOS: dict[str, str] = {
    "SmilingWolf/wd-eva02-large-tagger-v3": "fireicewolf/wd-eva02-large-tagger-v3",
    "SmilingWolf/wd-vit-large-tagger-v3": "fireicewolf/wd-vit-large-tagger-v3",
    "SmilingWolf/wd-swinv2-tagger-v3": "fireicewolf/wd-swinv2-tagger-v3",
    "SmilingWolf/wd-vit-tagger-v3": "fireicewolf/wd-vit-tagger-v3",
    "SmilingWolf/wd-convnext-tagger-v3": "fireicewolf/wd-convnext-tagger-v3",
    "SmilingWolf/wd-v1-4-moat-tagger-v2": "fireicewolf/wd-v1-4-moat-tagger-v2",
    "SmilingWolf/wd-v1-4-swinv2-tagger-v2": "fireicewolf/wd-v1-4-swinv2-tagger-v2",
    "SmilingWolf/wd-v1-4-convnextv2-tagger-v2": (
        "fireicewolf/wd-v1-4-convnextv2-tagger-v2"
    ),
    "SmilingWolf/wd-v1-4-convnext-tagger-v2": "fireicewolf/wd-v1-4-convnext-tagger-v2",
    "SmilingWolf/wd-v1-4-vit-tagger-v2": "fireicewolf/wd-v1-4-vit-tagger-v2",
    "SmilingWolf/wd-v1-4-convnext-tagger": "fireicewolf/wd-v1-4-convnext-tagger",
    "SmilingWolf/wd-v1-4-vit-tagger": "fireicewolf/wd-v1-4-vit-tagger",
}
WD14_DOWNLOAD_SOURCES = frozenset({"auto", "huggingface", "modelscope"})

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_RATING_CATEGORY = 9
_GENERAL_CATEGORY = 0
_CHARACTER_CATEGORY = 4
_CUDA_PROVIDER = "CUDAExecutionProvider"
_CPU_PROVIDER = "CPUExecutionProvider"


class CudaUnavailableError(RuntimeError):
    """Raised when CUDA execution was requested explicitly but isn't available."""


class TaggerModelDownloadError(RuntimeError):
    """Raised when every configured source fails to provide a tagger asset."""


def resolve_download_sources(
    model_id: str,
    source: str,
    *,
    preferences: DownloadPreferences | None = None,
) -> tuple[str, ...]:
    """Return the ordered source candidates for one WD14 model.

    ``auto`` honors Settings -> Network -> Prefer ModelScope, but retains a
    Hugging Face fallback. Explicit source selection never silently changes
    providers. ModelScope ids are mapped only for the curated mirrors verified
    to contain both ``model.onnx`` and ``selected_tags.csv``.
    """
    requested = source.strip().lower()
    if requested not in WD14_DOWNLOAD_SOURCES:
        expected = ", ".join(sorted(WD14_DOWNLOAD_SOURCES))
        raise ValueError(f"unknown download source {source!r}; expected one of: {expected}")
    has_modelscope_mirror = model_id in WD14_MODELSCOPE_REPOS
    if requested == "modelscope":
        if not has_modelscope_mirror:
            raise TaggerModelDownloadError(
                f"no verified ModelScope mirror is configured for WD14 model {model_id!r}; "
                "use a model from the built-in catalogue or select Hugging Face"
            )
        return ("modelscope",)
    if requested == "huggingface":
        return ("huggingface",)
    prefs = preferences or download_preferences()
    if prefs.prefer_modelscope and has_modelscope_mirror:
        return ("modelscope", "huggingface")
    return ("huggingface",)


def _resolve_providers(device: str, available: list[str]) -> list[str]:
    """Translate `device` ('auto'/'cpu'/'cuda') into an ONNX providers list.

    `available` is what `ort.get_available_providers()` returns; we return a
    list ordered by priority. CUDA falls through to CPU automatically in auto
    mode, but raises in explicit `cuda` mode so the user notices a missing
    `onnxruntime-gpu` install.
    """
    device = device.lower()
    if device == "cpu":
        return [_CPU_PROVIDER]
    if device == "cuda":
        if _CUDA_PROVIDER not in available:
            msg = (
                "device='cuda' was requested but CUDAExecutionProvider is not available. "
                "Install onnxruntime-gpu (`pip uninstall onnxruntime` first, "
                "then `pip install onnxruntime-gpu`) and ensure CUDA 12.x is on your system."
            )
            raise CudaUnavailableError(msg)
        return [_CUDA_PROVIDER, _CPU_PROVIDER]
    if device != "auto":
        msg = f"unknown device {device!r}; expected 'auto', 'cpu', or 'cuda'"
        raise ValueError(msg)
    if _CUDA_PROVIDER in available:
        return [_CUDA_PROVIDER, _CPU_PROVIDER]
    return [_CPU_PROVIDER]


@dataclass(frozen=True, slots=True)
class TagPrediction:
    name: str
    score: float


@dataclass(frozen=True, slots=True)
class TagResult:
    image: Path
    rating: TagPrediction | None
    general: list[TagPrediction]
    character: list[TagPrediction]

    def caption(self, *, underscores: bool = False, include_character: bool = True) -> str:
        """Format as a kohya-style comma-separated tag string."""
        tags = list(self.general)
        if include_character:
            tags = list(self.character) + tags
        names = [t.name if underscores else t.name.replace("_", " ") for t in tags]
        return ", ".join(names)


@dataclass(slots=True)
class WD14Tagger:
    model_id: str = DEFAULT_MODEL
    general_threshold: float = 0.35
    character_threshold: float = 0.85
    device: str = "auto"  # auto | cpu | cuda
    source: str = "auto"  # auto | huggingface | modelscope

    _session: ort.InferenceSession | None = field(default=None, init=False, repr=False)
    _input_name: str = field(default="", init=False, repr=False)
    _input_size: int = field(default=448, init=False, repr=False)
    _tag_names: list[str] = field(default_factory=list, init=False, repr=False)
    _tag_categories: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32), init=False, repr=False
    )
    _active_provider: str = field(default="", init=False, repr=False)
    _active_download_source: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self.source = self.source.strip().lower()
        if self.source not in WD14_DOWNLOAD_SOURCES:
            expected = ", ".join(sorted(WD14_DOWNLOAD_SOURCES))
            raise ValueError(
                f"unknown download source {self.source!r}; expected one of: {expected}"
            )

    @property
    def active_provider(self) -> str:
        """Which ExecutionProvider the loaded session is actually using."""
        return self._active_provider

    @property
    def active_download_source(self) -> str:
        """Source that supplied the loaded checkpoint assets."""
        return self._active_download_source

    def _download_asset(
        self,
        filename: str,
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        """Download one required file with configured source preference/fallback."""
        from lorahub.core.tagging import download_status  # noqa: PLC0415

        preferences = download_preferences()
        resolved_candidates = resolve_download_sources(
            self.model_id,
            self.source,
            preferences=preferences,
        )
        if self._active_download_source:
            candidates = (
                self._active_download_source,
                *(
                    source
                    for source in resolved_candidates
                    if source != self._active_download_source
                ),
            )
        else:
            candidates = resolved_candidates
        failures: list[tuple[str, str]] = []
        for source in candidates:
            if should_stop is not None and should_stop():
                raise InterruptedError("stopped by user")
            download_status.mark_start(self.model_id, filename)
            try:
                if source == "modelscope":
                    repo_id = WD14_MODELSCOPE_REPOS[self.model_id]

                    def report(downloaded: int, total: int) -> None:
                        if total > 0:
                            download_status.mark_total(self.model_id, filename, total)
                        download_status.mark_downloaded(self.model_id, filename, downloaded)

                    path = modelscope_download_file(
                        repo_id,
                        filename,
                        token=preferences.modelscope_token,
                        proxy=preferences.proxy,
                        should_stop=should_stop,
                        on_progress=report,
                    )
                else:
                    path = hf_download(
                        repo_id=self.model_id,
                        filename=filename,
                        tqdm_class=download_status.tqdm_class_for(
                            self.model_id,
                            filename,
                            should_stop,
                        ),
                    )
            except InterruptedError as exc:
                download_status.mark_error(self.model_id, filename, exc)
                raise InterruptedError("stopped by user") from exc
            except Exception as exc:  # noqa: BLE001
                download_status.mark_error(self.model_id, filename, exc)
                detail = redact_command_text(str(exc)).strip() or type(exc).__name__
                failures.append((source, detail))
                continue
            download_status.mark_done(self.model_id, filename)
            self._active_download_source = source
            return path

        details = "; ".join(f"{source}: {error}" for source, error in failures)
        if self.source == "modelscope":
            hint = "Verify ModelScope connectivity, proxy, and access token settings."
        elif self.source == "huggingface":
            hint = "Verify Hugging Face endpoint, proxy, and access token settings."
        else:
            hint = (
                "Set Settings -> Network -> Prefer ModelScope or pass "
                "'--source modelscope' when Hugging Face is unavailable."
            )
        raise TaggerModelDownloadError(
            f"failed to download {self.model_id}/{filename}. {details}. {hint}"
        )

    def load(self, *, should_stop: Callable[[], bool] | None = None) -> None:
        """Eagerly download and warm up the model. Called automatically on first tag."""
        if self._session is not None:
            return
        if should_stop is not None and should_stop():
            raise InterruptedError("stopped by user")
        import onnxruntime as ort  # noqa: PLC0415

        # The ONNX file is the heavy one (~1.3GB for eva02-large); the tags
        # CSV is tiny but follows the same source and progress path.
        model_path = self._download_asset("model.onnx", should_stop=should_stop)
        if should_stop is not None and should_stop():
            raise InterruptedError("stopped by user")
        labels_path = self._download_asset("selected_tags.csv", should_stop=should_stop)

        if should_stop is not None and should_stop():
            raise InterruptedError("stopped by user")

        providers = _resolve_providers(self.device, ort.get_available_providers())
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._active_provider = self._session.get_providers()[0]
        spec = self._session.get_inputs()[0]
        self._input_name = spec.name
        # spec.shape is [N, H, W, 3]; second dim is the size we resize to.
        if isinstance(spec.shape[1], int) and spec.shape[1] > 0:
            self._input_size = int(spec.shape[1])

        names: list[str] = []
        cats: list[int] = []
        with Path(labels_path).open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                names.append(row["name"])
                cats.append(int(row["category"]))
        self._tag_names = names
        self._tag_categories = np.array(cats, dtype=np.int32)

    def tag_image(self, image_path: Path) -> TagResult:
        self.load()
        assert self._session is not None
        arr = _preprocess_image(image_path, self._input_size)
        probs = self._session.run(None, {self._input_name: arr})[0][0]
        return self._select_tags(image_path, probs)

    def predict_tags(self, image_path: Path) -> list[str]:
        """`BaseTagger` adapter — flat tag list with characters first."""
        result = self.tag_image(image_path)
        return [t.name for t in result.character] + [t.name for t in result.general]

    def tag_directory(
        self,
        directory: Path,
        *,
        recursive: bool = False,
        write_caption: bool = True,
        skip_existing: bool = True,
        underscores: bool = False,
        include_character: bool = True,
        on_progress: Callable[[Path, TagResult], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[TagResult]:
        results: list[TagResult] = []
        for img in _iter_images(directory, recursive=recursive):
            if should_stop is not None and should_stop():
                raise InterruptedError("stopped by user")
            caption_path = img.with_suffix(".txt")
            if skip_existing and caption_path.exists() and caption_path.stat().st_size > 0:
                continue
            result = self.tag_image(img)
            if write_caption:
                caption_path.write_text(
                    result.caption(underscores=underscores, include_character=include_character),
                    encoding="utf-8",
                )
            results.append(result)
            if on_progress is not None:
                on_progress(img, result)
        return results

    def _select_tags(self, image_path: Path, probs: np.ndarray) -> TagResult:
        cats = self._tag_categories
        names = self._tag_names

        rating: TagPrediction | None = None
        rating_mask = cats == _RATING_CATEGORY
        if rating_mask.any():
            rating_scores = probs[rating_mask]
            rating_names = [names[i] for i in np.flatnonzero(rating_mask)]
            best = int(np.argmax(rating_scores))
            rating = TagPrediction(rating_names[best], float(rating_scores[best]))

        general = _filter_category(probs, cats, names, _GENERAL_CATEGORY, self.general_threshold)
        character = _filter_category(
            probs, cats, names, _CHARACTER_CATEGORY, self.character_threshold
        )
        return TagResult(image=image_path, rating=rating, general=general, character=character)


def _filter_category(
    probs: np.ndarray,
    cats: np.ndarray,
    names: list[str],
    category: int,
    threshold: float,
) -> list[TagPrediction]:
    mask = cats == category
    if not mask.any():
        return []
    indices = np.flatnonzero(mask)
    scores = probs[indices]
    keep = scores >= threshold
    kept_indices = indices[keep]
    kept_scores = scores[keep]
    order = np.argsort(-kept_scores)
    return [TagPrediction(names[int(kept_indices[i])], float(kept_scores[i])) for i in order]


def _preprocess_image(path: Path, target_size: int) -> np.ndarray:
    with Image.open(path) as raw:
        image = raw.convert("RGB")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    canvas = canvas.resize((target_size, target_size), Image.Resampling.BICUBIC)
    arr = np.asarray(canvas, dtype=np.float32)
    arr = arr[:, :, ::-1]  # RGB -> BGR
    return np.expand_dims(arr, axis=0)


def _iter_images(directory: Path, *, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for p in sorted(directory.glob(pattern)):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            yield p
