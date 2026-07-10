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

from lorahub.core.net import hf_download

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

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_RATING_CATEGORY = 9
_GENERAL_CATEGORY = 0
_CHARACTER_CATEGORY = 4
_CUDA_PROVIDER = "CUDAExecutionProvider"
_CPU_PROVIDER = "CPUExecutionProvider"


class CudaUnavailableError(RuntimeError):
    """Raised when CUDA execution was requested explicitly but isn't available."""


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

    _session: ort.InferenceSession | None = field(default=None, init=False, repr=False)
    _input_name: str = field(default="", init=False, repr=False)
    _input_size: int = field(default=448, init=False, repr=False)
    _tag_names: list[str] = field(default_factory=list, init=False, repr=False)
    _tag_categories: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32), init=False, repr=False
    )
    _active_provider: str = field(default="", init=False, repr=False)

    @property
    def active_provider(self) -> str:
        """Which ExecutionProvider the loaded session is actually using."""
        return self._active_provider

    def load(self, *, should_stop: Callable[[], bool] | None = None) -> None:
        """Eagerly download and warm up the model. Called automatically on first tag."""
        if self._session is not None:
            return
        if should_stop is not None and should_stop():
            raise InterruptedError("stopped by user")
        import onnxruntime as ort  # noqa: PLC0415

        from lorahub.core.tagging import download_status  # noqa: PLC0415

        # Wire the HF download progress through the in-process status
        # board so the web UI's floating download toast can show the
        # actual byte count instead of an indeterminate spinner. The
        # ONNX file is the heavy one (~700MB for eva02-large); the
        # tags CSV is tiny but reported anyway for completeness.
        try:
            model_path = hf_download(
                repo_id=self.model_id,
                filename="model.onnx",
                tqdm_class=download_status.tqdm_class_for(
                    self.model_id,
                    "model.onnx",
                    should_stop,
                ),
            )
        except BaseException as exc:
            download_status.mark_error(self.model_id, "model.onnx", exc)
            raise
        if should_stop is not None and should_stop():
            raise InterruptedError("stopped by user")
        try:
            labels_path = hf_download(
                repo_id=self.model_id,
                filename="selected_tags.csv",
                tqdm_class=download_status.tqdm_class_for(
                    self.model_id,
                    "selected_tags.csv",
                    should_stop,
                ),
            )
        except BaseException as exc:
            download_status.mark_error(self.model_id, "selected_tags.csv", exc)
            raise

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
