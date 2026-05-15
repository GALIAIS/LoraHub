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
from huggingface_hub import hf_hub_download
from PIL import Image

if TYPE_CHECKING:
    import onnxruntime as ort

DEFAULT_MODEL = "SmilingWolf/wd-v1-4-vit-tagger-v2"
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_RATING_CATEGORY = 9
_GENERAL_CATEGORY = 0
_CHARACTER_CATEGORY = 4


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
    providers: tuple[str, ...] = ("CPUExecutionProvider",)

    _session: ort.InferenceSession | None = field(default=None, init=False, repr=False)
    _input_name: str = field(default="", init=False, repr=False)
    _input_size: int = field(default=448, init=False, repr=False)
    _tag_names: list[str] = field(default_factory=list, init=False, repr=False)
    _tag_categories: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32), init=False, repr=False
    )

    def load(self) -> None:
        """Eagerly download and warm up the model. Called automatically on first tag."""
        if self._session is not None:
            return
        import onnxruntime as ort  # noqa: PLC0415

        model_path = hf_hub_download(repo_id=self.model_id, filename="model.onnx")
        labels_path = hf_hub_download(repo_id=self.model_id, filename="selected_tags.csv")

        self._session = ort.InferenceSession(model_path, providers=list(self.providers))
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
    ) -> list[TagResult]:
        results: list[TagResult] = []
        for img in _iter_images(directory, recursive=recursive):
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
