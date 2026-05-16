"""JoyTag tagger — booru-style auto-tagger backed by `fancyfeast/joytag`.

Uses PyTorch (already a hard dependency for any LoRA training rig) so we
don't pull in transformers / timm just for inference. The HuggingFace repo
ships ``model.safetensors`` (a custom ViT-B/16 + sigmoid head) plus
``top_tags.txt`` (one tag per line, ~5000 entries). Default predict
threshold is 0.4, matching the upstream README.

NOTE — model architecture is the load-bearing TODO here. The upstream
checkout pairs ``model.safetensors`` with a ``Models.py`` module that
defines a custom ``VisionModel`` class; weights are stored under that
class's parameter names (``vision_model.*`` etc.) and are *not* a stock
HuggingFace ``transformers.ViTModel`` checkpoint. Until we vendor or
reimplement that class, ``load()`` will:

1. Download ``model.safetensors`` and ``top_tags.txt`` from the Hub.
2. Read the safetensors header so we can describe what's there.
3. Raise ``JoyTagModelError`` with a clear message pointing at this TODO.

Once the class is in place, the rest of the file (preprocessing,
``predict_tags``, ``tag_directory``) plumbs the result through unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

if TYPE_CHECKING:
    import torch

DEFAULT_MODEL = "fancyfeast/joytag"
DEFAULT_THRESHOLD = 0.4
INPUT_SIZE = 448
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class JoyTagModelError(RuntimeError):
    """Raised when the JoyTag weights can't be turned into a usable model.

    Most often: the upstream ``Models.py`` class definition isn't available
    in this process, so we can't materialise a module that matches the
    safetensors parameter names.
    """


@dataclass(frozen=True, slots=True)
class JoyTagPrediction:
    name: str
    score: float


@dataclass(frozen=True, slots=True)
class JoyTagResult:
    image: Path
    tags: list[JoyTagPrediction]

    def caption(self, *, underscores: bool = False, **_unused: Any) -> str:
        """Format as a kohya-style comma-separated tag string.

        ``**_unused`` swallows ``include_character=`` for symmetry with the
        WD14 result type — JoyTag doesn't separate categories, so the flag
        is meaningless here, but call sites stay polymorphic.
        """
        names = [t.name if underscores else t.name.replace("_", " ") for t in self.tags]
        return ", ".join(names)


@dataclass(slots=True)
class JoyTagger:
    model_id: str = DEFAULT_MODEL
    predict_threshold: float = DEFAULT_THRESHOLD
    device: str = "auto"  # auto | cpu | cuda

    _model: Any = field(default=None, init=False, repr=False)
    _tag_names: list[str] = field(default_factory=list, init=False, repr=False)
    _active_provider: str = field(default="", init=False, repr=False)
    _torch_device: Any = field(default=None, init=False, repr=False)

    @property
    def active_provider(self) -> str:
        """e.g. ``"cuda"`` or ``"cpu"``. Empty until `load()` ran."""
        return self._active_provider

    def load(self) -> None:
        """Download weights + tag list and build the inference module.

        See module docstring for the architecture caveat — this currently
        raises ``JoyTagModelError`` until the upstream model class is
        wired in.
        """
        if self._model is not None:
            return

        # Download first so failures during model construction still leave
        # the safetensors blob cached locally for the next attempt.
        weights_path = hf_hub_download(repo_id=self.model_id, filename="model.safetensors")
        tags_path = hf_hub_download(repo_id=self.model_id, filename="top_tags.txt")

        with Path(tags_path).open(encoding="utf-8") as fh:
            self._tag_names = [line.strip() for line in fh if line.strip()]

        # Inspect the safetensors header so the error message can hint at
        # what's actually in the file.
        param_summary = _safetensors_param_summary(Path(weights_path))

        # Resolve the runtime now so ``active_provider`` is populated even
        # though we never finished loading. Catch torch-missing as part of
        # the same ``JoyTagModelError`` umbrella so callers don't need a
        # second except clause.
        try:
            self._torch_device = _resolve_torch_device(self.device)
            self._active_provider = str(self._torch_device)
        except ImportError as exc:
            msg = (
                "JoyTag requires torch for inference but ``import torch`` failed: "
                f"{exc}. Install torch (any CUDA build works) and retry."
            )
            raise JoyTagModelError(msg) from exc

        # TODO: vendor or reimplement fancyfeast/joytag's ``Models.VisionModel``
        # so we can ``load_state_dict(...)`` the safetensors here. Until then
        # we fail fast with enough context for the caller to swap in WD14.
        msg = (
            "JoyTag inference is not yet wired up. The model weights at "
            f"{weights_path} were downloaded successfully ({param_summary}), "
            "but the matching ``VisionModel`` architecture from "
            "fancyfeast/joytag's Models.py has not been ported into LoraHub yet. "
            "Pass tagger='wd14' for now or open an issue."
        )
        raise JoyTagModelError(msg)

    def tag_image(self, image_path: Path) -> JoyTagResult:
        """Run inference on a single image. Calls `load()` lazily."""
        self.load()
        assert self._model is not None
        import torch  # noqa: PLC0415

        tensor = _preprocess_image(image_path).to(self._torch_device)
        with torch.no_grad():
            logits = self._model(tensor.unsqueeze(0))
            if isinstance(logits, dict):
                # Upstream module returns ``{'tags': logits}``.
                logits = logits.get("tags", next(iter(logits.values())))
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        return self._select_tags(image_path, probs)

    def predict_tags(self, image_path: Path) -> list[str]:
        """`BaseTagger` adapter — flat list of tag names above threshold."""
        return [p.name for p in self.tag_image(image_path).tags]

    def tag_directory(
        self,
        directory: Path,
        *,
        recursive: bool = False,
        write_caption: bool = True,
        skip_existing: bool = True,
        underscores: bool = False,
        include_character: bool = True,  # noqa: ARG002 — symmetry with WD14
        on_progress: Callable[[Path, JoyTagResult], None] | None = None,
    ) -> list[JoyTagResult]:
        results: list[JoyTagResult] = []
        for img in _iter_images(directory, recursive=recursive):
            caption_path = img.with_suffix(".txt")
            if skip_existing and caption_path.exists() and caption_path.stat().st_size > 0:
                continue
            result = self.tag_image(img)
            if write_caption:
                caption_path.write_text(
                    result.caption(underscores=underscores),
                    encoding="utf-8",
                )
            results.append(result)
            if on_progress is not None:
                on_progress(img, result)
        return results

    def _select_tags(self, image_path: Path, probs: np.ndarray) -> JoyTagResult:
        if probs.shape[0] != len(self._tag_names):
            msg = (
                f"model output has {probs.shape[0]} probs but tag list has "
                f"{len(self._tag_names)} entries; the wrong top_tags.txt was loaded."
            )
            raise JoyTagModelError(msg)
        keep = np.flatnonzero(probs >= self.predict_threshold)
        order = keep[np.argsort(-probs[keep])]
        tags = [JoyTagPrediction(self._tag_names[int(i)], float(probs[i])) for i in order]
        return JoyTagResult(image=image_path, tags=tags)


def _resolve_torch_device(device: str) -> torch.device:
    """Translate ``'auto'/'cpu'/'cuda'`` into a concrete ``torch.device``."""
    import torch  # noqa: PLC0415

    device = device.lower()
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            msg = (
                "device='cuda' was requested but no CUDA-capable GPU is visible to torch. "
                "Install a CUDA build of pytorch or pass device='cpu'."
            )
            raise JoyTagModelError(msg)
        return torch.device("cuda")
    if device != "auto":
        msg = f"unknown device {device!r}; expected 'auto', 'cpu', or 'cuda'"
        raise ValueError(msg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safetensors_param_summary(path: Path) -> str:
    """Read just the safetensors JSON header so we can describe the file.

    The format is: 8-byte little-endian uint64 header length, then a UTF-8
    JSON object whose keys are tensor names. We don't load any tensors —
    this is purely for the error message and never costs more than a few KB.
    """
    try:
        with path.open("rb") as fh:
            (header_len,) = np.frombuffer(fh.read(8), dtype="<u8")
            header = json.loads(fh.read(int(header_len)).decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return f"unreadable safetensors header: {exc}"
    tensor_keys = [k for k in header if k != "__metadata__"]
    sample = ", ".join(tensor_keys[:3])
    return (
        f"{len(tensor_keys)} tensors; first keys: {sample}"
        if tensor_keys
        else "header parsed but no tensors found"
    )


def _preprocess_image(path: Path) -> torch.Tensor:
    """JoyTag-style preprocessing: square-pad, resize 448, normalise to [0,1].

    Closely follows the reference inference snippet in fancyfeast/joytag's
    README. ImageNet mean/std normalisation is intentionally skipped — the
    upstream model uses raw [0,1] tensors.
    """
    import torch  # noqa: PLC0415

    with Image.open(path) as raw:
        image = raw.convert("RGB")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    canvas = canvas.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BICUBIC)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    # HWC -> CHW
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _iter_images(directory: Path, *, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for p in sorted(directory.glob(pattern)):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            yield p


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_THRESHOLD",
    "INPUT_SIZE",
    "JoyTagModelError",
    "JoyTagPrediction",
    "JoyTagResult",
    "JoyTagger",
]
