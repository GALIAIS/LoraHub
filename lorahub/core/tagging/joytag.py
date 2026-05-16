"""JoyTag tagger — booru-style auto-tagger backed by `fancyfeast/joytag`.

Uses PyTorch (already a hard dependency for any LoRA training rig) so we
don't pull in ``transformers`` / ``timm`` just for inference. The HuggingFace
repo ships ``model.safetensors`` (a custom 12-layer ViT with a CNN stem and
sigmoid head), ``config.json`` (architecture hyperparameters), and
``top_tags.txt`` (one tag per line, ~5800 entries). Default predict threshold
is 0.4, matching the upstream README.

The model architecture lives in :mod:`lorahub.core.tagging._joytag_model` —
a trimmed inference-only port of fancyfeast/joytag's ``Models.ViT``. State
dict keys (``patch_embeddings.*``, ``blocks.N.{norm1,qkv_proj,out_proj,
skip_init1,norm2,mlp,skip_init2}``, ``norm.*``, ``head.*``) match upstream
exactly so the safetensors checkpoint loads via ``load_state_dict`` without
any remapping.
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

        Resolves the torch device first so ``active_provider`` is populated
        early (and we surface a clean error if torch is missing). Then pulls
        ``config.json`` / ``model.safetensors`` / ``top_tags.txt`` from the
        Hub, builds the vendored :class:`JoyTagViT`, and loads the state dict
        in eval mode on the chosen device.
        """
        if self._model is not None:
            return

        # Resolve torch first so a missing-torch env fails before downloading
        # several hundred MB of weights.
        try:
            self._torch_device = _resolve_torch_device(self.device)
            self._active_provider = str(self._torch_device)
        except ImportError as exc:
            msg = (
                "JoyTag requires torch for inference but ``import torch`` failed: "
                f"{exc}. Install torch (any CUDA build works) and retry."
            )
            raise JoyTagModelError(msg) from exc

        weights_path = hf_hub_download(repo_id=self.model_id, filename="model.safetensors")
        tags_path = hf_hub_download(repo_id=self.model_id, filename="top_tags.txt")
        config_path = hf_hub_download(repo_id=self.model_id, filename="config.json")

        with Path(tags_path).open(encoding="utf-8") as fh:
            self._tag_names = [line.strip() for line in fh if line.strip()]

        with Path(config_path).open(encoding="utf-8") as fh:
            config = json.load(fh)

        self._model = _load_vision_model(
            config=config,
            weights_path=Path(weights_path),
            device=self._torch_device,
        )

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


def _load_vision_model(
    *,
    config: dict[str, Any],
    weights_path: Path,
    device: torch.device,
) -> Any:
    """Build the JoyTag ViT and load the safetensors state dict into it.

    Split out from ``JoyTagger.load`` so tests can monkey-patch this with a
    dummy ``nn.Module`` and avoid a real download. Imports happen inside the
    function so the module remains importable in a torch-less env.
    """
    # noqa: PLC0415 — lazy imports so the module imports cleanly without torch.
    try:
        from safetensors.torch import load_file  # noqa: PLC0415
    except ImportError as exc:
        msg = (
            "JoyTag needs the ``safetensors`` package to read model weights but "
            f"``import safetensors.torch`` failed: {exc}. ``pip install safetensors``."
        )
        raise JoyTagModelError(msg) from exc

    from lorahub.core.tagging._joytag_model import (  # noqa: PLC0415
        build_joytag_vit,
        load_joytag_state_dict,
    )

    try:
        model = build_joytag_vit(config)
    except (KeyError, TypeError, ValueError) as exc:
        msg = (
            f"failed to build JoyTag ViT from config.json: {exc}. The config "
            "shape may have drifted from upstream — open an issue."
        )
        raise JoyTagModelError(msg) from exc

    state_dict = load_file(str(weights_path), device="cpu")
    try:
        load_joytag_state_dict(model, state_dict)
    except RuntimeError as exc:
        msg = (
            "failed to load JoyTag state dict — vendored architecture and "
            f"checkpoint don't agree: {exc}"
        )
        raise JoyTagModelError(msg) from exc

    model.eval()
    return model.to(device)


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
