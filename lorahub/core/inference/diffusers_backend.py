"""Generic diffusers-based preview inference backend.

Catch-all path for arches diffusers' ``AutoPipelineForText2Image``
recognises out of the box: SDXL, FLUX, SD3, SD1.5/SD2 and friends. The
backend exists primarily so non-Anima configs don't have to fall back
to the bare placeholder PNG when the user just wants a sanity check.

Two intentional limitations:
  * **Image-only**: video arches (Wan, HunyuanVideo, LTX, ...) are
    declared unsupported. ``AutoPipelineForText2Image`` doesn't know
    about them and a real video preview path needs a separate pipeline
    + a different render loop. The registry will skip this backend for
    those arches and the worker emits ``preview_unavailable``.
  * **Lazy diffusers import**: diffusers is *not* a hard dependency of
    lorahub, so we import on first call and fail the ``is_available``
    check when the import errors. That means: ``pip install lorahub``
    on a host without diffusers installed still launches jobs cleanly,
    just without rich previews for non-Anima configs.

The render path keeps the pipeline pinned to a single base-model id per
``(arch, base_id)`` pair via a tiny in-process LRU. Loading SDXL once
costs a few seconds; reusing it across checkpoints is what makes live
previews tractable on top of training.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from lorahub.core.inference import PromptSpec
from lorahub.core.inference.registry import register_backend

log = logging.getLogger(__name__)


# diffusers AutoPipelineForText2Image natively maps these arches today.
# We map lorahub arch -> conservative default pretrained id so a config
# that doesn't pin one still gets *some* preview pipeline. When the
# config carries an explicit checkpoint path we prefer that over the
# default id.
_ARCH_TO_DEFAULT_REPO: dict[str, str] = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sd2": "stabilityai/stable-diffusion-2-1",
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
    "flux": "black-forest-labs/FLUX.1-dev",
}

# Arches the registry should skip outright. Mostly video pipelines —
# diffusers ships dedicated classes for these and we don't wire them
# up here on purpose (separate cut).
_UNSUPPORTED_ARCHES: frozenset[str] = frozenset(
    {
        "wan",
        "hunyuan_video",
        "hunyuan_video_15",
        "ltx_video",
        "ltx2",
        "cosmos",
        "cosmos_predict2",
        "anima",  # delegated to the dedicated Anima backend
    }
)


class DiffusersInferenceBackend:
    """Generic diffusers ``AutoPipelineForText2Image`` driver.

    One instance per job — keeps a lazily loaded pipeline cached so
    consecutive renders against the same base model don't re-pay the
    weight-loading cost.
    """

    name = "diffusers"

    def __init__(
        self,
        *,
        arch: str,
        base_model_id: str | None = None,
        lora_strength: float = 1.0,
        device: str | None = None,
    ) -> None:
        self.arch = arch
        self.base_model_id = base_model_id
        self.lora_strength = lora_strength
        self.device = device
        self._pipeline: Any | None = None

    # --------------------------------------------------------------------- #
    # InferenceBackend Protocol
    # --------------------------------------------------------------------- #

    def is_available(self, *, arch: str) -> bool:
        if arch in _UNSUPPORTED_ARCHES:
            return False
        if arch not in _ARCH_TO_DEFAULT_REPO:
            return False
        # diffusers must be importable. Heavy deps (torch / transformers)
        # are imported transitively — we only probe the top-level here.
        try:
            import diffusers  # noqa: F401, PLC0415
        except ImportError:
            return False
        return True

    def render(
        self,
        *,
        lora_path: Path,
        spec: PromptSpec,
        out_path: Path,
        default_steps: int,
        default_cfg: float,
    ) -> None:
        # Defer the heavy import so a missing diffusers / torch never
        # blocks lorahub at import time.
        try:
            from diffusers import AutoPipelineForText2Image  # noqa: PLC0415
        except ImportError as exc:
            raise InferenceUnavailable(
                "diffusers is not installed; cannot render preview"
            ) from exc

        pipe = self._load_pipeline(AutoPipelineForText2Image)
        # Attach the freshly written LoRA. ``load_lora_weights`` handles
        # the safetensors format; we unload before re-loading so a new
        # checkpoint cleanly replaces the previous one. Some pipelines
        # don't support unload (or none has been loaded yet) — either
        # way, the suppress contains it.
        with contextlib.suppress(Exception):
            pipe.unload_lora_weights()
        try:
            pipe.load_lora_weights(str(lora_path))
        except Exception as exc:  # noqa: BLE001
            raise InferenceUnavailable(
                f"diffusers could not load LoRA weights from {lora_path}: {exc}"
            ) from exc

        kwargs: dict[str, Any] = {
            "prompt": spec.prompt,
            "num_inference_steps": spec.steps or default_steps,
            "guidance_scale": spec.cfg or default_cfg,
            "width": spec.width,
            "height": spec.height,
        }
        if spec.negative:
            kwargs["negative_prompt"] = spec.negative
        if spec.seed is not None:
            try:
                import torch  # noqa: PLC0415

                kwargs["generator"] = torch.Generator(
                    device=self.device or "cpu"
                ).manual_seed(spec.seed)
            except ImportError:
                # Fall through without a generator — diffusers will
                # default-seed itself.
                pass
        if self.lora_strength != 1.0:
            kwargs["cross_attention_kwargs"] = {"scale": self.lora_strength}

        result = pipe(**kwargs)
        # diffusers returns a pipeline-output object with `.images`.
        images = getattr(result, "images", None)
        if not images:
            raise InferenceUnavailable("diffusers pipeline returned no images")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(out_path, format="PNG", optimize=True)
        log.info(
            "[diffusers-inference] rendered %s (arch=%s, base=%s)",
            out_path.name,
            self.arch,
            self.base_model_id or _ARCH_TO_DEFAULT_REPO.get(self.arch, "?"),
        )

    # --------------------------------------------------------------------- #
    # Pipeline cache
    # --------------------------------------------------------------------- #

    def _load_pipeline(self, auto_pipeline: Any) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        repo = self.base_model_id or _ARCH_TO_DEFAULT_REPO.get(self.arch)
        if not repo:
            raise InferenceUnavailable(
                f"no diffusers default repo registered for arch={self.arch!r}"
            )
        log.info("[diffusers-inference] loading %s for arch=%s", repo, self.arch)
        try:
            import torch  # noqa: PLC0415

            dtype = (
                torch.float16
                if torch.cuda.is_available()
                else torch.float32
            )
            pipe = auto_pipeline.from_pretrained(repo, torch_dtype=dtype)
        except ImportError:
            pipe = auto_pipeline.from_pretrained(repo)
        if self.device is not None:
            try:
                pipe = pipe.to(self.device)
            except Exception:  # noqa: BLE001
                log.warning(
                    "[diffusers-inference] could not move pipeline to %s; staying on default device",
                    self.device,
                )
        self._pipeline = pipe
        return pipe


class InferenceUnavailable(RuntimeError):  # noqa: N818
    """Raised when diffusers prerequisites are missing at render time.

    Distinct from a generic ``InferenceFailed`` so the worker can decide
    later whether to surface this as an error or as a one-shot
    ``preview_unavailable`` event. Today the worker treats it as a
    regular failure log.
    """


# --------------------------------------------------------------------------- #
# Registry hook — picked up by lorahub.core.inference at import time.
# --------------------------------------------------------------------------- #


def _factory(*, arch: str, config: Any, workspace: Path | None) -> Any:
    if arch in _UNSUPPORTED_ARCHES:
        return None
    if arch not in _ARCH_TO_DEFAULT_REPO:
        return None
    # Pull the base-model checkpoint hint from the config so the user's
    # locally pinned model wins over the conservative default repo id.
    base_id: str | None = None
    if config is not None:
        try:
            ckpt = getattr(config.base_model, "checkpoint", None)
            if ckpt is not None:
                ckpt_path = Path(str(ckpt))
                # Use a local checkpoint path verbatim if it exists; otherwise
                # treat the string as a repo id (huggingface short form).
                base_id = str(ckpt_path) if ckpt_path.exists() else str(ckpt)
        except AttributeError:
            base_id = None
    backend = DiffusersInferenceBackend(arch=arch, base_model_id=base_id)
    if not backend.is_available(arch=arch):
        return None
    return backend


register_backend("diffusers", _factory)
