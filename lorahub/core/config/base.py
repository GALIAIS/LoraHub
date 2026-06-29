"""Base model + architecture-specific component path configs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ._shared import _CAMEL_CONFIG


class ArchPathsConfig(BaseModel):
    """Arch-specific component paths shared by both backends.

    Empty by default — only set the fields your model actually uses.
    Both compilers consume the same fields and emit them under the names
    the corresponding upstream expects (kohya argv, dp TOML keys).
    """

    model_config = _CAMEL_CONFIG

    # FLUX / SD3 / FLUX2
    clip_l: Path | None = None
    clip_g: Path | None = None
    t5xxl: Path | None = None
    ae: Path | None = None  # FLUX autoencoder

    # Generic (Anima / Wan / HunyuanImage / chroma transformer-style)
    transformer: Path | None = None
    text_encoder: Path | None = None
    llm: Path | None = None  # Anima Qwen3, HunyuanVideo LLM
    byt5: Path | None = None  # HunyuanImage byT5

    # Anima-specific
    qwen3: Path | None = None
    t5_tokenizer: Path | None = None
    llm_adapter: Path | None = None

    # Token length caps
    t5xxl_max_token_length: int | None = Field(default=None, ge=1)
    qwen3_max_token_length: int | None = Field(default=None, ge=1)
    t5_max_token_length: int | None = Field(default=None, ge=1)

    # Attention masking + dropout — FLUX/SD3
    apply_t5_attn_mask: bool = False
    apply_lg_attn_mask: bool = False
    t5_dropout_rate: float = Field(0.0, ge=0.0, lt=1.0)
    clip_l_dropout_rate: float = Field(0.0, ge=0.0, lt=1.0)
    clip_g_dropout_rate: float = Field(0.0, ge=0.0, lt=1.0)

    # SD3 positional-embed crop
    pos_emb_random_crop_rate: float = Field(0.0, ge=0.0, lt=1.0)
    enable_scaled_pos_embed: bool = False

    # FLUX dev distilled guidance scale baked into the LoRA
    guidance_scale: float | None = Field(default=None, gt=0)

    # Place TE on a specific device / dtype (SD3 separates TEs from UNet)
    t5xxl_device: str | None = None
    t5xxl_dtype: Literal["fp16", "bf16", "fp32", "fp8"] | None = None

    # VAE memory tweaks (Anima / HunyuanImage / Wan)
    vae_chunk_size: int | None = Field(default=None, ge=1)
    vae_disable_cache: bool = False
    text_encoder_cpu: bool = False


class BaseModelConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    # The arch literal mirrors the union of upstream-supported model families
    # across kohya sd-scripts and diffusion-pipe. Backends are responsible for
    # rejecting arches they do not implement (kohya rejects dp-only entries
    # like `wan` or `hunyuan_video`; dp rejects kohya-only entries like sd15).
    # Old config values stay valid; new values follow each upstream's docs.
    arch: Literal[
        # kohya sd-scripts (README "Supported Models")
        "sd15",
        "sd2",
        "sdxl",
        "sd3",
        "flux",
        "lumina",
        "hunyuan_image",
        "anima",
        # diffusion-pipe (docs/supported_models.md) -- additional entries
        "flux2",
        "chroma",
        "hidream",
        "omnigen2",
        "auraflow",
        "qwen_image",
        "cosmos",
        "cosmos_predict2",
        "hunyuan_video",
        "hunyuan_video_15",
        "ltx_video",
        "ltx2",
        "wan",
        "z_image",
        "ernie_image",
        "krea2",
    ] = "sdxl"
    # SDXL sub-architectures sharing the SDXL backbone but trained on
    # different finetune lineages (Pony/Illustrious/NoobAI/Animagine).
    # Backends still treat these as SDXL; the variant only nudges
    # default learning rates and a couple of CLI flags.
    arch_variant: Literal["", "pony", "illustrious", "noobai", "animagine"] = ""
    checkpoint: Path
    vae: Path | None = None
    # Per-component checkpoint paths for arches that ship as multi-file
    # bundles (FLUX = clip_l + t5xxl + ae; SD3 = clip_l/g + t5xxl;
    # Anima = qwen3 + qwen_image_vae + transformer; HunyuanImage = byt5 +
    # text_encoder; ...). Free-form bag because upstream keeps adding
    # entries. The kohya/dp compilers each render the keys their
    # respective upstream understands.
    arch_paths: ArchPathsConfig = Field(default_factory=lambda: ArchPathsConfig())
