"""Runtime / training-loop knobs (attention, dataloader, optimisation, output)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ._shared import _CAMEL_CONFIG


class AttentionConfig(BaseModel):
    """Selects the attention kernel for the training forward+backward pass.

    SageAttention and other backward-incompatible kernels are not in
    this enum because there is no separate sampling-stage attention
    selector — sample images reuse this same training backend. If the
    backend's forward is unsafe in eval mode, the trainer surfaces it
    as a launch warning. ``flash3`` / ``flash4`` require Hopper /
    Blackwell hardware respectively; the runtime gates them by
    compute-capability and falls back to ``flash`` (FlashAttention 2)
    when the host can't run the chosen kernel.
    """

    model_config = _CAMEL_CONFIG

    training: Literal[
        "auto",     # pick the best available kernel for this GPU
        "torch",    # naive torch attention — debugging only
        "sdpa",     # F.scaled_dot_product_attention (PyTorch native)
        "flex",     # torch.nn.attention.flex_attention (PyTorch 2.5+)
        "xformers",
        "flash",    # FlashAttention 2 (Ampere/Ada/Hopper)
        "flash3",   # FlashAttention 3 (Hopper-only)
        "flash4",   # FlashAttention 4 beta (Hopper/Blackwell)
    ] = "auto"
    # Some backends require a memory-split attention path on kohya
    # (notably xformers). Mirrors --split_attn.
    split: bool = False


class DataLoaderConfig(BaseModel):
    """DataLoader / cache pipeline knobs.

    kohya: --max_data_loader_n_workers, --persistent_data_loader_workers,
    --vae_batch_size, --text_encoder_batch_size. dp: caching_batch_size,
    map_num_proc.
    """

    model_config = _CAMEL_CONFIG

    num_workers: int = Field(8, ge=0)
    persistent_workers: bool = False
    vae_batch_size: int = Field(1, ge=1)
    text_encoder_batch_size: int | None = Field(default=None, ge=1)
    cache_shuffle_num: int = Field(0, ge=0)
    map_num_proc: int | None = Field(default=None, ge=1)


class AugmentationConfig(BaseModel):
    """Image augmentation (kohya only). dp doesn't currently consume any of these."""

    model_config = _CAMEL_CONFIG

    flip: bool = False
    color: bool = False
    random_crop: bool = False
    # `min_face_size,target_size,max_face_size` triple, kohya format.
    face_crop_aug_range: str | None = None
    # Use image alpha channel as masked-loss mask.
    alpha_mask: bool = False


class OptimizationConfig(BaseModel):
    """Training-time speed / VRAM knobs that sit alongside the optimiser.

    Each field lines up with an upstream argv or TOML key the per-backend
    compilers already know how to emit. Defaults match upstream defaults
    so existing configs keep producing identical commands.
    """

    model_config = _CAMEL_CONFIG

    # PyTorch 2 graph compilation. kohya: --torch_compile. dp:
    # `pipeline_model.compile(dynamic=True)` is currently unconditional
    # in upstream's train.py, so dp ignores this knob (kept for parity
    # of UI/config shape).
    torch_compile: bool = False
    # Fused LoRA backward + optimizer step. kohya: --fused_backward_pass.
    # Saves one gradient buffer; LoRA-compatible. dp does not have an
    # equivalent argv yet (its DeepSpeed pipeline orders bwd/step
    # internally) — passing this through dp is a no-op.
    fused_backward_pass: bool = False
    # Train all parameters in bf16 (model + grads + optimizer states).
    # kohya: --full_bf16. dp: optim_dtype="bf16".
    full_bf16: bool = False
    # Same idea, fp16 path (older GPUs).
    full_fp16: bool = False
    # Number of transformer blocks temporarily offloaded to CPU during
    # the forward pass. kohya FLUX/SD3: --blocks_to_swap. dp:
    # `blocks_to_swap` in TOML's [general] block (already supported via
    # backend.diffusion_pipe.blocks_to_swap; this top-level mirror lets
    # us share configs across backends without two source-of-truth keys).
    blocks_to_swap: int = Field(0, ge=0)
    # FP8 base model weight load (FLUX / SD3 / HunyuanImage). VRAM -40%.
    fp8_base: bool = False
    fp8_base_unet: bool = False
    # HunyuanImage scaled FP8 (different math from --fp8_base).
    fp8_scaled: bool = False
    # HunyuanImage VL text encoder in FP8.
    fp8_vl_text_encoder: bool = False
    # Memory-strategy hints (kohya).
    lowram: bool = False
    highvram: bool = False
    # SDXL VAE in fp32 (the half-precision VAE corrupts colours on some
    # SDXL finetunes; this is the canonical workaround).
    no_half_vae: bool = False
    # safetensors loading without mmap (NFS / network filesystems).
    disable_mmap_load_safetensors: bool = False
    # Gradient checkpointing offloaded to CPU (kohya).
    cpu_offload_checkpointing: bool = False
    # Anima-specific unsloth-flavoured offload.
    unsloth_offload_checkpointing: bool = False
    # Cache text encoder outputs to RAM / disk (kohya). Disk version frees
    # VRAM completely and lets 6GB cards train SDXL.
    cache_text_encoder_outputs: bool = False
    cache_text_encoder_outputs_to_disk: bool = False


class OutputConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    name: str = "lora_output"
    save_every_n_epochs: int = Field(1, ge=1)
    # Step-level save cadence (kohya / dp).
    save_every_n_steps: int | None = Field(default=None, ge=1)
    # dp: examples-level save cadence.
    save_every_n_examples: int | None = Field(default=None, ge=1)
    # Retain only the most recent N checkpoints (kohya).
    save_last_n_epochs: int | None = Field(default=None, ge=1)
    save_last_n_steps: int | None = Field(default=None, ge=1)
    save_dtype: Literal["fp16", "bf16", "float"] = "fp16"
    output_dir: Path | None = None
    # Free-form metadata stamped onto the LoRA file. kohya: --metadata_*.
    training_comment: str | None = None
    no_metadata: bool = False
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_output_name(cls, value: str) -> str:
        name = value.strip()
        invalid_chars = set('<>:"/\\|?*')
        reserved = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }
        if not name or name in {".", ".."}:
            raise ValueError("output name is required")
        if len(name) > 96:
            raise ValueError("output name must be at most 96 characters")
        if name[-1] in {" ", "."}:
            raise ValueError("output name cannot end with a space or dot")
        if any(char in invalid_chars or ord(char) < 32 for char in name):
            raise ValueError("output name contains an invalid path character")
        if name.split(".", 1)[0].upper() in reserved:
            raise ValueError("output name is reserved by Windows")
        return name
