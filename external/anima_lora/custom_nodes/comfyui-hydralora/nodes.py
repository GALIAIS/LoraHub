"""Anima ComfyUI custom nodes.

Three single-purpose loader nodes that each take a MODEL, apply one kind
of Anima-trained intervention, and return a MODEL. Chain them when a
workflow needs more than one.

  - ``AnimaAdapterLoader``: LoRA / HydraLoRA / ReFT (auto-detected from
    the safetensors keys + metadata). Installs ComfyUI weight patches for
    plain LoRA, per-Linear forward hooks for HydraLoRA live routing
    (σ-conditional and/or FeRA-style FEI-conditional on the Hydra stack),
    and per-block forward hooks for ReFT.
  - ``AnimaFeraLoader``: author-faithful FeRA (Yin et al., arXiv:2511.17979)
    — global router on the latent's spectral energy + per-Linear stacked
    independent experts. Different network family from
    ``AnimaAdapterLoader``'s Hydra/FEI variant: incompatible save format,
    mutually exclusive with HydraLoRA-moe (load one, not both).
  - ``AnimaPostfixLoader``: prefix / postfix / cond context splicing.
    Wraps ``diffusion_model.forward`` to splice learned vectors into the
    T5-compatible crossattn embedding after the LLM adapter, CFG-safe via
    ``cond_or_uncond``.

Adapter and postfix loaders were previously bundled in a single node
with toggle booleans; they were split in v3.0.0 so each does one thing
and users can bypass / reorder them with ComfyUI's standard MODEL-chain
wiring. ``AnimaFeraLoader`` was added in v3.1.0.
"""

import folder_paths

from .adapter import apply_adapter
from .fera import apply_fera
from .postfix import apply_postfix


class AnimaAdapterLoader:
    """Apply an Anima adapter (LoRA / HydraLoRA / ReFT) to a MODEL.

    Auto-detects which components the safetensors file contains and
    routes each to its correct application path:

      - Plain LoRA → ``ModelPatcher.add_patches``
      - HydraLoRA → per-Linear ``forward_hook`` (live router replay,
        including σ-conditional bias and FeRA-style FEI routing when
        the checkpoint's metadata declares ``ss_use_fei_router=true``)
      - ReFT → per-block ``forward_hook`` on the DiT's blocks

    Postfix / prefix / cond files load through ``AnimaPostfixLoader``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "adapter": (
                    loras,
                    {
                        "tooltip": (
                            "Anima adapter file. May contain any combination "
                            "of LoRA, HydraLoRA (*_moe.safetensors), "
                            "ChimeraHydra (*_chimera.safetensors — dual-pool "
                            "content + frequency routing), and ReFT "
                            "(residual-stream) weights."
                        )
                    },
                ),
                "strength_lora": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Strength for LoRA / Hydra weight patches.",
                    },
                ),
                "strength_reft": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Strength for ReFT residual-stream edits.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Anima adapter loader. Auto-detects LoRA / HydraLoRA / ChimeraHydra "
        "/ ReFT in the safetensors file. HydraLoRA installs per-Linear "
        "forward hooks that compute the trained per-sample router gate "
        "from each Linear's input and blend per-expert lora_up heads — "
        "full live routing including σ-conditional bias and FeRA-style "
        "FEI-conditional content routing when the checkpoint declares "
        "it. ChimeraHydra (*_chimera.safetensors, ss_use_chimera_hydra=true) "
        "additionally runs a network-level FreqRouter on FEI+σ each step, "
        "splits experts into content (K_c, per-Linear) + frequency (K_f, "
        "global) pools, and dispatches the concat gate through the same "
        "Hydra einsum. ReFT installs per-block forward hooks. For prefix "
        "/ postfix / cond context splicing, chain an AnimaPostfixLoader "
        "after this node."
    )

    def apply(self, model, adapter, strength_lora, strength_reft):
        new_model = model.clone()
        file_path = folder_paths.get_full_path("loras", adapter)
        apply_adapter(new_model, file_path, strength_lora, strength_reft)
        return (new_model,)


class AnimaFeraLoader:
    """Apply a FeRA adapter to a MODEL — either author-faithful or plan2.

    FeRA (Yin et al., arXiv:2511.17979): one **global router** consumes
    the latent's Frequency-Energy Indicator each denoising step and emits
    a single ``(B, num_experts)`` gate that every adapted Linear reuses
    for that step. Each adapted Linear carries **independent** stacked
    low-rank experts (``lora_down: (E, r, in)``, ``lora_up: (E, out, r)``)
    and adds ``Σ_k w_k · U_k @ D_k @ x`` to the frozen base.

    Loads two save formats with identical inference semantics:

      * Author-faithful (``networks.methods.fera``) — N-band FEI,
        ``router.net.*``, stacked-Parameter ``lora_down``/``lora_up``.
      * Plan2 stacked-experts (``networks.lora_anima`` with
        ``ss_network_spec=stacked_experts_global_fei``) — 2-band FEI,
        ``global_router.net.*``, per-expert split
        ``lora_downs.{i}.weight`` / ``lora_ups.{i}.weight``.

    Distinct from ``AnimaAdapterLoader``'s FEI-on-Hydra variant: that
    one routes per-Linear on Hydra's shared-A stack, this one routes
    globally on independent experts. Mutually exclusive with HydraLoRA-
    moe at the inference layer — load one, not both.
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "adapter": (
                    loras,
                    {
                        "tooltip": (
                            "FeRA checkpoint — either author-faithful "
                            "(networks.methods.fera; router.net.* + "
                            "lora_unet_*.lora_down/lora_up) or plan2 "
                            "stacked_experts_global_fei (global_router.net.* + "
                            "lora_unet_*.lora_downs.{i}.weight / .lora_ups.{i}.weight, "
                            "typically named *_moe.safetensors). Both use "
                            "an independent-A stacked-expert layout with a "
                            "single network-level FEI router."
                        )
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": (
                            "Scales the gated expert correction added to "
                            "each adapted Linear (mirrors the training-side "
                            "multiplier; 0 short-circuits to the frozen base)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Anima FeRA loader (author-faithful — Yin et al., arXiv:2511.17979). "
        "Installs a single model-level forward_pre_hook that computes the "
        "per-step Frequency-Energy Indicator and global router gates, plus "
        "per-Linear forward_hooks that add the gated stacked-expert "
        "correction. Mutually exclusive with HydraLoRA — for FEI-on-Hydra "
        "checkpoints, use AnimaAdapterLoader. For prefix / postfix / cond, "
        "chain an AnimaPostfixLoader after this node."
    )

    def apply(self, model, adapter, strength):
        new_model = model.clone()
        file_path = folder_paths.get_full_path("loras", adapter)
        apply_fera(new_model, file_path, strength)
        return (new_model,)


class AnimaPostfixLoader:
    """Apply an Anima prefix / postfix / cond file to a MODEL.

    Wraps ``diffusion_model.forward`` to splice the learned vectors into
    the T5-compatible crossattn embedding after the LLM adapter + pad-to-512
    step. Positive-batch rows only (CFG-safe via ``cond_or_uncond`` from
    ``transformer_options``). Mode (prefix / postfix / cond) is
    auto-detected from the safetensors keys.

    Chain after ``AnimaAdapterLoader`` when a workflow needs both — the
    postfix wrapper sees the model with adapter modifications already in
    place.
    """

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "postfix": (
                    loras,
                    {
                        "tooltip": (
                            "Postfix / prefix / cond file (prefix_embeds, "
                            "postfix_embeds, or cond_mlp.* keys)."
                        )
                    },
                ),
                "strength_postfix": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Strength multiplier for the postfix vectors.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Anima postfix loader. Splices learned prefix / postfix / cond "
        "vectors into the T5-compatible crossattn embedding after the "
        "LLM adapter. Mode auto-detected from safetensors keys. "
        "Positive-batch only (CFG-safe). For LoRA / HydraLoRA / ReFT "
        "adapters, chain an AnimaAdapterLoader before this node."
    )

    def apply(self, model, postfix, strength_postfix):
        new_model = model.clone()
        file_path = folder_paths.get_full_path("loras", postfix)
        apply_postfix(new_model, file_path, strength_postfix)
        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "AnimaAdapterLoader": AnimaAdapterLoader,
    "AnimaFeraLoader": AnimaFeraLoader,
    "AnimaPostfixLoader": AnimaPostfixLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaAdapterLoader": "Anima Adapter Loader",
    "AnimaFeraLoader": "Anima FeRA Loader",
    "AnimaPostfixLoader": "Anima Postfix Loader",
}
