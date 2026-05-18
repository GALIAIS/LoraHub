"""Save-pipeline orchestrator for the LoRA / Ortho / Hydra / DoRA family.

The per-variant save logic — Cayley distillation, MoE write layout,
DoRA/qkv defuse — lives on the variant's module class in
``networks/lora_modules/`` (``OrthoLoRAModule.distill_save_state_dict``,
``HydraLoRAModule.build_moe_state_dict``, etc). This file is the thin
ordering layer that calls them and writes the resulting file(s).

Ordering of the conversion pipeline is load-bearing:

  1. ``ChimeraHydraLoRAModule.distill_save_state_dict``
     (gated on co-located ``.S_q_c`` + ``.S_q_f``)
  2. ``StackedExpertsLoRAModule.distill_save_state_dict``
     (gated on 3-D ``.S_p`` AND 3-D ``.S_q``)
  3. ``OrthoHydraLoRAModule.distill_save_state_dict``
     (gated on 3-D ``.S_p`` AND 2-D ``.S_q``)
  4. ``OrthoLoRAModule.distill_save_state_dict``
     (gated on 2-D ``.S_p``)
  5. legacy sig-type OrthoLoRA → standard LoRA
     (gated on ``.base_lambda``; kept here because it touches the
     deprecated ``lora_deprecated.OrthoLoRAModule`` save layout that no
     live module class owns)

The ``.S_p`` / ``.S_q`` dimensionality is the discriminator — every step
checks both dims explicitly so the matchers never overlap on the same
prefix. Step 5 handles legacy checkpoints from
``lora_deprecated.OrthoLoRAModule``; current training never emits those
keys, but the converter is kept so old artifacts remain re-bakeable.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch

from library.log import setup_logging
from networks.lora_modules import (
    ChimeraHydraLoRAModule,
    HydraLoRAModule,
    OrthoHydraLoRAModule,
    OrthoLoRAModule,
    StackedExpertsLoRAModule,
)
from networks.lora_modules.lora import rename_dora_and_defuse_standard

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy: sig-type OrthoLoRA → standard LoRA via 2r-dim SVD.
#
# Kept here (not on a module class) because the live ``OrthoLoRAModule``
# never emits these keys — they belong to the deprecated
# ``lora_deprecated.OrthoLoRAModule``, which is gone from the runtime path.
# ---------------------------------------------------------------------------


def _convert_legacy_ortho_to_lora(
    state_dict: Dict[str, torch.Tensor], dtype: Optional[torch.dtype]
) -> None:
    prefixes = set()
    for key in state_dict.keys():
        if key.endswith(".base_lambda"):
            prefixes.add(key[: -len(".base_lambda")])

    for prefix in prefixes:
        P = state_dict[f"{prefix}.p_layer.weight"]  # (out, r)
        Q = state_dict[f"{prefix}.q_layer.weight"]  # (r, in)
        lam = state_dict[f"{prefix}.lambda_layer"]
        P_base = state_dict[f"{prefix}.base_p_weight"]
        Q_base = state_dict[f"{prefix}.base_q_weight"]
        lam_base = state_dict[f"{prefix}.base_lambda"]
        alpha = state_dict.get(f"{prefix}.alpha")
        rank = Q.shape[0]

        # ΔW = P·diag(λ)·Q − P_base·diag(λ_base)·Q_base is rank ≤ 2r. SVD
        # works in the small 2r-dim column/row space instead of on the full
        # (out × in) matrix: ΔW = [P|P_base] @ M @ [Q; Q_base], then SVD of M.
        svd_device = "cuda" if torch.cuda.is_available() else "cpu"
        save_dtype = dtype if dtype is not None else P.dtype

        P_cat = torch.cat([P, P_base], dim=1).float().to(svd_device)  # (out, 2r)
        Q_cat = torch.cat([Q, Q_base], dim=0).float().to(svd_device)  # (2r, in)
        lam_diag = torch.diag(lam.squeeze(0).float().to(svd_device))
        lam_base_diag = torch.diag(lam_base.squeeze(0).float().to(svd_device))

        M = torch.zeros(2 * rank, 2 * rank, device=svd_device)
        M[:rank, :rank] = lam_diag
        M[rank:, rank:] = -lam_base_diag

        Qp, Rp = torch.linalg.qr(P_cat)
        Qq, Rq = torch.linalg.qr(Q_cat.T)

        core = Rp @ M @ Rq.T
        Uc, Sc, Vhc = torch.linalg.svd(core)

        lora_up = (
            (Qp @ Uc[:, :rank] * Sc[:rank].sqrt().unsqueeze(0))
            .to(save_dtype)
            .cpu()
            .contiguous()
        )
        lora_down = (
            (Sc[:rank].sqrt().unsqueeze(1) * Vhc[:rank, :] @ Qq.T)
            .to(save_dtype)
            .cpu()
            .contiguous()
        )

        for suffix in (
            "p_layer.weight",
            "q_layer.weight",
            "lambda_layer",
            "base_p_weight",
            "base_q_weight",
            "base_lambda",
        ):
            state_dict.pop(f"{prefix}.{suffix}", None)

        state_dict[f"{prefix}.lora_up.weight"] = lora_up
        state_dict[f"{prefix}.lora_down.weight"] = lora_down
        if alpha is not None:
            state_dict[f"{prefix}.alpha"] = alpha


# ---------------------------------------------------------------------------
# Back-compat shim: tests/test_global_router.py imports this name directly
# to exercise the StackedExperts MoE writer in isolation.
# ---------------------------------------------------------------------------


def _build_stacked_experts_state_dict(
    state_dict: Dict[str, torch.Tensor],
    dtype: Optional[torch.dtype],
) -> Dict[str, torch.Tensor]:
    """Thin shim → :meth:`StackedExpertsLoRAModule.build_moe_state_dict`."""
    return StackedExpertsLoRAModule.build_moe_state_dict(state_dict, dtype)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def save_network_weights(
    state_dict: Dict[str, torch.Tensor],
    *,
    file: str,
    dtype: Optional[torch.dtype],
    metadata: Optional[Dict[str, str]],
    save_variant: str,
) -> None:
    """Run the full save pipeline: distill chain + variant write.

    Mutates ``state_dict`` in place.
    """
    if metadata is not None and len(metadata) == 0:
        metadata = None

    # Distill chain. Order is load-bearing — see module docstring.
    ChimeraHydraLoRAModule.distill_save_state_dict(state_dict, dtype)
    StackedExpertsLoRAModule.distill_save_state_dict(state_dict, dtype)
    OrthoHydraLoRAModule.distill_save_state_dict(state_dict, dtype)
    OrthoLoRAModule.distill_save_state_dict(state_dict, dtype)
    _convert_legacy_ortho_to_lora(state_dict, dtype)

    # Variant dispatch.
    #   * ``stacked_experts_global_fei``: independent-A per-expert
    #     ``(lora_downs.{i}, lora_ups.{i})`` → ``*_moe.safetensors``.
    #   * ``chimera_hydra_moe``: dual-A per-pool ``lora_{down,up}_{c,f}`` +
    #     ``freq_router.*`` → ``*_chimera.safetensors``.
    #   * ``hydra_moe`` / ``ortho_hydra_to_hydra``: shared-A Hydra
    #     ``(lora_down, lora_ups.{i})`` → ``*_moe.safetensors``.
    #   * standard: rename DoRA + defuse qkv → ``*.safetensors``.
    #
    # Auto-fallback for hydra: any ``.lora_up_weight`` key surviving the
    # distill chain implies a Hydra payload. Kept for callers that don't
    # plumb ``save_variant`` through.
    is_stacked_experts_variant = save_variant == "stacked_experts_global_fei"
    is_chimera_variant = save_variant == "chimera_hydra_moe"
    is_hydra_variant = (
        save_variant in ("hydra_moe", "ortho_hydra_to_hydra")
        or (
            not is_chimera_variant
            and any(k.endswith(".lora_up_weight") for k in state_dict.keys())
        )
    ) and not is_stacked_experts_variant

    if is_stacked_experts_variant:
        se_file = os.path.splitext(file)[0] + "_moe.safetensors"
        se_sd = StackedExpertsLoRAModule.build_moe_state_dict(state_dict, dtype)
        from safetensors.torch import save_file as sf_save

        sf_save(se_sd, se_file, metadata or {})
        logger.info(f"StackedExperts full format saved to {se_file}")
        return

    if is_chimera_variant:
        chimera_file = os.path.splitext(file)[0] + "_chimera.safetensors"
        chimera_sd = ChimeraHydraLoRAModule.build_moe_state_dict(
            state_dict, dtype
        )
        from safetensors.torch import save_file as sf_save

        sf_save(chimera_sd, chimera_file, metadata or {})
        logger.info(f"ChimeraHydra full format saved to {chimera_file}")
        return

    if is_hydra_variant:
        hydra_file = os.path.splitext(file)[0] + "_moe.safetensors"
        hydra_sd = HydraLoRAModule.build_moe_state_dict(state_dict, dtype)
        from safetensors.torch import save_file as sf_save

        sf_save(hydra_sd, hydra_file, metadata or {})
        logger.info(f"HydraLoRA full format saved to {hydra_file}")
        # The _moe file is the only useful artifact for HydraLoRA —
        # a uniform expert average defeats layer-local routing.
        return

    # Standard (lora / ortho / dora) write path.
    rename_dora_and_defuse_standard(state_dict)

    if dtype is not None:
        for key in list(state_dict.keys()):
            v = state_dict[key].detach().clone().to("cpu").to(dtype)
            state_dict[key] = v

    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import save_file
        from library.training.hashing import precalculate_safetensors_hashes

        if metadata is None:
            metadata = {}
        model_hash, legacy_hash = precalculate_safetensors_hashes(
            state_dict, metadata
        )
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

        save_file(state_dict, file, metadata)
    else:
        torch.save(state_dict, file)
