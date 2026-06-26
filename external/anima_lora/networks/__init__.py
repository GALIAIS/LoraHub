"""NetworkSpec registry for LoRA adapter-method dispatch.

Replaces the flag-cascade in ``networks.lora_anima.create_network`` with a
declarative map. Each entry pairs an adapter variant name with the module
class it instantiates and a ``save_variant`` label consumed by
``networks.lora_save``.

Flag precedence (evaluated top to bottom, first match wins):

    use_chimera_hydra                    → chimera_hydra
    use_ia3 / use_lokr / use_loha / use_full / use_diag_oft / use_boft /
    use_glora / use_vera                 → atomic decomposition variant
    use_dylora                           → dylora
    use_moe_style="independent_A"        → stacked_experts_global_fei
    use_moe_style="shared_A" + use_ortho → ortho_hydra
    use_moe_style="shared_A"             → hydra
    use_dora                             → dora
    use_ortho                            → ortho
    (none)                               → lora

The legacy ``use_hydra`` / ``use_sigma_router`` / ``use_fei_router``
kwargs were retired in plan2 task #6 — see ``LoRANetworkCfg.from_kwargs``
for the rejection message.

DoRA (Liu et al. ICML'24) was reintroduced as ``use_dora`` after the
LoraHub Phase-1 algorithm-set expansion (LoKr / LoHA / DoRA / IA3). It
shares the standard save layout with plain LoRA — the magnitude column
is renamed ``.dora_scale`` at save time so ComfyUI's stock LoRA loader
picks it up unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type

from networks.lora_modules import (
    BOFTModule,
    ChimeraHydraLoRAModule,
    DiagOFTModule,
    DoRAModule,
    DyLoRAModule,
    FullModule,
    GLoRAModule,
    HydraLoRAModule,
    IA3Module,
    LoHAModule,
    LoKrModule,
    LoRAModule,
    OrthoHydraLoRAModule,
    OrthoLoRAModule,
    StackedExpertsLoRAModule,
    VeRAModule,
)


@dataclass(frozen=True)
class NetworkSpec:
    """Descriptor for one adapter variant.

    Attributes:
        name: Stable identifier stamped on the network and written to
            metadata as ``ss_network_spec``. Also the key into
            ``NETWORK_REGISTRY``.
        module_class: Concrete ``LoRAModule`` subclass the network will
            instantiate per target module.
        save_variant: Label keyed into ``networks.lora_save.SAVE_HANDLERS``
            — selects the serialization pipeline for this variant.
        kwarg_flags: Tuple of kwargs this variant consumes beyond
            ``SHARED_KWARG_FLAGS``. Combined with the shared set by
            ``all_network_kwargs()`` to populate argparse schema and
            forward TOML-level args into ``create_network``. Single
            source of truth for what keys train.py recognizes.
        post_init: Optional hook run after the network is built; receives
            ``(network, kwargs)``. Used for variant-specific attribute
            attachment (e.g. hydra balance loss weight).
    """

    name: str
    module_class: Type
    save_variant: str = "standard"
    kwarg_flags: Tuple[str, ...] = ()
    post_init: Optional[Callable[[Any, Mapping[str, Any]], None]] = None


# Kwargs every LoRA-family variant consumes in ``create_network``: core
# targeting knobs + cross-cutting add-ons (ReFT, channel scaling,
# LoRA+, T-LoRA). Cross-cutting because these compose on top of any
# variant rather than belonging to a single one.
SHARED_KWARG_FLAGS: Tuple[str, ...] = (
    # Core network targeting / knobs
    "train_llm_adapter",
    "exclude_patterns",
    "include_patterns",
    "layer_start",
    "layer_end",
    "rank_dropout",
    "module_dropout",
    "verbose",
    # Regex-driven per-module rank / lr overrides
    "network_reg_dims",
    "network_reg_lrs",
    # HydraLoRA router (+ σ-router MLP) LR multiplier on top of unet_lr / reg_lr
    "network_router_lr_scale",
    # LoRA+
    "loraplus_lr_ratio",
    "loraplus_unet_lr_ratio",
    "loraplus_text_encoder_lr_ratio",
    # T-LoRA (timestep-dependent rank masking)
    "use_timestep_mask",
    "per_sample_timestep_mask",
    "min_rank",
    "alpha_rank_scale",
    # Per-channel input pre-scaling (SmoothQuant-style)
    "per_channel_scaling",
    "channel_stats_path",
    "channel_scaling_alpha",
    # Memory-saving down-projection autograd (classic LoRA only; bitwise-equal grads)
    "use_custom_down_autograd",
    # SVD-Down init for plain LoRA: "kaiming" | "weight_svd".
    "down_init",
    # Variant selectors (read by resolve_network_spec)
    # LyCORIS-compatible selector aliases. ``algo`` mirrors
    # lycoris.kohya's public network_arg surface; the ``use_*`` flags
    # below remain anima_lora's canonical internal selectors.
    "algo",
    "use_ortho",
    "use_dora",
    "use_ia3",
    "use_lokr",
    "use_loha",
    "use_dylora",
    "use_full",
    "use_diag_oft",
    "use_boft",
    "use_glora",
    "use_vera",
    # LoKr factor (only consumed when use_lokr=True). Picks the (a, c)
    # split of out_dim and (b, d) split of in_dim. Larger factor → more
    # expressive W₁ matrix, smaller LoRA leg.
    "lokr_factor",
    # LyCORIS-compatible spelling for LoKr factor. Normalized to
    # ``lokr_factor`` by ``_normalize_lycoris_kwargs``.
    "factor",
    # BOFT factor (only consumed when use_boft=True). Number of
    # butterfly stages composed into the rotation; ≥ log_2(out_dim)
    # is sufficient to span SO(out_dim). Default 4 follows upstream.
    "boft_factors",
    # PSOFT-style Cayley-init magnitude (consumed by OrthoHydra +
    # StackedExperts in ortho mode).
    "ortho_init_std",
    # Three-axis routing config (see plan2.md §three-axis-config). Drives
    # `LoRANetworkCfg.from_kwargs` translation; `resolve_network_spec` also
    # dispatches on `use_moe_style="independent_A"` → `stacked_experts_global_fei`.
    "use_moe_style",
    "route_per_layer",
    "router_source",
    # GlobalRouter knobs (consumed only when route_per_layer=False).
    "router_hidden_dim",
    "router_tau",
    # FECL knobs (FeRA auxiliary loss; opt-in via fera_fecl_weight > 0).
    "fera_fecl_weight",
    "fera_num_bands",
    # ReFT add-on (composes with any variant)
    "add_reft",
    "reft_dim",
    "reft_alpha",
    "reft_layers",
    # REPA-style auxiliary alignment (composes with any variant). The factory
    # ``_maybe_attach_repa_head`` reads ``use_repa`` to decide whether to attach
    # the head, then sizes/weights it from the rest. Selection of the hooked
    # block index and the loss weight live as top-level argparse flags
    # (``--repa_layer``, ``--repa_weight``) — the network module doesn't need
    # them, only the adapter / loss handler does.
    "use_repa",
    "repa_dit_dim",
    "repa_hidden_dim",
    "repa_encoder_dim",
    "repa_lr_scale",
)


_LYCORIS_ALGO_TO_FLAG: Dict[str, Optional[str]] = {
    "lora": None,
    "locon": None,
    "tlora": "use_timestep_mask",
    "asr_tlora": "use_timestep_mask",
    "loha": "use_loha",
    "lokr": "use_lokr",
    "ia3": "use_ia3",
    "dylora": "use_dylora",
    "full": "use_full",
    "diag_oft": "use_diag_oft",
    "diag-oft": "use_diag_oft",
    "boft": "use_boft",
    "glora": "use_glora",
}


def _post_init_hydra(network: Any, kwargs: Mapping[str, Any]) -> None:
    blw = kwargs.get("balance_loss_weight")
    target = float(blw) if blw is not None else 0.01
    warmup = kwargs.get("balance_loss_warmup_ratio")
    warmup_ratio = float(warmup) if warmup is not None else 0.0
    network._balance_loss_target_weight = target
    network._balance_loss_warmup_ratio = warmup_ratio
    # Hold the balance penalty at 0 during the warmup window so the router can
    # specialize before load-balancing kicks in; flipped to `target` by
    # LoRANetwork.step_balance_loss_warmup once global_step crosses the ratio.
    network._balance_loss_weight = 0.0 if warmup_ratio > 0.0 else target
    network._use_hydra = True
    # FECL weight surface: ``_fera_fecl_loss`` reads
    # ``ctx.network.fecl_weight``. Mirror the cfg value here (and fall back
    # to the kwarg when no cfg is present — keeps unit tests that hand a
    # network without a full cfg working).
    cfg_weight = getattr(getattr(network, "cfg", None), "fera_fecl_weight", None)
    if cfg_weight is not None:
        network.fecl_weight = float(cfg_weight)
    else:
        network.fecl_weight = float(kwargs.get("fera_fecl_weight", 0.0) or 0.0)

    # ChimeraHydra: stamp the chimera flag + per-pool balance weights for
    # ``get_balance_loss`` to consume. Falls back to the shared
    # ``balance_loss_weight`` (the OrthoHydra default) when the user didn't
    # set explicit per-pool weights — matches the proposal §"Balance loss"
    # ("``w_c`` keeps the current ortho-hydra value; ``w_f`` starts at the
    # same value, tunable").
    cfg = getattr(network, "cfg", None)
    if cfg is not None and getattr(cfg, "use_chimera_hydra", False):
        network._use_chimera_hydra = True
        w_c = cfg.balance_w_content if cfg.balance_w_content is not None else target
        w_f = cfg.balance_w_freq if cfg.balance_w_freq is not None else target
        network._balance_w_content = float(w_c)
        network._balance_w_freq = float(w_f)
    else:
        network._use_chimera_hydra = False


_HYDRA_KWARG_FLAGS: Tuple[str, ...] = (
    "num_experts",
    "balance_loss_weight",
    "balance_loss_warmup_ratio",
    "expert_init_std",
    # Unified layer filter — scopes which Linears participate in routed
    # adaptation (Hydra MoE leaves + σ / FEI feature concatenation).
    "router_targets",
    # σ-conditional router add-on (router_source="sigma")
    "sigma_feature_dim",
    "per_bucket_balance_weight",
    "num_sigma_buckets",
    "specialize_experts_by_sigma_buckets",
    "sigma_bucket_boundaries",
    # FEI-conditional router (router_source="fei"; FeRA-style content-aware)
    "fei_feature_dim",
    "fei_sigma_low_div",
)

_CHIMERA_KWARG_FLAGS: Tuple[str, ...] = (
    "use_chimera_hydra",
    "num_experts_content",
    "num_experts_freq",
    # Per-pool balance weights. Fall back to balance_loss_weight when unset.
    "balance_w_content",
    "balance_w_freq",
    # FreqRouter init magnitude (small N(0, std)) — non-zero so the freq
    # pool differentiates at step 0.
    "freq_router_init_std",
    # Per-modality LayerNorm on FreqRouter input. Active only when both
    # FEI and σ feature blocks are enabled — equalizes the variance budget
    # so the higher-dim σ block doesn't fan-in-overpower the 2-D FEI simplex.
    "freq_router_layer_norm",
    # Per-pool router LR multipliers — stack on top of network_router_lr_scale.
    # Defaults to 1.0 (no-op). Bump content when the per-layer router stays
    # near-uniform too long (std=0.01 init is slow to break symmetry).
    "network_content_router_lr_scale",
    "network_freq_router_lr_scale",
)


NETWORK_REGISTRY: Dict[str, NetworkSpec] = {
    "lora": NetworkSpec(
        name="lora",
        module_class=LoRAModule,
        save_variant="standard",
    ),
    "ortho": NetworkSpec(
        name="ortho",
        module_class=OrthoLoRAModule,
        save_variant="ortho_to_lora",
    ),
    # DoRA (Liu et al. ICML'24) — shares the ``standard`` save variant
    # because its delta layout (lora_down + lora_up + magnitude) maps
    # one-to-one onto the plain LoRA layout after the magnitude is
    # renamed ``.dora_scale`` (see ``rename_dora_keys`` in
    # ``lora_modules.lora``).
    "dora": NetworkSpec(
        name="dora",
        module_class=DoRAModule,
        save_variant="standard",
    ),
    # IA3 (Liu et al. NeurIPS'22) — per-output rescaling with no LoRA
    # legs. Saves under a dedicated ``ia3`` variant because the on-disk
    # tensor (``.ia3_weight``) doesn't share structure with the LoRA
    # / DoRA / Ortho families' ``lora_down`` + ``lora_up`` shape.
    "ia3": NetworkSpec(
        name="ia3",
        module_class=IA3Module,
        save_variant="ia3",
    ),
    # LoKr (LyCORIS, arXiv:2212.10650) — Kronecker decomposition.
    # Linear-only in the Phase-1 cut. Custom save_variant because the
    # on-disk keys are ``lokr_w1`` / ``lokr_w2_a`` / ``lokr_w2_b``,
    # neither shape nor name overlapping the LoRA family.
    "lokr": NetworkSpec(
        name="lokr",
        module_class=LoKrModule,
        save_variant="lokr",
    ),
    # LoHA (FedPara, arXiv:2108.06098) — Hadamard product of two LoRA
    # pairs. Linear-only here. Saves under ``loha`` for the same shape
    # mismatch reason as LoKr.
    "loha": NetworkSpec(
        name="loha",
        module_class=LoHAModule,
        save_variant="loha",
    ),
    # DyLoRA (Valipour et al. EACL'23) — random rank truncation during
    # training, full rank at inference. Saves under ``standard`` because
    # the on-disk shape is bit-identical to plain LoRA.
    "dylora": NetworkSpec(
        name="dylora",
        module_class=DyLoRAModule,
        save_variant="standard",
    ),
    # Full — free-Δ wrapper (not a LoRA). Saves under a dedicated
    # ``full`` variant; the on-disk key is ``delta`` (out × in) per
    # Linear, which the standard rename + qkv defuse path doesn't
    # touch.
    "full": NetworkSpec(
        name="full",
        module_class=FullModule,
        save_variant="full",
    ),
    # Diag-OFT (Qiu et al. NeurIPS'23) — block-diagonal orthogonal
    # rotation of the host weight via Cayley parameterisation. Saves
    # under ``oft`` because the on-disk key is ``oft_skew`` (K, r, r)
    # — neither shape nor name overlapping the LoRA family.
    "diag_oft": NetworkSpec(
        name="diag_oft",
        module_class=DiagOFTModule,
        save_variant="oft",
    ),
    # BOFT (Liu et al. arXiv:2311.06243) — m-stage butterfly
    # composition of block-diagonal orthogonals. Saves under ``boft``
    # for the same reason; key is ``boft_skew`` (m, K, r, r).
    "boft": NetworkSpec(
        name="boft",
        module_class=BOFTModule,
        save_variant="boft",
    ),
    # GLoRA-light (Chavan et al. arXiv:2306.07967) — LoRA + per-rank
    # diagonal gate. Saves under ``glora`` because the on-disk shape
    # adds a ``glora_gate`` (r,) sibling to the standard LoRA pair.
    "glora": NetworkSpec(
        name="glora",
        module_class=GLoRAModule,
        save_variant="glora",
    ),
    # VeRA (Kopiczko et al. ICLR'24, arXiv:2310.11454) — frozen random
    # A / B + two trainable scale vectors. Saves under ``vera``;
    # checkpoint carries A / B (persistent buffers) plus the scales.
    "vera": NetworkSpec(
        name="vera",
        module_class=VeRAModule,
        save_variant="vera",
    ),
    "hydra": NetworkSpec(
        name="hydra",
        module_class=HydraLoRAModule,
        save_variant="hydra_moe",
        kwarg_flags=_HYDRA_KWARG_FLAGS,
        post_init=_post_init_hydra,
    ),
    "ortho_hydra": NetworkSpec(
        name="ortho_hydra",
        module_class=OrthoHydraLoRAModule,
        save_variant="ortho_hydra_to_hydra",
        kwarg_flags=_HYDRA_KWARG_FLAGS,
        post_init=_post_init_hydra,
    ),
    # ChimeraHydra: dual-pool additive routing on the OrthoHydra Cayley
    # parameterization (proposal: docs/proposal/chimera_hydra.md). Training
    # builds Cayley params via ``ChimeraHydraLoRAModule``; save distills
    # them to the Hydra-MoE on-disk layout (shared ``lora_down`` + per-expert
    # ``lora_ups.{i}.weight``, q/k/v defused, top-level ``freq_router.*``)
    # written to a ``*_chimera.safetensors`` sibling. Load goes through
    # ``HydraLoRAModule`` with ``num_experts_content > 0`` (the dual-pool
    # runtime form added in this commit), so the Cayley class is training-
    # only — checkpoint resume silently loses the orthogonal parameterization
    # (matches the OrthoHydra → Hydra trade-off).
    "chimera_hydra": NetworkSpec(
        name="chimera_hydra",
        module_class=ChimeraHydraLoRAModule,
        save_variant="chimera_hydra_moe",
        kwarg_flags=_HYDRA_KWARG_FLAGS + _CHIMERA_KWARG_FLAGS,
        post_init=_post_init_hydra,
    ),
    # FeRA paper-faithful: independent-A stacked experts, single network-level
    # router fed by FEI(z_t). See plan2.md §three-axis-config — selected via
    # ``use_moe_style="independent_A"``. Save handler stub: task #4 wires the
    # real serialization path; until then this spec is reachable only via the
    # gui-methods/fera.toml variant (also a task #6 deliverable).
    "stacked_experts_global_fei": NetworkSpec(
        name="stacked_experts_global_fei",
        module_class=StackedExpertsLoRAModule,
        save_variant="stacked_experts_global_fei",
        kwarg_flags=_HYDRA_KWARG_FLAGS,
        post_init=_post_init_hydra,
    ),
}


def all_network_kwargs() -> Tuple[str, ...]:
    """Return the union of shared + per-variant kwargs, sorted.

    Single source of truth for train.py — populates the argparse schema
    and the TOML → ``net_kwargs`` forwarding list, so adding a new kwarg
    to a ``NetworkSpec`` (or to ``SHARED_KWARG_FLAGS``) automatically
    makes it visible to training without touching train.py.
    """
    merged: set[str] = set(SHARED_KWARG_FLAGS)
    for spec in NETWORK_REGISTRY.values():
        merged.update(spec.kwarg_flags)
    return tuple(sorted(merged))


def _parse_bool_flag(kwargs: Mapping[str, Any], key: str) -> bool:
    v = kwargs.get(key, False)
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).lower() == "true"


def _normalize_lycoris_kwargs(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Accept LyCORIS/kohya style ``network_args`` on anima_lora.

    LoraHub's typed compiler emits the canonical ``use_lokr=true`` style
    flags. Direct users often bring configs from ``lycoris.kohya`` with
    ``algo=lokr`` / ``factor=8``. Normalize those aliases here so both
    surfaces hit the same native anima_lora modules and save handlers.
    """
    normalized = dict(kwargs)
    raw_algo = normalized.get("algo")
    if raw_algo is not None and str(raw_algo).strip():
        algo = str(raw_algo).strip().lower()
        if algo.startswith("lycoris_"):
            algo = algo.removeprefix("lycoris_")
        if algo not in _LYCORIS_ALGO_TO_FLAG:
            raise ValueError(
                f"LyCORIS algo={raw_algo!r} is not supported by this "
                "anima_lora build. Supported values: "
                f"{sorted(_LYCORIS_ALGO_TO_FLAG)}"
            )
        flag = _LYCORIS_ALGO_TO_FLAG[algo]
        if flag is not None and flag not in normalized:
            normalized[flag] = "true"

    if "factor" in normalized and "lokr_factor" not in normalized:
        normalized["lokr_factor"] = normalized["factor"]

    return normalized


def resolve_network_spec(kwargs: Mapping[str, Any]) -> NetworkSpec:
    """Resolve which NetworkSpec to instantiate from create_network kwargs.

    Precedence is deterministic and documented in the module docstring.
    Raises on mutually-exclusive combinations.

    Honors the ``use_moe_style`` axis (plan2.md §three-axis-config):
    ``"independent_A"`` routes to ``stacked_experts_global_fei`` (FeRA);
    ``"shared_A"`` plus ``use_ortho`` routes to ``ortho_hydra``; bare
    ``"shared_A"`` routes to ``hydra``. The legacy ``use_hydra`` kwarg was
    retired in plan2 task #6 — ``LoRANetworkCfg.from_kwargs`` raises if a
    TOML still carries it.

    ``use_chimera_hydra=True`` short-circuits to the chimera variant. The
    chimera config requires ``use_moe_style="shared_A"`` semantics under
    the hood (OrthoHydra parameterization), but uses K_c + K_f instead of
    a single ``num_experts`` — the user only sets the chimera flag.
    """
    kwargs = _normalize_lycoris_kwargs(kwargs)

    use_ortho = _parse_bool_flag(kwargs, "use_ortho")
    use_dora = _parse_bool_flag(kwargs, "use_dora")
    use_ia3 = _parse_bool_flag(kwargs, "use_ia3")
    use_lokr = _parse_bool_flag(kwargs, "use_lokr")
    use_loha = _parse_bool_flag(kwargs, "use_loha")
    use_dylora = _parse_bool_flag(kwargs, "use_dylora")
    use_full = _parse_bool_flag(kwargs, "use_full")
    use_diag_oft = _parse_bool_flag(kwargs, "use_diag_oft")
    use_boft = _parse_bool_flag(kwargs, "use_boft")
    use_glora = _parse_bool_flag(kwargs, "use_glora")
    use_vera = _parse_bool_flag(kwargs, "use_vera")
    use_chimera = _parse_bool_flag(kwargs, "use_chimera_hydra")
    if use_chimera:
        return NETWORK_REGISTRY["chimera_hydra"]
    # Atomic-decomposition variants: each owns its own forward and
    # save layout, so they can't compose with LoRA legs / Ortho /
    # MoE / DoRA. Reject the combo loudly.
    _atomic = {
        "use_ia3": use_ia3,
        "use_lokr": use_lokr,
        "use_loha": use_loha,
        "use_full": use_full,
        "use_diag_oft": use_diag_oft,
        "use_boft": use_boft,
        "use_glora": use_glora,
        "use_vera": use_vera,
    }
    enabled_atomic = [k for k, v in _atomic.items() if v]
    if len(enabled_atomic) > 1:
        raise ValueError(
            f"Mutually exclusive variants enabled: {enabled_atomic}"
        )
    if enabled_atomic:
        if use_dora or use_ortho or use_dylora:
            raise ValueError(
                f"{enabled_atomic[0]}=True is mutually exclusive with "
                "use_dora / use_ortho / use_dylora (no LoRA legs to compose with)."
            )
        raw_moe = kwargs.get("use_moe_style")
        if raw_moe and str(raw_moe).lower() not in ("false", "none", ""):
            raise ValueError(
                f"{enabled_atomic[0]}=True is mutually exclusive with "
                f"use_moe_style={raw_moe!r}."
            )
        # Strip the ``use_`` prefix to get the registry key.
        return NETWORK_REGISTRY[enabled_atomic[0][len("use_") :]]
    if use_dylora:
        if use_dora or use_ortho:
            raise ValueError(
                "use_dylora=True is mutually exclusive with use_dora / use_ortho"
                " (rank-truncation prefix doesn't compose with magnitude / Cayley distill)."
            )
        raw_moe = kwargs.get("use_moe_style")
        if raw_moe and str(raw_moe).lower() not in ("false", "none", ""):
            raise ValueError(
                "use_dylora=True is mutually exclusive with use_moe_style"
                f"={raw_moe!r} (per-expert ups don't share a single rank prefix)."
            )
        return NETWORK_REGISTRY["dylora"]

    raw_moe = kwargs.get("use_moe_style")
    if isinstance(raw_moe, str):
        moe_style = raw_moe.strip()
        if moe_style.lower() in ("false", "none", ""):
            moe_style = ""
    elif raw_moe is False or raw_moe is None:
        moe_style = ""
    else:
        raise ValueError(
            f"use_moe_style={raw_moe!r}: expected False, 'shared_A', or 'independent_A'."
        )
    if moe_style not in ("", "shared_A", "independent_A"):
        raise ValueError(
            f"use_moe_style={raw_moe!r}: expected False, 'shared_A', or 'independent_A'."
        )

    if moe_style == "independent_A":
        if use_dora:
            raise ValueError(
                "use_dora=True is mutually exclusive with use_moe_style='independent_A'"
                " (FeRA's stacked experts do not share the DoRA magnitude layout)."
            )
        return NETWORK_REGISTRY["stacked_experts_global_fei"]
    if moe_style == "shared_A":
        if use_dora:
            raise ValueError(
                "use_dora=True is mutually exclusive with use_moe_style='shared_A'"
                " (Hydra's per-expert ups don't compose with DoRA's per-Linear magnitude)."
            )
        return (
            NETWORK_REGISTRY["ortho_hydra"] if use_ortho else NETWORK_REGISTRY["hydra"]
        )
    if use_dora:
        if use_ortho:
            raise ValueError(
                "use_dora=True with use_ortho=True is not supported — DoRA's"
                " magnitude column doesn't round-trip through OrthoLoRA's"
                " Cayley distill keys. Pick one."
            )
        return NETWORK_REGISTRY["dora"]
    if use_ortho:
        return NETWORK_REGISTRY["ortho"]
    return NETWORK_REGISTRY["lora"]


__all__ = [
    "NetworkSpec",
    "NETWORK_REGISTRY",
    "SHARED_KWARG_FLAGS",
    "all_network_kwargs",
    "resolve_network_spec",
]
