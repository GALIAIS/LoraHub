"""Frozen configuration object for ``LoRANetwork``.

Replaces the 25-arg ``LoRANetwork.__init__`` and the per-kwarg parse pile in
``factory.create_network`` / ``create_network_from_weights``. Two construction
sites — ``from_kwargs`` (fresh training; absorbs the str→bool/int/float casts
that train.py's ``net_kwargs`` produces) and ``from_weights`` (warm-start /
inference; values come from checkpoint key sniffing).

Frozen by intent: every field here is fixed for the run. Mutable runtime
state (``multiplier``, LoRA+ ratios, hit counters, σ caches, post-build attrs
written by ``spec.post_init``) stays as plain attributes on the network.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Type, Union

import torch

from networks.lora_modules import LoRAModule

# Three-axis routing config (see plan2.md §three-axis-config).
MoEStyle = Union[Literal[False], Literal["shared_A"], Literal["independent_A"]]
RouterSource = Literal["input", "sigma", "fei", "none"]

logger = logging.getLogger(__name__)


def _as_bool(value: Any, *, default: bool = False) -> bool:
    """Parse a kwarg that may arrive as ``"true"`` / ``"false"`` / bool / None."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _as_moe_style(value: Any) -> MoEStyle:
    """Parse the three-valued ``use_moe_style`` kwarg.

    Accepts: ``False`` / ``None`` / ``"false"`` / ``""`` → ``False``;
    the literal strings ``"shared_A"`` / ``"independent_A"`` pass through.
    """
    if value is None or value is False:
        return False
    if isinstance(value, str):
        v = value.strip()
        if v.lower() in ("false", "none", ""):
            return False
        if v in ("shared_A", "independent_A"):
            return v
    raise ValueError(
        f"use_moe_style={value!r}: expected False, 'shared_A', or 'independent_A'."
    )


def _as_router_source(value: Any) -> RouterSource:
    """Parse the ``router_source`` kwarg. Empty / None → ``"none"``."""
    if value is None:
        return "none"
    if isinstance(value, str):
        v = value.strip()
        if v == "":
            return "none"
        if v in ("input", "sigma", "fei", "none"):
            return v  # type: ignore[return-value]
    raise ValueError(
        f"router_source={value!r}: expected 'input', 'sigma', 'fei', or 'none'."
    )


def _as_str_list(value: Any) -> Optional[List[str]]:
    """Parse a kwarg that's either a python-literal list, single string, or None."""
    if value is None:
        return None
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError):
        return [value] if isinstance(value, str) else None
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def _as_float_list(value: Any) -> Optional[List[float]]:
    """Parse a kwarg that's either a TOML list, python-literal list string, or None.

    TOML arrays come through as native lists; CLI-stringified lists parse via
    ast.literal_eval. Raises on malformed input rather than silently dropping
    it, since a wrong σ-bucket boundary list would change band assignments
    without surfacing an error.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"Could not parse list-of-floats kwarg: {value!r} ({exc})"
            ) from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"Expected list of floats, got {type(value).__name__}: {value!r}"
        )
    return [float(v) for v in value]


def _validate_sigma_bucket_boundaries(
    boundaries: List[float], num_sigma_buckets: int
) -> None:
    """Validate a custom σ-bucket boundary list. Raises ValueError on any
    violation: wrong length, non-zero start, non-one end, or non-strictly-
    increasing edges.
    """
    if len(boundaries) != num_sigma_buckets + 1:
        raise ValueError(
            "sigma_bucket_boundaries must have length num_sigma_buckets + 1 = "
            f"{num_sigma_buckets + 1}, got {len(boundaries)}."
        )
    if abs(boundaries[0]) > 1e-6:
        raise ValueError(
            f"sigma_bucket_boundaries[0] must be 0.0, got {boundaries[0]}."
        )
    if abs(boundaries[-1] - 1.0) > 1e-6:
        raise ValueError(
            f"sigma_bucket_boundaries[-1] must be 1.0, got {boundaries[-1]}."
        )
    for i in range(len(boundaries) - 1):
        if boundaries[i + 1] <= boundaries[i]:
            raise ValueError(
                "sigma_bucket_boundaries must be strictly increasing; "
                f"violated at index {i}: {boundaries[i]} >= {boundaries[i + 1]}."
            )


def _parse_kv_pairs(kv_pair_str: str, *, is_int: bool) -> Dict[str, Any]:
    """Parse "key1=val1,key2=val2" into a dict, casting values to int/float."""
    pairs: Dict[str, Any] = {}
    for pair in kv_pair_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            logger.warning(f"Invalid format: {pair}, expected 'key=value'")
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            pairs[key] = int(value) if is_int else float(value)
        except ValueError:
            logger.warning(f"Invalid value for {key}: {value}")
    return pairs


# Default exclude regex appended to user-supplied excludes in `from_kwargs`.
# Skips embedders / norms / modulation projectors that are never adapted.
_DEFAULT_EXCLUDE = (
    r".*(_modulation|_norm|_embedder|final_layer|adaln_fused_down|adaln_up_|"
    r"pooled_text_proj).*"
)

@dataclass(frozen=True)
class LoRANetworkCfg:
    """Run-fixed configuration for a ``LoRANetwork``.

    Field groupings mirror the comment blocks in ``factory.create_network``:
    core / targeting / dropouts / regex overrides / T-LoRA / ReFT / Hydra /
    σ-router / channel scaling / logging.
    """

    # core LoRA
    lora_dim: int = 4
    alpha: float = 1.0
    module_class: Type = LoRAModule
    # warm-start path supplies these from the checkpoint; fresh path leaves None
    modules_dim: Optional[Dict[str, int]] = None
    modules_alpha: Optional[Dict[str, float]] = None

    # targeting
    train_llm_adapter: bool = False
    exclude_patterns: List[str] = field(default_factory=list)
    include_patterns: Optional[List[str]] = None
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None

    # dropouts
    dropout: Optional[float] = None
    rank_dropout: Optional[float] = None
    module_dropout: Optional[float] = None

    # per-module rank / lr regex overrides
    reg_dims: Optional[Dict[str, int]] = None
    reg_lrs: Optional[Dict[str, float]] = None

    # T-LoRA
    use_timestep_mask: bool = False
    min_rank: int = 1
    alpha_rank_scale: float = 1.0

    # ReFT
    add_reft: bool = False
    reft_dim: int = 4
    reft_alpha: Optional[float] = None
    reft_layers: object = "all"

    # Hydra (MoE)
    num_experts: int = 4
    # Gaussian perturb std applied to fused per-expert `lora_up_weight` at
    # init in plain HydraLoRA only (NOT OrthoHydra disjoint or fallback) —
    # paper baseline knob; production training should leave at 0.0.
    expert_init_std: float = 0.0
    router_lr_scale: float = 1.0
    # Single regex that scopes which Linear modules participate in routed
    # adaptation. Matched modules become HydraLoRA leaves; non-matching
    # modules fall back to plain LoRA / OrthoLoRA. Sigma- and FEI-feature
    # router inputs piggyback on the same set — there is no separate sub-
    # filter for σ vs FEI vs Hydra anymore. ``None`` = apply MoE everywhere.
    router_targets: Optional[str] = None
    hydra_router_names: Optional[List[str]] = None
    per_bucket_balance_weight: float = 0.3
    num_sigma_buckets: int = 3
    # Hard expert/timestep partition: when on, the E experts are split into
    # ``num_sigma_buckets`` bands of ``E // num_sigma_buckets`` experts each
    # using interleaved layout (expert e → band ``e mod num_sigma_buckets``);
    # for a sample at σ in band b, only the experts in that band can win the
    # gate (out-of-band logits masked to -inf before softmax). Soft routing
    # still operates *within* a band. Independent of (and composes with) the
    # σ-feature router. Requires ``num_experts % num_sigma_buckets == 0``.
    specialize_experts_by_sigma_buckets: bool = False
    # Optional custom σ-bucket boundaries. Length must equal
    # ``num_sigma_buckets + 1``, strictly increasing, starting at 0.0 and
    # ending at 1.0. Defaults (None) to uniform ``linspace(0, 1, B+1)``.
    # Lets you spend more capacity on a chosen σ regime — e.g.
    # ``[0.0, 0.5, 0.8, 1.0]`` gives a wide low-σ band and progressively
    # narrower mid/high-σ bands while expert count per band stays equal.
    sigma_bucket_boundaries: Optional[List[float]] = None

    # Three-axis routing config (see plan2.md). ``use_moe_style`` picks the
    # expert layout — ``False`` (no experts), ``"shared_A"`` (Hydra: one
    # ``lora_down`` + per-expert ``lora_up``), ``"independent_A"`` (FeRA:
    # stacked per-expert ``lora_down`` and ``lora_up``). ``route_per_layer``
    # picks router location: ``True`` (today's Hydra per-Linear default) or
    # ``False`` (one network-level router, FeRA-style). ``router_source``
    # picks the gate input: ``"input"`` (per-Linear input vector — only valid
    # with ``route_per_layer=True``), ``"sigma"`` (sinusoidal σ features),
    # ``"fei"`` (FEI(z_t) simplex), or ``"none"``.
    use_moe_style: MoEStyle = False
    route_per_layer: bool = False
    router_source: RouterSource = "none"

    # PSOFT-style Cayley/SVD parameterization (per-module bool). Selects
    # ``ortho`` mode on ``StackedExpertsLoRAModule`` when paired with
    # ``use_moe_style="independent_A"``; for the non-MoE / shared_A cells the
    # ``ortho``-ness is already encoded in the chosen module class
    # (``OrthoLoRA`` / ``OrthoHydra``) and this field is informational.
    use_ortho: bool = False
    ortho_init_std: float = 0.02

    # σ-conditional router parameters (consumed when ``router_source="sigma"``).
    # Layer scope is shared with Hydra and FEI via ``router_targets`` above.
    sigma_feature_dim: int = 16
    sigma_router_names: Optional[List[str]] = None

    # FEI-conditional router parameters (consumed when ``router_source="fei"``).
    # ``fei_feature_dim`` defaults to 2 = the simplex ``(e_low, e_high)`` from
    # ``library.runtime.fei.compute_fei_2band``. Default
    # ``fei_sigma_low_div=4.0`` for σ_low scaling — chosen by the
    # 2026-05-13 dataset sweep on real training latents (highest
    # std(e_low) at low/mid t). 8.0 remains a Pareto choice. See
    # ``[[project_fera_probe_2band_decision]]``.
    fei_feature_dim: int = 2
    fei_sigma_low_div: float = 4.0
    fei_router_names: Optional[List[str]] = None

    # GlobalRouter parameters (consumed when ``route_per_layer=False``).
    # Two-layer MLP feeding softmax/τ — same shape as FeRA's
    # ``SoftFrequencyRouter``. Final layer is zero-init so step-0 gates are
    # uniform; combined with zero-init expert ups this guarantees ΔW=0 at
    # the first optimizer step.
    router_hidden_dim: int = 64
    router_tau: float = 0.7

    # FECL (Frequency-Energy Consistency Loss) — opt-in auxiliary loss for
    # the FeRA family. Default ``0.0`` keeps it disabled (the 2-band path
    # collapses to a content-free scalar; bench at 3 bands if revisiting).
    # ``library/training/losses.py::_fera_fecl_loss`` reads
    # ``network.fecl_weight`` (set in ``_post_init_hydra``) and applies it
    # to the unscaled scalar computed by ``library/training/fecl.compute_fecl``.
    fera_fecl_weight: float = 0.0
    fera_num_bands: int = 3

    # ChimeraHydra (dual-pool additive routing — see
    # ``docs/proposal/chimera_hydra.md``). ``use_chimera_hydra=True`` swaps
    # the OrthoHydra path for the chimera module, which splits ``num_experts``
    # into a content pool (``num_experts_content``, routed by the per-layer
    # rank-R router) and a freq pool (``num_experts_freq``, routed by the
    # network-level ``FreqRouter`` fed by FEI + sinusoidal-σ features). Total
    # E = K_c + K_f. Per-pool balance loss weights are tracked separately so
    # each pool spreads under independent pressure (a single combined term
    # would let one pool flatten to uniform while the other concentrates).
    use_chimera_hydra: bool = False
    num_experts_content: int = 3
    num_experts_freq: int = 3
    balance_w_content: Optional[float] = None  # falls back to balance_loss_weight
    balance_w_freq: Optional[float] = None  # falls back to balance_loss_weight
    # FreqRouter init magnitude. Non-zero so the freq pool differentiates
    # immediately as FEI/σ vary across the batch — zero-weight init would
    # be a fixed point under the additive composition (see proposal §"Init").
    freq_router_init_std: float = 0.1
    # Per-modality LayerNorm on the FreqRouter input. Active only when both
    # FEI and σ feature blocks are enabled (variance balance is the whole
    # point — with one modality off LN either no-ops or destroys the 2-D
    # FEI simplex's magnitude). Parameterless (``elementwise_affine=False``)
    # so the state_dict format is unchanged; the on/off semantics live in
    # the ``ss_chimera_freq_router_layer_norm`` metadata stamp.
    freq_router_layer_norm: bool = True
    # Per-pool router LR multipliers (chimera-only). Stack on top of the
    # global ``router_lr_scale``: effective LR = ``unet_lr × router_lr_scale
    # × <pool>_router_lr_scale``. Default 1.0 = preserves the previous
    # uniform scaling. Useful when the content pool stays near-uniform
    # (small per-layer router LR with std=0.01 init can take many steps to
    # leave the symmetric initialization) — bumping ``content`` to 5–10×
    # is a faster lever than raising ``balance_w_content``.
    content_router_lr_scale: float = 1.0
    freq_router_lr_scale: float = 1.0

    # SmoothQuant-style per-channel input pre-scaling
    channel_scales_dict: Optional[Dict[str, torch.Tensor]] = None

    # logging
    verbose: bool = False

    @classmethod
    def from_kwargs(
        cls,
        kwargs: Mapping[str, Any],
        *,
        network_dim: Optional[int],
        network_alpha: Optional[float],
        neuron_dropout: Optional[float],
        module_class: Type,
        channel_scales_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> "LoRANetworkCfg":
        """Build cfg from train.py's stringified ``net_kwargs`` dict."""
        if network_dim is None:
            network_dim = 4
        if network_alpha is None:
            network_alpha = 1.0

        train_llm_adapter = _as_bool(kwargs.get("train_llm_adapter"))

        exclude_patterns = _as_str_list(kwargs.get("exclude_patterns")) or []
        exclude_patterns.append(_DEFAULT_EXCLUDE)
        include_patterns = _as_str_list(kwargs.get("include_patterns"))

        layer_start = kwargs.get("layer_start")
        layer_start = int(layer_start) if layer_start is not None else None
        layer_end = kwargs.get("layer_end")
        layer_end = int(layer_end) if layer_end is not None else None

        rank_dropout = kwargs.get("rank_dropout")
        rank_dropout = float(rank_dropout) if rank_dropout is not None else None
        module_dropout = kwargs.get("module_dropout")
        module_dropout = float(module_dropout) if module_dropout is not None else None

        use_timestep_mask = _as_bool(kwargs.get("use_timestep_mask"))
        min_rank = kwargs.get("min_rank")
        min_rank = int(min_rank) if min_rank is not None else 1
        alpha_rank_scale = kwargs.get("alpha_rank_scale")
        alpha_rank_scale = (
            float(alpha_rank_scale) if alpha_rank_scale is not None else 1.0
        )

        add_reft = _as_bool(kwargs.get("add_reft"))
        reft_dim = kwargs.get("reft_dim")
        reft_dim = int(reft_dim) if reft_dim is not None else network_dim
        reft_alpha = kwargs.get("reft_alpha")
        reft_alpha = float(reft_alpha) if reft_alpha is not None else None
        reft_layers = kwargs.get("reft_layers", "all")

        num_experts = kwargs.get("num_experts")
        num_experts = int(num_experts) if num_experts is not None else 4
        expert_init_std = float(kwargs.get("expert_init_std", 0.0))

        router_lr_scale = kwargs.get("network_router_lr_scale")
        router_lr_scale = float(router_lr_scale) if router_lr_scale is not None else 1.0

        _legacy_router_keys = [
            k
            for k in ("hydra_router_layers", "sigma_router_layers", "fei_router_layers")
            if k in kwargs
        ]
        if _legacy_router_keys:
            raise ValueError(
                f"{_legacy_router_keys} are no longer supported — the three "
                "router layer filters were consolidated into a single "
                "`router_targets` regex. Replace them with one `router_targets = "
                "...` entry in your method TOML."
            )
        router_targets = kwargs.get("router_targets", None)
        per_bucket_balance_weight = kwargs.get("per_bucket_balance_weight")
        per_bucket_balance_weight = (
            float(per_bucket_balance_weight)
            if per_bucket_balance_weight is not None
            else 0.3
        )
        num_sigma_buckets = int(kwargs.get("num_sigma_buckets", 3))
        specialize_experts_by_sigma_buckets = _as_bool(
            kwargs.get("specialize_experts_by_sigma_buckets")
        )
        sigma_bucket_boundaries = _as_float_list(
            kwargs.get("sigma_bucket_boundaries")
        )
        if specialize_experts_by_sigma_buckets:
            if num_sigma_buckets <= 1:
                raise ValueError(
                    "specialize_experts_by_sigma_buckets requires num_sigma_buckets > 1, "
                    f"got num_sigma_buckets={num_sigma_buckets}."
                )
            if num_experts % num_sigma_buckets != 0:
                raise ValueError(
                    "specialize_experts_by_sigma_buckets requires num_experts to be "
                    f"divisible by num_sigma_buckets, got num_experts={num_experts}, "
                    f"num_sigma_buckets={num_sigma_buckets}."
                )
            if sigma_bucket_boundaries is not None:
                _validate_sigma_bucket_boundaries(
                    sigma_bucket_boundaries, num_sigma_buckets
                )
        elif sigma_bucket_boundaries is not None:
            logger.warning(
                "sigma_bucket_boundaries set but "
                "specialize_experts_by_sigma_buckets is off — boundaries ignored."
            )
            sigma_bucket_boundaries = None

        sigma_feature_dim = int(kwargs.get("sigma_feature_dim", 16))

        fei_feature_dim = int(kwargs.get("fei_feature_dim", 2))
        fei_sigma_low_div = float(kwargs.get("fei_sigma_low_div", 4.0))

        # GlobalRouter knobs (only consumed when ``route_per_layer=False``).
        router_hidden_dim = int(kwargs.get("router_hidden_dim", kwargs.get("router_hidden", 64)))
        router_tau = float(kwargs.get("router_tau", 0.7))

        use_ortho = _as_bool(kwargs.get("use_ortho"))
        ortho_init_std = float(kwargs.get("ortho_init_std", 0.02))

        # FECL knobs. Default off; turning it on requires `num_bands >= 3`
        # to be a meaningful objective (see compute_fecl docstring).
        fera_fecl_weight = float(kwargs.get("fera_fecl_weight", 0.0))
        fera_num_bands = int(kwargs.get("fera_num_bands", kwargs.get("num_bands", 3)))

        # ChimeraHydra knobs. ``num_experts`` (parent Hydra cfg) is treated
        # as a derived value when ``use_chimera_hydra=True`` — recomputed
        # below so users only set K_c / K_f.
        use_chimera_hydra = _as_bool(kwargs.get("use_chimera_hydra"))
        num_experts_content = int(kwargs.get("num_experts_content", 3))
        num_experts_freq = int(kwargs.get("num_experts_freq", 3))
        balance_w_content_raw = kwargs.get("balance_w_content")
        balance_w_content = (
            float(balance_w_content_raw) if balance_w_content_raw is not None else None
        )
        balance_w_freq_raw = kwargs.get("balance_w_freq")
        balance_w_freq = (
            float(balance_w_freq_raw) if balance_w_freq_raw is not None else None
        )
        freq_router_init_std = float(kwargs.get("freq_router_init_std", 0.1))
        freq_router_layer_norm = _as_bool(kwargs.get("freq_router_layer_norm", True))
        content_router_lr_scale = float(
            kwargs.get("network_content_router_lr_scale", 1.0)
        )
        freq_router_lr_scale = float(
            kwargs.get("network_freq_router_lr_scale", 1.0)
        )
        if use_chimera_hydra:
            if num_experts_content <= 0 or num_experts_freq <= 0:
                raise ValueError(
                    "use_chimera_hydra=True requires num_experts_content > 0 "
                    f"and num_experts_freq > 0 (got K_c={num_experts_content}, "
                    f"K_f={num_experts_freq})."
                )
            # Derive total E from the pool split so the rest of the
            # cfg machinery (warmup masks, balance loss accumulators, etc.)
            # sees a consistent num_experts.
            num_experts = num_experts_content + num_experts_freq

        # Three-axis routing resolution (plan2.md §three-axis-config). The
        # legacy ``use_hydra`` / ``use_sigma_router`` / ``use_fei_router``
        # kwargs were retired in plan2 task #6 — every shipped TOML uses the
        # new keys, and old `.safetensors` files (with ``ss_use_hydra`` etc.)
        # stop loading by design (no legacy compat shim).
        raw_moe_style = kwargs.get("use_moe_style")
        raw_route_per_layer = kwargs.get("route_per_layer")
        raw_router_source = kwargs.get("router_source")

        for legacy_key in ("use_hydra", "use_sigma_router", "use_fei_router"):
            if kwargs.get(legacy_key) is not None:
                raise ValueError(
                    f"Legacy router kwarg {legacy_key!r} is no longer "
                    "supported. Use the three-axis keys instead: "
                    "`use_moe_style` (False / 'shared_A' / 'independent_A'), "
                    "`route_per_layer` (true / false), and `router_source` "
                    "('none' / 'input' / 'sigma' / 'fei'). See plan2.md "
                    "§three-axis-config."
                )

        use_moe_style: MoEStyle = (
            _as_moe_style(raw_moe_style) if raw_moe_style is not None else False
        )

        if raw_router_source is not None:
            router_source: RouterSource = _as_router_source(raw_router_source)
        elif use_moe_style is not False:
            # Hydra's default router input is the per-Linear input vector.
            router_source = "input"
        else:
            router_source = "none"

        if raw_route_per_layer is not None:
            route_per_layer = _as_bool(raw_route_per_layer)
        else:
            # No-MoE means no router at all; Hydra defaults to per-layer.
            route_per_layer = use_moe_style is not False

        # ChimeraHydra: pin the three-axis cells to (shared_A, per-layer,
        # input) regardless of TOML wiring. The chimera content router IS
        # a per-layer shared_A Hydra router on pooled lx; the freq router
        # adds a second routing source on top via a dedicated network-level
        # mechanism. Stamping these three values means the save metadata
        # flows through the standard MoE branch and the loader can detect
        # the chimera-specific stamps without a parallel three-axis path.
        if use_chimera_hydra:
            if use_moe_style not in (False, "shared_A"):
                raise ValueError(
                    "use_chimera_hydra=True is only compatible with "
                    "use_moe_style='shared_A' (or unset); got "
                    f"use_moe_style={use_moe_style!r}."
                )
            if raw_route_per_layer is not None and not _as_bool(raw_route_per_layer):
                raise ValueError(
                    "use_chimera_hydra=True requires route_per_layer=True "
                    "(content router is per-layer)."
                )
            if raw_router_source is not None and raw_router_source != "input":
                raise ValueError(
                    "use_chimera_hydra=True requires router_source='input' "
                    "(content router reads pooled lx); σ/FEI are owned by "
                    "the network-level FreqRouter."
                )
            use_moe_style = "shared_A"
            route_per_layer = True
            router_source = "input"

        # Validate impossible combos.
        if use_moe_style is False and (
            route_per_layer or router_source != "none"
        ):
            raise ValueError(
                "Routing config requires use_moe_style != False; got "
                f"use_moe_style={use_moe_style!r}, route_per_layer={route_per_layer}, "
                f"router_source={router_source!r}."
            )
        if not route_per_layer and router_source == "input":
            raise ValueError(
                "router_source='input' requires route_per_layer=True — no "
                "network-level 'input' signal exists per DiT forward."
            )

        reg_dims_str = kwargs.get("network_reg_dims")
        reg_dims = _parse_kv_pairs(reg_dims_str, is_int=True) if reg_dims_str else None
        reg_lrs_str = kwargs.get("network_reg_lrs")
        reg_lrs = _parse_kv_pairs(reg_lrs_str, is_int=False) if reg_lrs_str else None

        verbose = _as_bool(kwargs.get("verbose"))

        return cls(
            lora_dim=network_dim,
            alpha=network_alpha,
            module_class=module_class,
            train_llm_adapter=train_llm_adapter,
            exclude_patterns=exclude_patterns,
            include_patterns=include_patterns,
            layer_start=layer_start,
            layer_end=layer_end,
            dropout=neuron_dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            reg_dims=reg_dims,
            reg_lrs=reg_lrs,
            use_timestep_mask=use_timestep_mask,
            min_rank=min_rank,
            alpha_rank_scale=alpha_rank_scale,
            add_reft=add_reft,
            reft_dim=reft_dim,
            reft_alpha=reft_alpha,
            reft_layers=reft_layers,
            num_experts=num_experts,
            expert_init_std=expert_init_std,
            router_lr_scale=router_lr_scale,
            router_targets=router_targets,
            per_bucket_balance_weight=per_bucket_balance_weight,
            num_sigma_buckets=num_sigma_buckets,
            specialize_experts_by_sigma_buckets=specialize_experts_by_sigma_buckets,
            sigma_bucket_boundaries=sigma_bucket_boundaries,
            use_moe_style=use_moe_style,
            route_per_layer=route_per_layer,
            router_source=router_source,
            sigma_feature_dim=sigma_feature_dim,
            fei_feature_dim=fei_feature_dim,
            fei_sigma_low_div=fei_sigma_low_div,
            router_hidden_dim=router_hidden_dim,
            router_tau=router_tau,
            use_ortho=use_ortho,
            ortho_init_std=ortho_init_std,
            fera_fecl_weight=fera_fecl_weight,
            fera_num_bands=fera_num_bands,
            use_chimera_hydra=use_chimera_hydra,
            num_experts_content=num_experts_content,
            num_experts_freq=num_experts_freq,
            balance_w_content=balance_w_content,
            balance_w_freq=balance_w_freq,
            freq_router_init_std=freq_router_init_std,
            freq_router_layer_norm=freq_router_layer_norm,
            content_router_lr_scale=content_router_lr_scale,
            freq_router_lr_scale=freq_router_lr_scale,
            channel_scales_dict=channel_scales_dict,
            verbose=verbose,
        )

    @classmethod
    def from_weights(
        cls,
        *,
        modules_dim: Dict[str, int],
        modules_alpha: Dict[str, float],
        module_class: Type,
        train_llm_adapter: bool,
        has_reft: bool,
        reft_dim: Optional[int],
        reft_block_indices,
        is_hydra_or_ortho_hydra: bool,
        hydra_num_experts: int,
        sigma_feature_dim_detected: Optional[int],
        sigma_router_names: Optional[List[str]],
        hydra_router_names: Optional[List[str]],
        channel_scales_dict: Optional[Dict[str, torch.Tensor]],
        specialize_experts_by_sigma_buckets: bool = False,
        num_sigma_buckets: Optional[int] = None,
        sigma_bucket_boundaries: Optional[List[float]] = None,
        fei_feature_dim: int = 0,
        fei_sigma_low_div: Optional[float] = None,
        fei_router_names: Optional[List[str]] = None,
        is_stacked_experts: bool = False,
        # Three-axis stamps from save metadata. All three must be present
        # for MoE checkpoints — pre-plan2 artifacts stop loading by design.
        new_use_moe_style: Optional[str] = None,
        new_route_per_layer: Optional[bool] = None,
        new_router_source: Optional[str] = None,
        # ChimeraHydra stamps. Present only on chimera checkpoints — when
        # set the loader builds ``ChimeraHydraLoRAModule`` instead of
        # ``OrthoHydraLoRAModule`` and the network attaches a FreqRouter.
        is_chimera_hydra: bool = False,
        num_experts_content: Optional[int] = None,
        num_experts_freq: Optional[int] = None,
        freq_router_layer_norm: bool = False,
    ) -> "LoRANetworkCfg":
        """Build cfg from a checkpoint key-sniff (warm-start / inference path).

        Mirrors the ``LoRANetwork(...)`` call previously embedded in
        ``create_network_from_weights``. Per-module dims / alphas come from
        ``modules_dim`` / ``modules_alpha``, so ``lora_dim`` / ``alpha`` here
        are placeholders. Training-time schedules (warmup, T-LoRA) stay off
        in the warm-start path.

        ``specialize_experts_by_sigma_buckets`` / ``num_sigma_buckets`` /
        ``sigma_bucket_boundaries`` come from safetensors metadata stamped by
        ``save_weights`` — the partition leaves no tensor footprint
        (``_expert_band`` / ``_sigma_edges`` are non-persistent) so it has to
        be reconstructed from those scalars at load time.

        For non-MoE checkpoints (plain LoRA / OrthoLoRA / T-LoRA / ReFT) the
        three-axis stamps are not stamped at save time; absence is taken as
        ``(False, False, "none")``. MoE checkpoints (Hydra / OrthoHydra /
        StackedExperts) must carry all three stamps — plan2 task #6 retired
        the legacy ``ss_use_hydra`` / ``ss_use_fei_router`` fallback.
        """
        if (
            new_use_moe_style is not None
            and new_route_per_layer is not None
            and new_router_source is not None
        ):
            use_moe_style: MoEStyle = _as_moe_style(new_use_moe_style)
            route_per_layer = bool(new_route_per_layer)
            router_source: RouterSource = _as_router_source(new_router_source)
        elif is_hydra_or_ortho_hydra or is_stacked_experts:
            raise RuntimeError(
                "MoE checkpoint is missing the three-axis routing stamps "
                "(ss_use_moe_style / ss_route_per_layer / ss_router_source). "
                "Pre-plan2 checkpoints stop loading by design — retrain the "
                "adapter to produce the new metadata."
            )
        else:
            use_moe_style = False
            route_per_layer = False
            router_source = "none"

        # ChimeraHydra requires both pool sizes to be stamped at save time;
        # absence on a flagged checkpoint indicates malformed metadata.
        if is_chimera_hydra:
            if num_experts_content is None or num_experts_freq is None:
                raise RuntimeError(
                    "ChimeraHydra checkpoint missing ss_num_experts_content / "
                    "ss_num_experts_freq metadata — checkpoint is malformed."
                )
            if (
                hydra_num_experts
                and hydra_num_experts != num_experts_content + num_experts_freq
            ):
                raise RuntimeError(
                    "ChimeraHydra checkpoint K_c + K_f mismatch: stamped "
                    f"K_c={num_experts_content}, K_f={num_experts_freq}, "
                    f"detected num_experts={hydra_num_experts}."
                )

        return cls(
            lora_dim=4,
            alpha=1.0,
            module_class=module_class,
            modules_dim=modules_dim,
            modules_alpha=modules_alpha,
            train_llm_adapter=train_llm_adapter,
            add_reft=has_reft,
            reft_dim=reft_dim if reft_dim is not None else 4,
            reft_layers=sorted(reft_block_indices) if has_reft else "all",
            num_experts=hydra_num_experts if is_hydra_or_ortho_hydra else 4,
            channel_scales_dict=channel_scales_dict,
            use_moe_style=use_moe_style,
            route_per_layer=route_per_layer,
            router_source=router_source,
            sigma_feature_dim=(
                sigma_feature_dim_detected
                if sigma_feature_dim_detected is not None
                else 128
            ),
            sigma_router_names=sigma_router_names,
            hydra_router_names=hydra_router_names,
            specialize_experts_by_sigma_buckets=specialize_experts_by_sigma_buckets,
            num_sigma_buckets=(
                int(num_sigma_buckets) if num_sigma_buckets else 3
            ),
            sigma_bucket_boundaries=sigma_bucket_boundaries,
            fei_feature_dim=int(fei_feature_dim),
            fei_sigma_low_div=(
                float(fei_sigma_low_div) if fei_sigma_low_div is not None else 4.0
            ),
            fei_router_names=fei_router_names,
            use_chimera_hydra=is_chimera_hydra,
            num_experts_content=(
                int(num_experts_content) if num_experts_content is not None else 3
            ),
            num_experts_freq=(
                int(num_experts_freq) if num_experts_freq is not None else 3
            ),
            freq_router_layer_norm=bool(freq_router_layer_norm),
        )
