# LoRANetwork: the module-assembly / training-orchestration core of the LoRA
# adapter stack for Anima. Targets DiT blocks (and optionally text-encoder
# attention) with pluggable per-module classes supplied by a NetworkSpec.

import logging
import math
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import torch

from library.log import setup_logging
from library.training.metrics import MetricContext
from networks import NETWORK_REGISTRY, NetworkSpec, lora_save
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.loading import (
    _parse_reft_layers,
    _refuse_split_hydra_keys,
    _refuse_split_stacked_experts_keys,
    _refuse_unfused_attn_lora_keys,
    _stack_lora_ups,
)
from networks.lora_modules import (
    ChimeraHydraInferenceModule,
    ChimeraHydraLoRAModule,
    HydraLoRAModule,
    LoRAModule,
    OrthoHydraLoRAModule,
    OrthoLoRAModule,
    ReFTModule,
    StackedExpertsLoRAModule,
    _sigma_sinusoidal_features,
)

setup_logging()
logger = logging.getLogger(__name__)

_BLOCK_IDX_RE = re.compile(r"blocks\.(\d+)\.")


class GlobalRouter(torch.nn.Module):
    """Single network-level router feeding every routing-aware module.

    Two-layer MLP → softmax/τ — same parameterization as FeRA's
    ``SoftFrequencyRouter``. Final layer is zero-init so step-0 gates
    are uniform across experts. Combined with zero-init expert ups (free
    mode) or zero-init ``lambda_layer`` (ortho mode) this guarantees
    ΔW=0 at the first optimizer step (clean residual baseline).

    Owned by ``LoRANetwork`` when ``cfg.route_per_layer=False`` and
    ``cfg.use_moe_style`` selects an MoE layout. Reads the per-step
    routing signal (FEI simplex / sinusoidal σ features) supplied by
    the train loop via ``set_fei`` / ``set_sigma``, and broadcasts the
    resulting gates ``(B, E)`` to every routing-aware module's
    ``_routing_weights`` buffer via ``LoRANetwork.set_routing_weights``.

    Exposes ``_last_gates`` / ``_last_input`` for the metrics layer
    (``LoRANetwork.metrics`` and the future FECL handler in task #5);
    both are detached and overwritten per forward.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        *,
        hidden_dim: int = 64,
        tau: float = 0.7,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(
                f"GlobalRouter: input_dim must be > 0, got {input_dim}"
            )
        if num_experts <= 1:
            raise ValueError(
                f"GlobalRouter: num_experts must be > 1, got {num_experts}"
            )
        self.input_dim = int(input_dim)
        self.num_experts = int(num_experts)
        self.tau = float(tau)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_experts),
        )
        # Uniform-at-init: zero the output layer so softmax(0/τ) = 1/E.
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

        # Per-step diagnostics. Overwritten on every forward; readable by
        # ``LoRANetwork.metrics`` and the FECL loss handler. Detached at
        # write so holding the reference across the step boundary doesn't
        # pin autograd state. ``_last_fei`` is an alias of ``_last_input``
        # under the FEI router source — wired in ``forward``.
        self._last_gates: Optional[torch.Tensor] = None
        self._last_input: Optional[torch.Tensor] = None
        self._last_fei: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim). Promote to fp32 for the matmul + softmax —
        # bf16 logits + softmax(τ<1) underflow at low energies. Inference
        # casts the parent LoRANetwork to bf16, which would otherwise drag
        # the router weights along; re-pin to fp32 on first forward so the
        # matmul dtype matches the upcast input.
        if self.net[0].weight.dtype != torch.float32:
            self.net.float()
        x32 = x.float()
        logits = self.net(x32)
        gates = torch.softmax(logits / self.tau, dim=-1)
        self._last_gates = gates.detach()
        # ``_last_input`` is the raw routing-signal tensor that fed this
        # forward — FEI simplex (router_source="fei") or sinusoidal-σ
        # features ("sigma"). Aliased as ``_last_fei`` for the FECL handler
        # / plan2 task #5 — keeps the diagnostic surface stable across
        # router-source variants.
        self._last_input = x32.detach()
        self._last_fei = self._last_input
        return gates


class FreqRouter(torch.nn.Module):
    """ChimeraHydra freq-pool router (one per network).

    Two-layer MLP feeding softmax/τ over the ``K_f`` freq experts. Input is
    ``concat(FEI(z_t), sinusoidal-σ-features)`` — both functions of the
    per-step σ/z_t. The router lives at network top level and broadcasts
    ``π_f`` to every chimera module's ``_freq_routing_weights`` buffer; the
    broadcast preserves grad_fn so ``∂L_denoise/∂π_f`` reaches the router's
    parameters along the same path FeRA's GlobalRouter uses (eq. 6-7, 11).

    Critical: the output layer uses NON-zero init (small N(0, std)). Unlike
    GlobalRouter (which zero-inits to guarantee ΔW=0 at step 0), a
    zero-init freq router would be a fixed point of the additive
    composition — the freq pool would receive uniform gates that fail to
    differentiate the experts and the gradient `∂L/∂W_router` would never
    leave zero. The chimera proposal mandates non-zero output init for
    exactly this reason (see proposal §"Init").

    Per-modality LayerNorm (``apply_layer_norm=True``): when both
    ``fei_dim`` and ``sigma_dim`` are > 0, each modality's slice of the
    concat input is passed through a parameterless ``LayerNorm`` before
    the MLP. The 2-D FEI simplex and the 16/32-D sinusoidal-σ block have
    different per-channel variance budgets at init (variance contribution
    scales as ``n_channels``), so without LN the higher-dim σ block can
    fan-in-overpower FEI ~``sigma_dim/fei_dim``× at init. LN is
    intentionally parameterless (``elementwise_affine=False``) — keeps the
    save/load surface unchanged, no metadata stamp needed for the LN
    weights themselves (only for the on/off flag).
    """

    def __init__(
        self,
        input_dim: int,
        num_freq_experts: int,
        *,
        hidden_dim: int = 32,
        tau: float = 1.0,
        init_std: float = 0.1,
        fei_dim: int = 0,
        sigma_dim: int = 0,
        apply_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"FreqRouter: input_dim must be > 0, got {input_dim}")
        if num_freq_experts <= 1:
            raise ValueError(
                f"FreqRouter: num_freq_experts must be > 1, got {num_freq_experts}"
            )
        self.input_dim = int(input_dim)
        self.num_freq_experts = int(num_freq_experts)
        self.tau = float(tau)
        self.fei_dim = int(fei_dim)
        self.sigma_dim = int(sigma_dim)
        # LN only fires when both modalities are present — its job is
        # variance balance across the concat, which is a no-op (or worse,
        # destructive on the 2-D simplex) when only one modality is in
        # play. The dim-sum check guards against the rebuild path where
        # fei_dim+sigma_dim wasn't threaded through; in that case LN stays
        # off and the router behaves like the pre-LN build.
        self.apply_layer_norm = bool(apply_layer_norm) and (
            self.fei_dim > 0
            and self.sigma_dim > 0
            and self.fei_dim + self.sigma_dim == self.input_dim
        )
        # SiLU (proposal §Routers): smoother than ReLU on small-input MLPs
        # and consistent with the DiT's own activation choice.
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, num_freq_experts),
        )
        with torch.no_grad():
            # Hidden layer keeps default Linear init. Only the output layer
            # gets the small-std non-zero init that breaks the freq-pool
            # cold-start fixed point — see class docstring.
            torch.nn.init.normal_(self.net[-1].weight, std=float(init_std))
            torch.nn.init.zeros_(self.net[-1].bias)

        # Parameterless per-modality LN. elementwise_affine=False keeps the
        # state_dict free of ln_* keys, so old (LN-off) checkpoints stay
        # load-compatible — the on/off semantics are carried by the
        # ``apply_layer_norm`` flag (stamped to metadata), not by tensor
        # presence in the state_dict.
        self.ln_fei: Optional[torch.nn.LayerNorm] = None
        self.ln_sigma: Optional[torch.nn.LayerNorm] = None
        if self.apply_layer_norm:
            self.ln_fei = torch.nn.LayerNorm(self.fei_dim, elementwise_affine=False)
            self.ln_sigma = torch.nn.LayerNorm(self.sigma_dim, elementwise_affine=False)

        # Per-step diagnostics, parallel to GlobalRouter.
        self._last_gates: Optional[torch.Tensor] = None
        self._last_input: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # See GlobalRouter.forward — fp32 compute is load-bearing for the
        # softmax(logits / τ) precision at small τ. Inference casts the
        # parent LoRANetwork to bf16; re-pin router weights to fp32.
        if self.net[0].weight.dtype != torch.float32:
            self.net.float()
        x32 = x.float()
        if self.apply_layer_norm:
            fei_part = self.ln_fei(x32[..., : self.fei_dim])
            sigma_part = self.ln_sigma(x32[..., self.fei_dim : self.fei_dim + self.sigma_dim])
            x32 = torch.cat([fei_part, sigma_part], dim=-1)
        logits = self.net(x32)
        gates = torch.softmax(logits / self.tau, dim=-1)
        self._last_gates = gates.detach()
        self._last_input = x32.detach()
        return gates


class LoRANetwork(torch.nn.Module):
    # Target modules: DiT blocks, embedders, final layer. embedders and final layer are excluded by default.
    ANIMA_TARGET_REPLACE_MODULE = [
        "Block",
        "PatchEmbed",
        "TimestepEmbedding",
        "FinalLayer",
    ]
    # Target modules: LLM Adapter blocks
    ANIMA_ADAPTER_TARGET_REPLACE_MODULE = ["LLMAdapterTransformerBlock"]
    # Target modules for text encoder (Qwen3)
    TEXT_ENCODER_TARGET_REPLACE_MODULE = [
        "Qwen3Attention",
        "Qwen3MLP",
        "Qwen3SdpaAttention",
        "Qwen3FlashAttention2",
    ]

    LORA_PREFIX_ANIMA = "lora_unet"  # ComfyUI compatible
    LORA_PREFIX_TEXT_ENCODER = "lora_te"  # Qwen3

    def __init__(
        self,
        text_encoders: list,
        unet,
        cfg: LoRANetworkCfg,
        *,
        multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # Mutable runtime state — explicitly NOT in cfg. ``set_multiplier`` and
        # ``set_loraplus_lr_ratio`` write these post-construction; per-step
        # diagnostics (hit counters, σ caches) accumulate during training.
        self.multiplier = multiplier
        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None
        self._channel_scale_misses: List[str] = []
        self._channel_scale_hits: int = 0
        self._sigma_router_hits: int = 0
        self._hydra_router_hits: int = 0
        self._hydra_router_misses: int = 0
        self._last_sigma: Optional[torch.Tensor] = None
        # Hydra up-weight grad-norm snapshot (T-LoRA / σ-bucket conflict
        # diagnostic). Filled by ``capture_up_grad_stats`` between backward
        # and ``optimizer.zero_grad``; consumed by the ``hydra_up_grad``
        # metric. Values stay on-device until ``get_up_grad_stats`` runs the
        # D2H — capture happens every sync step but the metric only reads on
        # log steps, so the sync was the per-step bottleneck.
        self._last_up_grad_stats: Dict[str, object] = {}
        # Per-step cache for ``get_router_stats`` — both the progress-bar
        # postfix and the metrics layer call it on log steps. Cleared in
        # ``clear_step_caches`` so the next forward recomputes.
        self._router_stats_cache: Optional[Dict[str, object]] = None
        # Separate cache for the chimera dual-pool router stats — different
        # reduction (mean gates per pool, not argmax-histogram) and different
        # entropy normalization (per-pool log(K_pool)). Same lifecycle.
        self._chimera_router_stats_cache: Optional[Dict[str, object]] = None

        # Local aliases for the closure body and the post-closure ReFT block.
        # Reading via `cfg.foo` works too; aliases just keep the diff small.
        module_class = cfg.module_class
        modules_dim = cfg.modules_dim
        modules_alpha = cfg.modules_alpha
        dropout = cfg.dropout
        rank_dropout = cfg.rank_dropout
        module_dropout = cfg.module_dropout
        verbose = cfg.verbose
        alpha = cfg.alpha
        lora_dim = cfg.lora_dim
        train_llm_adapter = cfg.train_llm_adapter
        add_reft = cfg.add_reft
        reft_dim = cfg.reft_dim
        reft_alpha = cfg.reft_alpha
        reft_layers = cfg.reft_layers

        # Unified routing scope. ``cfg.router_targets`` is the single regex
        # that governs which Linears participate in routed adaptation (Hydra
        # MoE leaves + σ-feature concat + FEI-feature concat all share it).
        # From-weights path supplies an explicit name set per router family
        # (different families may have different module memberships in older
        # checkpoints); when present, the explicit set wins over the regex.
        _router_re = (
            re.compile(cfg.router_targets) if cfg.router_targets else None
        )

        self._sigma_router_names = (
            set(cfg.sigma_router_names) if cfg.sigma_router_names else None
        )
        self._sigma_router_re = (
            _router_re
            if (
                cfg.router_source == "sigma"
                and _router_re is not None
                and self._sigma_router_names is None
            )
            else None
        )

        self._fei_router_names = (
            set(cfg.fei_router_names) if cfg.fei_router_names else None
        )
        self._fei_router_re = (
            _router_re
            if (
                cfg.router_source == "fei"
                and _router_re is not None
                and self._fei_router_names is None
            )
            else None
        )
        self._fei_router_hits = 0
        # Modules built with ``use_global_router=True`` (shared_A +
        # ``route_per_layer=False``): the per-layer router is skipped and gates
        # arrive via the network-level ``GlobalRouter``. Counted separately
        # from ``_fei_router_hits`` because the per-layer FEI cat is bypassed.
        self._global_router_hits = 0
        # Retained as a network attr (library/inference/adapters.py reads it
        # via getattr); derived from cfg.router_source.
        self.use_fei_router = cfg.router_source == "fei"
        self.use_sigma_router = cfg.router_source == "sigma"
        # Shared-A Hydra layout + network-level router (FEI-on-Hydra global).
        # Toggle for the per-module construction loop below; lets Hydra /
        # OrthoHydra modules skip ``self.router`` and consume gates from the
        # ``GlobalRouter`` instead. Mirrors the FeRA (independent_A) routing
        # location without changing the underlying Hydra parameter layout.
        self._use_global_router_for_hydra = (
            cfg.use_moe_style == "shared_A"
            and not cfg.route_per_layer
            and cfg.router_source != "none"
        )

        # Per-module HydraLoRA gating. Matching modules get the Hydra class;
        # non-matching modules fall back to plain LoRA / OrthoLoRA so MoE
        # capacity is concentrated where specialization is actually learnable.
        # Fresh path: regex over `original_name`. From-weights path: explicit
        # name set detected from checkpoint keys. Explicit set wins. None on
        # both = apply MoE everywhere (legacy).
        self._hydra_router_names = (
            set(cfg.hydra_router_names) if cfg.hydra_router_names else None
        )
        self._hydra_router_re = (
            _router_re if (_router_re is not None and self._hydra_router_names is None)
            else None
        )

        if modules_dim is not None:
            logger.info("create LoRA network from weights")
        else:
            logger.info(
                f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}"
            )
            logger.info(
                f"neuron dropout: p={dropout}, rank dropout: p={rank_dropout}, module dropout: p={module_dropout}"
            )

        # compile regular expression if specified
        def str_to_re_patterns(patterns: Optional[List[str]]) -> List[re.Pattern]:
            re_patterns = []
            if patterns is not None:
                for pattern in patterns:
                    try:
                        re_pattern = re.compile(pattern)
                    except re.error as e:
                        logger.error(f"Invalid pattern '{pattern}': {e}")
                        continue
                    re_patterns.append(re_pattern)
            return re_patterns

        exclude_re_patterns = str_to_re_patterns(cfg.exclude_patterns)
        include_re_patterns = str_to_re_patterns(cfg.include_patterns)

        # create module instances
        def create_modules(
            is_unet: bool,
            text_encoder_idx: Optional[int],
            root_module: torch.nn.Module,
            target_replace_modules: List[str],
            default_dim: Optional[int] = None,
        ) -> Tuple[List[LoRAModule], List[str]]:
            prefix = (
                self.LORA_PREFIX_ANIMA if is_unet else self.LORA_PREFIX_TEXT_ENCODER
            )

            # First pass: collect candidate modules
            candidates = []
            for name, module in root_module.named_modules():
                if (
                    target_replace_modules is None
                    or module.__class__.__name__ in target_replace_modules
                ):
                    if target_replace_modules is None:
                        module = root_module

                    for child_name, child_module in module.named_modules():
                        is_linear = isinstance(child_module, torch.nn.Linear)
                        is_conv2d = isinstance(child_module, torch.nn.Conv2d)
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            original_name = (name + "." if name else "") + child_name
                            # Strip torch.compile wrapper from module path
                            original_name = original_name.replace("_orig_mod.", "")
                            lora_name = f"{prefix}.{original_name}".replace(".", "_")

                            # exclude/include filter
                            excluded = any(
                                pattern.fullmatch(original_name)
                                for pattern in exclude_re_patterns
                            )
                            included = any(
                                pattern.fullmatch(original_name)
                                for pattern in include_re_patterns
                            )
                            if excluded and not included:
                                if verbose:
                                    logger.info(f"exclude: {original_name}")
                                continue

                            # layer range filter: skip blocks outside [layer_start, layer_end)
                            if is_unet and (
                                cfg.layer_start is not None or cfg.layer_end is not None
                            ):
                                block_match = _BLOCK_IDX_RE.match(original_name)
                                if block_match:
                                    block_idx = int(block_match.group(1))
                                    if (
                                        cfg.layer_start is not None
                                        and block_idx < cfg.layer_start
                                    ):
                                        if verbose:
                                            logger.info(
                                                f"layer_range exclude: {original_name} (block {block_idx} < {cfg.layer_start})"
                                            )
                                        continue
                                    if (
                                        cfg.layer_end is not None
                                        and block_idx >= cfg.layer_end
                                    ):
                                        if verbose:
                                            logger.info(
                                                f"layer_range exclude: {original_name} (block {block_idx} >= {cfg.layer_end})"
                                            )
                                        continue

                            dim = None
                            alpha_val = None

                            if modules_dim is not None:
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha_val = modules_alpha[lora_name]
                            else:
                                if cfg.reg_dims is not None:
                                    for reg, d in cfg.reg_dims.items():
                                        if re.fullmatch(reg, original_name):
                                            dim = d
                                            alpha_val = alpha
                                            logger.info(
                                                f"Module {original_name} matched with regex '{reg}' -> dim: {dim}"
                                            )
                                            break
                                if dim is None:
                                    if is_linear or is_conv2d_1x1:
                                        dim = (
                                            default_dim
                                            if default_dim is not None
                                            else lora_dim
                                        )
                                        alpha_val = alpha

                            if dim is None or dim == 0:
                                if is_linear or is_conv2d_1x1:
                                    candidates.append(
                                        (
                                            lora_name,
                                            None,
                                            None,
                                            None,
                                            original_name,
                                            True,
                                        )
                                    )  # skipped
                                continue

                            candidates.append(
                                (
                                    lora_name,
                                    child_module,
                                    dim,
                                    alpha_val,
                                    original_name,
                                    False,
                                )
                            )

                    if target_replace_modules is None:
                        break

            # Second pass: create LoRA modules with progress bar
            from tqdm import tqdm

            loras = []
            skipped = []
            non_skipped = [
                (ln, cm, d, a, on) for ln, cm, d, a, on, skip in candidates if not skip
            ]
            skipped = [ln for ln, cm, d, a, on, skip in candidates if skip]

            label = (
                "DiT"
                if is_unet
                else f"TE{text_encoder_idx + 1}"
                if text_encoder_idx is not None
                else "model"
            )
            for lora_name, child_module, dim, alpha_val, original_name in tqdm(
                non_skipped, desc=f"Creating {label} LoRA", leave=False
            ):
                # Per-module class resolution: when the network's nominal class
                # is Hydra (MoE), narrow it to only the layers in the hydra
                # filter. Non-matching layers fall back to plain LoRA /
                # OrthoLoRA so router overhead + balance-loss pressure are
                # concentrated on sites where specialization is learnable.
                effective_module_class = module_class
                if (
                    module_class
                    in (
                        HydraLoRAModule,
                        OrthoHydraLoRAModule,
                        ChimeraHydraLoRAModule,
                        ChimeraHydraInferenceModule,
                    )
                    and is_unet
                ):
                    if self._hydra_router_names is not None:
                        hydra_on = lora_name in self._hydra_router_names
                    elif self._hydra_router_re is not None:
                        hydra_on = bool(self._hydra_router_re.search(original_name))
                    else:
                        hydra_on = True
                    if hydra_on:
                        self._hydra_router_hits += 1
                    else:
                        self._hydra_router_misses += 1
                        if module_class is HydraLoRAModule:
                            effective_module_class = LoRAModule
                        elif module_class is ChimeraHydraInferenceModule:
                            # Load path. Unrouted leg was saved as plain LoRA
                            # (OrthoLoRA distilled to ``.lora_down.weight`` +
                            # ``.lora_up.weight`` at save time — see
                            # ``_convert_ortho_to_lora``).
                            effective_module_class = LoRAModule
                        else:
                            # Train path (ChimeraHydraLoRAModule) and
                            # OrthoHydra: unrouted leg uses the OrthoLoRA
                            # Cayley parameterization.
                            effective_module_class = OrthoLoRAModule

                extra_kwargs = {}
                if effective_module_class == OrthoLoRAModule:
                    pass  # no extra kwargs — SVD init reads from org_module directly
                elif effective_module_class == ChimeraHydraLoRAModule:
                    # Pool split is the chimera's only constructor surface;
                    # σ/FEI feature dims are 0 by design (the network-level
                    # FreqRouter owns those axes — see chimera.py module
                    # docstring). The pool sum must equal cfg.num_experts
                    # by ``LoRANetworkCfg.from_kwargs`` invariant.
                    extra_kwargs["num_experts_content"] = cfg.num_experts_content
                    extra_kwargs["num_experts_freq"] = cfg.num_experts_freq
                elif effective_module_class == ChimeraHydraInferenceModule:
                    # Inference (free-form) twin of the chimera training
                    # class. Same constructor surface — both pool sizes
                    # arrive from the chimera-stamped metadata via
                    # ``cfg.from_weights``.
                    extra_kwargs["num_experts_content"] = cfg.num_experts_content
                    extra_kwargs["num_experts_freq"] = cfg.num_experts_freq
                elif effective_module_class == OrthoHydraLoRAModule:
                    extra_kwargs["num_experts"] = cfg.num_experts
                    if self._use_global_router_for_hydra:
                        extra_kwargs["use_global_router"] = True
                        self._global_router_hits += 1
                elif effective_module_class == HydraLoRAModule:
                    extra_kwargs["num_experts"] = cfg.num_experts
                    if cfg.expert_init_std > 0.0:
                        extra_kwargs["expert_init_std"] = cfg.expert_init_std
                    if self._use_global_router_for_hydra:
                        extra_kwargs["use_global_router"] = True
                        self._global_router_hits += 1
                    if cfg.use_chimera_hydra:
                        # Dual-pool runtime form (load path from a distilled
                        # chimera checkpoint — see factory.py is_chimera_hydra
                        # branch). HydraLoRAModule narrows its router to K_c
                        # outputs and registers _freq_routing_weights for the
                        # network-level FreqRouter broadcast. σ/FEI feature
                        # dims must stay 0 here — FreqRouter owns those axes.
                        extra_kwargs["num_experts_content"] = cfg.num_experts_content
                elif effective_module_class == StackedExpertsLoRAModule:
                    # Independent-A (FeRA). Gates arrive via the network-level
                    # ``GlobalRouter`` through the shared ``_routing_weights``
                    # buffer — no per-Linear router knob to set. ``num_experts``
                    # must match ``cfg.num_experts`` (and therefore the
                    # GlobalRouter's output width) or the routing-weight
                    # broadcast inside ``forward`` shape-mismatches.
                    extra_kwargs["num_experts"] = cfg.num_experts
                    extra_kwargs["ortho"] = cfg.use_ortho
                    if cfg.use_ortho:
                        extra_kwargs["ortho_init_std"] = cfg.ortho_init_std

                # Hard σ-band expert partition: applied to every Hydra/
                # OrthoHydra module (independent of the σ-feature router
                # regex). Each module owns the partition; the network-level
                # ``set_sigma`` propagates ``_sigma`` to enable per-step band
                # selection. Validation (E % N == 0) lives in cfg parsing.
                if (
                    cfg.specialize_experts_by_sigma_buckets
                    and effective_module_class
                    in (HydraLoRAModule, OrthoHydraLoRAModule)
                    and is_unet
                ):
                    extra_kwargs["specialize_experts_by_sigma_buckets"] = True
                    extra_kwargs["num_sigma_buckets"] = cfg.num_sigma_buckets
                    if cfg.sigma_bucket_boundaries is not None:
                        extra_kwargs["sigma_bucket_boundaries"] = (
                            cfg.sigma_bucket_boundaries
                        )

                # σ-conditional router: only widen the router input with
                # sinusoidal(σ) features on modules whose name matches the
                # layer filter (cross_attn.q / self_attn.qkv by default — see
                # B0 pre-analysis in timestep-hydra.md). From-weights path uses
                # an explicit name set; fresh-from-kwargs path uses a regex
                # over original_name. Gated on the effective class so a
                # hydra-excluded module can't pick up σ either. Skipped under
                # ``use_global_router`` — the network-level router consumes
                # the routing signal once and the per-Linear cat is dead.
                if (
                    cfg.router_source == "sigma"
                    and effective_module_class
                    in (
                        HydraLoRAModule,
                        OrthoHydraLoRAModule,
                    )
                    and is_unet
                    and not self._use_global_router_for_hydra
                ):
                    if self._sigma_router_names is not None:
                        enable = lora_name in self._sigma_router_names
                    elif self._sigma_router_re is not None:
                        enable = bool(self._sigma_router_re.search(original_name))
                    else:
                        enable = True
                    if enable:
                        extra_kwargs["sigma_feature_dim"] = cfg.sigma_feature_dim
                        self._sigma_router_hits += 1

                # FEI-conditional router (FeRA-style). Same gating as σ —
                # widen the router input with the per-sample FEI simplex on
                # modules whose name matches the layer filter. The FEI tensor
                # itself is computed once per step in the train/inference loop
                # and propagated via ``LoRANetwork.set_fei``. Skipped under
                # ``use_global_router`` — the GlobalRouter reads FEI directly
                # at the network level and per-Linear cat is dead.
                if (
                    cfg.router_source == "fei"
                    and effective_module_class
                    in (
                        HydraLoRAModule,
                        OrthoHydraLoRAModule,
                    )
                    and is_unet
                    and not self._use_global_router_for_hydra
                ):
                    if self._fei_router_names is not None:
                        enable_fei = lora_name in self._fei_router_names
                    elif self._fei_router_re is not None:
                        enable_fei = bool(self._fei_router_re.search(original_name))
                    else:
                        enable_fei = True
                    if enable_fei:
                        extra_kwargs["fei_feature_dim"] = cfg.fei_feature_dim
                        self._fei_router_hits += 1

                # Per-channel scaling is DiT-only: the bench script hooks DiT
                # linears, text encoder activations are never calibrated.
                if cfg.channel_scales_dict is not None and is_unet:
                    _cs = cfg.channel_scales_dict.get(lora_name)
                    if _cs is not None:
                        extra_kwargs["channel_scale"] = _cs
                        self._channel_scale_hits += 1
                    else:
                        self._channel_scale_misses.append(lora_name)

                lora = effective_module_class(
                    lora_name,
                    child_module,
                    self.multiplier,
                    dim,
                    alpha_val,
                    dropout=dropout,
                    rank_dropout=rank_dropout,
                    module_dropout=module_dropout,
                    **extra_kwargs,
                )
                lora.original_name = original_name
                loras.append(lora)

            return loras, skipped

        # Create LoRA for text encoders (Qwen3 - typically not trained for Anima)
        # Skip for OrthoLoRA since SVD init is expensive and TE modules are discarded in apply_to anyway
        self.text_encoder_loras: List[LoRAModule] = []
        skipped_te = []
        if text_encoders is not None and module_class not in (
            OrthoLoRAModule,
            OrthoHydraLoRAModule,
            ChimeraHydraLoRAModule,
            ChimeraHydraInferenceModule,
        ):
            for i, text_encoder in enumerate(text_encoders):
                if text_encoder is None:
                    continue
                logger.info(f"create LoRA for Text Encoder {i + 1}:")
                te_loras, te_skipped = create_modules(
                    False,
                    i,
                    text_encoder,
                    LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE,
                )
                logger.info(
                    f"create LoRA for Text Encoder {i + 1}: {len(te_loras)} modules."
                )
                self.text_encoder_loras.extend(te_loras)
                skipped_te += te_skipped

        # Create LoRA for DiT blocks
        target_modules = list(LoRANetwork.ANIMA_TARGET_REPLACE_MODULE)
        if train_llm_adapter:
            target_modules.extend(LoRANetwork.ANIMA_ADAPTER_TARGET_REPLACE_MODULE)

        self.unet_loras: List[LoRAModule]
        self.unet_loras, skipped_un = create_modules(True, None, unet, target_modules)

        logger.info(f"create LoRA for Anima DiT: {len(self.unet_loras)} modules.")
        if verbose:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:60} {lora.lora_dim}, {lora.alpha}")

        skipped = skipped_te + skipped_un
        if verbose and len(skipped) > 0:
            logger.warning(f"dim (rank) is 0, {len(skipped)} LoRA modules are skipped:")
            for name in skipped:
                logger.info(f"\t{name}")

        if cfg.channel_scales_dict is not None:
            logger.info(
                f"per_channel_scaling: {self._channel_scale_hits} DiT modules "
                f"received calibration-based input scaling"
            )
            if self._channel_scale_misses:
                logger.warning(
                    f"per_channel_scaling: {len(self._channel_scale_misses)} DiT modules "
                    f"have no calibration stats (first: {self._channel_scale_misses[:3]}). "
                    f"These will train without input rebalancing — regenerate the stats "
                    f"file with `python archive/bench/analyze_lora_input_channels.py "
                    f"--dump_channel_stats <path>` if this is unexpected."
                )

        # Create ReFT modules on the DiT residual stream (block outputs), following
        # Wu et al. (2024) §3.3 — one intervention per selected block, not per
        # internal Linear. Selection is controlled by ``reft_layers``.
        self.unet_refts: List[ReFTModule] = []
        self.text_encoder_refts: List[ReFTModule] = []
        if add_reft:
            dit_blocks = getattr(unet, "blocks", None)
            if dit_blocks is None or len(dit_blocks) == 0:
                raise ValueError(
                    "add_reft=True but DiT has no .blocks attribute to wrap. "
                    "Block-level ReFT requires a transformer with a `blocks` ModuleList."
                )
            num_blocks = len(dit_blocks)
            selected_indices = _parse_reft_layers(reft_layers, num_blocks)

            reft_alpha_value = reft_alpha if reft_alpha is not None else alpha
            for idx in selected_indices:
                block = dit_blocks[idx]
                block_embed_dim = getattr(block, "x_dim", None)
                if block_embed_dim is None:
                    raise ValueError(
                        f"Block {idx} ({type(block).__name__}) has no `x_dim`; "
                        "cannot infer embed_dim for ReFT."
                    )
                reft_name = f"reft_unet_blocks_{idx}"
                reft = ReFTModule(
                    reft_name,
                    block,
                    embed_dim=block_embed_dim,
                    multiplier=multiplier,
                    reft_dim=reft_dim,
                    alpha=reft_alpha_value,
                    dropout=dropout,
                    module_dropout=module_dropout,
                )
                reft.original_name = f"blocks.{idx}"
                self.unet_refts.append(reft)
            logger.info(
                f"create ReFT for Anima DiT: {len(self.unet_refts)}/{num_blocks} "
                f"blocks (reft_dim={reft_dim}, layers={reft_layers!r})"
            )

        # assertion: no duplicate names
        names = set()
        for lora in (
            self.text_encoder_loras
            + self.unet_loras
            + self.text_encoder_refts
            + self.unet_refts
        ):
            assert lora.lora_name not in names, (
                f"duplicated lora name: {lora.lora_name}"
            )
            names.add(lora.lora_name)

        # Alias each sigma-aware module's ``_sigma`` / ``_sigma_features``
        # buffer to a single network-level shared tensor. ``set_sigma`` then
        # updates the shared tensor in place once and every aliased module
        # buffer sees the new value through shared storage — instead of
        # ~56 per-module ``copy_`` calls per training step.
        self._wire_shared_sigma_buffers()
        self._wire_shared_fei_buffers()
        self._wire_shared_routing_buffers()
        self._wire_shared_freq_routing_buffers()

        # Build the network-level GlobalRouter when the cfg selects MoE
        # without per-Linear routers. The input dim is derived from the
        # routing signal: ``"fei"`` → ``fei_feature_dim`` simplex,
        # ``"sigma"`` → ``sigma_feature_dim`` sinusoidal features.
        # Routing-aware modules: ``independent_A`` (StackedExperts) always
        # consume the broadcast gates; ``shared_A`` (Hydra / OrthoHydra)
        # consumes them when built with ``use_global_router=True``.
        self.global_router: Optional[GlobalRouter] = None
        if cfg.use_moe_style is not False and not cfg.route_per_layer:
            if cfg.router_source == "fei":
                router_input_dim = int(cfg.fei_feature_dim)
            elif cfg.router_source == "sigma":
                router_input_dim = int(cfg.sigma_feature_dim)
            else:
                router_input_dim = 0
            if router_input_dim > 0 and cfg.num_experts > 1:
                self.global_router = GlobalRouter(
                    input_dim=router_input_dim,
                    num_experts=int(cfg.num_experts),
                    hidden_dim=int(cfg.router_hidden_dim),
                    tau=float(cfg.router_tau),
                )
                logger.info(
                    f"GlobalRouter: source={cfg.router_source!r}, "
                    f"input_dim={router_input_dim}, "
                    f"num_experts={cfg.num_experts}, "
                    f"hidden={cfg.router_hidden_dim}, τ={cfg.router_tau:.2f}, "
                    f"routing-aware modules={len(self._routing_aware_loras)}"
                )

        # ChimeraHydra FreqRouter: one per network, broadcasts ``π_f`` over
        # the freq pool of every chimera module. Input is
        # ``concat(FEI, sinusoidal-σ-features)`` — owned by the freq router
        # exclusively (the per-layer content router never sees σ/FEI). Built
        # only when at least one chimera module was actually constructed; the
        # router_targets regex can narrow the chimera class to a subset of
        # layers (others fall back to OrthoLoRA).
        self.freq_router: Optional[FreqRouter] = None
        if cfg.use_chimera_hydra and self._chimera_aware_loras:
            freq_input_dim = int(cfg.fei_feature_dim) + int(cfg.sigma_feature_dim)
            if freq_input_dim <= 0:
                raise ValueError(
                    "use_chimera_hydra=True requires fei_feature_dim + "
                    f"sigma_feature_dim > 0 for the FreqRouter input (got "
                    f"FEI={cfg.fei_feature_dim}, σ={cfg.sigma_feature_dim})."
                )
            self.freq_router = FreqRouter(
                input_dim=freq_input_dim,
                num_freq_experts=int(cfg.num_experts_freq),
                hidden_dim=int(cfg.router_hidden_dim),
                tau=float(cfg.router_tau),
                init_std=float(cfg.freq_router_init_std),
                fei_dim=int(cfg.fei_feature_dim),
                sigma_dim=int(cfg.sigma_feature_dim),
                apply_layer_norm=bool(cfg.freq_router_layer_norm),
            )
            # Force the per-step conditioning hook to fire set_fei every
            # step (router_conditioning.py reads this flag). Chimera ties
            # σ + FEI together for the freq router input, so the set_fei
            # path is where we re-fire FreqRouter.
            self.use_fei_router = True
            logger.info(
                f"ChimeraHydra FreqRouter: input_dim={freq_input_dim} "
                f"(FEI={cfg.fei_feature_dim} + σ={cfg.sigma_feature_dim}), "
                f"K_f={cfg.num_experts_freq}, hidden={cfg.router_hidden_dim}, "
                f"τ={cfg.router_tau:.2f}, init_std={cfg.freq_router_init_std}, "
                f"LN={self.freq_router.apply_layer_norm}, "
                f"chimera modules={len(self._chimera_aware_loras)}"
            )

    def _wire_shared_sigma_buffers(self) -> None:
        """Replace each HydraLoRA / OrthoHydraLoRA module's ``_sigma`` and
        ``_sigma_features`` buffers with references to a single network-level
        tensor (per sigma_feature_dim for the features). Modules then read the
        same tensor object as their own attribute, so an in-place ``copy_`` on
        the network's shared buffer flows to every module without a Python
        propagation loop.

        Run once at the end of ``__init__`` — before any forward fires, so
        Dynamo / cudagraphs capture the aliased data pointer on first compile
        and never see a per-module pointer-mismatch event.
        """
        sigma_loras: List[torch.nn.Module] = []
        by_dim: Dict[int, List[torch.nn.Module]] = {}
        for lora in self.unet_loras + self.text_encoder_loras:
            if "_sigma" not in lora._buffers:
                continue
            sigma_loras.append(lora)
            d = int(getattr(lora, "sigma_feature_dim", 0))
            if d > 0 and "_sigma_features" in lora._buffers:
                by_dim.setdefault(d, []).append(lora)
        self._sigma_aware_loras = sigma_loras
        self._sigma_aware_loras_by_dim = by_dim
        if not sigma_loras:
            self._shared_sigma = None
            self._shared_sigma_features: Dict[int, torch.Tensor] = {}
            return

        # Pick the first module's placeholder buffer as the canonical shared
        # tensor; rebind every other module's buffer to the same object. The
        # placeholder is shape (1,) / (1, dim) — set_sigma replaces it with a
        # full-shape tensor on the first call (and re-aliases at the same time).
        shared_sigma = sigma_loras[0]._buffers["_sigma"]
        for lora in sigma_loras:
            lora._buffers["_sigma"] = shared_sigma
        self._shared_sigma = shared_sigma

        self._shared_sigma_features = {}
        for dim, loras in by_dim.items():
            shared_feat = loras[0]._buffers["_sigma_features"]
            for lora in loras:
                lora._buffers["_sigma_features"] = shared_feat
            self._shared_sigma_features[dim] = shared_feat

    def _wire_shared_fei_buffers(self) -> None:
        """Replace each FEI-aware module's ``_fei`` buffer with a single
        network-level shared tensor (per FEI feature dim).

        Mirrors ``_wire_shared_sigma_buffers``. ``set_fei`` writes to one
        shared buffer per dim; aliased module ``_fei`` buffers see the
        update through shared storage. The aliasing-recovery dance from
        ``set_sigma`` (rebind whenever shape or device drift breaks the
        identity) applies here too — ``Module._apply`` (``.to(device)``)
        independently reallocates buffers and silently breaks the link if
        we don't identity-check. See ``[[project_set_sigma_aliasing_bug]]``.
        """
        fei_loras: List[torch.nn.Module] = []
        by_dim: Dict[int, List[torch.nn.Module]] = {}
        for lora in self.unet_loras + self.text_encoder_loras:
            d = int(getattr(lora, "fei_feature_dim", 0))
            if d <= 0:
                continue
            if "_fei" not in lora._buffers:
                continue
            fei_loras.append(lora)
            by_dim.setdefault(d, []).append(lora)
        self._fei_aware_loras = fei_loras
        self._fei_aware_loras_by_dim = by_dim
        if not fei_loras:
            self._shared_fei: Dict[int, torch.Tensor] = {}
            return

        # One shared placeholder per dim — ``set_fei`` rebinds to full-shape
        # ``(B, dim)`` on first call.
        self._shared_fei = {}
        for dim, loras in by_dim.items():
            shared_feat = loras[0]._buffers["_fei"]
            for lora in loras:
                lora._buffers["_fei"] = shared_feat
            self._shared_fei[dim] = shared_feat

    def _wire_shared_routing_buffers(self) -> None:
        """Alias every routing-aware module's ``_routing_weights`` buffer to
        one network-level shared tensor.

        Mirrors ``_wire_shared_sigma_buffers`` / ``_wire_shared_fei_buffers``.
        ``StackedExpertsLoRAModule.__init__`` registers a ``(1, E)`` uniform
        placeholder; this pass picks the first module's buffer as canonical
        and rebinds every other module to the same object. ``set_routing_weights``
        then updates one shared tensor per step; aliased module buffers
        see the new gates through shared storage.

        All routing-aware modules in our build share the same ``num_experts``
        by construction (driven by ``cfg.num_experts``), so a single shared
        tensor is enough — no per-dim split like ``_shared_fei``.
        """
        routing_loras: List[torch.nn.Module] = []
        for lora in self.unet_loras + self.text_encoder_loras:
            if "_routing_weights" not in lora._buffers:
                continue
            routing_loras.append(lora)
        self._routing_aware_loras = routing_loras
        if not routing_loras:
            self._shared_routing_weights: Optional[torch.Tensor] = None
            return

        canonical = routing_loras[0]._buffers["_routing_weights"]
        for lora in routing_loras:
            lora._buffers["_routing_weights"] = canonical
        self._shared_routing_weights = canonical

    def _wire_shared_freq_routing_buffers(self) -> None:
        """Alias every chimera module's ``_freq_routing_weights`` buffer to one
        shared tensor.

        Parallel to ``_wire_shared_routing_buffers`` but on the chimera-
        specific buffer name. FreqRouter broadcasts ``π_f`` once per step via
        direct slot assignment; aliased buffers on every chimera module see
        the new gates through shared storage. The buffer must carry the
        router's grad_fn (NO .detach(), NO .copy_()) so ``∂L_denoise/∂π_f``
        reaches the FreqRouter — same contract as
        ``router_state._set_routing_weights``.
        """
        freq_loras: List[torch.nn.Module] = []
        for lora in self.unet_loras + self.text_encoder_loras:
            if "_freq_routing_weights" not in lora._buffers:
                continue
            freq_loras.append(lora)
        self._chimera_aware_loras = freq_loras
        if not freq_loras:
            self._shared_freq_routing_weights: Optional[torch.Tensor] = None
            return

        canonical = freq_loras[0]._buffers["_freq_routing_weights"]
        for lora in freq_loras:
            lora._buffers["_freq_routing_weights"] = canonical
        self._shared_freq_routing_weights = canonical

    def prepare_network(self, args):
        if getattr(args, "lora_fp32_accumulation", False):
            logger.warning(
                "--lora_fp32_accumulation is deprecated and has no effect; "
                "fp32 accumulation is now unconditional in LoRA/Hydra/ReFT "
                "bottleneck matmuls. Remove the flag from your config."
            )

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier
        for reft in self.text_encoder_refts + self.unet_refts:
            reft.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def fuse_weights(self):
        """Merge all LoRA deltas into base model weights for zero-overhead inference."""
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.fuse_weight()

    def unfuse_weights(self):
        """Remove all LoRA deltas from base model weights."""
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.unfuse_weight()

    def set_timestep_mask(self, timesteps: torch.Tensor, max_timestep: float = 1.0):
        """Compute and set timestep-dependent rank mask on all modules."""
        if not self.cfg.use_timestep_mask:
            return

        max_rank = self.cfg.lora_dim
        # Reuse a single GPU-resident mask to avoid ~200 CPU→GPU transfers per step
        mask = getattr(self, "_shared_timestep_mask", None)
        if mask is None or mask.device != timesteps.device:
            mask = torch.zeros(1, max_rank, device=timesteps.device)
            self._shared_timestep_mask = mask
            self._timestep_mask_arange = torch.arange(max_rank, device=timesteps.device)
            for lora in self.text_encoder_loras + self.unet_loras:
                lora._timestep_mask = mask

        # Compute threshold r entirely on device — avoids GPU→CPU .item() sync and
        # keeps the effective rank as a tensor so the mask build stays static-shape.
        t = timesteps.float().mean()
        frac = ((max_timestep - t) / max_timestep).clamp(min=0.0, max=1.0)
        r = (
            frac.pow(self.cfg.alpha_rank_scale) * (max_rank - self.cfg.min_rank)
            + self.cfg.min_rank
        )
        r = r.clamp(max=float(max_rank))
        mask.copy_((self._timestep_mask_arange < r).to(mask.dtype).unsqueeze(0))

    def set_reft_timestep_mask(
        self, timesteps: torch.Tensor, max_timestep: float = 1.0
    ):
        """Compute and set timestep-dependent mask on ReFT modules."""
        if not self.cfg.use_timestep_mask:
            return
        refts = self.text_encoder_refts + self.unet_refts
        if not refts:
            return
        reft_dim = self.cfg.reft_dim

        mask = getattr(self, "_shared_reft_mask", None)
        if mask is None or mask.device != timesteps.device:
            mask = torch.zeros(1, reft_dim, device=timesteps.device)
            self._shared_reft_mask = mask
            self._reft_mask_arange = torch.arange(reft_dim, device=timesteps.device)
            for reft in refts:
                reft._timestep_mask = mask

        t = timesteps.float().mean()
        frac = ((max_timestep - t) / max_timestep).clamp(min=0.0, max=1.0)
        r = frac.pow(self.cfg.alpha_rank_scale) * (reft_dim - 1) + 1
        r = r.clamp(max=float(reft_dim))
        mask.copy_((self._reft_mask_arange < r).to(mask.dtype).unsqueeze(0))

    def clear_timestep_mask(self):
        """Restore full-rank masks on every LoRA / ReFT module.

        Each module's ``_timestep_mask`` is a Tensor by construction (default
        all-ones buffer at init, rebound to the shared live-updated mask when
        ``set_timestep_mask`` runs). Clearing fills the shared masks with ones
        in place — modules that were rebound immediately see the neutral mask
        via the shared reference; modules with local defaults are already
        neutral. Never set to None: the always-a-Tensor invariant is what
        keeps the adapter forward free of a None-vs-Tensor guard under
        ``compile_mode=full``.
        """
        shared = getattr(self, "_shared_timestep_mask", None)
        if shared is not None:
            shared.fill_(1.0)
        shared_reft = getattr(self, "_shared_reft_mask", None)
        if shared_reft is not None:
            shared_reft.fill_(1.0)

    def set_sigma(self, sigmas: torch.Tensor) -> None:
        """Stash per-sample σ on every HydraLoRA module whose router accepts σ.

        Mirrors ``set_timestep_mask`` — one call per training step. σ and the
        sinusoidal-features tensor are stored in network-level shared buffers
        whose storage is aliased into every sigma-aware module's ``_sigma`` /
        ``_sigma_features`` (see ``_wire_shared_sigma_buffers``), so the
        update is one in-place ``copy_`` per shared tensor instead of a
        per-module Python loop.

        IMPORTANT: write in place rather than rebinding. Inductor captures
        the buffers as static cudagraph inputs and re-records the whole graph
        if the data pointer changes — rebinding every step caused per-step
        re-record under ``compile_inductor_mode=reduce-overhead``
        (cudagraph_trees log: "static input data pointer changed"). Pointer
        only changes on the first call (placeholder → full-shape) and on a
        rare batch-shape change; both re-alias every module to the new tensor.

        Aliasing-recovery: ``Module._apply`` (i.e. ``network.to(device)``)
        reallocates each registered buffer independently, breaking the
        identity established by ``_wire_shared_sigma_buffers``. The
        ``self._shared_sigma`` Python attribute is *not* touched by
        ``Module._apply`` (it isn't a registered buffer), so post-``.to(...)``
        we may have a stale CPU shared tensor while the modules' ``_sigma``
        buffers all live on GPU and are no longer aliased to anything. Detect
        this on every call (cheap identity check against the canonical module's
        live buffer) and force the rebind path to re-establish aliasing —
        otherwise the in-place ``copy_`` writes to the orphaned CPU tensor
        and every module silently keeps reading its own zero-initialized
        ``_sigma``. This bug only manifests at B=1 (placeholder shape (1,)
        matches runtime shape so the historical rebind path was skipped),
        which is why σ-band partition and σ-feature router were both dead at
        ``batch_size=1`` despite the unit tests passing in eager mode.
        """
        sigmas = sigmas.detach()
        self._last_sigma = sigmas
        # Either path needs per-module ``_sigma``: σ-feature concat router
        # (sigma_feature_dim>0) and hard σ-band expert partition. Skip the
        # propagation entirely when neither is configured.
        if not (
            self.cfg.router_source == "sigma"
            or self.cfg.specialize_experts_by_sigma_buckets
        ):
            return
        sigma_loras = self._sigma_aware_loras
        if not sigma_loras:
            return

        # Canonical = the live buffer on the first sigma-aware module. After
        # ``network.to(device)`` this is the GPU-allocated tensor; before any
        # device move it's still the CPU placeholder from
        # ``_wire_shared_sigma_buffers``.
        canonical = sigma_loras[0]._buffers["_sigma"]
        cast = sigmas.to(dtype=canonical.dtype, device=canonical.device)
        # Rebind whenever (a) the shared attribute lost identity with the
        # canonical (e.g. ``.to()`` rebinding broke aliasing) or (b) the
        # shape changed (placeholder → full batch). Both branches need to
        # re-alias every module so the next call's fast path actually
        # propagates.
        needs_rebind = (
            self._shared_sigma is not canonical
            or canonical.shape != cast.shape
        )
        if needs_rebind:
            new_sigma = cast.detach().clone()
            for lora in sigma_loras:
                lora._buffers["_sigma"] = new_sigma
            self._shared_sigma = new_sigma
            shared_sigma = new_sigma
        else:
            canonical.copy_(cast)
            shared_sigma = canonical

        for dim, loras in self._sigma_aware_loras_by_dim.items():
            canonical_feat = loras[0]._buffers["_sigma_features"]
            feat = _sigma_sinusoidal_features(shared_sigma, dim).detach()
            cast_feat = feat.to(
                dtype=canonical_feat.dtype, device=canonical_feat.device
            )
            feat_needs_rebind = (
                self._shared_sigma_features.get(dim) is not canonical_feat
                or canonical_feat.shape != cast_feat.shape
            )
            if feat_needs_rebind:
                new_feat = cast_feat.clone()
                for lora in loras:
                    lora._buffers["_sigma_features"] = new_feat
                self._shared_sigma_features[dim] = new_feat
            else:
                canonical_feat.copy_(cast_feat)

    def clear_sigma(self) -> None:
        """Reset cached σ to zeros.

        Never set to None: ``_sigma`` stays a Tensor so the unconditional
        sinusoidal path in ``_compute_gate`` has no None-vs-Tensor guard to
        recompile on under ``compile_mode=full``. Used in eval / validation
        and by inference teardown (``clear_hydra_sigma``). Zero in place to
        keep the cudagraph data pointer stable (see ``set_sigma`` note).

        Like ``set_sigma``, must operate on the *live* per-module buffer —
        ``Module._apply`` (``.to(device)``) breaks the init-time aliasing,
        and ``self._shared_sigma`` may then point at an orphaned CPU tensor
        whose zeroing wouldn't reach any module. Zero the canonical module
        buffer instead and re-establish aliasing if it was broken.
        """
        self._last_sigma = None
        if not self._sigma_aware_loras:
            return
        sigma_loras = self._sigma_aware_loras
        canonical = sigma_loras[0]._buffers["_sigma"]
        if self._shared_sigma is not canonical:
            for lora in sigma_loras:
                lora._buffers["_sigma"] = canonical
            self._shared_sigma = canonical
        canonical.zero_()
        for dim, loras in self._sigma_aware_loras_by_dim.items():
            canonical_feat = loras[0]._buffers["_sigma_features"]
            if self._shared_sigma_features.get(dim) is not canonical_feat:
                for lora in loras:
                    lora._buffers["_sigma_features"] = canonical_feat
                self._shared_sigma_features[dim] = canonical_feat
            zero_feat = _sigma_sinusoidal_features(canonical, dim)
            cast_feat = zero_feat.to(
                dtype=canonical_feat.dtype, device=canonical_feat.device
            )
            if canonical_feat.shape == cast_feat.shape:
                canonical_feat.copy_(cast_feat)
            else:
                new_feat = cast_feat.detach().clone()
                for lora in loras:
                    lora._buffers["_sigma_features"] = new_feat
                self._shared_sigma_features[dim] = new_feat

    def set_fei(self, fei: torch.Tensor) -> None:
        """Stash per-sample FEI ``[B, fei_dim]`` on every FEI-aware module.

        Parallel to ``set_sigma`` — one call per training/inference step.
        Same shared-buffer aliasing recovery: identity-check ``self._shared_fei``
        against the canonical module's live buffer, rebind on shape change
        or after ``Module._apply`` orphans the link
        (``[[project_set_sigma_aliasing_bug]]``).

        ``fei`` must be ``(B, fei_feature_dim)`` matching
        ``cfg.fei_feature_dim`` (default 2 for the simplex). Caller is the
        train/inference loop running ``library.runtime.fei.compute_fei_2band``
        on ``z_t`` once per step.

        When ``cfg.route_per_layer=False`` and a ``GlobalRouter`` is wired,
        the router fires on the fresh FEI and its gates are broadcast to
        every routing-aware module via ``set_routing_weights`` in the same
        call — one entry point for the FeRA-style global-router path.
        """
        fei = fei.detach()
        # Fast-path: if there are no per-Linear FEI consumers, no global
        # router, and no chimera FreqRouter needing FEI, nothing to do.
        has_per_layer_fei = bool(getattr(self, "_fei_aware_loras", None))
        global_fei_router = (
            self.global_router
            if (
                self.global_router is not None
                and self.cfg.router_source == "fei"
                and not self.cfg.route_per_layer
            )
            else None
        )
        chimera_freq_router = (
            self.freq_router
            if (
                getattr(self, "freq_router", None) is not None
                and getattr(self, "_chimera_aware_loras", None)
            )
            else None
        )
        if not (
            has_per_layer_fei
            or global_fei_router is not None
            or chimera_freq_router is not None
        ):
            return
        if not (
            self.use_fei_router
            or global_fei_router is not None
            or chimera_freq_router is not None
        ):
            return

        # Per-layer FEI broadcast (legacy path — FEI-on-Hydra Phase 1).
        if has_per_layer_fei:
            # Group loras by their feature dim — every fei-aware module
            # currently in our network shares the same dim (cfg-level), but
            # the loop is robust to a future per-layer dim override.
            for dim, loras in self._fei_aware_loras_by_dim.items():
                canonical = loras[0]._buffers["_fei"]
                cast = fei.to(dtype=canonical.dtype, device=canonical.device)
                if cast.dim() == 1:
                    cast = cast.unsqueeze(0)
                if cast.shape[-1] != dim:
                    raise ValueError(
                        f"set_fei: fei.shape[-1]={cast.shape[-1]} != fei_feature_dim={dim}"
                    )
                current_shared = self._shared_fei.get(dim)
                needs_rebind = (
                    current_shared is not canonical
                    or canonical.shape != cast.shape
                )
                if needs_rebind:
                    new_fei = cast.detach().clone()
                    for lora in loras:
                        lora._buffers["_fei"] = new_fei
                    self._shared_fei[dim] = new_fei
                else:
                    canonical.copy_(cast)

        # Global router (FeRA-style): fire on fresh FEI and broadcast gates.
        # Router runs WITH grad so the autograd path ``L_denoise → y_t →
        # α_{t,m} → g_φ`` (FeRA eq. 6-7, 11) reaches the GlobalRouter params.
        # ``set_routing_weights`` reassigns each expert module's buffer slot
        # to the live ``gates`` tensor (no detach, no in-place copy).
        if global_fei_router is not None:
            gates = global_fei_router(fei)
            self.set_routing_weights(gates)

        # ChimeraHydra FreqRouter: input is concat(FEI, sinusoidal-σ-features).
        # σ already arrived through ``set_sigma`` (which fires before
        # ``set_fei`` in ``apply_router_conditioning``); the freq router lives
        # at network level and computes its features fresh each step rather
        # than relying on per-module shared σ-feature buffers (chimera modules
        # are built with ``sigma_feature_dim=0`` since the freq router owns
        # the σ axis exclusively).
        if chimera_freq_router is not None:
            sigma = self._last_sigma
            if sigma is None:
                raise RuntimeError(
                    "ChimeraHydra FreqRouter requires set_sigma to fire before "
                    "set_fei within the same step (apply_router_conditioning "
                    "preserves this order — check custom call sites)."
                )
            sigma_dim = int(self.cfg.sigma_feature_dim)
            sigma_feat = _sigma_sinusoidal_features(sigma, sigma_dim)
            # Match the FEI tensor's device/dtype and batch axis. Both should
            # share the same B by construction (one σ per sample, one FEI per
            # sample), so a straight cat is correct.
            fei_cast = fei.to(device=sigma_feat.device, dtype=sigma_feat.dtype)
            if fei_cast.dim() == 1:
                fei_cast = fei_cast.unsqueeze(0)
            router_in = torch.cat([fei_cast, sigma_feat], dim=-1)
            freq_gates = chimera_freq_router(router_in)
            self.set_freq_routing_weights(freq_gates)

    def clear_fei(self) -> None:
        """Reset cached FEI to zeros without rebinding pointers.

        Same in-place-zero pattern as ``clear_sigma`` — keeps cudagraph
        data pointers stable. Re-establishes aliasing if ``Module._apply``
        broke it since the last call.
        """
        if not getattr(self, "_fei_aware_loras", None):
            return
        for dim, loras in self._fei_aware_loras_by_dim.items():
            canonical = loras[0]._buffers["_fei"]
            current_shared = self._shared_fei.get(dim)
            if current_shared is not canonical:
                for lora in loras:
                    lora._buffers["_fei"] = canonical
                self._shared_fei[dim] = canonical
            canonical.zero_()

    def set_routing_weights(self, weights: torch.Tensor) -> None:
        """Broadcast a ``(B, E)`` gate tensor to every routing-aware module.

        Called either:
          * Internally by ``set_fei`` when ``cfg.route_per_layer=False`` and
            ``cfg.router_source="fei"`` — the GlobalRouter fires on the
            fresh FEI and its output is broadcast here.
          * Externally by future ``"sigma"`` global-router paths /
            inference callers needing to push pre-computed gates.

        Assigns the SAME live ``weights`` tensor reference to every routing-
        aware module's ``_routing_weights`` buffer slot (no detach, no in-
        place copy). This is what gives the GlobalRouter its gradient path:
        ``L_denoise`` backprop flows through ``y_t = Σ α_{t,m} E_m(z_t)``
        (FeRA eq. 7) into ``α``, then through the assigned buffer reads
        into ``g_φ``'s parameters. The cudagraph-pointer-stability story is
        intentionally traded away here — gates are a tiny ``(B, E)`` tensor
        and the autograd path is what makes the router train at all.
        """
        if not getattr(self, "_routing_aware_loras", None):
            return
        routing_loras = self._routing_aware_loras
        canonical_buf = routing_loras[0]._buffers["_routing_weights"]
        w = weights.to(dtype=canonical_buf.dtype, device=canonical_buf.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        for lora in routing_loras:
            lora._routing_weights = w
        self._shared_routing_weights = w

    def clear_routing_weights(self) -> None:
        """Reset gates to uniform ``1/E`` in place.

        Called between training steps (or by inference teardown). Pointer
        stays stable for cudagraph capture; re-aliases if ``Module._apply``
        broke the link.
        """
        if not getattr(self, "_routing_aware_loras", None):
            return
        routing_loras = self._routing_aware_loras
        canonical = routing_loras[0]._buffers["_routing_weights"]
        if self._shared_routing_weights is not canonical:
            for lora in routing_loras:
                lora._buffers["_routing_weights"] = canonical
            self._shared_routing_weights = canonical
        E = int(canonical.shape[-1])
        canonical.fill_(1.0 / max(E, 1))

    def set_freq_routing_weights(self, weights: torch.Tensor) -> None:
        """Broadcast ``π_f`` from the FreqRouter to every chimera module.

        Direct slot assignment (NO .detach(), NO .copy_()) so the buffer
        carries the router's grad_fn — same contract as
        ``set_routing_weights`` for the GlobalRouter. The chimera module's
        ``_compute_gate`` reads ``_freq_routing_weights`` to build the
        ``[π_c | π_f]`` concatenation, so the autograd path
        ``L_denoise → out_f → π_f → FreqRouter params`` is intact.
        """
        if not getattr(self, "_chimera_aware_loras", None):
            return
        freq_loras = self._chimera_aware_loras
        canonical_buf = freq_loras[0]._buffers["_freq_routing_weights"]
        w = weights.to(dtype=canonical_buf.dtype, device=canonical_buf.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        for lora in freq_loras:
            lora._freq_routing_weights = w
        self._shared_freq_routing_weights = w

    def clear_freq_routing_weights(self) -> None:
        """Reset chimera freq gates to uniform ``1/K_f`` in place."""
        if not getattr(self, "_chimera_aware_loras", None):
            return
        freq_loras = self._chimera_aware_loras
        canonical = freq_loras[0]._buffers["_freq_routing_weights"]
        if self._shared_freq_routing_weights is not canonical:
            for lora in freq_loras:
                lora._buffers["_freq_routing_weights"] = canonical
            self._shared_freq_routing_weights = canonical
        K_f = int(canonical.shape[-1])
        canonical.fill_(1.0 / max(K_f, 1))

    def clear_step_caches(self) -> None:
        """Drop per-step tensor references (``_last_gate``) and invalidate
        memoized router-stats caches between training steps.

        Called unconditionally from the training loop before each forward,
        for two reasons:

        (1) ``_last_gate`` caches a tensor produced inside the compiled
        forward — under ``torch.compile(mode='reduce-overhead')`` that tensor
        lives in the inductor cudagraph memory pool. Holding a Python
        reference across the step boundary prevents ``cudagraph_trees`` from
        reclaiming pool memory and silently demotes the run to the eager
        fallback path. Call must precede ``cudagraph_mark_step_begin()``.

        (2) ``_router_stats_cache`` / ``_chimera_router_stats_cache`` memoize
        per-step router diagnostics so the progress-bar postfix and the TB
        logging layer share one D2H sync. Without per-step invalidation
        these freeze at their first computed values — and on runs without
        cudagraph mode (``_cudagraph_mark_step=False``) the invalidation has
        no other trigger, so TB shows the same usage/entropy on every log
        step.

        ``_sigma`` is intentionally *not* cleared: it's rebound by
        ``set_sigma`` before every forward, the caller passes a tensor from
        outside the compiled region (the flow-matching sampler's ``timesteps``,
        not a pool-allocated intermediate), and keeping it a Tensor at all
        times is what lets the adapter ``_compute_gate`` drop the None-vs-
        Tensor guard under ``compile_mode=full``.

        Safe to call unconditionally — consumers (balance loss, router stats)
        read ``_last_gate`` only within the step that wrote it.
        """
        self._last_sigma = None
        self._router_stats_cache = None
        self._chimera_router_stats_cache = None
        for lora in self.unet_loras + self.text_encoder_loras:
            if hasattr(lora, "_last_gate"):
                lora._last_gate = None
        # Drop the GlobalRouter's per-step transients for the same reason —
        # ``_last_gates`` / ``_last_input`` are detached tensors that may live
        # in the inductor cudagraph memory pool; holding a Python reference
        # across the step boundary blocks pool reclamation.
        if self.global_router is not None:
            self.global_router._last_gates = None
            self.global_router._last_input = None
            self.global_router._last_fei = None
        # Same treatment for the chimera FreqRouter.
        if getattr(self, "freq_router", None) is not None:
            self.freq_router._last_gates = None
            self.freq_router._last_input = None

    def step_balance_loss_warmup(self, global_step: int, max_train_steps: int) -> None:
        """Activate the MoE load-balance penalty once training crosses the
        warmup window. Step function: ``_balance_loss_weight`` holds at 0
        during the first ``_balance_loss_warmup_ratio`` of steps, then flips
        to ``_balance_loss_target_weight``. No-op unless both attributes are
        attached (hydra post_init) and the ratio is > 0.

        Letting the router specialize before the penalty kicks in avoids
        pinning it to uniform at init; flipping the penalty on after warmup
        keeps a diverged router from collapsing to a single expert.
        """
        target = float(getattr(self, "_balance_loss_target_weight", 0.0) or 0.0)
        ratio = float(getattr(self, "_balance_loss_warmup_ratio", 0.0) or 0.0)
        if ratio <= 0.0 or max_train_steps <= 0 or target <= 0.0:
            return
        warmup_steps = int(max_train_steps * ratio)
        self._balance_loss_weight = 0.0 if global_step < warmup_steps else target

    @staticmethod
    def _switch_balance(gate: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer balance: E · Σ_i frac_i · mean_gate_i. Scalar."""
        num_experts = gate.shape[-1]
        expert_idx = gate.argmax(dim=-1)  # (B,)
        frac = torch.zeros(num_experts, device=gate.device, dtype=gate.dtype)
        frac.scatter_add_(0, expert_idx, torch.ones_like(expert_idx, dtype=gate.dtype))
        frac = frac / gate.shape[0]
        gate_mean = gate.mean(dim=0)  # (num_experts,)
        return num_experts * (frac * gate_mean).sum()

    def get_balance_loss(self) -> torch.Tensor:
        """Switch-Transformer load-balancing loss averaged over HydraLoRA modules.

        Global term aggregates gates over the full batch. When σ-conditional
        routing is on, also adds a per-σ-bucket term so global balance can't
        mask per-bucket collapse (expert i only at high σ, expert j only at
        low σ: globally balanced but per-bucket one-hot). Buckets are fixed
        thresholds on σ∈[0,1]; for N=3 that's [1/3, 2/3]. Under logit-normal
        σ sampling this is ~30/40/30 — close enough to equal-frequency for v1.

        ChimeraHydra path: ``_use_chimera_hydra=True`` splits each module's
        cached gate into the content slice ``[:K_c]`` and the freq slice
        ``[K_c:]``, then accumulates two independent switch losses weighted
        by ``_balance_w_content`` / ``_balance_w_freq``. A single combined
        term would let the optimizer satisfy the constraint by flattening
        one pool to uniform while concentrating the other — see proposal
        §"Balance loss".
        """
        # Chimera: dual-pool branch. Computed entirely separately from the
        # legacy single-pool / σ-bucket aggregation since the weights and
        # accumulation are independent per pool.
        if getattr(self, "_use_chimera_hydra", False):
            return self._get_chimera_balance_loss()

        total = None
        per_bucket_total = None
        count = 0
        per_bucket_count = 0

        sigma = self._last_sigma  # (B,) or None
        num_buckets = self.cfg.num_sigma_buckets
        bucket_w = float(self.cfg.per_bucket_balance_weight or 0.0)
        want_per_bucket = (
            self.cfg.router_source == "sigma"
            and sigma is not None
            and num_buckets > 1
            and bucket_w > 0.0
        )
        if want_per_bucket:
            thresholds = torch.linspace(0.0, 1.0, num_buckets + 1, device=sigma.device)[
                1:-1
            ]
            bucket_ids = torch.bucketize(sigma.float(), thresholds)  # (B,) in [0, N)

        for lora in self.unet_loras + self.text_encoder_loras:
            gate = getattr(lora, "_last_gate", None)
            if gate is None:
                continue
            term = self._switch_balance(gate)
            total = term if total is None else total + term
            count += 1

            if want_per_bucket and getattr(lora, "sigma_feature_dim", 0) > 0:
                # Only penalize per-bucket collapse on modules that actually
                # have σ-conditional routing capacity to collapse.
                module_bucket_sum = None
                module_bucket_count = 0
                for b in range(num_buckets):
                    mask = bucket_ids == b
                    if int(mask.sum()) < 2:
                        # Not enough samples to meaningfully measure balance
                        # in this bucket on this step; skip.
                        continue
                    bterm = self._switch_balance(gate[mask])
                    module_bucket_sum = (
                        bterm
                        if module_bucket_sum is None
                        else module_bucket_sum + bterm
                    )
                    module_bucket_count += 1
                if module_bucket_sum is not None:
                    per_bucket_total = (
                        module_bucket_sum / module_bucket_count
                        if per_bucket_total is None
                        else per_bucket_total + module_bucket_sum / module_bucket_count
                    )
                    per_bucket_count += 1

        if total is None:
            return torch.tensor(0.0)
        out = total / count
        if per_bucket_total is not None and per_bucket_count > 0:
            out = out + bucket_w * (per_bucket_total / per_bucket_count)
        return out

    def _get_chimera_balance_loss(self) -> torch.Tensor:
        """Dual-pool switch loss for the chimera path.

        Each module's cached gate is shape ``(B, K_c + K_f)``: the first
        ``K_c`` entries are π_c (per-layer content router), the rest are
        π_f (broadcast freq router). Compute Switch balance independently
        on each slice, average across modules, then combine with the
        per-pool weights ``_balance_w_content`` / ``_balance_w_freq``.

        Mathematically equivalent to ``w_c · L_balance(π_c) + w_f · L_balance(π_f)``
        from the proposal §Balance loss — independent terms force each
        pool to spread on its own pressure, unlike a single combined term
        which can collapse one pool to uniform while the other concentrates.

        Warmup contract is *asymmetric*: the per-Linear content router
        gates through ``_balance_loss_weight`` (held at 0 during the
        warmup window, then flipped to target). The freq pool bypasses
        warmup entirely — the network-level FreqRouter has its own
        symmetry-breaker (FEI input + ``freq_router_init_std``), so its
        load-balance pressure can safely fire from step 0. Trainer-side
        ``_hydra_balance_loss`` therefore consumes this scalar directly
        without re-multiplying by the outer warmup gate.
        """
        K_c_default = int(getattr(self.cfg, "num_experts_content", 0))
        total_c = None
        total_f = None
        count_c = 0
        count_f = 0
        for lora in self.unet_loras + self.text_encoder_loras:
            gate = getattr(lora, "_last_gate", None)
            if gate is None:
                continue
            K_c = int(getattr(lora, "num_experts_content", K_c_default))
            if K_c <= 0:
                continue
            gate_c = gate[..., :K_c]
            gate_f = gate[..., K_c:]
            if gate_c.shape[-1] > 1:
                term_c = self._switch_balance(gate_c)
                total_c = term_c if total_c is None else total_c + term_c
                count_c += 1
            if gate_f.shape[-1] > 1:
                term_f = self._switch_balance(gate_f)
                total_f = term_f if total_f is None else total_f + term_f
                count_f += 1

        if total_c is None and total_f is None:
            return torch.tensor(0.0)
        w_c = float(getattr(self, "_balance_w_content", 0.0) or 0.0)
        w_f = float(getattr(self, "_balance_w_freq", 0.0) or 0.0)
        # Apply the warmup gate to the CONTENT pool only; freq fires from
        # step 0 (see docstring). ``_balance_loss_weight`` is the warmup-
        # gated outer multiplier — content rides it, freq ignores it.
        # Trainer consumes the result of this method directly (no further
        # multiplication on the chimera path).
        outer = float(getattr(self, "_balance_loss_weight", 0.0) or 0.0)
        out = torch.tensor(0.0)
        if total_c is not None and count_c > 0:
            out = out + outer * w_c * (total_c / count_c)
        if total_f is not None and count_f > 0:
            out = out + w_f * (total_f / count_f)
        return out

    def get_router_entropy(self) -> Optional[float]:
        """Mean per-sample normalized entropy of hydra router gates, averaged
        across modules. Returns None when no hydra module has cached a gate
        this step. Thin wrapper over :meth:`get_router_stats` kept for the
        progress-bar postfix path; prefer ``get_router_stats`` for logging.

        Chimera path returns the simple mean of the per-pool entropies
        (each already normalized to [0, 1] by ``log(K_pool)``) so the
        postfix shows a sensible scalar; the legacy single-vector entropy
        would read >1.0 on chimera (concat sums to 2, not 1).
        """
        if getattr(self, "_use_chimera_hydra", False):
            cstats = self.get_chimera_router_stats()
            if not cstats:
                return None
            parts = [cstats[k] for k in ("content_entropy", "freq_entropy") if k in cstats]
            if not parts:
                return None
            return sum(parts) / len(parts)
        stats = self.get_router_stats()
        return stats.get("entropy_mean") if stats else None

    def get_router_stats(
        self,
    ) -> Dict[str, Union[float, List[float], List[List[float]], List[int]]]:
        """Per-step router diagnostics aggregated across hydra modules.

        Returns a dict with:
          - entropy_mean / entropy_p05 / entropy_p50 / entropy_p95: normalized
            per-module entropy (0 = one-hot collapse, 1 = uniform), pooled
            across modules.
          - margin_mean: mean top1-top2 softmax gap, averaged over batch then
            modules. High margin = confident routing; near-zero = effectively
            random.
          - expert_usage: length-E vector of argmax frequency averaged across
            modules. Sums to ~1.0. Flat distribution = balanced; a column
            near 0 means that expert is never picked (collapse).
          - expert_usage_per_bucket: (num_buckets, E) list of argmax frequency
            per σ-bucket (low→high σ), averaged across modules. Empty buckets
            (no batch samples in that σ range this step) report zeros.
          - bucket_counts: per-bucket sample count (length num_buckets). Useful
            sanity for the per-bucket usage row — a bucket with 0 samples this
            step has a meaningless usage row.

          Per-bucket entries omitted when σ wasn't set this step or
          ``num_sigma_buckets <= 1``.

        Empty dict when no hydra module cached a gate this step.

        Vectorized: per-module gates with matching shape are stacked into one
        ``(M, B, E)`` tensor and reduced in a single pass per metric. The
        previous per-module Python loop emitted ~9 small kernels per Hydra
        module (clamp / log / sum / topk(2) / argmax / scatter_add_ / ones_like
        / div), stalling the post-step boundary by ~500 launches at the
        56-module default stack. This implementation issues a constant ~10
        launches regardless of module count (see
        ``docs/optimizations/nsys_analysis_0503.md``).

        Result is memoized on ``self._router_stats_cache`` and invalidated by
        ``clear_step_caches`` so the progress-bar postfix and the logging
        metric share one computation per step.
        """
        if self._router_stats_cache is not None:
            return self._router_stats_cache

        # Collect gates with matching expert count. Modules with mismatched E
        # are skipped (aggregating usage vectors of different length isn't
        # meaningful) — same policy as the previous per-module loop.
        gates: List[torch.Tensor] = []
        E_ref: Optional[int] = None
        for lora in self.unet_loras + self.text_encoder_loras:
            gate = getattr(lora, "_last_gate", None)
            if gate is None:
                continue
            E = gate.shape[-1]
            if E <= 1:
                continue
            if E_ref is None:
                E_ref = E
            elif E != E_ref:
                continue
            gates.append(gate)

        if not gates:
            return {}

        g = torch.stack(gates, dim=0)  # (M, B, E)
        M, B, E = g.shape

        sigma = self._last_sigma  # (B,) or None
        num_buckets = int(self.cfg.num_sigma_buckets)
        want_per_bucket = sigma is not None and num_buckets > 1
        # When ``specialize_experts_by_sigma_buckets`` is on, each sample can
        # only route to its band's ``E / num_buckets`` experts (others masked
        # to -inf pre-softmax). Normalizing entropy by ``log(E)`` then caps
        # the achievable max at ``log(experts_per_band) / log(E)`` (e.g. 0.44
        # for E=12, num_buckets=4) — making "uniform within band" look like
        # collapse. Normalize by the actual reachable support instead.
        band_partition_active = bool(
            self.cfg.specialize_experts_by_sigma_buckets and num_buckets > 1
        )
        effective_E = (E // num_buckets) if band_partition_active else E
        norm = math.log(effective_E) if effective_E > 1 else 1.0

        p = g.float().clamp_min(1e-12)
        # (M,) per-module mean entropy, normalized to [0, 1] over reachable support
        H_per_module = -(p * p.log()).sum(dim=-1).mean(dim=-1) / norm
        # (M, B, 2) top-2 in one batched topk → (M,) mean margin
        top2 = p.topk(2, dim=-1).values
        margin_per_module = (top2[..., 0] - top2[..., 1]).mean(dim=-1)
        # Argmax usage: one_hot + sum → (M, E) histograms in one pass
        expert_idx = g.argmax(dim=-1)  # (M, B)
        usage_per_module = torch.nn.functional.one_hot(expert_idx, num_classes=E).to(
            g.dtype
        ).sum(dim=1) / float(B)  # (M, E)

        H_per_module = H_per_module.detach()
        q_probs = torch.tensor(
            [0.05, 0.5, 0.95], device=H_per_module.device, dtype=H_per_module.dtype
        )
        q = torch.quantile(H_per_module, q_probs)  # (3,)
        # Single packed summary: [mean_H, p05, p50, p95, margin_mean]. One DtoH.
        summary = torch.stack(
            [H_per_module.mean(), q[0], q[1], q[2], margin_per_module.detach().mean()]
        ).cpu()
        usage_mean = usage_per_module.detach().mean(dim=0).cpu().tolist()
        out: Dict[str, Union[float, List[float], List[List[float]], List[int]]] = {
            "entropy_mean": float(summary[0]),
            "entropy_p05": float(summary[1]),
            "entropy_p50": float(summary[2]),
            "entropy_p95": float(summary[3]),
            "margin_mean": float(summary[4]),
            "expert_usage": usage_mean,
        }

        if want_per_bucket and sigma is not None:
            thresholds = torch.linspace(0.0, 1.0, num_buckets + 1, device=sigma.device)[
                1:-1
            ]
            bucket_ids = torch.bucketize(sigma.float(), thresholds).clamp(
                0, num_buckets - 1
            )  # (B,)
            bucket_counts_t = torch.zeros(
                num_buckets, device=sigma.device, dtype=torch.long
            )
            bucket_counts_t.scatter_add_(
                0, bucket_ids, torch.ones_like(bucket_ids, dtype=torch.long)
            )
            # Per-bucket argmax frequency, normalized within each bucket so
            # each row sums to ~1 (or 0 for empty buckets). Flat scatter_add
            # over (M, num_buckets * E) avoids a per-module loop.
            bucket_ids_dev = bucket_ids.to(expert_idx.device)
            flat_idx = bucket_ids_dev[None, :] * E + expert_idx  # (M, B)
            bu = torch.zeros(M, num_buckets * E, device=g.device, dtype=g.dtype)
            bu.scatter_add_(1, flat_idx, torch.ones_like(flat_idx, dtype=g.dtype))
            bu = bu.view(M, num_buckets, E)
            bc = bucket_counts_t.to(g.dtype).clamp_min(1).view(1, num_buckets, 1)
            bucket_usage_mean = (bu / bc).detach().mean(dim=0).cpu().tolist()
            out["expert_usage_per_bucket"] = bucket_usage_mean
            out["bucket_counts"] = bucket_counts_t.cpu().tolist()

        self._router_stats_cache = out
        return out

    def get_chimera_router_stats(
        self,
    ) -> Dict[str, Union[float, List[float]]]:
        """Per-pool router diagnostics for the chimera dual-pool routing.

        Chimera's ``_last_gate`` is ``cat([π_c, π_f])`` of shape
        ``(B, K_c + K_f)`` — two independent softmaxes glued together, so the
        whole vector sums to 2 and ``argmax`` across the concat is doubly
        misleading: it (a) only ever names one slot per sample so the
        per-pool view collapses to a single histogram summing to 1, and
        (b) biases toward whichever pool happens to have higher initialization
        variance (FreqRouter's ``init_std=0.1`` vs the content router's 0.01).
        This routine reports each pool independently, using **mean gates**
        (same approach as ``fera/expert_usage`` — see
        ``[[project_fera_expert_usage_mean_gates]]``):

          * Content pool: aggregate ``π_c = gate[..., :K_c]`` across every
            chimera module that cached a gate this step. Mean over modules
            then over batch gives the per-content-expert usage vector
            ``(K_c,)``, which sums to ~1.
          * Freq pool: the FreqRouter broadcasts the same ``π_f`` to every
            module, so we read it once from ``freq_router._last_gates``
            (shape ``(B, K_f)``) — no module aggregation needed. Mean over
            batch gives the per-freq-expert usage vector ``(K_f,)``.

        Returned entropy / margin are normalized per pool: entropy by
        ``log(K_pool)`` (so [0, 1] is the right range — the legacy
        ``hydra/router_entropy`` would read >1.0 on chimera because it
        divided the sum-to-2 vector's entropy by ``log(E)``, not
        ``log(2·K_pool)``).

        Empty dict on non-chimera networks or when no gate has been cached
        this step.
        """
        if not getattr(self, "_use_chimera_hydra", False):
            return {}
        if self._chimera_router_stats_cache is not None:
            return self._chimera_router_stats_cache

        out: Dict[str, Union[float, List[float]]] = {}
        K_c_default = int(getattr(self.cfg, "num_experts_content", 0))

        # --- Content pool: aggregate π_c across modules ----------------------
        pi_c_list: List[torch.Tensor] = []
        K_c_ref: Optional[int] = None
        for lora in self.unet_loras + self.text_encoder_loras:
            gate = getattr(lora, "_last_gate", None)
            if gate is None:
                continue
            K_c = int(getattr(lora, "num_experts_content", K_c_default))
            if K_c <= 0:
                continue
            if K_c_ref is None:
                K_c_ref = K_c
            elif K_c != K_c_ref:
                continue
            pi_c_list.append(gate[..., :K_c])

        if pi_c_list and K_c_ref is not None and K_c_ref > 1:
            pi_c = torch.stack(pi_c_list, dim=0).float().clamp_min(1e-12)  # (M, B, K_c)
            norm_c = math.log(K_c_ref)
            H_c_per_mod = -(pi_c * pi_c.log()).sum(dim=-1).mean(dim=-1) / norm_c
            top2_c = pi_c.topk(2, dim=-1).values
            margin_c_per_mod = (top2_c[..., 0] - top2_c[..., 1]).mean(dim=-1)
            usage_c = pi_c.mean(dim=(0, 1))  # (K_c,)
            summary_c = torch.stack(
                [H_c_per_mod.mean().detach(), margin_c_per_mod.mean().detach()]
            ).cpu()
            out["content_entropy"] = float(summary_c[0])
            out["content_margin"] = float(summary_c[1])
            out["content_usage"] = usage_c.detach().cpu().tolist()

        # --- Freq pool: single broadcast tensor from FreqRouter --------------
        fr = getattr(self, "freq_router", None)
        pi_f = fr._last_gates if fr is not None else None
        if pi_f is not None and pi_f.dim() == 2 and pi_f.shape[-1] > 1:
            K_f = int(pi_f.shape[-1])
            pf = pi_f.float().clamp_min(1e-12)
            norm_f = math.log(K_f)
            H_f = (-(pf * pf.log()).sum(dim=-1).mean() / norm_f).detach()
            top2_f = pf.topk(2, dim=-1).values
            margin_f = (top2_f[..., 0] - top2_f[..., 1]).mean().detach()
            usage_f = pf.mean(dim=0).detach()
            summary_f = torch.stack([H_f, margin_f]).cpu()
            out["freq_entropy"] = float(summary_f[0])
            out["freq_margin"] = float(summary_f[1])
            out["freq_usage"] = usage_f.cpu().tolist()

        self._chimera_router_stats_cache = out
        return out

    def capture_up_grad_stats(self) -> None:
        """Snapshot per-expert grad-norm on Hydra up-weights.

        Diagnoses the T-LoRA × σ-bucket interaction: under the σ-band
        partition, a high-σ-band expert only fires at high σ, where T-LoRA
        clamps the rank to ``min_rank``. Rank columns ``[min_rank, R)`` of
        that expert's ``lora_up`` should then accumulate near-zero gradient
        — those columns are dead capacity. Reading ``lora_up_weight.grad``
        and splitting the L2 norm at the ``min_rank`` boundary makes the
        effect directly visible.

        Must be called between ``accelerator.backward(loss)`` and
        ``optimizer.zero_grad`` — once ``zero_grad(set_to_none=True)`` has
        run, ``.grad`` is ``None``.

        Stash format (read by ``library/training/metrics.py``):
          ``below`` : (E,) Σ_modules ‖grad[e, :, :min_rank]‖²  (only when T-LoRA active)
          ``above`` : (E,) Σ_modules ‖grad[e, :, min_rank:]‖²  (only when T-LoRA active)
          ``total`` : (E,) Σ_modules ‖grad[e, ...]‖²
          ``sp_total`` : (E,) Σ_modules ‖S_p.grad[e]‖²  (OrthoHydra)
          ``below_band`` / ``above_band`` / ``total_band`` / ``sp_total_band``
            (B,) per-σ-band sums (sum over experts assigned to band b).
            Only present when ``specialize_experts_by_sigma_buckets`` is on.
          ``min_rank`` : float, snapshot of ``cfg.min_rank`` for context.
          ``num_buckets`` : float, snapshot of ``cfg.num_sigma_buckets``.

        Square-norms (sum-of-squares) are reported, not L2 norms — the
        metric layer takes ``sqrt`` after summation. This keeps aggregation
        across modules correct (concatenation-of-grads norm = sqrt of
        sum-of-squares per chunk).
        """
        if not getattr(self, "_use_hydra", False):
            self._last_up_grad_stats = {}
            return

        use_tlora = bool(self.cfg.use_timestep_mask)
        min_rank = int(self.cfg.min_rank) if use_tlora else 0
        max_rank = int(self.cfg.lora_dim)
        # Clamp min_rank: a misconfig like min_rank > lora_dim would make the
        # "above" slice empty and the "below" slice full-rank, silently turning
        # the diagnostic into a no-op. Pin to [0, R].
        min_rank = max(0, min(min_rank, max_rank))
        has_tlora_split = use_tlora and 0 < min_rank < max_rank

        # Collect grads first; reduce in a few fused passes at the end. The
        # naive per-module loop launched ~4–7 tiny kernels per Hydra module
        # and stalled the post-backward / pre-optimizer boundary by hundreds
        # of ms on log steps (see docs/optimizations/nsys_analysis_0503.md).
        up_grads: List[torch.Tensor] = []  # each (E, out_i, R)
        sp_grads: List[torch.Tensor] = []  # each (E, r, r)
        expert_band_ref: Optional[torch.Tensor] = None

        for lora in self.unet_loras + self.text_encoder_loras:
            up = getattr(lora, "lora_up_weight", None)
            sp = getattr(lora, "S_p", None)
            up_grad = up.grad if isinstance(up, torch.nn.Parameter) else None
            sp_grad = sp.grad if isinstance(sp, torch.nn.Parameter) else None
            if up_grad is not None:
                up_grads.append(up_grad.detach())
            if sp_grad is not None and sp_grad.dim() == 3:
                # (E, r, r) — OrthoHydra rotation generator. No clean rank-col
                # split (Cayley couples all entries), so we report total only.
                # Plain OrthoLoRA's S_p is (r, r) with no expert axis — skipped
                # by the dim==3 check, since this diagnostic is per-expert.
                sp_grads.append(sp_grad.detach())
            if expert_band_ref is None:
                band = getattr(lora, "_expert_band", None)
                if band is not None:
                    expert_band_ref = band.detach()

        if not up_grads and not sp_grads:
            self._last_up_grad_stats = {}
            return

        total_per_exp: Optional[torch.Tensor] = None
        below_per_exp: Optional[torch.Tensor] = None
        above_per_exp: Optional[torch.Tensor] = None
        sp_total_per_exp: Optional[torch.Tensor] = None
        device_ref: Optional[torch.device] = None

        if up_grads:
            # All entries share E and R; only out_i varies. Cat along the out
            # axis into one (E, sum_out, R) tensor and reduce in one pass.
            big_up = torch.cat(up_grads, dim=1).float()
            sq_up = big_up.square()
            total_per_exp = sq_up.sum(dim=(1, 2))
            device_ref = total_per_exp.device
            if has_tlora_split:
                # Slices are views into ``sq_up``; sum along (out, rank-slice).
                below_per_exp = sq_up[:, :, :min_rank].sum(dim=(1, 2))
                above_per_exp = sq_up[:, :, min_rank:].sum(dim=(1, 2))

        if sp_grads:
            # All entries share (E, r, r). Stack into (M, E, r, r) and reduce
            # over modules + r×r in one pass.
            big_sp = torch.stack(sp_grads, dim=0).float()
            sp_total_per_exp = big_sp.square().sum(dim=(0, 2, 3))
            if device_ref is None:
                device_ref = sp_total_per_exp.device

        # Stash on-device tensors only — the D2H happens in
        # ``get_up_grad_stats`` so non-log steps avoid the
        # ``cudaStreamSynchronize`` that .cpu().tolist() forces.
        out: Dict[str, object] = {
            "min_rank": [float(min_rank)],
            "num_buckets": [float(self.cfg.num_sigma_buckets)],
        }
        if total_per_exp is not None:
            out["total"] = total_per_exp
        if below_per_exp is not None and above_per_exp is not None:
            out["below"] = below_per_exp
            out["above"] = above_per_exp
        if sp_total_per_exp is not None:
            out["sp_total"] = sp_total_per_exp

        # Per-band aggregation: scatter the per-expert sum-of-squares along
        # _expert_band. Only meaningful when σ-bucket partition is active —
        # otherwise the band assignment is undefined and per-band rows would
        # be misleading.
        if (
            expert_band_ref is not None
            and bool(self.cfg.specialize_experts_by_sigma_buckets)
            and int(self.cfg.num_sigma_buckets) > 1
        ):
            B = int(self.cfg.num_sigma_buckets)
            band = expert_band_ref.to(device_ref)

            def _scatter_to_band(per_exp: torch.Tensor) -> torch.Tensor:
                buf = torch.zeros(B, device=per_exp.device, dtype=per_exp.dtype)
                buf.scatter_add_(0, band, per_exp)
                return buf

            if total_per_exp is not None:
                out["total_band"] = _scatter_to_band(total_per_exp)
            if below_per_exp is not None and above_per_exp is not None:
                out["below_band"] = _scatter_to_band(below_per_exp)
                out["above_band"] = _scatter_to_band(above_per_exp)
            if sp_total_per_exp is not None:
                out["sp_total_band"] = _scatter_to_band(sp_total_per_exp)

        self._last_up_grad_stats = out

    def get_up_grad_stats(self) -> Dict[str, List[float]]:
        """Materialize the on-device stash from ``capture_up_grad_stats``.

        D2H is deferred to here so non-log steps don't pay the sync — the
        capture must run between backward and zero_grad (when ``.grad`` is
        live), but the metric only consumes the result on log steps.
        """
        raw = self._last_up_grad_stats
        if not raw:
            return {}
        materialized: Dict[str, List[float]] = {}
        for k, v in raw.items():
            if torch.is_tensor(v):
                materialized[k] = v.detach().cpu().tolist()
            else:
                materialized[k] = list(v)  # type: ignore[arg-type]
        return materialized

    def get_ortho_regularization(self) -> torch.Tensor:
        """Sum orthogonality regularization from all OrthoLoRA and ReFT modules."""
        total_reg = torch.tensor(0.0, device=next(self.parameters()).device)
        count = 0
        for lora in self.text_encoder_loras + self.unet_loras:
            if hasattr(lora, "regularization"):
                p_reg, q_reg = lora.regularization()
                total_reg = total_reg + p_reg + q_reg
                count += 1
        for reft in self.text_encoder_refts + self.unet_refts:
            total_reg = total_reg + reft.regularization()
            count += 1
        return total_reg / max(count, 1)

    def metrics(self, ctx: MetricContext) -> dict[str, float]:
        """Emit log-step keys owned by the LoRA network.

        Covers ortho regularization, hydra balance loss, router stats, and
        hydra up-weight grad-norm diagnostics. Each block returns nothing
        if its driver is off (``_ortho_reg_weight == 0``, ``_use_hydra ==
        False``, etc.) so the cost on inactive paths is one attr check.
        """
        out: dict[str, float] = {}

        # Ortho regularization magnitude.
        ortho_w = float(getattr(self, "_ortho_reg_weight", 0.0) or 0.0)
        if ortho_w > 0.0:
            v = self.get_ortho_regularization()
            if torch.is_tensor(v):
                v = v.detach().item()
            out["reg/ortho"] = float(v)
            out["reg/ortho_weighted"] = float(ortho_w * v)

        # Hydra balance loss magnitude.
        bal_w = float(getattr(self, "_balance_loss_weight", 0.0) or 0.0)
        if bal_w > 0.0:
            v = self.get_balance_loss()
            if torch.is_tensor(v):
                v = v.detach().item()
            out["reg/balance"] = float(v)
            out["reg/balance_weighted"] = float(bal_w * v)

        if not getattr(self, "_use_hydra", False):
            return out

        # Router diagnostics. Chimera takes a different path because its
        # ``_last_gate`` is a concat of two independent softmaxes, so the
        # argmax-histogram aggregation under ``hydra/*`` is doubly misleading
        # (sums to 1 instead of 2; biased toward whichever pool has higher
        # init variance — see ``get_chimera_router_stats`` docstring).
        if getattr(self, "_use_chimera_hydra", False):
            cstats = self.get_chimera_router_stats()
            if cstats:
                if "content_entropy" in cstats:
                    out["chimera/content_entropy"] = float(cstats["content_entropy"])
                    out["chimera/content_margin"] = float(cstats["content_margin"])
                    for i, v in enumerate(cstats.get("content_usage", [])):
                        out[f"chimera/content_usage/{i}"] = float(v)
                if "freq_entropy" in cstats:
                    out["chimera/freq_entropy"] = float(cstats["freq_entropy"])
                    out["chimera/freq_margin"] = float(cstats["freq_margin"])
                    for i, v in enumerate(cstats.get("freq_usage", [])):
                        out[f"chimera/freq_usage/{i}"] = float(v)
        else:
            stats = self.get_router_stats()
            if stats:
                out["hydra/router_entropy"] = float(stats["entropy_mean"])
                out["hydra/router_entropy_p05"] = float(stats["entropy_p05"])
                out["hydra/router_entropy_p50"] = float(stats["entropy_p50"])
                out["hydra/router_entropy_p95"] = float(stats["entropy_p95"])
                out["hydra/router_margin"] = float(stats["margin_mean"])
                for i, v in enumerate(stats.get("expert_usage", [])):
                    out[f"hydra/expert_usage/{i}"] = float(v)
                for b, row in enumerate(stats.get("expert_usage_per_bucket", [])):
                    for i, v in enumerate(row):
                        out[f"hydra/expert_usage_b{b}/{i}"] = float(v)
                for b, c in enumerate(stats.get("bucket_counts", [])):
                    out[f"hydra/bucket_count/{b}"] = float(c)

        # Hydra up-weight grad norms by rank region and σ-band.
        up = self.get_up_grad_stats()
        if up:
            eps = 1e-12

            def _emit_per_expert(prefix: str, sq: list[float]) -> None:
                for i, v in enumerate(sq):
                    out[f"hydra/up_grad/{prefix}/exp{i}"] = float(v) ** 0.5

            def _emit_per_band(prefix: str, sq: list[float]) -> None:
                for b, v in enumerate(sq):
                    out[f"hydra/up_grad/{prefix}/band{b}"] = float(v) ** 0.5

            if "total" in up:
                _emit_per_expert("total", up["total"])
            if "below" in up and "above" in up:
                _emit_per_expert("below", up["below"])
                _emit_per_expert("above", up["above"])
                for i, (b_, a_) in enumerate(zip(up["below"], up["above"])):
                    out[f"hydra/up_grad/above_below_ratio/exp{i}"] = float(
                        a_
                    ) ** 0.5 / (float(b_) ** 0.5 + eps)
            if "sp_total" in up:
                _emit_per_expert("sp_total", up["sp_total"])
            if "total_band" in up:
                _emit_per_band("total", up["total_band"])
            if "below_band" in up and "above_band" in up:
                _emit_per_band("below", up["below_band"])
                _emit_per_band("above", up["above_band"])
                for b, (bv, av) in enumerate(zip(up["below_band"], up["above_band"])):
                    out[f"hydra/up_grad/above_below_ratio/band{b}"] = float(
                        av
                    ) ** 0.5 / (float(bv) ** 0.5 + eps)
            if "sp_total_band" in up:
                _emit_per_band("sp_total", up["sp_total_band"])

        # GlobalRouter stats — for stacked-experts + per-network routing
        # (plan2 §three-axis-config). Mirrors the per-Linear hydra keys but
        # under the ``fera/`` namespace so dashboards can compare across
        # variants. ``_last_gates`` is populated by ``GlobalRouter.forward``;
        # absent (None) outside of a step that fired the router.
        if (
            self.global_router is not None
            and self.global_router._last_gates is not None
        ):
            gates = self.global_router._last_gates  # (B, E) detached
            if gates.dim() == 2 and gates.shape[1] > 1:
                g = gates.float().clamp_min(1e-12)
                E = int(g.shape[-1])
                # Per-batch normalized entropy.
                norm = math.log(E)
                H = -(g * g.log()).sum(dim=-1).mean() / norm
                # Top1-Top2 margin (confidence).
                top2 = g.topk(2, dim=-1).values
                margin = (top2[..., 0] - top2[..., 1]).mean()
                # Per-expert mean gate weight. argmax-histogram breaks exact
                # ties to index 0 and misreports a uniform router as
                # "100% expert 0"; mean(gates) reflects the actual soft
                # distribution and still sums to 1.
                usage = g.mean(dim=0)
                summary = torch.stack([H.detach(), margin.detach()]).cpu()
                out["fera/router_entropy"] = float(summary[0])
                out["fera/router_margin"] = float(summary[1])
                for i, v in enumerate(usage.detach().cpu().tolist()):
                    out[f"fera/expert_usage/{i}"] = float(v)

        return out

    @staticmethod
    def _strip_orig_mod_keys(state_dict):
        """Strip torch.compile '_orig_mod_' from state_dict keys for compat with old checkpoints."""
        new_sd = {}
        for key, val in state_dict.items():
            new_key = re.sub(r"(?<=_)_orig_mod_", "", key)
            new_sd[new_key] = val
        return new_sd

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        state_dict = self._strip_orig_mod_keys(state_dict)
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        for key in list(weights_sd.keys()):
            if key.endswith(".dora_scale"):
                new_key = key.replace(".dora_scale", ".magnitude")
                weights_sd[new_key] = weights_sd.pop(key)

        # Stack per-expert hydra ups into fused lora_up_weight (training form).
        # Also stacks per-expert ``.lora_downs.{i}.weight`` for the
        # StackedExperts (independent-A) layout — no-op for Hydra.
        weights_sd = _stack_lora_ups(weights_sd)
        # Refuse split stacked-experts first (its discriminator is per-expert
        # ``lora_down_weight`` 3-D, which the hydra refuser would otherwise
        # short-circuit on the absent shared ``lora_down.weight``).
        weights_sd = _refuse_split_stacked_experts_keys(weights_sd)
        # Refuse split hydra attn keys BEFORE the regular refuser: hydra splits
        # carry no lora_up.weight, so the regular path would skip them anyway,
        # but running hydra first means any non-hydra attention still goes
        # through the normal code path cleanly.
        weights_sd = _refuse_split_hydra_keys(weights_sd)
        # Refuse unfused attn projections (inverse of save_weights defusing).
        weights_sd = _refuse_unfused_attn_lora_keys(weights_sd)

        info = self.load_state_dict(weights_sd, False)
        return info

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info(
                f"enable LoRA for text encoder: {len(self.text_encoder_loras)} modules"
            )
        else:
            self.text_encoder_loras = []
            self.text_encoder_refts = []

        if apply_unet:
            logger.info(f"enable LoRA for DiT: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []
            self.unet_refts = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        # ReFT wraps each selected DiT Block's forward, so the chain is:
        #   Block.__call__ -> ReFT.forward -> original Block.forward
        #   (inside which LoRA-wrapped Linears still fire normally).
        for reft in self.text_encoder_refts + self.unet_refts:
            reft.apply_to()
            self.add_module(reft.lora_name, reft)

    def is_mergeable(self):
        return True

    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(LoRANetwork.LORA_PREFIX_ANIMA):
                apply_unet = True

        if apply_text_encoder:
            logger.info("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info("enable LoRA for DiT")
        else:
            self.unet_loras = []

        # Pre-group checkpoint keys by LoRA module prefix (avoid O(modules * keys) scan)
        # Keys are "{module_name}.{param}" where module_name has no dots (dots → underscores)
        grouped_sd: dict[str, dict[str, torch.Tensor]] = {}
        for key, value in weights_sd.items():
            prefix, dot, suffix = key.partition(".")
            if not dot:
                continue
            if prefix not in grouped_sd:
                grouped_sd[prefix] = {}
            grouped_sd[prefix][suffix] = value

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = grouped_sd.get(lora.lora_name, {})
            if sd_for_lora:
                lora.merge_to(sd_for_lora, dtype, device)

        logger.info("weights are merged")

    def set_loraplus_lr_ratio(
        self, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio
    ):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.loraplus_unet_lr_ratio = loraplus_unet_lr_ratio
        self.loraplus_text_encoder_lr_ratio = loraplus_text_encoder_lr_ratio

        logger.info(
            f"LoRA+ UNet LR Ratio: {self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio}"
        )
        logger.info(
            f"LoRA+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}"
        )

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        if text_encoder_lr is None or (
            isinstance(text_encoder_lr, list) and len(text_encoder_lr) == 0
        ):
            text_encoder_lr = [default_lr]
        elif isinstance(text_encoder_lr, float) or isinstance(text_encoder_lr, int):
            text_encoder_lr = [float(text_encoder_lr)]
        elif len(text_encoder_lr) == 1:
            pass  # already a list with one element

        self.requires_grad_(True)

        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, loraplus_ratio):
            param_groups = {"lora": {}, "plus": {}, "router": {}}
            reg_groups = {}
            reg_lrs_list = (
                list(self.cfg.reg_lrs.items()) if self.cfg.reg_lrs is not None else []
            )
            router_scale = float(self.cfg.router_lr_scale)
            # Chimera content-router multiplier (stacks on router_scale). The
            # per-Linear ``router.*`` group below collects chimera's content
            # router params (chimera modules own the only per-Linear
            # ``router.*`` in their network). Off-by-default for non-chimera
            # runs so plain Hydra is unaffected.
            content_router_scale = (
                float(self.cfg.content_router_lr_scale)
                if getattr(self.cfg, "use_chimera_hydra", False)
                else 1.0
            )
            router_lr_mult = router_scale * content_router_scale

            def _is_router_param(pname: str) -> bool:
                # named_parameters() yields top-level names like "router.weight"
                # — no leading dot. σ features live inside router.weight now
                # (columns [lora_dim:] of the weight), so there's a single path.
                return pname.startswith("router.")

            for lora in loras:
                matched_reg_lr = None
                for i, (regex_str, reg_lr) in enumerate(reg_lrs_list):
                    if re.fullmatch(regex_str, lora.original_name):
                        matched_reg_lr = (i, reg_lr)
                        logger.info(
                            f"Module {lora.original_name} matched regex '{regex_str}' -> LR {reg_lr}"
                        )
                        break

                for name, param in lora.named_parameters():
                    is_router = _is_router_param(name)
                    if matched_reg_lr is not None:
                        reg_idx, reg_lr = matched_reg_lr
                        group_key = f"reg_lr_{reg_idx}"
                        if group_key not in reg_groups:
                            reg_groups[group_key] = {
                                "lora": {},
                                "plus": {},
                                "router": {},
                                "lr": reg_lr,
                            }
                        if is_router:
                            reg_groups[group_key]["router"][
                                f"{lora.lora_name}.{name}"
                            ] = param
                        elif loraplus_ratio is not None and (
                            "lora_up" in name
                            or "p_layer" in name
                            or "learned_source" in name
                        ):
                            reg_groups[group_key]["plus"][
                                f"{lora.lora_name}.{name}"
                            ] = param
                        else:
                            reg_groups[group_key]["lora"][
                                f"{lora.lora_name}.{name}"
                            ] = param
                        continue

                    if is_router:
                        param_groups["router"][f"{lora.lora_name}.{name}"] = param
                    elif loraplus_ratio is not None and (
                        "lora_up" in name
                        or "p_layer" in name
                        or "learned_source" in name
                    ):
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            params = []
            descriptions = []
            for group_key, group in reg_groups.items():
                reg_lr = group["lr"]
                for key in ("lora", "plus", "router"):
                    param_data = {"params": group[key].values()}
                    if len(param_data["params"]) == 0:
                        continue
                    if key == "plus":
                        param_data["lr"] = (
                            reg_lr * loraplus_ratio
                            if loraplus_ratio is not None
                            else reg_lr
                        )
                    elif key == "router":
                        param_data["lr"] = reg_lr * router_lr_mult
                    else:
                        param_data["lr"] = reg_lr
                    if (
                        param_data.get("lr", None) == 0
                        or param_data.get("lr", None) is None
                    ):
                        logger.info("NO LR skipping!")
                        continue
                    params.append(param_data)
                    desc = f"reg_lr_{group_key.split('_')[-1]}"
                    descriptions.append(
                        desc
                        + (
                            " plus"
                            if key == "plus"
                            else (" router" if key == "router" else "")
                        )
                    )

            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}
                if len(param_data["params"]) == 0:
                    continue
                if lr is not None:
                    if key == "plus":
                        param_data["lr"] = lr * loraplus_ratio
                    elif key == "router":
                        param_data["lr"] = lr * router_lr_mult
                    else:
                        param_data["lr"] = lr
                if (
                    param_data.get("lr", None) == 0
                    or param_data.get("lr", None) is None
                ):
                    logger.info("NO LR skipping!")
                    continue
                params.append(param_data)
                descriptions.append(
                    "plus" if key == "plus" else ("router" if key == "router" else "")
                )
            return params, descriptions

        if self.text_encoder_loras:
            loraplus_ratio = (
                self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio
            )
            te1_loras = [
                lora
                for lora in self.text_encoder_loras
                if lora.lora_name.startswith(self.LORA_PREFIX_TEXT_ENCODER)
            ]
            if len(te1_loras) > 0:
                logger.info(
                    f"Text Encoder 1 (Qwen3): {len(te1_loras)} modules, LR {text_encoder_lr[0]}"
                )
                params, descriptions = assemble_params(
                    te1_loras, text_encoder_lr[0], loraplus_ratio
                )
                all_params.extend(params)
                lr_descriptions.extend(
                    ["textencoder 1" + (" " + d if d else "") for d in descriptions]
                )

        if self.unet_loras:
            params, descriptions = assemble_params(
                self.unet_loras,
                unet_lr if unet_lr is not None else default_lr,
                self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio,
            )
            all_params.extend(params)
            lr_descriptions.extend(
                ["unet" + (" " + d if d else "") for d in descriptions]
            )

        if self.text_encoder_refts:
            params, descriptions = assemble_params(
                self.text_encoder_refts,
                text_encoder_lr[0],
                self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio,
            )
            all_params.extend(params)
            lr_descriptions.extend(
                ["reft textencoder" + (" " + d if d else "") for d in descriptions]
            )

        if self.unet_refts:
            params, descriptions = assemble_params(
                self.unet_refts,
                unet_lr if unet_lr is not None else default_lr,
                self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio,
            )
            all_params.extend(params)
            lr_descriptions.extend(
                ["reft unet" + (" " + d if d else "") for d in descriptions]
            )

        # HydraLoRA per-module routers are submodules of HydraLoRAModule instances,
        # so they are already captured by the unet_loras param group above.

        if getattr(self, "repa_head", None) is not None:
            repa_params = list(self.repa_head.parameters())
            if len(repa_params) > 0:
                repa_lr_scale = float(getattr(self, "_repa_lr_scale", 1.0))
                base_lr = unet_lr if unet_lr is not None else default_lr
                if base_lr is None or base_lr == 0:
                    logger.info("REPA head: no base LR, skipping param group")
                else:
                    repa_lr = float(base_lr) * repa_lr_scale
                    all_params.append({"params": repa_params, "lr": repa_lr})
                    lr_descriptions.append("repa head")
                    logger.info(
                        f"REPA head param group: lr={repa_lr:.2e} "
                        f"({repa_lr_scale}x of unet_lr={base_lr})"
                    )

        # GlobalRouter (route_per_layer=False) lives on the network, not on
        # per-Linear LoRA modules, so the assemble_params loop above misses it.
        # Add it explicitly with the same router_lr_scale convention used for
        # per-Linear routers (unet_lr × router_lr_scale).
        if getattr(self, "global_router", None) is not None:
            gr_params = list(self.global_router.parameters())
            if len(gr_params) > 0:
                router_scale = float(self.cfg.router_lr_scale)
                base_lr = unet_lr if unet_lr is not None else default_lr
                if base_lr is None or base_lr == 0:
                    logger.info("GlobalRouter: no base LR, skipping param group")
                else:
                    gr_lr = float(base_lr) * router_scale
                    all_params.append({"params": gr_params, "lr": gr_lr})
                    lr_descriptions.append("global router")
                    logger.info(
                        f"GlobalRouter param group: lr={gr_lr:.2e} "
                        f"({router_scale}x of unet_lr={base_lr})"
                    )

        # ChimeraHydra FreqRouter mirrors the GlobalRouter param-group
        # treatment. Same router_lr_scale convention so a single knob tunes
        # both router families.
        if getattr(self, "freq_router", None) is not None:
            fr_params = list(self.freq_router.parameters())
            if len(fr_params) > 0:
                router_scale = float(self.cfg.router_lr_scale)
                freq_scale = float(self.cfg.freq_router_lr_scale)
                base_lr = unet_lr if unet_lr is not None else default_lr
                if base_lr is None or base_lr == 0:
                    logger.info("FreqRouter: no base LR, skipping param group")
                else:
                    fr_lr = float(base_lr) * router_scale * freq_scale
                    all_params.append({"params": fr_params, "lr": fr_lr})
                    lr_descriptions.append("chimera freq router")
                    logger.info(
                        f"ChimeraHydra FreqRouter param group: lr={fr_lr:.2e} "
                        f"({router_scale}x router_lr_scale × {freq_scale}x "
                        f"freq_router_lr_scale of unet_lr={base_lr})"
                    )

        return all_params, lr_descriptions

    def enable_gradient_checkpointing(self):
        pass  # not supported

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        spec: NetworkSpec = getattr(self, "_network_spec", NETWORK_REGISTRY["lora"])
        if metadata is None:
            metadata = {}
        if metadata:
            metadata["ss_network_spec"] = spec.name

        # Hard σ-band partition lives in non-persistent buffers (`_expert_band`)
        # and a Python attr (`_sigma_band_partition`); nothing of it survives
        # the state_dict write. Emit the two scalars needed to re-register the
        # partition at load time so inference (`make test`) and the ComfyUI
        # node can reconstruct the per-sample band mask. Only stamped when the
        # partition is on, so older non-band checkpoints stay byte-identical.
        if self.cfg.specialize_experts_by_sigma_buckets:
            metadata["ss_specialize_experts_by_sigma_buckets"] = "true"
            metadata["ss_num_sigma_buckets"] = str(int(self.cfg.num_sigma_buckets))
            if self.cfg.sigma_bucket_boundaries is not None:
                import json as _json

                metadata["ss_sigma_bucket_boundaries"] = _json.dumps(
                    list(self.cfg.sigma_bucket_boundaries)
                )

        # Three-axis routing config (plan2 §three-axis-config). Stamped on
        # every save so the loader can reconstruct the exact router layout
        # without key-sniffing — particularly important for distinguishing
        # ``stacked_experts_global_fei`` (independent-A) from ``hydra``
        # (shared-A) at a glance.
        if self.cfg.use_moe_style is not False:
            metadata["ss_use_moe_style"] = str(self.cfg.use_moe_style)
            metadata["ss_route_per_layer"] = (
                "true" if self.cfg.route_per_layer else "false"
            )
            metadata["ss_router_source"] = str(self.cfg.router_source)

        # FEI router params (router-source-specific scalars the loader needs
        # to size the router input). Stamped for both per-Linear and global
        # FEI routers.
        if self.cfg.router_source == "fei" and self.cfg.fei_feature_dim > 0:
            metadata["ss_fei_feature_dim"] = str(int(self.cfg.fei_feature_dim))
            metadata["ss_fei_sigma_low_div"] = str(float(self.cfg.fei_sigma_low_div))

        # ChimeraHydra: the pool split is the only non-key info the loader
        # cannot reconstruct from state_dict (P_bases shape encodes E = K_c +
        # K_f but not the split point). FreqRouter weights survive as plain
        # ``freq_router.*`` keys without dedicated handling. FEI/σ feature
        # dims are also stamped so the loader can re-size the freq router
        # input — they live outside the standard ``router_source`` flow
        # (chimera uses BOTH simultaneously).
        if self.cfg.use_chimera_hydra:
            metadata["ss_use_chimera_hydra"] = "true"
            metadata["ss_num_experts_content"] = str(int(self.cfg.num_experts_content))
            metadata["ss_num_experts_freq"] = str(int(self.cfg.num_experts_freq))
            metadata["ss_chimera_fei_feature_dim"] = str(int(self.cfg.fei_feature_dim))
            metadata["ss_chimera_sigma_feature_dim"] = str(
                int(self.cfg.sigma_feature_dim)
            )
            metadata["ss_chimera_fei_sigma_low_div"] = str(
                float(self.cfg.fei_sigma_low_div)
            )
            # FreqRouter input LN flag. Parameterless LN leaves no tensor
            # footprint in the state_dict, so the loader can't sniff it from
            # weights — has to come from metadata. Default-off on rebuild
            # when absent preserves pre-LN checkpoint inference.
            metadata["ss_chimera_freq_router_layer_norm"] = (
                "true" if self.cfg.freq_router_layer_norm else "false"
            )

        state_dict = self.state_dict()
        # Drop training-only auxiliary heads from the on-disk artifact. ``repa_head``
        # is a 3-layer MLP used only for the REPA alignment loss during training; it
        # carries no ``lora_unet_*`` prefix, isn't consumed at inference or merge,
        # and adds ~21MB of dead weight to the file. Resume-from-checkpoint will
        # re-init the head from random — re-converges in a few hundred steps.
        for key in [k for k in state_dict if k.startswith("repa_head.")]:
            del state_dict[key]
        lora_save.save_network_weights(
            state_dict,
            file=file,
            dtype=dtype,
            metadata=metadata,
            save_variant=spec.save_variant,
        )

    def backup_weights(self):
        loras: List[LoRAModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                org_module._lora_org_weight = org_module.weight.detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        loras: List[LoRAModule] = self.text_encoder_loras + self.unet_loras
        with torch.no_grad():
            for lora in loras:
                org_module = lora.org_module_ref[0]
                if not org_module._lora_restored:
                    org_module.weight.data.copy_(org_module._lora_org_weight)
                    org_module._lora_restored = True

    def pre_calculation(self):
        loras: List[LoRAModule] = self.text_encoder_loras + self.unet_loras
        with torch.no_grad():
            for lora in loras:
                org_module = lora.org_module_ref[0]
                lora_weight = lora.get_weight().to(
                    org_module.weight.device, dtype=org_module.weight.dtype
                )
                org_module.weight.data.add_(lora_weight)

                org_module._lora_restored = False
                lora.enabled = False

    def apply_max_norm_regularization(self, max_norm_value, device):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()
        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (
                    (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2))
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(
                    down.permute(1, 0, 2, 3), up
                ).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)
