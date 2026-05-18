# networks/

Pluggable adapter implementations selected at runtime via the `network_module` config key (plus, for the LoRA family, the three-axis routing cfg). Each subdirectory is a self-contained adapter family; `attention_dispatch.py` is the shared backend router used by both training and inference.

## Layout

| Path | Role |
|------|------|
| `lora_anima/` | LoRA network creation, module targeting, timestep-masking orchestration, global routing. Split into `network.py`, `factory.py`, `loading.py`, and `config.py`. |
| `lora_modules/` | Per-variant module implementations: `lora.py`, `ortho.py`, `hydra.py`, `reft.py`, `stacked_experts.py`, `chimera.py`, plus `base.py` and `custom_autograd.py`. Each module class owns its own save-pipeline hook (`distill_save_state_dict` / `build_moe_state_dict`) — the Cayley/SVD math and per-pool MoE layout live next to the variant that defined them. |
| `attn_fuse.py` | `AttnFuseSpec` + `iter_split_groups` + `match_fused_spec` — single source of truth for the runtime-fused `qkv_proj`/`kv_proj` ↔ on-disk split `q/k/v_proj` layout. Sits at the `networks/` top level so save (`lora_save.py`) and load (`lora_anima/loading.py`) both reach it without a cross-package import. |
| `lora_save.py`, `lora_utils.py` | Thin save-pipeline orchestrator + shared helpers. `lora_save.save_network_weights` calls each variant's `distill_save_state_dict` in fixed order, then dispatches to the matching `build_moe_state_dict`. Owns only the legacy sig-type OrthoLoRA distill (no live module class for it) and the variant-write sibling-file naming. |
| `methods/postfix.py` | Continuous postfix tuning: learns N vectors appended to adapter cross-attention (modes: hidden, embedding, cfg, dual). |
| `methods/ip_adapter.py` | IP-Adapter: PE-Core-L14-336 vision encoder + Perceiver resampler + per-block `to_k_ip`/`to_v_ip`. |
| `methods/easycontrol.py` | EasyControl: per-block cond LoRA on self-attn (q/k/v/o) + FFN + scalar `b_cond` logit-bias gate; two-stream block forward at training, KV-cache prefill at inference. |
| `methods/repa.py` | REPA auxiliary alignment loss (Yu et al., arXiv:2410.06940). Adds a single hook on `unet.blocks[args.repa_layer]` + small MLP head; cosine-aligns the pooled DiT mid-block feature against the spatial-mean cached PE-Core feature. Toggled by `use_repa = true` in `configs/methods/lora.toml`. |
| `methods/soft_tokens.py`, `methods/ip_adapter_pe_lora.py` | Soft tokens (SoftREPA parameterization) + the PE-LoRA delta path used by IP-Adapter / Anima Tagger. |
| `attention_dispatch.py` | Unified `dispatch_attention()` — backend router (SDPA / xformers / FA2 / FA3 / sageattn / flex). |
| `spectrum.py` | Spectrum inference acceleration (Chebyshev feature forecasting). See root CLAUDE.md §Spectrum and `docs/methods/spectrum.md`. |

## Three-axis routing surface (plan2)

As of commit `1dca212`, the LoRA-family routing flags collapsed into three orthogonal cfg axes consumed by `lora_anima/config.py::LoRANetworkCfg.from_kwargs` and dispatched by `__init__.py::resolve_network_spec`:

| Knob | Values | Meaning |
|---|---|---|
| `use_moe_style` | `False` / `"shared_A"` / `"independent_A"` | Expert layout — no experts, Hydra-style shared `lora_down`, or stacked per-expert `(lora_down, lora_up)`. |
| `route_per_layer` | `True` / `False` | Router location — per-Linear (Hydra default) or one network-level router. |
| `router_source` | `"none"` / `"input"` / `"sigma"` / `"fei"` | What signal the router reads — Linear input, σ-features, FEI on `z_t`, or no router. |

Variants that exist as cells in this matrix:

| Variant | `use_moe_style` | `route_per_layer` | `router_source` | Network module / path |
|---|---|---|---|---|
| Plain LoRA / OrthoLoRA / T-LoRA / ReFT | `False` | — | `"none"` | `lora_anima` + `lora_modules/` (LoRA, ortho, ReFT) |
| HydraLoRA (paper) | `"shared_A"` | `True` | `"input"` | `lora_anima` + `lora_modules/hydra.py` |
| σ-router on Hydra | `"shared_A"` | `True` | `"sigma"` | same |
| FEI-on-Hydra (lora.toml default) | `"shared_A"` | `True` | `"fei"` | same |
| **FeRA (author-faithful)** | `"independent_A"` | `False` | `"fei"` | `lora_anima` + `lora_modules/stacked_experts.py` + `GlobalRouter` |

Pre-plan2 metadata stamps (`ss_use_hydra`, `ss_use_fei_router`, `ss_network_module = "networks.methods.fera"`) **no longer load** — the legacy fallback was removed in plan2 task #6. The new stamps are `ss_use_moe_style` / `ss_route_per_layer` / `ss_router_source`.

`ortho` stays a per-module bool — set `use_ortho=true` to get the PSOFT-style Cayley-rotated SVD parameterization (applies to OrthoLoRA, OrthoHydra, and `StackedExpertsLoRAModule`).

## LoRA variants

All live in `lora_modules/`. Stack freely via toggle flags in `configs/methods/lora.toml`.

- **LoRA** (`lora.py::LoRAModule`) — Classic low-rank: `y = x + (x @ down @ up) * scale * multiplier`.
- **OrthoLoRA** (`ortho.py::OrthoLoRAModule`, `OrthoHydraLoRAModule`) — SVD-based orthogonal parameterization with orthogonality regularization (linear layers only). Saved as plain LoRA via thin SVD on ΔW at save time. See `docs/methods/psoft-integrated-ortholora.md`.
- **T-LoRA** — Not a separate class. A `_timestep_mask` buffer on `LoRAModule` / `OrthoLoRAModule` (registered in `base.py`) is rebound to a shared live-updated mask by `lora_anima/network.py::LoRANetwork.set_timestep_mask`. Effective rank varies with denoising step via a power-law schedule. **Training-only** — inference runs full rank at every t (baking into DiT is bit-equivalent). See `docs/methods/timestep_mask.md`.
- **HydraLoRA** (`hydra.py`) — MoE-style multi-head routing: shared `lora_down` + per-expert `lora_up_i` heads, layer-local router on the adapted Linear's input (`router_source="input"`) or σ-features / FEI features (`"sigma"` / `"fei"`). Requires `cache_llm_adapter_outputs=true`. Produces a `*_moe.safetensors` sibling for router-live inference. See `docs/methods/hydra-lora.md`.
- **Stacked experts / FeRA** (`stacked_experts.py::StackedExpertsLoRAModule`) — Independent-A layout: each expert owns its own `(lora_down, lora_up)`, stacked as `(E, …)` Parameters consumed in one `einsum`. Routed by `GlobalRouter` (one network-level router fed by FEI of `z_t`). Supports both free and PSOFT-style ortho parameterization. See `docs/experimental/fera.md`.
- **ReFT** (`reft.py`) — Block-level residual-stream intervention (LoReFT, Wu et al. NeurIPS 2024). One `ReFTModule` per selected DiT block wraps the block's `forward` and adds `R^T·(ΔW·h + b)·scale` to the output; orthogonality regularized on `R`. Additive side-channel, composes with any LoRA variant, lives in the same `.safetensors`. Vanilla ComfyUI can't load ReFT (weight-patcher silently drops `reft_*` keys) — use the `AnimaAdapterLoader` custom node (`custom_nodes/comfyui-hydralora/`).

## GlobalRouter (network-level routing)

`lora_anima/network.py::GlobalRouter` — `Linear(F_in → H) → ReLU → Linear(H → E) → softmax/τ`. Built when `cfg.route_per_layer=False` and `cfg.use_moe_style != False`. Final layer is zero-init so step-0 gates are uniform; warmup is the symmetry-breaker.

Hook site: `LoRANetwork.set_fei(z_t)` runs the FEI computation (via `library/runtime/fei.py`) and the router once, then writes the resulting `(B, num_experts)` tensor by reference into each routing-aware module's `_routing_weights` buffer. One Python-level write propagates to every adapted Linear that step — that's the architectural commitment of the "global router" design and the failure mode to watch for (router collapse → every layer collapses together).

Training-loop call: `train.py` fires `network.set_fei(noisy_model_input)` at the per-step σ/FEI hook block when the cfg has `route_per_layer=False` and `router_source="fei"`. Inference: `library/inference/generation.py` mirrors the same call before each Euler step.

## Attn fuse spec (qkv/kv fuse↔split)

`attn_fuse.py::AttnFuseSpec` + `iter_split_groups` + `match_fused_spec` is the single source of truth for the runtime-fused `qkv_proj` (self-attn) / `kv_proj` (cross-attn) ↔ on-disk split `q/k/v_proj` layout. ComfyUI's cosmos backbone uses the split layout while Anima's training-side DiT uses the fused projections; save always writes split, load always re-fuses. Both `lora_save.py` and `loading.py` walk the same specs, so adding a new fused projection only needs one entry here.

## Attention dispatch

`attention_dispatch.py::dispatch_attention()` routes to the active backend (torch SDPA, xformers, flash-attn v2/v3, sageattn, flex attention). **Tensor layout differs by backend** — BHLD for SDPA/sageattn, BLHD for xformers/flash-attn — so callers must hand tensors to the dispatcher in a known layout and the dispatcher transposes as needed. Check the backend branches before adding new attention call sites.

FA4 (flash-attention-sm120) was evaluated and is currently disabled — see `docs/optimizations/fa4.md`. The KV-trim + LSE-correction path that depended on FA4 is dormant in the same dispatcher (`crossattn_full_len`, the `flash4` branch); see fa4.md for what would need to be re-enabled to restore it.

## Timestep masking — when to update what

T-LoRA's mask is a single CPU/GPU buffer shared across all adapted Linears, updated once per denoising step from `lora_anima/network.py`. Anything that calls into LoRA modules during a forward must have the mask set for the current `t` already — `factory.py` and `network.py` are the only places that should be poking `set_timestep_mask` / `clear_timestep_mask`. New adapter variants that want timestep awareness should reuse the same buffer pattern (register as a buffer in `base.py`, read it inside `forward`) rather than threading `t` through every call site.
