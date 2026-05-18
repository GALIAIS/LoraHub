# FeRA — frequency-energy constrained routing

Port of Yin et al., *FeRA: Frequency-Energy Constrained Routing for Effective Diffusion Adaptation Fine-Tuning* (arXiv:2511.17979). DiT runs frozen; every adapted Linear is replaced by a small MoE of **independent rank-r experts**, and a **single global router** consumes the spectral state of the current latent `z_t` and emits one `(B, num_experts)` gate that every adapted layer reuses for that step.

Reference: `FeRA/` in the repo root (paper authors' code: `FeRA/fera/{layer,model,utils,config}.py`).

As of plan2 (commit `1dca212`), FeRA is no longer a standalone `network_module`. It lives as one cell of the LoRA-family three-axis routing matrix on `networks.lora_anima`:

| Axis | Value for author-faithful FeRA |
|------|---|
| `use_moe_style` | `"independent_A"` (each expert owns `(lora_down, lora_up)`) |
| `route_per_layer` | `False` (one network-level router) |
| `router_source` | `"fei"` (gate input = `FEI(z_t)`) |

Same three knobs also cover the earlier FEI-on-Hydra variant (shared-A + per-Linear router on FEI features) — see "Variants in the three-axis matrix" below.

## Quick start

```bash
make lora-gui GUI_PRESETS=fera                # configs/gui-methods/fera.toml + preset default
python tasks.py lora-gui fera                 # cross-platform
make lora-gui GUI_PRESETS=fera PRESET=low_vram

make test-hydra                               # router-live inference against the latest
                                              # output/anima_fera*.safetensors — the
                                              # stacked-experts checkpoint flows through
                                              # the same `_is_hydra_moe` safetensors sniff
                                              # because both layouts ship `lora_ups.{i}.weight`
```

There is **no** `make fera` toggle on `configs/methods/lora.toml` directly — the LoRA-family methods file ships the FEI-on-Hydra shared-A cell uncommented as the default LoRA stack. The author-faithful FeRA cell (`independent_A`) lives in `configs/gui-methods/fera.toml` so the `lora-gui` path is the canonical entry.

## What it actually does

```
                                 z_t  (B, C, H_lat, W_lat)
                                  │
                  FrequencyEnergyIndicator (DoG pyramid → simplex)   (library/runtime/fei.py)
                                  │
                          e_t (B, num_bands)
                                  │
                        GlobalRouter (Linear→ReLU→Linear→softmax/τ)
                                  │
                       w  (B, num_experts)            ◄── one global router
                                  │
       ┌──────────────────────────┴──────────────────────────┐
       ▼                          ▼                          ▼
┌─ StackedExperts ┐       ┌─ StackedExperts ┐       ┌─ StackedExperts ┐
│  attn.qkv       │   …   │  attn.out       │   …   │  mlp.layer1     │   …
│  base(x)+       │       │  base(x)+       │       │  base(x)+       │
│  Σ_k w_k·       │       │  Σ_k w_k·       │       │  Σ_k w_k·       │
│   E_k(x)        │       │   E_k(x)        │       │   E_k(x)        │
└─────────────────┘       └─────────────────┘       └─────────────────┘
  ^ independent             ^ independent             ^ independent
   (down_k, up_k)            (down_k, up_k)            (down_k, up_k)
```

Per adapted Linear:

```
StackedExpertsLoRAModule(x) = base(x) + multiplier · Σ_{k=0..E-1} w_k · LoRAExpert_k(x)
LoRAExpert_k(x) = lora_up_k( lora_down_k(x) ) · (alpha / rank)
```

The base Linear is frozen. Each expert has its own independent `(lora_down, lora_up)` pair — **no shared-A pooling** like HydraLoRA. At init `lora_up_k = 0` so the residual contribution is exactly zero at step 0 regardless of routing weights.

## Variants in the three-axis matrix

The three-axis surface (plan2.md §three-axis-config) covers both points in the FEI design space and several others:

| Variant | `use_moe_style` | `route_per_layer` | `router_source` |
|---|---|---|---|
| Plain LoRA / OrthoLoRA / T-LoRA / ReFT | `False` | — | `"none"` |
| HydraLoRA (paper) | `"shared_A"` | `True` | `"input"` |
| σ-router on Hydra | `"shared_A"` | `True` | `"sigma"` |
| FEI-on-Hydra (lora.toml default) | `"shared_A"` | `True` | `"fei"` |
| **FeRA (this doc)** | `"independent_A"` | `False` | `"fei"` |

`LoRANetworkCfg.from_kwargs` rejects `use_moe_style=False` combined with any router knob, and `route_per_layer=False, router_source="input"` (no "global input" per DiT forward). `ortho` stays a per-module bool — set `use_ortho=true` to get the PSOFT-style Cayley-rotated SVD parameterization on each expert.

## Implementation map

| File | Role |
|------|------|
| `networks/lora_anima/network.py` | `LoRANetwork` (the shared LoRA network) + `GlobalRouter` (Linear→ReLU→Linear→softmax/τ). On `use_moe_style="independent_A"` + `route_per_layer=False` + `router_source="fei"`, the network builds one `GlobalRouter` and fires it from `set_fei(z_t)` once per step; the resulting `(B, num_experts)` tensor is written by reference into every routing-aware module's `_routing_weights` buffer. |
| `networks/lora_modules/stacked_experts.py` | `StackedExpertsLoRAModule` — independent-A expert layout. Owns `lora_downs` / `lora_ups` as `(E, …)` stacked Parameters consumed in one `einsum`. Supports both free and PSOFT-style ortho parameterization (shared `Q_basis`/`P_basis` SVD bases + per-expert Cayley `S_q, S_p` + per-expert diagonal `λ`). |
| `networks/attn_fuse.py` | `AttnFuseSpec` + `iter_split_groups` — single source of truth for the runtime-fused `qkv_proj`/`kv_proj` ↔ on-disk split `q/k/v_proj` layout. Save and load both walk these specs; centralizing them keeps the two scanners from drifting. |
| `networks/lora_save.py` | `stacked_experts_global_fei` save handler — writes split-attention `lora_downs.{i}` / `lora_ups.{i}` + the global router state dict + the three plan2 metadata stamps (`ss_use_moe_style` / `ss_route_per_layer` / `ss_router_source`). Distills the ortho parameterization to plain `(lora_down, lora_up)` keys for inference. |
| `networks/__init__.py` | `resolve_network_spec` dispatches `use_moe_style="independent_A"` to the `stacked_experts_global_fei` NetworkSpec. |
| `library/runtime/fei.py` | `gaussian_blur_2d` + DoG kernel cache + `compute_fei` helper. Shared with the FEI-on-Hydra variant — single kernel cache keyed by `(σ_low, σ_mid, kernel_size)`. |
| `library/inference/generation.py` | Calls `compute_and_set_hydra_fei(anima, latents)` per Euler step (right after `set_hydra_sigma`); that helper (in `library/inference/adapters.py`) runs the DoG pyramid on `z_t` and forwards the resulting `(B, fei_feature_dim)` tensor to `network.set_fei`, which then fires the GlobalRouter and broadcasts the gates to every adapted Linear. |
| `library/inference/models.py` | `_is_hydra_moe` matches the `lora_ups.{i}.weight` key pattern, which now also catches plan2 stacked-experts checkpoints. Both layouts go through the same router-live dynamic-hook inference path (no static merge). |
| `library/training/losses.py` | `_fera_fecl_loss` reads either a pre-computed scalar from `ctx.aux['fecl_loss']` (legacy) or a `ctx.aux['fera']['z_base']` payload (current); composer auto-activates on `LoRANetwork` with `use_moe_style="independent_A"`. |
| `train.py` | At the per-step σ/FEI hook block: computes FEI features from `noisy_model_input` and calls `network.set_fei(_fei)` when the cfg has `route_per_layer=False, router_source="fei"`. |
| `configs/gui-methods/fera.toml` | Default config for the author-faithful cell. |
| `configs/methods/lora.toml` | Default LoRA-family stack — ships the shared-A FEI-on-Hydra cell uncommented (LoRA + OrthoLoRA + T-LoRA + Hydra(shared_A, FEI)). |
| `bench/fera/` | Diagnostic probes: `probe_fei.py` (3-bucket inference probe), `probe_fei_dataset.py` (training-distribution σ_low sweep), `probe_fei_3band_dataset.py`, `probe_closed_loop.py`, `refactor_lowdim_forward.py`, `expressivity_analysis.py`. Pre-network-module work that settled the 2-band collapse and σ_low rule. |

## Parameter count

Per adapted Linear: `E · r · (D_in + D_out)`. Default `E=4, r=32` on Anima's 28 blocks × 5 Linears gives:

```
28 blocks × 5 Linears × 4 experts × 32 rank × (D_in + D_out) avg
≈  28 × 5 × 4 × 32 × (2048 + 2048)  ≈  147 M    (experts)
+  router: (2 → 64) + (64 → 4)      ≈  390      (negligible)
```

Far heavier than vanilla LoRA at the same rank because the per-expert `(down, up)` pair is independent — switching to `use_ortho=true` drops the trainable surface by ~3 OOM via shared SVD bases + per-expert `(S_q, S_p, λ)` (see [`docs/methods/psoft-integrated-ortholora.md`](../methods/psoft-integrated-ortholora.md)).

## Knobs (`configs/gui-methods/fera.toml`)

| Param | Default | Notes |
|---|---|---|
| `network_dim` (rank) | 32 | Each expert is `(in → r → out)`. Independent across experts. |
| `network_alpha` | 32 | Same as rank ⇒ scale `α/r = 1`. |
| `use_moe_style` | `"independent_A"` | Picked up by `resolve_network_spec` → `stacked_experts_global_fei`. |
| `route_per_layer` | `False` | One network-level `GlobalRouter`. |
| `router_source` | `"fei"` | Router input = `FEI(z_t)`. |
| `num_experts` | 4 | Paper used 3; default raised to 4 here. |
| `use_ortho` | `true` | PSOFT-style Cayley parameterization on each expert. Drops trainable params ~3 OOM. |
| `fera_num_bands` | 3 | Paper default. Drop to 2 for Anima's bench-validated bimodal split (see [[project_fera_probe_2band_decision]]). |
| `fei_feature_dim` | 2 | Router input width. Set to `num_bands` for the raw simplex, or 2 for the `(e_low, e_high)` low-dim projection. |
| `fei_sigma_low_div` | 4.0 | `σ_low = min(H_lat, W_lat) / fei_sigma_low_div`. Default picked from the 2026-05-13 dataset sweep (`bench/fera/results/20260513-1649-dataset-sweep/`) — `div=4` yields the highest router std(e_low) at low/mid t on real training latents. NOT the paper's pixel-domain `min(H, W)/128` (that's SD2-512-specific). |
| `router_tau` | 0.7 | Softmax temperature. Lower → sharper expert specialization. |
| `router_hidden_dim` | 64 | Router MLP hidden width. |
| `network_router_lr_scale` | 10 | Router LR = `network_router_lr_scale × unet_lr`. |
| `balance_loss_weight` | 3e-7 | Switch-transformer-style load-balance loss. |
| `balance_loss_warmup_ratio` | 0.4 | Linear ramp over first 40% of training. |
| `fera_fecl_weight` | 0.0 | FECL aux loss weight (paper used 0.1–0.2). Activates the base-pass forward inside the loss composer — 2× per-step forward cost. Default 0 = disabled. |
| `router_targets` | regex over qkv/q/kv/output/MLP | Which adapted Linears feed the routing-weight broadcast. Restrict to ablate. |
| `compile_mode` | `"full"` | The `linalg.solve` inside the ortho Cayley path may force a graph break per StackedExperts module; bench compile-on vs compile-off before relying on it. |

## σ_low rule (why not the paper's κ)

The paper picks DoG kernel scale `κ = min(H, W) / 128` — a pixel-domain constant tuned for SD2 at 512×512. Anima trains under constant-token bucketing (`H_lat · W_lat ≈ 4096`) at varied aspect ratios, so a fixed pixel σ would land on different fractions of the latent grid per bucket. Bench probes (`bench/fera/results/20260512-1814-fera-pilot/`, `…20260512-1827-fera-midwide/`) validated the latent-domain rule on inference trajectories; the 2026-05-13 dataset sweep (`bench/fera/probe_fei_dataset.py`, results under `…20260513-1649-dataset-sweep/`) then picked `div=4` over `div=8` for real training-distribution router signal:

```
σ_low = min(H_lat, W_lat) / fei_sigma_low_div     (default fei_sigma_low_div = 4.0)
```

Aspect invariance held across 1024², 832×1248, 1248×832 at the inference probe (mean `|Δ FEI|` < 0.02 between mirror buckets). The dataset sweep ranked divisors by population std(e_low) at flow-matching training-input t: **div=4 highest** (0.131 at t=0.05), div=8 second (0.112), paper-style div=128 worst (0.020). This is the same rule the FEI-on-Hydra variant uses.

## 3 bands vs 2 bands

Author paper picks `num_bands = 3` (low / mid / high). On Anima flow-matching latents the mid band is structurally near-empty (`e_mid ≤ 8%` at `σ_mid = 4`, `≤ 1.5%` at `σ_mid = 8`) — see [[project_fera_probe_2band_decision]]. Anima's velocity target `(image − noise)` is bimodal by construction (concentrated at very-low image structure + very-high noise), so the third band carries no routing-useful signal.

The shipped `configs/methods/lora.toml` default (FEI-on-Hydra) sets `fera_num_bands = 2` directly. `configs/gui-methods/fera.toml` keeps `num_bands = 3` for paper fidelity; if held-out gate-entropy + per-expert utilization shows expert collapse, drop to 2.

## FECL

The paper's frequency-energy consistency loss (eq. 10) is wired into the loss composer. When `fera_fecl_weight > 0` and the active network is a `LoRANetwork` with `use_moe_style="independent_A"`, every training step does:

1. **Main forward** with FeRA routing active → `z_fera` (with grad).
2. **`network.clear_routing_weights()`** to fall every routing-aware module through to its frozen base.
3. **Second no-grad forward** under the same autocast → `z_base`.
4. **`network.set_fei(z_t)`** to restore the gates — load-bearing under gradient checkpointing, which replays the main forward during backward and needs the same routing weights.
5. **Stash `z_base` in `ctx.aux['fera']['z_base']`**; the composer's `_fera_fecl_loss` handler runs the band decomposition + paper-eq.10 itself.
6. Handler multiplies by `fera_fecl_weight` and adds to the total — same `_STAGE_SCALAR_BROADCAST` stage as ortho / hydra-balance / soft-tokens-contrastive.

FECL bandwise distribution: pushes the adapter correction `δ = z_fera − z_base` to concentrate its energy in the same bands the residual `r = z_fera − target` has energy in. Encourages experts to spend their capacity where the FM loss is currently failing rather than perturbing bands that are already correct.

Two caveats:

- The 2× forward cost is real — at default 4 epochs that's an extra full training run worth of compute. The base pass is `no_grad` but still has to do the full DiT forward.
- At `num_bands = 2` the loss degenerates to a single scalar ratio (the two band shares sum to 1, so weighted (Δshare)² is content-free across bands). Setting `fera_fecl_weight > 0` while `fera_num_bands = 2` will train but the FECL contribution can't differentiate experts spectrally — the term becomes purely a magnitude regularizer on the correction. If you want 2-band FeRA, set `fera_fecl_weight = 0`.

Setting `fera_fecl_weight = 0` skips the base-pass entirely (the gate is checked before the second forward), so the inactive case has zero overhead.

## Save format

`output/ckpt/<output_name>.safetensors` keys (`output_name = "anima_fera"` by default):

```
# GlobalRouter — fp32, lives at network root
global_router.net.0.weight                    (router_hidden_dim, fei_feature_dim)
global_router.net.0.bias                      (router_hidden_dim,)
global_router.net.2.weight                    (num_experts, router_hidden_dim)
global_router.net.2.bias                      (num_experts,)

# Per-adapted-Linear stacked experts (distilled — inference, ComfyUI, Spectrum)
lora_unet_<dotted_path>.lora_downs.{0..E-1}   (r, in)         bf16
lora_unet_<dotted_path>.lora_ups.{0..E-1}     (out, r)        bf16

# Native ortho-mode keys (training resume only, when use_ortho=true)
lora_unet_<dotted_path>.S_p                   (E, r, r)       fp32
lora_unet_<dotted_path>.S_q                   (E, r, r)       fp32
lora_unet_<dotted_path>.lambda_layer          (E, r)          fp32
lora_unet_<dotted_path>.P_basis               (out, r)        fp32
lora_unet_<dotted_path>.Q_basis               (r, in)         fp32
```

**ComfyUI compatibility — split q/k/v on disk.** ComfyUI's cosmos backbone uses split `q_proj`/`k_proj`/`v_proj` Linears while Anima's training-side DiT uses fused `qkv_proj` (self-attn) and `kv_proj` (cross-attn). `attn_fuse.py::AttnFuseSpec` is the single source of truth; the save handler always writes the **split** layout (slicing the fused `lora_up` along its output axis to `[Q | K | V]` matching `Attention.compute_qkv`'s `unflatten(..., (3, n_heads, head_dim)).unbind(dim=-3)` order) and `loading.py` re-fuses on load. The training-side DiT (which adapts the fused projections) receives a single stacked Parameter.

**Plan2 metadata stamps:**

```
ss_network_module       = "networks.lora_anima"
ss_use_moe_style        = "independent_A"
ss_route_per_layer      = "False"
ss_router_source        = "fei"
ss_num_experts          = "4"
ss_fera_num_bands       = "3"
ss_fei_sigma_low_div    = "4.0"
ss_use_ortho            = "True"        # if ortho mode
```

`factory.from_weights_metadata` rebuilds the network from these stamps without the original TOML. Pre-plan2 checkpoints (with `ss_use_hydra` / `ss_use_fei_router` / `ss_network_module = "networks.methods.fera"`) **no longer load** — the legacy fallback was removed in plan2 task #6. The frozen base Linear weights are not saved (they belong to the DiT — see "Frozen base ownership" below).

## Frozen base ownership

`StackedExpertsLoRAModule` keeps a reference to the original `nn.Linear` via `object.__setattr__` so it bypasses `nn.Module`'s child-tracking. The base layer's weights stay where they always were — owned by the DiT — and don't leak into `LoRANetwork.state_dict()`. Consequences:

- The trained FeRA file only carries the router + expert deltas + (in ortho mode) the SVD bases.
- Loading FeRA against a DiT that doesn't match the architecture silently produces broken outputs — no shape check on the base.
- Static merge of router-mixed experts into the DiT is not supported. `scripts/merge_to_dit.py` refuses unless `--allow-partial`, which drops the FeRA portion.

## Compatibility

| Component | Compat | Notes |
|---|---|---|
| Training loop | ✅ | `train.py` calls `network.set_fei(noisy_model_input)` at the σ/FEI hook block when the cfg has `route_per_layer=False, router_source="fei"`. |
| Standard inference | ✅ | `library/inference/models.py::_is_hydra_moe` matches the `lora_ups.{i}.weight` keys and takes the dynamic-hook route. `set_fei` fires per step from `generation.py`. |
| Spectrum inference | ⚠ | Per-step `set_fei` is wired, but on a Spectrum cached step the FEI/gate is updated while the cached features may have been forecast from a different gate distribution. Bench against `--spectrum` before relying on it. |
| `torch.compile` | ✅ vanilla / ⚠ ortho | Vanilla stacked-experts is a base + single `einsum`; shape-static under constant-token bucketing. Ortho mode adds a `linalg.solve` Cayley step that may force a graph break per StackedExperts module — bench compile-on/off. |
| `blocks_to_swap` | ✅ | `StackedExpertsLoRAModule` replaces the original Linear in-place; block swap moves it and its experts together. |
| `gradient_checkpointing` | ✅ | The adapter is a thin Linear-replacement; checkpointing at block granularity wraps it correctly. FECL base-pass is replayed during backward, hence the `set_fei` restore in step 4 above. |
| Modulation guidance | ✅ orthogonal | AdaLN path is untouched. |
| T-LoRA / OrthoLoRA / ReFT | ⚠ partial | `use_ortho=true` is part of the same cfg surface and works inside FeRA. T-LoRA timestep masking and ReFT are designed against shared-A / plain-LoRA layouts; verify the toggle on a small bench before stacking. |
| DCW (scalar / v4) | ✅ orthogonal | Sampler-level correction; composes with anything upstream of the Euler step. |
| ComfyUI | ✅ | The **Anima Adapter Loader** node (`custom_nodes/comfyui-hydralora/`) auto-detects stacked-experts checkpoints via the same `lora_ups.{i}` sniff and installs the dynamic per-Linear hook. |
| Other Hydra-moe loaded simultaneously | ❌ | `models.py` refuses two moe files in one `--lora_weight` list — pick one router scheme. |

## What to measure

The bet is that the global router on latent spectral state captures *per-prompt* routing variance the σ-router can't — populations at the same σ get the same Hydra gate but different FEI gates. The whole point hinges on whether this content-aware variance translates to a quality lift.

1. **Router gate entropy across training.** Should stabilize above zero, with consistent per-prompt-type variation (scenery vs portrait vs flat-style routes to different gate distributions). Collapse → one expert always wins → FeRA reduces to a plain LoRA with extra unused params. Logged via `fera/router_entropy`, `fera/router_margin`, `fera/expert_usage/*` (added in plan2 task #4).
2. **Per-expert utilization on a held-out prompt set.** Histogram of `argmax_k w_k` (or weighted utilization) across prompts. Useful answer: experts specialize by *content type*, not by σ-stage (that's what Hydra does).
3. **Per-prompt routing stability across seeds.** Two seeds of the same prompt should produce similar gate distributions (gate is a function of `z_t`, which differs by seed but converges to similar spectral shape). If gates drift wildly seed-to-seed, the router is noise-sensitive — tighten `router_tau` or grow `router_hidden_dim`.
4. **A/B vs FEI-on-Hydra (`configs/methods/lora.toml` default).** Same dataset, matched epochs/lr. The author-faithful path is heavier (independent A) and globally-routed; FEI-on-Hydra is lighter and locally-routed. Whichever wins tells us which axis matters more on Anima.
5. **Sample quality vs `make lora`.** The hard test — is FeRA better than the plain LoRA-family default? Use the same prompt set used for `make test` and look at structural quality + prompt following + style coherence. FM val-MSE is uninformative on Anima (see [[project_fm_val_loss_uninformative]]).

## Hyperparameters worth sweeping

| Knob | Default | Range to try | Why |
|---|---|---|---|
| `num_experts` | 4 | 2, 3, 4, 6 | Paper: 3. Watch for expert utilization saturating ⇒ too many. |
| `fera_num_bands` | 3 | 2, 3 | 2 = Anima-validated. 3 only useful if router actually splits along mid-band. |
| `network_dim` (rank) | 32 | 8, 16, 32, 64 | Independent per-expert → rank multiplied by `E`. |
| `router_tau` | 0.7 | 0.3, 0.7, 1.0, 2.0 | Lower τ → sharper specialization but more sensitive to FEI noise. |
| `router_hidden_dim` | 64 | 32, 64, 128 | Router input is only `fei_feature_dim` floats; the bottleneck is usually expressive enough. |
| `fei_sigma_low_div` | 4.0 | 2, 4, 8, 16 | Higher → tighter low band (more high-freq picked up there). 2026-05-13 dataset sweep picked 4 over 8 on training latents; both 4 and 8 are in the Pareto region. |
| `router_targets` | all attn + MLP | MLP-only, attn-only | Ablate which sites benefit from FeRA gating. MLP-only mirrors the FEI-on-Hydra default. |
| `fera_fecl_weight` | 0.0 | 0, 0.1, 0.2 | Activates the FECL base-pass + composer term (2× per-step forward cost). Only meaningful at `num_bands ≥ 3` (at 2 bands the term collapses to a magnitude regularizer). |
| `use_ortho` | `true` | `true`, `false` | Cayley-rotated SVD basis vs free `(down, up)`. Ortho drops trainable params ~3 OOM. |
| `multiplier` (inference) | 1.0 | 0.0, 0.5, 1.0, 1.5 | `0.0` short-circuits to frozen base for clean ablation. |

## Files

- `networks/lora_anima/network.py` — `LoRANetwork` + `GlobalRouter`.
- `networks/lora_modules/stacked_experts.py` — `StackedExpertsLoRAModule`.
- `networks/attn_fuse.py` — `AttnFuseSpec`, the qkv/kv fuse↔split spec.
- `networks/lora_save.py` — `stacked_experts_global_fei` save handler.
- `networks/__init__.py` — `resolve_network_spec` + `NETWORK_REGISTRY`.
- `configs/gui-methods/fera.toml` — author-faithful FeRA cell config.
- `configs/methods/lora.toml` — default LoRA-family stack (ships shared-A FEI-on-Hydra uncommented).
- `library/runtime/fei.py` — DoG kernels + FEI computation, shared across both cells.
- `library/training/losses.py::_fera_fecl_loss` — FECL handler.
- `bench/fera/` — diagnostic probes; see [[project_fera_probe_2band_decision]] for the σ_low / num_bands findings.
- `FeRA/` — paper authors' reference implementation (read-only, for diffing).
- `plan2.md` — plan for retiring `networks/methods/fera.py` into `lora_anima` via three-axis cfg (commit `1dca212`).

## Citation

```
@article{yin2025fera,
  title={FeRA: Frequency-Energy Constrained Routing for Effective Diffusion Adaptation Fine-Tuning},
  author={Yin, Bo and Hu, Xiaobin and Zhou, Xingyu and Jiang, Peng-Tao and Liao, Yue
          and Zhu, Junwei and Zhang, Jiangning and Tai, Ying and Wang, Chengjie
          and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2511.17979},
  year={2025}
}
```
