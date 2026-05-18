# ChimeraHydra: dual-pool additive MoE on the OrthoHydra basis

A re-parameterization of HydraLoRA's expert pool. Each adapted `Linear` keeps the **shared `lora_down` + stacked `lora_up`** layout from `hydralora.md`, but the $E$ experts are split into two **disjoint pools** routed by two **structurally different routers**:

| Pool          | Size      | Router          | Router input            | What it sees       |
| ------------- | --------- | --------------- | ----------------------- | ------------------ |
| **Content**   | $K_c$     | per-Linear      | RMS-pooled rank-$r$ `lx` | sample/content axis |
| **Frequency** | $K_f$     | network-level   | FEI of $z_t$ (+σ-features, currently off) | noise-level axis |

The two pools' outputs are **added**. No multiplicative gate, no σ-band overlap mask, no curriculum. Specialization is enforced by **router-input separation**: the content router cannot see noise level, the freq router cannot see pooled text features, so each pool's experts can only differentiate along its own axis.

Recap from `hydralora.md`: a HydraLoRA module replaces a single $B$ with $E$ stacked heads + a layer-local router that emits a per-sample softmax over them. ChimeraHydra keeps the exact same einsum kernel but constructs the gate from **two disjoint sources** stitched together: `gate = cat([π_c, π_f], dim=-1)`. The forward pass is mathematically identical to single-pool routing with a partitioned $E = K_c + K_f$ gate vector.

> Currently experimental — see `docs/proposal/chimera_hydra.md` for the bench plan and `docs/experimental/chimera-hydra.md` for the user-facing entry points (`make exp-chimera`, `make lora-gui GUI_PRESETS=chimera_hydra`).

---

## 1. Why two pools

A single HydraLoRA router has to do two jobs at once. Anima's denoising flow has two roughly orthogonal sources of variance: **content** (which style / artist / subject the sample is from) and **noise level** ($\sigma_t$, which controls whether the model is doing coarse layout at high $\sigma$ or texture refinement at low $\sigma$). Plain HydraLoRA's router reads only the pooled rank-$r$ activation; it can pick up content-axis signal but has no direct view of $\sigma$. The σ-router variant (`router_source = "sigma"`) and FEI-on-Hydra variant (`router_source = "fei"`) replace the input axis but pay the symmetric price — they lose content-axis signal.

A single router can't do both because **its input is one tensor**. The router weight is rank-$E$; if the input is content-only, the output is content-conditioned; if the input is FEI, the output is σ-conditioned. There is no middle ground where one $E$-way softmax learns both axes — that would require the experts themselves to be 2D-indexed, which is exactly the staged-2D design ChimeraHydra supersedes (multiplicative gate $g_c \odot g_t$ with phased training; see proposal §"Supersedes").

ChimeraHydra's structural answer: **two routers, one pool each, additive composition**. The content router specializes the first $K_c$ experts along the sample axis. The freq router specializes the next $K_f$ experts along the noise axis. The two pools sum their contributions, so a single forward pass produces an effective delta with both kinds of conditioning folded in.

---

## 2. The math

Per adapted Linear, ChimeraHydra stores the same Cayley-parameterized OrthoHydra state (`ortholora.md` §2 + `hydralora.md` §5.2) over $E = K_c + K_f$ experts:

| Component                                | Shape                          | Trainable | Role                                    |
| ---------------------------------------- | ------------------------------ | --------- | --------------------------------------- |
| `Q_basis` (frozen SVD)                   | $(r,\, d_\text{in})$           | buffer    | Shared down basis                       |
| `P_bases` (frozen SVD, partitioned)      | $(E,\, d_\text{out},\, r)$      | buffer    | Per-expert disjoint up basis slices     |
| `S_q`                                    | $(r,\, r)$                     | yes       | Down-side Cayley seed                   |
| `S_p`                                    | $(E,\, r,\, r)$                | yes       | Per-expert up-side Cayley seeds         |
| `lambda_layer`                           | $(1,\, r)$                     | yes       | Diagonal scale, zero-init               |
| `router.weight` (content)                | $(K_c,\, r)$                   | yes       | Layer-local gate logits                 |
| `router.bias`                            | $(K_c,)$                       | yes       | Zero-init                               |
| `_freq_routing_weights` (buffer)         | $(B,\, K_f)$                   | no        | Slot for FreqRouter broadcast           |

The forward (`networks/lora_modules/chimera.py:155–232`, simplified) splits cleanly into a Cayley solve, a router stage, and a two-pool einsum-fold:

```python
# Effective bases (Cayley rotation of frozen SVD slices)
R_q, R_p = cayley_batched(S_q, S_p)               # (r,r) + (E,r,r)
Q_eff    = R_q @ Q_basis                          # (r, d_in)
P_eff    = P_bases @ R_p                          # (E, d_out, r)

# Down projection
lx       = F.linear(x, Q_eff)                     # (B, L, r)

# Gate construction: two routers, disjoint outputs
pi_c     = softmax(content_router(rms_pool(lx)))  # (B, K_c)   — per-Linear
pi_f     = _freq_routing_weights                  # (B, K_f)   — network broadcast
gate     = cat([pi_c, pi_f], dim=-1)              # (B, E)

# T-LoRA: mask content branch only, leave freq branch full rank
P_eff_c, P_eff_f = P_eff[:K_c], P_eff[K_c:]
P_combined_c = einsum("bc,cor->bor", gate[..., :K_c], P_eff_c) * mask_t   # (B, d_out, r)
P_combined_f = einsum("bf,for->bor", gate[..., K_c:], P_eff_f)            # (B, d_out, r)

# Single bmm over the additive P_combined
P_combined = P_combined_c + P_combined_f
out        = bmm(lx * lambda_layer, P_combined.transpose(1, 2))           # (B, L, d_out)

return org_forward(x) + out * multiplier * scale
```

Three properties to keep in mind:

- **Disjoint by index.** The first $K_c$ slices of `P_bases` belong to the content pool; the next $K_f$ to the freq pool. OrthoHydra's existing partitioning logic (`ortho.py:137–404`) at `num_experts = E` already produces this layout — chimera just relabels the first $K_c$ as content and the rest as freq.
- **Two folds, one bmm.** The two pools' P_combined tensors are added before the final `bmm`. T-LoRA's content-only mask is folded into `P_combined_c` on the rank axis (broadcast `(1, 1, r)` over `(B, d_out, r)`) so the freq branch keeps full rank without a separate kernel. One bmm per Linear regardless of mask state — shape-static under `torch.compile`.
- **Math-identical to single-pool HydraLoRA.** If you handed the same `cat([π_c, π_f])` gate to plain HydraLoRA at $E = K_c + K_f$, you'd get the same output. The chimera-specific bit is *where the gate halves come from*, not how the kernel consumes them.

---

## 3. The two routers

### 3.1 Content router (per-Linear)

A small `Linear(r → K_c)` per adapted module, identical in shape and policy to HydraLoRA's layer-local router but **narrowed to $K_c$ outputs**. Input is RMS-pooled rank-$r$ post-`lora_down` activation — the same hot-path policy `hydralora.md` §3 justifies (sample-level content survives the 4096-token sequence; rank-$r$ space has no large outliers; softmax stable in bf16). Weight initialized at `std=0.01` so starting gates are near-uniform; bias zero-init.

The content router **only ever sees pooled $lx$**. σ and FEI never reach it — those features are owned by the FreqRouter exclusively. This is the structural specialization guarantee: the content router has no axis on which to express noise-conditioning even if the optimizer wanted it to.

### 3.2 FreqRouter (network-level)

One `Linear → SiLU → Linear` MLP per network. Input is the per-step noise-level feature concatenation:

$$
\text{router}_f\ :\ \mathbb{R}^{F_\text{in}}\ \to\ \mathbb{R}^{K_f},\qquad
F_\text{in}\ =\ \text{fei\_feature\_dim} + \text{sigma\_feature\_dim}
$$

Defaults: `fei_feature_dim = 2` (the 2-band DoG simplex `e_low, e_high` from `library/runtime/fei.py`), `sigma_feature_dim = 16` in the GUI variant but **`sigma_feature_dim = 0` in the canonical bench config** (`configs/methods/chimera.toml`). At the current bench setting the FreqRouter input is therefore **FEI-only** (2-dim) — the sinusoidal-σ slice is wired through `set_fei` but disabled. The σ axis is reachable for the FreqRouter through FEI (which is itself a function of $z_t$ that varies strongly with $\sigma$), so dropping the explicit sinusoidal-σ slice doesn't sever the freq router from noise level, it just removes the redundant direct view.

Output Linear init: `N(0, std=0.1)` — **non-zero is load-bearing**. A zero-init FreqRouter would be a fixed point of the additive composition: uniform $\pi_f$ ⇒ symmetric expert gradients ⇒ the router weights never escape zero. The chimera proposal mandates non-zero output init for exactly this reason (`chimera.py::FreqRouter` class docstring). Unlike `GlobalRouter` for FeRA (which zero-inits to guarantee $\Delta W = 0$ at step 0), the chimera freq pool starts near-uniform but *not at* uniform so FEI variation across the batch immediately writes gradient into the router.

Per step the FreqRouter is fired once by `LoRANetwork.set_fei` (`networks/lora_anima/network.py:1248–1362`) and the resulting `(B, K_f)` tensor is **slot-assigned** into every chimera module's `_freq_routing_weights` buffer. The slot assignment preserves grad_fn so `∂L_denoise/∂π_f` flows back through the buffer to the FreqRouter's parameters — same contract as `GlobalRouter` for FeRA (eq. 6–7, 11 in the FeRA paper).

### 3.3 Why the σ slice is currently off

The bench-line ChimeraHydra entry runs with `sigma_feature_dim = 0`. The proposal's "F ≈ 32" target includes the 16-dim sinusoidal slice, but the bench config disables it to start from the smaller FEI-only input and bring the σ channel back if FEI alone is too narrow a signal for $K_f$-way differentiation. As of writing, no run has demonstrated the FreqRouter is starved without the σ slice — when it does, the toggle to add it back is a single config edit. The code path is exercised every step regardless (see `network.py:1352` — `_sigma_sinusoidal_features(sigma, 0)` returns an empty tensor of shape `(B, 0)` and the cat with FEI produces a `(B, 2)` input).

---

## 4. T-LoRA per-branch composition

When `use_timestep_mask = true` (the default), the rank mask $\text{mask}_t(\sigma)$ from `timestep-mask.md` is applied **to the content branch only**. The freq branch retains full rank at every $t$:

$$
\text{content}\ :\quad P_\text{combined}^{(c)}\ \cdot\ \text{mask}_t\quad\text{(broadcast over rank axis)}\\[2pt]
\text{freq}\ \ \ \ \ :\quad P_\text{combined}^{(f)}\quad\text{(unmasked)}
$$

The proposal's argument: T-LoRA mitigates high-$\sigma$ memorization of layout/identity, which is exactly the content branch's risk surface. The freq branch *wants* high rank at high $\sigma$ to learn coarse-stage features. Per-branch masking is the structural composition that falls out for free because the pools are physically separate.

Implementation note: the mask is folded into `P_combined_c` on the rank axis (`P_combined_c = einsum(...) * mask_t.view(1, 1, -1)`) rather than masking `lx` before the einsum. This lets the two pools' `P_combined` tensors be added before a single `bmm`, saving a kernel launch and an `(B, L, r)` saved-for-backward activation. `_timestep_mask` is always-a-tensor (`base.py`), so the path stays shape-static under `torch.compile` even at the T-LoRA flip points.

---

## 5. Per-pool balance loss

Without pressure, training can collapse one pool entirely while the other concentrates — the standard MoE collapse failure, with a chimera-specific risk that **one pool's collapse is a local minimum for the other's balance**: if the freq pool flattens to uniform, the additive composition reduces to a single-router OrthoHydra, and the content pool will trivially satisfy any single-pool balance constraint.

The fix is **per-pool balance loss**:

$$
\mathcal{L}_\text{balance}\ =\ w_c \cdot K_c \cdot \sum_{i=1}^{K_c} f_i^{(c)}\, \bar{g}_i^{(c)}\ +\ w_f \cdot K_f \cdot \sum_{j=1}^{K_f} f_j^{(f)}\, \bar{g}_j^{(f)}
$$

Each pool gets its own Switch-Transformer-style coefficient. The accumulator (`networks/lora_anima/network.py:_get_chimera_balance_loss`) splits each module's cached `_last_gate` at index $K_c$ and runs the two halves through independent balance terms. Defaults: `balance_w_content = 2e-5` (matches the `[[project_hydra_balance_weight_ceiling]]` safe range), `balance_w_freq = 2e-5`. The outer `balance_loss_weight` multiplier stays at `1.0` so the per-pool weights are the only effective scalars.

A single combined balance term would not work: the optimizer could trivially satisfy it by flattening one pool to uniform (which makes that pool's contribution to the term vanish) while the other concentrates. Two independent terms force pressure on both axes simultaneously.

---

## 6. Cold-start risk and diagnostics

Two random-init routers ⇒ risk one settles into a usable distribution while the other oscillates near uniform and never wakes up. Three structural mitigations:

1. **Per-pool balance loss** (§5).
2. **Non-zero FreqRouter init** (§3.2): `freq_router_init_std = 0.1`. Output near-uniform but not at uniform ⇒ FEI variation across the batch immediately differentiates.
3. **Forced FEI-pipeline activation**: `cfg.use_chimera_hydra = True` sets `use_fei_router = True` in `LoRANetwork.__init__` regardless of `cfg.router_source`, so `apply_router_conditioning` fires `set_sigma → set_fei` every step. Without this, an off-by-default FEI pipeline would never propagate to the FreqRouter.

The live diagnostic is per-pool normalized gate entropy in the first 1k steps. The chimera-aware path runs through `get_chimera_router_stats` (separate from the standard `get_router_stats` because per-pool entropy normalizes by `log(K_pool)`, not `log(E)`). Freq-pool entropy persistently $> 0.998$ after warmup ⇒ the FreqRouter has no signal the content router didn't already capture via $lx$-σ correlation — the "freq pool redundant" failure mode the proposal's `C-fei` falsification cell is designed to catch.

---

## 7. File format — save distills, load re-hydrates

Save (`networks/lora_save.py::chimera_hydra_moe`) runs the OrthoHydra Cayley → Hydra distillation (`_convert_ortho_hydra_to_hydra`: fold $(S_p, S_q, P_\text{bases}, Q_\text{basis}, \lambda)$ into shared `lora_down` + per-expert `lora_ups.{i}`), then defuses the fused attention projections (`_build_hydra_moe_state_dict`: split `qkv_proj` / `kv_proj` into per-component `q/k/v_proj`, cloning shared `lora_down`/`alpha`/`router.*` into each split). The expert axis runs `[content_0 … content_{K_c-1} | freq_0 … freq_{K_f-1}]`. Top-level `freq_router.*` keys flow through both conversion steps unchanged.

The on-disk layout matches the existing HydraLoRA MoE keyspace exactly. The only chimera-specific bits are:

- `router.weight` is $K_c$-narrowed instead of $E$-wide.
- A top-level `freq_router.{net.0,net.2}.{weight,bias}` block.
- `ss_use_chimera_hydra = "true"` metadata stamp, plus per-pool sizes and feature dims.

Load (`library/inference/models.py::_is_chimera_moe` → `networks/lora_anima/factory.py`) sniffs `ss_use_chimera_hydra` from metadata, then **overrides** `module_class = HydraLoRAModule` (instead of the Cayley `ChimeraHydraLoRAModule` used at training). The runtime form is `HydraLoRAModule` with `num_experts_content > 0` set: its router is narrowed to $K_c$ and it registers a `_freq_routing_weights` buffer for the network's FreqRouter to write into. The Cayley class is therefore **training-only** — checkpoint resume silently drops the orthogonal parameterization and continues on the distilled form, matching the OrthoHydra → Hydra precedent.

This dual-pool runtime form is detected purely by the metadata stamp: nothing in the safetensors key layout distinguishes a chimera file from a standard hydra-MoE file. The `router.weight.shape[0] = K_c < E` is the only structural difference, and even that is only interpretable in conjunction with `ss_num_experts_content`. The ComfyUI `comfyui-hydralora` node uses the same metadata sniff.

---

## 8. Compile friendliness

Two einsum folds + a single `bmm` per Linear. T-LoRA's mask is folded into `P_combined_c` rather than masking `lx`, so both pools share the same `(B, L, r) @ (B, r, d_out)` `bmm` regardless of mask state. The freq routing buffer is shape-`(B, K_f)`, slot-assigned per step (not in-place copied) — the pointer identity changes but the shape doesn't, so dynamo doesn't recompile. Standard chimera training runs at the same compile budget as OrthoHydra at `num_experts = E`.

`set_fei` fires the FreqRouter **with grad** (not inside `torch.no_grad`) so the autograd path `L_denoise → out_f → π_f → FreqRouter.params` reaches the router parameters through the slot-assigned `_freq_routing_weights` buffer. This matches the FeRA `GlobalRouter` contract (`[[project_fera_router_gradient_path]]`).

---

## 9. Composition

| Stacks with             | How it composes                                                                                                |
| ----------------------- | -------------------------------------------------------------------------------------------------------------- |
| **T-LoRA**              | Per-branch — mask on content, full rank on freq. Built-in (§4).                                                |
| **OrthoLoRA**           | `use_ortho = true` is the chimera default. Both pools share the Cayley parameterization on the OrthoHydra basis. |
| **ReFT**                | Designed against shared-A / plain-LoRA layouts. Verify on small bench before stacking.                          |
| **Spectrum**            | Cached steps skip transformer blocks → the FreqRouter doesn't fire on those steps. Same caveat as FeRA.        |
| **Modulation guidance** | Orthogonal. Touches AdaLN only.                                                                                |
| **DCW (scalar / v4)**   | Orthogonal. Sampler-level correction.                                                                          |
| **Static merge to DiT** | ❌ MoE — sample-dependent gates can't be folded into a Linear weight.                                          |
| **FeRA in same ckpt**   | ❌ One MoE scheme per checkpoint.                                                                              |

---

## 10. Configuration

`configs/methods/chimera.toml` (canonical bench, `make exp-chimera`):

```toml
use_chimera_hydra = true
num_experts_content = 4
num_experts_freq = 2

# Per-pool balance — independent of the outer balance multiplier.
balance_loss_weight = 1.0
balance_w_content = 2e-7
balance_w_freq = 0

# FreqRouter input. σ slice is currently off — FreqRouter sees FEI(2) only.
fei_feature_dim = 2
fei_sigma_low_div = 4.0
sigma_feature_dim = 0          # ← wired but disabled in the bench config

freq_router_init_std = 0.1     # non-zero is load-bearing (§3.2)

# Cayley on the unrouted leg; T-LoRA on the content branch.
use_ortho = true
use_timestep_mask = true
min_rank = 8
```

`configs/gui-methods/chimera_hydra.toml` is the GUI-friendly variant — same activation flag, but `K_c = K_f = 3`, `balance_w_freq = 2e-5`, and `sigma_feature_dim = 16` (the σ slice is on in the GUI variant). Pick whichever matches your run.

The three-axis fields (`use_moe_style` / `route_per_layer` / `router_source`) are **auto-pinned** to `("shared_A", true, "input")` by `LoRANetworkCfg.from_kwargs` whenever `use_chimera_hydra = true`. Passing any other value for those fields raises — the chimera flag is the only routing knob you set.

---

## 11. Minimal mental model

1. OrthoHydra with the $E$ experts relabeled into two disjoint pools by index: first $K_c$ are content, next $K_f$ are freq.
2. Two routers feed disjoint halves of the gate vector. Content router is per-Linear (sees pooled rank-$r$ `lx`); freq router is network-level (sees FEI of $z_t$, optionally + sinusoidal-σ; σ slice off in the current bench).
3. Two `einsum` folds, two `P_combined` tensors, summed; one `bmm` produces the final delta.
4. T-LoRA mask on content only; freq branch always full rank.
5. Per-pool balance loss ($w_c$, $w_f$ independent) prevents one-pool collapse.
6. Cayley at training, distilled to standard Hydra-MoE layout at save. Loaded as `HydraLoRAModule(num_experts_content > 0)` + a top-level FreqRouter — metadata stamp `ss_use_chimera_hydra` is the only thing that distinguishes a chimera file from a stock hydra-MoE file on disk.
