# Soft Tokens — per-layer × per-t learnable text tokens (SoftREPA)

Per-layer, time-indexed soft tokens in T5-compatible space. DiT is frozen. ~1M trainable params at default config (n_layers=10, K=4, D=1024, n_t_buckets=100). Each of the first `n_layers` DiT blocks gets its own learned (K, D) token bank plus a per-(t-bucket, layer) D-vector offset, spliced into `crossattn_emb` for that block alone. Trained with plain FM **plus** the paper's InfoNCE contrastive objective on diffusion-loss logits, blended at a small weight.

Reference: Lee et al., *Aligning Text to Image in Diffusion Models is Easier Than You Think* (arXiv:2503.08250, NeurIPS 2025) — "SoftREPA". We adopt the full methodology — both the per-layer × per-t parameterization and the InfoNCE objective — but blend the contrastive term as a small-λ regularizer on top of plain FM (paper used pure contrastive, which produced an SD3 FID regression — see §"Contrastive integration" below).

## Quick start

```bash
make exp-soft-tokens                    # default preset
python tasks.py exp-soft-tokens         # cross-platform
```

v1 is **training-only**. `inference.py` cannot load these checkpoints — `create_network_from_weights(for_inference=True)` raises `NotImplementedError`. Wire up the per-step block hook inside the denoising loop if early training looks promising.

## What it is

For each block `k ∈ [0, n_layers)`, the cross-attention input is replaced by a layer-specific variant:

```
s^(k, t)         = tokens[k] + t_offsets[bucket(t), k]      # shape (K, D)
crossattn_emb_k  = splice(crossattn_emb, s^(k, t))
block_k(x, ..., crossattn_emb_k)                            # original block, modified text input
```

`tokens ∈ ℝ^(n_layers × K × D)` is the base bank; `t_offsets ∈ ℝ^(n_t_buckets × n_layers × D)` is a per-(bucket, layer) D-vector broadcast across the K-token axis. Zero-init on `t_offsets` means at step 0 the layer banks reduce to their base values — no time conditioning until gradients learn it.

```
                       Soft Tokens
              ┌─────────────────────────────┐
              │  DiT Block 0                │
              │  ┌─────────────┐            │
crossattn ────┼─►│ +s^(0,t)    │  cross    │
   (B,S,D)    │  │   spliced   │  attn ──► │ ──► x'
              │  └─────────────┘            │
              ├─────────────────────────────┤
              │  DiT Block 1                │
              │  ┌─────────────┐            │
crossattn ────┼─►│ +s^(1,t)    │  cross    │
   (B,S,D)    │  │   spliced   │  attn ──► │ ──► x''
              │  └─────────────┘            │
              ├─────────────────────────────┤
              │  ...                        │
              ├─────────────────────────────┤
              │  DiT Block (n_layers..N-1)  │  no splice — block sees
              │  cross-attn (unmodified)    │  the original crossattn
              └─────────────────────────────┘
```

The crossattn passed in is unchanged across blocks (Anima is **not** joint-stream MM-DiT — text features don't evolve through blocks). Each block independently sees a different splice; no strip/re-prepend dance.

## Parameter count

```
n_layers · K · D       (base tokens)
+ n_t_buckets · n_layers · D   (t-offsets, broadcast across K)
```

Defaults: 10 · 4 · 1024 + 100 · 10 · 1024 ≈ 41k + 1.05M ≈ **1.05M params**. 30–60× lighter than a typical LoRA.

## Implementation map

| File | Role |
|------|------|
| `networks/methods/soft_tokens.py` | `SoftTokensNetwork` — per-(layer, t) token bank, splice hook, save/load. `SoftTokensMethodAdapter` — paper InfoNCE forwards (k extra DiT calls with rolled text). |
| `library/training/losses.py::_soft_tokens_contrastive_loss` | Loss-registry handler that multiplies the adapter's stashed contrastive scalar by `network.contrastive_weight`. Active in `_STAGE_SCALAR_BROADCAST` when method is `soft_tokens` and `contrastive_weight > 0`. |
| `apply_to(text_encoders, unet)` | Walks `unet.blocks[:n_layers]`, replaces each `block.forward` with a wrapper that splices `s^(k, t)` into `crossattn_emb` before calling the original (ReFT-pattern monkey-patch). |
| `append_postfix(crossattn_emb, seqlens, timesteps)` | Receives `timesteps` from `train.py`'s existing per-step hook; computes `(n_layers, B, K, D)` step-scoped tokens and caches them on the network. **Returns `crossattn_emb` unchanged** — splicing happens inside the block hooks. |
| `_make_block_hook(layer_idx, org_forward)` | Closure that reads the cached step tokens at `layer_idx`, splices into `crossattn_emb`, calls the original block forward. |
| `configs/methods/soft_tokens.toml` | Default config (n_layers=10, K=4, n_t_buckets=100, splice_position=end_of_sequence, lr=1e-3, 4 epochs). |
| `configs/gui-methods/soft_tokens.toml` | Sibling for `make lora-gui GUI_PRESETS=soft_tokens`. |
| `scripts/experimental_tasks/training.py::cmd_soft_tokens` | Task entry-point. |
| `tasks.py` `exp-soft-tokens` | Make/CLI registration. |

## Splice position

Two options, mirroring postfix:

| Mode | Where | Trade-off |
|---|---|---|
| `end_of_sequence` (default) | overwrite the K tail slots `[S-K, S)` of the zero-padding region | Static splice index → maximally compile-friendly. Caption-position-agnostic. Preserves the strongest front-of-padding attention sinks intact. |
| `front_of_padding` | place K tokens at `[seqlens[i], seqlens[i]+K)` per sample (`scatter`) | Caption-position-aware. Displaces the strongest sinks. Per-sample variable indices via the cached `crossattn_seqlens`. Incompatible with `trim_crossattn_kv=true`. |

Toggle via `network_args = ["splice_position=front_of_padding"]`. The choice is metadata-tagged (`ss_splice_position`) so checkpoints round-trip with the right splice mode.

Anima's text-encoder padding invariant (zero-padded positions act as cross-attention sinks) means writing into the padded tail is *not* a no-op — those slots receive attention mass and the soft tokens get exposure to every spatial query. See the "Text encoder padding" note in the root CLAUDE.md.

## Contrastive integration

The paper's InfoNCE objective (§3.1, eq. 13–14) is wired up as `SoftTokensMethodAdapter`. For each anchor `i`, the adapter runs `k` extra DiT forwards with text rolled by `j ∈ {1..k}` along the batch axis — same `(x_t, ε, t)`, varying `y` — and builds a `(1+k)`-way softmax over negative diffusion-loss logits:

```
l(x, y) = exp(−‖v_θ(x_t, t, y, s) − (ε − x_0)‖² / τ)
L_contrastive = −E_i[ log( exp(l(x_i, y_i)) / Σ_j∈{i,(i+1),..,(i+k)} exp(l(x_i, y_j)) ) ]
L_total = L_FM(matched) + λ · L_contrastive
```

The soft tokens themselves don't depend on text content, so the same `_step_layer_tokens` cached during the matched forward is reused across negative forwards (the per-block hook re-splices into the rolled crossattn). Per-sample seqlens roll with the text.

**Why hybrid (`L_FM + λ · L_contrastive`) rather than pure contrastive (paper):**
- The paper showed an SD3 FID regression at paper-strength contrastive (FID 31.59 → 36.21 on COCO val5K) while ImageReward gained ~14%. SD3 is the closest analogue to Anima.
- Anima's caption distribution (booru-style tags / structured prompts) is narrower than COCO's natural-language captions, so in-batch negatives are weaker (more semantic overlap). Pure contrastive on weak negatives over-tunes the discriminative signal at the cost of fidelity.
- Keeping plain FM at full strength preserves fidelity; the contrastive term acts as a small-λ regularizer that pushes layer-tokens to make matched (x, y) pairs more discriminable than mismatched ones — the load-bearing signal that drives **per-layer specialization**.

**Cost.** Each step does `1 + min(k, B-1)` DiT forwards instead of 1. At default `k=1`, batch-size 2+: 2× FM-only forward cost. At `k=2`, batch-size 3+: 3×. Memory scales linearly because each negative forward keeps activations live until backward (we don't `wants_split_backward` — all `(1+k)` candidates contribute to a single InfoNCE loss).

**Splice constraint.** The contrastive path currently requires `splice_position=end_of_sequence`. The rolled-text trick reconstructs the pre-splice tensor from the post-splice tensor by zeroing the K tail slots, which works for EOS (constant tail) but not FOP (per-sample variable splice indices, scatter is not trivially invertible). FOP-with-contrastive is a TODO.

**Disabling contrastive.** Set `contrastive_weight=0` in `network_args` to drop back to plain FM with just the parameterization. The adapter short-circuits at `weight ≤ 0` and the loss composer omits the term.

## Why a separate module from `postfix.py`

Postfix splices **once** at the cached adapter output (training-time and inference-time, in `train.py:762` and `library/inference/generation.py`). Soft tokens splice **per-block** via a monkey-patched `Block.forward`. Different surface entirely — keeping them separate avoids muddying the postfix abstraction. Both modules expose `append_postfix(...)` so `train.py`'s existing per-step trainer hook routes timesteps to either family without code changes.

## Why no slot-collapse

The existing postfix module logs an aggressive guard against K-slot permutation symmetry collapse (`anima_postfix.safetensors` was effectively K=1 due to zero-init + symmetric splice — see the postfix module docstring and the `slot_embed_init_std` knob). Soft tokens **structurally avoid** this: tokens at different `(k, t)` pairs are consumed at different positions in the network and gradients differ from step 1, so no symmetry to break.

## Compatibility

| Component | Compat | Notes |
|---|---|---|
| Training loop | ✅ | `train.py` already passes `timesteps=...` into `append_postfix` (legacy `cond-timestep` postfix mode); soft tokens piggyback on the same hook. |
| Standard inference | ❌ v1 | `for_inference=True` raises `NotImplementedError`. Per-step block hook would need to fire inside `library/inference/generation.py::generate_body`. |
| Spectrum inference | ❌ v1 | Same blocker as standard inference — Spectrum's actual-step forwards would need the per-step hook too. |
| `torch.compile` (`_run_blocks`) | ✅ | `end_of_sequence` keeps `crossattn_emb` shape static; the cached `_step_layer_tokens` is read as a runtime tensor with static shape. `front_of_padding` uses `scatter` with dynamic per-sample indices but static buffer shape — also compile-clean. |
| `blocks_to_swap` | ❌ method-forced 0 | The hook captures each `Block` by reference at `apply_to()` time; a swapped block is a different object instance, so the hook would fire on the wrong tensor. |
| `gradient_checkpointing` | ✅ | The hook is the outermost wrapper; the original `forward` (which itself runs `checkpoint(_forward, ...)`) is called underneath, and the spliced `crossattn_emb` is part of the saved input graph. |
| Modulation guidance | ✅ orthogonal | Modulation = per-block AdaLN path; soft tokens = K/V input path per block. |
| T-LoRA / OrthoLoRA / ReFT | n/a | Soft tokens freeze the DiT; LoRA-family methods are not stacked in this config. |
| `trim_crossattn_kv=true` | ⚠ EOS only | Soft tokens overwrite the zero-padding tail (no seqlen change), so trim's KV slice is correct. With `front_of_padding` the tokens land *inside* what trim treats as real-text region; leave trim off. |

## Evaluation

What to measure to know if this is doing anything:

1. **`|t_offsets|` at convergence as a function of bucket**: flat/near-zero → time conditioning collapsed (the per-layer base tokens absorbed everything; SoftREPA's `use_dc_t=True` won't be load-bearing). Curve should grow away from zero, ideally with structure across t.
2. **Per-layer token norm**: `‖tokens[k]‖` should differ across `k`. If they converge to a single shared bank, we're effectively running a single-layer postfix and the per-layer parameterization is dead weight.
3. **Held-out prompt-following**: this is the load-bearing question. The existing DCW v4 calibrator targets the same axis (text-image alignment, prompt-following) but at inference time. If soft tokens move the same metrics, they're a training-time alternative. If not, they're parameter overhead.
4. **Anatomy / style breakdown**: REPA helped anatomy on Anima but broke anime style (vision-encoder photo-prior leak). Soft tokens have no external visual prior, so the failure mode shouldn't recur — but they also can't reproduce the anatomy gain. The plausible win is text alignment, not structural quality. If anatomy *also* improves, that's a surprise worth tracking.
5. **Splice position A/B**: `end_of_sequence` vs `front_of_padding`. Front-of-padding displaces the strongest sinks and might give the tokens more attention mass at the cost of disturbing what the pretrained model relies on. Worth a short bench before committing to a default.

## Hyperparameters worth sweeping

| Knob | Default | Range to try | Why |
|---|---|---|---|
| `n_layers` | 10 | 5, 10, 14, 28 | SoftREPA used 5/24 layers on SD3. 10/28 is proportional. Going to 28 (all blocks) doubles params and tests whether deep-block tokens do anything. |
| `network_dim` (K) | 4 | 1, 4, 8, 16 | SoftREPA used m=4 on SD3. K=1 collapses to "per-layer prefix vector" — clean ablation. |
| `n_t_buckets` | 100 | 0 (disable t-cond), 20, 100 | Setting `t_offsets.weight.requires_grad_(False)` is a clean ablation for whether time conditioning is load-bearing. |
| `init_std` | 0.02 | 0.0, 0.02, 0.1 | Zero-init = strict identity at step 0 (block sees zeroed padding tail). 0.02 = small perturbation. 0.1 = aggressive. |
| `splice_position` | `end_of_sequence` | both | See §"Splice position" above. FOP currently incompatible with contrastive. |
| `learning_rate` | 1e-3 | 1e-4 to 5e-3 | Soft tokens are tiny + zero-inited offsets; high LR is fine. |
| `contrastive_weight` | 0.05 | 0, 0.01, 0.05, 0.1, 0.5 | Paper-strength is closer to 1.0 (pure contrastive); we default low to preserve fidelity. 0 disables contrastive entirely (parameterization-only ablation). |
| `contrastive_k` | 1 | 1, 2, 3 | Each k-increment adds one DiT forward per step. Capped at `batch_size - 1` at runtime. |
| `contrastive_tau` | 1.0 | 0.5, 1.0, 2.0 | Temperature in `exp(−MSE / τ)`. Smaller τ → sharper logits, stronger contrastive pressure. |
