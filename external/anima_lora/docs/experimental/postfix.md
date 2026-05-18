# Postfix (cond+ortho)

Caption-conditional postfix with structural orthogonality. The DiT runs frozen; only `cond_mlp` trains. `K` continuous vectors are computed per-caption from the pooled cached adapter output and spliced into the cross-attention input (replacing zero-padding slots) — extra "prompt tokens" the DiT attends to alongside the real caption.

The toggle-block path (`make exp-postfix`) and the clean per-variant path (`make lora-gui GUI_PRESETS=postfix_ortho_cond`) both resolve to this mode. Earlier `postfix` / `prefix` / `cond-timestep` modes are still in `networks/methods/postfix.py` but are not the shipping path and aren't documented here.

## What's actually computed

```
crossattn_emb  (B, S, D)
       │
       ├─ maxabs-pool over content slots ─►  pooled (B, D)
       │                                        │
       │                                        ▼
       │                       LayerNorm(D) → Linear → GELU → Linear
       │                                        │
       │                                        ▼
       │                          cond_out (B, K(K-1)/2 + 1)
       │                                        │
       │       split: skew-seed S(c) ∈ ℝ^{K×K}_skew, λ(c) ∈ ℝ
       │                                        │
       │                  R(c) = Cayley(S(c) − S(c)ᵀ) ∈ O(K)
       │                                        │
       │                                        ▼
       │             postfix(c) = R(c) · basis · λ(c)
       │                       (B, K, D)
       │                                        │
       └────────────── splice ◄─────────────────┘
                  (overwrites zero-padding region;
                   end_of_sequence or front_of_padding)
```

Per caption,

    postfix(c) · postfix(c)ᵀ = λ(c)² · I_K

exactly — K orthogonal directions of uniform magnitude. The optimizer cannot collapse the K-rank capacity onto one slot; `λ(c)` is the only knob it has to scale the whole bundle up or down per caption.

### Three load-bearing choices

These are not stylistic — each fixes a concrete failure observed earlier in the family:

- **Frozen row-orthonormal basis.** The trainable surface is `cond_mlp` only (the basis is computed once and `register_buffer`-frozen, persisted at fp32 inside the safetensors). This collapses the parameter count from `K·D` to `K(K-1)/2 + 1` per-caption rotation/scale scalars and structurally guarantees orthogonality at every step, replacing earlier soft regularizers that didn't hold up.
- **Maxabs pooling over content slots.** Mean-pool drags every caption onto the T5 corpus DC offset (`bench 20260511-1004`: cross-caption cos μ ≈ 0.84, peaking at 0.9986 after the first Linear). Maxabs picks per-channel the token with largest `|·|` (sign preserved, padding set to `-1` so it can't win), which keeps both polarities of caption-distinct deflections. With maxabs the same diagnostic dropped to cos μ ≈ 0.865.
- **LayerNorm on the pooled input.** Strips the residual DC offset before the first `Linear` so caption deltas survive instead of being projected as a shared mean across hidden units. γ=1, β=0 init keeps the rest of `cond_mlp`'s zero-init behavior intact (the final `Linear` still starts at zero, so step 0 is `S(c)=0, λ(c)=λ_init`).

### Why `lambda_init = 0.3`

The final `Linear` of `cond_mlp` is zero-init, so without intervention `λ(c)` starts at exactly 0 and the postfix is empty at step 0. Earlier runs (`bench/postfix_ortho/results/20260511-1622-cond-v2-ln-final`) saw `λ(c)` mean drift from 0.50 (epoch 1) down to 0.034 (epoch 2 final) — the network never had an amplitude to *defend*, only one to *grow from zero*, and any L2/weight-decay pressure shrinks it back. Biasing the last-layer `λ(c)` bias to ~0.3 gives the optimizer a non-trivial starting magnitude to either keep using or actively kill. The skew-seed bias entries stay zero (so `S(c)=0 → R=I → postfix(c) = basis · λ_init` at step 0).

### Why no per-slot positional bias

Splice positions sit on zero-padding slots that carry no positional encoding into the DiT's cross-attention K-projection — so in principle the K postfix tokens could be permutation-symmetric in softmax (proposal §B). An earlier v3 added a `slot_pos` parameter to break this symmetry; it grew to dominate (`slot_pos / postfix` norm ratio 1.82) and crushed caption diversity back to cos μ=0.994. It also produced a cudagraph aliasing failure under the compiled hot path. In practice the K Cayley-rotated SVD basis rows already produce K distinct cross-attention keys from content alone (`bench 20260511-1622-cond-v2-ln-final` preserved cos μ=0.39 across captions with no `slot_pos`), so the parameter was removed. Legacy v3/v4 checkpoints load with `slot_pos` ignored and a warning.

## Knobs

From `configs/methods/postfix.toml`:

| Param | Default | Notes |
|-------|---------|-------|
| `network_dim` (K) | 48 | Postfix slots. K ≤ D required. |
| `mode` | `cond` | cond+ortho is selected by `mode=cond` + `ortho=true`. |
| `ortho` | true | Switches in the Cayley + frozen-basis path. |
| `ortho_basis` | `svd_te` | `random` (QR on Gaussian) or `svd_te` (top-K right singular vectors of cached T5 outputs, row-shuffled by seed). |
| `te_cache_dir` | `post_image_dataset/lora` | Where `svd_te` reads `_anima_te.safetensors` from. |
| `svd_num_files` | 1024 | Files sampled to build the SVD basis. |
| `ortho_basis_seed` | 0 | Deterministic row-shuffle of the top-K singular vectors. Breaks "slot 0 = principal direction" inductive bias the same way OrthoHydra's `e mod B` interleaves bands. |
| `cond_hidden_dim` | 1024 | MLP hidden width. Matched to D so the first `Linear` isn't a bottleneck against the `K(K-1)/2 + 1` output. |
| `splice_position` | `front_of_padding` | `end_of_sequence` places at `[S-K, S)` (caption-position-agnostic); `front_of_padding` places at `[seqlens[i], seqlens[i]+K)` (caption-position-aware). |
| `lambda_init` | 0.3 | Bias on the `λ(c)` output channel so step 0 has non-trivial magnitude. |

`learning_rate = 5e-4`, `max_train_epochs = 2`, `blocks_to_swap = 0`, `cache_llm_adapter_outputs = true`. Compile is on (`compile_mode = "full"`, `compile_inductor_mode = "reduce-overhead"`); `compile_hot_path` wraps `_compute_ortho_cond_postfix` (the `cond_mlp` + Cayley + matmul + cast sequence) into a single shape-static graph at fixed K, D, B.

The GUI variant (`configs/gui-methods/postfix_ortho_cond.toml`) ships a smaller K=32, `cond_hidden=256`, `end_of_sequence` splice, no `lambda_init` (defaults to 0). It's a separate experiment point, not a recommended override of the toggle-block defaults.

## Inference

```bash
python inference.py \
    --postfix_weight output/anima_postfix_ortho_v4.safetensors \
    --prompt "..." \
    ...
```

Or `make exp-test-postfix` against the most recent `output/anima_postfix*.safetensors`. The save format embeds `ss_mode`, `ss_ortho`, `ss_ortho_basis`, and `ss_lambda_init` in safetensors metadata; `create_network_from_weights` re-derives K and D from `ortho_basis.shape`, so the loader doesn't need the original training config.

The basis is saved at fp32 regardless of training dtype — bf16 truncation of basis entries breaks the `‖postfix · postfix.ᵀ − λ² · I_K‖_F < 1e-4` gate on round-trip.

## Files

- `networks/methods/postfix.py` — `PostfixNetwork` (cond+ortho branch + `_compute_ortho_cond_postfix` compile target), save/load, metadata.
- `configs/methods/postfix.toml` — toggle-block path (`make exp-postfix`).
- `configs/gui-methods/postfix_ortho_cond.toml` — clean per-variant config.
- `docs/proposal/orthogonal_postfix.md` — design rationale and bench history.
- `archive/bench/postfix/` — diagnostic scripts (`analyze_cond_postfix.py`, `analyze_ortho_postfix.py`) and `initial_postfix_problems.md`.
