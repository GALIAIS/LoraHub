# LoRA Research Workspace

This folder is for LoraHub-owned LoRA algorithm experiments for the modified
`anima_lora` backend. Code here is not part of the production network registry
until it has a measured gain and a migration path.

## Papers Read

Primary sources used before changing this research plan:

- LoRA: freezes base weights and trains low-rank update matrices, motivated by
  rank deficiency in downstream adaptation.
  https://arxiv.org/abs/2106.09685
- AdaLoRA: fixed equal rank per matrix is suboptimal; dynamically allocate rank
  budget by importance.
  https://arxiv.org/abs/2303.10512
- DyLoRA: trains LoRA blocks so multiple ranks can be used without retraining.
  https://arxiv.org/abs/2210.07558
- DoRA: separates weight magnitude and direction; improves LoRA capacity and
  stability while keeping inference overhead low.
  https://arxiv.org/abs/2402.09353
- SVDiff: diffusion personalization can reduce overfitting by training a compact
  singular-value parameter space instead of broad weight updates.
  https://arxiv.org/abs/2303.11305
- T-LoRA: diffusion personalization overfits more at higher/noisier timesteps;
  timestep-dependent rank masking and orthogonal initialization improve concept
  fidelity/text alignment.
  https://arxiv.org/abs/2507.05964
- LyCORIS: a Stable Diffusion LoRA extension library/paper grouping several
  decomposition methods beyond conventional LoRA, including LoHa, LoKr, IA3,
  DyLoRA and GLoRA-style variants.
  https://arxiv.org/abs/2309.14859
- FedPara / LoHa basis: Hadamard product of low-rank factors increases
  effective expressivity while keeping parameter count controlled.
  https://arxiv.org/abs/2108.06098

## LyCORIS Notes

LyCORIS is relevant here because it is not just another LoRA name. It is a
collection of SD-oriented parameter-efficient adapter variants with practical
save/load conventions. The current `anima_lora` fork already mirrors much of
that surface:

- `algorithm=lycoris_loha` / `use_loha=true` maps to `LoHAModule`.
- `algorithm=lycoris_lokr` / `use_lokr=true` maps to `LoKrModule`.
- `algorithm=lycoris_ia3` / `use_ia3=true` maps to `IA3Module`.
- `algorithm=lycoris_dylora` / `use_dylora=true` maps to `DyLoRAModule`.
- `algorithm=lycoris_glora` / `use_glora=true` maps to `GLoRAModule`.
- `algorithm=lycoris_diag_oft` / `use_diag_oft=true` maps to `DiagOFTModule`.
- `algorithm=lycoris_boft` / `use_boft=true` maps to `BOFTModule`.

What the paper/library changes for our research plan:

- Do not invent another decomposition first. LoHa/LoKr/OFT already cover the
  main "more expressive than plain LoRA" axis.
- Prefer checkpoint-compatible improvements around rank scheduling,
  initialization, loss regularization, and routing before adding a new save
  layout.
- Treat LyCORIS variants as baselines, not just options. Any new algorithm must
  beat at least current `tlora`, `loha`, and `lokr` on the same dataset seed.
- Keep anime/style LoRA evaluation practical: preview quality, prompt
  adherence, overfit signs, and load compatibility matter more than a lower
  training loss alone.

Current local implication:

- ASR-LoRA remains the first patch because it improves the local T-LoRA
  timestep-rank implementation without competing with LyCORIS decomposition
  variants.
- Spectral-Stable LoRA should be tested against LoHa/LoKr. If LoHa/LoKr already
  solve the collapse on the target dataset, skip the new regularizer.
- Layerwise rank budget should reuse LyCORIS-compatible checkpoints. Static
  layer multipliers are acceptable only if the saved tensor shapes stay the
  same.

## Current Baseline

The production path already has:

- Classic LoRA: `lora_down -> lora_up`, zero-init up, standard save layout.
- OrthoLoRA: Cayley/SVD-style orthogonal low-rank update, distilled back to
  standard LoRA keys on save.
- T-LoRA: timestep-dependent rank mask, currently one shared mask per batch.
- DoRA, DyLoRA, GLoRA, IA3, LoKr, LoHA, OFT/BOFT, VeRA, Hydra/Chimera/FeRA.
- Stability patches for Anima DiT fp16: LoRA/LoHA fp32 islands and DiT block
  fp32 islands on V100.

That means new work should not re-create another plain low-rank adapter.

The first useful target is the local T-LoRA implementation. The T-LoRA paper's
central observation is timestep-dependent overfitting: noisy/high timesteps
should receive less rank capacity than clean/low timesteps. The current code
does this with a single mask computed from the batch mean timestep. That is a
cheap approximation, but it throws away per-sample timestep information when a
batch mixes noise levels.

## Research Track 1: ASR-LoRA

Working name: Adaptive Sigma-Rank LoRA.

Goal: keep the standard LoRA checkpoint format while making T-LoRA rank usage
depend on each sample's timestep/noise level instead of the batch mean.

Forward form:

```text
y = W0(x) + up(mask(t) * down(x)) * scale
```

Where `mask(t)` is per sample:

```text
r(t) = min_rank + ((max_t - t) / max_t)^alpha * (rank - min_rank)
mask[b, i] = i < r(t_b)
```

Shape rules:

- Linear `(B, T, R)` bottleneck: mask is `(B, 1, R)`.
- Linear `(B, R)` bottleneck: mask is `(B, R)`.
- Conv `(B, R, H, W)` bottleneck: mask is `(B, R, 1, 1)`.

Paper basis:

- From T-LoRA: rank should shrink at noisy timesteps to reduce overfitting.
- From DyLoRA: dynamic rank should preserve deployability over a rank range.
- From AdaLoRA: fixed equal rank is not always the right allocation.

Local extension:

- T-LoRA describes timestep-dependent rank masking. This research step narrows
  the local implementation gap by replacing batch-mean masking with per-sample
  masking. It is not a new checkpoint format.

Why this is the first experiment:

- It keeps existing LoRA weights and save format.
- It does not add routers or new trainable parameters.
- It directly fixes information loss in current local T-LoRA.
- It can be toggled off without changing old checkpoints.
- It is cheaper and less invasive than AdaLoRA-style importance scoring.

Initial code:

- `rank_mask.py::rank_budget`
- `rank_mask.py::per_sample_rank_mask`

Production integration, if the test wins:

1. Add `per_sample_timestep_mask=false` as a network arg.
2. Update `LoRANetwork.set_timestep_mask()` to build `(B, 1, R)` masks when
   enabled; keep the current shared `(1, R)` mask as default.
3. Update LoRA-family forwards to accept either shared or per-sample masks.
4. Add compiler/UI surface only after a smoke test shows no regression.

Do not add:

- No new trainable router.
- No per-layer rank search loop.
- No new save format.

## Research Track 2: Layerwise Rank Budget

Problem: every adapted block currently receives the same rank schedule. Style
LoRA often needs more capacity in middle/high-level blocks and less in early
projection layers.

Minimal version:

```text
effective_rank(layer, t) = rank_mask(t) * layer_multiplier[layer]
```

No new parameters. Multipliers can start from a static map:

- self-attn qkv: 0.75
- cross-attn q/k/v/out: 1.0
- FFN projections: 0.5

Paper basis:

- AdaLoRA shows rank budget should differ by weight importance.
- Custom Diffusion shows diffusion personalization can work by focusing a small
  subset of attention parameters rather than broadly updating everything.

Local extension:

- Start with static multipliers derived from layer names, not learned pruning.
  Only move to importance scoring if the static map fails.

Do not implement until ASR-LoRA has a baseline result.

## Research Track 3: Spectral-Stable LoRA

Problem: style LoRA overfits by letting a few rank directions dominate. Current
channel scaling and SVD init help, but there is no direct constraint on rank
direction collapse.

Minimal version:

```text
loss += lambda * ||normalize(A A^T) - I||_F
```

Only apply to `lora_down` of Linear layers. This should be a loss add-on, not a
new checkpoint format.

Paper basis:

- T-LoRA uses orthogonal initialization to make adapter components more
  independent.
- SVDiff reduces diffusion overfitting by operating in a compact singular-value
  space.
- DoRA's direction/magnitude split is another signal that update geometry
  matters, not just parameter count.

Local extension:

- Add a small orthogonality penalty before trying another adapter class.

## Evaluation

Each candidate must be compared against current `algorithm=tlora` with the same
dataset and seed:

- 60-image V100 smoke: confirm no NaN/black preview.
- 32G style config: compare loss curve, validation curve, and preview stability.
- Same checkpoint format load test in LoRA test page.
- Wall-clock overhead must stay under 3 percent for ASR-LoRA.

Promotion rule:

Only move code from this folder into `lora_modules` / `lora_anima` when it
passes one real training run and one save/load inference check.

## Next Concrete Patch

Implement `per_sample_timestep_mask=false` behind a hidden network arg:

```text
--network_args use_timestep_mask=true per_sample_timestep_mask=true
```

Default stays false. The patch is acceptable only if old T-LoRA checkpoints
load unchanged and standard LoRA behavior is byte-for-byte unaffected when the
flag is false.
