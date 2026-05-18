# postfix_residual — per-image inversion as a *probe* of the residual manifold

Sibling proposal to [`postfix_residual_for_directedit.md`](postfix_residual_for_directedit.md).
Same decomposition `ψ_src = T5(tags) + postfix(image)`, but a different
question. The companion proposal asks **how to deliver** an image-conditional
postfix (amortized encoder). This proposal asks **whether the decomposition is
even viable**, and **what shape** the postfix space needs to be — by inverting
the postfix tail per image against the frozen DiT and characterizing the
resulting set of tails.

This is inversion-as-measurement, not inversion-as-product. There is no
sidecar workflow, no CLI for batch inversion across the dataset, no DirectEdit
or ComfyUI integration. The deliverable is a set of numbers and the design
implications they imply for the encoder.

## TL;DR

```
prefix = T5(tags)           # frozen
tail   = Q @ diag(s)        # K=48 trainable scales, Q is a fixed orthonormal basis
ψ      = concat(prefix, tail)
optimize s against L_FM(image, ψ)    # per-image; ~30–90 s on a 5060 Ti
```

Run this on N ≈ 30–50 source images. Then answer:

1. **Ceiling.** Does `T5(tags) + inverted_tail` close the dry-run gap that
   `T5(tags)` alone leaves? How does the ceiling scale with K?
2. **Manifold.** What is the intrinsic dimensionality of the inverted-tail
   set? Do tails cluster by visual content? Are seed-replicates of the same
   image *functionally* equivalent even when raw tails differ?
3. **Lane discipline.** Without explicit regularization (λ_zero), does the
   tail spontaneously stay decorrelated from the text channel, or does it
   overgrow into tag-redundant content?

Each answer has a specific design consequence — see "Decision tree from probe
outputs" below.

## Why a probe before either encoder

The companion encoder proposal commits to (a) PE feature pooling, (b) `K=48`
tail length, (c) `λ_zero` regularization as the load-bearing lever against
lane overgrowth, (d) FM loss + optional `L_directedit_dry` as the supervision
signal. Each commitment is a guess about the geometry of the residual.
Per-image inversion measures that geometry directly, **before** any of those
guesses get baked into a training pipeline.

The three risks the companion proposal imports — lane overgrowth, slot
collapse re-validation on PE input, FM-from-noise as a soft supervision — are
all answered (or sharpened) by probe results. Concretely:

| Companion-proposal guess | What the probe tells you |
|---|---|
| `K=48` is the right tail length | Ceiling-vs-K curve. If ceiling saturates at K=8, K=48 is wasted capacity and a lane-overgrowth risk. |
| MLP-from-pooled-PE has enough capacity | Intrinsic dim of inverted-S vectors. If d_intrinsic ≪ K·D, you don't need a (PE → K·D) MLP — a (PE → d_intrinsic) MLP + decoder suffices. |
| λ_zero will hold the tail in its lane | Whether *unregularized* per-image inversion stays decorrelated from T5(tags) at convergence. If it doesn't, the lane is a soft constraint and the encoder needs an explicit decorrelation loss. |
| FM-from-noise is "good enough" for the encoder | If per-image inversion (which *also* uses FM loss) can't close the dry-run gap, the encoder can't either. Cheap negative result. |

If the probe says "the decomposition is viable and the manifold is low-dim
and content-clustered," the encoder design becomes much more informed (and
much smaller). If the probe says "the decomposition tops out far from the
dry-run target," the whole `T5(tags) + postfix(image)` framing needs
rethinking before more engineering — and the companion proposal blocks on
that result.

## What this is

- A measurement instrument: per-image optimization of K orthogonally-constrained
  tail scales against the frozen DiT, producing inverted-S vectors that we then
  analyze in aggregate.
- A small experiment (N ≈ 30–50 images, plus 3-seed replicates on a subset)
  that runs once, drops a bench report, and either greenlights or kills the
  encoder direction.

## What this isn't

- **Not a deployable DirectEdit primitive.** No sidecar `.safetensors` per
  training image, no first-load optimization at edit time, no ComfyUI socket.
  The 30 inverted tails from the probe are diagnostic artifacts; they aren't
  shipped to users.
- **Not classical embedding inversion** (`archive/inversion/invert_embedding.py`).
  That optimizes the *entire* 512-length sequence and yields a ψ that has no
  text-editable decomposition. The probe freezes the prefix so the
  decomposition question is well-posed.
- **Not the amortized encoder** (`postfix_residual_for_directedit.md`). That
  is downstream — its shape depends on probe outputs.
- **Not the existing K=8 reference inversion** (`archive/inversion/invert_reference.py`,
  `make invert-ref`). That path inverts K=8 *prefix* slots against a single
  reference image and produces a textual-inversion-style subject token, with
  no orthogonality constraint and no decomposition into "text prefix + image
  residual." The probe reuses much of that script's optimization loop but
  asks an orthogonal question: not "what K vectors stand for this subject"
  but "what K-dim subspace closes the dry-run gap on top of the actual
  tags." See "Prior art" below for the lineage.

## Prior art in tree

`orthogonal_postfix.md` §5 already flags this as a small follow-up:

> Adding an `--ortho` flag to [`invert_reference.py`] re-runs the same workload
> with the symmetry broken. If the orthogonal version produces meaningfully
> different (richer? more image-distinctive?) inversions, that's evidence
> the original K=8 path was capacity-limited all along…

This proposal is essentially that follow-up, scoped harder: not "ortho-ify
the existing reference inversion to see if it gets richer," but **use
ortho-tail inversion as the instrument that decides whether the
`T5(tags) + postfix(image)` decomposition has a real ceiling and what shape
the encoder's output space should take.**

Practical consequence: the implementation is mostly a copy of
`archive/inversion/invert_reference.py` with three changes — (1) prefix is
the *actual cached T5(tags)*, not a templated `"a photo"`; (2) the K trainable
slots become `Q @ diag(s)` instead of free `(K, D)` parameters; (3) the
analysis runs over N images jointly, not per-image-in-isolation. Items (1)
and (2) come directly from `orthogonal_postfix.md`'s parameterization.

## Architecture (the probe instrument)

Reuse the existing postfix carrier (`networks/methods/postfix.py` —
`splice_position=end_of_sequence` or `front_of_padding`). Drop the `cond_mlp`
(no image features feed it; the trainable parameters *are* the tail). Use
the simplest ortho parameterization:

```
tail = Q @ diag(s)               # Q: (K, D) frozen orthonormal
                                 # s: (K,)   trainable scales (per-slot magnitudes)
```

Total trainable params per image: K (≤ 48). Fewest possible — structural
orthogonality, no Cayley rotation needed for v0 since we're not asking the
tail to *rotate* within its basis, just to *select magnitudes* along the K
fixed orthonormal directions. (Cayley `Q @ R @ diag(s)` is a v0.5 ablation
if v0 ceiling underperforms — adds K(K-1)/2 params.)

Choice of `Q`:

- **SVD of cached T5 corpus** (v0 default) — top-K right singular vectors
  of stacked `_anima_te.safetensors` across the training set. Aligns the
  tail subspace with "directions T5 actually uses for this corpus" and
  matches what cond+ortho postfix already does (`ortho_basis=svd_te`).
- **Random orthonormal** (v0.5 ablation) — sanity check that SVD basis
  isn't doing the heavy lifting via some content-leakage path.

Splice: reuse `end_of_sequence` (caption-position-agnostic) — same as the
default postfix path. No new splice code.

## Optimization

Lift the loop from `archive/inversion/invert_embedding.py` (or
`invert_reference.py` — closer fit since it already freezes a prefix). Per
image:

```
L = L_FM (frozen DiT, ψ = concat(T5(tags), Q @ diag(s)), random t per microbatch)
```

Two probe variants:

- **Unregularized** (primary): no λ_zero penalty. The question is whether
  the tail *spontaneously* stays in its lane. Whatever it does, that's the
  ground truth about the loss landscape.
- **Regularized** (auxiliary): `L + λ_zero · ‖s‖²`, sweep λ_zero ∈ {0.001,
  0.01, 0.1}. Measures the cost of pushing the tail toward minimality — i.e.
  what reconstruction quality you sacrifice to guarantee lane discipline.

Hyperparameters (ported from `archive/inversion/invert_embedding.py`
defaults):

- `--steps 100` (extend to 500 if ceiling hasn't plateaued)
- `--lr 0.01`, `--lr_schedule cosine`
- `--grad_accum 4`, `--timesteps_per_step 1`
- `--sigma_sampling uniform`, sweep `--sigma_min ∈ {0, 0.1, 0.2}` (the
  small-tail signal is easily drowned at low σ; P-GRAFT-style low-σ skip is
  more relevant here than for full inversion)
- `s` init: zeros (tail starts at exactly zero — clean baseline). Compare
  against `s ~ N(0, 0.149²)` (matches the archive script's `init_jitter_std`)
  as an ablation.

Early-stop on **dry-run LPIPS / DreamSim**, not on FM loss
(`project_fm_val_loss_uninformative` — FM val loss doesn't track quality on
Anima). The FM curve will plateau before the perceptual gap closes.

## Probe metrics

The N ≈ 30–50 inverted-S vectors get analyzed jointly. Each row below is one
quantity the bench report writes out:

| Metric | What it answers | Decision implication |
|---|---|---|
| **Dry-run LPIPS / DreamSim**, baseline (`T5(tags)`) vs probe (`T5(tags) + tail`), per image | Ceiling: does the decomposition close the gap? | If gap doesn't close → decomposition is dead, kill encoder direction. |
| **Ceiling vs K** curve, K ∈ {4, 8, 16, 32, 48} | How much tail capacity does the residual *actually* need? | Encoder's K is `argmin K s.t. ceiling plateaued`. |
| **Intrinsic dimensionality** of `{s_i}_i^N`: PCA spectrum, %variance @ first j components | Is the residual a low-dim manifold? | If d_intrinsic ≪ K, encoder bottleneck is `(PE → d_intrinsic) → K-decoder`, not `PE → K·D`. |
| **Content-clustering**: cluster inverted-S vectors (k-means / agglomerative) and check whether visually-similar images cluster together (manual eyeball or a CLIP-distance proxy on the source images) | Does the manifold have semantic structure encoder can latch onto? | If yes → encoder can use a smooth MLP. If no (random-looking) → encoder needs either huge capacity or the framing is wrong. |
| **Multi-seed functional equivalence**: re-invert 5 images with 3 seeds each. Compare raw cosine of `s` and **functional cosine** (cosine of `Q @ diag(s)` outputs after splicing, measured at DiT block-N activations or at sampled images via LPIPS). | Is the inversion problem many-to-one (cosmetic ambiguity) or one-to-one (geometric ambiguity)? | Many-to-one functional but high raw variance → encoder targets need alignment (Hungarian / canonical-form pre-processing) for L2 distillation to be well-conditioned. Already-tight → use raw L2. |
| **Lane-discipline (cross-channel correlation)**: cosine of `tail` against the top text-channel directions (singular vectors of T5(tags) for that image). | Does the unregularized tail naturally avoid the text subspace? | High correlation → lane is a soft constraint; encoder needs explicit decorrelation loss. Low correlation → λ_zero is sufficient, no decorrelation term needed. |
| **Tail-only reconstruction** (`ψ = tail` with prefix zeroed): does it reconstruct? | Sharper lane test — if the tail alone can drive recovery, it's a full encoder hiding inside a "residual." | Should fail. If it succeeds, the encoder will overgrow whether λ_zero is set or not. |
| **Optimization cost** | Wall-clock per image at the probe defaults | Cost framing — feasibility of the *probe itself*, not a product cost. Order-of-magnitude only. |

The probe is small enough that this fits in one bench run with a few seeded
re-runs for the multi-seed test. No iterative pipeline.

## Decision tree from probe outputs

The point of the probe is that each outcome has a specific consequence:

```
Does ceiling close meaningfully? ──────── No ─→ Decomposition is dead.
                                                 The companion encoder
                                                 proposal blocks here.
                                                 Pivot or close.
            │
            Yes
            ▼
Where does ceiling plateau in K? ──────── K* (call this the encoder budget)
            │
            ▼
Intrinsic dim ≪ K* ? ──────────────────── Yes ─→ Encoder = (PE → d_intrinsic
                                                 latent) + (latent → tail
                                                 decoder via the same Q
                                                 basis). Smaller, better-
                                                 conditioned target.
                                                 │
                                          No ─→ Encoder = (PE → K*·D
                                                 directly). Larger MLP /
                                                 resampler.
            │
            ▼
Multi-seed functional cosine high? ─────  Yes ─→ Encoder distills against
                                                 raw inverted-S (cheap L2).
                                                 │
                                          No ─→ Need canonical-form alignment
                                                 (e.g. Hungarian on slot
                                                 ordering, or learn the
                                                 encoder against a *function*
                                                 of inverted-S, not raw).
            │
            ▼
Lane stays disciplined unregularized? ─── Yes ─→ Encoder can train with λ_zero
                                                 alone (companion proposal's
                                                 plan).
                                          No ─→ Encoder needs explicit
                                                 tag-decorrelation loss.
                                                 Add `‖tail · T5(tags)ᵀ‖²`
                                                 or similar.
            │
            ▼
Manifold has content clusters? ────────── Yes ─→ Encoder MLP from pooled PE
                                                 is plausible (smooth target).
                                          No ─→ Need full-token (resampler-
                                                 style) input, not pooled.
                                                 Or framing is wrong.
```

## Why this answers things the encoder proposal can't

The companion proposal lists "lane overgrowth," "slot-collapse re-validation
on PE input," and "FM-from-noise as wrong supervision" as risks to manage in
training. Each is at-best partially observable in an encoder training run:

- **Lane overgrowth** under encoder training is confounded with capacity:
  if the encoder underfits the residual it also under-overgrows. You can't
  cleanly tell "λ_zero needs to be higher" from "the encoder is too small."
  Per-image inversion has unlimited capacity per image (it's the loss
  optimum on a small parameter set), so its lane behavior is the *intrinsic*
  lane behavior of the loss landscape.
- **Slot collapse** is structurally absent here (Q @ diag(s) is orthonormal
  by construction; collapse cannot occur). So the probe doesn't re-validate
  the cond+ortho LN+maxabs fix on PE input — it bypasses the question
  entirely. *The companion proposal still needs that validation;* the probe
  just establishes that the underlying manifold is rich enough to be worth
  validating against.
- **FM-from-noise supervision quality** — if FM loss can't drive per-image
  inversion to a good ceiling (where it has every advantage: unlimited
  capacity, no generalization burden, no amortization), no encoder using FM
  loss will close the gap either. This is a clean negative-result path.

## Implementation steps

1. **Lift the optimization loop** from `archive/inversion/invert_reference.py`
   into `library/inference/postfix_inversion.py`. The reference script
   already freezes a prefix and optimizes K slots — closer fit than
   `invert_embedding.py`. Changes vs the source:
   - Prefix is loaded from `{stem}_anima_te.safetensors` (the actual cached
     tags-conditioned T5 output), not from an encoded `--template`.
   - K slots become `Q @ diag(s)` where `Q` is loaded from an SVD-of-T5
     basis file (compute once for the corpus and cache); `s` is the only
     leaf parameter.
   - Splice happens at `end_of_sequence` (matching default postfix), not
     at the front of the sequence (where `invert_reference.py`'s prefix
     mode places its K slots).
2. **New CLI** `scripts/invert_postfix_tail.py` — single-image and
   `--image_dir` modes (the latter just for batched probe runs, **not** to
   produce sidecars for downstream use):
   ```
   --image_dir post_image_dataset/lora    # cached latents + cached T5
   --num_images 30 --shuffle --seed 0
   --K 48 --basis svd_te --basis_path output/probes/postfix_tail/svd_basis_K48.pt
   --steps 100 --lr 0.01 --grad_accum 4
   --lambda_zero 0.0                       # primary; sweep separately
   --sigma_min 0.0
   --output_dir output/probes/postfix_tail/<run-id>/
   ```
   Outputs per image: `{stem}_s.safetensors` (just the K-vector) +
   `{stem}_loss.csv`. No splice-ready postfix tensor saved — analysis works
   on `s` directly, and reconstruction-for-metrics rebuilds `Q @ diag(s)`
   on the fly.
3. **Multi-seed mode** `--seeds 0,1,2 --replicate_images 5` runs 3 seeds
   each over a fixed subset of 5 images — input to the multi-seed
   functional-equivalence metric.
4. **K-sweep mode** `--K_sweep 4,8,16,32,48` runs the same N images at each
   K (with appropriate `Q` per K). Input to the ceiling-vs-K curve.
5. **Probe analyzer** `bench/postfix_tail_probe/analyze.py` consumes the
   output directory, computes all metrics in the table above, writes a
   one-page `report.md` + JSON envelope under
   `bench/postfix_tail_probe/results/<run-id>/`. Uses `bench/_common.py`.
6. **Bench harness** `bench/postfix_tail_probe/run_probe.py` is the
   single entrypoint — runs the CLI in the right modes and chains the
   analyzer. One command produces the full probe report.
7. **`make` target** `make exp-probe-postfix-tail` (experimental, since
   this is a one-shot diagnostic). Wire under `scripts/experimental_tasks/`
   to match the `exp-*` family convention.

No DirectEdit changes, no ComfyUI changes, no GUI changes. The probe lives
entirely in `scripts/` + `bench/` + `library/inference/`.

## Failure modes (of the probe itself)

1. **Probe says "ceiling closes" but the encoder later fails anyway.** The
   probe gives an upper bound on amortized performance, not a guarantee.
   A positive probe is necessary, not sufficient.
2. **Multi-seed functional cosine is *low* even after canonical
   alignment.** The inversion problem is fundamentally many-to-one with
   geometric (not just cosmetic) ambiguity. This is a hard signal that the
   amortized encoder will train against an ill-defined target. The decision
   tree above handles the "many-to-one functional but high raw variance"
   case; the "many-to-one *and* functionally divergent" case is the bad
   one — pivot or accept high encoder variance.
3. **Sigma_min sensitivity dominates results.** If the ceiling moves a lot
   with `sigma_min`, the probe is measuring the optimizer's sensitivity to
   the timestep schedule rather than the manifold's geometry. Report
   sigma_min sweep as part of the bench output and call this out
   explicitly.
4. **N too small to characterize manifold.** N=30 is OK for ceiling and
   intrinsic-dim (PCA on 30×K with K=48 is well-conditioned). It's
   borderline for content-clustering (need enough images per cluster).
   Scale to N=100 if clustering is the deciding metric.

## Relationship to other proposals

- **`postfix_residual_for_directedit.md`** (companion): downstream of this
  probe. Its concrete design choices (`K=48`, pooled-PE input, λ_zero as
  the lane lever, FM + `L_directedit_dry` supervision) are guesses; the
  probe either supports or invalidates each one. **Run this first; do not
  build the encoder before the probe report exists.**
- **`orthogonal_postfix.md`** §5 (line 192–201): explicitly proposes adding
  `--ortho` to `invert_reference.py` as a small follow-up if the structural
  fix passes v1. This proposal is that follow-up scoped harder — as a
  probe of the residual manifold, not as a method tweak. The Cayley-rotated
  basis from `orthogonal_postfix.md` is what `Q` is (or simpler — diag
  scales over the same SVD basis).
- **`img2emb_plan.md`**: orthogonal goal (full image embedding replacing
  text). Could in principle be probed the same way (invert the entire
  prefix region per image), but `archive/inversion/invert_embedding.py`
  already does that for analysis purposes; img2emb's question is
  amortization quality, not whether the geometry exists.
- **`archive/inversion/invert_reference.py`** (`make invert-ref`): the
  prior art. Already inverts K=8 vectors per image. The probe replaces
  free-K with ortho-K and reframes the analysis from "what subject token
  did we get" to "what is the manifold of inverted residuals."

## Cost

Per-image optimization at probe defaults: ~30–90 s on a 5060 Ti
(estimate from `archive/inversion/invert_reference.py`'s observed cost at
K=8, scaled for K=48 with `Q @ diag(s)` being lighter than free `(K, D)`).

Total probe wall-clock:
- Primary run: N=30 images × ~60 s = ~30 min.
- K-sweep (5 values × N=30): ~2.5 hours.
- Multi-seed (5 images × 3 seeds): ~15 min.
- Lambda_zero sweep (3 values × N=30): ~1.5 hours.

Call it half a day of GPU time for the full probe report, including the
analyzer's CPU work. That's the right scale for a measurement that decides
whether to commit a multi-week encoder training pipeline.

## What this proposal explicitly does *not* commit to

- No sidecar `.safetensors` artifacts shipped per training image. The
  inverted tails are diagnostic data, not deployable conditioning.
- No DirectEdit CLI flag, no ComfyUI socket, no GUI integration.
- No Phase B encoder distillation pipeline. That belongs to the
  companion proposal *after* the probe report exists.
- No commitment to the Cayley `Q @ R @ diag(s)` upgrade. v0 stops at
  `Q @ diag(s)` unless the ceiling looks capacity-limited; the upgrade is
  itself a probe finding, not a presumed-needed component.
