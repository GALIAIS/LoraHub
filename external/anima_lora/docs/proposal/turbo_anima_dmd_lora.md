# Turbo Anima — Decoupled DMD distillation with co-LoRA student + fake

A proposal to distill the 28-step Anima teacher (CFG=4) into a 4–8 step
generator via Decoupled DMD (Liu et al., arXiv:2511.22677, "CFG Augmentation
as the Spear, Distribution Matching as the Shield"), using **LoRA for both
the student and the fake score model** on a single frozen DiT.

This is a new training pipeline — not a new adapter family. It composes with
the existing adapter ecosystem at inference: `turbo_anima.safetensors` can
be loaded alongside any trained concept LoRA on the same frozen DiT, the
same way LCM-LoRA composes with style LoRAs.

## Why this is the next move

Three independent signals converge on Decoupled DMD as the right distillation
prior for Anima specifically:

1. **Anima's production config (CFG=4, 28 steps) is the exact regime
   Decoupled DMD addresses.** The paper shows that DMD's success comes from
   "baking the CFG pattern" into the student via the CA term
   `(α−1)(s_cond_real − s_uncond_real)`, not from distribution matching per
   se. With `α=4` and a 28→4 step collapse, that bake target is large and
   well-defined.

2. **The reverse-forward distribution drift observed in `output/dcw/`
   supports their Decoupled-Hybrid schedule.** Calibration traces show
   Anima's iterative inference state matches the training-time x_t
   distribution up through step ~25, then drifts at steps 26–28 — the same
   SNR-t inference-drift that DCW corrects at the sampler boundary. The
   τ_CA > t constraint cashes this in directly: at small generator-t (late
   in the student's compressed trajectory), the student is operating near
   the drifted regime, so evaluating CA at small τ would compute the
   teacher's CFG gap on out-of-distribution states. Renoising *up* into
   τ > t puts the CA evaluation back into the regime where the
   training-inference gap is minimal and the teacher's score is
   well-calibrated. The schedule is doing correctness work, not just
   concentrating gradient — which also makes a testable prediction
   (CA-branch variance should rise sharply if τ_CA is allowed to drop near
   small generator-t).

3. **Co-LoRA simplifies the fake's modeling problem.** Standard DMD trains
   the fake from teacher init as a full model to track G_θ's drifting
   distribution. With student-as-LoRA, the fake only needs to model "what
   changes when student LoRA is applied to the frozen base" — a strictly
   smaller target than full distribution tracking. The frozen base does the
   heavy lifting for both.

The deployment story is the bonus: a small `turbo_anima_lora.safetensors`
that any user can stack on top of their existing concept LoRA at inference,
same model surgery they already do.

## Background: what Decoupled DMD claims

The DMD-in-practice gradient (Eq. 3 in the paper) decomposes algebraically
into two terms (Eq. 6):

```
∇L_DMD = E[ −( Δ_real-fake + (α−1)·Δ_cfg ) · ∂G_θ(z_t)/∂θ ]

  Δ_real-fake = s_cond_real(x_τ) − s_cond_fake(x_τ)          # DM
  Δ_cfg       = s_cond_real(x_τ) − s_uncond_real(x_τ)        # CA
```

The paper's two empirical claims:

- **CA is the engine.** Training with `Δ_cfg` alone (no `Δ_real-fake`)
  converts a multi-step model into a usable few-step generator quickly, then
  collapses into artifacts after a few k iterations (Fig 2). The
  few-step-conversion ability is almost entirely from CA.
- **DM is a regularizer, not the engine.** Training with `Δ_real-fake` alone
  is unstable. Its job in the combined loss is to cancel artifacts CA
  introduces. Their Fig 3 shows that simpler regularizers (mean/variance
  matching, GAN) also stabilize CA — DM is just a particularly well-behaved
  choice.

The actionable recipe from Sec 4.3 / Table 1:

- **τ_CA > t**: CA's re-noising timestep is sampled strictly noisier than
  the generator's current step. This concentrates CA on still-unresolved
  content (the engine focuses on what needs converting).
- **τ_DM ∈ [0, 1]**: DM's re-noising timestep spans the full noise range.
  This lets DM correct global artifacts (color drift, oversaturation) that
  inherit from earlier steps regardless of the current step.

Their Table 1 row 4 ("τ_CA > t, τ_DM ∈ [0,1]") is the winning configuration
on Lumina-Image-2.0. That's the same generation of DiT as Anima (flow
matching, single-stream cross-attention), so the schedule should transfer
cleanly.

## Design

### Phase split

| Phase | Target steps | Student rank | Fake rank | Purpose |
|-------|-------------|--------------|-----------|---------|
| v0    | 8           | 128          | 128       | Conservative target; matches Z-Image's 8-step claim. Validate the pipeline works. |
| v1    | 4           | 128          | 128       | Production target. |
| v2    | 4           | 64 or 256    | =student  | Rank sweep, only if v1 lands. |

8-step first because it's the safer engineering target — if 8-step fails to
match the teacher, 4-step will fail harder, and we want to debug the
training loop before debugging capacity. Z-Image is the existence proof for
8-step.

### Three roles on one frozen DiT

```
                       frozen Anima DiT (no grad)
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   teacher view          student view          fake view
   (no adapter,          (student LoRA          (fake LoRA
    CFG α=4 at           applied)                applied)
    inference)
```

All three forwards reuse the same frozen DiT weights — the LoRA attachments
swap. This is the same monkey-patching pattern as
`networks/lora_anima/network.py:apply_to`, just with two LoRA networks
toggling on/off per call.

VRAM accounting at bf16:
- Frozen DiT: ~5 GB (no grad, no optimizer state)
- Student LoRA rank-128: ~150 MB params + grad + Adam state ≈ 1.2 GB
- Fake LoRA rank-128: same ≈ 1.2 GB
- Activations for one batch at 1024² with grad checkpointing: ~6 GB
- Total: ~15 GB. Fits on a 16 GB card.

### Target modules

Both student and fake LoRA target the same modules:
- All attention projections: `to_q`, `to_k`, `to_v`, `to_out` (cross + self).
- FFN layers: `layer1`, `layer2`.
- Every block (no skip pattern).

This is the same target surface as the LoRA family default in
`networks/lora_anima/factory.py`. No reason to be exotic here — the
student needs to absorb a global functional change (28→4 step trajectory
remapping), not a localized concept.

### Co-LoRA capacity argument

Why fake rank ≥ student rank, not less:

The fake's job is to provide a fake-score estimate `s_cond_fake(x_τ)` that
tracks the student's actual output distribution. The DM term
`(s_cond_real − s_cond_fake)` is the corrective signal — if the fake
under-fits the student, DM produces noisy gradients and stops canceling
CA's artifacts. The paper's Fig 2 row 1 shows what this looks like
empirically: high-freq checkerboard noise that compounds over training.

LoRA is a rank-r perturbation of the frozen base. The student LoRA at
rank r_s defines a manifold of student-perturbed scores; the fake LoRA at
rank r_f needs to be at least r_s-expressive to track it pointwise. We
make this concrete: **`r_fake = r_student`** at minimum. v2 could try
`r_fake = 2 · r_student` if v1 stability is marginal.

This is the opposite of the usual LoRA intuition where smaller-rank is
better-regularized. Here the fake isn't a generator — it's a *score
tracker*. Its rank constraint is a capacity ceiling on DM regularization
strength, not a stability prior.

### Schedule (Decoupled-Hybrid, Table 1 row 4)

Per training step, the generator current timestep `t` is drawn from the
flow-matching prior. Then:

```python
# Generator step — single-call DMD2 forward.
# Anima's network is a *velocity* predictor (v_t = ε − x_0), not an x0
# predictor, so we explicitly convert to the clean-end endpoint before
# renoising. The conversion uses the same x_t = (1 − t)·x_0 + t·ε
# convention as `library/runtime/noise.py`:
#     x_0 = x_t − t · v_t
# This is the DMD2-style single-call generator; we do NOT unroll the
# 4/8-step inference sampler at training time — gradient is one ODE step
# from the sampled t.
x_t = sample_noise_at(t)                    # (1-t)·x_0 + t·ε from a data sample
v_student = student_lora(x_t, t, c)         # student velocity prediction
x_pred = x_t - t * v_student                # endpoint (clean-x0 estimate)

# CA branch — τ_CA > t (focused engine)
tau_ca = sample_uniform(t, 1.0)             # strictly noisier than t
x_renoised_ca = renoise(x_pred, tau_ca)
with torch.no_grad():
    s_real_cond_ca   = teacher(x_renoised_ca, tau_ca, c)
    s_real_uncond_ca = teacher(x_renoised_ca, tau_ca, c_null)
delta_cfg = s_real_cond_ca - s_real_uncond_ca

# DM branch — τ_DM ∈ [0, 1] (comprehensive regularizer)
tau_dm = sample_uniform(0.0, 1.0)
x_renoised_dm = renoise(x_pred, tau_dm)
with torch.no_grad():
    s_real_cond_dm = teacher(x_renoised_dm, tau_dm, c)
s_fake_cond_dm = fake_lora(x_renoised_dm, tau_dm, c)   # has grad to fake

delta_dm = s_real_cond_dm - s_fake_cond_dm

# DMD gradient on student, with CA warmup (see [optim].alpha_warmup_steps).
alpha_eff = alpha * min(1.0, step / alpha_warmup_steps) + 1.0 * (1.0 - min(1.0, step / alpha_warmup_steps))
grad_signal = delta_dm + (alpha_eff - 1.0) * delta_cfg
# Backprop −grad_signal · ∂G_θ/∂θ through student.
```

The α warmup linearly ramps `alpha_eff` from `1.0` (CA disabled — pure DM)
to the target `alpha = 4.0` over the first `alpha_warmup_steps` (default
`1000`). At step 0 the gradient is `delta_dm` alone, so the student starts
inside the regime DM can regularize before the large CA signal kicks in.
This is structural, not a mitigation — LoRA capacity is smaller than the
paper's full-fine-tune student, so `(α − 1) = 3` from step 0 is enough to
NaN the student before any usable image structure forms (see R2 below for
the diagnostic).

Fake training (one optimizer step per generator step, same as standard DMD2):
```python
# Standard score-matching loss on x_pred — fake learns to denoise the
# student's output distribution.
fake_loss = flow_matching_loss(fake_lora, x_pred.detach(), c)
```

Two optimizer states — one for student LoRA, one for fake LoRA. Both
AdamW. No shared params.

### Re-noising primitive

For flow matching with `x_t = (1 − t) · x_0 + t · ε`, re-noising the
student's endpoint estimate `x_pred = x_t − t · v_student` to noise level
`τ` is:
```
x_τ = (1 − τ) · x_pred + τ · ε,   ε ~ N(0, I)
```
Same primitive as in `library/runtime/noise.py`'s training-time noise
sampler — just applied to the student's predicted clean image instead of
to a dataset latent. Note `x_pred` is *not* the network output directly;
it's the endpoint converted from the velocity output (line 173 above).

### Inference at production time

The trained student LoRA is used as a normal LoRA at inference, with the
caller setting `num_inference_steps = 4` (or 8) and `cfg = 1.0` (CFG is
baked in — the paper's whole point). It should work with:
- Existing concept LoRAs (linear LoRA composition, ranks add).
- DCW v4 (`library/inference/dcw_calibrator.py`): tunable; the per-step λ
  schedule was calibrated on a 28-step trajectory and will need its own
  4/8-step recalibration. Out of scope for v1.
- Spectrum (`networks/spectrum.py`): incompatible by construction —
  Spectrum's Chebyshev cache assumes ≥16 steps. Don't try to stack.
- Modulation guidance: tunable; the distilled `pooled_text_proj` should
  still help, but a turbo student may have re-learned the modulation
  pathway implicitly. Test, don't assume.

## Config surface

New method file `configs/methods/turbo.toml` (experimental until validated):

```toml
network_module = "networks.methods.turbo_dmd"

[network_args]
student_rank = 128
fake_rank = 128
target_modules = ["q", "k", "v", "out", "ffn1", "ffn2"]

[dmd]
teacher_steps = 28
teacher_cfg = 4.0
student_steps = 8            # v0; flip to 4 for v1
alpha = 4.0                  # CA scale, matches teacher CFG
tau_ca_strategy = "above_t"  # τ_CA > t
tau_dm_strategy = "uniform"  # τ_DM ∈ [0, 1]

[optim]
student_lr = 1e-5
fake_lr = 1e-5
fake_steps_per_student_step = 1
alpha_warmup_steps = 1000    # linear ramp alpha_eff: 1.0 → alpha over these steps
```

GUI variant `configs/gui-methods/turbo_8step.toml` for the v0 entry point,
`turbo_4step.toml` once v1 lands.

Make targets (under the `exp-*` umbrella until validated):
```
make exp-turbo                 # default: 8-step
make exp-turbo STUDENT_STEPS=4 # 4-step
make exp-test-turbo            # inference with the latest turbo LoRA
```

## Validation plan

Phased. Each phase gates the next. Bench scripts go under `bench/turbo/`,
following the `bench/_common.py` envelope from `CONTRIBUTING.md`.

### Phase 0: single-prompt overfit (1 day)

Goal: prove the loop runs and converges on one prompt.

- Single fixed prompt, batch size 1, 2k iterations.
- Compare student@4-step output to teacher@28-step output on a fixed seed.
- Pass: student output is recognizably the teacher's image (same subject,
  same composition). Quality drop is acceptable.
- Fail: student diverges, fake collapses, or training NaNs. Likely fix:
  CA scaling — drop `α` from 4.0 to 2.0, recover stability, then ramp.

### Phase 1: 100-prompt sweep (3 days)

Goal: confirm generalization off the overfit prompt.

- 100 prompts from `image_dataset/` captions, batch size 1, 10k iterations.
- Image Reward + HPS v2.1 on the 100 prompts at student@4-step vs
  teacher@28-step.
- Per-aspect ratio breakdown (1024², 832×1248, 1248×832) — Anima quality
  is aspect-dependent (per `project_dcw_cfg_aspect_signflip` memory),
  distillation may behave differently per aspect.
- Pass: student Image Reward ≥ 80% of teacher's Image Reward, no aspect
  scores below 60% of teacher's.
- Fail: aspect-specific collapse → check fake's per-aspect coverage in
  the training batch; rank-too-low at one aspect → bump student rank to
  256, retrain Phase 1.

### Phase 2: full HPS bench (1 week)

Goal: match the paper's Table 1 methodology.

- 1k COCO-prompt sample, both Image Reward and HPS v2.1 + DPG.
- Run all 4 schedule configs from Table 1 (DMD-baseline, Decoupled-Full,
  Coupled-Constrained, Decoupled-Hybrid) as an ablation. Replicates the
  paper's claim on Anima.
- Pass: Decoupled-Hybrid wins (or ties) on aggregate, matching the
  paper's row 4 result on Lumina.

### Phase 3: composition test (2 days)

Goal: confirm the deployment story.

- Pick 3 existing concept LoRAs from `output/ckpt/`.
- Test (turbo only), (concept only @ 28-step), (turbo + concept @ 4-step).
- Subjective eyeball test — does the concept survive?
- Pass: concept LoRA recognizably present in 4-step turbo composition.
- Fail: concept washed out → likely the student baked the unconditional
  bias too strongly. Mitigation: train the student with concept-LoRA-on
  augmentation a fraction of the time (deferred to a future revision).

Skip rule: if Phase 1 fails after one rank bump, kill the proposal. The
8→4 step gap is what makes turbo worth shipping; if 4-step quality is
unrecoverable, the project's value collapses and we're better off
investigating Hyper-SD-style consistency distillation instead.

## Risks and failure modes

Ordered by likelihood, most to least.

### R1: Fake under-tracks student → DM is weak → CA artifacts

Highest-likelihood failure. Symptoms: student image shows high-freq
checkerboard or oversaturation that grows over training, while fake_loss
on the standard score-matching objective looks fine.

Cause: the fake's standard score-matching loss measures the fake's ability
to denoise the student's outputs *averaged over noise levels*. That isn't
the same as the fake matching the student's score *at the specific τ_DM
levels DM is evaluated at*. Rank ceiling of LoRA fake amplifies this gap.

Mitigation: (a) bump fake rank to 2× student rank if student rank ≥ 128.
(b) Add an explicit consistency check at the τ_DM samples — log
`||s_real_cond_dm − s_fake_cond_dm||` per timestep bucket; if any bucket
diverges, the fake isn't covering that region.

### R2: CA gradient too large at α=4 → student diverges (even with warmup)

Likelihood: medium. Symptoms: student NaNs after the warmup window closes
(~step 1k+), or fake_loss spikes catastrophically once `alpha_eff`
approaches the target.

Cause: `α − 1 = 3` is a large multiplier on `Δ_cfg`. The paper used the
same value on SDXL — but our LoRA student has less capacity to absorb the
signal than their full-fine-tune student. The default 1k-step warmup is
the first line of defense; this risk is about the warmup not being
enough.

Mitigation: (a) extend `alpha_warmup_steps` to 2k–4k. (b) Cap the final
α at 2.0–2.5 (Z-Image used 7.5 reduced to ~4 with their schedule; we have
less headroom). (c) If NaNs occur *during* the ramp (step < warmup), the
issue isn't α magnitude — look at LR or the fake's tracking, not at α.

### R3: 4-step is fundamentally not enough capacity

Likelihood: medium for the 4-step v1, low for the 8-step v0. Symptoms: v0
passes Phase 1, v1 fails Phase 1 with student quality at <50% of teacher.

This is the failure mode where the proposal partially succeeds (8-step
turbo ships, 4-step doesn't). 8-step at 5060 Ti inference is still a
~3–4× speedup over 28-step, so v0-only is a real outcome, just less
exciting than the original ambition.

### R4: LoRA student saturates below full-fine-tune student

Likelihood: medium-low. Symptoms: Phase 2 shows the LoRA turbo is, e.g.,
85% of teacher Image Reward while the paper's full-fine-tune Z-Image
hits ~95%.

Mitigation: this is the cost of the co-LoRA approach. If it ships at 85%
with the composition story (Phase 3) working, it's still useful. If it
ships at 85% *and* composition is broken, kill it — at that point the
honest move is full-fine-tune distillation, which is a different (much
more expensive) project.

### R5: Late-step instability from τ_CA > t

Likelihood: low. Symptoms: when student's `t` is sampled near 1.0 (clean
end), τ_CA's `uniform(t, 1.0)` collapses to a near-empty interval and
gradient is noisy.

Mitigation: clamp `tau_ca = max(uniform(t, 1.0), t + 0.05)` — guarantees
a minimum re-noising gap. If t > 0.95, skip the CA branch entirely (DM
alone for that step). Both are minor implementation details, not
structural.

## Out of scope

- **Full-fine-tune student baseline.** Justified by VRAM, not science.
  If co-LoRA fails completely we revisit this, but it's a much larger
  project (multi-day per checkpoint, multiple GPUs).
- **DCW recalibration at 4/8-step.** The current DCW v4 fusion head was
  trained against 28-step trajectories. A turbo-specific DCW would need
  its own `make dcw` run after v1 ships. Don't conflate the two.
- **Self-distillation (Z-Image-style chained distillation).** The Z-Image
  paper chains multiple distillation rounds. Defer until single-round
  results are in.
- **GAN-regularizer variant.** The paper showed GAN can replace DM (Fig
  3); we're sticking with DM because GAN-as-regularizer added training
  instability they themselves note ("collapsed after 4k iterations").
  Not the experiment to run when we're already taking a co-LoRA bet.
- **Negative prompt / multi-CFG bake.** The paper bakes single-CFG; we
  do the same. Concept-LoRA stacking handles user-side prompt tuning.

## File-level plan

New files:
- `networks/methods/turbo_dmd.py` — `TurboDMDNetwork` wrapping two LoRA
  networks (student, fake). Reuses `networks/lora_modules/` building
  blocks. Implements `forward_student` / `forward_fake` plus the
  attachment toggle hooks.
- `scripts/distill_turbo.py` — main training loop. Modeled on
  `scripts/distill_mod/distill.py:1` for the frozen-DiT + adapter-only
  pattern. Differences: two adapters, two optimizers, the renoise step,
  CA+DM gradient assembly.
- `configs/methods/turbo.toml`, `configs/gui-methods/turbo_8step.toml`,
  `configs/gui-methods/turbo_4step.toml` — config surface.
- `bench/turbo/measure_turbo_quality.py`, `bench/turbo/README.md` — Image
  Reward + HPS bench. Reuses `bench/_common.py` envelope.
- `docs/methods/turbo.md` — promote here from `proposal/` once v0
  passes Phase 1.

Touched files:
- `tasks.py` + `scripts/experimental_tasks/training.py` — add `exp-turbo`
  / `exp-test-turbo` commands.
- `Makefile` — same.
- `inference.py` — no changes expected; turbo LoRA loads through the
  existing LoRA adapter path.

## Open questions before kicking off

1. **Teacher caching.** Do we precompute teacher trajectories for the
   training image set, or run teacher inference live each step? Live is
   simpler and matches the paper; cached is faster per step but burns
   disk. Default: live, with cache as a phase-2 optimization if needed.
2. **Fake init.** Zero-init the fake LoRA, or init from a copy of the
   student LoRA after, say, 500 iterations? Standard DMD inits fake from
   teacher; we have no equivalent for "LoRA-of-teacher" so zero is the
   natural baseline. Bears watching.
3. **Validation prompts.** Reuse the 100-prompt set from the DCW bench
   (`bench/dcw/results/`) or sample fresh from `image_dataset/`? Default:
   fresh, to avoid coupling turbo validation to DCW's particular prompt
   distribution.

None of these are blocking. Defaults above can be revisited after Phase 0
runs.
