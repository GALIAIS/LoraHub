# LoRA Algorithm Research Design

This document defines research-only LoRA algorithm directions for the modified
`anima_lora` backend. Nothing here is a production algorithm until real A/B
training proves it is useful.

## Rules

- Do not add a public `algorithm` enum before measured wins.
- Do not change checkpoint format unless the gain cannot be achieved with the
  existing LoRA/LyCORIS save layout.
- Compare every candidate against the current best baseline with the same
  dataset, seed, prompts, resolution, precision, and total steps.
- Promote only after a full train run plus save/load inference check.
- Loss alone is not enough. Preview quality, prompt adherence, stability, and
  speed all count.

## Baselines

Current production baselines to beat:

- Speed baseline: plain LoRA or ASR-T-LoRA at the same rank.
- Character baseline: DoRA / OrthoLoRA, depending on current template.
- Style baseline: LoHA from `configs/anima_style_32gb_loha.yaml`.
- Quality baseline: LoHA/LoKr/ASR-T-LoRA at the largest stable rank that fits.

## Evaluation Metrics

Required numeric fields for every experiment result:

- `seconds_per_step`
- `total_train_seconds`
- `quality_score`
- `prompt_score`
- `identity_score` for character runs
- `style_score` for style runs
- `val_loss`
- `nan`
- `black_preview_count`
- `oom`
- `checkpoint_load_ok`

Required artifacts:

- training config
- seed
- dataset id/path
- final LoRA checkpoint
- preview grid
- base-model comparison grid
- raw log

## Candidate A: Fast Adaptive Rank LoRA

Goal: increase training speed without visible quality loss.

Use case:

- large datasets
- quick iteration
- weak GPUs
- runs where user wants "good enough" fast

Initial recipe:

- Use ASR-T-LoRA.
- Lower base rank by 25%.
- Use per-sample timestep rank masking.
- Keep high-sigma/noisy steps at lower effective rank.
- Keep low-sigma/clean steps near full effective rank.
- Keep existing checkpoint layout.

Current config plan:

- `plan = fast`
- `algorithm = asr_tlora`
- `network_dim_scale = 0.75`
- `min_rank_ratio = 0.50`
- `alpha_rank_scale = 0.80`
- `caption_dropout_rate = 0.10`

Promotion bar:

- at least 12% faster than baseline
- no quality-score drop
- no prompt-score drop beyond noise
- no new NaN/black-preview failures

Likely failure mode:

- underfitting detailed clothing, hands, or fine style texture
- lower rank may look acceptable in loss but worse in previews

Next implementation only if recipe wins:

- add layer-aware rank shrink so attention projections keep more capacity than
  MLP projections during fast training

## Candidate B: Character Fidelity LoRA

Goal: improve character/identity consistency without making outputs stiff.

Use case:

- one character
- small to medium dataset
- identity, face, outfit, or mascot consistency

Initial recipe:

- Use DoRA as baseline because direction/magnitude separation often preserves
  identity better than plain LoRA at the same rank.
- Lower caption dropout than style training.
- Keep rank stable; do not aggressively mask rank.
- Prefer prompt adherence over global style bleed.

Current config plan:

- `plan = balanced_character`
- `algorithm = dora`
- `network_dim_scale = 1.00`
- `caption_dropout_rate = 0.08`
- `useTimestepMask = false`

Promotion bar:

- identity score at least 0.08 above baseline
- prompt score does not drop more than 0.03
- speed overhead <= 5%
- checkpoint load works in the LoRA test page

Likely failure mode:

- overfitting one pose or outfit
- faces improve while scene/prompt control worsens

Next implementation only if recipe is close but not enough:

- add character-focused layer targeting: more capacity on cross-attention and
  selected self-attention, less on broad MLP layers

## Candidate C: Style Fidelity LoRA

Goal: learn transferable style without memorizing dataset subjects.

Use case:

- artist/style LoRA
- few-shot style references
- style should transfer across people, landscape, object, and lighting prompts

Initial recipe:

- Use LoHA baseline because Hadamard low-rank factors are strong for texture,
  color coupling, and brush-like style.
- Use stronger caption dropout than character runs.
- Validate against base-model comparison grids.

Current config plan:

- `plan = balanced_style`
- `algorithm = loha`
- `network_dim_scale = 1.00`
- `caption_dropout_rate = 0.18`
- `useTimestepMask = false`

Promotion bar:

- style score at least 0.10 above baseline
- prompt score does not drop more than 0.03
- speed overhead <= 8%
- preview grid shows transfer across subjects

Likely failure mode:

- style becomes subject memorization
- style only appears when prompt resembles training captions
- LoHA overhead may not justify the quality gain on small datasets

Next implementation only if recipe wins:

- test static layer-rank allocation:
  - cross-attention: full capacity
  - self-attention: medium capacity
  - MLP: reduced capacity

## Candidate D: High Quality Adaptive LoRA

Goal: slower training with clearly better final quality.

Use case:

- final production LoRA
- users willing to spend more time
- dataset is curated and stable
- target is highest visual quality, not quick iteration

Initial recipe:

- Use larger ASR-T-LoRA.
- Increase rank by 50%.
- Use per-sample timestep rank masking so high-rank capacity is spent mostly
  where it matters.
- Enable gradient checkpointing if needed for VRAM.

Current config plan:

- `plan = quality`
- `algorithm = asr_tlora`
- `network_dim_scale = 1.50`
- `min_rank_ratio = 0.35`
- `alpha_rank_scale = 1.25`
- `caption_dropout_rate = 0.14`
- `gradientCheckpointing = true`

Promotion bar:

- quality score at least 0.18 above baseline
- speed overhead <= 35%
- no NaN/black previews
- visible improvement in final preview grid

Likely failure mode:

- better training loss but worse generalization
- slower without enough visible improvement
- larger rank amplifies bad captions or duplicate images

Next implementation only if recipe wins:

- add an orthogonality penalty on LoRA down directions:
  `lambda * ||normalize(A A^T) - I||_F`

## Candidate E: Spectral Stable LoRA

Goal: reduce rank-direction collapse and improve stability.

Status: design only. Do not implement before A-D produce real results.

Core idea:

- Keep LoRA tensor shapes unchanged.
- Add a small auxiliary loss to discourage all rank directions from converging
  into one dominant direction.
- Apply only to Linear LoRA down weights.

Promotion bar:

- fixes an observed collapse case
- does not slow training by more than 5%
- does not require new checkpoint keys

Why deferred:

- adds training loss plumbing
- harder to attribute gains
- unnecessary if LoHA/DoRA/ASR recipes already solve the target problem

## Experiment Workflow

1. Pick baseline config.
2. Generate candidate config with `experiment_plans.py`.
3. Run baseline and candidate with same seed.
4. Save raw logs and preview grids.
5. Append one JSONL result record.
6. Run `passes_promotion_gate(...)`.
7. If gate fails, do not implement production code.
8. If gate passes, repeat once on a second dataset.
9. Only then design production integration.

Example:

```bash
python external/anima_lora/networks/lora_research/experiment_plans.py \
  --base configs/anima_style_32gb_loha.yaml \
  --plan quality \
  --out runs/research/quality.yaml
```

## Promotion Decision

A candidate can move toward production only when all are true:

- passes its numeric promotion bar
- has no NaN, black preview, OOM, or load failure
- visibly improves the preview grid
- does not require users to understand research-only knobs
- has a migration path that keeps old configs working

If the gain is small, keep it as a template recipe, not a new algorithm.
