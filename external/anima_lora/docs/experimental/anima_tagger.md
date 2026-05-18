# Anima Tagger — multi-label tagger trained on Anima's caption distribution

A small classifier that maps an image to a comma-separated tag string in
exactly the format Anima's training-time T5 saw. Used as the case-1 ψ_src
provider for DirectEdit, and as a standalone captioner for LoRA dataset
prep / prompt scaffolding via the `comfyui-anima-tagger` ComfyUI node.

Status: **shipped**. A trained checkpoint lives at
`models/captioners/anima-tagger-v1/` (4,937-tag vocab, residual macro-F1
~0.192 on val with softmax groups argmax-only). End-to-end PE-LoRA training
path, typed tag-group routing, and a curator-side role-marker scanner are
all wired.

## Why this exists

DirectEdit's invert/edit primitive is robust to ψ_src corruption — even
shuffled or tag-dropped source captions reconstruct the source image at
~99% pixel fidelity. But edit *leverage* (whether ψ_tar = ψ_src + edit-tag
actually applies the change) collapses when ψ_src is structurally far
from Anima's training-time embedding manifold. Generic booru taggers
were bad enough at this to be the live blocker; this tagger replaces
that role with an Anima-distribution head.

## Architecture

```
PIL image → PIL LANCZOS-resize to PE-Core bucket size
         → IMAGE_TRANSFORMS (= [-1, 1])
         → frozen PE-Core-L14-336 → patch tokens [T, 1024]
         → mean-pool over T → feature [1024]
         ──────────────────────────────────────────────────  trunk (frozen)
         → LayerNorm + Linear(1024, 1024) + GELU + Dropout
         ────────────────────────────────────────────────── shared trunk_h
         ├→ Linear(1024, n_tags)       → tag_logits     ──── multi-label head
         └→ Linear(1024, 3)            → rating_logits  ──── 3-class rating head

Per-tag F1-calibrated threshold sweep at the end of training picks the
inference threshold for each output dimension. Tags belonging to a
softmax group are excluded from the sweep — they're argmax-only at
inference.
```

Total trained params at default `n_tags=4937, d_hidden=1024`: **~6.1M**
(frozen path). The end-to-end `--pe_lora_rank > 0` path adds a low-rank
delta over the trailing PE-Core blocks; alpha/rank/layers configurable.

The shipped checkpoint runs the frozen path (`pe_lora: false`), trained
for 100 epochs at `lr=2e-4`, `batch_size=64`, `lambda_rating=0.1`.

### Why a shared trunk for both heads

Rating prediction and tag prediction look at the same kinds of visual
content — lots of the rating signal is also expressible as tag
co-occurrence. A shared trunk gives the rating gradient a path into the
same representation the tag head reads from, at the cost of one extra
Linear at the head split. Empirically this is what gelcrawl's quality
classifier does too (`gelcrawl/classify.py`); we reuse that pattern.

### Why mean-pool over patch tokens

PE-Core's CLS token is contrastive-image-text trained — useful for
retrieval, not optimized for multi-label classification. Mean-pool over
the patch tokens gives a content-weighted summary; head capacity is
enough that the pooling choice doesn't bottleneck.

### Why a sqrt(neg/pos) BCE pos-weight

Anima's tag distribution has a heavy long-tail. Default BCE-with-logits
treats every tag-output identically, so common tags (1girl) dominate the
gradient. Inverse-frequency weights (`n_neg/n_pos`) over-correct and
explode rare-tag gradients. `sqrt(n_neg/n_pos)` is the standard middle
ground — softens the long-tail without overshoot.

## Code layout

`library/captioning/` (inference + shared schema):

| File | Role |
|---|---|
| `anima_tagger.py` | `AnimaTagger` — public inference class. Exposes `predict`/`predict_caption`. Loads checkpoint, encoder, vocab, thresholds, rules, optional groups, optional PE-LoRA delta from one directory. Implements all post-prediction refinements (group argmax, character floor, original-fallback, girls-count cap, top-1 artist/copyright). |
| `anima_tagger_model.py` | `AnimaTaggerConfig` + `AnimaTaggerHead` (trunk + tag head + rating head). |
| `anima_tagger_data.py` | `TaggerManifest`, `FeatureCacheBuilder`, `CachedFeatureDataset`, `ImageCacheBuilder`, `CachedImageDataset`, `BucketBatchSampler`, `pil_resize_to_bucket`. |
| `tag_rules.py` | `tag_rules.yaml` loader/applier (replacements, always-remove, clothing dedup, `category_overrides`, `coverage_ignore`). |
| `tag_groups.py` | `tag_groups.yaml` loader; `TagGroup`/`TagGroups`/`ResolvedGroup`; modes `softmax`, `softmax_when_solo`, `multilabel`. |

`scripts/anima_tagger/` (CLI + training pipeline — invoke as
`python -m scripts.anima_tagger.cli`):

| File | Role |
|---|---|
| `cli.py` | Argparse + 7-mode dispatcher (`build_vocab`, `build_features`, `build_resized`, `train`, `calibrate`, `predict`, `scan_role_markers`). Loads `anima_lora/.env` so `CAPTION_CORPUS_DIR` resolves before defaults are computed. |
| `vocab.py` | Caption discovery, tag categorization (rating literal → `@` artist → count regex → `category_overrides` → tag cache → `general` fallback), `min_freq` cut, train/val split, manifest build, group resolution against the kept vocab, coverage scan. |
| `caches.py` | `cmd_build_features` (pooled PE features → `.cache/pooled-pe/`) and `cmd_build_resized` (LANCZOS-resized uint8 images → `.cache/resized-pe/`). |
| `train_cached.py` | Frozen-encoder fast path: full train/val features pushed to VRAM once, sliced by index — no DataLoader. |
| `train_pe_lora.py` | End-to-end PE-LoRA path: bucket-grouped image batches, two AdamW param groups (head at `--lr`, LoRA at `--pe_lora_lr`). |
| `train_common.py` | `GroupRouter`, `compute_grouped_loss` (BCE + per-group CE with mask-out so each (sample, tag) is supervised by exactly one term), `eval_split` (residual macro-F1 + per-group argmax accuracy), `pos_weight_sqrt`, `rating_class_weights`, `save_history_plot`. |
| `calibrate.py` | Per-tag F1-optimal threshold sweep on val (skips softmax-group tags). |
| `predict.py` | Single-image debug entry; samples a random val stem when `--image` is omitted. |
| `role_markers.py` | Read-only curator helper — scans the trained vocab + manifest for character-typed tags that behave like affiliation markers and emits a YAML stub ready to paste into `tag_rules.yaml`. |
| `constants.py` | Tag-type ID map, `RATINGS`, `SLOT_ORDER`, count-tag regex, `find_image_for_caption`. |

## Configuration via `.env`

External corpus paths are routed via `CAPTION_CORPUS_DIR`. Add to
`anima_lora/.env`:

```
CAPTION_CORPUS_DIR=/path/to/external/caption/corpus
```

Expected layout:

| Path | What it is | Consumer |
|---|---|---|
| `<corpus>/retrieved/{artist}/{stem}.{webp,jpg,png,jpeg}` | Source images, paired with `.txt` captions | Training input + label. ~12k images. |
| `<corpus>/retrieved/{artist}/{stem}.txt` | Booru-style caption per image, in Anima format (`rating, count, characters, copyrights, @artists, generals`) | Multi-hot label after `tag_rules` normalization. |
| `<corpus>/retrieved/.tag_cache.json` | `tag → integer type id` (0=general, 1=artist, 3=copyright, 4=character, 5=metadata, 6=deprecated) | Vocab categorization + canonical-emit-slot routing. |
| `<corpus>/tag_rules.yaml` | Replacements + always-remove + clothing dedup + `category_overrides` + `coverage_ignore` | Vocab-build time and inference safety net. Snapshotted into the checkpoint. |
| `<corpus>/tag_groups.yaml` | Typed groupings (`eye_color`, `hair_color`, `hair_length`, `rating`, `top_garment`, …) | Group routing during training + inference. Snapshotted into the checkpoint. |
| `<corpus>/selected/` (optional) | Curated subset (already deduped) | Additional caption source. |

`image_dataset/` (Anima's training set) is also scanned by default.

`CAPTION_CORPUS_DIR` is **not committed** — it's per-user. The trained
checkpoint snapshots `rules.yaml` + `groups.yaml` so inference has zero
runtime dependency on the corpus dir.

## Training pipeline

Seven modes, run independently via `python -m scripts.anima_tagger.cli`:

```bash
# 1. Build the vocabulary + train/val split + per-stem manifest +
#    resolved typed groups.
python -m scripts.anima_tagger.cli --mode build_vocab --min_freq 5

# 2a. Frozen-encoder path: cache pooled PE-Core features per stem.
python -m scripts.anima_tagger.cli --mode build_features

# 2b. End-to-end PE-LoRA path: cache LANCZOS-resized uint8 images instead.
python -m scripts.anima_tagger.cli --mode build_resized

# 3a. Train the head on cached features (frozen encoder; default).
python -m scripts.anima_tagger.cli --mode train --epochs 100

# 3b. End-to-end PE-LoRA training (requires --mode build_resized first).
python -m scripts.anima_tagger.cli --mode train \
    --pe_lora_rank 16 --pe_lora_layers 4 --pe_lora_lr 1e-4 --epochs 30

# 4. Sweep per-tag F1-optimal thresholds on val (skips softmax-group tags).
python -m scripts.anima_tagger.cli --mode calibrate

# 5. Single-image sanity check.
python -m scripts.anima_tagger.cli --mode predict --image foo.png --show_scores

# 6. Curator helper — find character tags that behave like affiliation markers.
python -m scripts.anima_tagger.cli --mode scan_role_markers --out_yaml stub.yaml
```

All artifacts go to `--out_dir` (default
`models/captioners/anima-tagger-v1/`):

```
models/captioners/anima-tagger-v1/
├── vocab.json              # tag list + category + median emit pos + groups
├── rules.yaml              # snapshot of tag_rules.yaml at vocab-build time
├── groups.yaml             # snapshot of tag_groups.yaml (optional)
├── dataset.json            # per-stem (image_path, multi_hot, rating) manifest
├── .cache/pooled-pe/       # per-stem [d_enc] safetensors  (frozen path)
├── .cache/resized-pe/      # per-stem uint8 [C, H, W] safetensors  (PE-LoRA path)
├── model.safetensors       # AnimaTaggerHead state_dict
├── pe_lora.safetensors     # PE-LoRA delta (only when trained with --pe_lora_rank > 0)
├── config.json             # model + training metadata (incl. pe_lora flags)
├── thresholds.safetensors  # per-tag F1-calibrated thresholds + val_f1
├── train_history.json      # per-epoch loss + val metrics
└── train_history.png       # 2-panel loss + val-F1 / rating-acc plot
```

The checkpoint is **fully self-contained** — no runtime dependency on
`CAPTION_CORPUS_DIR`, moveable across machines.

### Frozen-encoder path (`train_cached.py`)

The whole train/val feature tensor (~50 MB at 12k × 1024 × float32) lives
in VRAM after one push — no per-step dataloader. Rating-CE is class-weighted
(inverse frequency normalized to mean=1); tag-BCE uses
`sqrt(n_neg/n_pos)` per-tag pos-weight. Best macro-F1 on val saved.

### End-to-end PE-LoRA path (`train_pe_lora.py`)

Set `--pe_lora_rank > 0` and the trainer ignores the pooled cache,
reads pre-resized `uint8 [C,H,W]` from `.cache/resized-pe/`, and runs
PE-Core + mean-pool + head per step. `inject_pe_lora` (from
`networks/methods/ip_adapter_pe_lora.py`) injects a low-rank delta on the
trailing `--pe_lora_layers` resblocks (default 4) targeting QKV / attn-out
/ MLP (each toggleable). `BucketBatchSampler` groups same-shape images
into shape-homogeneous batches so the encoder forwards stay
recompile-free.

Two AdamW param groups — head at `--lr`, LoRA at `--pe_lora_lr` (default
`1e-4`). `pe_lora.safetensors` is saved alongside `model.safetensors`,
and `config.json` records every PE-LoRA flag so inference can reconstruct
the encoder state exactly.

### Group routing (`GroupRouter`)

`tag_groups.yaml` declares typed groups with one of three modes:

* **`softmax_when_solo`** — K-way CE over the group's logits when the
  sample is single-subject (`solo`/`1girl`/`1boy`/`1other` fires AND no
  multi-count tag fires) AND no `escape:` tag fires; falls back to BCE
  per-tag otherwise. Used for groups that are mutually exclusive
  on a single subject (eye color, hair color, hair length, primary
  garment) but irrelevant when an explicit escape applies (e.g.
  `heterochromia` for eye_color, `multicolored hair` for hair_color).
* **`softmax`** — always K-way CE (modulo `escape:`). Used for genuinely
  exclusive groups like rating.
* **`multilabel`** — left in BCE; the group only exists for
  introspection / UI grouping.

`compute_grouped_loss` runs BCE on every (sample, tag) and masks off the
positions where CE supervises that pair, so each cell is supervised by
exactly one term. `eval_split` reports macro-F1 over **residual**
(BCE-supervised) tags only and per-group argmax accuracy separately —
softmax-group tags' sigmoid scores are untrained noise, so the
flat macro-F1 the cached path reports is conservatively low (best 0.192
on the shipped checkpoint, with 5 active softmax groups).

### Calibration

`calibrate.py` sweeps thresholds in `[0.05, 0.95]` step `0.05` per tag and
picks the F1-maximizing one on val. Tags with no positive val examples,
zero achievable F1, or membership in a softmax group keep `default=0.5`
(softmax-group tags are routed by argmax at inference, so a
sigmoid-threshold value never fires for them). Tag-block size of 256
caps memory.

### Role-marker scan

`role_markers.py` is a read-only curator helper. It reads `vocab.json`
+ `dataset.json` and ranks every `category=='character'` tag by its
conditional co-occurrence with another character tag on **solo**
training samples (using the same `solo`/`1girl`/`1boy`/`1other` predicate
the trainer applies). Each candidate is auto-bucketed:

* **A_costume** — candidate shares a name prefix with a top partner →
  variant of an existing base. Curate via `tag_rules.yaml` `dedup:`.
* **D_role** — broad partner pool (≥ `--min_role_partners` distinct
  partners) → affiliation marker mistyped as character (`sensei (blue
  archive)`, `producer (idolmaster)`, `doctor (arknights)`). Curate via
  `tag_rules.yaml` `remove:`.
* **C_pair** — narrow partner pool (top-1 partner ≥
  `--pair_dominance` of co-occurrences) → genuine couple/sibling pair.
  Leave alone.
* **B_review** — everything else; eyeball.

`--out_yaml stub.yaml` writes a YAML stub split into pasteable sections
(A as dedup blocks, D under `remove:`, B/C as commented hints). No files
in the checkpoint dir are mutated.

## Inference

```python
from library.captioning import AnimaTagger
from PIL import Image

tagger = AnimaTagger("models/captioners/anima-tagger-v1")
caption = tagger.predict_caption(Image.open("foo.png"))
# → "sensitive, 1girl, hatsune miku, vocaloid, @some_artist, blue eyes, ..."

debug = tagger.predict(Image.open("foo.png"))
# → {"rating": "...", "rating_scores": {...}, "scores": {...},
#    "kept": {...}, "groups": {"eye_color": "blue eyes", ...}}
```

`AnimaTagger.predict`:

1. PIL → bucket-resize → IMAGE_TRANSFORMS → frozen PE-Core (+ optional
   PE-LoRA delta loaded from `pe_lora.safetensors` when `config.pe_lora`
   is true) → mean-pool → trunk → tag_logits + rating_logits.
2. `sigmoid(tag_logits) ≥ thresholds` → `kept`; `argmax(rating_logits)`
   → rating.
3. **Group-aware refinement.** For each loaded `softmax`/`softmax_when_solo`
   group, when the gating predicate applies (single-subject for
   `softmax_when_solo`, always for `softmax`, both modulo escape tags),
   replace any sigmoid-admitted members with the single argmax winner
   over the group's logits.
4. **Girls-count cap.** When `kept` contains digit-prefixed `Ngirls`, trim
   character predictions to the top-`max(N)` by score — caps the
   independent-sigmoid leakage on gender-ambiguous art.
5. **Character floor + original fallback.** Any character below
   `character_floor` (default `0.5`, sits above some F1 thresholds as
   low as `0.05` for noisy long-tail characters) is dropped. When that
   empties the character slot AND no copyright tag survives, add
   `original` (booru convention for non-IP work) so the caption still
   has a slot-filling copyright.
6. **Top-1 artist + top-1 copyright.** Independent sigmoid heads can
   admit several borderline tags; collapse to the highest-scoring one
   (booru convention is one artist / one copyright per work).

`predict_caption` then slots tags by canonical category order
(`rating, count, character, copyright, artist, general`), within-slot by
median emit position from the training corpus, re-applies `tag_rules` as
a safety net (the dedup map already fired during training-data
normalization, but the model could in principle predict both `bra` and
`black bra`), replaces underscores with spaces, and joins with `, `.

## Wired-up touchpoints

### CLI driver

`scripts/experimental_tasks/inference.py::cmd_test_directedit` runs the
Anima Tagger on the source image to seed `--prompt_src`:

```bash
make exp-test-directedit PROMPT='glasses'
```

Requires `models/captioners/anima-tagger-v1/model.safetensors`. The
driver exits with a clear error if the checkpoint is missing — train it
via `python -m scripts.anima_tagger.cli`.

`scripts/edit.py` itself doesn't tag — it takes `--prompt_src` directly.
Tagging only happens in the make-target driver (CLI) or the ComfyUI node
(see below).

### ComfyUI nodes (`custom_nodes/comfyui-anima-tagger/`)

Two nodes share the `ANIMA_TAGGER` socket type:

| Node | Inputs | Outputs |
|------|--------|---------|
| `AnimaTaggerLoader` | `tagger_dir` (STRING) | `tagger` (ANIMA_TAGGER) |
| `AnimaTaggerCaption` | `tagger` (ANIMA_TAGGER), `image` (IMAGE) | `caption` (STRING) |

The package supports two install shapes:

1. **Inside the anima_lora repo** (dev / monorepo). Imports the live
   `library.captioning.anima_tagger`.
2. **Standalone** (just this directory dropped into vanilla ComfyUI
   `custom_nodes/`). Falls back to a bundled inference subset under
   `_vendor/`, regenerated by `python scripts/sync_vendor.py` from the
   live tree before bumping the node version.

PE-Core-L14-336 (~1 GB) is auto-fetched from `facebook/PE-Core-L14-336`
on first use into the `pe_ckpt` path on the loader.

`AnimaTaggerCaption` outputs a STRING that drops into any text input —
DirectEdit's `ANIMA_TAGGER` socket, `CLIPTextEncode` for prompt
scaffolding, or `Save Text File` for LoRA dataset pre-fill.

## Known limitations

1. **Rating-class imbalance.** Train-corpus rating mix is ~67% explicit
   / ~32% sensitive / ~0.6% general. Class-weighted CE compensates
   partially. If `general`-rating accuracy matters downstream, oversample
   at training time.
2. **Per-tag positives are thin for the long tail.** At `min_freq=5`
   each long-tail tag has 5–20 positives; calibrated thresholds for those
   tags are noisier than for high-frequency ones. `--min_freq 10` is a
   knob to revisit if F1 disappoints.
3. **No bench harness yet.** `bench/anima_tagger/` per the standard
   envelope (cf. `bench/_common.py::write_result`) is the next thing to
   add — should report F1 on a held-out set plus a downstream
   "edit-success-rate" metric on a small DirectEdit set.
4. **Long-tail characters benefit from `character_floor`.** Some F1
   thresholds settle as low as `0.05` for noisy long-tail characters;
   the post-prediction floor (default `0.5`) is what stops borderline
   guesses from leaking into ψ_src on stylized / gender-ambiguous art.
   Lowering the floor recovers recall at the cost of precision.

## Open design questions

1. **DINOv3 trunk swap.** gelcrawl's `classify.py` uses DINOv3 ViT-L/16@224
   and works well in this domain. If F1 saturates and we suspect the
   trunk is the limit, swap encoders — `--encoder` flag already plumbs
   through the loader registry.
2. **Embedding output instead of tag string.** `predict_caption` emits a
   string that gets re-tokenized by T5. We could add a head producing
   `[K, D_t5]` continuous tokens directly — but that's the img2emb design
   (`docs/proposal/img2emb_plan.md`) and hits the same structural
   challenges. Stick with tag-string output for now.
