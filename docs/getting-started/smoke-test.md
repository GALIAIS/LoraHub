---
title: End-to-end smoke test
description: Run the full data → caption → train → preview pipeline against a real character set.
---

# End-to-end smoke test

Once a backend is installed (set `LORAHUB_KOHYA_SD_SCRIPTS` /
`LORAHUB_DIFFUSION_PIPE` or copy `.env.example`) and a base model is on disk,
the full path from zero to a trained LoRA + live previews looks like this:

```powershell
# 1. Pull a character's images from BangumiBase
lorahub fetch-bangumi azurlaneanime 5 --output ./datasets/laffey --limit 50

# 2. Caption every image — pick one of:

#    a) Classic auto-tag (WD14 / WD-v3 ONNX, fastest):
lorahub tag ./datasets/laffey

#    b) Smart caption (WD14 + vision LLM, Anima-format):
curl -X POST http://127.0.0.1:18765/api/image-studio/ai/smart-caption \
     -H 'Content-Type: application/json' \
     -d '{"path":"./datasets/laffey","captionMode":"character","triggerWord":"laffey"}'

# 3. Scaffold a config and edit it (point baseModel.checkpoint at your model)
lorahub init smoke
notepad configs/smoke.yaml

# 4. Sanity check
lorahub validate configs/smoke.yaml
lorahub info     configs/smoke.yaml

# 5. Train
lorahub train    configs/smoke.yaml
```

!!! success "Reference timing"
    On an RTX 4070 Laptop (8 GB VRAM) using IllustriousXL as the base, the
    full path produces a 21 MB SDXL LoRA file in under 3 minutes — verified
    with 3 BangumiBase images of "laffey (azur lane)" at 512×512, 2 steps.

## Need test data fast?

`lorahub fetch-bangumi` pulls a single character's image set from the
[BangumiBase](https://huggingface.co/BangumiBase) Hugging Face datasets —
pre-clustered, MIT-licensed, ready for smoke testing.

```powershell
# List characters in a show
lorahub fetch-bangumi azurlaneanime

# Grab character 5 with up to 50 images
lorahub fetch-bangumi azurlaneanime 5 --output ./datasets/akagi --limit 50

# Or pull the 8 preview thumbnails first to identify the character
lorahub fetch-bangumi azurlaneanime 5 --preview --output ./datasets/akagi
```

Each image lands next to an empty `.txt` caption file — fill them in (or
auto-caption with the next steps) before training.

## Auto-tag with WD14 / JoyTag

`lorahub tag` runs a tagger over a directory and writes kohya-style `.txt`
captions next to each image.

```powershell
# Default thresholds (general=0.35, character=0.85), skips images that already have captions
lorahub tag ./datasets/akagi

# Re-tag everything from scratch with a tighter general threshold
lorahub tag ./datasets/akagi --overwrite --general 0.45

# Skip the character tag if you're training a style or concept LoRA
lorahub tag ./datasets/akagi --no-include-character

# JoyTag (PyTorch backend, ~5800-tag vocabulary, default 0.4 threshold)
lorahub tag ./datasets/akagi --tagger joytag --joytag-threshold 0.4
```

Default WD14 model is `SmilingWolf/wd-eva02-large-tagger-v3`. CPU inference
handles hundreds of images at ~1 s/image; for batch throughput install the
GPU runtime:

```powershell
pip uninstall onnxruntime
pip install lorahub[gpu]              # or: pip install onnxruntime-gpu
lorahub tag ./datasets/akagi --device cuda
```

`--device auto` picks GPU when `onnxruntime-gpu` and a CUDA 12.x runtime are
present, otherwise falls back to CPU. `--device cuda` forces GPU and errors
out with an actionable message if it isn't available.

## Smart caption (WD14 + vision LLM)

The Image Studio's smart-caption pipeline combines WD14 with a configured
vision LLM to produce Anima-format captions:

```
masterpiece, best quality, score_7, <safe|sensitive|nsfw>,
<1girl/solo/character>, @<trigger>,
<2-3 sentence natural-language description>,
<remaining general tags>
```

Three modes:

- **style** — describe the medium and rendering on purpose so the model
  binds the style to the trigger word.
- **character** — skip fixed identity features (hair / eye colour, signature
  outfit) so the model learns them from the latent.
- **general** — describe everything; useful when the dataset isn't
  trigger-word-driven.

Each line you produce can be re-rendered into a preview image by the live
preview worker if you train via diffusion-pipe (see the next section).

## Live previews during training

When you train with `diffusion-pipe`, lorahub spins up a background worker
that watches `runs/<job>/output/{step|epoch}{N}/` and renders one PNG per
prompt for every new checkpoint. Default prompts live at
`configs/sample_prompts/anima_default.txt`; switch the trigger inside the
file to retarget.

The worker reacts in <1 s to `checkpoint_saved` events and falls back to a
5 s polling tick. Per-checkpoint render budget keeps training throughput
within ~30% of baseline. Skipped renders (low free VRAM, cancellation) are
silently rescheduled — only true crashes raise an error event.

The PNGs land in `workspace/samples/step{N}_{idx}.png` and surface live in
the **Sample Gallery** of the Jobs page.
