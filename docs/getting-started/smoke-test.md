---
title: End-to-end smoke test
description: Run the full data → tag → train pipeline against a real character set.
---

# End-to-end smoke test

Once you have kohya-ss/sd-scripts installed (set `LORAHUB_KOHYA_SD_SCRIPTS` or
copy `.env.example`) and an SDXL base model on disk, the full path from zero
to a trained LoRA looks like this:

```powershell
# 1. Pull a character's images from BangumiBase
lorahub fetch-bangumi azurlaneanime 5 --output ./datasets/laffey --limit 50

# 2. Auto-tag every image
lorahub tag ./datasets/laffey

# 3. Scaffold a recipe and edit it (point base_model.checkpoint at your SDXL .safetensors)
lorahub init smoke
notepad smoke.yaml

# 4. Sanity check
lorahub validate smoke.yaml
lorahub info     smoke.yaml

# 5. Train
lorahub train    smoke.yaml
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
auto-tag with `lorahub tag`) before training.

## Auto-tag a dataset

`lorahub tag` runs the WD14 / WD-v3 ONNX tagger over a directory and writes
kohya-style `.txt` captions next to each image.

```powershell
# Default thresholds (general=0.35, character=0.85), skips images that already have a non-empty caption
lorahub tag ./datasets/akagi

# Re-tag everything from scratch with a tighter general threshold
lorahub tag ./datasets/akagi --overwrite --general 0.45

# Skip the character tag if you're training a style or concept LoRA
lorahub tag ./datasets/akagi --no-include-character
```

The first run downloads ~400 MB of ONNX weights from Hugging Face (cached for
subsequent runs). CPU inference handles hundreds of images at ~1 s/image; for
batch throughput install the GPU runtime:

```powershell
pip uninstall onnxruntime
pip install lorahub[gpu]              # or: pip install onnxruntime-gpu
lorahub tag ./datasets/akagi --device cuda
```

`--device auto` picks GPU when `onnxruntime-gpu` and a CUDA 12.x runtime are
present, otherwise falls back to CPU. `--device cuda` forces GPU and errors
out with an actionable message if it isn't available.

### JoyTag (PyTorch backend)

LoraHub also ships a JoyTag adapter that hosts the
[`fancyfeast/joytag`](https://huggingface.co/fancyfeast/joytag) ViT model
end-to-end in PyTorch — useful when you want richer booru tags
(~5800-tag vocabulary, default 0.4 threshold) than WD14. The architecture is
vendored under `lorahub/core/tagging/_joytag_model.py` so no
`timm` / `einops` / `transformers` extras are pulled in. Install the optional
`tagging` extras to get PyTorch + safetensors:

```powershell
pip install "lorahub[tagging]"
# or pick a CUDA wheel manually from https://pytorch.org/get-started/locally/
```

The first run downloads ~700 MB of safetensors weights plus `config.json` and
`top_tags.txt` from the Hub.
