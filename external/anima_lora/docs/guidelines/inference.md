# Inference Reference

## Basic usage

```bash
python inference.py \
    --dit models/diffusion_models/anima-base-v1.0.safetensors \
    --text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
    --vae models/vae/qwen_image_vae.safetensors \
    --lora_weight output/ckpt/anima_lora.safetensors \
    --prompt "your prompt" \
    --image_size 1024 1024 \
    --infer_steps 28 \
    --flow_shift 1.0 \
    --guidance_scale 4.0 \
    --sampler er_sde \
    --attn_mode flash \
    --save_path output/tests
```

The canned `make test` target picks the latest bakeable LoRA in `output/ckpt/`
and runs against the values above (`INFERENCE_BASE` in
`scripts/tasks/_common.py`). The `inference.py` argparse defaults differ
(`--infer_steps 50`, `--flow_shift 5.0`, `--guidance_scale 3.5`,
`--attn_mode torch`, `--sampler euler`) — `make test` is the more
representative starting point.

## Test targets

| Target | What it runs |
|--------|--------------|
| `make test` | Latest LoRA |
| `make test SPECTRUM=1` | Latest LoRA + Spectrum acceleration |
| `make test MOD=1` | Latest LoRA + distilled `pooled_text_proj` (modulation guidance). Composes with `SPECTRUM=1`. |
| `make test NOLORA=1` | Bare DiT (skips `--lora_weight`). Compose with `MOD=1` for a mod-only sample. |
| `make test-hydra` | Latest HydraLoRA / FeRA `*_moe.safetensors` (router-live) |
| `make test-merge` | Inference against a baked DiT under `MODEL_DIR=` |
| `make test-dcw` | Latest LoRA + DCW scalar bias correction (λ = -0.015) |
| `make test-dcw-v4` | Latest LoRA + DCW v4 learnable calibrator (auto-resolves head) |
| `make test-spectrum-dcw` | Spectrum + DCW scalar |
| `make test-dcw-v4-spectrum` | Spectrum + DCW v4 |
| `make exp-test-postfix` | Postfix tuning (also `-exp`, `-func` variants) |
| `make exp-test-ip REF_IMAGE=...` | IP-Adapter (image-conditioned) |
| `make exp-test-easycontrol REF_IMAGE=...` | EasyControl |
| `make exp-test-directedit PROMPT='...'` | DirectEdit on a random source image |
| `make exp-test-directedit-dry` | DirectEdit reconstruction sanity check |

## Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--lora_weight` | — | Adapter path(s); space-separated for stacking |
| `--lora_multiplier` | 1.0 | Scalar (one per weight or one for all) |
| `--postfix_weight` | — | Postfix-tuning vectors |
| `--ip_adapter_weight` | — | IP-Adapter checkpoint |
| `--ip_image` | — | Reference image for IP-Adapter |
| `--ip_scale` | 1.0 | IP-Adapter image-CFG scale |
| `--ip_image_match_size` | off | Match target H/W to the ref image bucket |
| `--easycontrol_weight` | — | EasyControl checkpoint |
| `--easycontrol_image` | — | Reference image for EasyControl |
| `--easycontrol_scale` | 1.0 | EasyControl scale |
| `--easycontrol_image_match_size` | off | Match target H/W to ref bucket |
| `--infer_steps` | 50 | Denoising steps (28 via `make test`) |
| `--guidance_scale` | 3.5 | Text CFG scale (4.0 via `make test`) |
| `--flow_shift` | 5.0 | Flow-matching schedule shift (1.0 via `make test`) |
| `--sampler` | `euler` | `euler` (deterministic ODE), `er_sde` (stochastic), `lcm` (x0 re-noise — distilled few-step models) |
| `--attn_mode` | `torch` | `torch` / `flash` / `flex` / `sageattn` / `xformers` (`sdpa` alias) |
| `--from_file` | — | Batch prompts from a text file |
| `--interactive` | off | REPL prompt loop |
| `--compile` | off | `torch.compile` speedup |
| `--text_encoder_cpu` | off | Keep the text encoder on CPU (low-VRAM) |
| `--vae_chunk_size` | — | VAE decode tile size |
| `--vae_disable_cache` | off | Skip the per-tile cache |
| `--no_metadata` | off | Don't embed training metadata in the PNG |
| `--save_path` | — | Output directory |

`--fp8` and `--prefix_weight` were removed. The legacy `--dcw_v4 <head>`
alias still parses but maps to `--dcw_calibrator`.

## Modulation guidance

Distilled `pooled_text_proj` MLP steers AdaLN coefficients at inference. See
[`../methods/mod-guidance.md`](../methods/mod-guidance.md).

| Flag | Description |
|------|-------------|
| `--pooled_text_proj` | Path to the distilled MLP |
| `--mod_w` | Guidance strength (positive boosts) |
| `--mod_pos_prompt` / `--mod_neg_prompt` | Text targets for the AdaLN delta |
| `--mod_start_layer` / `--mod_end_layer` | Layer band to apply the delta on |
| `--mod_taper` / `--mod_taper_scale` / `--mod_final_w` | Schedule shaping |

## Spectrum acceleration

Training-free; Chebyshev feature forecasting skips most blocks on cached
steps. See [`../methods/spectrum.md`](../methods/spectrum.md).

| Flag | Description |
|------|-------------|
| `--spectrum` | Enable |
| `--spectrum_warmup` | Steps before caching starts |
| `--spectrum_window_size` / `--spectrum_flex_window` | Adaptive window schedule |
| `--spectrum_w` / `--spectrum_m` / `--spectrum_lam` | Forecasting hyperparameters |
| `--spectrum_stop_caching_step` | Last cached step |
| `--spectrum_calibration` | Bias adjustment |

## DCW (sampler-level SNR-t bias correction)

Composes with every adapter and with Spectrum. Two modes — see
[`../methods/dcw.md`](../methods/dcw.md).

| Flag | Description |
|------|-------------|
| `--dcw` | Scalar mode (one global λ) |
| `--dcw_lambda` | λ value (`make test-dcw` default: -0.015) |
| `--dcw_band_mask` | Where to apply: `LL` (default) / `HF` / `all` |
| `--dcw_schedule` | Per-step shaping (e.g. flat / cosine) |
| `--dcw_calibrator` | Path to a v4 fusion-head safetensors (replaces scalar) |
| `--dcw_calibrator_gain` | Multiplicative scale on the calibrator's α̂ |

The shipped scalar default (-0.015) is tuned for CFG=1; at production CFG=4
the bias direction is (CFG × aspect)-dependent — non-square aspects want
small **positive** λ. The Spectrum ComfyUI node ships +0.01 as its scalar
default. Prefer `--dcw_calibrator` for production runs.

## P-GRAFT

Dynamic-hook variant that lets you cut LoRA off mid-trajectory (originally
for prefix inversion; useful as a debug switch).

```bash
python inference.py ... \
    --pgraft \
    --lora_cutoff_step 37    # LoRA active for steps 0–36, disabled 37+
```

## Tiled decode

For high-resolution outputs that don't fit the VAE in one shot:

```bash
--tiled_diffusion --tile_size 1024 --tile_overlap 64
```

## Prompt-file format

One prompt per line; flags can be appended inline.

```
a girl standing in a field --w 1024 --h 1024 --s 50 --g 5.0
another prompt --seed 42 --flow_shift 4.0
```

## LoRA in ComfyUI

Plain Anima LoRA `.safetensors` use kohya-ss `lora_unet_` key naming and
load directly into ComfyUI's stock `LoraLoader` — no conversion. For
HydraLoRA / FeRA / ReFT / postfix / prefix checkpoints (extra `router.*`,
`reft_*`, stacked `lora_ups.N.*` keys that the stock loader drops), use the
**Anima Adapter Loader** in `custom_nodes/comfyui-hydralora/`.

Spectrum KSampler + mod-guidance + in-node DCW (scalar default +0.01, plus
an `auto` mode that runs the v4 fusion head) live in
[ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler).
