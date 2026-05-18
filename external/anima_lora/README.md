# anima_lora

[한국어](README.ko.md) · 📖 [가이드북 (Windows 초보자용 한국어 종합 가이드)](docs/guidelines/가이드북.md)

LoRA / T-LoRA training and inference engine for the [Anima](https://huggingface.co/circlestone-labs/Anima) diffusion model (DiT-based, flow-matching).

Four things this repo aims to do well:

1. **Fast LoRA training** on consumer GPUs — full-model `torch.compile` with CUDAGraph capture, end to end.
2. **Solid conventional implementations** — LoRA, OrthoLoRA, and T-LoRA stack together and bake losslessly into a standalone DiT checkpoint.
3. **Recent methods, engineered for Anima** — Spectrum inference, DCW calibrator, OrthoHydraLoRA, and modulation guidance, each implemented end-to-end against Anima's compile/CUDAGraph contract rather than dropped in as a toy port.
4. **A broad experimental surface** — ReFT, postfix/prefix tuning, IP-Adapter, EasyControl, embedding inversion, img2emb, GRAFT.

> **At-a-glance diagrams** for every method (DiT internals, LoRA, OrthoLoRA, T-LoRA, HydraLoRA, ReFT, Spectrum, modulation, compile optimizations) live in [`docs/structure_images/`](docs/structure_images/) — paired with prose walkthroughs in [`docs/structure/`](docs/structure/).

---

## 1. Fast training

**13.4 GB peak VRAM · 1.1 s/step** on a single RTX 5060 Ti while **rank=32 1MP resolution lora training** — achieved by co-designing the data pipeline, attention, and compiler stack so Dynamo sees one static shape for the whole run.

| Lever | Summary |
|---|---|
| Constant-token bucketing | All buckets target `(H/16)×(W/16) ≈ 4096` patches; batches zero-pad to exactly 4096. One static shape, no compile recompilation. |
| Max-padded text encoder | Text outputs padded to 512 and zero-filled — the pretrained DiT uses zero keys as cross-attn sinks, so trimming breaks it. Also gives the compiler another fixed dim. |
| Per-block `torch.compile` (default) | Each DiT block compiled independently with Inductor. Combined with static tokens this eliminates guard recompilation. |
| Full-model compile + CUDAGraph (opt-in) | Set `compile_mode = "full"` + `compile_inductor_mode = "reduce-overhead"` and Inductor sees the whole 28-block stack while `cudagraph_trees` captures one graph that replays every step — no per-block kernel boundary, no per-step launch overhead. Forces the static-shape contract end to end; incompatible with `gradient_checkpointing` and `blocks_to_swap`. See [full_model_cudagraph.md](docs/optimizations/full_model_cudagraph.md). |
| Compile-friendly hot path | Audited every forward for patterns dynamo can't trace cleanly — `einops.rearrange` replaced with explicit `.unflatten()/.permute()` chains, `torch.autocast` context managers replaced with direct `.to(dtype)` casts, dict `.items()` loops hoisted out of compiled regions, FA4 wrapped in `@torch.compiler.disable` for clean graph breaks. |
| Flash Attention 2 | `flash_attn` 2.x with SDPA fallback. FA4 evaluated and removed — see [fa4.md](docs/optimizations/fa4.md). |

Compile pipeline details in [docs/optimizations/for_compile.md](docs/optimizations/for_compile.md); full-model + CUDAGraph design in [docs/optimizations/full_model_cudagraph.md](docs/optimizations/full_model_cudagraph.md).

---

## 2. Solid conventional implementations

The default training config stacks **LoRA + OrthoLoRA + T-LoRA** together. All three fold losslessly into a standalone DiT checkpoint via thin-SVD export at save time, so you can ship ComfyUI-compatible `*_merged.safetensors` with no adapter loader dependency.

| Variant | Pitch | Details |
|---|---|---|
| **LoRA** | Classic low-rank, rank 16–32. | — |
| **OrthoLoRA** | SVD-parameterized with orthogonality regularization; exports as plain LoRA. | [psoft-integrated-ortholora.md](docs/methods/psoft-integrated-ortholora.md) |
| **T-LoRA** | Timestep-dependent rank masking — low rank at high noise, full rank at low noise. Training-only mask, so merge is bit-equivalent. | [timestep_mask.md](docs/methods/timestep_mask.md) |

**Side-by-side** — same prompt, `er_sde` 30 steps, `cfg=4.0`, 1024². Each LoRA trained at rank 16 for 2 epochs on a 20% subset with training seed 42; inference seeds `{41, 42, 43}`. Reproduce with `python archive/bench_methods.py`.

|  | **LoRA** | **OrthoLoRA + T-LoRA** |
|:---:|:---:|:---:|
| seed 41 | <img src="docs/side_by_side/lora/20260423-154854-014_41_.png" width="320"> | <img src="docs/side_by_side/ortho_tlora/20260423-155545-258_41_.png" width="320"> |
| seed 42 | <img src="docs/side_by_side/lora/20260423-154938-584_42_.png" width="320"> | <img src="docs/side_by_side/ortho_tlora/20260423-155631-762_42_.png" width="320"> |
| seed 43 | <img src="docs/side_by_side/lora/20260423-155024-080_43_.png" width="320"> | <img src="docs/side_by_side/ortho_tlora/20260423-155718-280_43_.png" width="320"> |

<details>
<summary>Base model and individual variants (plain, OrthoLoRA, T-LoRA)</summary>

|  | **plain (base)** | **OrthoLoRA** | **T-LoRA** |
|:---:|:---:|:---:|:---:|
| seed 41 | <img src="docs/side_by_side/plain/20260423-160513-382_41_.png" width="240"> | <img src="docs/side_by_side/ortholora/20260423-155109-338_41_.png" width="240"> | <img src="docs/side_by_side/tlora/20260423-155327-834_41_.png" width="240"> |
| seed 42 | <img src="docs/side_by_side/plain/20260423-160556-697_42_.png" width="240"> | <img src="docs/side_by_side/ortholora/20260423-155155-526_42_.png" width="240"> | <img src="docs/side_by_side/tlora/20260423-155413-304_42_.png" width="240"> |
| seed 43 | <img src="docs/side_by_side/plain/20260423-160640-759_43_.png" width="240"> | <img src="docs/side_by_side/ortholora/20260423-155241-905_43_.png" width="240"> | <img src="docs/side_by_side/tlora/20260423-155458-996_43_.png" width="240"> |

</details>

**Merging**:

```bash
make merge                                  # bake latest LoRA at multiplier 1.0
make merge ADAPTER_DIR=output/ckpt MULTIPLIER=0.8
```

Refuses non-linear-delta variants (ReFT / HydraLoRA `_moe` / postfix / prefix) by default; `--allow-partial` drops those and bakes only the LoRA portion.

---

## 3. Recent methods, engineered for Anima

Four recent papers picked up, implemented against Anima end-to-end, and shipped with the engineering they need to be actually usable — not toy reimplementations.

| Method | What it is | Engineering notes | Doc |
|---|---|---|---|
| **Spectrum inference** | Training-free ~3.75× speedup via Chebyshev polynomial feature forecasting (Han et al., CVPR 2026). On cached steps every transformer block is skipped — only `t_embedder` + `final_layer` + `unpatchify` run. | `register_forward_pre_hook` on `final_layer` captures block outputs without monkey-patching the model; adaptive window schedule concentrates real forwards on early high-noise steps. Stable ComfyUI node in a separate repo: [ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler). | [spectrum.md](docs/methods/spectrum.md) |
| **DCW calibrator** | Sampler-level SNR-t bias correction (Yu et al., CVPR 2026) — mixes each Euler step's `prev_sample` toward the model's `x0_pred` along the LL Haar band. Two modes: scalar `λ` (offline-tuned) and **v4 learnable** per-prompt calibrator with online observation. | v4 head conditions on `(aspect, prompt, observed prefix gap)` and fires after `k=7` warmup steps. Bias direction characterized as **(CFG × aspect)-dependent** on Anima — paper-direction at CFG=4 non-square, paper-opposite at CFG=1 / 1024². Trained per-checkpoint via `make dcw`. | [dcw.md](docs/methods/dcw.md) |
| **OrthoHydraLoRA** | MoE-style multi-head LoRA with orthogonalized experts and layer-local routing — shared `lora_down`, per-expert `lora_up_i`, learned per-sample router. Targets multi-style training without the cross-style bleed a single low-rank subspace produces. Original paper: [arXiv:2605.03252](https://arxiv.org/abs/2605.03252). | Saves two side-by-side files: `anima_hydra.safetensors` (baked-down LoRA, ComfyUI drop-in) and `anima_hydra_moe.safetensors` (full multi-head). Live routing in ComfyUI via the bundled **Anima Adapter Loader** node (`custom_nodes/comfyui-hydralora/`), which installs per-Linear forward hooks reproducing `HydraLoRAModule.forward`. | [hydra-lora.md](docs/methods/hydra-lora.md) |
| **Modulation guidance** | Distill a `pooled_text_proj` MLP that steers AdaLN modulation coefficients toward quality-positive directions (Starodubcev et al., ICLR 2026). Teacher sees real cross-attention; student sees zeroed cross-attention but receives pooled text through modulation. | Trained with `make distill-mod` against the frozen DiT. Inference applies the projection at AdaLN time so it composes with any LoRA variant; `make test MOD=1` runs a sample with it enabled (composes with `SPECTRUM=1`). | [mod-guidance.md](docs/methods/mod-guidance.md) |

---

## 4. Experimental surface

Each ships with a doc — see the link for usage, flags, and caveats.

| Feature | What it is | Doc |
|---|---|---|
| **ReFT** | Block-level residual-stream intervention (LoReFT, NeurIPS 2024). Composes with any LoRA variant. | [reft.md](docs/methods/reft.md) |
| **Postfix (cond+ortho)** | Caption-conditional postfix vectors with structural orthogonality (Cayley-rotated frozen SVD basis). DiT frozen; only `cond_mlp` trains. | [postfix.md](docs/experimental/postfix.md) |
| **IP-Adapter** | Decoupled image cross-attention (Ye et al. 2023). DiT frozen; trains Perceiver resampler + per-block `to_k_ip`/`to_v_ip`. | [ip-adapter.md](docs/experimental/ip-adapter.md) |
| **EasyControl** | Extended self-attention image conditioning. DiT frozen; trains per-block cond LoRA on self-attn + FFN + scalar `b_cond` gate. | [easycontrol.md](docs/experimental/easycontrol.md) |
| **Embedding inversion** | Optimize a text embedding to match a target image through the frozen DiT. | [invert.md](docs/methods/invert.md) |
| **img2emb resampler** | Learn a reference-image → embedding mapping via TIPSv2-L/14 features + anchor injection. | [archive/img2emb/README.md](archive/img2emb/README.md) |
| **GRAFT** | Rejection-sampling fine-tuning — train, generate, curate survivors, retrain. | [graft-guideline.md](docs/guidelines/graft-guideline.md) |

> **Want to contribute?** Two areas where outside help would have outsized impact: **IP-Adapter productionization** (tests, public reference checkpoint, lighter vision encoder) and **EasyControl adapters** (canny / depth / pose / … — each control type is one self-contained PR). See [CONTRIBUTING.md → Priority areas](CONTRIBUTING.md#priority-areas).

---

## Setup

```bash
uv sync                   # Python 3.13 with pre-built flash attention 2
hf auth login
make download-models      # DiT + Qwen3 text encoder + QwenImage VAE into models/
# place training images in image_dataset/ with .txt caption sidecars
make gui                  # recommended — config editor + dataset browser + training monitor
```

`uv sync` resolves to **torch 2.12 nightly + CUDA 13.2 on Linux** and **torch 2.11 stable + CUDA 13.0 on Windows**. Windows users who want the ~10% throughput win from CUDA 13.2 + nightly torch can switch by editing the comment-toggled lines in `pyproject.toml` (search for `cuda132 opt-in`) and re-running `uv sync`. Prerequisites (CUDA 13.2 toolkit + a trimmed FA2 wheel) are in [docs/optimizations/cuda132.md](docs/optimizations/cuda132.md).

CLI path:

```bash
make preprocess           # VAE-compatible resize & validation
make lora                 # or: PRESET=fast_16gb make lora / PRESET=low_vram make lora / make exp-postfix
make test                 # sample generation with the latest trained LoRA
```

Config chain: `configs/base.toml → configs/presets.toml[<preset>] → configs/methods/<method>.toml → CLI args`. Override with `PRESET=low_vram make lora` or `--network_dim 32 --max_train_epochs 64`. Full flag reference in [docs/guidelines/training.md](docs/guidelines/training.md) and [docs/guidelines/inference.md](docs/guidelines/inference.md).

---

## Documentation

| Doc | Contents |
|-----|----------|
| [guidelines/training.md](docs/guidelines/training.md) | Training flags, LoRA variants, caption shuffle, masked loss, dataset config |
| [guidelines/inference.md](docs/guidelines/inference.md) | Inference flags, P-GRAFT, prompt files, LoRA format conversion |
| [guidelines/graft-guideline.md](docs/guidelines/graft-guideline.md) | GRAFT curation workflow |
| [optimizations/](docs/optimizations/) | Compile pipeline, FA4 post-mortem, CUDA 13.2 |
| [methods/](docs/methods/) | One doc per method — HydraLoRA, ReFT, Spectrum, inversion, mod guidance, postfix/prefix, T-LoRA, OrthoLoRA |

---

## License

Toolkit code: [MIT](LICENSE).

Anima / CircleStone **base model weights** ship under the **CircleStone Labs Non-Commercial License v1.0** and are not relicensed by this repo. Any LoRA, fine-tune, or merged checkpoint trained from those weights is a Derivative and inherits the non-commercial terms. See [NOTICE](NOTICE).
