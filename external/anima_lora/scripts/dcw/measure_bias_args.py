"""Argument parser for ``scripts/dcw/measure_bias.py``.

Split out from ``measure_bias.py`` to keep the entry-point script focused
on orchestration. The ``description`` is passed in by the caller so the
``--help`` text remains the measure_bias module docstring.
"""

from __future__ import annotations

import argparse


def parse_args(description: str | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dit",
        type=str,
        default="models/diffusion_models/anima-base-v1.0.safetensors",
        help="DiT .safetensors path (default: anima-base-v1.0).",
    )
    p.add_argument(
        "--lora_weight",
        type=str,
        nargs="+",
        default=None,
        help="Optional LoRA / HydraLoRA adapter(s) to stack on the base DiT. "
        "Auto-detects HydraLoRA moe (lora_ups.* keys) and attaches router-live "
        "via dynamic forward hooks; plain LoRA goes through the same dynamic path "
        "(math-equivalent to static merge for this measurement).",
    )
    p.add_argument(
        "--lora_multiplier",
        type=float,
        nargs="+",
        default=[1.0],
        help="Multiplier per --lora_weight entry (broadcast if a single value).",
    )
    p.add_argument(
        "--dataset_dir",
        type=str,
        default="post_image_dataset/lora",
        help="Directory with cached *_anima.npz + *_anima_te.safetensors pairs.",
    )
    p.add_argument(
        "--text_variant",
        type=int,
        default=0,
        help="Cached caption variant (crossattn_emb_v<N>); 0 = canonical.",
    )
    p.add_argument(
        "--attn_mode",
        type=str,
        default="flash",
        help="torch | sdpa | xformers | sage | flash",
    )
    p.add_argument("--n_images", type=int, default=2, help="Cached samples to use")
    p.add_argument("--n_seeds", type=int, default=1, help="Seeds per sample")
    p.add_argument(
        "--image_h",
        type=int,
        default=None,
        help="Restrict to cached samples with this image-space height (the "
        "<H> in <stem>_<H>x<W>_anima.npz). Required (with --image_w) for "
        "--compile to converge to a single graph and for direct cross-run "
        "comparability of velocity norms.",
    )
    p.add_argument(
        "--image_w",
        type=int,
        default=None,
        help="Restrict to cached samples with this image-space width.",
    )
    p.add_argument(
        "--shuffle_seed",
        type=int,
        default=None,
        help="Deterministically shuffle the candidate pool before truncating "
        "to --n_images. Default None preserves alphabetical-first selection "
        "(legacy behavior). Used by `make dcw` to broaden prompt diversity "
        "beyond the alphabetical-first 35 stems per bucket.",
    )
    p.add_argument(
        "--exclude_stems",
        type=str,
        default=None,
        help="Path to a text file (one stem per line) listing cached samples "
        "to skip when building the candidate pool. Used by `make dcw` to "
        "dedup against prior runs' manifest.json so incremental gathers grow "
        "the calibration pool monotonically. Lines starting with '#' and "
        "blank lines are ignored.",
    )
    p.add_argument(
        "--infer_steps",
        type=int,
        default=28,
        help="Inference schedule length (v2 prod env = 28).",
    )
    p.add_argument(
        "--flow_shift", type=float, default=1.0, help="σ shift (matches inference.py)."
    )
    p.add_argument("--seed_base", type=int, default=1234)
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile the DiT before the bench loop. Each unique latent "
        "(H, W) pays a one-time warm-up; steady-state is much faster. Best "
        "amortized when n_images is moderate (every sample's ~150+ forwards "
        "run at the same shape, and dynamo auto-flips to dynamic shapes after "
        "the second distinct (H, W)). Pass --no-compile to disable.",
    )
    p.add_argument(
        "--dcw_sweep",
        action="store_true",
        help="Also run LL-only DCW-corrected trajectories at --dcw_scalers "
        "(one_minus_sigma schedule). Used by v2 §A4 to estimate S_pop(σ_i).",
    )
    p.add_argument(
        "--dcw_scalers",
        type=float,
        nargs="+",
        default=[0.010],
        help="λ values to sweep when --dcw_sweep is set (negative on Anima; "
        "v2 §A4 uses {0, -0.015, -0.020, -0.025}).",
    )
    p.add_argument(
        "--baseline_lambda",
        type=float,
        default=0.0,
        help="LL-only DCW λ applied (one_minus_sigma schedule) on every step "
        "of the seed-batched 'baseline' reverse trajectory. Default 0 keeps "
        "the legacy no-DCW baseline. Set to e.g. 0.01 to collect data under "
        "the make-test-dcw scalar so the trained head emits a residual α̂ "
        "on top — eliminates the v4 dead-zone mismatch (warmup steps see "
        "the same correction as the rest, and g_obs is observed on the "
        "trajectory inference will actually see). Stamped into result.json "
        "args; the trainer reads it back and bakes it into the safetensors "
        "meta so the calibrator applies the matching baseline at inference. "
        "Ignored under --dcw_sweep (sweep mode tests the configured scalars).",
    )
    p.add_argument(
        "--dump_per_sample_gaps",
        action="store_true",
        help="Dump per-(traj, step) baseline LL/LH/HL/HH gap arrays "
        "(shape (n_images*n_seeds, n_steps)) to gaps_per_sample.npz. "
        "Consumed by the dcw-learnable-calibrator analysis scripts "
        "(transfer hypothesis, PCA, S_pop).",
    )
    p.add_argument(
        "--save_plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write gap_curves.png. Default on. `make dcw` passes "
        "--no-save_plot per bucket and emits a single pooled plot at "
        "the end of the data-gen phase instead.",
    )
    p.add_argument(
        "--save_images",
        action="store_true",
        help="Decode the final reverse-trajectory latent for each "
        "(sample, seed, config) row and save as PNG under "
        "<run_dir>/images/. Loads the VAE transiently after the bench "
        "loop completes, decodes one row at a time, frees the VAE. "
        "Lets you visually compare baseline vs. each --dcw_scalers config "
        "at matched (sample, seed) — i.e., actually see whether a "
        "gap-narrowing λ improves perceptual quality, not just integrated "
        "|gap|. ~25 MB extra peak RAM at 832×1248 × 48 rows.",
    )
    p.add_argument(
        "--vae",
        type=str,
        default="models/vae/qwen_image_vae.safetensors",
        help="VAE path used by --save_images to decode final latents.",
    )
    p.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale. 1.0 = single conditional "
        "forward (matches v1 calibration). >1 live-encodes the "
        "unconditional embed at startup and runs an extra DiT forward "
        "per step, combining as v_uncond + s · (v_cond − v_uncond) "
        "(matches inference.py). v2 §A1 production env = 4.0.",
    )
    p.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Unconditional prompt for CFG > 1 (default '' matches "
        "inference.py default).",
    )

    # Modulation guidance (off by default — base-DiT calibration target).
    # When --pooled_text_proj is set, mirrors inference.py's mod-guidance
    # pipeline so v2 §A1 can run a production-mod-on cross-check.
    g_mod = p.add_argument_group("modulation guidance (optional)")
    g_mod.add_argument(
        "--pooled_text_proj",
        type=str,
        default="",
        help="Path to trained pooled_text_proj weights (.safetensors). "
        "Default enables modulation guidance with the production-baseline "
        "0429 checkpoint and the pos/neg prompts below. Pass an empty "
        "string (--pooled_text_proj '') to disable for the base-DiT "
        "calibration measurement.",
    )
    g_mod.add_argument(
        "--text_encoder",
        type=str,
        default="models/text_encoders/qwen_3_06b_base.safetensors",
        help="Qwen3 text encoder path; only loaded when mod guidance is on, "
        "freed after the steering delta is computed.",
    )
    g_mod.add_argument("--mod_w", type=float, default=3.0)
    g_mod.add_argument(
        "--mod_pos_prompt", type=str, default="absurdres, masterpiece, score_9"
    )
    g_mod.add_argument(
        "--mod_neg_prompt",
        type=str,
        default="worst quality, low quality, score_1, score_2, score_3",
    )
    g_mod.add_argument("--mod_start_layer", type=int, default=8)
    g_mod.add_argument("--mod_end_layer", type=int, default=27)
    g_mod.add_argument("--mod_taper", type=int, default=0)
    g_mod.add_argument("--mod_taper_scale", type=float, default=0.25)
    g_mod.add_argument("--mod_final_w", type=float, default=0.0)

    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label appended to the run dir (<out_root>/<ts>-<label>/).",
    )
    p.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="Override the run-dir root. Default bench/dcw/results/. "
        "`make dcw` redirects to output/dcw/ since calibration "
        "trajectories are runtime artifacts, not published bench results.",
    )
    return p.parse_args()
