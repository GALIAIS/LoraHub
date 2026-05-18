#!/usr/bin/env python3
"""Cross-platform task runner -- replaces Makefile for Windows compatibility.

Usage:
    python tasks.py <command> [extra args...]

Examples:
    python tasks.py lora
    python tasks.py lora --network_dim 32 --max_train_epochs 64
    python tasks.py test
    python tasks.py test                     # add SPECTRUM=1 to enable Spectrum
    python tasks.py test                     # add MOD=1 to enable modulation guidance
    python tasks.py test                     # add NOLORA=1 to run against the bare DiT
    python tasks.py download-models
    python tasks.py exp-postfix              # experimental method
    python tasks.py exp-test-ip ref.png      # experimental inference

Command implementations live under ``scripts/tasks/`` (shipped methods) and
``scripts/experimental_tasks/`` (unstable methods exposed under ``exp-*``).
This file is just a name → callable dispatch table.
"""

import sys

from scripts.experimental_tasks import inference as exp_inference
from scripts.experimental_tasks import training as exp_training
from scripts.tasks import (
    dcw,
    downloads,
    gui,
    inference,
    masking,
    preprocess,
    tagger,
    training,
    utilities,
)

COMMANDS = {
    # ── Training ──────────────────────────────────────────────────────
    "lora": (
        training.cmd_lora,
        "LoRA family (lora|tlora|tlora_rf|hydralora via configs/methods/lora.toml)",
    ),
    "lora-gui": (
        training.cmd_lora_gui,
        "Train from a self-contained configs/gui-methods/<variant>.toml "
        "(variant from GUI_PRESETS env or 1st positional; e.g. tlora, hydralora, reft, postfix_exp).",
    ),
    # ── Inference ─────────────────────────────────────────────────────
    "test": (
        inference.cmd_test,
        "Inference with latest LoRA. SPECTRUM=1 enables Spectrum acceleration; "
        "MOD=1 adds the latest distilled pooled_text_proj (modulation guidance); "
        "NOLORA=1 runs against the bare DiT (skips --lora_weight).",
    ),
    "test-hydra": (
        inference.cmd_test_hydra,
        "Inference with latest HydraLoRA moe (router-live)",
    ),
    "test-merge": (
        inference.cmd_test_merge,
        "Inference with latest *_merged.safetensors (MODEL_DIR=..., default 'output_temp')",
    ),
    "test-dcw": (
        inference.cmd_test_dcw,
        "Inference with latest LoRA + DCW post-step bias correction. "
        "Honors SPECTRUM=1 / MOD=1 / NOLORA=1.",
    ),
    "test-smc-cfg": (
        inference.cmd_test_smc_cfg,
        "Inference with latest LoRA + SMC-CFG (sliding-mode control CFG, arXiv:2603.03281). "
        "Honors SPECTRUM=1 / MOD=1 / NOLORA=1.",
    ),
    "test-dcw-v4": (
        inference.cmd_test_dcw_v4,
        "Inference with DCW v4 learnable calibrator (auto-resolves fusion_head.safetensors). "
        "No LoRA by default; SPECTRUM=1 / MOD=1 / NOLORA=0 (attach latest LoRA) all compose.",
    ),
    "dcw": (
        dcw.cmd_dcw,
        "Calibrate DCW v4: sample 5 aspect buckets (default 130×1 seeds, "
        "shuffle_seed=0) + train fusion head",
    ),
    "dcw-train": (
        dcw.cmd_dcw_train,
        "Train-only on existing bench/dcw/results/ pool (~30s, no sampling)",
    ),
    "test-spectrum-dcw": (
        inference.cmd_test_spectrum_dcw,
        "Spectrum-accelerated inference + DCW post-step bias correction",
    ),
    "test-dcw-v4-spectrum": (
        inference.cmd_test_dcw_v4_spectrum,
        "Spectrum-accelerated inference + DCW v4 learnable calibrator (auto-resolves fusion_head.safetensors)",
    ),
    # ── Preprocess ────────────────────────────────────────────────────
    "preprocess": (
        preprocess.cmd_preprocess,
        "Full preprocessing (resize + VAE + text embeddings)",
    ),
    "preprocess-resize": (
        preprocess.cmd_preprocess_resize,
        "Resize images to bucket resolutions",
    ),
    "preprocess-vae": (preprocess.cmd_preprocess_vae, "Cache VAE latents"),
    "preprocess-te": (preprocess.cmd_preprocess_te, "Cache text encoder embeddings"),
    "preprocess-pooled": (
        preprocess.cmd_preprocess_pooled,
        "Cache pooled text embeddings sidecar (consumed by distill-mod). No GPU.",
    ),
    "preprocess-pe": (
        preprocess.cmd_preprocess_pe,
        "Cache PE-Core vision-encoder features into the LoRA cache dir. "
        "Consumed by REPA (--use_repa) and IP-Adapter live-disk mode.",
    ),
    # ── Anima Tagger ──────────────────────────────────────────────────
    "preprocess-tagger": (
        tagger.cmd_preprocess_tagger,
        "Build the Anima Tagger vocab/manifest + cache pooled PE features "
        "+ cache resized images (build_vocab + build_features + "
        "build_resized). Needs CAPTION_CORPUS_DIR in .env.",
    ),
    "tagger": (
        tagger.cmd_tagger,
        "Two-stage train: Stage 1 head-only on cached pooled features, "
        "then Stage 2 PE-LoRA warm-started from Stage 1. Requires "
        "`make preprocess-tagger` first.",
    ),
    "test-tagger": (
        tagger.cmd_test_tagger,
        "Predict tags for a single image (--image <path>) or sample a random "
        "val-split stem. Pass --show_scores for rating + top-K kept tags.",
    ),
    # ── Downloads ─────────────────────────────────────────────────────
    "download-models": (downloads.cmd_download_models, "Download all models"),
    "download-anima": (downloads.cmd_download_anima, "Download Anima model"),
    "download-sam3": (downloads.cmd_download_sam3, "Download SAM3 model"),
    "download-mit": (downloads.cmd_download_mit, "Download MIT model"),
    "download-pe": (
        downloads.cmd_download_pe,
        "Download PE-Core-L14-336 (img2emb encoder)",
    ),
    "download-pe-spatial": (
        downloads.cmd_download_pe_spatial,
        "Download PE-Spatial-B16-512 (Anima Tagger aux encoder)",
    ),
    # ── Masking ───────────────────────────────────────────────────────
    "mask": (
        masking.cmd_mask,
        "Run SAM + MIT (via tempdir) and write merged masks under post_image_dataset/masks/",
    ),
    "mask-clean": (
        masking.cmd_mask_clean,
        "Remove post_image_dataset/masks/",
    ),
    # ── GUI ───────────────────────────────────────────────────────────
    "gui": (gui.cmd_gui, "Launch PySide6 GUI"),
    "gui-shortcut": (
        gui.cmd_gui_shortcut,
        "Create a Windows desktop shortcut that launches the GUI (no console window)",
    ),
    # ── Utilities ─────────────────────────────────────────────────────
    "merge": (
        utilities.cmd_merge,
        "Bake latest LoRA (ADAPTER_DIR=..., default 'output/ckpt') into base DiT",
    ),
    "comfy-batch": (utilities.cmd_comfy_batch, "Run ComfyUI batch workflow"),
    "distill-prep": (
        utilities.cmd_distill_prep,
        'Pre-stage artifacts for distill-mod: T5("") uncond sidecar + '
        "teacher-synthetic clean latents pool (--skip_synth / --skip_uncond to "
        "stage only one).",
    ),
    "distill-mod": (
        utilities.cmd_distill_mod,
        "Distill pooled_text_proj MLP for modulation guidance",
    ),
    "test-unit": (utilities.cmd_test_unit, "Run smoke/unit tests (pytest tests/)"),
    "export-logs": (
        utilities.cmd_export_logs,
        "Dump TB scalar logs to JSON (RUN=<dir>, ALL=1 for every subrun, JSONL=1 for line-delimited)",
    ),
    "print-config": (
        utilities.cmd_print_config,
        "Dump merged config (METHOD=<name> PRESET=<name>) with provenance",
    ),
    "update": (
        utilities.cmd_update,
        "Update from GitHub release (preserves datasets/output/models, prompts on "
        "config conflicts, runs uv sync). Pass --dry-run / --version v1.0 / --no-sync.",
    ),
    "vendor-sync": (
        utilities.cmd_vendor_sync,
        "Refresh custom_nodes/*/_vendor/ from live library/* (run before publishing nodes)",
    ),
    # ── Experimental ──────────────────────────────────────────────────
    # Unstable methods kept under exp-* so they don't pollute the main command
    # surface. May produce broken output, change without notice, or be removed.
    "exp-postfix": (
        exp_training.cmd_postfix,
        "[experimental] Postfix tuning (mode selected in configs/methods/postfix.toml)",
    ),
    "exp-turbo": (
        exp_training.cmd_turbo,
        "[experimental] Decoupled DMD2 distillation — bakes CFG=4 / 28-step Anima "
        "into a 4-step LoRA student (configs/methods/turbo.toml). "
        "Single-GPU bespoke loop (bypasses train.py/accelerate, like distill-mod).",
    ),
    "exp-fera": (
        exp_training.cmd_fera,
        "[experimental] Author-faithful FeRA (independent-A stacked experts + "
        "global FEI router; configs/gui-methods/fera.toml)",
    ),
    "exp-soft-tokens": (
        exp_training.cmd_soft_tokens,
        "[experimental] SoftREPA-style per-layer × per-t soft tokens (training-only v1)",
    ),
    "exp-chimera": (
        exp_training.cmd_chimera,
        "[experimental] ChimeraHydra dual-pool additive routing "
        "(content + freq pools on OrthoHydra; configs/methods/chimera.toml)",
    ),
    "exp-ip-adapter": (
        exp_training.cmd_ip_adapter,
        "[experimental] IP-Adapter training (decoupled image cross-attention)",
    ),
    "exp-ip-adapter-preprocess": (
        exp_training.cmd_ip_adapter_preprocess,
        "[experimental] Alias for `preprocess` + `preprocess-pe` (IP-Adapter "
        "reuses the LoRA pipeline's caches under post_image_dataset/lora/).",
    ),
    "exp-easycontrol": (
        exp_training.cmd_easycontrol,
        "[experimental] EasyControl training (extended self-attn KV with VAE-encoded reference)",
    ),
    "exp-easycontrol-preprocess": (
        exp_training.cmd_easycontrol_preprocess,
        "[experimental] Full EasyControl preprocess: latents + text emb. "
        "Source: easycontrol-dataset/  Cache: post_image_dataset/easycontrol/.",
    ),
    "exp-test-postfix": (
        exp_inference.cmd_test_postfix,
        "[experimental] Inference with latest postfix weight",
    ),
    "exp-test-turbo": (
        exp_inference.cmd_test_turbo,
        "[experimental] Inference with latest turbo student LoRA at 4 steps, cfg=1.0 "
        "(CFG is baked into the student).",
    ),
    "exp-test-postfix-exp": (
        exp_inference.cmd_test_postfix_exp,
        "[experimental] Inference with latest postfix-exp weight",
    ),
    "exp-test-postfix-func": (
        exp_inference.cmd_test_postfix_func,
        "[experimental] Inference with latest postfix-func weight",
    ),
    "exp-test-ip": (
        exp_inference.cmd_test_ip,
        "[experimental] Inference with latest IP-Adapter weight. Usage: exp-test-ip <ref_image> [--prompt ... --ip_scale ...]",
    ),
    "exp-test-easycontrol": (
        exp_inference.cmd_test_easycontrol,
        "[experimental] Inference with latest EasyControl weight. Usage: exp-test-easycontrol <ref_image> [--prompt ... --easycontrol_scale ...]",
    ),
    "exp-test-directedit": (
        exp_inference.cmd_test_directedit,
        "[experimental] DirectEdit on a random source image. PROMPT='...' supplies the edit "
        "instruction (appended to the Anima Tagger source caption). REF_IMAGE=path overrides the "
        "random pick. Usage: exp-test-directedit [ref_image] [extra...]",
    ),
    "exp-test-directedit-dry": (
        exp_inference.cmd_test_directedit_dry,
        "[experimental] DirectEdit functional check: random source image + random crossattn "
        "embed (no TE, no captioner); ψ_tar == ψ_src so output should reconstruct the source. "
        "REF_IMAGE=path overrides the random pick. Usage: exp-test-directedit-dry [ref_image] [extra...]",
    ),
    "exp-invert-directedit": (
        exp_inference.cmd_invert_directedit,
        "[experimental] Probe: invert the K-slot orthogonal postfix tail per image, then run "
        "DirectEdit dry mode twice (baseline T5(tags) vs T5(tags)+tail). Outputs side-by-side under "
        "output/tests/invert_directedit/<stem>/. Env: N_IMAGES (default 1), REF_IMAGE, K (default 48), "
        "INVERT_STEPS, LAMBDA_ZERO, SIGMA_MIN, BASIS, SEED.",
    ),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python tasks.py <command> [extra args...]\n")
        print("Commands:")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:30s} {desc}")
        print("\nExtra arguments are forwarded to the underlying command.")
        print("Example: python tasks.py lora --network_dim 32 --max_train_epochs 64")
        sys.exit(0)

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run 'python tasks.py --help' for available commands", file=sys.stderr)
        sys.exit(1)

    extra = sys.argv[2:]
    fn, desc = COMMANDS[command]
    if extra and extra[0] in ("-h", "--help"):
        print(f"python tasks.py {command} -- {desc}\n")
        if fn.__doc__:
            print(fn.__doc__.strip())
        else:
            print("(no detailed help available)")
        print(
            "\nUnrecognised flags are forwarded verbatim to the underlying script. "
            "Run the underlying script with --help for its full flag set."
        )
        sys.exit(0)
    fn(extra)


if __name__ == "__main__":
    main()
