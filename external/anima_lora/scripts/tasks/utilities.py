"""Misc utility entry-points: merge, comfy-batch, distill-prep, distill-mod,
test-unit, update, export-logs, print-config."""

from __future__ import annotations

import os

from ._common import PY, _preset, bespoke_preset_flags, run


def cmd_merge(extra):
    """Bake latest LoRA in ADAPTER_DIR (env, default 'output/ckpt') into the base DiT."""
    adapter_dir = os.environ.get("ADAPTER_DIR", "output/ckpt")
    multiplier = os.environ.get("MULTIPLIER", "1.0")
    run(
        [
            PY,
            "scripts/merge_to_dit.py",
            "--adapter_dir",
            adapter_dir,
            "--multiplier",
            multiplier,
            *extra,
        ]
    )


def cmd_comfy_batch(extra):
    workflow = extra[0] if extra else "workflows/modhydra.json"
    remaining = extra[1:] if extra else []
    run([PY, "scripts/comfy_batch.py", workflow, *remaining])


def cmd_distill_prep(extra):
    """Pre-stage artifacts for ``make distill-mod``.

    Phase 1: emits ``post_image_dataset/lora/_anima_uncond_te.safetensors``
    (T5("") cross-attn baseline) — consumed as the student's unconditional
    text input, replacing the zeroed-crossattn shortcut.

    Phase 2: emits teacher-synthesized clean latents under
    ``post_image_dataset/distill_mod_synth/`` (same NPZ layout as
    ``cache_latents.py``). Train with
    ``make distill-mod ARGS='--synth_data_dir post_image_dataset/distill_mod_synth'``
    to fit on the teacher's manifold (paper-faithful; removes real-vs-teacher
    gap that floors val loss).

    Skip flags forwarded via ``extra``: ``--skip_uncond``, ``--skip_synth``,
    ``--max_samples N``, etc.
    """
    run([PY, "-m", "scripts.distill_mod.prep", *extra])


def cmd_distill_mod(extra):
    """Distill the pooled_text_proj MLP for modulation guidance.

    Honors ``PRESET`` (default ``default``) — translates ``blocks_to_swap`` and
    ``gradient_checkpointing`` from ``configs/presets.toml`` into CLI flags so
    ``make distill-mod PRESET=low_vram`` enables grad ckpt + unsloth offload.
    Trailing ``extra`` args are appended last, so user CLI overrides win.

    Saves to ``output/ckpt/pooled_text_proj.safetensors`` so ``make test MOD=1``
    picks it up automatically.
    """
    preset_flags = bespoke_preset_flags(_preset())
    run(
        [
            PY,
            "-m",
            "scripts.distill_mod.distill",
            "--data_dir",
            "post_image_dataset/lora",
            "--dit_path",
            "models/diffusion_models/anima-base-v1.0.safetensors",
            "--output_path",
            "output/ckpt/pooled_text_proj.safetensors",
            "--attn_mode",
            "flash",
            *preset_flags,
            *extra,
        ]
    )


def cmd_test_unit(extra):
    run([PY, "-m", "pytest", "-q", "tests/", *extra])


def cmd_update(extra):
    """Update anima_lora from a GitHub release (preserves datasets/output/models;
    prompts on configs/methods/ + configs/gui-methods/ conflicts; runs uv sync)."""
    run([PY, "scripts/update.py", *extra])


def cmd_vendor_sync(extra):
    """Refresh custom_nodes/*/_vendor/ trees from the live library.* sources.

    Run before bumping a custom-node version / publishing — the bundled
    vendor copies (tagger + directedit) are how the ComfyUI nodes import
    their inference subset when not running inside the anima_lora repo.
    """
    run([PY, "scripts/sync_vendor.py", *extra])


def cmd_export_logs(extra):
    """Dump TB scalar logs to JSON. RUN=<dir> (default output/logs), ALL=1, JSONL=1."""
    run_path = os.environ.get("RUN", "output/logs")
    cmd = [PY, "scripts/export_logs_json.py", run_path]
    if os.environ.get("ALL"):
        cmd.append("--all")
    if os.environ.get("JSONL"):
        cmd.append("--jsonl")
    run([*cmd, *extra])


def cmd_print_config(extra):
    method = os.environ.get("METHOD", "lora")
    preset = _preset()
    run(
        [
            PY,
            "train.py",
            "--method",
            method,
            "--preset",
            preset,
            "--print-config",
            "--no-config-snapshot",
            *extra,
        ]
    )
