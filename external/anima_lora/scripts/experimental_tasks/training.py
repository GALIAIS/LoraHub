"""Experimental training entry-points: postfix, ip-adapter, easycontrol.

These are wired up under ``make exp-*`` / ``python tasks.py exp-*`` to keep
the unstable methods visually separate from the shipped ones (lora family,
modulation guidance, hydra). Each ``cmd_*`` is a thin shim that translates env
vars + extra argv into the right ``train.py`` (via ``accelerate launch``) or
``preprocess/*.py`` call.
"""

from __future__ import annotations

from scripts.tasks import preprocess as _preprocess
from scripts.tasks._common import PY, _preset, bespoke_preset_flags, run, train


def cmd_postfix(extra):
    train("postfix", extra)


def cmd_turbo(extra):
    """Turbo Anima — Decoupled DMD2 distillation (proposal: docs/proposal/turbo_anima_dmd_lora.md).

    Bypasses train.py / accelerate (single-GPU bespoke loop, like distill-mod).
    Reads ``configs/methods/turbo.toml``; trailing args are forwarded so user
    CLI flags override TOML values, e.g.::

        make exp-turbo                                  # defaults: rank=48, 4-step
        make exp-turbo ARGS="--student_rank 64 --iterations 5000"
        make exp-turbo ARGS="--single_prompt_idx 0"     # Phase 0 single-prompt overfit

    Honors ``PRESET`` (default ``default``) — translates ``blocks_to_swap`` and
    ``gradient_checkpointing`` from ``configs/presets.toml`` into CLI flags so
    ``make exp-turbo PRESET=low_vram`` enables grad ckpt + unsloth offload, and
    ``PRESET=half/quarter/tenth`` shrinks the dataset via ``--sample_ratio``.
    ``extra`` is appended last, so user CLI overrides win.
    """
    preset_flags = bespoke_preset_flags(_preset())
    run([PY, "scripts/distill_turbo.py", *preset_flags, *extra])


def cmd_fera(extra):
    """Author-faithful FeRA (Yin et al., arXiv:2511.17979).

    Drives ``configs/gui-methods/fera.toml`` — independent-A stacked experts
    + a single network-level GlobalRouter fed by FEI(z_t). Lives on the
    LoRA-family network module via the ``stacked_experts_global_fei`` spec
    (selected by ``use_moe_style="independent_A"``). plan2 §three-axis-config.
    """
    train("fera", extra, methods_subdir="gui-methods")


def cmd_soft_tokens(extra):
    train("soft_tokens", extra)


def cmd_chimera(extra):
    """ChimeraHydra (dual-pool additive routing — docs/proposal/chimera_hydra.md).

    Drives ``configs/methods/chimera.toml``: OrthoHydra split into a content
    pool (K_c=3, per-layer rank-R router on pooled lx) and a freq pool
    (K_f=3, network-level FreqRouter on concat(FEI, σ-features)). Pool
    outputs are added (no multiplicative gate, no σ-band overlap mask).
    Single-phase co-training; per-pool balance loss; T-LoRA mask on the
    content branch only.
    """
    train("chimera", extra)


def cmd_ip_adapter(extra):
    train("ip_adapter", extra)


def cmd_ip_adapter_preprocess(extra):
    """Full IP-Adapter preprocess.

    IP-Adapter shares the LoRA pipeline's data layout — source images live in
    ``image_dataset/`` and caches in ``post_image_dataset/lora/``. This is just
    a convenience alias for ``make preprocess`` + ``make preprocess-pe`` so the
    GUI's IP-Adapter tab and ``make exp-ip-adapter-preprocess`` keep working.
    """
    _preprocess.cmd_preprocess(extra)
    _preprocess.cmd_preprocess_pe(extra)


def cmd_easycontrol(extra):
    train("easycontrol", extra)


def cmd_easycontrol_preprocess(extra):
    """Full EasyControl preprocess: VAE latents + text-encoder outputs.

    Source: ``easycontrol-dataset/``  Caches: ``post_image_dataset/easycontrol/``.
    """
    src = "easycontrol-dataset"
    dst = "post_image_dataset/easycontrol"
    run(
        [
            PY,
            "preprocess/cache_latents.py",
            "--dir",
            src,
            "--cache_dir",
            dst,
            "--vae",
            "models/vae/qwen_image_vae.safetensors",
            "--batch_size",
            "4",
            "--chunk_size",
            "64",
        ]
    )
    run(
        [
            PY,
            "preprocess/cache_text_embeddings.py",
            "--dir",
            src,
            "--cache_dir",
            dst,
            "--qwen3",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit",
            "models/diffusion_models/anima-base-v1.0.safetensors",
            "--caption_shuffle_variants",
            "4",
            "--caption_tag_dropout_rate",
            "0.1",
        ]
    )
