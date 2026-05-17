"""Compile a semantic `TrainingConfig` into kohya-ss command-line arguments.

This is the most important translation layer in the project: upper layers
stay stable while kohya parameter names drift. Keep it a pure function so
we can unit-test thoroughly without touching disk or GPU.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lorahub.core.config.schema import TrainingConfig

logger = logging.getLogger(__name__)

# Map our optimizer names to kohya's --optimizer_type values
_OPTIMIZER_MAP: dict[str, str] = {
    "adamw": "AdamW",
    "adamw8bit": "AdamW8bit",
    "lion": "Lion",
    "lion8bit": "Lion8bit",
    "prodigy": "Prodigy",
    "dadaptation": "DAdaptation",
}

# Arches whose sd-scripts entry script ships the `--blocks_to_swap` flag.
# FLUX / SD3 / Lumina / Anima / Hunyuan Image expose CPU-offload-per-block;
# SD1.x / SD2.x / SDXL do not. Emitting `--blocks_to_swap` to a script that
# doesn't accept it would make argparse explode at launch time, so we drop
# the flag (with a warning) for the unsupported arches.
_BLOCKS_TO_SWAP_ARCHES: frozenset[str] = frozenset({
    "flux",
    "sd3",
    "lumina",
    "anima",
    "hunyuan_image",
})

# Map our network types to kohya's --network_module
_NETWORK_MODULE_MAP: dict[str, str] = {
    "lora": "networks.lora",
    "locon": "lycoris.kohya",
    "loha": "lycoris.kohya",
    "dora": "networks.lora",
}


class CompilationError(ValueError):
    """Raised when a config cannot be expressed in kohya's argument vocabulary."""


def compile_config(
    cfg: TrainingConfig,
    workspace: Path,
) -> tuple[str, list[str], dict[Path, str]]:
    """Translate a recipe into (script_name, argv, files_to_write).

    `files_to_write` is a mapping of absolute path to file content that the
    caller must write before launching the subprocess (currently just
    `<workspace>/dataset.toml`). Returning it instead of writing it ourselves
    keeps the compiler a pure function.
    """
    script = _pick_script(cfg.base_model.arch)
    args: list[str] = []
    files: dict[Path, str] = {}

    _emit_model_args(cfg, args)
    _emit_dataset_args(cfg, workspace, args, files)
    _emit_network_args(cfg, args)
    _emit_optimizer_args(cfg, args)
    _emit_schedule_args(cfg, args)
    _emit_precision_args(cfg, args)
    _emit_loss_args(cfg, args)
    _emit_output_args(cfg, workspace, args)
    _emit_sampling_args(cfg, workspace, args)
    _emit_resume_args(cfg, args)
    _emit_validation_args(cfg, args)
    _emit_optimization_args(cfg, args)
    _emit_variant_args(cfg, args)
    _emit_extra_args(cfg, args)

    return script, args, files


# Map our base_model.arch literals to the kohya sd-scripts entry script that
# trains a network/LoRA for that family. Mirrors the upstream README's
# "Supported Models" table -- see https://github.com/kohya-ss/sd-scripts.
# Arches not in this map are diffusion-pipe-exclusive and `_pick_script`
# raises with a pointer back to the dp backend.
_KOHYA_SCRIPT_MAP: dict[str, str] = {
    "sd15": "train_network.py",
    "sd2": "train_network.py",
    "sdxl": "sdxl_train_network.py",
    "sd3": "sd3_train_network.py",
    "flux": "flux_train_network.py",
    "lumina": "lumina_train_network.py",
    "hunyuan_image": "hunyuan_image_train_network.py",
    "anima": "anima_train_network.py",
}


def _pick_script(arch: str) -> str:
    script = _KOHYA_SCRIPT_MAP.get(arch)
    if script is not None:
        return script
    msg = (
        f"kohya does not support arch={arch!r}; use diffusion-pipe "
        f"(supported kohya arches: {sorted(_KOHYA_SCRIPT_MAP)})"
    )
    raise CompilationError(msg)


def _emit_model_args(cfg: TrainingConfig, args: list[str]) -> None:
    args += [f"--pretrained_model_name_or_path={cfg.base_model.checkpoint}"]
    if cfg.base_model.vae is not None:
        args += [f"--vae={cfg.base_model.vae}"]


def _emit_dataset_args(
    cfg: TrainingConfig,
    workspace: Path,
    args: list[str],
    files: dict[Path, str],
) -> None:
    """Generate a kohya dataset.toml + point the CLI at it.

    kohya's plain `--train_data_dir` mode expects images under `<n>_<concept>/`
    subdirectories. Our recipes describe a flat directory plus a num_repeats
    field, so we emit a dataset.toml (which supports flat dirs) and stop
    passing the legacy resolution/bucket/caption flags that conflict with it.
    """
    toml_path = (workspace / "dataset.toml").resolve()
    files[toml_path] = _build_dataset_toml(cfg)
    args.append(f"--dataset_config={toml_path}")


def _build_dataset_toml(cfg: TrainingConfig) -> str:
    ds = cfg.dataset
    res = (
        f"{ds.resolution[0]}"
        if len(ds.resolution) == 1
        else f"[{ds.resolution[0]}, {ds.resolution[1]}]"
    )

    parts = [
        "[general]",
        f"shuffle_caption = {str(ds.caption.shuffle).lower()}",
        f'caption_extension = "{ds.caption.ext}"',
        "keep_tokens = 0",
        "",
        "[[datasets]]",
        f"resolution = {res}",
        f"batch_size = {cfg.schedule.batch_size}",
    ]
    if ds.bucket.enabled:
        parts += [
            "enable_bucket = true",
            f"min_bucket_reso = {ds.bucket.min_size}",
            f"max_bucket_reso = {ds.bucket.max_size}",
            f"bucket_reso_steps = {ds.bucket.step}",
        ]
    if ds.caption.drop_rate > 0:
        parts.append(f"caption_dropout_rate = {ds.caption.drop_rate}")

    parts += [
        "",
        "  [[datasets.subsets]]",
        f'  image_dir = "{_toml_escape(ds.source)}"',
        f"  num_repeats = {ds.num_repeats}",
        f'  caption_extension = "{ds.caption.ext}"',
        "",
    ]
    return "\n".join(parts)


def _toml_escape(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')



def _emit_network_args(cfg: TrainingConfig, args: list[str]) -> None:
    n = cfg.network
    module = _NETWORK_MODULE_MAP.get(n.type)
    if module is None:
        msg = f"unsupported network.type: {n.type}"
        raise CompilationError(msg)

    args += [
        f"--network_module={module}",
        f"--network_dim={n.rank}",
        f"--network_alpha={n.alpha}",
    ]

    network_args: list[str] = []
    if n.type == "locon":
        network_args.append("algo=locon")
    elif n.type == "loha":
        network_args.append("algo=loha")
    elif n.type == "dora":
        network_args.append("dora_wd=True")

    # Conv rank/alpha for lycoris flavours. The schema validator already
    # guarantees these are None for plain lora/dora, so no extra guard.
    if n.conv_dim is not None:
        network_args.append(f"conv_dim={n.conv_dim}")
    if n.conv_alpha is not None:
        network_args.append(f"conv_alpha={n.conv_alpha}")

    # sd-scripts spells dropout knobs as `dropout` / `rank_dropout` /
    # `module_dropout` inside `--network_args`. Only emit when > 0 so
    # default recipes don't grow noise in the launch argv.
    if n.network_dropout > 0:
        network_args.append(f"dropout={n.network_dropout}")
    if n.rank_dropout > 0:
        network_args.append(f"rank_dropout={n.rank_dropout}")
    if n.module_dropout > 0:
        network_args.append(f"module_dropout={n.module_dropout}")

    if not n.target_text_encoder:
        args += ["--network_train_unet_only"]
    elif not n.target_unet:
        args += ["--network_train_text_encoder_only"]

    if network_args:
        args += ["--network_args"] + network_args

    # `--scale_weight_norms` is a top-level sd-scripts flag, not a
    # `--network_args` key, so it goes straight on `args`.
    if n.scale_weight_norms is not None:
        args.append(f"--scale_weight_norms={n.scale_weight_norms}")


def _emit_optimizer_args(cfg: TrainingConfig, args: list[str]) -> None:
    o = cfg.optimizer
    opt_type = _OPTIMIZER_MAP.get(o.type.lower())
    if opt_type is None:
        msg = f"unsupported optimizer.type: {o.type}"
        raise CompilationError(msg)

    args += [
        f"--optimizer_type={opt_type}",
        f"--learning_rate={o.lr.unet}",
        f"--unet_lr={o.lr.unet}",
        f"--text_encoder_lr={o.lr.text_encoder}",
        f"--lr_scheduler={o.schedule}",
        f"--lr_warmup_steps={o.warmup_steps}",
    ]

    # kohya's `--optimizer_args` takes a sequence of `key=value` tokens after
    # the flag. We always render betas / weight_decay / eps so the YAML stays
    # the source of truth, then merge any free-form `optimizer_args` items
    # (user keys win over the dedicated fields when names collide).
    extra: dict[str, str] = {
        "betas": f"{o.betas[0]},{o.betas[1]}",
        "weight_decay": str(o.weight_decay),
        "eps": str(o.eps),
    }
    extra.update(o.optimizer_args)
    args.append("--optimizer_args")
    args += [f"{k}={v}" for k, v in extra.items()]


def _emit_schedule_args(cfg: TrainingConfig, args: list[str]) -> None:
    s = cfg.schedule
    args += [
        f"--max_train_epochs={s.epochs}",
        f"--train_batch_size={s.batch_size}",
        f"--gradient_accumulation_steps={s.grad_accum}",
    ]
    if s.max_steps is not None:
        args += [f"--max_train_steps={s.max_steps}"]


def _emit_precision_args(cfg: TrainingConfig, args: list[str]) -> None:
    if cfg.precision != "fp32":
        args += [f"--mixed_precision={cfg.precision}"]
    if cfg.gradient_checkpointing:
        args += ["--gradient_checkpointing"]
    if cfg.cache_latents:
        args += ["--cache_latents"]


def _emit_loss_args(cfg: TrainingConfig, args: list[str]) -> None:
    """Emit loss-shaping flags (--min_snr_gamma, --noise_offset, etc).

    None-valued / zero-valued / sd-scripts-default-valued fields are omitted
    so the user's recipe stays additive over the kohya defaults: writing
    `loss: {}` keeps every behaviour kohya ships with.
    """
    loss = cfg.loss
    if loss.min_snr_gamma is not None:
        args.append(f"--min_snr_gamma={loss.min_snr_gamma}")
    if loss.noise_offset > 0:
        args.append(f"--noise_offset={loss.noise_offset}")
    if loss.ip_noise_gamma is not None:
        args.append(f"--ip_noise_gamma={loss.ip_noise_gamma}")
    if loss.prior_loss_weight != 1.0:
        args.append(f"--prior_loss_weight={loss.prior_loss_weight}")
    if loss.loss_type != "l2":
        args.append(f"--loss_type={loss.loss_type}")
    if loss.debiased_estimation:
        args.append("--debiased_estimation_loss")
    if loss.masked_loss:
        args.append("--masked_loss")
    if loss.scale_v_pred_loss_like_noise_pred:
        args.append("--scale_v_pred_loss_like_noise_pred")
    if loss.v_parameterization:
        args.append("--v_parameterization")


def _emit_output_args(cfg: TrainingConfig, workspace: Path, args: list[str]) -> None:
    out = cfg.output
    output_dir = out.output_dir if out.output_dir is not None else workspace / "output"
    args += [
        f"--output_dir={output_dir}",
        f"--output_name={out.name}",
        f"--save_every_n_epochs={out.save_every_n_epochs}",
        "--save_model_as=safetensors",
        f"--save_precision={out.save_dtype}",
        f"--logging_dir={workspace / 'logs'}",
    ]


def _emit_sampling_args(cfg: TrainingConfig, workspace: Path, args: list[str]) -> None:
    s = cfg.sampling
    if not s.enabled or s.prompts_file is None:
        return
    args += [
        f"--sample_every_n_epochs={s.every_n_epochs}",
        f"--sample_prompts={s.prompts_file}",
        "--sample_sampler=euler_a",
    ]


def _emit_resume_args(cfg: TrainingConfig, args: list[str]) -> None:
    """Emit the kohya `--save_state*` flags so a run can be resumed.

    sd-scripts writes the state directory inside `--output_dir` next to
    the safetensors as `<output_name>-state`. We only enable local state
    saving here; the resume route scans the same directory at restart
    time. `--save_state_to_huggingface` is intentionally not surfaced.
    """
    r = cfg.resume
    if r.save_state:
        args.append("--save_state")
    if r.save_state_at_end:
        args.append("--save_state_on_train_end")
    if r.save_state_every_n_epochs is not None:
        args.append(f"--save_state_every_n_epochs={r.save_state_every_n_epochs}")


def _emit_validation_args(cfg: TrainingConfig, args: list[str]) -> None:
    """Emit sd-scripts' validation-split flags when `dataset.val_split > 0`.

    sd-scripts 1.x exposes a held-out validation split via three flags:
    `--validation_split_percentage` (integer percent, 0 disables it),
    `--validate_every_n_epochs`, and `--max_validation_steps`. We round the
    fractional `val_split` to the nearest percent so a YAML value of `0.10`
    lands on `10` instead of `10.0`. `validation.max_samples` is mapped to
    `--max_validation_steps` because that is the per-eval-pass cap kohya
    actually accepts; users can always override via `backend.extra_args`.
    """
    ds = cfg.dataset
    if ds.val_split <= 0.0:
        return

    percent = max(1, round(ds.val_split * 100))
    args.append(f"--validation_split_percentage={percent}")

    v = cfg.validation
    args.append(f"--validate_every_n_epochs={v.every_n_epochs}")
    if v.max_samples is not None:
        args.append(f"--max_validation_steps={v.max_samples}")


def _emit_variant_args(cfg: TrainingConfig, args: list[str]) -> None:
    """Inject argv tweaks specific to an SDXL sub-architecture.

    Conservative for now: only the Pony lineage gets `--clip_skip=2`,
    which matches how Pony was trained and is the most-cited recipe
    delta the community converges on. Illustrious/NoobAI/Animagine
    don't add argv yet - their tuning lives in the scaffolder's LR
    defaults. User overrides via `backend.extra_args` win because
    `_emit_extra_args` runs after this hook.
    """
    variant = cfg.base_model.arch_variant
    if variant == "pony":
        args.append("--clip_skip=2")


def _emit_optimization_args(cfg: TrainingConfig, args: list[str]) -> None:
    """Emit kohya speed/VRAM toggles from ``cfg.optimization``.

    All four knobs map to flags that exist in (some) sd-scripts entry
    scripts. We emit them when set; ``--blocks_to_swap`` is gated on the
    handful of architectures whose train script defines the argparse
    option (FLUX/SD3/Lumina/Anima/HunyuanImage). For other arches (notably
    SDXL/SD1.x/SD2.x) sd-scripts has no such flag and emitting it would
    make argparse abort, so we skip + warn instead.

    ``--full_bf16`` coexists with ``--mixed_precision``: sd-scripts treats
    it as an additive escalation that pushes optimizer state and grads to
    bf16 on top of the mixed-precision forward, so both flags landing on
    the argv is intentional.
    """
    o = cfg.optimization
    if o.torch_compile:
        args.append("--torch_compile")
    if o.fused_backward_pass:
        args.append("--fused_backward_pass")
    if o.full_bf16:
        args.append("--full_bf16")
    if o.blocks_to_swap > 0:
        if cfg.base_model.arch in _BLOCKS_TO_SWAP_ARCHES:
            args.append(f"--blocks_to_swap={o.blocks_to_swap}")
        else:
            logger.warning(
                "kohya: --blocks_to_swap is not supported for arch=%r; "
                "ignoring optimization.blocks_to_swap=%d "
                "(supported arches: %s)",
                cfg.base_model.arch,
                o.blocks_to_swap,
                sorted(_BLOCKS_TO_SWAP_ARCHES),
            )


def _emit_extra_args(cfg: TrainingConfig, args: list[str]) -> None:
    """Append user-provided escape-hatch args verbatim. Last write wins."""
    for key, value in cfg.backend.extra_args.items():
        flag = f"--{key}" if not key.startswith("--") else key
        if value is True:
            args.append(flag)
        elif value is False or value is None:
            continue
        else:
            args.append(f"{flag}={value}")

