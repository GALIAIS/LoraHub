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

# Env var read by ``_attn_patch`` inside the kohya subprocess to swap in
# the FA3 / FA4 dispatcher. Empty / unset means "leave attention alone".
_ATTN_OVERRIDE_ENV = "LORAHUB_KOHYA_ATTN_OVERRIDE"

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

# Arches that pass through `add_dit_training_arguments` (FLUX / SD3 / Lumina /
# Anima / HunyuanImage). These scripts accept the flow-matching weighting
# knobs (`--weighting_scheme`, `--logit_mean`, `--logit_std`, `--mode_scale`)
# and the text-encoder caching flags. SDXL has its own copy of the caching
# flags via `add_sdxl_training_arguments`; SD1/SD2 has neither.
_DIT_ARCHES: frozenset[str] = frozenset({
    "flux",
    "sd3",
    "lumina",
    "anima",
    "hunyuan_image",
})

# Arches whose entry script (directly or transitively) accepts
# `--cache_text_encoder_outputs(_to_disk)`. This is SDXL plus everything in
# `_DIT_ARCHES`; plain `train_network.py` (SD1.5 / SD2) lacks it.
_CACHE_TE_ARCHES: frozenset[str] = _DIT_ARCHES | {"sdxl"}

# Arches whose entry script accepts `--disable_mmap_load_safetensors`. Same
# coverage as the TE caching flag — emitted by `add_sdxl_training_arguments`
# (SDXL) and `add_dit_training_arguments` (the dit family). SD1.5 / SD2 do
# not see it, so we gate on the same set.
_MMAP_DISABLE_ARCHES: frozenset[str] = _CACHE_TE_ARCHES

# FLUX-specific argv added by `library.flux_train_utils.add_flux_train_arguments`.
_FLUX_ARCHES: frozenset[str] = frozenset({"flux"})

# SD3 ships its own attention-mask + positional-embed knobs via
# `library.sd3_train_utils.add_sd3_training_arguments`.
_SD3_ARCHES: frozenset[str] = frozenset({"sd3"})

# Anima-only argv (`anima_train_network.py` + `library.anima_train_utils`).
_ANIMA_ARCHES: frozenset[str] = frozenset({"anima"})

# Hunyuan Image-only argv (`hunyuan_image_train_network.py`).
_HUNYUAN_IMAGE_ARCHES: frozenset[str] = frozenset({"hunyuan_image"})

# Arches that share the FLUX/SD3 dropout knobs (`--t5_dropout_rate`,
# `--clip_l_dropout_rate`, `--clip_g_dropout_rate`, `--apply_t5_attn_mask`).
# FLUX defines `--apply_t5_attn_mask`; SD3 redefines it plus `--apply_lg_attn_mask`
# and the per-encoder dropout rates.
_T5_DROPOUT_ARCHES: frozenset[str] = frozenset({"flux", "sd3"})
_CLIP_DROPOUT_ARCHES: frozenset[str] = frozenset({"sd3"})

# Map our network types to kohya's --network_module
_NETWORK_MODULE_MAP: dict[str, str] = {
    "lora": "networks.lora",
    "locon": "lycoris.kohya",
    "loha": "lycoris.kohya",
    "lokr": "lycoris.kohya",
    "dora": "networks.lora",
}


class CompilationError(ValueError):
    """Raised when a config cannot be expressed in kohya's argument vocabulary."""


def _require_training_dataset(cfg: TrainingConfig) -> None:
    """Reject an incomplete source/subset selection before writing dataset.toml."""
    if cfg.dataset.subsets:
        missing = [
            str(index + 1)
            for index, subset in enumerate(cfg.dataset.subsets)
            if subset.path is None
        ]
        if missing:
            raise CompilationError(
                "kohya requires dataset.subsets[].path for every active subset "
                f"(missing: {', '.join(missing)})"
            )
        return
    if cfg.dataset.source is None:
        raise CompilationError(
            "kohya requires dataset.source when no dataset subsets are configured"
        )


def compile_config(
    cfg: TrainingConfig,
    workspace: Path,
) -> tuple[str, list[str], dict[Path, str], dict[str, str]]:
    """Translate a recipe into (script_name, argv, files_to_write, env).

    `files_to_write` is a mapping of absolute path to file content that the
    caller must write before launching the subprocess (currently just
    `<workspace>/dataset.toml`). Returning it instead of writing it ourselves
    keeps the compiler a pure function.

    `env` carries process-level overrides the runner must merge into the
    spawn environment — currently used by the FA3 / FA4 attention path so
    the in-process monkey-patch (`_attn_patch.py`) knows which dispatcher
    to substitute. Always a dict; empty when no overrides are needed.
    """
    script = _pick_script(cfg.base_model.arch)
    _require_training_dataset(cfg)
    args: list[str] = []
    files: dict[Path, str] = {}
    env: dict[str, str] = {}

    _emit_model_args(cfg, args)
    _emit_arch_paths_args(cfg, args)
    _emit_dataset_args(cfg, workspace, args, files)
    _emit_dataloader_args(cfg, args)
    _emit_augmentation_args(cfg, args)
    _emit_caption_args(cfg, args)
    _emit_bucket_args(cfg, args)
    _emit_network_args(cfg, args)
    _emit_per_module_lr_args(cfg, args)
    _emit_optimizer_args(cfg, args)
    _emit_schedule_args(cfg, args)
    _emit_precision_args(cfg, args)
    _emit_loss_args(cfg, args)
    _emit_flow_match_args(cfg, args)
    _emit_attention_args(cfg, args, env)
    _emit_output_args(cfg, workspace, args)
    _emit_metadata_args(cfg, args)
    _emit_sampling_args(cfg, workspace, args)
    _emit_resume_args(cfg, args)
    _emit_validation_args(cfg, args)
    _emit_optimization_args(cfg, args)
    _emit_variant_args(cfg, args)
    _emit_extra_args(cfg, args)
    _emit_monitoring_args(cfg, args)

    return script, args, files, env


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
        f"keep_tokens = {ds.caption.keep_tokens}",
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
        if ds.bucket.skip_image_resolution is not None:
            skip_resolution = ds.bucket.skip_image_resolution
            if isinstance(skip_resolution, tuple):
                rendered = f"[{skip_resolution[0]}, {skip_resolution[1]}]"
            else:
                rendered = str(skip_resolution)
            parts.append(f"skip_image_resolution = {rendered}")
    if ds.caption.drop_rate > 0:
        parts.append(f"caption_dropout_rate = {ds.caption.drop_rate}")

    subset_rows: list[tuple[Path, int, str | None]]
    if ds.subsets:
        subset_rows = [
            (subset.path, subset.num_repeats, subset.caption_prefix)
            for subset in ds.subsets
        ]
    else:
        subset_rows = [(ds.source, ds.num_repeats, None)]

    for image_dir, num_repeats, caption_prefix in subset_rows:
        parts += [
            "",
            "  [[datasets.subsets]]",
            f'  image_dir = "{_toml_escape(image_dir)}"',
            f"  num_repeats = {num_repeats}",
            f'  caption_extension = "{ds.caption.ext}"',
        ]
        if caption_prefix:
            parts.append(f'  caption_prefix = "{_toml_escape(caption_prefix)}"')

    if ds.reg_source is not None:
        parts += [
            "",
            "  [[datasets.subsets]]",
            f'  image_dir = "{_toml_escape(ds.reg_source)}"',
            "  num_repeats = 1",
            "  is_reg = true",
            f'  caption_extension = "{ds.caption.ext}"',
        ]

    parts.append("")
    return "\n".join(parts)


def _toml_escape(path: Path | str) -> str:
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
    elif n.type == "lokr":
        network_args.append("algo=lokr")
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

    # Continue training from existing LoRA / merge bases. `init_from` maps
    # to kohya's `--network_weights`; `dim_from_weights` is a flag that
    # tells the trainer to read rank from that file (so the schema field
    # holds the same path but emits a different kohya argv pair).
    if n.init_from is not None:
        args.append(f"--network_weights={n.init_from}")
    if n.dim_from_weights is not None:
        # kohya's --dim_from_weights is a store_true flag; the path is
        # taken from --network_weights. If the user supplied a path here
        # but no `init_from`, route it through --network_weights too so
        # the flag has somewhere to read from.
        if n.init_from is None:
            args.append(f"--network_weights={n.dim_from_weights}")
        args.append("--dim_from_weights")
    if n.base_weights:
        args.append("--base_weights")
        args += [str(p) for p in n.base_weights]
        if n.base_weights_multiplier:
            args.append("--base_weights_multiplier")
            args += [str(m) for m in n.base_weights_multiplier]


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

    # max_grad_norm has a non-zero kohya default of 1.0 — emit only when the
    # recipe moves it (most users either keep it at 1.0 or disable clipping
    # by passing 0). This keeps the existing argv tight.
    if o.max_grad_norm != 1.0:
        args.append(f"--max_grad_norm={o.max_grad_norm}")

    # Custom scheduler module / args (kohya: --lr_scheduler_type / _args).
    if o.scheduler_module is not None:
        args.append(f"--lr_scheduler_type={o.scheduler_module}")
    if o.scheduler_args:
        args.append("--lr_scheduler_args")
        args += [f"{k}={v}" for k, v in o.scheduler_args.items()]
    # cosine_with_restarts cycle count -- only emit when user moved it
    if o.scheduler_num_cycles != 1:
        args.append(f"--lr_scheduler_num_cycles={o.scheduler_num_cycles}")
    if o.scheduler_power != 1.0:
        args.append(f"--lr_scheduler_power={o.scheduler_power}")
    if o.scheduler_timescale is not None:
        args.append(f"--lr_scheduler_timescale={o.scheduler_timescale}")
    if o.scheduler_min_lr_ratio is not None:
        args.append(f"--lr_scheduler_min_lr_ratio={o.scheduler_min_lr_ratio}")

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
    if s.seed is not None:
        args.append(f"--seed={s.seed}")
    if s.lr_decay_steps is not None:
        args.append(f"--lr_decay_steps={s.lr_decay_steps}")


def _emit_precision_args(cfg: TrainingConfig, args: list[str]) -> None:
    if cfg.precision != "fp32":
        args += [f"--mixed_precision={cfg.precision}"]
    if cfg.gradient_checkpointing:
        args += ["--gradient_checkpointing"]
    if cfg.cache_latents:
        args += ["--cache_latents"]
    if cfg.cache_latents_to_disk:
        args += ["--cache_latents_to_disk"]
    if cfg.skip_cache_check:
        args += ["--skip_cache_check"]
    if cfg.cache_info:
        args += ["--cache_info"]
    if cfg.train_inpainting:
        args += ["--train_inpainting"]


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
    if loss.noise_offset_random_strength:
        args.append("--noise_offset_random_strength")
    if loss.multires_noise_iterations is not None:
        args.append(f"--multires_noise_iterations={loss.multires_noise_iterations}")
    # multires_noise_discount has a non-zero default (0.3); emit only when the
    # user moved it OR when multires noise is on (kohya ignores it otherwise
    # but it's harmless).
    if loss.multires_noise_discount != 0.3:
        args.append(f"--multires_noise_discount={loss.multires_noise_discount}")
    if loss.adaptive_noise_scale is not None:
        args.append(f"--adaptive_noise_scale={loss.adaptive_noise_scale}")
    if loss.ip_noise_gamma is not None:
        args.append(f"--ip_noise_gamma={loss.ip_noise_gamma}")
    if loss.ip_noise_gamma_random_strength:
        args.append("--ip_noise_gamma_random_strength")
    if loss.zero_terminal_snr:
        args.append("--zero_terminal_snr")
    if loss.min_timestep is not None:
        args.append(f"--min_timestep={loss.min_timestep}")
    if loss.max_timestep is not None:
        args.append(f"--max_timestep={loss.max_timestep}")
    if loss.prior_loss_weight != 1.0:
        args.append(f"--prior_loss_weight={loss.prior_loss_weight}")
    if loss.loss_type != "l2":
        args.append(f"--loss_type={loss.loss_type}")
    if loss.huber_schedule is not None:
        args.append(f"--huber_schedule={loss.huber_schedule}")
    if loss.huber_c is not None:
        args.append(f"--huber_c={loss.huber_c}")
    if loss.huber_scale is not None:
        args.append(f"--huber_scale={loss.huber_scale}")
    if loss.debiased_estimation:
        args.append("--debiased_estimation_loss")
    if loss.masked_loss:
        args.append("--masked_loss")
    if loss.scale_v_pred_loss_like_noise_pred:
        args.append("--scale_v_pred_loss_like_noise_pred")
    if loss.v_parameterization:
        args.append("--v_parameterization")
    if loss.v_pred_like_loss is not None:
        args.append(f"--v_pred_like_loss={loss.v_pred_like_loss}")


def _emit_attention_args(
    cfg: TrainingConfig,
    args: list[str],
    env: dict[str, str],
) -> None:
    """Translate ``cfg.attention.training`` into kohya argv + env overrides.

    kohya sd-scripts expose attention selection as a mix of dedicated
    booleans (`--xformers`, `--sdpa`) and a free-form `--attn_mode` flag.
    The exact spelling differs by entry script, but every modern script
    (sdxl/sd3/flux/lumina/anima/hunyuan_image) understands `--xformers`
    and `--sdpa`; the FLUX and SD3 trainers additionally accept
    `--attn_mode=flash` for FlashAttention 2.

    FA3 and FA4 are not first-class in kohya. We emit `--attn_mode flash`
    (so sd-scripts loads its FA path) and stash the requested upgrade in
    the ``LORAHUB_KOHYA_ATTN_OVERRIDE`` env var. The companion module
    ``_attn_patch.py`` reads that var inside the subprocess and swaps the
    attention dispatcher to ``flash_attn_interface`` (FA3) or the FA4 API
    before kohya's modules import it. Loading the patch is the runner's
    responsibility (see ``runner.py``).

    `flex` (`torch.nn.attention.flex_attention`) has no kohya argv yet, so
    we drop back to sdpa with a warning rather than emit a flag the trainer
    will reject. `auto` emits no attention argv so kohya keeps its default
    (sdpa for most arches today).
    """
    backend = cfg.attention.training
    split = cfg.attention.split

    if backend == "auto":
        # Trust kohya's own default; nothing to emit. `--split_attn` only
        # makes sense alongside an explicit attention choice.
        return

    if backend == "torch":
        args.append("--attn_mode=torch")
        return

    if backend == "sdpa":
        # `--sdpa` is the historically-stable spelling across every
        # sd-scripts entry; `--attn_mode=sdpa` is accepted by the newer
        # FLUX/SD3 trainers but `--sdpa` works everywhere.
        args.append("--sdpa")
        return

    if backend == "flex":
        logger.warning(
            "attention.training='flex' is not supported by kohya; "
            "falling back to sdpa"
        )
        args.append("--sdpa")
        return

    if backend == "xformers":
        args.append("--xformers")
        if split:
            args.append("--split_attn")
        return

    if backend == "flash":
        args.append("--attn_mode=flash")
        return

    if backend in ("flash3", "flash4"):
        # kohya itself only ships FA2 wiring. We tell sd-scripts to load
        # `--attn_mode=flash` and let the in-process patch promote the
        # dispatcher to FA3/FA4 via flash_attn_interface or the FA4 API.
        args.append("--attn_mode=flash")
        env[_ATTN_OVERRIDE_ENV] = backend
        return


def _emit_output_args(cfg: TrainingConfig, workspace: Path, args: list[str]) -> None:
    out = cfg.output
    output_dir = out.output_dir if out.output_dir is not None else workspace / "output"
    args += [
        f"--output_dir={output_dir}",
        f"--output_name={out.name}",
        "--save_model_as=safetensors",
        f"--save_precision={out.save_dtype}",
        f"--logging_dir={workspace / 'logs'}",
    ]
    # Save cadence. When the user picks step-level cadence we skip the
    # epoch flag entirely so kohya doesn't double-save (epoch boundary +
    # step boundary both firing on the same iteration would yield two
    # safetensors files for one weight set).
    if out.save_every_n_steps is not None:
        args.append(f"--save_every_n_steps={out.save_every_n_steps}")
    else:
        args.append(f"--save_every_n_epochs={out.save_every_n_epochs}")
    if out.save_last_n_epochs is not None:
        args.append(f"--save_last_n_epochs={out.save_last_n_epochs}")
    if out.save_last_n_steps is not None:
        args.append(f"--save_last_n_steps={out.save_last_n_steps}")
    if out.training_comment is not None:
        args.append(f"--training_comment={out.training_comment}")
    if out.no_metadata:
        args.append("--no_metadata")


def _emit_sampling_args(cfg: TrainingConfig, workspace: Path, args: list[str]) -> None:
    s = cfg.sampling
    if not s.enabled or s.prompts_file is None:
        return
    args += [
        f"--sample_prompts={s.prompts_file}",
        f"--sample_sampler={s.sample_sampler or 'euler_a'}",
    ]
    if s.every_n_epochs is not None:
        args.append(f"--sample_every_n_epochs={s.every_n_epochs}")
    if s.every_n_steps is not None:
        args.append(f"--sample_every_n_steps={s.every_n_steps}")
    if s.at_first:
        args.append("--sample_at_first")


def _emit_monitoring_args(cfg: TrainingConfig, args: list[str]) -> None:
    """Forward ``cfg.monitoring`` to upstream sd-scripts wandb flags.

    Mirrors the official sd-scripts CLI surface (``library/train_util.py``
    ``add_logging_arguments``): ``--log_with``, ``--log_tracker_name``,
    ``--wandb_run_name``. ``--logging_dir`` is already emitted by
    ``_emit_output_args``, so a wandb-enabled run reuses the same
    workspace logs directory and accelerate's bootstrap will set
    ``WANDB_DIR`` accordingly.

    Identity fields not exposed via the sd-scripts CLI (entity / tags /
    notes / run_id / group / job_type / mode / resume / base_url) flow
    through ``WANDB_*`` env vars injected by
    ``lorahub.api.wandb_env.wandb_env``. Secrets (``--wandb_api_key``)
    are deliberately not emitted; the job runner injects
    ``WANDB_API_KEY`` so the api key never lands in the recorded argv.
    """
    monitoring = cfg.monitoring
    if not monitoring.enable_wandb:
        return
    args.append("--log_with=wandb")
    if monitoring.project:
        args.append(f"--log_tracker_name={monitoring.project}")
    if monitoring.run_name:
        args.append(f"--wandb_run_name={monitoring.run_name}")


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
    if r.resume_from is not None:
        args.append(f"--resume={r.resume_from}")
    if r.save_last_n_epochs_state is not None:
        args.append(f"--save_last_n_epochs_state={r.save_last_n_epochs_state}")
    if r.save_last_n_steps_state is not None:
        args.append(f"--save_last_n_steps_state={r.save_last_n_steps_state}")
    if r.skip_until_initial_step:
        args.append("--skip_until_initial_step")
    if r.initial_epoch is not None:
        args.append(f"--initial_epoch={r.initial_epoch}")
    if r.initial_step is not None:
        args.append(f"--initial_step={r.initial_step}")


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
    if v.every_n_steps is not None:
        args.append(f"--validate_every_n_steps={v.every_n_steps}")
    if v.max_samples is not None:
        args.append(f"--max_validation_steps={v.max_samples}")
    if v.seed is not None:
        args.append(f"--validation_seed={v.seed}")


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
    arch = cfg.base_model.arch
    if o.torch_compile:
        args.append("--torch_compile")
    if o.fused_backward_pass:
        args.append("--fused_backward_pass")
    if o.full_bf16:
        args.append("--full_bf16")
    if o.full_fp16:
        args.append("--full_fp16")
    if o.lowram:
        args.append("--lowram")
    if o.highvram:
        args.append("--highvram")
    if o.no_half_vae:
        args.append("--no_half_vae")
    if o.cpu_offload_checkpointing:
        args.append("--cpu_offload_checkpointing")

    # FP8 base — universal across kohya scripts that pass through
    # `train_util.add_training_arguments` / `train_network.py`. Hunyuan Image
    # rejects --fp8_base / --fp8_base_unet at runtime in favour of
    # --fp8_scaled, but we still let users pass them through if they
    # explicitly opt in (the script will raise a clear NotImplementedError).
    if o.fp8_base:
        args.append("--fp8_base")
    if o.fp8_base_unet:
        args.append("--fp8_base_unet")

    # Hunyuan Image-only scaled fp8 + VL text-encoder fp8.
    if o.fp8_scaled:
        if arch in _HUNYUAN_IMAGE_ARCHES:
            args.append("--fp8_scaled")
        else:
            _warn_unsupported(arch, "optimization.fp8_scaled", "--fp8_scaled", "hunyuan_image")
    if o.fp8_vl_text_encoder:
        if arch in _HUNYUAN_IMAGE_ARCHES:
            # NOTE: the upstream argv is `--fp8_vl`, not `--fp8_vl_text_encoder`.
            # The schema field uses the more descriptive name; we translate here.
            args.append("--fp8_vl")
        else:
            _warn_unsupported(arch, "optimization.fp8_vl_text_encoder", "--fp8_vl", "hunyuan_image")

    # Anima-only unsloth offload.
    if o.unsloth_offload_checkpointing:
        if arch in _ANIMA_ARCHES:
            args.append("--unsloth_offload_checkpointing")
        else:
            _warn_unsupported(
                arch,
                "optimization.unsloth_offload_checkpointing",
                "--unsloth_offload_checkpointing",
                "anima",
            )

    # disable_mmap_load_safetensors lives on SDXL + DiT scripts; SD1/SD2 lack it.
    if o.disable_mmap_load_safetensors:
        if arch in _MMAP_DISABLE_ARCHES:
            args.append("--disable_mmap_load_safetensors")
        else:
            _warn_unsupported(
                arch,
                "optimization.disable_mmap_load_safetensors",
                "--disable_mmap_load_safetensors",
                sorted(_MMAP_DISABLE_ARCHES),
            )

    # Text-encoder output caching is universal except for SD1/SD2.
    if o.cache_text_encoder_outputs:
        if arch in _CACHE_TE_ARCHES:
            args.append("--cache_text_encoder_outputs")
        else:
            _warn_unsupported(
                arch,
                "optimization.cache_text_encoder_outputs",
                "--cache_text_encoder_outputs",
                sorted(_CACHE_TE_ARCHES),
            )
    if o.cache_text_encoder_outputs_to_disk:
        if arch in _CACHE_TE_ARCHES:
            args.append("--cache_text_encoder_outputs_to_disk")
        else:
            _warn_unsupported(
                arch,
                "optimization.cache_text_encoder_outputs_to_disk",
                "--cache_text_encoder_outputs_to_disk",
                sorted(_CACHE_TE_ARCHES),
            )

    if o.blocks_to_swap > 0:
        if arch in _BLOCKS_TO_SWAP_ARCHES:
            args.append(f"--blocks_to_swap={o.blocks_to_swap}")
        else:
            logger.warning(
                "kohya: --blocks_to_swap is not supported for arch=%r; "
                "ignoring optimization.blocks_to_swap=%d "
                "(supported arches: %s)",
                arch,
                o.blocks_to_swap,
                sorted(_BLOCKS_TO_SWAP_ARCHES),
            )


def _warn_unsupported(
    arch: str,
    field: str,
    flag: str,
    supported: object,
) -> None:
    """One-line warning for fields that don't map to argv on this arch."""
    logger.warning(
        "kohya: %s is not supported for arch=%r; ignoring %s "
        "(supported arches: %s)",
        flag,
        arch,
        field,
        supported,
    )


def _emit_extra_args(cfg: TrainingConfig, args: list[str]) -> None:
    """Append user-provided escape-hatch args verbatim. Last write wins.

    Accepts native bool values as well as their string forms ("true" /
    "false", case-insensitive) so the YAML editor and the form-driven
    KeyValueTextArea (which only emits strings) feed argparse the same
    shape: ``True`` -> single store_true flag, ``False`` / ``None`` ->
    omitted, anything else -> ``--flag=value``.
    """
    for key, value in cfg.backend.extra_args.items():
        flag = f"--{key}" if not key.startswith("--") else key
        normalized = value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "true":
                normalized = True
            elif lowered == "false":
                normalized = False
        if normalized is True:
            args.append(flag)
        elif normalized is False or normalized is None:
            continue
        else:
            args.append(f"{flag}={normalized}")


# --------------------------------------------------------------------------- #
# Arch-specific path bag (`cfg.base_model.arch_paths`).
# --------------------------------------------------------------------------- #


def _emit_arch_paths_args(cfg: TrainingConfig, args: list[str]) -> None:
    """Translate `cfg.base_model.arch_paths` into argv per arch.

    Each kohya entry script accepts a different subset of these flags. We
    split the work between FLUX/SD3/Anima/HunyuanImage helpers; everything
    else runs on the universal `--vae` knob (already covered by the model
    helper).
    """
    arch = cfg.base_model.arch
    p = cfg.base_model.arch_paths

    if arch in _FLUX_ARCHES:
        _emit_flux_paths(p, args)
    elif arch in _SD3_ARCHES:
        _emit_sd3_paths(p, args)
    elif arch in _ANIMA_ARCHES:
        _emit_anima_paths(p, args)
    elif arch in _HUNYUAN_IMAGE_ARCHES:
        _emit_hunyuan_image_paths(p, args)

    # Per-encoder dropout rates and attention masks straddle FLUX + SD3 (the
    # SD3 train util redefines the FLUX flags), so they're handled in the
    # arch-specific helpers above. Nothing to do here for other arches.
    _warn_unused_arch_paths(arch, p)


def _emit_flux_paths(p, args: list[str]) -> None:
    """FLUX argv: --clip_l, --t5xxl, --ae, plus the encoder dropout / mask /
    guidance scale knobs added by `library.flux_train_utils`."""
    if p.clip_l is not None:
        args.append(f"--clip_l={p.clip_l}")
    if p.t5xxl is not None:
        args.append(f"--t5xxl={p.t5xxl}")
    if p.ae is not None:
        args.append(f"--ae={p.ae}")
    if p.t5xxl_max_token_length is not None:
        args.append(f"--t5xxl_max_token_length={p.t5xxl_max_token_length}")
    if p.apply_t5_attn_mask:
        args.append("--apply_t5_attn_mask")
    if p.guidance_scale is not None:
        args.append(f"--guidance_scale={p.guidance_scale}")
    if p.t5_dropout_rate > 0:
        args.append(f"--t5_dropout_rate={p.t5_dropout_rate}")
    if p.clip_l_dropout_rate > 0:
        args.append(f"--clip_l_dropout_rate={p.clip_l_dropout_rate}")


def _emit_sd3_paths(p, args: list[str]) -> None:
    """SD3 argv: --clip_l, --clip_g, --t5xxl, the encoder masks/dropouts and
    SD3.5-specific positional-embed knobs."""
    if p.clip_l is not None:
        args.append(f"--clip_l={p.clip_l}")
    if p.clip_g is not None:
        args.append(f"--clip_g={p.clip_g}")
    if p.t5xxl is not None:
        args.append(f"--t5xxl={p.t5xxl}")
    if p.t5xxl_max_token_length is not None:
        args.append(f"--t5xxl_max_token_length={p.t5xxl_max_token_length}")
    if p.apply_t5_attn_mask:
        args.append("--apply_t5_attn_mask")
    if p.apply_lg_attn_mask:
        args.append("--apply_lg_attn_mask")
    if p.t5_dropout_rate > 0:
        args.append(f"--t5_dropout_rate={p.t5_dropout_rate}")
    if p.clip_l_dropout_rate > 0:
        args.append(f"--clip_l_dropout_rate={p.clip_l_dropout_rate}")
    if p.clip_g_dropout_rate > 0:
        args.append(f"--clip_g_dropout_rate={p.clip_g_dropout_rate}")
    if p.pos_emb_random_crop_rate > 0:
        args.append(f"--pos_emb_random_crop_rate={p.pos_emb_random_crop_rate}")
    if p.enable_scaled_pos_embed:
        args.append("--enable_scaled_pos_embed")
    if p.t5xxl_device is not None:
        args.append(f"--t5xxl_device={p.t5xxl_device}")
    if p.t5xxl_dtype is not None:
        args.append(f"--t5xxl_dtype={p.t5xxl_dtype}")


def _emit_anima_paths(p, args: list[str]) -> None:
    """Anima argv: --qwen3, --llm_adapter_path, --t5_tokenizer_path, plus
    the Qwen3/T5 max-token-length knobs and the VAE memory tweaks. Note the
    upstream spelling differs from our schema field names (e.g. `llm_adapter`
    -> `--llm_adapter_path`)."""
    if p.qwen3 is not None:
        args.append(f"--qwen3={p.qwen3}")
    if p.llm_adapter is not None:
        args.append(f"--llm_adapter_path={p.llm_adapter}")
    if p.t5_tokenizer is not None:
        args.append(f"--t5_tokenizer_path={p.t5_tokenizer}")
    if p.qwen3_max_token_length is not None:
        args.append(f"--qwen3_max_token_length={p.qwen3_max_token_length}")
    if p.t5_max_token_length is not None:
        args.append(f"--t5_max_token_length={p.t5_max_token_length}")
    if p.vae_chunk_size is not None:
        args.append(f"--vae_chunk_size={p.vae_chunk_size}")
    if p.vae_disable_cache:
        args.append("--vae_disable_cache")


def _emit_hunyuan_image_paths(p, args: list[str]) -> None:
    """Hunyuan Image argv: --text_encoder, --byt5, --text_encoder_cpu,
    --vae_chunk_size."""
    if p.text_encoder is not None:
        args.append(f"--text_encoder={p.text_encoder}")
    if p.byt5 is not None:
        args.append(f"--byt5={p.byt5}")
    if p.text_encoder_cpu:
        args.append("--text_encoder_cpu")
    if p.vae_chunk_size is not None:
        args.append(f"--vae_chunk_size={p.vae_chunk_size}")


def _warn_unused_arch_paths(arch: str, p) -> None:
    """Surface a single warning when the user filled an arch-only path field
    that doesn't apply to the chosen arch.

    We only warn for the high-signal mismatches (FLUX paths on a non-FLUX
    arch, etc.) — full coverage of every cross-product would just add noise.
    """
    if arch not in _FLUX_ARCHES and arch not in _SD3_ARCHES and (
        p.clip_l is not None or p.t5xxl is not None or p.ae is not None
    ):
        logger.warning(
            "kohya: arch_paths.clip_l/t5xxl/ae are FLUX/SD3-only; "
            "ignoring on arch=%r",
            arch,
        )
    if arch not in _ANIMA_ARCHES and (
        p.qwen3 is not None or p.llm_adapter is not None or p.t5_tokenizer is not None
    ):
        logger.warning(
            "kohya: arch_paths.qwen3/llm_adapter/t5_tokenizer are "
            "Anima-only; ignoring on arch=%r",
            arch,
        )
    if arch not in _HUNYUAN_IMAGE_ARCHES and p.byt5 is not None:
        logger.warning(
            "kohya: arch_paths.byt5 is HunyuanImage-only; ignoring on arch=%r",
            arch,
        )


# --------------------------------------------------------------------------- #
# Flow-matching hyperparameters (`cfg.flow_match`).
# --------------------------------------------------------------------------- #


def _emit_flow_match_args(cfg: TrainingConfig, args: list[str]) -> None:
    """Translate `cfg.flow_match` into argv on the DiT-family arches.

    The flow-matching knobs (`--timestep_sampling`, `--sigmoid_scale`,
    `--model_prediction_type`, `--discrete_flow_shift`) are added by FLUX /
    HunyuanImage / Anima entry scripts. SD3 ships an extra `--training_shift`.
    The Diffusers-style weighting knobs (`--weighting_scheme`, `--logit_*`,
    `--mode_scale`) live in `add_dit_training_arguments` and so apply to
    every DiT arch.

    SD1.x / SD2.x / SDXL aren't flow-matching trainers; we drop the args
    silently when set on those arches because the existing flow_match block
    in a recipe usually means "this is a multi-arch recipe template" rather
    than user error.
    """
    arch = cfg.base_model.arch
    fm = cfg.flow_match

    if arch not in _DIT_ARCHES:
        # User filled fields that don't apply -- warn once if any is set.
        if any(
            v is not None
            for v in (
                fm.timestep_sampling,
                fm.sigmoid_scale,
                fm.model_prediction_type,
                fm.discrete_flow_shift,
                fm.training_shift,
                fm.weighting_scheme,
                fm.logit_mean,
                fm.logit_std,
                fm.mode_scale,
            )
        ):
            logger.warning(
                "kohya: flow_match fields are only consumed by FLUX / SD3 / "
                "Lumina / Anima / HunyuanImage; ignoring on arch=%r",
                arch,
            )
        return

    if fm.timestep_sampling is not None:
        args.append(f"--timestep_sampling={fm.timestep_sampling}")
    if fm.sigmoid_scale is not None:
        args.append(f"--sigmoid_scale={fm.sigmoid_scale}")
    if fm.model_prediction_type is not None:
        args.append(f"--model_prediction_type={fm.model_prediction_type}")
    if fm.discrete_flow_shift is not None:
        args.append(f"--discrete_flow_shift={fm.discrete_flow_shift}")

    # SD3-only.
    if fm.training_shift is not None:
        if arch in _SD3_ARCHES:
            args.append(f"--training_shift={fm.training_shift}")
        else:
            _warn_unsupported(arch, "flow_match.training_shift", "--training_shift", "sd3")

    # Diffusers-style weighting (`add_dit_training_arguments`).
    if fm.weighting_scheme is not None:
        args.append(f"--weighting_scheme={fm.weighting_scheme}")
    if fm.logit_mean is not None:
        args.append(f"--logit_mean={fm.logit_mean}")
    if fm.logit_std is not None:
        args.append(f"--logit_std={fm.logit_std}")
    if fm.mode_scale is not None:
        args.append(f"--mode_scale={fm.mode_scale}")


# --------------------------------------------------------------------------- #
# DataLoader (`cfg.dataloader`) — universal across kohya entry scripts.
# --------------------------------------------------------------------------- #


def _emit_dataloader_args(cfg: TrainingConfig, args: list[str]) -> None:
    """Emit the data-loading throughput knobs.

    `num_workers` defaults to 8 in both our schema and kohya — emit only when
    the user moves it. `vae_batch_size` defaults to 1 likewise.
    `text_encoder_batch_size` lives in `add_dit_training_arguments`, so we
    gate it on the DiT family / SDXL.
    """
    d = cfg.dataloader
    if d.num_workers != 8:
        args.append(f"--max_data_loader_n_workers={d.num_workers}")
    if d.persistent_workers and d.num_workers > 0:
        args.append("--persistent_data_loader_workers")
    if d.vae_batch_size != 1:
        args.append(f"--vae_batch_size={d.vae_batch_size}")
    if d.text_encoder_batch_size is not None:
        arch = cfg.base_model.arch
        if arch in _CACHE_TE_ARCHES:
            args.append(f"--text_encoder_batch_size={d.text_encoder_batch_size}")
        else:
            _warn_unsupported(
                arch,
                "dataloader.text_encoder_batch_size",
                "--text_encoder_batch_size",
                sorted(_CACHE_TE_ARCHES),
            )


# --------------------------------------------------------------------------- #
# Augmentation (`cfg.augmentation`) — universal across kohya entry scripts.
# --------------------------------------------------------------------------- #


def _emit_augmentation_args(cfg: TrainingConfig, args: list[str]) -> None:
    """Per-image augmentation flags. All live on `add_dataset_arguments`."""
    a = cfg.augmentation
    if a.flip:
        args.append("--flip_aug")
    if a.color:
        args.append("--color_aug")
    if a.random_crop:
        args.append("--random_crop")
    if a.face_crop_aug_range is not None:
        args.append(f"--face_crop_aug_range={a.face_crop_aug_range}")
    if a.alpha_mask:
        args.append("--alpha_mask")


# --------------------------------------------------------------------------- #
# Caption knobs (`cfg.dataset.caption`) — extends the existing dataset.toml
# emission with argv-only kohya knobs.
# --------------------------------------------------------------------------- #


def _emit_caption_args(cfg: TrainingConfig, args: list[str]) -> None:
    """Caption-related kohya argv that aren't expressible in dataset.toml.

    Most dataset-level caption knobs live in the toml (shuffle, drop_rate,
    extension); these argv flags govern dropout cadence, secondary tokens,
    wildcards, prefix/suffix, token warmup, and weighted captions.
    """
    c = cfg.dataset.caption
    if c.dropout_every_n_epochs > 0:
        args.append(f"--caption_dropout_every_n_epochs={c.dropout_every_n_epochs}")
    if c.tag_dropout_rate > 0:
        args.append(f"--caption_tag_dropout_rate={c.tag_dropout_rate}")
    if c.keep_tokens > 0:
        args.append(f"--keep_tokens={c.keep_tokens}")
    if c.keep_tokens_separator is not None:
        args.append(f"--keep_tokens_separator={c.keep_tokens_separator}")
    if c.secondary_separator is not None:
        args.append(f"--secondary_separator={c.secondary_separator}")
    if c.enable_wildcard:
        args.append("--enable_wildcard")
    if c.prefix is not None:
        args.append(f"--caption_prefix={c.prefix}")
    if c.suffix is not None:
        args.append(f"--caption_suffix={c.suffix}")
    if c.max_token_length is not None:
        args.append(f"--max_token_length={c.max_token_length}")
    if c.token_warmup_min is not None:
        args.append(f"--token_warmup_min={c.token_warmup_min}")
    if c.token_warmup_step is not None:
        args.append(f"--token_warmup_step={c.token_warmup_step}")
    if c.weighted:
        args.append("--weighted_captions")


# --------------------------------------------------------------------------- #
# Bucket knobs (`cfg.dataset.bucket`) — extends the dataset.toml emission
# with argv-only kohya flags.
# --------------------------------------------------------------------------- #


def _emit_bucket_args(cfg: TrainingConfig, args: list[str]) -> None:
    """Bucket-related kohya argv that aren't inside dataset.toml.

    `enable_bucket`, `min_bucket_reso`, `max_bucket_reso`, `bucket_reso_steps`,
    and `skip_image_resolution` are emitted via dataset.toml. The remaining
    knobs (`bucket_no_upscale`, `resize_interpolation`) are top-level CLI
    flags that the trainer consumes alongside the toml.
    """
    b = cfg.dataset.bucket
    if not b.enabled:
        return
    if b.no_upscale:
        args.append("--bucket_no_upscale")
    if b.resize_interpolation is not None:
        args.append(f"--resize_interpolation={b.resize_interpolation}")


# --------------------------------------------------------------------------- #
# Anima per-module learning-rate overrides (`cfg.network.module_lr`).
# --------------------------------------------------------------------------- #


def _emit_per_module_lr_args(cfg: TrainingConfig, args: list[str]) -> None:
    """Emit the Anima per-module LR argv (`--llm_adapter_lr`, `--self_attn_lr`,
    etc). Other arches don't expose these flags so we warn + skip when the
    user populates the field on a non-Anima arch."""
    lr = cfg.network.module_lr
    if lr is None:
        return
    arch = cfg.base_model.arch
    if arch not in _ANIMA_ARCHES:
        if any(
            v is not None
            for v in (lr.llm_adapter, lr.self_attn, lr.cross_attn, lr.mlp, lr.mod)
        ):
            logger.warning(
                "kohya: network.module_lr fields are Anima-only; "
                "ignoring on arch=%r",
                arch,
            )
        return
    if lr.llm_adapter is not None:
        args.append(f"--llm_adapter_lr={lr.llm_adapter}")
    if lr.self_attn is not None:
        args.append(f"--self_attn_lr={lr.self_attn}")
    if lr.cross_attn is not None:
        args.append(f"--cross_attn_lr={lr.cross_attn}")
    if lr.mlp is not None:
        args.append(f"--mlp_lr={lr.mlp}")
    if lr.mod is not None:
        args.append(f"--mod_lr={lr.mod}")


# --------------------------------------------------------------------------- #
# Output metadata (`cfg.output.metadata`) — `--metadata_*` model-spec flags.
# --------------------------------------------------------------------------- #


_METADATA_KEYS: frozenset[str] = frozenset({
    "title",
    "author",
    "description",
    "license",
    "tags",
    "usage_hint",
    "thumbnail",
    "merged_from",
    "trigger_phrase",
    "preprocessor",
    "is_negative_embedding",
})


def _emit_metadata_args(cfg: TrainingConfig, args: list[str]) -> None:
    """Forward `output.metadata` keys onto kohya's `--metadata_<key>` flags.

    Unknown keys are passed through verbatim so the recipe stays
    forward-compatible with new sai_model_spec entries — kohya rejects bogus
    keys at argparse time, surfacing the error to the user.
    """
    md = cfg.output.metadata
    for key, value in md.items():
        if key in _METADATA_KEYS:
            args.append(f"--metadata_{key}={value}")
        else:
            args.append(f"--metadata_{key}={value}")
            logger.warning(
                "kohya: output.metadata.%s is not a known sai_model_spec "
                "field; passing through verbatim",
                key,
            )
