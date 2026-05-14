"""Compile a semantic `RecipeConfig` into kohya-ss command-line arguments.

This is the most important translation layer in the project: upper layers
stay stable while kohya parameter names drift. Keep it a pure function so
we can unit-test thoroughly without touching disk or GPU.
"""

from __future__ import annotations

from pathlib import Path

from lorahub.core.config.schema import RecipeConfig

# Map our optimizer names to kohya's --optimizer_type values
_OPTIMIZER_MAP: dict[str, str] = {
    "adamw": "AdamW",
    "adamw8bit": "AdamW8bit",
    "lion": "Lion",
    "lion8bit": "Lion8bit",
    "prodigy": "Prodigy",
    "dadaptation": "DAdaptation",
}

# Map our network types to kohya's --network_module
_NETWORK_MODULE_MAP: dict[str, str] = {
    "lora": "networks.lora",
    "locon": "lycoris.kohya",
    "loha": "lycoris.kohya",
    "dora": "networks.lora",
}


class CompilationError(ValueError):
    """Raised when a recipe cannot be expressed in kohya's argument vocabulary."""


def compile_recipe(
    recipe: RecipeConfig,
    workspace: Path,
) -> tuple[str, list[str]]:
    """Translate a recipe into (script_name, argv_for_subprocess).

    `script_name` is the kohya entry script to run (e.g. `sdxl_train_network.py`).
    `argv` is the full argument list, paths already resolved relative to
    `workspace` where appropriate.
    """
    script = _pick_script(recipe.base_model.arch)
    args: list[str] = []

    _emit_model_args(recipe, args)
    _emit_dataset_args(recipe, args)
    _emit_network_args(recipe, args)
    _emit_optimizer_args(recipe, args)
    _emit_schedule_args(recipe, args)
    _emit_precision_args(recipe, args)
    _emit_output_args(recipe, workspace, args)
    _emit_sampling_args(recipe, workspace, args)
    _emit_extra_args(recipe, args)

    return script, args


def _pick_script(arch: str) -> str:
    match arch:
        case "sdxl":
            return "sdxl_train_network.py"
        case "sd15":
            return "train_network.py"
        case "flux":
            return "flux_train_network.py"
        case "sd3":
            return "sd3_train_network.py"
        case _:
            msg = f"unsupported base_model.arch: {arch}"
            raise CompilationError(msg)


def _emit_model_args(recipe: RecipeConfig, args: list[str]) -> None:
    args += [f"--pretrained_model_name_or_path={recipe.base_model.checkpoint}"]
    if recipe.base_model.vae is not None:
        args += [f"--vae={recipe.base_model.vae}"]


def _emit_dataset_args(recipe: RecipeConfig, args: list[str]) -> None:
    ds = recipe.dataset
    args += [f"--train_data_dir={ds.source}"]

    if len(ds.resolution) == 1:
        args += [f"--resolution={ds.resolution[0]}"]
    else:
        args += [f"--resolution={ds.resolution[0]},{ds.resolution[1]}"]

    if ds.bucket.enabled:
        args += [
            "--enable_bucket",
            f"--min_bucket_reso={ds.bucket.min_size}",
            f"--max_bucket_reso={ds.bucket.max_size}",
            f"--bucket_reso_steps={ds.bucket.step}",
        ]

    if ds.caption.shuffle:
        args += ["--shuffle_caption"]
    if ds.caption.drop_rate > 0:
        args += [f"--caption_dropout_rate={ds.caption.drop_rate}"]
    if ds.caption.ext != ".txt":
        args += [f"--caption_extension={ds.caption.ext}"]


def _emit_network_args(recipe: RecipeConfig, args: list[str]) -> None:
    n = recipe.network
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

    if not n.target_text_encoder:
        args += ["--network_train_unet_only"]
    elif not n.target_unet:
        args += ["--network_train_text_encoder_only"]

    if network_args:
        args += ["--network_args"] + network_args


def _emit_optimizer_args(recipe: RecipeConfig, args: list[str]) -> None:
    o = recipe.optimizer
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


def _emit_schedule_args(recipe: RecipeConfig, args: list[str]) -> None:
    s = recipe.schedule
    args += [
        f"--max_train_epochs={s.epochs}",
        f"--train_batch_size={s.batch_size}",
        f"--gradient_accumulation_steps={s.grad_accum}",
    ]
    if s.max_steps is not None:
        args += [f"--max_train_steps={s.max_steps}"]


def _emit_precision_args(recipe: RecipeConfig, args: list[str]) -> None:
    if recipe.precision != "fp32":
        args += [f"--mixed_precision={recipe.precision}"]
    if recipe.gradient_checkpointing:
        args += ["--gradient_checkpointing"]
    if recipe.cache_latents:
        args += ["--cache_latents"]


def _emit_output_args(recipe: RecipeConfig, workspace: Path, args: list[str]) -> None:
    out = recipe.output
    output_dir = out.output_dir if out.output_dir is not None else workspace / "output"
    args += [
        f"--output_dir={output_dir}",
        f"--output_name={out.name}",
        f"--save_every_n_epochs={out.save_every_n_epochs}",
        "--save_model_as=safetensors",
        f"--save_precision={out.save_dtype}",
        f"--logging_dir={workspace / 'logs'}",
    ]


def _emit_sampling_args(recipe: RecipeConfig, workspace: Path, args: list[str]) -> None:
    s = recipe.sampling
    if not s.enabled or s.prompts_file is None:
        return
    args += [
        f"--sample_every_n_epochs={s.every_n_epochs}",
        f"--sample_prompts={s.prompts_file}",
        "--sample_sampler=euler_a",
    ]


def _emit_extra_args(recipe: RecipeConfig, args: list[str]) -> None:
    """Append user-provided escape-hatch args verbatim. Last write wins."""
    for key, value in recipe.backend.extra_args.items():
        flag = f"--{key}" if not key.startswith("--") else key
        if value is True:
            args.append(flag)
        elif value is False or value is None:
            continue
        else:
            args.append(f"{flag}={value}")

