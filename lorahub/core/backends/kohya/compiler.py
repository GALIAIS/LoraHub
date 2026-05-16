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
) -> tuple[str, list[str], dict[Path, str]]:
    """Translate a recipe into (script_name, argv, files_to_write).

    `files_to_write` is a mapping of absolute path to file content that the
    caller must write before launching the subprocess (currently just
    `<workspace>/dataset.toml`). Returning it instead of writing it ourselves
    keeps the compiler a pure function.
    """
    script = _pick_script(recipe.base_model.arch)
    args: list[str] = []
    files: dict[Path, str] = {}

    _emit_model_args(recipe, args)
    _emit_dataset_args(recipe, workspace, args, files)
    _emit_network_args(recipe, args)
    _emit_optimizer_args(recipe, args)
    _emit_schedule_args(recipe, args)
    _emit_precision_args(recipe, args)
    _emit_output_args(recipe, workspace, args)
    _emit_sampling_args(recipe, workspace, args)
    _emit_resume_args(recipe, args)
    _emit_validation_args(recipe, args)
    _emit_variant_args(recipe, args)
    _emit_extra_args(recipe, args)

    return script, args, files


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


def _emit_dataset_args(
    recipe: RecipeConfig,
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
    files[toml_path] = _build_dataset_toml(recipe)
    args.append(f"--dataset_config={toml_path}")


def _build_dataset_toml(recipe: RecipeConfig) -> str:
    ds = recipe.dataset
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
        f"batch_size = {recipe.schedule.batch_size}",
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


def _emit_resume_args(recipe: RecipeConfig, args: list[str]) -> None:
    """Emit the kohya `--save_state*` flags so a run can be resumed.

    sd-scripts writes the state directory inside `--output_dir` next to
    the safetensors as `<output_name>-state`. We only enable local state
    saving here; the resume route scans the same directory at restart
    time. `--save_state_to_huggingface` is intentionally not surfaced.
    """
    r = recipe.resume
    if r.save_state:
        args.append("--save_state")
    if r.save_state_at_end:
        args.append("--save_state_on_train_end")
    if r.save_state_every_n_epochs is not None:
        args.append(f"--save_state_every_n_epochs={r.save_state_every_n_epochs}")


def _emit_validation_args(recipe: RecipeConfig, args: list[str]) -> None:
    """Emit sd-scripts' validation-split flags when `dataset.val_split > 0`.

    sd-scripts 1.x exposes a held-out validation split via three flags:
    `--validation_split_percentage` (integer percent, 0 disables it),
    `--validate_every_n_epochs`, and `--max_validation_steps`. We round the
    fractional `val_split` to the nearest percent so a YAML value of `0.10`
    lands on `10` instead of `10.0`. `validation.max_samples` is mapped to
    `--max_validation_steps` because that is the per-eval-pass cap kohya
    actually accepts; users can always override via `backend.extra_args`.
    """
    ds = recipe.dataset
    if ds.val_split <= 0.0:
        return

    percent = max(1, round(ds.val_split * 100))
    args.append(f"--validation_split_percentage={percent}")

    v = recipe.validation
    args.append(f"--validate_every_n_epochs={v.every_n_epochs}")
    if v.max_samples is not None:
        args.append(f"--max_validation_steps={v.max_samples}")


def _emit_variant_args(recipe: RecipeConfig, args: list[str]) -> None:
    """Inject argv tweaks specific to an SDXL sub-architecture.

    Conservative for now: only the Pony lineage gets `--clip_skip=2`,
    which matches how Pony was trained and is the most-cited recipe
    delta the community converges on. Illustrious/NoobAI/Animagine
    don't add argv yet - their tuning lives in the scaffolder's LR
    defaults. User overrides via `backend.extra_args` win because
    `_emit_extra_args` runs after this hook.
    """
    variant = recipe.base_model.arch_variant
    if variant == "pony":
        args.append("--clip_skip=2")


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

