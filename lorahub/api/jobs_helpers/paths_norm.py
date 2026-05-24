"""Recipe-path absolutisation.

Training subprocesses (kohya / diffusion-pipe / anima_lora) run with
their cwd pinned to the backend's own repo, so a recipe path like
``./models/foo.safetensors`` would otherwise be looked up under
``diffusion-pipe/models/`` instead of the lorahub project root. The
helpers here resolve every path field on a ``TrainingConfig`` against
the API server's cwd before the cfg ever reaches a compiler.
"""

from __future__ import annotations

from pathlib import Path

from lorahub.core.config.schema import TrainingConfig


def _absolutise(p: Path | str | None, base: Path) -> Path | None:
    """Resolve a recipe-relative path against the project root.

    When the literal resolved path doesn't exist on disk but the
    same path under ``base/models/`` does, prefer the latter — this
    rescues recipe yamls that store bare relative paths like
    ``circlestone-labs__Anima/foo.safetensors`` (the old picker
    output before the prefix fix) without forcing every user to
    re-save.
    """
    if p is None:
        return None
    path = Path(str(p)).expanduser()
    if path.is_absolute():
        return path
    resolved = (base / path).resolve()
    if not resolved.exists():
        # Try the models/ fallback. Only kicks in when the literal
        # path is missing — never overrides an existing file.
        models_fallback = (base / "models" / path).resolve()
        if models_fallback.exists():
            return models_fallback
    return resolved


def _normalize_recipe_paths(cfg: TrainingConfig, base: Path | None = None) -> TrainingConfig:
    """Make every path field in `cfg` absolute, anchored at `base`.

    Mutates a *copy* of the cfg (Pydantic models are effectively
    mutable; we still touch fields in place but only after the cfg
    snapshot has been captured for persistence by callers that care).
    """
    base_dir = (base or Path.cwd()).resolve()

    cfg.base_model.checkpoint = _absolutise(cfg.base_model.checkpoint, base_dir)  # type: ignore[assignment]
    if cfg.base_model.vae is not None:
        cfg.base_model.vae = _absolutise(cfg.base_model.vae, base_dir)
    paths = cfg.base_model.arch_paths
    for fname in (
        "clip_l", "clip_g", "t5xxl", "ae", "transformer", "text_encoder",
        "llm", "byt5", "qwen3", "t5_tokenizer", "llm_adapter",
    ):
        cur = getattr(paths, fname, None)
        if cur is not None:
            setattr(paths, fname, _absolutise(cur, base_dir))

    cfg.dataset.source = _absolutise(cfg.dataset.source, base_dir)  # type: ignore[assignment]
    if cfg.dataset.conditioning_dir is not None:
        cfg.dataset.conditioning_dir = _absolutise(cfg.dataset.conditioning_dir, base_dir)
    if cfg.dataset.reg_source is not None:
        cfg.dataset.reg_source = _absolutise(cfg.dataset.reg_source, base_dir)
    for sub in cfg.dataset.subsets:
        sub.path = _absolutise(sub.path, base_dir)  # type: ignore[assignment]
        if sub.mask_path is not None:
            sub.mask_path = _absolutise(sub.mask_path, base_dir)

    if cfg.output.output_dir is not None:
        cfg.output.output_dir = _absolutise(cfg.output.output_dir, base_dir)

    # Free-form dp model_paths bag — every value is a path string.
    if cfg.backend.diffusion_pipe is not None:
        mp = cfg.backend.diffusion_pipe.model_paths
        if mp:
            cfg.backend.diffusion_pipe.model_paths = {
                k: str(_absolutise(v, base_dir)) for k, v in mp.items()
            }

    if cfg.network.init_from is not None:
        cfg.network.init_from = _absolutise(cfg.network.init_from, base_dir)
    if cfg.network.dim_from_weights is not None:
        cfg.network.dim_from_weights = _absolutise(cfg.network.dim_from_weights, base_dir)
    cfg.network.base_weights = [
        _absolutise(p, base_dir) for p in cfg.network.base_weights  # type: ignore[misc]
    ]

    if cfg.resume.resume_from is not None:
        cfg.resume.resume_from = _absolutise(cfg.resume.resume_from, base_dir)

    if cfg.sampling.prompts_file is not None:
        cfg.sampling.prompts_file = _absolutise(cfg.sampling.prompts_file, base_dir)

    return cfg


__all__ = ["_absolutise", "_normalize_recipe_paths"]
