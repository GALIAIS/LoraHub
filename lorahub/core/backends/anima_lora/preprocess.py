"""Auto-preprocess for anima_lora — bring the LoRA cache directory up to date.

anima_lora's training loop reads a four-file-per-image cache layout under
``post_image_dataset/lora/``:

    {stem}_{WxH}_anima.npz       (VAE latent, written by cache_latents.py)
    {stem}_anima_te.safetensors  (Qwen3 + T5 TE outputs, cache_text_embeddings.py)
    {stem}_anima_pe.safetensors  (PE-Core features, cache_pe_encoder.py — only
                                  IP-Adapter / DCW need it; the LoRA path skips it)

Upstream's flow assumes the user runs ``make preprocess`` once before the
first ``make lora``; LoraHub instead auto-fills the cache from the user's
raw image directory the moment a recipe is launched. This keeps
``cfg.dataset.source`` consistent with kohya / dp (always points at the
raw image directory), removes the "two source paths depending on backend"
foot-gun, and lets users switch backends without touching their dataset.

The helper here only wires the **standard LoRA path**:

    raw images (cfg.dataset.source)
        → resize_images.py  → <workspace>/post_image_dataset/resized/
        → cache_latents.py  → <workspace>/post_image_dataset/lora/{stem}_{WxH}_anima.npz
        → cache_text_embeddings.py → <workspace>/post_image_dataset/lora/{stem}_anima_te.safetensors

PE feature caching (`cache_pe_encoder.py`) is intentionally not run from
here — only the IP-Adapter / DCW v4 paths use those sidecars and they
need extra args (centroid, etc.) that aren't surfaced in the standard
LoraHub recipe schema. Future cut: detect ``method=ip_adapter`` and
chain the PE step automatically.

Bucket-fingerprint invalidation
-------------------------------

The latent ``.npz`` keys are bucket-specific (``latents_{H/8}x{W/8}``),
so when the user flips ``enable_native_flatten`` / ``bucket_table`` /
``static_token_count`` between training runs, the previous-run npz no
longer contains the keys the new bucket assignment will request. We
write a ``_lorahub_bucket_fingerprint.json`` next to the cache and
treat any mismatch as "cache is stale, rebuild from scratch" — better
to spend the preprocess time again than to hit
``ValueError: latents_NxN not found in {npz_path}`` mid-training.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lorahub.core.backends._common.runner import (
    EventListener,
    SubprocessRunner,
)
from lorahub.core.backends.anima_lora.bootstrap import AnimaLoraEnv
from lorahub.core.backends.anima_lora.parser import parse_line
from lorahub.core.config.schema import BaseModelConfig
from lorahub.core.events import EventType, TrainingEvent

__all__ = [
    "PreprocessError",
    "ensure_cache",
]


# Filename for the bucket fingerprint sidecar. Lives in the cache_dir
# alongside the latent / TE caches.
_FINGERPRINT_FILENAME = "_lorahub_bucket_fingerprint.json"


# Image extensions the LoRA pipeline accepts. Kept in lock-step with
# upstream's ``library/datasets/image_utils.IMAGE_EXTENSIONS``; the set
# duplicated here so we don't import torch just to compute "is this a
# JPEG". Diverging will only mean LoraHub's "missing cache" detection is
# slightly looser than upstream's actual processing — never fatal.
_IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
)


class PreprocessError(RuntimeError):
    """A preprocess subprocess exited non-zero, or no images were found."""


def _list_images(image_dir: Path) -> list[Path]:
    """Recursive image enumeration mirroring upstream's preprocess scripts.

    Upstream walks subfolders when ``--recursive`` is passed; we always
    pass ``--recursive`` so this listing must include nested layouts.
    """
    if not image_dir.is_dir():
        return []
    return sorted(
        p
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


def _missing_caches(
    images: list[Path],
    cache_dir: Path,
) -> list[Path]:
    """Return the subset of images that don't have a TE-cache sidecar yet.

    We use ``{stem}_anima_te.safetensors`` as the gate because it's the
    last file written in upstream's preprocess chain (resize → VAE
    latent → TE embedding). If TE is present, the latent has to be
    too — and any preprocess re-run is idempotent, so it's safe to
    declare "everything is cached" once TE exists.

    The latent cache filename includes resolution (``{WxH}_anima.npz``)
    which we'd have to predict from the bucket manager to match exactly;
    keying off TE keeps the check simple and correct.
    """
    if not cache_dir.is_dir():
        return list(images)
    out: list[Path] = []
    for img in images:
        te = cache_dir / f"{img.stem}_anima_te.safetensors"
        if not te.is_file():
            out.append(img)
    return out


# Type alias for the SubprocessRunner factory that ``ensure_cache`` uses.
# Tests inject a stub here so we can exercise the orchestration logic
# without spawning a real Python subprocess.
RunnerFactory = Callable[..., SubprocessRunner]


def _default_runner_factory(**kwargs: object) -> SubprocessRunner:
    return SubprocessRunner(**kwargs)  # type: ignore[arg-type]


def _bucket_fingerprint(opts: Any | None) -> dict[str, Any]:
    """Snapshot of every option that influences the bucket selection.

    When any of these change, every ``.npz`` written by an earlier run
    becomes useless: the new bucket assignment will look up keys
    (``latents_{H/8}x{W/8}``) that simply don't exist in the on-disk
    arrays. We persist this dict next to the cache and force a
    rebuild whenever it disagrees with the current recipe.

    ``opts`` is duck-typed (``AnimaLoraOptions`` from
    ``lorahub.core.config.schema``); we intentionally don't import the
    type to keep ``preprocess.py`` cheap to import in tests. ``None``
    means "no options provided" → we fall back to a permissive
    fingerprint that always invalidates conservatively.
    """
    if opts is None:
        return {"version": 1, "_no_opts": True}
    return {
        "version": 1,
        "static_token_count": getattr(opts, "static_token_count", None),
        "enable_native_flatten": bool(
            getattr(opts, "enable_native_flatten", False),
        ),
        "bucket_table": getattr(opts, "bucket_table", None),
    }


def _fingerprint_path(cache_dir: Path) -> Path:
    return cache_dir / _FINGERPRINT_FILENAME


def _read_fingerprint(cache_dir: Path) -> dict[str, Any] | None:
    """Read the previously-persisted fingerprint, or ``None`` on miss."""
    path = _fingerprint_path(cache_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_fingerprint(cache_dir: Path, fp: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _fingerprint_path(cache_dir).write_text(
        json.dumps(fp, indent=2, sort_keys=True), encoding="utf-8",
    )


def _wipe_stale_caches(cache_dir: Path, on_event: EventListener | None) -> None:
    """Delete every ``*_anima*.{npz,safetensors}`` under ``cache_dir``.

    Called when the bucket fingerprint changes — the npz arrays were
    keyed on the previous bucket table and are guaranteed to mismatch
    the new training run. We leave non-cache files (the fingerprint
    sidecar itself, any user-dropped notes) alone.
    """
    if not cache_dir.is_dir():
        return
    removed = 0
    for path in cache_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if not (
            name.endswith("_anima.npz")
            or name.endswith("_anima_te.safetensors")
            or name.endswith("_anima_pe.safetensors")
        ):
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    if on_event is not None and removed > 0:
        on_event(
            TrainingEvent(
                type=EventType.log,
                payload={
                    "level": "info",
                    "source": "preprocess",
                    "message": (
                        f"anima_lora: bucket config changed — "
                        f"wiped {removed} stale cache file(s) "
                        f"under {cache_dir}; will re-run preprocess"
                    ),
                },
            )
        )


def _run_step(
    *,
    label: str,
    argv: list[str],
    cwd: Path,
    workspace: Path,
    on_event: EventListener | None,
    runner_factory: RunnerFactory,
) -> None:
    """Spawn one preprocess script, stream its output, raise on non-zero exit.

    Re-uses the LoraHub-wide :class:`SubprocessRunner` so events from
    preprocess look the same on the bus as training events (operators
    can see resize / cache progress in the live UI). The only twist:
    we call ``runner.wait()`` so :func:`ensure_cache` blocks until the
    step finishes before launching train.py.
    """
    # Block ``done`` events from this preprocess subprocess from
    # reaching the upstream listener: jobs_helpers treats *any* `done`
    # as the training process exiting, which closes the events sink
    # and tears down the GPU sampler. The next preprocess step's
    # `_safe_emit` then raises "JsonlEventSink used outside its
    # context". We still let log / progress events through so the UI
    # can show resize / cache progress.
    upstream: EventListener | None = on_event

    def filtered_listener(ev: TrainingEvent) -> None:
        if ev.type is EventType.done:
            return
        if upstream is not None:
            upstream(ev)

    listener: EventListener = filtered_listener if upstream is not None else (lambda _e: None)
    if upstream is not None:
        upstream(
            TrainingEvent(
                type=EventType.log,
                payload={
                    "level": "info",
                    "source": "preprocess",
                    "message": f"anima_lora preprocess: {label} starting",
                },
            )
        )
    runner = runner_factory(
        argv=argv,
        workspace=workspace,
        on_event=listener,
        parse_line=parse_line,
        cwd=cwd,
        thread_label=f"anima_lora_preprocess_{label}",
    )
    runner.start()
    result = runner.wait()
    if result.returncode != 0:
        msg = (
            f"anima_lora preprocess step {label!r} failed "
            f"(returncode={result.returncode}); see preceding log lines"
        )
        raise PreprocessError(msg)


def _resize_argv(repo: Path, source: Path, resized: Path) -> list[str]:
    return [
        str(repo / "preprocess" / "resize_images.py"),
        "--src",
        str(source.resolve()),
        "--dst",
        str(resized.resolve()),
        "--recursive",
    ]


def _cache_latents_argv(
    repo: Path, resized: Path, cache: Path, vae: Path
) -> list[str]:
    return [
        str(repo / "preprocess" / "cache_latents.py"),
        "--dir",
        str(resized.resolve()),
        "--cache_dir",
        str(cache.resolve()),
        "--vae",
        str(vae.resolve()),
        "--recursive",
    ]


def _cache_te_argv(
    repo: Path, source: Path, cache: Path, qwen3: Path, dit: Path | None
) -> list[str]:
    """``cache_text_embeddings.py`` reads .txt sidecars from the source dir.

    Note: source (raw images dir, has captions) — not the resized output
    — because upstream copies captions only when ``--no_copy_captions``
    is unset; we let upstream's TE step read them from the master copy
    so dataset edits don't require a resize pass.
    """
    argv = [
        str(repo / "preprocess" / "cache_text_embeddings.py"),
        "--dir",
        str(source.resolve()),
        "--cache_dir",
        str(cache.resolve()),
        "--qwen3",
        str(qwen3.resolve()),
        "--recursive",
    ]
    if dit is not None:
        argv += ["--dit", str(dit.resolve())]
    return argv


def ensure_cache(
    *,
    image_dir: Path,
    workspace: Path,
    base_model: BaseModelConfig,
    env: AnimaLoraEnv,
    on_event: EventListener | None = None,
    runner_factory: RunnerFactory | None = None,
    opts: Any | None = None,
) -> None:
    """Bring ``<workspace>/post_image_dataset/lora`` up to date.

    Detects whether every image in ``image_dir`` already has a matching
    ``{stem}_anima_te.safetensors`` cache file under
    ``<workspace>/post_image_dataset/lora`` AND whether the bucket-
    selection options match what the cache was originally built with.
    When everything is current, the function returns immediately.
    Otherwise it runs upstream's three preprocess scripts in sequence:

        1. ``preprocess/resize_images.py``     → resized PNGs
        2. ``preprocess/cache_latents.py``     → VAE latent caches
        3. ``preprocess/cache_text_embeddings.py`` → TE caches

    Each step streams events through ``on_event`` (so the UI shows
    progress) and the function blocks until the step finishes. Failures
    raise :class:`PreprocessError` so the caller can abort the launch
    cleanly instead of feeding train.py incomplete caches.

    ``opts`` is the active ``AnimaLoraOptions``. It's used to compute
    the bucket fingerprint — when ``enable_native_flatten`` /
    ``bucket_table`` / ``static_token_count`` differ from the previous
    run, the latent ``.npz`` keys won't match the new bucket
    assignment, so we wipe the cache and rebuild. Without this guard
    the user hits ``ValueError: latents_NxN not found`` mid-training.
    """
    image_dir = image_dir.resolve()
    workspace = workspace.resolve()

    images = _list_images(image_dir)
    if not images:
        msg = (
            f"anima_lora preprocess: no images found under {image_dir}; "
            "populate the directory with .jpg/.png/.webp files (and "
            "matching .txt captions) before launching"
        )
        raise PreprocessError(msg)

    cache_dir = workspace / "post_image_dataset" / "lora"
    resized_dir = workspace / "post_image_dataset" / "resized"

    # Bucket fingerprint check — when the bucket table changed since
    # the last run, every npz in the cache is keyed on stale shapes.
    # Wipe and rebuild rather than letting train.py crash mid-step on
    # a missing latents_HxW key.
    new_fp = _bucket_fingerprint(opts)
    old_fp = _read_fingerprint(cache_dir)
    fingerprint_mismatch = old_fp is not None and old_fp != new_fp

    if fingerprint_mismatch:
        if on_event is not None:
            on_event(
                TrainingEvent(
                    type=EventType.log,
                    payload={
                        "level": "warning",
                        "source": "preprocess",
                        "message": (
                            "anima_lora: bucket config changed since "
                            f"last cache build (was {old_fp}, now "
                            f"{new_fp}); wiping latent / TE caches "
                            "and re-running preprocess to avoid "
                            "'latents_NxN not found' at training start"
                        ),
                    },
                )
            )
        _wipe_stale_caches(cache_dir, on_event)
        # Force the missing-images path below to see every image as missing
        # so the preprocess steps run.
        missing = list(images)
    else:
        missing = _missing_caches(images, cache_dir)

    if not missing:
        if on_event is not None:
            on_event(
                TrainingEvent(
                    type=EventType.log,
                    payload={
                        "level": "info",
                        "source": "preprocess",
                        "message": (
                            f"anima_lora cache hit ({len(images)} images "
                            f"already cached under {cache_dir})"
                        ),
                    },
                )
            )
        # Make sure the fingerprint is on disk even when we skipped — old
        # caches built before this guard existed have no fingerprint
        # sidecar, so we backfill on the first launch that takes the hit
        # path. Future runs will compare against this snapshot.
        if old_fp is None:
            _write_fingerprint(cache_dir, new_fp)
        return

    if on_event is not None:
        on_event(
            TrainingEvent(
                type=EventType.log,
                payload={
                    "level": "info",
                    "source": "preprocess",
                    "message": (
                        f"anima_lora preprocess: {len(missing)}/"
                        f"{len(images)} images need cache; populating "
                        f"{cache_dir}"
                    ),
                },
            )
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    resized_dir.mkdir(parents=True, exist_ok=True)

    factory = runner_factory or _default_runner_factory
    python = str(env.python_executable)
    repo = env.repo_path

    # Step 1: resize raw images → resized_dir.
    _run_step(
        label="resize",
        argv=[python, *_resize_argv(repo, image_dir, resized_dir)],
        cwd=repo,
        workspace=workspace,
        on_event=on_event,
        runner_factory=factory,
    )

    # Step 2: VAE latent caches against resized images. Requires the
    # ``base_model.arch_paths.ae`` path to be set, otherwise we cannot
    # locate the QwenImage VAE.
    ae_path = base_model.arch_paths.ae
    if ae_path is None:
        msg = (
            "anima_lora preprocess: base_model.archPaths.ae must point at "
            "the QwenImage VAE checkpoint to run cache_latents.py"
        )
        raise PreprocessError(msg)
    _run_step(
        label="cache_latents",
        argv=[python, *_cache_latents_argv(repo, resized_dir, cache_dir, ae_path)],
        cwd=repo,
        workspace=workspace,
        on_event=on_event,
        runner_factory=factory,
    )

    # Step 3: TE caches. Reads .txt sidecars from the *raw* image_dir
    # (upstream's resize pass copies captions to resized_dir but
    # cache_text_embeddings.py is happy to read from either; using the
    # raw dir means a caption edit is picked up immediately without a
    # resize re-run).
    qwen3_path = base_model.arch_paths.qwen3
    if qwen3_path is None:
        msg = (
            "anima_lora preprocess: base_model.archPaths.qwen3 must point "
            "at the Qwen3 text encoder to run cache_text_embeddings.py"
        )
        raise PreprocessError(msg)
    _run_step(
        label="cache_text_embeddings",
        argv=[
            python,
            *_cache_te_argv(
                repo, image_dir, cache_dir, qwen3_path, dit=base_model.checkpoint
            ),
        ],
        cwd=repo,
        workspace=workspace,
        on_event=on_event,
        runner_factory=factory,
    )

    # Persist the bucket fingerprint so the next launch can detect
    # config changes and rebuild appropriately.
    _write_fingerprint(cache_dir, new_fp)

    if on_event is not None:
        on_event(
            TrainingEvent(
                type=EventType.log,
                payload={
                    "level": "info",
                    "source": "preprocess",
                    "message": (
                        f"anima_lora preprocess complete; cache populated "
                        f"at {cache_dir}"
                    ),
                },
            )
        )
