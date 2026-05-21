"""anima_lora preview backend — subprocess-based image generation.

Plugs the vendored ``external/anima_lora/inference.py`` into LoraHub's
inference registry so preview rendering for arch=anima goes through
upstream's purpose-built pipeline (Spectrum acceleration, DCW, etc.)
instead of LoraHub's in-process Anima backend.

Why a separate backend instead of replacing ``anima.py``:

* The in-process ``AnimaInferenceBackend`` binds against
  ``library.anima`` from the user's *training* venv. anima_lora needs
  torch 2.11 nightly + CUDA 13 — not something our main venv usually
  has. Going through subprocess matches how training already works.
* Registry priority: this backend registers AFTER ``anima.py`` and
  with a higher precedence (factory order — earlier in the list = ran
  first). When the vendored copy + ``LORAHUB_ANIMA_LORA_PYTHON`` are
  available, this wins. Otherwise we fall through to the in-process
  Anima backend (or the diffusers / stub chain after that).

Subprocess contract:

    <python> external/anima_lora/inference.py \
        --dit <DiT path> \
        --vae <VAE path> \
        --text_encoder <Qwen3 path> \
        --lora_weight <lora.safetensors> \
        --prompt "<text>" \
        --image_size <H> <W> \
        --infer_steps N \
        --guidance_scale C \
        --save_path <out.png> \
        --seed <S>

Paths come from ``cfg.base_model.arch_paths`` (DiT / VAE / Qwen3) which
the recipe already populates for training.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from lorahub.core.backends.anima_lora import bootstrap as _al_bootstrap
from lorahub.core.backends.errors import BootstrapError
from lorahub.core.inference import PromptSpec
from lorahub.core.inference.registry import register_backend

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SEC = 600  # 10 minutes — enough for a single 1024x1024 sample.


class AnimaLoraInferenceBackend:
    """Spawns ``external/anima_lora/inference.py`` for each preview render.

    Each ``render`` call is a one-shot subprocess; we don't keep a long
    running worker for previews. anima_lora's startup cost is sizable
    (DiT + VAE + Qwen3 load), but previews are infrequent (one per
    saved checkpoint) so a fresh process per call keeps memory bounded
    and avoids lifecycle complexity.
    """

    name = "anima_lora"

    def __init__(
        self,
        *,
        env: _al_bootstrap.AnimaLoraEnv,
        dit_path: Path,
        vae_path: Path,
        text_encoder_path: Path,
        timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self._env = env
        self._dit = dit_path
        self._vae = vae_path
        self._text_encoder = text_encoder_path
        self._timeout = timeout_sec

    def is_available(self, *, arch: str) -> bool:
        """Only ``anima`` arch — same gate as the training backend."""
        if arch != "anima":
            return False
        # All three model artefacts must exist on disk; missing any
        # would just give us an opaque inference.py traceback.
        return (
            self._dit.exists()
            and self._vae.exists()
            and self._text_encoder.exists()
        )

    def render(
        self,
        *,
        lora_path: Path,
        spec: PromptSpec,
        out_path: Path,
        default_steps: int,
        default_cfg: float,
    ) -> None:
        """Generate one preview by spawning inference.py.

        Honours ``spec.steps`` / ``spec.cfg`` / ``spec.seed`` overrides
        and falls back to the worker's defaults when those are ``None``.
        Raises ``RuntimeError`` if the subprocess fails — the caller
        (PreviewWorker) catches it and emits a log event.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        argv = self._build_argv(lora_path, spec, out_path, default_steps, default_cfg)
        log.info(
            "anima_lora preview render: lora=%s prompt=%r → %s",
            lora_path.name,
            spec.prompt[:60],
            out_path,
        )
        try:
            proc = subprocess.run(
                argv,
                cwd=self._env.repo_path,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            msg = (
                f"anima_lora preview timed out after {self._timeout}s "
                f"(prompt {spec.prompt[:40]!r})"
            )
            raise RuntimeError(msg) from exc
        if proc.returncode != 0:
            tail = "\n".join(
                (proc.stderr or proc.stdout or "").splitlines()[-15:]
            )
            msg = (
                f"anima_lora inference.py exited {proc.returncode} for "
                f"prompt {spec.prompt[:40]!r}; tail:\n{tail}"
            )
            raise RuntimeError(msg)
        if not out_path.is_file():
            msg = (
                f"anima_lora inference.py returned 0 but {out_path} was "
                f"not written"
            )
            raise RuntimeError(msg)

    def _build_argv(
        self,
        lora_path: Path,
        spec: PromptSpec,
        out_path: Path,
        default_steps: int,
        default_cfg: float,
    ) -> list[str]:
        steps = spec.steps if spec.steps is not None else default_steps
        cfg = spec.cfg if spec.cfg is not None else default_cfg
        argv: list[str] = [
            str(self._env.python_executable),
            str(self._env.script("inference.py")),
            "--dit", str(self._dit),
            "--vae", str(self._vae),
            "--text_encoder", str(self._text_encoder),
            "--lora_weight", str(lora_path),
            "--prompt", spec.prompt,
            "--image_size", str(spec.height), str(spec.width),
            "--infer_steps", str(steps),
            "--guidance_scale", repr(float(cfg)),
            "--save_path", str(out_path),
        ]
        if spec.seed is not None:
            argv += ["--seed", str(spec.seed)]
        if spec.negative:
            argv += ["--negative_prompt", spec.negative]
        return argv


def _anima_lora_factory(
    *, arch: str, recipe: Any, workspace: Any
) -> AnimaLoraInferenceBackend | None:
    """Registry factory for the anima_lora backend.

    Skips when:
      * arch != "anima"
      * the vendored copy (or env override) doesn't resolve cleanly
      * the recipe lacks the DiT / VAE / Qwen3 paths upstream needs

    Returning ``None`` lets the registry fall through to the next
    backend (the in-process anima.py one) — preview rendering
    degrades gracefully instead of failing hard.
    """
    if arch != "anima":
        return None
    if recipe is None:
        return None

    # Resolve the vendored copy + python interpreter. Recipe-level
    # overrides live on backend.python_executable / backend.repo_path
    # the same way the training backend reads them.
    backend_cfg = getattr(recipe, "backend", None)
    config_python = getattr(backend_cfg, "python_executable", None) if backend_cfg else None
    config_path = getattr(backend_cfg, "repo_path", None) if backend_cfg else None
    try:
        env = _al_bootstrap.resolve(
            config_path=config_path,
            config_python=config_python,
        )
    except BootstrapError:
        log.info(
            "anima_lora preview backend skipped: vendored copy not resolvable"
        )
        return None

    # Pull DiT / VAE / Qwen3 paths from the recipe's BaseModelConfig.
    bm = getattr(recipe, "base_model", None)
    if bm is None:
        return None
    paths = getattr(bm, "arch_paths", None)
    if paths is None:
        return None

    # anima_lora's `--dit` is the same file LoraHub treats as the main
    # checkpoint. `--vae` maps to ``ae`` (FLUX-style aliasing — anima
    # uses the QwenImage VAE under the same field name as flux).
    dit = bm.checkpoint
    vae = paths.ae
    text_encoder = paths.qwen3
    if dit is None or vae is None or text_encoder is None:
        log.info(
            "anima_lora preview backend skipped: missing dit/vae/qwen3 paths"
        )
        return None

    return AnimaLoraInferenceBackend(
        env=env,
        dit_path=dit,
        vae_path=vae,
        text_encoder_path=text_encoder,
    )


# Register at import time. Insert order matters — registry tries
# factories in order, first non-None wins. We use `prepend=True` so
# this subprocess-based backend takes priority over the in-process
# anima.py one when both are usable; if our `is_available()` returns
# False (vendored copy missing or paths absent), the registry falls
# through to anima.py and then to the diffusers / stub chain.
register_backend("anima_lora", _anima_lora_factory, prepend=True)


__all__ = ["AnimaLoraInferenceBackend"]
