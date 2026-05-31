"""Anima inference backend for the lorahub preview worker.

Implements the `AnimaInference` Protocol used by `PreviewWorker`, but
delegates the actual diffusion-and-VAE-decode work to sd-scripts'
`anima_minimal_inference.py` via subprocess. The reasons for not
importing the inference path directly:

* `anima_minimal_inference.py` lives in the sd-scripts venv (separate
  Python env from lorahub) — different torch / xformers / diffusers
  pins which we don't want to entangle.
* Each preview render sits between training steps; running it in a
  short-lived subprocess gives clean CUDA-context teardown so we
  never leak VRAM between previews.
* Subprocess lets us hard-cap with `subprocess.Popen.wait(timeout=...)`
  so a hung diffusion call can't stall the worker thread.

The one piece this module does in-process is the LoRA-format bridge:
diffusion-pipe saves PEFT-style LoRA (`diffusion_model.*.lora_{A,B}.weight`,
no alpha tensor — read from `adapter_config.json`) but
`anima_minimal_inference.py` expects sd-scripts/ComfyUI flat keys
(`lora_unet_*.{lora_down,lora_up,alpha}`). We rewrite the safetensors
file once per checkpoint into a sibling `lorahub_converted.safetensors`
and feed THAT to the inference script.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lorahub.core.inference import PromptSpec

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# LoRA format conversion: dp peft -> sd-scripts/ComfyUI
# --------------------------------------------------------------------------- #


_DM_PREFIX_RE = re.compile(r"^(?:diffusion_model|transformer)\.")
_LORA_KEY_RE = re.compile(r"^(.*)\.lora_(A|B)\.weight$")


def convert_dp_lora_to_kohya(
    dp_lora_path: Path,
    out_path: Path,
    *,
    rank_fallback: int = 16,
    alpha_fallback: float = 16.0,
) -> Path:
    """Convert a diffusion-pipe PEFT LoRA into kohya/sd-scripts format.

    Inputs:
      dp_lora_path: `runs/.../{step|epoch}N/adapter_model.safetensors`
                    written by dp Saver (peft state dict + a
                    `diffusion_model.` prefix + only `format=pt` metadata).
      out_path:     where to write the converted file.

    Output is a safetensors with keys shaped like

        lora_unet_blocks_0_self_attn_q_proj.lora_down.weight   # (rank, in)
        lora_unet_blocks_0_self_attn_q_proj.lora_up.weight     # (out, rank)
        lora_unet_blocks_0_self_attn_q_proj.alpha              # scalar tensor

    which `anima_minimal_inference.load_dit_model` then merges into the
    base in a single pass via `load_safetensors_with_lora_and_fp8`.

    LLM adapter modules — `diffusion_model.llm_adapter.blocks.X...` —
    map to the same `lora_unet_*` family because in Anima the adapter
    is part of the DiT graph (`anima_models.Anima.llm_adapter`).
    `o_proj` (LLM adapter) and `output_proj` (DiT Block) are preserved
    verbatim; both names exist as concrete submodules upstream.
    """
    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pull LoRA hyperparameters from the sibling adapter_config.json,
    # which is what peft.save_pretrained writes alongside the weights.
    rank = rank_fallback
    alpha = alpha_fallback
    cfg_path = dp_lora_path.parent / "adapter_config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            rank = int(cfg.get("r", rank))
            alpha = float(cfg.get("lora_alpha", alpha))
        except (json.JSONDecodeError, ValueError, TypeError):
            log.warning("could not parse %s — using fallback rank/alpha", cfg_path)

    # Group the dp keys by base module path; collect lora_A / lora_B pairs.
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(dp_lora_path), framework="pt") as f:
        for k in f.keys():
            stripped = _DM_PREFIX_RE.sub("", k)
            m = _LORA_KEY_RE.match(stripped)
            if m is None:
                continue  # ignore stray keys (no embedding/bias from dp)
            base, side = m.group(1), m.group(2)
            pairs.setdefault(base, {})[side] = f.get_tensor(k)

    out_state: dict[str, torch.Tensor] = {}
    alpha_t = torch.tensor(alpha)

    for base, ab in pairs.items():
        a = ab.get("A")
        b = ab.get("B")
        if a is None or b is None:
            log.warning("dp lora %s: missing A or B for %s — dropped", dp_lora_path, base)
            continue
        # Rewrite the dotted base path into the flat kohya prefix.
        # `blocks.0.self_attn.q_proj` -> `lora_unet_blocks_0_self_attn_q_proj`
        flat = "lora_unet_" + base.replace(".", "_")
        # sd-scripts naming: lora_down = A, lora_up = B
        out_state[f"{flat}.lora_down.weight"] = a.contiguous()
        out_state[f"{flat}.lora_up.weight"] = b.contiguous()
        out_state[f"{flat}.alpha"] = alpha_t.clone()

    # `format=pt` is required for safetensors to round-trip through
    # `safe_open(framework="pt")` cleanly on the consumer side.
    save_file(
        out_state,
        str(out_path),
        metadata={"format": "pt", "lorahub_source": "diffusion-pipe-peft"},
    )
    log.info(
        "converted dp lora %s -> kohya %s (%d modules, rank=%d alpha=%.1f)",
        dp_lora_path.name,
        out_path.name,
        len(pairs),
        rank,
        alpha,
    )
    return out_path


# --------------------------------------------------------------------------- #
# Inference backend (subprocess wrapper around anima_minimal_inference.py)
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AnimaInferenceConfig:
    """Where to find the upstream inference script and its model files.

    All fields are absolute paths so subprocess invocation is
    reproducible across cwd changes.
    """

    sd_scripts_python: Path  # python from the sd-scripts venv
    sd_scripts_repo: Path    # checkout root with anima_minimal_inference.py
    transformer_path: Path   # base Anima safetensors (~4 GB)
    vae_path: Path           # qwen_image_vae.safetensors
    text_encoder_path: Path  # qwen_3_06b_base.safetensors
    # Fixed inference behaviour:
    timeout_per_image_s: float = 180.0
    min_free_vram_mib: int = 6500   # skip when free VRAM is below this
    user_strength: float = 1.0      # external multiplier on top of dp's alpha=rank scale


class AnimaInferenceBackend:
    """Bridge implementing the `AnimaInference` Protocol from the
    preview worker. One instance per job; safe to call `render`
    sequentially from a single thread (the worker guarantees this)."""

    name = "anima"

    def __init__(self, config: AnimaInferenceConfig) -> None:
        self.config = config
        self._sanity_checked = False

    def is_available(self, *, arch: str) -> bool:
        """Anima only runs against ``arch == "anima"``; all paths must exist.

        Mirrors the ``_sanity_check`` body but returns a bool instead of
        raising — the registry uses this gate to fall through to the
        next backend when prerequisites are absent.
        """
        if arch != "anima":
            return False
        cfg = self.config
        for path in (
            cfg.sd_scripts_python,
            cfg.sd_scripts_repo / "anima_minimal_inference.py",
            cfg.transformer_path,
            cfg.vae_path,
            cfg.text_encoder_path,
        ):
            if not Path(path).exists():
                return False
        return True

    # PreviewWorker calls this for each (lora, prompt) pair.
    def render(
        self,
        *,
        lora_path: Path,
        spec: PromptSpec,
        out_path: Path,
        default_steps: int,
        default_cfg: float,
    ) -> None:
        cfg = self.config

        if not self._sanity_checked:
            self._sanity_check()
            self._sanity_checked = True

        if not _has_enough_vram(cfg.min_free_vram_mib):
            log.warning(
                "[anima-inference] free VRAM below %d MiB, skipping %s prompt %d",
                cfg.min_free_vram_mib,
                lora_path.parent.name,
                spec.index,
            )
            raise InferenceSkipped("insufficient free VRAM for preview render")

        # Convert the dp LoRA into kohya format if not already done.
        # The conversion is keyed by mtime so a re-saved checkpoint
        # at the same step regenerates the converted file.
        kohya_lora = lora_path.parent / "lorahub_converted.safetensors"
        if not kohya_lora.exists() or kohya_lora.stat().st_mtime < lora_path.stat().st_mtime:
            convert_dp_lora_to_kohya(lora_path, kohya_lora)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Render via anima_minimal_inference.py. The script prints to a
        # save_path directory then names the file deterministically; we
        # render into a temporary scratch dir and move the produced PNG
        # into out_path so naming stays under our control.
        scratch_dir = out_path.parent / f"_scratch_{out_path.stem}"
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)
        scratch_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(cfg.sd_scripts_python),
            "anima_minimal_inference.py",
            "--dit", str(cfg.transformer_path),
            "--vae", str(cfg.vae_path),
            "--text_encoder1", str(cfg.text_encoder_path),
            "--prompt", spec.prompt,
            "--negative_prompt", spec.negative or "",
            "--image_size", f"{spec.height}x{spec.width}",
            "--steps", str(spec.steps or default_steps),
            "--guidance_scale", str(spec.cfg or default_cfg),
            "--seed", str(spec.seed if spec.seed is not None else 42),
            "--save_path", str(scratch_dir),
            "--output_type", "image",
            "--lora_weights", str(kohya_lora),
            "--lora_multiplier", str(cfg.user_strength),
        ]
        if spec.sampler is not None:
            cmd.extend(["--sampler", spec.sampler])
        if spec.flow_shift is not None:
            cmd.extend(["--flow_shift", str(spec.flow_shift)])
        log.info(
            "[anima-inference] render %s prompt %d -> %s",
            lora_path.parent.name,
            spec.index,
            out_path.name,
        )
        started = time.time()
        env = os.environ.copy()
        # Disable tqdm spam — the subprocess output is captured anyway.
        env.setdefault("TQDM_DISABLE", "1")
        try:
            res = subprocess.run(
                cmd,
                cwd=str(cfg.sd_scripts_repo),
                env=env,
                capture_output=True,
                text=True,
                timeout=cfg.timeout_per_image_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            shutil.rmtree(scratch_dir, ignore_errors=True)
            raise InferenceFailed(
                f"anima_minimal_inference.py timed out after "
                f"{cfg.timeout_per_image_s:.0f}s"
            ) from None

        if res.returncode != 0:
            tail = (res.stderr or res.stdout or "").splitlines()[-12:]
            shutil.rmtree(scratch_dir, ignore_errors=True)
            raise InferenceFailed(
                f"anima_minimal_inference.py exited {res.returncode}: "
                + "\n".join(tail)
            )

        # Pick the only PNG the script wrote and atomically rename it
        # to our canonical out_path.
        produced = sorted(scratch_dir.glob("*.png"))
        if not produced:
            shutil.rmtree(scratch_dir, ignore_errors=True)
            raise InferenceFailed(
                "anima_minimal_inference.py succeeded but produced no PNG"
            )
        produced[0].replace(out_path)
        shutil.rmtree(scratch_dir, ignore_errors=True)

        log.info(
            "[anima-inference] rendered %s in %.1fs",
            out_path.name,
            time.time() - started,
        )

    def _sanity_check(self) -> None:
        cfg = self.config
        missing: list[str] = []
        if not cfg.sd_scripts_python.exists():
            missing.append(f"sd_scripts_python={cfg.sd_scripts_python}")
        script = cfg.sd_scripts_repo / "anima_minimal_inference.py"
        if not script.is_file():
            missing.append(f"inference script={script}")
        if not cfg.transformer_path.is_file():
            missing.append(f"transformer={cfg.transformer_path}")
        if not cfg.vae_path.is_file():
            missing.append(f"vae={cfg.vae_path}")
        if not cfg.text_encoder_path.is_file():
            missing.append(f"text_encoder={cfg.text_encoder_path}")
        if missing:
            raise InferenceFailed(
                "anima inference prerequisites missing:\n  - "
                + "\n  - ".join(missing)
            )


class InferenceFailed(RuntimeError):
    """Raised by AnimaInferenceBackend.render when the subprocess errors
    out (timeout, non-zero exit, missing PNG, missing prerequisite)."""


class InferenceSkipped(RuntimeError):
    """Raised when the render was deliberately skipped (VRAM pressure,
    cancel signal). The worker treats this as benign — no error event
    emitted, just move on to the next prompt or checkpoint."""


# --------------------------------------------------------------------------- #
# VRAM probe (pynvml — already an indirect dep through nvidia-smi paths)
# --------------------------------------------------------------------------- #


def _has_enough_vram(min_free_mib: int) -> bool:
    """Check that GPU 0 has at least `min_free_mib` MiB free.

    Falls back to 'assume yes' when pynvml isn't available — better to
    let the subprocess fail with OOM than block previews on a
    speculative gate.
    """
    try:
        import pynvml  # type: ignore[import-not-found] # noqa: PLC0415
    except ImportError:
        return True
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mib = int(info.free) // (1024 * 1024)
        pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        return True
    log.debug("[anima-inference] free VRAM = %d MiB", free_mib)
    return free_mib >= min_free_mib


# --------------------------------------------------------------------------- #
# Convenience constructor from a TrainingConfig + workspace
# --------------------------------------------------------------------------- #


def build_backend_from_config(
    *, config: Any, workspace: Path
) -> AnimaInferenceBackend | None:
    """Resolve sd-scripts paths from settings and the dp model paths
    block in the config; return None if any prerequisite is missing.

    `config` is a TrainingConfig (typed loose to keep this module free
    of cyclic imports).
    """
    # sd-scripts python: prefer the configured backend.python_executable,
    # fall back to env var, fall back to the shell `python`.
    sd_scripts_python_env = os.environ.get("LORAHUB_SD_SCRIPTS_PYTHON")
    if config.backend.python_executable is not None:
        py = Path(str(config.backend.python_executable))
    elif sd_scripts_python_env:
        py = Path(sd_scripts_python_env)
    else:
        py = Path("python")

    sd_scripts_repo_env = os.environ.get("LORAHUB_KOHYA_SD_SCRIPTS")
    if config.backend.repo_path is not None:
        repo = Path(str(config.backend.repo_path))
    elif sd_scripts_repo_env:
        repo = Path(sd_scripts_repo_env)
    else:
        return None

    # Anima base paths come straight from the dp model_paths bag (these
    # are the raw upstream strings — transformer_path / vae_path / llm_path).
    if config.backend.diffusion_pipe is None:
        return None
    mp = config.backend.diffusion_pipe.model_paths or {}
    transformer = mp.get("transformer_path") or config.base_model.checkpoint
    vae = mp.get("vae_path")
    te = mp.get("llm_path")
    if not transformer or not vae or not te:
        return None

    cfg = AnimaInferenceConfig(
        sd_scripts_python=py,
        sd_scripts_repo=repo,
        transformer_path=Path(str(transformer)),
        vae_path=Path(str(vae)),
        text_encoder_path=Path(str(te)),
        timeout_per_image_s=180.0,
        min_free_vram_mib=6500,
    )
    return AnimaInferenceBackend(cfg)


# --------------------------------------------------------------------------- #
# Registry hook — picked up by lorahub.core.inference at import time.
# --------------------------------------------------------------------------- #


def _anima_factory(
    *, arch: str, config: Any, workspace: Any
) -> AnimaInferenceBackend | None:
    """Registry factory for the Anima backend.

    Only returns a backend for ``arch == "anima"`` — the registry skips
    the rest and falls through to the next entry (typically diffusers).
    Same prerequisite gate as ``build_backend_from_config`` so missing
    sd-scripts paths don't surface a half-built backend.
    """
    if arch != "anima":
        return None
    if config is None or workspace is None:
        return None
    backend = build_backend_from_config(config=config, workspace=Path(workspace))
    if backend is None:
        return None
    if not backend.is_available(arch=arch):
        return None
    return backend


# Late import to dodge the cycle: registry imports PromptSpec from
# ``lorahub.core.inference``, which is what this module imports as well.
# By the time this module reaches its bottom, both are defined.
from lorahub.core.inference.registry import register_backend  # noqa: E402

register_backend("anima", _anima_factory)
