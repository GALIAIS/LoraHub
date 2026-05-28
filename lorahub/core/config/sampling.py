"""Sampling preview configs (PromptSpec, SamplingOutputs, SamplingConfig)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ._shared import _CAMEL_CONFIG


class PromptSpec(BaseModel):
    """One sampling prompt persisted in yaml.

    The trainer-side prompts file (kohya `--sample_prompts`) is a plain
    text format with `--w` / `--h` / `--d` / `--s` / `--l` / `--n`
    flags. We let users author prompts directly in the config instead
    of pointing at a sibling .txt — the launcher materialises the
    kohya-style file under workspace/prompts.txt at job-start time so
    no upstream tooling has to change.

    Per-row seed semantics mirror SamplingConfig.seed: -1 means
    "random per run", drawn at runtime; any other int is honoured
    verbatim. Width/height fall back to the ambient SamplingConfig
    resolution.
    """

    model_config = _CAMEL_CONFIG

    prompt: str
    negative: str | None = None
    cfg: float | None = Field(default=None, gt=0)
    steps: int | None = Field(default=None, ge=1)
    seed: int | None = None
    width: int | None = Field(default=None, ge=64)
    height: int | None = Field(default=None, ge=64)


class SamplingOutputs(BaseModel):
    """Toggles for the four preview-output features.

    Each is independent; users mix and match. Defaults reflect "useful
    but cheap" — grid stitching + PNG metadata are on; base-compare
    and cross-ckpt animation are off because both spend extra GPU /
    disk for richer artefacts that not every run needs.
    """

    model_config = _CAMEL_CONFIG

    grid_stitching: bool = True
    base_compare: bool = False
    cross_ckpt_animation: bool = False
    png_metadata: bool = True


class SamplingConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    enabled: bool = True
    every_n_epochs: int = Field(1, ge=1)
    # Step-level sampling cadence (kohya: --sample_every_n_steps).
    every_n_steps: int | None = Field(default=None, ge=1)
    # Generate a baseline before training starts (kohya: --sample_at_first).
    at_first: bool = False
    # Legacy: external prompts.txt path. New configs should populate
    # ``prompts`` instead — the launcher materialises a prompts.txt
    # under workspace/ from that list at job-start. We keep the
    # field for back-compat: when set and ``prompts`` is empty the
    # legacy path is used unchanged.
    prompts_file: Path | None = None
    prompts: list[PromptSpec] = Field(default_factory=list)
    resolution: list[int] = Field(default_factory=lambda: [1024, 1024])
    # ComfyUI-style sentinel: -1 means "draw a fresh random integer at
    # job-start". Anything else is honoured verbatim. The launcher logs
    # the resolved seed so reproducing a run is still possible.
    seed: int = -1
    # Optional trigger word substituted into every ``prompts[].prompt``
    # in place of the literal ``${TRIGGER}`` placeholder at job-start.
    # Empty / unset → the launcher tries to recover one by reading the
    # first comma-separated token of the dataset's .txt captions and
    # picking the most common value. Failing both paths the placeholder
    # (and any trailing ", ") is stripped, leaving a generic prompt.
    # Lets the same default config template adapt to character / style
    # LoRAs without hand-editing every prompt row.
    trigger_word: str | None = None
    outputs: SamplingOutputs = Field(default_factory=SamplingOutputs)
    # NOTE: ``sampling.attention`` was removed — sample-stage attention
    # backend selection was schema-only (no compiler ever wired it
    # through to the trainer). Existing YAML files carrying it load
    # cleanly because pydantic's default ``extra="ignore"`` policy
    # silently drops the unknown key. Sample images now always reuse
    # ``cfg.attention.training``.

    # diffusion-pipe doesn't generate preview images on its own. When
    # `enable_live_inference` is on, the lorahub job runner starts a
    # background watcher that polls the workspace `output/step*` dirs
    # and runs an in-process Anima inference for every new checkpoint
    # using the prompt list at `prompts_file`. The PNGs land under
    # `workspace/samples/` and a `sample_ready` event is emitted so
    # the analysis-tab gallery picks them up live.
    #
    # Off by default — turning it on adds GPU pressure during the
    # narrow window between checkpoints; only useful with the dp
    # backend (kohya already produces previews via --sample_prompts).
    enable_live_inference: bool = False
    inference_steps: int = Field(24, ge=1)
    inference_cfg: float = Field(5.0, gt=0)

    # Side-band SVD over each saved LoRA checkpoint. Cheap (< 1 s on
    # a typical adapter, runs on a daemon thread on the API host) and
    # produces a useful "is the adapter actually learning anything"
    # signal: effective_rank / top1_energy / fro_norm trends.
    # Off-switch is here so air-gapped users with adapters that have
    # several hundred LoRA pairs can opt out.
    spectrum_analysis: bool = True
