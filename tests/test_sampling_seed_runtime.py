"""Lock down the runtime seed semantics for sampling/preview.

The bug we're guarding against: ``cfg.sampling.prompts[*].seed = -1``
used to be resolved into a fixed random integer at job-start, which was
then frozen into ``prompts.txt`` as ``--d <N>`` for every preview tick.
Result — every epoch's preview render reset ``torch.manual_seed`` to the
same value, and the rendered images looked pixel-identical even as the
LoRA evolved.

The fix collapses ``-1`` to ``None`` so the prompt-file materialiser
skips ``--d`` for that row, which makes ``_sample_image_inference``
fall through to ambient RNG and produce a fresh sample each epoch.
Concrete seeds (``42``, ``1234``) still pass through verbatim for users
who want a reproducible preview.

These tests cover the lifecycle hook + the prompt-file materialiser
together because that's the contract the trainer ultimately consumes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lorahub.api.jobs_helpers.lifecycle import (
    _materialise_prompts_file,
    _resolve_runtime_seeds,
)
from lorahub.core.config.schema import TrainingConfig


def _kohya_cfg(workspace: Path, *, prompts: list[dict] | None = None,
               sampling_seed: int = -1) -> TrainingConfig:
    ckpt = workspace / "sdxl.safetensors"
    ckpt.write_bytes(b"")
    data = workspace / "data"
    data.mkdir(exist_ok=True)
    (data / "stub.png").write_bytes(b"")
    payload = {
        "base_model": {"arch": "sdxl", "checkpoint": str(ckpt)},
        "dataset": {"source": str(data)},
        "schedule": {"epochs": 1, "batch_size": 1, "grad_accum": 1},
        "sampling": {
            "enabled": True,
            "seed": sampling_seed,
            "prompts": prompts or [],
        },
        "output": {"name": "lora_output"},
        "backend": {"type": "kohya"},
    }
    return TrainingConfig.model_validate(payload)


# --------------------------------------------------------------------------- #
# _resolve_runtime_seeds
# --------------------------------------------------------------------------- #


def test_top_level_seed_minus_one_still_drawn(tmp_path: Path) -> None:
    """``sampling.seed`` is the training seed; ``-1`` must still be
    resolved to a concrete integer so the snapshot is reproducible."""
    cfg = _kohya_cfg(tmp_path, sampling_seed=-1)
    _resolve_runtime_seeds(cfg)
    assert isinstance(cfg.sampling.seed, int)
    assert cfg.sampling.seed != -1
    assert cfg.sampling.seed >= 0


def test_top_level_seed_explicit_value_preserved(tmp_path: Path) -> None:
    cfg = _kohya_cfg(tmp_path, sampling_seed=12345)
    _resolve_runtime_seeds(cfg)
    assert cfg.sampling.seed == 12345


def test_prompt_seed_minus_one_collapses_to_none(tmp_path: Path) -> None:
    """The per-row ``-1`` sentinel must NOT be frozen — it should be
    dropped to ``None`` so the prompt-file materialiser omits ``--d``
    and the trainer falls through to ambient RNG."""
    cfg = _kohya_cfg(
        tmp_path,
        prompts=[{"prompt": "a portrait", "seed": -1}],
    )
    _resolve_runtime_seeds(cfg)
    assert cfg.sampling.prompts[0].seed is None


def test_prompt_seed_concrete_value_preserved(tmp_path: Path) -> None:
    cfg = _kohya_cfg(
        tmp_path,
        prompts=[{"prompt": "a portrait", "seed": 42}],
    )
    _resolve_runtime_seeds(cfg)
    assert cfg.sampling.prompts[0].seed == 42


def test_prompt_seed_unset_stays_unset(tmp_path: Path) -> None:
    cfg = _kohya_cfg(
        tmp_path,
        prompts=[{"prompt": "a portrait"}],
    )
    _resolve_runtime_seeds(cfg)
    assert cfg.sampling.prompts[0].seed is None


# --------------------------------------------------------------------------- #
# _materialise_prompts_file
# --------------------------------------------------------------------------- #


def test_materialised_prompts_omit_d_flag_when_seed_is_none(
    tmp_path: Path,
) -> None:
    """A ``None`` per-row seed must not appear as ``--d`` in prompts.txt;
    that's what lets the trainer randomise per epoch."""
    cfg = _kohya_cfg(
        tmp_path,
        prompts=[{"prompt": "a portrait", "seed": -1}],
    )
    _resolve_runtime_seeds(cfg)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _materialise_prompts_file(cfg, workspace)

    body = (workspace / "prompts.txt").read_text(encoding="utf-8")
    assert "a portrait" in body
    assert "--d " not in body  # the trailing space guards against substrings


def test_materialised_prompts_emit_d_flag_when_seed_is_concrete(
    tmp_path: Path,
) -> None:
    cfg = _kohya_cfg(
        tmp_path,
        prompts=[{"prompt": "a portrait", "seed": 42}],
    )
    _resolve_runtime_seeds(cfg)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _materialise_prompts_file(cfg, workspace)

    body = (workspace / "prompts.txt").read_text(encoding="utf-8")
    assert "--d 42" in body


def test_materialised_multiline_prompt_remains_one_prompt(
    tmp_path: Path,
) -> None:
    cfg = _kohya_cfg(
        tmp_path,
        prompts=[{"prompt": "placeholder"}],
    )
    # Runtime trigger substitution happens after schema validation, so guard
    # the final file boundary independently from the schema normalizer.
    cfg.sampling.prompts[0].prompt = "a detailed portrait\nwith dramatic lighting"
    cfg.sampling.prompts[0].negative = "bad anatomy\nblurry"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _materialise_prompts_file(cfg, workspace)

    lines = (workspace / "prompts.txt").read_text(encoding="utf-8").splitlines()
    assert lines == [
        "a detailed portrait with dramatic lighting --n bad anatomy blurry"
    ]


def test_prompt_schema_folds_multiline_text(tmp_path: Path) -> None:
    cfg = _kohya_cfg(
        tmp_path,
        prompts=[{"prompt": "first\n  second", "negative": "bad\t anatomy"}],
    )

    assert cfg.sampling.prompts[0].prompt == "first second"
    assert cfg.sampling.prompts[0].negative == "bad anatomy"


def test_idempotent_resolve_runtime_seeds(tmp_path: Path) -> None:
    """Running the resolver twice must not bounce a None back to a
    drawn integer — once None always None for the prompt row."""
    cfg = _kohya_cfg(
        tmp_path,
        prompts=[{"prompt": "x", "seed": -1}],
    )
    _resolve_runtime_seeds(cfg)
    first = cfg.sampling.seed
    _resolve_runtime_seeds(cfg)
    # Top-level seed only draws on -1, so re-running keeps it stable.
    assert cfg.sampling.seed == first
    assert cfg.sampling.prompts[0].seed is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
