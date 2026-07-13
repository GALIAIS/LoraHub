"""Tests for kohya / diffusion-pipe cross-field policies + the
backend-aware advisor prompt selection.

The anima_lora policy / advisor surface is covered by
``test_config_advisor_llm.py``. This file mirrors the same shape for
the two other backends:

* one focused test per significant kohya rule
* one focused test per significant dp rule
* the advisor's system prompt picks the right backend cheat sheet
* the advisor round-trips the proposed full config through the
  matching backend's policies (kohya proposal goes through kohya
  policies, not anima's)
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from lorahub.api.config_advisor_llm import (
    AdvisorRequest,
    HardwareContext,
    _BACKEND_GUIDES,
    _resolve_policy_check,
    _system_prompt_for_backend,
    build_user_prompt,
    run_advisor,
)
from lorahub.core.backends.diffusion_pipe.policies import (
    check_cross_field_conflicts as dp_check,
)
from lorahub.core.backends.kohya.policies import (
    check_cross_field_conflicts as kohya_check,
)
from lorahub.core.config.schema import TrainingConfig


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _kohya_cfg(**override: Any) -> TrainingConfig:
    """Smallest valid kohya config we can build from the schema.

    We assemble the dict manually (vs loading from configs/) so the
    test isn't coupled to whatever configs happen to ship.
    """
    payload: dict[str, Any] = {
        "baseModel": {"arch": "sdxl", "checkpoint": "/tmp/sdxl.safetensors"},
        "dataset": {
            "source": "/tmp/dataset",
            "resolution": [1024, 1024],
            "bucket": {"enabled": True, "min": 256, "max": 2048, "step": 64},
            "caption": {"strategy": "tag_file", "ext": ".txt"},
            "numRepeats": 1,
        },
        "network": {"type": "lora", "rank": 16, "alpha": 16},
        "optimizer": {"type": "AdamW", "lr": {"unet": 1.0e-4, "textEncoder": 5.0e-5}},
        "schedule": {"epochs": 8, "batchSize": 1},
        "output": {"name": "test_kohya", "saveEveryNEpochs": 2},
        "backend": {"type": "kohya"},
    }
    for path, value in override.items():
        parts = path.split(".")
        target = payload
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = value
    return TrainingConfig.model_validate(payload)


def _dp_cfg(**override: Any) -> TrainingConfig:
    payload: dict[str, Any] = {
        "baseModel": {"arch": "flux", "checkpoint": "/tmp/flux.safetensors"},
        "dataset": {
            "source": "/tmp/dataset",
            "resolution": [1024, 1024],
            "caption": {"strategy": "tag_file", "ext": ".txt"},
            "numRepeats": 1,
        },
        "network": {"type": "lora", "rank": 16, "alpha": 16},
        "schedule": {"epochs": 8, "batchSize": 1},
        "output": {"name": "test_dp"},
        "backend": {
            "type": "diffusion-pipe",
            "diffusionPipe": {},
        },
    }
    for path, value in override.items():
        parts = path.split(".")
        target = payload
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = value
    return TrainingConfig.model_validate(payload)


# --------------------------------------------------------------------- #
# kohya policies
# --------------------------------------------------------------------- #


def test_kohya_clean_baseline_is_silent() -> None:
    """A reasonable default kohya config must not light any rule."""
    cfg = _kohya_cfg()
    issues = kohya_check(cfg)
    assert issues == [], [i.message for i in issues]


def test_kohya_alpha_dim_ratio_warns() -> None:
    cfg = _kohya_cfg(**{"network.rank": 4, "network.alpha": 64})  # 16x ratio
    assert any(
        i.severity == "warning" and "network.alpha" in i.field
        for i in kohya_check(cfg)
    )


def test_kohya_bucket_min_gt_max_is_error() -> None:
    cfg = _kohya_cfg(
        **{"dataset.bucket": {"enabled": True, "min": 1024, "max": 256}},
    )
    assert any(
        i.severity == "error" and "bucket" in i.field for i in kohya_check(cfg)
    )


def test_kohya_caption_drop_rate_too_high_warns() -> None:
    cfg = _kohya_cfg(**{"dataset.caption.dropRate": 0.7})
    assert any(
        i.severity == "warning" and "dropRate" in i.field
        for i in kohya_check(cfg)
    )


def test_kohya_save_every_exceeds_epochs_warns() -> None:
    cfg = _kohya_cfg(
        **{"schedule.epochs": 3, "output.saveEveryNEpochs": 10},
    )
    assert any(
        "saveEveryNEpochs" in i.field for i in kohya_check(cfg)
    )


def test_kohya_persistent_workers_requires_worker_processes() -> None:
    cfg = _kohya_cfg(
        **{"dataloader.numWorkers": 0, "dataloader.persistentWorkers": True},
    )

    assert any(
        i.severity == "error" and i.field == "dataloader.persistentWorkers"
        for i in kohya_check(cfg)
    )


def test_kohya_attention_overlap_warns() -> None:
    cfg = _kohya_cfg(
        **{"backend.extraArgs": {"xformers": True, "sdpa": True}},
    )
    assert any(
        i.severity == "warning" and "attention" in i.message.lower()
        or "extraArgs" in i.field
        for i in kohya_check(cfg)
    )


def test_kohya_unsupported_conditioning_and_subset_fields_are_rejected() -> None:
    cfg = _kohya_cfg(
        **{
            "dataset.conditioningDir": "/tmp/reference",
            "dataset.subsets": [
                {
                    "path": "/tmp/subset",
                    "conditioningDataDir": "/tmp/reference",
                    "maskPath": "/tmp/masks",
                    "arBuckets": [1.0],
                }
            ],
        }
    )
    issues = kohya_check(cfg)
    assert any(i.severity == "error" and i.field == "dataset.conditioningDir" for i in issues)
    assert sum(i.severity == "error" and i.field == "dataset.subsets" for i in issues) == 3


def test_kohya_skips_when_backend_is_anima() -> None:
    """Loading an anima cfg through kohya policies must stay silent —
    rules read kohya-shaped fields that don't exist on anima."""
    cfg = _kohya_cfg(**{"backend.type": "anima_lora"})
    assert kohya_check(cfg) == []


# --------------------------------------------------------------------- #
# diffusion-pipe policies
# --------------------------------------------------------------------- #


def test_dp_clean_baseline_is_silent() -> None:
    cfg = _dp_cfg()
    assert dp_check(cfg) == []


def test_dp_pipeline_stages_requires_reentrant_ckpt() -> None:
    cfg = _dp_cfg(
        **{
            "backend.diffusionPipe": {
                "pipelineStages": 2,
                "reentrantActivationCheckpointing": False,
            }
        }
    )
    assert any(
        i.severity == "error" and "reentrantActivationCheckpointing" in i.field
        for i in dp_check(cfg)
    )


def test_dp_blocks_to_swap_with_compile_is_error() -> None:
    cfg = _dp_cfg(
        **{"backend.diffusionPipe": {"blocksToSwap": 12, "compile": True}},
    )
    assert any(
        i.severity == "error" and "compile" in i.field
        for i in dp_check(cfg)
    )


def test_dp_partition_split_length_mismatch_is_error() -> None:
    cfg = _dp_cfg(
        **{
            "backend.diffusionPipe": {
                "pipelineStages": 4,
                "partitionMethod": "manual",
                "partitionSplit": [1, 2],  # 2 entries, expected 3
                "reentrantActivationCheckpointing": True,
            }
        }
    )
    assert any(
        i.severity == "error" and "partitionSplit" in i.field
        for i in dp_check(cfg)
    )


def test_dp_manual_partition_requires_split() -> None:
    cfg = _dp_cfg(
        **{
            "backend.diffusionPipe": {
                "pipelineStages": 2,
                "partitionMethod": "manual",
                "reentrantActivationCheckpointing": True,
            }
        }
    )
    assert any(
        i.severity == "error" and i.field == "backend.diffusionPipe.partitionSplit"
        for i in dp_check(cfg)
    )


def test_dp_partition_split_requires_manual_mode() -> None:
    cfg = _dp_cfg(**{"backend.diffusionPipe": {"partitionSplit": [10]}})
    assert any(
        i.severity == "error" and i.field == "backend.diffusionPipe.partitionSplit"
        for i in dp_check(cfg)
    )


def test_dp_unsupported_conditioning_and_regularization_data_are_rejected() -> None:
    cfg = _dp_cfg(
        **{
            "dataset.regSource": "/tmp/reg",
            "dataset.subsets": [
                {"path": "/tmp/subset", "conditioningDataDir": "/tmp/reference"}
            ],
        }
    )
    issues = dp_check(cfg)
    assert any(i.severity == "error" and i.field == "dataset.regSource" for i in issues)
    assert any(i.severity == "error" and i.field == "dataset.subsets" for i in issues)


def test_dp_val_split_is_rejected_instead_of_silently_ignored() -> None:
    cfg = _dp_cfg(**{"dataset.valSplit": 0.1})
    assert any(
        i.severity == "error" and i.field == "dataset.valSplit"
        for i in dp_check(cfg)
    )


def test_dp_eval_overlap_warns() -> None:
    cfg = _dp_cfg(
        **{
            "backend.diffusionPipe": {
                "evalEveryNEpochs": 2,
                "evalEveryNSteps": 100,
            }
        }
    )
    assert any(
        i.severity == "warning" and "eval" in i.field.lower()
        for i in dp_check(cfg)
    )


def test_dp_max_ar_lt_min_ar_is_error() -> None:
    cfg = _dp_cfg(
        **{"backend.diffusionPipe": {"minAr": 2.0, "maxAr": 0.5}},
    )
    assert any(
        i.severity == "error" and "maxAr" in i.field
        for i in dp_check(cfg)
    )


def test_dp_uncond_fraction_too_high_warns() -> None:
    cfg = _dp_cfg(**{"backend.diffusionPipe": {"uncondFraction": 0.7}})
    assert any(
        i.severity == "warning" and "uncondFraction" in i.field
        for i in dp_check(cfg)
    )


def test_dp_skips_when_backend_is_kohya() -> None:
    """An anima / kohya cfg must not light dp rules."""
    cfg = _dp_cfg(**{"backend.type": "kohya"})
    # Strip the dp options bag so cfg is a plausible kohya config.
    payload = cfg.model_dump(by_alias=True, mode="json")
    payload["backend"] = {"type": "kohya"}
    fresh = TrainingConfig.model_validate(payload)
    assert dp_check(fresh) == []


# --------------------------------------------------------------------- #
# Advisor system prompt picks the right cheat sheet
# --------------------------------------------------------------------- #


def test_system_prompt_includes_anima_specific_locks() -> None:
    prompt = _system_prompt_for_backend("anima_lora")
    assert "vaeChunkSize" in prompt
    assert "QwenImage" in prompt


def test_system_prompt_for_kohya_excludes_anima_locks() -> None:
    prompt = _system_prompt_for_backend("kohya")
    # No anima-specific lock-rule lines.
    assert "vaeChunkSize" not in prompt
    assert "networks.lora_anima" not in prompt
    # And does include kohya-shaped guidance.
    assert "kohya" in prompt
    assert "fused" in prompt or "8bit" in prompt


def test_system_prompt_for_dp_uses_dp_guide() -> None:
    prompt = _system_prompt_for_backend("diffusion-pipe")
    assert "pipelineStages" in prompt
    assert "DeepSpeed" in prompt
    # Tolerates the schema-key spelling too.
    same = _system_prompt_for_backend("diffusion_pipe")
    assert "pipelineStages" in same


def test_system_prompt_unknown_backend_falls_back() -> None:
    prompt = _system_prompt_for_backend(None)
    assert prompt == _system_prompt_for_backend("anima_lora")


def test_resolve_policy_check_routes_correctly() -> None:
    assert _resolve_policy_check("kohya") is kohya_check
    assert _resolve_policy_check("diffusion-pipe") is dp_check
    assert _resolve_policy_check("diffusion_pipe") is dp_check
    # unknown / empty falls back to anima
    from lorahub.core.backends.anima_lora.policies import (
        check_cross_field_conflicts as anima_check,
    )

    assert _resolve_policy_check("anima_lora") is anima_check
    assert _resolve_policy_check("") is anima_check


def test_user_prompt_mentions_target_backend() -> None:
    """The user message must tell the LLM which backend to stay
    within so it doesn't propose cross-backend knobs."""
    req = AdvisorRequest(
        current_config={
            "baseModel": {"arch": "sdxl"},
            "backend": {"type": "kohya"},
        },
        intent="character LoRA",
        hardware=HardwareContext(gpu_name="RTX 4090", vram_mib=24576),
    )
    prompt = build_user_prompt(req)
    assert "kohya" in prompt
    assert "sdxl" in prompt


# --------------------------------------------------------------------- #
# Advisor end-to-end with a kohya config + mocked LLM
# --------------------------------------------------------------------- #


def test_run_advisor_uses_kohya_policies_for_kohya_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An LLM proposal that targets kohya must be round-tripped
    through kohya policies, not anima — otherwise we'd surface
    irrelevant warnings (anima reads AnimaLoraOptions; kohya doesn't
    populate it)."""
    from lorahub.api.ai_store import AIRoute
    from lorahub.api import config_advisor_llm as advisor_mod

    full_cfg = _kohya_cfg().model_dump(by_alias=True, mode="json")
    # Inject a known kohya conflict: drop_rate >= 0.5
    full_cfg["dataset"]["caption"]["dropRate"] = 0.8

    class _StubStore:
        def get_route(self, task_id: str):  # type: ignore[no-untyped-def]
            return AIRoute(
                task_id="config.recommend", provider_id="p", model_id="m",
            )

    class _R:
        text = json.dumps(
            {
                "rationale": "More dropout for regularisation.",
                "patches": [
                    {
                        "field": "dataset.caption.dropRate",
                        "value": 0.8,
                        "reason": "stronger reg",
                    }
                ],
                "fullConfig": full_cfg,
            }
        )
        provider_id = "p"
        model_id = "m"

    monkeypatch.setattr(advisor_mod, "invoke", lambda *a, **kw: _R())

    outcome = run_advisor(
        _StubStore(),  # type: ignore[arg-type]
        AdvisorRequest(
            current_config={
                "baseModel": {"arch": "sdxl"},
                "backend": {"type": "kohya"},
            },
            intent="character LoRA",
        ),
    )
    # The kohya rule fired (caption dropRate >= 0.5).
    assert any(
        "dropRate" in i["field"] and i["severity"] == "warning"
        for i in outcome.validation_issues
    ), outcome.validation_issues
