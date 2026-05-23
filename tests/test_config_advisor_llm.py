"""Tests for the policies module + LLM-driven config advisor.

The policies side gets one focused test per cross-field rule so a
future schema change that drops or renames a knob can't silently
delete a check. The advisor side covers the prompt builder shape +
strict JSON parser tolerance + the full ``run_advisor`` orchestrator
with the AI client stubbed (no network calls).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import yaml

from lorahub.api.config_advisor_llm import (
    AdvisorError,
    AdvisorRequest,
    DatasetContext,
    HardwareContext,
    build_user_prompt,
    parse_response,
    run_advisor,
)
from lorahub.core.backends.anima_lora.policies import check_cross_field_conflicts
from lorahub.core.config.schema import TrainingConfig


# --------------------------------------------------------------------- #
# Helper: minimal valid TrainingConfig
# --------------------------------------------------------------------- #


def _base_cfg(**override: Any) -> TrainingConfig:
    """Build a TrainingConfig from anima_lora_default.yaml + overrides."""
    from pathlib import Path

    raw = yaml.safe_load(
        Path("configs/anima_lora_default.yaml").read_text(encoding="utf-8")
    )
    # Apply nested overrides via simple dotted-path walk.
    for key, value in override.items():
        parts = key.split(".")
        target = raw
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = value
    return TrainingConfig.model_validate(raw)


# --------------------------------------------------------------------- #
# Policies — one rule per test, kept terse.
# --------------------------------------------------------------------- #


def test_policy_compile_full_blocked_by_grad_ckpt() -> None:
    cfg = _base_cfg(
        **{
            "backend.animaLora.compileMode": "full",
            "backend.animaLora.gradientCheckpointing": True,
        }
    )
    issues = check_cross_field_conflicts(cfg)
    assert any(
        i.severity == "error" and "compile_mode='full'" in i.message
        for i in issues
    )


def test_policy_blocks_to_swap_blocks_cpu_offload_ckpt() -> None:
    cfg = _base_cfg(
        **{
            "backend.animaLora.blocksToSwap": 12,
            "backend.animaLora.cpuOffloadCheckpointing": True,
        }
    )
    issues = check_cross_field_conflicts(cfg)
    assert any(
        i.severity == "error" and "互斥" in i.message
        for i in issues
    )


def test_policy_min_rank_exceeds_dim_is_error() -> None:
    cfg = _base_cfg(
        **{
            "backend.animaLora.networkDim": 8,
            "backend.animaLora.lora": {
                "useOrtho": True,
                "useTimestepMask": True,
                "minRank": 16,
                "alphaRankScale": 1.0,
            },
        }
    )
    issues = check_cross_field_conflicts(cfg)
    assert any(
        i.severity == "error" and "minRank" in i.field
        for i in issues
    )


def test_policy_alpha_dim_ratio_warning() -> None:
    cfg = _base_cfg(
        **{
            "backend.animaLora.networkDim": 8,
            "backend.animaLora.networkAlpha": 64.0,  # 8x ratio — way out of band
        }
    )
    issues = check_cross_field_conflicts(cfg)
    assert any(
        i.severity == "warning" and "networkAlpha" in i.field
        for i in issues
    )


def test_policy_caption_dropout_too_high() -> None:
    cfg = _base_cfg(**{"backend.animaLora.captionDropoutRate": 0.7})
    issues = check_cross_field_conflicts(cfg)
    assert any("captionDropoutRate" in i.field for i in issues)


def test_policy_save_every_n_exceeds_max_epochs() -> None:
    cfg = _base_cfg(
        **{
            "backend.animaLora.maxTrainEpochs": 4,
            "backend.animaLora.saveEveryNEpochs": 10,
        }
    )
    issues = check_cross_field_conflicts(cfg)
    assert any("saveEveryNEpochs" in i.field for i in issues)


def test_policy_clean_default_yaml_emits_no_issues() -> None:
    cfg = _base_cfg()
    issues = check_cross_field_conflicts(cfg)
    # The shipped default recipe must not light up any rule —
    # otherwise users get noise the first time they open the form.
    assert issues == [], [i.message for i in issues]


# --------------------------------------------------------------------- #
# Advisor — prompt builder
# --------------------------------------------------------------------- #


def test_build_prompt_includes_hardware_dataset_intent() -> None:
    req = AdvisorRequest(
        current_config={"baseModel": {"arch": "anima"}, "schedule": {"epochs": 8}},
        intent="character LoRA, ~4h budget",
        hardware=HardwareContext(gpu_name="RTX 4090", vram_mib=24576),
        dataset=DatasetContext(
            path="./datasets/foo", image_count=58,
            caption_coverage=0.95, average_long_edge=1024,
        ),
    )
    prompt = build_user_prompt(req)
    assert "RTX 4090" in prompt
    assert "24576 MiB" in prompt
    assert "58" in prompt
    assert "character LoRA" in prompt
    # The current config gets embedded so the LLM has a starting point.
    assert "anima" in prompt


def test_build_prompt_handles_missing_optional_inputs() -> None:
    """When the user hasn't filled hardware / dataset, the prompt
    still renders — we just omit those sections instead of inserting
    'None' literals."""
    req = AdvisorRequest(current_config={"x": 1})
    prompt = build_user_prompt(req)
    assert "用户没有特别说明" in prompt
    assert "未知" not in prompt or "GPU 型号: 未知" not in prompt  # only when hw is supplied


# --------------------------------------------------------------------- #
# Advisor — JSON parser tolerance
# --------------------------------------------------------------------- #


def test_parse_response_accepts_clean_json() -> None:
    payload = json.dumps(
        {
            "rationale": "ok",
            "patches": [{"field": "schedule.epochs", "value": 12, "reason": "more steps"}],
            "fullConfig": {"x": 1},
        }
    )
    resp = parse_response(payload)
    assert resp.rationale == "ok"
    assert len(resp.patches) == 1
    assert resp.patches[0].field == "schedule.epochs"


def test_parse_response_strips_code_fences() -> None:
    payload = (
        "```json\n"
        + json.dumps({"rationale": "x", "patches": [], "fullConfig": {}})
        + "\n```"
    )
    resp = parse_response(payload)
    assert resp.rationale == "x"


def test_parse_response_rejects_non_json() -> None:
    with pytest.raises(AdvisorError, match="JSON"):
        parse_response("hello world")


def test_parse_response_rejects_bad_schema() -> None:
    payload = json.dumps({"rationale": "ok"})  # missing patches / fullConfig is OK
    # but a malformed patch entry should fail
    payload_bad = json.dumps(
        {
            "rationale": "ok",
            "patches": [{"field": "", "value": 1, "reason": "x"}],  # empty field
            "fullConfig": {},
        }
    )
    parse_response(payload)  # OK — defaults fill in
    with pytest.raises(AdvisorError):
        parse_response(payload_bad)


# --------------------------------------------------------------------- #
# Advisor — orchestrator with mocked AI client
# --------------------------------------------------------------------- #


def test_run_advisor_round_trips_through_validate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator hands the LLM-shaped JSON back through schema
    validation + cross-field conflict check, and forwards any issues
    so the UI can flag "applied, but heads-up" cases.
    """
    # 1. Fake AIRoute so run_advisor doesn't bail before invoke.
    from lorahub.api.ai_store import AIRoute, AIStore
    from lorahub.api import config_advisor_llm as advisor_mod

    class _StubStore:
        def get_route(self, task_id: str):  # type: ignore[no-untyped-def]
            if task_id == "config.recommend":
                return AIRoute(
                    task_id="config.recommend",
                    provider_id="stub-provider",
                    model_id="stub-model",
                )
            return None

    # 2. Stub invoke() — return a deterministic JSON response.
    full_cfg = yaml.safe_load(
        (
            __import__("pathlib").Path("configs/anima_lora_default.yaml")
        ).read_text(encoding="utf-8")
    )
    # Bump epochs in the proposed full config so the round-trip
    # exercise is non-trivial.
    full_cfg["schedule"]["epochs"] = 12
    fake_text = json.dumps(
        {
            "rationale": "Epochs were too low.",
            "patches": [
                {
                    "field": "schedule.epochs",
                    "value": 12,
                    "reason": "Need more steps for character convergence.",
                }
            ],
            "fullConfig": full_cfg,
        }
    )

    class _StubResult:
        text = fake_text
        provider_id = "stub-provider"
        model_id = "stub-model"

    def fake_invoke(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _StubResult()

    monkeypatch.setattr(advisor_mod, "invoke", fake_invoke)

    req = AdvisorRequest(
        current_config={"baseModel": {"arch": "anima"}},
        intent="character LoRA, want longer training",
        hardware=HardwareContext(gpu_name="RTX 4090", vram_mib=24576),
    )
    outcome = run_advisor(_StubStore(), req)  # type: ignore[arg-type]
    assert outcome.provider_id == "stub-provider"
    assert outcome.model_id == "stub-model"
    assert outcome.rationale == "Epochs were too low."
    assert len(outcome.patches) == 1
    assert outcome.patches[0]["field"] == "schedule.epochs"
    assert outcome.full_config["schedule"]["epochs"] == 12
    # Default recipe is policy-clean, so there should be no issues
    # surfaced from the round-trip.
    assert outcome.validation_issues == []


def test_run_advisor_surfaces_policy_violation_in_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM proposes something that triggers a cross-field rule —
    advisor returns the patches *and* the issues so the UI shows
    the trade-off."""
    from lorahub.api.ai_store import AIRoute
    from lorahub.api import config_advisor_llm as advisor_mod

    full_cfg = yaml.safe_load(
        (
            __import__("pathlib").Path("configs/anima_lora_default.yaml")
        ).read_text(encoding="utf-8")
    )
    # Inject a known conflict: caption_dropout_rate=0.8 (rule fires
    # at >= 0.5).
    full_cfg.setdefault("backend", {}).setdefault("animaLora", {})[
        "captionDropoutRate"
    ] = 0.8

    class _StubStore:
        def get_route(self, task_id: str):  # type: ignore[no-untyped-def]
            if task_id == "config.recommend":
                return AIRoute(
                    task_id="config.recommend",
                    provider_id="p",
                    model_id="m",
                )
            return None

    class _R:
        text = json.dumps(
            {
                "rationale": "Drop more captions for regularisation.",
                "patches": [
                    {
                        "field": "backend.animaLora.captionDropoutRate",
                        "value": 0.8,
                        "reason": "test",
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
        AdvisorRequest(current_config={}, intent="x"),
    )
    assert any("captionDropoutRate" in i["field"] for i in outcome.validation_issues)


def test_run_advisor_raises_when_route_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Empty:
        def get_route(self, _task_id: str):  # type: ignore[no-untyped-def]
            return None

    with pytest.raises(AdvisorError, match="未配置"):
        run_advisor(
            _Empty(),  # type: ignore[arg-type]
            AdvisorRequest(current_config={}, intent=""),
        )


def test_run_advisor_falls_back_to_global_default_when_task_route_unbound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: app.py's lifespan seeds an empty AIRoute for
    ``config.recommend``, so ``store.get_route(...)`` returned a
    truthy stub (provider/model both null) and the old ``or``-based
    fallback path never triggered. Result: users with a fully-
    configured ``global.default`` saw "未绑定" the first time they
    clicked 智能推荐 even though they had a working LLM bound.

    Verify the new resolution: stub config.recommend → fall back
    to global.default which has the binding."""
    from lorahub.api.ai_store import AIRoute
    from lorahub.api import config_advisor_llm as advisor_mod

    routes_by_id: dict[str, AIRoute] = {
        "config.recommend": AIRoute(
            task_id="config.recommend",
            provider_id=None,
            model_id=None,
        ),
        "global.default": AIRoute(
            task_id="global.default",
            provider_id="p",
            model_id="m",
        ),
    }

    class _Stub:
        def get_route(self, task_id: str):  # type: ignore[no-untyped-def]
            return routes_by_id.get(task_id)

    class _R:
        text = json.dumps(
            {"rationale": "ok", "patches": [], "fullConfig": {}}
        )
        provider_id = "p"
        model_id = "m"

    monkeypatch.setattr(advisor_mod, "invoke", lambda *a, **kw: _R())

    outcome = run_advisor(
        _Stub(),  # type: ignore[arg-type]
        AdvisorRequest(current_config={}, intent="x"),
    )
    # The fallback route's provider/model is what we actually used.
    assert outcome.provider_id == "p"
    assert outcome.model_id == "m"
