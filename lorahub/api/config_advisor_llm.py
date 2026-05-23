"""LLM-driven training-config advisor.

This is the *real*智能推荐: the user clicks the button, the backend
hands the LLM a prompt containing

    1. hardware budget (GPU model + VRAM)
    2. dataset stats (path, image count, average resolution, caption coverage)
    3. user intent (free-form text — "a character LoRA, ~4h budget")
    4. the full set of fields the schema understands, with their defaults,
       lock policy, and one-line description
    5. the user's current draft as the "starting point"

and asks for a JSON response of shape::

    {
      "rationale": "Two-paragraph summary of the reasoning.",
      "patches": [
        {"field": "backend.animaLora.networkDim", "value": 16,
         "reason": "8GB target..."},
        ...
      ],
      "fullConfig": {full updated training config as JSON / dict}
    }

The LLM is responsible for *all* the judgement (which knob matters at
which VRAM tier, character vs style trade-offs, EMA vs cudagraph_trees
trade-off etc.). The backend's job is purely: assemble the prompt
faithfully, call the model via the project-wide AI router, validate
the response is well-formed, and return it.

Design rules:

* No baked-in heuristics about VRAM tiers — the LLM decides.
* Locked-value fields are listed with the lock reason so the LLM
  doesn't propose changes that will get silently ignored.
* The response schema is strict JSON, validated server-side via
  pydantic. Malformed responses surface as a clear error rather
  than a confusing partial-apply.
* The patched config is round-tripped through the same
  ``check_cross_field_conflicts`` rule set used by /configs/validate
  — a recommendation that introduces a hard conflict gets flagged
  *along* with the recommendation so the UI can surface "applied,
  but heads-up: …".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from lorahub.api.ai_store import AIStore
from lorahub.core.ai.client import AIError, invoke
from lorahub.core.config.schema import TrainingConfig

log = logging.getLogger(__name__)


# Task ID under which the AI provider / model is configured.
# A user routes "config.recommend" to e.g. "deepseek/deepseek-v3" and
# this advisor picks it up automatically. See app.py's _LORAHUB_TASKS
# seed list for the full set.
ADVISOR_TASK_ID = "config.recommend"


_SYSTEM_PROMPT = """\
你是 LoraHub 的训练配置顾问。LoraHub 是一个基于 Anima(DiT + Qwen-Image VAE + Qwen3 文本编码器)
的扩散模型 LoRA 训练工作台,你只需要为 anima_lora 后端给出建议(用户的所有输入都会以这个后端为前提)。

## 你的任务

收到一份用户当前的 TrainingConfig YAML 草稿、目标硬件预算、数据集统计、与用户意图描述。
请输出一份**严格优于当前草稿**的完整新配置:既要避免 OOM 与已知字段冲突,也要兼顾收敛质量
与训练时长。把判断的依据写清楚——你不是在跑 benchmark,而是在帮一个普通用户从他的硬件
预算里挤出最好的训练效果。

## 输出格式(必须严格 JSON,不要任何 markdown 代码围栏)

```
{
  "rationale": "整体思路 4-8 句话。先说为什么这套配置匹配用户硬件 + 意图,再说权衡。",
  "patches": [
    {
      "field": "backend.animaLora.networkDim",
      "value": 16,
      "reason": "32GB 头部空间够,字符 LoRA 推荐 dim=16-32 区间;16 是最稳的中位选择。"
    }
  ],
  "fullConfig": { ... 完整的更新后 cfg,使用 camelCase ... }
}
```

字段路径用点号分隔 (例:`backend.animaLora.networkDim`、`schedule.batchSize`)。

## 必须遵守的硬约束

1. **锁定字段(locked-value)绝不要改**:用户改了 train.py 也不会理会,只会 silently ignore。
   后面会用 `## Locked fields` 段列出。
2. **架构 baseModel.arch 永远是 "anima"**——其他后端不在你的领域。
3. 严格 JSON 输出,不带 ```json 围栏、不带任何前后说明文字。
4. `fullConfig` 必须是完整的 TrainingConfig,可以被 schema 直接 validate。
   保留 baseModel / dataset / output 这些不可省略的顶层段。
5. 不要发明 schema 未定义的字段。

## 已知的字段间冲突 (必须避开)

* `backend.animaLora.compileMode = "full"` 与下列任一互斥:
  - `gradientCheckpointing = true`
  - `unslothOffloadCheckpointing = true`
  - `blocksToSwap > 0`
* `blocksToSwap > 0` 与 `cpuOffloadCheckpointing = true` 互斥
* `ema = true` 与 `compileInductorMode = "reduce-overhead"` 配合,
  cudagraph_trees liveness check 会失败,需要把 inductor mode 设为 "default"
* AdamW8bit / Lion8bit / 任何 8bit 优化器需要 bitsandbytes 包安装
* `validationSplitNum > 0` 但 `useCmmd = false` 是 wasted holdout
* `keepTokens > 0` 但 `useShuffledCaptionVariants = false` 是死字段
* `networkAlpha / networkDim` 的比值偏离 [0.25, 4.0] 会数值不稳

## 推荐选型的常识

* 8GB 卡常用:resolution 768,blocksToSwap≈24,gradientCheckpointing=true,
  AdamW8bit,batchSize 1 + gradAccum 4,关 sampling/validation/torch.compile
* 16GB 卡常用:resolution 1024,blocksToSwap≈12,gradientCheckpointing=true,
  AdamW,batchSize 1 + gradAccum 4,可开 sampling
* 24GB 卡常用:blocksToSwap=0,gradientCheckpointing=false,batchSize 2,
  可开 compileMode="blocks" + reduce-overhead
* 32GB+ 卡:都开,可上 networkDim 32 + LoHa rank 4 / DoRA + CMMD validation
* 字符 LoRA 通常 networkDim 16-32,style LoRA 4-16(LoHa rank=4 等效 dim=16)
* numRepeats × imageCount × epochs / (batch×accum) 应在 200-1000 步区间
* captionDropoutRate 常用 0.05-0.15,过高会忘 trigger,过低则不正则化
* keepTokens=1 锁定第一个 trigger token 不被 shuffle 抹掉,新手友好
"""


@dataclass(slots=True)
class HardwareContext:
    """Hardware budget passed into the prompt."""

    gpu_name: str | None
    vram_mib: int | None
    driver: str | None = None


@dataclass(slots=True)
class DatasetContext:
    """Dataset stats — only the LLM-relevant subset."""

    path: str | None
    image_count: int | None
    caption_coverage: float | None  # 0..1
    average_long_edge: int | None = None


@dataclass(slots=True)
class AdvisorRequest:
    """All inputs the advisor pipeline accepts.

    The orchestrator (route handler) is responsible for collecting
    these. Keeping the LLM-bound layer ignorant of WHERE the data
    came from (HTTP request body / lifespan singletons / GPU probe)
    means the advisor itself stays straightforwardly testable.
    """

    current_config: dict[str, Any]
    intent: str = ""
    hardware: HardwareContext | None = None
    dataset: DatasetContext | None = None


# ---------------------------------------------------------------------- #
# Prompt builder
# ---------------------------------------------------------------------- #


_LOCKED_FIELDS_NOTE = """\
## Locked fields(用户改了也不会生效,请保留默认值)

* `staticTokenCount`: 4096 — Anima DiT torch.compile 锁死 4096-token bucket map
* `vaeChunkSize`: 64 — QwenImage VAE memory layout 锁死 64
* `captionExtension`: ".txt" — 数据 pipeline 写死 .txt 后缀
* `saveModelAs`: "safetensors" — Anima 只能加载 safetensors
* `pathPattern`: "*" — 数据 pipeline 默认通配
* `networkModule`: "networks.lora_anima" — 唯一可用的 LoRA module
"""


def build_user_prompt(req: AdvisorRequest) -> str:
    """Render the per-call user message.

    Includes hardware / dataset / intent / current config — everything
    the LLM needs to make an informed call. Intentionally verbose:
    LLM context is cheap, ambiguity is expensive.
    """
    lines: list[str] = []
    lines.append("# 输入")
    lines.append("")
    if req.hardware is not None:
        lines.append("## 硬件")
        lines.append(
            f"- GPU 型号: {req.hardware.gpu_name or '未知'}"
        )
        lines.append(
            f"- VRAM: {req.hardware.vram_mib} MiB"
            if req.hardware.vram_mib else "- VRAM: 未知"
        )
        if req.hardware.driver:
            lines.append(f"- 驱动: {req.hardware.driver}")
        lines.append("")
    if req.dataset is not None:
        lines.append("## 数据集")
        if req.dataset.path:
            lines.append(f"- 路径: {req.dataset.path}")
        if req.dataset.image_count is not None:
            lines.append(f"- 图片数: {req.dataset.image_count}")
        if req.dataset.average_long_edge is not None:
            lines.append(f"- 平均长边: {req.dataset.average_long_edge}px")
        if req.dataset.caption_coverage is not None:
            cov_pct = round(req.dataset.caption_coverage * 100)
            lines.append(f"- caption 覆盖率: {cov_pct}%")
        lines.append("")
    lines.append("## 用户意图")
    lines.append(req.intent.strip() or "(用户没有特别说明,按通用 LoRA 训练处理)")
    lines.append("")
    lines.append("## 当前配置(用户起点)")
    lines.append("```json")
    lines.append(json.dumps(req.current_config, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append(_LOCKED_FIELDS_NOTE)
    lines.append("")
    lines.append("# 输出")
    lines.append("")
    lines.append("严格 JSON,不要任何前后文字。")
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Response schema
# ---------------------------------------------------------------------- #


class AdvisorPatch(BaseModel):
    field: str = Field(min_length=1, max_length=200)
    value: Any
    reason: str = Field(min_length=1, max_length=2000)


class AdvisorResponse(BaseModel):
    rationale: str = Field(default="", max_length=4000)
    patches: list[AdvisorPatch] = Field(default_factory=list, max_length=80)
    fullConfig: dict[str, Any] = Field(default_factory=dict)


def parse_response(text: str) -> AdvisorResponse:
    """Strict JSON parse + pydantic validate.

    LLMs love wrapping JSON in ```json fences even when explicitly
    told not to. We strip an outer fence if present, then parse;
    anything that still doesn't validate raises so the route can
    surface "LLM returned malformed JSON, please retry" rather than
    silently corrupting the user's config.
    """
    body = text.strip()
    # Tolerate ```json ... ``` and ``` ... ``` wrappers.
    if body.startswith("```"):
        # drop the first fence line and the trailing fence
        first_nl = body.find("\n")
        body = body[first_nl + 1 :] if first_nl >= 0 else body
        if body.rstrip().endswith("```"):
            body = body.rstrip()[: -3].rstrip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        msg = f"LLM 返回不是合法 JSON: {exc.msg} at line {exc.lineno}"
        raise AdvisorError(msg) from exc
    try:
        return AdvisorResponse.model_validate(data)
    except ValidationError as exc:
        msg = f"LLM 返回的 JSON 不符合 advisor 协议: {exc.errors(include_url=False)[:3]!r}"
        raise AdvisorError(msg) from exc


# ---------------------------------------------------------------------- #
# Orchestrator
# ---------------------------------------------------------------------- #


class AdvisorError(RuntimeError):
    """Wraps any user-actionable failure of the advisor pipeline."""


@dataclass(slots=True)
class AdvisorOutcome:
    """What the advisor pipeline returns to the route handler."""

    rationale: str
    patches: list[dict[str, Any]]
    full_config: dict[str, Any]
    # Validation issues the LLM-proposed full_config triggers (cross-field
    # rule set). Empty when the LLM produced a clean recipe.
    validation_issues: list[dict[str, Any]] = field(default_factory=list)
    # Provider / model that actually answered, for the UI to display.
    provider_id: str = ""
    model_id: str = ""
    # Round-trip latency in ms.
    elapsed_ms: int = 0


def run_advisor(
    store: AIStore,
    request: AdvisorRequest,
    *,
    timeout_s: float = 60.0,
) -> AdvisorOutcome:
    """Full advisor pipeline:

    1. Resolve the ``config.recommend`` AI route (provider + model).
    2. Render system + user prompts.
    3. Invoke via ``lorahub.core.ai.client.invoke``.
    4. Parse + validate the JSON response.
    5. Round-trip the proposed full config through the schema and the
       cross-field conflict rules so the UI can flag any LLM-introduced
       issues alongside the recommendation.
    """
    import time  # noqa: PLC0415

    route = store.get_route(ADVISOR_TASK_ID) or store.get_route("global.default")
    if route is None:
        msg = (
            f"AI 路由 '{ADVISOR_TASK_ID}' 未配置,且 'global.default' 也没有。"
            "请到 设置 → AI 服务商 → 路由 给 config.recommend 选择一个 provider + model。"
        )
        raise AdvisorError(msg)
    if not route.provider_id or not route.model_id:
        msg = (
            f"AI 路由 '{route.task_id}' 还没绑定 provider/model。"
            "请到 设置 → AI 服务商 → 路由 完成绑定。"
        )
        raise AdvisorError(msg)

    system_prompt = _system_prompt_for_route(route)
    user_prompt = build_user_prompt(request)

    started = time.monotonic()
    try:
        result = invoke(
            store,
            provider_id=route.provider_id,
            model_id=route.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            route=route,
            timeout=timeout_s,
        )
    except AIError as exc:
        msg = f"调用 AI 服务商失败: {exc}"
        raise AdvisorError(msg) from exc
    elapsed_ms = int((time.monotonic() - started) * 1000)

    response = parse_response(result.text or "")

    # Round-trip the proposed full config through schema + cross-field
    # rule set. The route handler / UI surfaces these as warnings
    # alongside the patches so users see "applied, but heads-up: …".
    validation_issues: list[dict[str, Any]] = []
    if response.fullConfig:
        try:
            cfg = TrainingConfig.model_validate(response.fullConfig)
            from lorahub.core.backends.anima_lora.policies import (  # noqa: PLC0415
                check_cross_field_conflicts,
            )

            for iss in check_cross_field_conflicts(cfg):
                validation_issues.append(
                    {
                        "severity": iss.severity.value,
                        "field": iss.field,
                        "message": iss.message,
                    }
                )
        except ValidationError as exc:
            # The LLM produced a config that doesn't even pass schema
            # validation. Surface the first 3 errors so the UI can
            # explain instead of silently failing the apply.
            for e in exc.errors(include_url=False)[:3]:
                validation_issues.append(
                    {
                        "severity": "error",
                        "field": ".".join(str(p) for p in e.get("loc", [])),
                        "message": str(e.get("msg", "")),
                    }
                )

    return AdvisorOutcome(
        rationale=response.rationale,
        patches=[p.model_dump() for p in response.patches],
        full_config=response.fullConfig,
        validation_issues=validation_issues,
        provider_id=route.provider_id,
        model_id=route.model_id,
        elapsed_ms=elapsed_ms,
    )


def _system_prompt_for_route(route: Any) -> str:
    """Use the user-customised system_prompt if they've written one,
    else fall back to the curated default. This matches how the rest
    of the AI surface (caption / diagnose) behaves so power users can
    tune wording without forking the codebase."""
    custom = (getattr(route, "system_prompt", None) or "").strip()
    if custom:
        return custom
    return _SYSTEM_PROMPT


__all__ = [
    "ADVISOR_TASK_ID",
    "AdvisorError",
    "AdvisorOutcome",
    "AdvisorRequest",
    "AdvisorResponse",
    "DatasetContext",
    "HardwareContext",
    "build_user_prompt",
    "parse_response",
    "run_advisor",
]
