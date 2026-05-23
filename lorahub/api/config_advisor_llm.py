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


_SYSTEM_PROMPT_HEAD = """\
你是 LoraHub 的训练配置顾问。LoraHub 是一个扩散模型 LoRA 训练工作台,支持三个后端:
``anima_lora`` (基于 Anima DiT + Qwen-Image VAE + Qwen3 文本编码器)、``kohya``
(基于 kohya-ss/sd-scripts,涵盖 SDXL / SD1.5 / SD3 / FLUX / Lumina 等主流架构)、
``diffusion-pipe`` (基于 tdrussell/diffusion-pipe,涵盖 Wan / HunyuanVideo / LTX
等视频模型与多种图像架构)。

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
2. **不要改 baseModel.arch**:用户已经选好了底模架构,你的任务是基于该架构调参。
3. **不要改 backend.type**:如果用户在 kohya 后端,你的建议必须是 kohya 字段;
   不能因为某个超参跨后端而切换后端。
4. 严格 JSON 输出,不带 ```json 围栏、不带任何前后说明文字。
5. `fullConfig` 必须是完整的 TrainingConfig,可以被 schema 直接 validate。
   保留 baseModel / dataset / output 这些不可省略的顶层段。
6. 不要发明 schema 未定义的字段。
"""


# Per-backend "field cheat sheet" inserted into the system prompt.
# Each entry covers the cross-field gotchas the backend's
# policies.py also enforces, so the LLM has the same heuristics the
# validator uses and won't propose a config that immediately fails
# downstream.
_BACKEND_GUIDES: dict[str, str] = {
    "anima_lora": """\
## anima_lora 后端 — 已知字段冲突

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
* `lora.minRank > networkDim` 会让 T-LoRA route layer 构造失败

## anima_lora 后端 — Locked fields(用户改了也不会生效)

* `staticTokenCount`: 4096 — Anima DiT torch.compile 锁死 4096-token bucket map
* `vaeChunkSize`: 64 — QwenImage VAE memory layout 锁死 64
* `captionExtension`: ".txt"
* `saveModelAs`: "safetensors"
* `pathPattern`: "*"
* `networkModule`: "networks.lora_anima"

## anima_lora 后端 — 推荐选型常识

* 8GB:resolution 768,blocksToSwap≈24,gradientCheckpointing=true,AdamW8bit,
  batchSize 1 + gradAccum 4,关 sampling/validation/torch.compile
* 16GB:resolution 1024,blocksToSwap≈12,gradientCheckpointing=true,AdamW,
  可开 sampling
* 24GB:blocksToSwap=0,gradientCheckpointing=false,batchSize 2,
  可开 compileMode="blocks" + reduce-overhead
* 32GB+:都开,可上 networkDim 32 + LoHa rank 4 / DoRA + CMMD validation
* 字符 LoRA 通常 networkDim 16-32,style LoRA 4-16(LoHa rank=4 等效 dim=16)
""",
    "kohya": """\
## kohya 后端 — 已知字段冲突

* `network.alpha / network.rank` 比值偏离 [0.25, 4.0] 会数值不稳;推荐 alpha == rank
* `network.rank < 4` 几乎学不到内容
* AdamW8bit / Lion8bit 需要 bitsandbytes (kohya sd-scripts 子环境里通常自带)
* fused 优化器(AdamWFused 等)需要 bf16/fp16,fp32 + fused 不会更快也不省显存
* `dataset.bucket.min > max` 会让 kohya 启动 assert 'bucket reso list is empty'
* `dataset.bucket.max` 比 resolution 长边大 2× 以上,极端长宽比图片显存暴涨
* `dataset.caption.keepTokens > 0` 但 shuffle=false → keepTokens 不起作用
* `dataset.caption.dropRate >= 0.5` 模型会忘记 trigger word
* `output.saveEveryNEpochs > schedule.epochs` 整次训练只在最后落一次盘
* `backend.extraArgs` 同时开 xformers / sdpa / flash 多个 attention 实现,kohya 只取
  最先识别的那个

## kohya 后端 — 推荐选型常识

* SDXL 8GB:resolution 1024,gradientCheckpointing=true,AdamW8bit,
  batchSize 1 + gradAccum 4,fp8 te,xformers
* SDXL 12GB:加大 networkDim 16 → 32,可关 grad-ckpt
* SDXL 24GB:batchSize 2-4,关 grad-ckpt,fp32 te
* FLUX/SD3 上 t5xxl_dtype=fp8 是标准搭配,不与 fp32 训练混用
* character LoRA 通常 rank 16-32,style 4-16
* mixed_precision 与 cfg.precision 必须一致(bf16 / fp16 二选一,fp32 不省显存)
""",
    "diffusion-pipe": """\
## diffusion-pipe 后端 — 已知字段冲突

* `pipelineStages > 1` 启用 pipeline parallel 时必须 `reentrantActivationCheckpointing = true`
  (DeepSpeed PP 调度器要求)
* `blocksToSwap > 0` 与 `compile = true` 互斥(cudagraph 与 swap 进出冲突)
* `partitionSplit` 长度必须等于 `pipelineStages - 1`,否则 DeepSpeed 启动 assert
* `evalEveryNEpochs / evalEveryNSteps / evalEveryNExamples` 只能取一个,
  其余被忽略
* `transformerDtype` 是 fp8 系列时不应配 fp32 全局精度
* `transformerDtype` 与 `diffusionModelDtype` 同时设且不同时,后者会被忽略
* `cachingBatchSize > 8` 在 8GB 卡上 VAE/TE encode 阶段会爆显存
* `cacheShuffleNum` 在 (0, 8) 区间形同不开,要么 0 要么 ≥16
* `dataset.bucket.maxAr < minAr` 数据集会被全丢弃
* `uncondFraction >= 0.5` 模型会大幅偏向无条件分布

## diffusion-pipe 后端 — 推荐选型常识

* HunyuanVideo / Wan 等视频模型常用:transformerDtype=float8_e4m3fn,
  pipelineStages=2 (~24GB 卡), reentrantActivationCheckpointing=true
* Flux / SDXL 等图像模型常用:transformerDtype=bfloat16,blocksToSwap=0
* video 上 imageMicroBatchSizePerGpu 与 evalMicroBatchSizePerGpu 要分别设
* uncondFraction 0.05-0.15 是标准 CFG 训练区间
* min_ar / max_ar 通常 0.5 / 2.0,极端比例数据集再放宽
""",
}


# Default fallback when ``cfg.backend.type`` is unrecognised.
_BACKEND_GUIDES["__default__"] = _BACKEND_GUIDES["anima_lora"]


def _system_prompt_for_backend(backend_type: str | None) -> str:
    """Compose the full system prompt by appending the backend-specific
    cheat sheet to the shared head. Falls back to anima_lora's sheet
    when the backend type is missing or unrecognised."""
    key = (backend_type or "").strip()
    # Tolerate both schema key spellings.
    if key == "diffusion_pipe":
        key = "diffusion-pipe"
    guide = _BACKEND_GUIDES.get(key) or _BACKEND_GUIDES["__default__"]
    return _SYSTEM_PROMPT_HEAD + "\n" + guide


# Legacy alias kept so a custom system_prompt the user already typed
# in Settings → AI providers continues to work as before — those
# settings stored a string verbatim under the route, not a closure.
_SYSTEM_PROMPT = _SYSTEM_PROMPT_HEAD + "\n" + _BACKEND_GUIDES["anima_lora"]


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


# (Locked-field rationale moved into ``_BACKEND_GUIDES`` above so the
# system prompt picks up the right list per backend. The legacy
# ``_LOCKED_FIELDS_NOTE`` constant was deleted.)


def build_user_prompt(req: AdvisorRequest) -> str:
    """Render the per-call user message.

    Includes hardware / dataset / intent / current config — everything
    the LLM needs to make an informed call. Intentionally verbose:
    LLM context is cheap, ambiguity is expensive.
    """
    lines: list[str] = []
    lines.append("# 输入")
    lines.append("")

    # Backend the user is currently on; the LLM is told not to switch.
    backend_type: str = ""
    cfg_backend = req.current_config.get("backend") if isinstance(req.current_config, dict) else None
    if isinstance(cfg_backend, dict):
        backend_type = str(cfg_backend.get("type") or "")
    arch = ""
    cfg_basemodel = req.current_config.get("baseModel") if isinstance(req.current_config, dict) else None
    if isinstance(cfg_basemodel, dict):
        arch = str(cfg_basemodel.get("arch") or "")
    if backend_type or arch:
        lines.append("## 目标后端 / 架构")
        if backend_type:
            lines.append(f"- backend.type: {backend_type}")
        if arch:
            lines.append(f"- baseModel.arch: {arch}")
        lines.append(
            "- 你的建议必须停留在该后端 + 架构组合下,不要跨后端切换。"
        )
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

    # Resolve which AIRoute to use. The seed list in app.py creates
    # an empty ``config.recommend`` row (provider/model both null) on
    # first boot, so ``store.get_route("config.recommend")`` returns
    # a truthy stub even before the user binds it. Treat that stub as
    # equivalent to "not configured" and fall back to the global
    # default — which is a much more common thing for users to have
    # already set up via tagging / caption flows.
    route = store.get_route(ADVISOR_TASK_ID)
    if route is None or not route.provider_id or not route.model_id:
        route = store.get_route("global.default") or route
    if route is None:
        msg = (
            f"AI 路由 '{ADVISOR_TASK_ID}' 未配置,且 'global.default' 也没有。"
            "请到 设置 → AI 服务商 → 路由 给 config.recommend 选择一个 provider + model。"
        )
        raise AdvisorError(msg)
    if not route.provider_id or not route.model_id:
        msg = (
            f"AI 路由 '{ADVISOR_TASK_ID}' 未绑定 provider/model,且 'global.default' 也没绑。"
            "请到 设置 → AI 服务商 → 路由 给其中一个完成绑定。"
        )
        raise AdvisorError(msg)

    # Pick the system prompt that matches the user's current backend
    # — anima_lora / kohya / diffusion-pipe each get their own field
    # cheat sheet so the LLM doesn't hallucinate cross-backend knobs.
    backend_type: str | None = None
    cfg_backend = (
        request.current_config.get("backend")
        if isinstance(request.current_config, dict) else None
    )
    if isinstance(cfg_backend, dict):
        backend_type = (cfg_backend.get("type") or None)  # type: ignore[assignment]
    system_prompt = _system_prompt_for_route(route, backend_type)
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
    # We pick the policies module that matches the backend the LLM was
    # told to stay within — running anima rules on a kohya proposal
    # would emit nonsense.
    validation_issues: list[dict[str, Any]] = []
    if response.fullConfig:
        try:
            cfg = TrainingConfig.model_validate(response.fullConfig)
            proposal_backend = (
                cfg.backend.type if cfg.backend and cfg.backend.type else (backend_type or "")
            )
            check = _resolve_policy_check(proposal_backend)

            for iss in check(cfg):
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


def _system_prompt_for_route(route: Any, backend_type: str | None = None) -> str:
    """Use the user-customised system_prompt if they've written one,
    else fall back to the curated default for the matching backend.
    Matches how the rest of the AI surface (caption / diagnose) behaves
    so power users can tune wording without forking the codebase."""
    custom = (getattr(route, "system_prompt", None) or "").strip()
    if custom:
        return custom
    return _system_prompt_for_backend(backend_type)


def _resolve_policy_check(backend_type: str):
    """Pick the cross-field rule set that matches ``backend_type``.

    Each backend module owns its own policies and gates them on
    ``cfg.backend.type`` so this lookup is mostly redundant — but
    selecting the right module up-front keeps imports targeted (no
    sense paying the kohya-policies import cost on an anima route).
    """
    from typing import Callable  # noqa: PLC0415

    key = (backend_type or "anima_lora").strip()
    if key == "diffusion_pipe":
        key = "diffusion-pipe"
    if key == "kohya":
        from lorahub.core.backends.kohya.policies import (  # noqa: PLC0415
            check_cross_field_conflicts as _check,
        )
        return _check
    if key == "diffusion-pipe":
        from lorahub.core.backends.diffusion_pipe.policies import (  # noqa: PLC0415
            check_cross_field_conflicts as _check,
        )
        return _check
    from lorahub.core.backends.anima_lora.policies import (  # noqa: PLC0415
        check_cross_field_conflicts as _check,
    )
    return _check


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
