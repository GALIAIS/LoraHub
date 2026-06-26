/**
 * anima_lora-specific config section.
 *
 * Only visible when `backend.type === "anima_lora"`. Mirrors every field
 * on `AnimaLoraOptions` in `lorahub/core/config/schema.py`.
 *
 * Layout follows the upstream merge chain so users edit in roughly the
 * same order they'd find a knob in `configs/methods/lora.toml` /
 * `configs/base.toml`:
 *   1. Method + preset (the two core choices)
 *   2. Network / optim / sampling — the train.py argv overrides
 *   3. Memory + attn + compile (the perf tradeoff knobs)
 *   4. Method sub-config (visible only when the matching method is selected)
 *   5. Turbo distillation (orthogonal — when populated, switches the whole
 *      pipeline from train.py to scripts/distill_turbo.py)
 */
import { memo, useCallback } from "react"
import { Sparkles } from "lucide-react"
import { ANIMA_LORA_LOCKS, LockBadge } from "../anima-lora-locks"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
import {
  EnumSelect,
  FloatInput,
  IntInput,
  PathInput,
  Row,
  Section,
  ToggleSwitch,
} from "../widgets"
import {
  AnimaLoraLockedDefaultsSection,
  AnimaLoraTurboSection,
} from "./backend-anima-lora-advanced"
import { AnimaLoraMethodConfig } from "./backend-anima-lora-methods"
import {
  AnimaLoraCacheSection,
  AnimaLoraCompileSection,
  AnimaLoraMemorySection,
} from "./backend-anima-lora-performance"
import {
  LR_SCHEDULER_OPTIONS,
  METHOD_OPTIONS,
  OPTIMIZER_OPTIONS,
  PRESET_OPTIONS,
  TARGET_PRESET_OPTIONS,
  TIMESTEP_OPTIONS,
  WEIGHTING_SCHEME_OPTIONS,
} from "./backend-anima-lora-options"
import { SuggestDialog } from "./suggest-dialog"

/** Look up the lock badge for a field key; returns ``null`` when the field
 *  is unrestricted (most non-base.toml knobs). */
function lockBadgeFor(field: string) {
  const meta = ANIMA_LORA_LOCKS[field]
  return meta ? <LockBadge meta={meta} /> : null
}


export const BackendAnimaLoraFields = memo(function BackendAnimaLoraFields({
  value = {},
  optimizer,
  set,
  errorMap,
}: {
  value: NonNullable<ConfigFormValue["backend"]>["animaLora"]
  optimizer: ConfigFormValue["optimizer"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  const method = v.method ?? "lora"
  const isFullFinetune = method === "full_finetune"

  // Switching method clears the previously-active sub-config so the
  // model_validator on the Python side doesn't reject "method=postfix
  // but ipAdapter sub-config also set" (it'd accept it but it's noise).
  const onMethodChange = useCallback(
    (next: string) => {
      set(["backend", "animaLora", "method"], next)
    },
    [set],
  )

  return (
    <>
      <div className="flex justify-end -mt-1 mb-2">
        <SuggestDialog set={set} backend="anima_lora" />
      </div>
      <Row
        label="训练方法"
        required
        description="选择 Anima 训练方法。"
        errors={errorMap.get("backend.animaLora.method")}
      >
        <EnumSelect value={method} onChange={onMethodChange} options={METHOD_OPTIONS} />
      </Row>
      <Row
        label="硬件预设"
        description="读取 anima_lora/configs/presets.toml。"
        errors={errorMap.get("backend.animaLora.preset")}
      >
        <EnumSelect
          value={v.preset ?? "default"}
          onChange={(p) => set(["backend", "animaLora", "preset"], p)}
          options={PRESET_OPTIONS}
        />
      </Row>

      <Row label="输出名" errors={errorMap.get("backend.animaLora.outputName")}>
        <PathInput
          value={v.outputName ?? ""}
          onChange={(s) => set(["backend", "animaLora", "outputName"], s || undefined)}
          placeholder="anima_lora"
        />
      </Row>

      <Row
        label="差异训练"
        description="启用 conditioning training。参考图目录在数据集子集内设置。"
      >
        <ToggleSwitch
          checked={v.conditioning ?? false}
          onCheckedChange={(b) =>
            set(["backend", "animaLora", "conditioning"], b)
          }
        />
      </Row>

      {/* === 通用网络参数 === */}
      <Section
        icon={<Sparkles className="size-3.5" />}
        title={isFullFinetune ? "训练范围" : "网络容量"}
        subtitle={isFullFinetune ? "Full model finetune" : "LoRA rank / alpha"}
      >
        {!isFullFinetune && (
          <>
            <Row label="网络维度 (rank)" errors={errorMap.get("backend.animaLora.networkDim")}>
              <FloatInput
                value={v.networkDim}
                onChange={(n) => set(["backend", "animaLora", "networkDim"], n)}
                placeholder="16"
                min={1}
              />
            </Row>
            <Row label="网络 alpha" errors={errorMap.get("backend.animaLora.networkAlpha")}>
              <FloatInput
                value={v.networkAlpha}
                onChange={(n) => set(["backend", "animaLora", "networkAlpha"], n)}
                placeholder="16"
                min={1}
              />
            </Row>
          </>
        )}
        <Row
          label={isFullFinetune ? "只训练 DiT" : "只训练 UNet"}
          description={
            isFullFinetune
              ? "开启时仅训练 Anima DiT。"
              : "开启时不训练 text encoder。"
          }
        >
          <ToggleSwitch
            checked={v.networkTrainUnetOnly ?? true}
            onCheckedChange={(c) => set(["backend", "animaLora", "networkTrainUnetOnly"], c)}
          />
        </Row>
        {!isFullFinetune && (
          <Row
            label="训练 block 范围"
            description="限制 Anima DiT block 范围。结束值不包含自身。"
            errors={[
              ...(errorMap.get("backend.animaLora.layerStart") ?? []),
              ...(errorMap.get("backend.animaLora.layerEnd") ?? []),
            ]}
          >
            <div className="flex flex-wrap items-center gap-2">
              <IntInput
                value={v.layerStart ?? null}
                onChange={(n) => set(["backend", "animaLora", "layerStart"], n)}
                placeholder="起始"
                min={0}
              />
              <span className="text-xs text-muted-foreground">到</span>
              <IntInput
                value={v.layerEnd ?? null}
                onChange={(n) => set(["backend", "animaLora", "layerEnd"], n)}
                placeholder="结束"
                min={0}
              />
            </div>
          </Row>
        )}
        {!isFullFinetune && (
          <Row
            label="目标模块"
            description="限制 adapter 注入模块。"
            errors={errorMap.get("backend.animaLora.targetPreset")}
          >
            <EnumSelect
              value={v.targetPreset ?? "all"}
              onChange={(next) =>
                set(["backend", "animaLora", "targetPreset"], next)
              }
              options={TARGET_PRESET_OPTIONS}
            />
          </Row>
        )}
      </Section>

      {/* === 优化器 + 调度 === */}
      <Section title="优化器 / 学习率 / 调度">
        <Row label="优化器类型">
          <EnumSelect
            value={v.optimizerType ?? "AdamW"}
            onChange={(s) => set(["backend", "animaLora", "optimizerType"], s)}
            options={OPTIMIZER_OPTIONS}
          />
        </Row>
        <Row label="学习率调度器">
          <EnumSelect
            value={v.lrScheduler ?? "constant"}
            onChange={(s) => set(["backend", "animaLora", "lrScheduler"], s)}
            options={LR_SCHEDULER_OPTIONS}
          />
        </Row>
        <Row label="学习率" errors={errorMap.get("backend.animaLora.learningRate")}>
          <FloatInput
            value={v.learningRate}
            onChange={(n) => set(["backend", "animaLora", "learningRate"], n)}
            placeholder="5e-5"
            step={1e-6}
          />
        </Row>
        <Row
          label="梯度裁剪"
          description="max_grad_norm。0 表示关闭。"
          errors={errorMap.get("optimizer.maxGradNorm")}
        >
          <FloatInput
            value={optimizer?.maxGradNorm ?? 1.0}
            onChange={(n) => set(["optimizer", "maxGradNorm"], n ?? 1.0)}
            placeholder="1.0"
            step={0.1}
            min={0}
          />
        </Row>
        <Row
          label="LR Warmup 比例"
          description="占总训练步数的比例。0.05 表示 5%。"
          errors={errorMap.get("backend.animaLora.lrWarmupRatio")}
        >
          <FloatInput
            value={v.lrWarmupRatio}
            onChange={(n) => set(["backend", "animaLora", "lrWarmupRatio"], n)}
            placeholder="0.05"
            step={0.01}
            min={0}
            max={1}
          />
        </Row>
        <Row label="最大训练轮数">
          <FloatInput
            value={v.maxTrainEpochs}
            onChange={(n) => set(["backend", "animaLora", "maxTrainEpochs"], n)}
            placeholder="8"
            min={1}
          />
        </Row>
        <Row label="每 N 轮保存">
          <FloatInput
            value={v.saveEveryNEpochs}
            onChange={(n) => set(["backend", "animaLora", "saveEveryNEpochs"], n)}
            placeholder="2"
            min={1}
          />
        </Row>
        <Row label="检查点保存频率" description="optimizer state 保存频率。">
          <FloatInput
            value={v.checkpointingEpochs}
            onChange={(n) => set(["backend", "animaLora", "checkpointingEpochs"], n)}
            placeholder="2"
            min={1}
          />
        </Row>
        <Row label="caption 丢弃率" description="训练时随机丢弃 caption 的概率。">
          <FloatInput
            value={v.captionDropoutRate}
            onChange={(n) => set(["backend", "animaLora", "captionDropoutRate"], n)}
            placeholder="0.1"
            step={0.01}
            min={0}
            max={1}
          />
        </Row>
      </Section>

      {/* === 流匹配采样 === */}
      <Section title="流匹配采样" subtitle="timestep 与损失权重">
        <Row label="时间步采样方式">
          <EnumSelect
            value={v.timestepSampling ?? "sigmoid"}
            onChange={(s) => set(["backend", "animaLora", "timestepSampling"], s)}
            options={TIMESTEP_OPTIONS}
          />
        </Row>
        <Row label="sigmoid 缩放" description="sigmoid timestep 采样缩放。">
          <FloatInput
            value={v.sigmoidScale}
            onChange={(n) => set(["backend", "animaLora", "sigmoidScale"], n)}
            placeholder="1.0"
            step={0.1}
          />
        </Row>
        <Row label="离散流偏移" description="flow matching shift。">
          <FloatInput
            value={v.discreteFlowShift}
            onChange={(n) => set(["backend", "animaLora", "discreteFlowShift"], n)}
            placeholder="1.0"
            step={0.1}
          />
        </Row>
        <Row
          label="加权方案"
          description="rectified-flow 损失加权。"
        >
          <EnumSelect
            value={v.weightingScheme ?? ""}
            onChange={(s) =>
              set(["backend", "animaLora", "weightingScheme"], s || null)
            }
            options={WEIGHTING_SCHEME_OPTIONS}
          />
        </Row>
        <Row
          label="min_snr_gamma"
          description="min_snr_rf 的 γ 阈值。留空不写入。"
        >
          <FloatInput
            value={v.minSnrGamma ?? undefined}
            onChange={(n) =>
              set(["backend", "animaLora", "minSnrGamma"], n ?? null)
            }
            placeholder="（留空 = 关闭)"
            step={0.5}
            min={0}
          />
        </Row>
        <Row
          label="方差减少损失权重"
          description="AsymFlow 方差减少损失权重。留空关闭。"
        >
          <FloatInput
            value={v.vrLossWeight ?? undefined}
            onChange={(n) =>
              set(["backend", "animaLora", "vrLossWeight"], n ?? null)
            }
            placeholder="（关闭)"
            step={0.1}
            min={0}
          />
        </Row>
      </Section>

      {/* === 训练增强（EMA / NaN guard / sample grid） === */}
      <Section
        title="训练增强"
        subtitle="EMA / NaN guard / 采样网格"
      >
        <Row
          label="启用 EMA"
          description="维护可训练参数的指数移动平均权重。"
        >
          <ToggleSwitch
            checked={!!v.ema}
            onCheckedChange={(c) => set(["backend", "animaLora", "ema"], c)}
          />
        </Row>
        {v.ema && (
          <>
            <Row
              label="EMA decay"
              description="EMA 衰减系数。"
            >
              <FloatInput
                value={v.emaDecay ?? 0.9999}
                onChange={(n) =>
                  set(["backend", "animaLora", "emaDecay"], n ?? 0.9999)
                }
                placeholder="0.9999"
                step={0.0001}
                min={0.9}
                max={0.99999}
              />
            </Row>
            <Row
              label="warmup decay"
              description="按训练步数缩放 EMA decay。"
            >
              <ToggleSwitch
                checked={v.emaUseNumUpdates ?? true}
                onCheckedChange={(c) =>
                  set(["backend", "animaLora", "emaUseNumUpdates"], c)
                }
              />
            </Row>
            <Row
              label="自动护栏"
              description="启用 EMA 时固定 compile_inductor_mode = default。"
            >
              <span className="text-xs text-muted-foreground">已启用</span>
            </Row>
          </>
        )}
        <Row
          label="启用 NaN guard"
          description="检查 loss / 梯度中的 NaN 与 Inf。"
        >
          <ToggleSwitch
            checked={!!v.nanGuard}
            onCheckedChange={(c) => set(["backend", "animaLora", "nanGuard"], c)}
          />
        </Row>
        {v.nanGuard && (
          <>
            <Row
              label="自动恢复"
              description="超阈值时降低 LR；可配合 EMA 还原权重。"
            >
              <ToggleSwitch
                checked={!!v.nanGuardRecover}
                onCheckedChange={(c) =>
                  set(["backend", "animaLora", "nanGuardRecover"], c)
                }
              />
            </Row>
            <Row
              label="连续异常上限"
              description="连续异常步数阈值。"
            >
              <IntInput
                value={v.nanGuardMaxConsecutive ?? 5}
                onChange={(n) =>
                  set(["backend", "animaLora", "nanGuardMaxConsecutive"], n ?? 5)
                }
                min={1}
                placeholder="5"
              />
            </Row>
          </>
        )}
        <Row
          label="采样网格图"
          description="采样后生成 contact-sheet PNG。"
        >
          <ToggleSwitch
            checked={!!v.sampleGrid}
            onCheckedChange={(c) => set(["backend", "animaLora", "sampleGrid"], c)}
          />
        </Row>
      </Section>

      <AnimaLoraCacheSection
        value={v}
        set={set}
        lockBadgeFor={lockBadgeFor}
      />
      <AnimaLoraCompileSection value={v} set={set} />
      <AnimaLoraMemorySection value={v} set={set} />
      <AnimaLoraMethodConfig
        method={method}
        value={v}
        set={set}
        errorMap={errorMap}
      />

      <AnimaLoraLockedDefaultsSection
        value={v}
        set={set}
        errorMap={errorMap}
        lockBadgeFor={lockBadgeFor}
      />
      <AnimaLoraTurboSection value={v} set={set} />
    </>
  )
})
