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
  set,
  errorMap,
}: {
  value: NonNullable<ConfigFormValue["backend"]>["animaLora"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  const method = v.method ?? "lora"

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
        description="LoRA 为默认堆叠（LoRA + OrthoLoRA + T-LoRA）；其余四种为上游论文级算法。选定后下方将展开其子配置。"
        errors={errorMap.get("backend.animaLora.method")}
      >
        <EnumSelect value={method} onChange={onMethodChange} options={METHOD_OPTIONS} />
      </Row>
      <Row
        label="硬件预设"
        description="对应 anima_lora/configs/presets.toml 中的 section。debug 预设仅取 0.1 % 数据，用于打通管线。"
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
        description="启用 conditioning training: 每张目标图与同名参考图配对(参考图目录在 数据集 → 子集 → 参考图目录 设置),train.py 把参考图加载到 batch['conditioning_images'] 供下游 loss 使用。适合图像编辑 / ControlNet 风格的成对训练。"
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
        title="网络容量"
        subtitle="LoRA rank / alpha"
      >
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
        <Row
          label="只训练 UNet"
          description="anima_lora 默认开启 — text encoder 不训练。"
        >
          <ToggleSwitch
            checked={v.networkTrainUnetOnly ?? true}
            onCheckedChange={(c) => set(["backend", "animaLora", "networkTrainUnetOnly"], c)}
          />
        </Row>
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
          label="LR Warmup 比例"
          description="占总训练步数的比例（0.05 = 5%）。比绝对步数更稳健，跨数据集大小同样表现。"
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
        <Row label="检查点保存频率" description="保存 optimizer state 的频率，用于断点续训。">
          <FloatInput
            value={v.checkpointingEpochs}
            onChange={(n) => set(["backend", "animaLora", "checkpointingEpochs"], n)}
            placeholder="2"
            min={1}
          />
        </Row>
        <Row label="caption 丢弃率" description="训练时随机丢弃 caption 的概率，用于增强泛化。">
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
      <Section title="流匹配采样" subtitle="Anima DiT 的 timestep + 损失权重">
        <Row label="时间步采样方式">
          <EnumSelect
            value={v.timestepSampling ?? "sigmoid"}
            onChange={(s) => set(["backend", "animaLora", "timestepSampling"], s)}
            options={TIMESTEP_OPTIONS}
          />
        </Row>
        <Row label="sigmoid 缩放" description="控制 sigmoid 采样的集中程度。">
          <FloatInput
            value={v.sigmoidScale}
            onChange={(n) => set(["backend", "animaLora", "sigmoidScale"], n)}
            placeholder="1.0"
            step={0.1}
          />
        </Row>
        <Row label="离散流偏移" description="Flow matching 的 shift 参数。">
          <FloatInput
            value={v.discreteFlowShift}
            onChange={(n) => set(["backend", "animaLora", "discreteFlowShift"], n)}
            placeholder="1.0"
            step={0.1}
          />
        </Row>
        <Row
          label="加权方案"
          description="rectified-flow 损失加权;min_snr_rf 是 Min-SNR-γ 整流流变体,需要配合下方 min_snr_gamma 使用。"
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
          description="Min-SNR-γ 整流流加权的 γ 阈值，推荐 5.0；仅当加权方案 = min_snr_rf 时生效。留空则该方案退化为等权。"
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
          description="可选 AsymFlow §5.2 方差减少损失。+40% step 计算成本,留空关闭。"
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
        subtitle="EMA 影子权重 / NaN guard 自愈 / 采样网格 — 全部可选"
      >
        <Row
          label="启用 EMA"
          description="对 LoRA 可训练参数维护一份指数移动平均影子；每个 ckpt 旁会同步写出 {name}_ema.safetensors，推理质量通常优于在线权重。约 2× LoRA 显存占用。"
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
              description="衰减系数。0.9999 适合常规 LoRA · 半衰期约 1 万步；短训（< 2k step）建议降至 0.999 / 0.99。"
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
              description="开启后前几百步用 min(decay, (1+t)/(10+t)) 缩放衰减，避免影子吸入早期噪声。"
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
              description="开启 EMA 时，LoraHub 强制 compile_inductor_mode = default，以避开 cudagraph_trees 与 EMA 的不兼容（否则会在 step 2 抛 RuntimeError）。无需手动设置。"
            >
              <span className="text-xs text-muted-foreground">已启用</span>
            </Row>
          </>
        )}
        <Row
          label="启用 NaN guard"
          description="在反向传播前与梯度裁剪后检查 loss / 梯度的 NaN / Inf。当连续超过阈值时按下方策略恢复或中止训练。"
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
              description="超阈值时：将每个参数组的 LR 减半，并（若 EMA 已启用）用影子权重还原在线参数；关闭则直接中止训练。"
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
              description="连续多少步出现 NaN / Inf 后才触发恢复或中止；偶发尖峰将被吸收。默认 5。"
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
          description="每轮采样后额外合成一张 contact-sheet PNG（单图仍各自落盘），便于一眼看进度。"
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
