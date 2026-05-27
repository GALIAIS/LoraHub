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
import { Lock, Sparkles } from "lucide-react"
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
import { SuggestDialog } from "./suggest-dialog"

/** Look up the lock badge for a field key; returns ``null`` when the field
 *  is unrestricted (most non-base.toml knobs). */
function lockBadgeFor(field: string) {
  const meta = ANIMA_LORA_LOCKS[field]
  return meta ? <LockBadge meta={meta} /> : null
}

const METHOD_OPTIONS = [
  { value: "lora", label: "LoRA · 默认堆叠 (LoRA + OrthoLoRA + T-LoRA)" },
  { value: "postfix", label: "Postfix · 自由参数 / 条件正交后缀" },
  { value: "chimera", label: "ChimeraHydra · 双池路由 MoE" },
  { value: "easycontrol", label: "EasyControl · 自注意力图像条件" },
  {
    value: "ip_adapter",
    label: "IP-Adapter · 图像交叉注意力 (PE-Core encoder)",
  },
] as const

const PRESET_OPTIONS = [
  { value: "default", label: "default · 标准 24 GB" },
  { value: "low_vram", label: "low_vram · 8 GB (grad ckpt + unsloth offload)" },
  { value: "graft", label: "graft · blocks_to_swap = 20" },
  { value: "half", label: "half · 50 % 数据 (实验)" },
  { value: "quarter", label: "quarter · 25 % 数据" },
  { value: "tenth", label: "tenth · 10 % 数据" },
  { value: "debug", label: "debug · 0.1 % 数据 (管线打通)" },
] as const

const TIMESTEP_OPTIONS = [
  { value: "sigmoid", label: "sigmoid · 默认" },
  { value: "uniform", label: "uniform" },
  { value: "logit_normal", label: "logit-normal" },
] as const

const LORA_ALGORITHM_OPTIONS = [
  { value: "ortho", label: "OrthoLoRA · 默认" },
  { value: "lora", label: "LoRA · 经典低秩 ΔW = BA" },
  { value: "dora", label: "DoRA · 方向 / 幅度分离" },
  { value: "dylora", label: "DyLoRA · 训练随机 rank 截断" },
  { value: "glora", label: "GLoRA · LoRA + 每秩对角 gate" },
  { value: "ia3", label: "IA³ · 每输出通道缩放" },
  { value: "lokr", label: "LoKr · Kronecker 积分解" },
  { value: "loha", label: "LoHA · Hadamard 积, r² 表达力" },
  { value: "diag_oft", label: "Diag-OFT · 块对角正交 (保 hyperspherical)" },
  { value: "boft", label: "BOFT · 蝴蝶级联正交" },
  { value: "vera", label: "VeRA · 冻结随机投影 + 缩放向量" },
  { value: "full", label: "Full · 自由 ΔW Parameter (baseline)" },
] as const

const WEIGHTING_SCHEME_OPTIONS = [
  { value: "", label: "关闭 · 等权 RF 损失 (默认)" },
  {
    value: "min_snr_rf",
    label: "Min-SNR-γ · 整流流变体 (需 min_snr_gamma)",
  },
  { value: "sigma_sqrt", label: "Sigma-Sqrt" },
  { value: "logit_normal", label: "Logit-Normal" },
  { value: "mode", label: "Mode" },
  { value: "cosmap", label: "CosMap" },
] as const

const ATTN_OPTIONS = [
  { value: "flash", label: "FlashAttention · 默认 (需 flash-attn)" },
  { value: "torch", label: "Torch SDPA · 无 flash-attn 时备选" },
  { value: "flex", label: "FlexAttention" },
  { value: "sageattn", label: "SageAttention · 仅推理" },
  { value: "xformers", label: "xFormers" },
] as const

const COMPILE_MODE_OPTIONS = [
  { value: "", label: "关闭 · 默认" },
  { value: "blocks", label: "blocks · 分块编译 (可与 grad ckpt 共存)" },
  {
    value: "full",
    label: "full · 全图编译 (与 grad ckpt / blocks_to_swap 互斥)",
  },
] as const

const COMPILE_INDUCTOR_OPTIONS = [
  { value: "", label: "默认" },
  { value: "default", label: "default" },
  { value: "reduce-overhead", label: "reduce-overhead · 推荐" },
  { value: "max-autotune", label: "max-autotune" },
] as const

const MIXED_PRECISION_OPTIONS = [
  { value: "bf16", label: "bf16 · 默认" },
  { value: "fp16", label: "fp16" },
  { value: "fp32", label: "fp32" },
] as const

const OPTIMIZER_OPTIONS = [
  { value: "AdamW", label: "AdamW · 默认" },
  { value: "AdamW8bit", label: "AdamW8bit" },
  { value: "Lion", label: "Lion" },
  { value: "Prodigy", label: "Prodigy" },
  { value: "CAME", label: "CAME · 显存友好二阶矩 (LyCORIS / 风格 LoRA 推荐)" },
] as const

const LR_SCHEDULER_OPTIONS = [
  { value: "constant", label: "constant · 默认" },
  { value: "cosine", label: "cosine" },
  { value: "cosine_with_restarts", label: "cosine_with_restarts" },
  { value: "linear", label: "linear" },
  { value: "polynomial", label: "polynomial" },
] as const

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

      {/* === 缓存 === */}
      <Section title="缓存" subtitle="latent / TE / LLM adapter 输出落盘">
        <Row
          label="缓存潜变量"
          labelBadge={lockBadgeFor("cacheLatents")}
          description="提前用 VAE 编码图片并存盘，大幅提速但占用额外硬盘空间。"
        >
          <ToggleSwitch
            checked={v.cacheLatents ?? true}
            onCheckedChange={(c) => set(["backend", "animaLora", "cacheLatents"], c)}
          />
        </Row>
        <Row
          label="潜变量缓存写盘"
          labelBadge={lockBadgeFor("cacheLatentsToDisk")}
          description="将潜变量缓存写到磁盘，释放内存。"
        >
          <ToggleSwitch
            checked={v.cacheLatentsToDisk ?? true}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "cacheLatentsToDisk"], c)
            }
          />
        </Row>
        <Row
          label="缓存文本编码器输出"
          labelBadge={lockBadgeFor("cacheTextEncoderOutputs")}
          description="预计算 TE 输出并缓存,避免训练时重复 forward。"
        >
          <ToggleSwitch
            checked={v.cacheTextEncoderOutputs ?? true}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "cacheTextEncoderOutputs"], c)
            }
          />
        </Row>
        <Row
          label="TE 缓存写盘"
          labelBadge={lockBadgeFor("cacheTextEncoderOutputsToDisk")}
          description="将文本编码器缓存写到磁盘，释放内存。"
        >
          <ToggleSwitch
            checked={v.cacheTextEncoderOutputsToDisk ?? true}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "cacheTextEncoderOutputsToDisk"], c)
            }
          />
        </Row>
        <Row
          label="缓存 LLM Adapter 输出"
          labelBadge={lockBadgeFor("cacheLlmAdapterOutputs")}
          description="预计算 LLM adapter 输出并缓存。"
        >
          <ToggleSwitch
            checked={v.cacheLlmAdapterOutputs ?? true}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "cacheLlmAdapterOutputs"], c)
            }
          />
        </Row>
        <Row
          label="打乱 caption 变体"
          description="每 epoch 使用不同的 caption shuffle 变体,增加数据多样性。"
        >
          <ToggleSwitch
            checked={v.useShuffledCaptionVariants ?? true}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "useShuffledCaptionVariants"], c)
            }
          />
        </Row>
        <Row
          label="静态 token 数"
          labelBadge={lockBadgeFor("staticTokenCount")}
          description="默认 4096 适配 1024² 训练。1536² 训练设 9240 + Bucket 表选 1536。开启 native-flatten 时本字段会被忽略（两条 bucket 路径互斥）。"
        >
          <FloatInput
            value={v.staticTokenCount}
            onChange={(n) => set(["backend", "animaLora", "staticTokenCount"], n)}
            placeholder="4096"
            min={1}
          />
        </Row>
        <Row
          label="VAE 分块大小"
          labelBadge={lockBadgeFor("vaeChunkSize")}
          description="QwenImage VAE memory layout 锁死 64。"
        >
          <FloatInput
            value={v.vaeChunkSize}
            onChange={(n) => set(["backend", "animaLora", "vaeChunkSize"], n)}
            placeholder="64"
            min={1}
          />
        </Row>
        <Row
          label="禁用 VAE 缓存"
          labelBadge={lockBadgeFor("vaeDisableCache")}
          description="关闭 VAE 内部 KV 缓存,拖慢 ~30% 但与官方行为一致。"
        >
          <ToggleSwitch
            checked={v.vaeDisableCache ?? false}
            onCheckedChange={(c) => set(["backend", "animaLora", "vaeDisableCache"], c)}
          />
        </Row>
      </Section>

      {/* === 注意力 / 编译 === */}
      <Section title="注意力 / torch.compile" subtitle="性能权衡旋钮">
        <Row label="注意力模式" description="不装 flash-attn 时选 torch。">
          <EnumSelect
            value={v.attnMode ?? "flash"}
            onChange={(s) => set(["backend", "animaLora", "attnMode"], s)}
            options={ATTN_OPTIONS}
          />
        </Row>
        <Row
          label="编译模式"
          description="full 与 gradient_checkpointing / blocks_to_swap 互斥，LoraHub 在编译期会校验。"
        >
          <EnumSelect
            value={v.compileMode ?? ""}
            onChange={(s) =>
              set(["backend", "animaLora", "compileMode"], s ? s : null)
            }
            options={COMPILE_MODE_OPTIONS}
          />
        </Row>
        <Row label="Inductor 模式">
          <EnumSelect
            value={v.compileInductorMode ?? ""}
            onChange={(s) =>
              set(
                ["backend", "animaLora", "compileInductorMode"],
                s ? s : null,
              )
            }
            options={COMPILE_INDUCTOR_OPTIONS}
          />
        </Row>
        <Row
          label="启用 native-flatten"
          description="走 4032+4200 双家族 bucket 表 + 不 padding 的 fake-5D 展平，block 栈编译为 2 张图（vs 现状 ~24 张）。在 RTX Pro 6000 / 4090 类卡上提速约 2×。与 staticTokenCount 互斥；切换后需要重做 dataset 缓存。"
        >
          <ToggleSwitch
            checked={v.enableNativeFlatten ?? false}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "enableNativeFlatten"], c)
            }
          />
        </Row>
        <Row
          label="Bucket 表"
          description="默认 · 按 native-flatten / staticTokenCount 自动选择（4032+4200 或 4096）。1536 · 9216+9240 双家族，用于 Anima v1.0 native 1536² 训练，12 个 entry 覆盖 ar 0.44–2.25。选 1536 时必须同时启用 native-flatten，或将 staticTokenCount 提至 9240 及以上。"
        >
          <EnumSelect
            value={v.bucketTable ?? ""}
            onChange={(s) =>
              set(["backend", "animaLora", "bucketTable"], s ? s : null)
            }
            options={[
              { value: "", label: "默认" },
              { value: "1536", label: "1536² native (9216+9240)" },
            ]}
          />
        </Row>
        <Row label="自定义 autograd" description="anima_lora 自定义内存优化 autograd · 默认开。">
          <ToggleSwitch
            checked={v.useCustomDownAutograd ?? true}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "useCustomDownAutograd"], c)
            }
          />
        </Row>
      </Section>

      {/* === 显存 / offload === */}
      <Section title="显存 / offload" subtitle="低显存训练相关">
        <Row label="块交换数" description="0=关。graft preset 默认 20。">
          <FloatInput
            value={v.blocksToSwap}
            onChange={(n) => set(["backend", "animaLora", "blocksToSwap"], n)}
            placeholder="0"
            min={0}
          />
        </Row>
        <Row label="梯度检查点">
          <ToggleSwitch
            checked={v.gradientCheckpointing ?? false}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "gradientCheckpointing"], c)
            }
          />
        </Row>
        <Row label="Unsloth offload 检查点" description="低显存预设的杀手锏。">
          <ToggleSwitch
            checked={v.unslothOffloadCheckpointing ?? false}
            onCheckedChange={(c) =>
              set(
                ["backend", "animaLora", "unslothOffloadCheckpointing"],
                c,
              )
            }
          />
        </Row>
        <Row label="CPU offload 检查点">
          <ToggleSwitch
            checked={v.cpuOffloadCheckpointing ?? false}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "cpuOffloadCheckpointing"], c)
            }
          />
        </Row>
        <Row label="混合精度">
          <EnumSelect
            value={v.mixedPrecision ?? "bf16"}
            onChange={(s) => set(["backend", "animaLora", "mixedPrecision"], s)}
            options={MIXED_PRECISION_OPTIONS}
          />
        </Row>
      </Section>

      {/* === Method = lora 子配置 === */}
      {method === "lora" && (
        <Section
          icon={<Sparkles className="size-3.5" />}
          title="method=lora 子配置"
          subtitle="选择 LoRA 家族的算法变体 + T-LoRA 时间步 mask"
          defaultOpen
        >
          <Row
            label="算法"
            description="决定如何参数化 ΔW。LoRA 为经典低秩；OrthoLoRA 在 LoRA 上叠加 SVD + Cayley 正交；DoRA 引入方向 / 幅度分离；其余原子分解类（IA³ / LoKr / LoHA / Full / OFT / BOFT / GLoRA / VeRA）各自采用独立的 ΔW 参数化。"
          >
            <EnumSelect
              value={v.lora?.algorithm ?? "ortho"}
              onChange={(s) =>
                set(["backend", "animaLora", "lora", "algorithm"], s)
              }
              options={LORA_ALGORITHM_OPTIONS}
            />
          </Row>
          {/* 算法专属字段 — 仅在选中对应算法时显示 */}
          {v.lora?.algorithm === "lokr" && (
            <Row
              label="LoKr factor"
              description="将 (out, in) 拆为 (a×c, b×d) 的最大块大小。8 是 LyCORIS 默认；越大 W₁ 表达力越强，LoRA 边项越小。"
            >
              <IntInput
                value={v.lora?.lokrFactor ?? 8}
                onChange={(n) =>
                  set(["backend", "animaLora", "lora", "lokrFactor"], n ?? 8)
                }
                min={1}
                placeholder="8"
              />
            </Row>
          )}
          {v.lora?.algorithm === "boft" && (
            <Row
              label="BOFT 蝴蝶层数"
              description="蝴蝶旋转的级联深度；m ≥ log₂(out_dim) 即可张满 SO(out_dim)。默认 4 已足够。"
            >
              <IntInput
                value={v.lora?.boftFactors ?? 4}
                onChange={(n) =>
                  set(["backend", "animaLora", "lora", "boftFactors"], n ?? 4)
                }
                min={1}
                placeholder="4"
              />
            </Row>
          )}
          {(v.lora?.algorithm === "lora" ||
            v.lora?.algorithm === "ortho" ||
            v.lora?.algorithm === "dora" ||
            v.lora?.algorithm === "dylora" ||
            v.lora?.algorithm === "glora") && (
            <Row
              label="启用时间步 mask"
              description="T-LoRA — 时间步相关的 rank mask,叠在任何带 LoRA legs 的算法上。"
            >
              <ToggleSwitch
                checked={v.lora?.useTimestepMask ?? true}
                onCheckedChange={(c) =>
                  set(["backend", "animaLora", "lora", "useTimestepMask"], c)
                }
              />
            </Row>
          )}
          {(v.lora?.algorithm === "lora" ||
            v.lora?.algorithm === "ortho" ||
            v.lora?.algorithm === "dora" ||
            v.lora?.algorithm === "dylora" ||
            v.lora?.algorithm === "glora") && (
            <Row label="T-mask 最小 rank">
              <IntInput
                value={v.lora?.minRank}
                onChange={(n) =>
                  set(["backend", "animaLora", "lora", "minRank"], n ?? 8)
                }
                placeholder="8"
                min={1}
              />
            </Row>
          )}
          {(v.lora?.algorithm === "lora" ||
            v.lora?.algorithm === "ortho" ||
            v.lora?.algorithm === "dora" ||
            v.lora?.algorithm === "dylora" ||
            v.lora?.algorithm === "glora") && (
            <Row label="alpha/rank 缩放">
              <FloatInput
                value={v.lora?.alphaRankScale}
                onChange={(n) =>
                  set(["backend", "animaLora", "lora", "alphaRankScale"], n)
                }
                placeholder="1.0"
                step={0.1}
              />
            </Row>
          )}
          <Row
            label="channel scaling α"
            description="OrthoLoRA / LoRA 输出 channel-wise 缩放系数，上游默认 0.5。降低可缓解梯度震荡。"
            errors={errorMap.get("backend.animaLora.lora.channelScalingAlpha")}
          >
            <FloatInput
              value={v.lora?.channelScalingAlpha}
              onChange={(n) =>
                set(
                  ["backend", "animaLora", "lora", "channelScalingAlpha"],
                  n ?? 0.5,
                )
              }
              placeholder="0.5"
              step={0.05}
              min={0}
              max={1}
            />
          </Row>
        </Section>
      )}

      {/* === Method = postfix 子配置 === */}
      {method === "postfix" && (
        <Section title="method=postfix 子配置" defaultOpen>
          <Row label="模式">
            <EnumSelect
              value={v.postfix?.mode ?? "cond"}
              onChange={(s) => set(["backend", "animaLora", "postfix", "mode"], s)}
              options={[
                { value: "cond", label: "Conditional · 受 caption 调制" },
                { value: "postfix", label: "Postfix · 自由 K×D 张量" },
              ]}
            />
          </Row>
          <Row label="条件隐藏维度">
            <FloatInput
              value={v.postfix?.condHiddenDim}
              onChange={(n) =>
                set(["backend", "animaLora", "postfix", "condHiddenDim"], n)
              }
              placeholder="1024"
              min={1}
            />
          </Row>
          <Row label="lambda 初始值">
            <FloatInput
              value={v.postfix?.lambdaInit}
              onChange={(n) =>
                set(["backend", "animaLora", "postfix", "lambdaInit"], n)
              }
              placeholder="0.3"
              step={0.05}
            />
          </Row>
          <Row label="TE 缓存目录">
            <PathInput
              value={v.postfix?.teCacheDir ?? ""}
              onChange={(s) =>
                set(
                  ["backend", "animaLora", "postfix", "teCacheDir"],
                  s || null,
                )
              }
              placeholder="post_image_dataset/lora"
            />
          </Row>
        </Section>
      )}

      {/* === Method = chimera 子配置 === */}
      {method === "chimera" && (
        <Section title="method=chimera 子配置" defaultOpen>
          <Row label="内容平衡权重">
            <FloatInput
              value={v.chimera?.balanceWContent}
              onChange={(n) =>
                set(["backend", "animaLora", "chimera", "balanceWContent"], n)
              }
              placeholder="2e-7"
              step={1e-8}
            />
          </Row>
          <Row label="频率平衡权重">
            <FloatInput
              value={v.chimera?.balanceWFreq}
              onChange={(n) =>
                set(["backend", "animaLora", "chimera", "balanceWFreq"], n)
              }
              placeholder="5e-7"
              step={1e-8}
            />
          </Row>
          <Row label="平衡损失预热比例">
            <FloatInput
              value={v.chimera?.balanceLossWarmupRatio}
              onChange={(n) =>
                set(
                  ["backend", "animaLora", "chimera", "balanceLossWarmupRatio"],
                  n,
                )
              }
              placeholder="0.4"
              step={0.05}
              min={0}
              max={1}
            />
          </Row>
        </Section>
      )}

      {/* === Method = easycontrol 子配置 === */}
      {method === "easycontrol" && (
        <Section title="method=easycontrol 子配置" defaultOpen>
          <Row label="条件门初始值" description="softmax gate 初始值 · -10 让 step 0 等同 baseline DiT。">
            <FloatInput
              value={v.easycontrol?.bCondInit}
              onChange={(n) =>
                set(["backend", "animaLora", "easycontrol", "bCondInit"], n)
              }
              placeholder="-10"
              step={0.5}
            />
          </Row>
          <Row label="条件 token 数">
            <FloatInput
              value={v.easycontrol?.condTokenCount}
              onChange={(n) =>
                set(
                  ["backend", "animaLora", "easycontrol", "condTokenCount"],
                  n,
                )
              }
              placeholder="4096"
              min={1}
            />
          </Row>
          <Row label="应用 FFN LoRA">
            <ToggleSwitch
              checked={v.easycontrol?.applyFfnLora ?? true}
              onCheckedChange={(c) =>
                set(
                  ["backend", "animaLora", "easycontrol", "applyFfnLora"],
                  c,
                )
              }
            />
          </Row>
          <Row label="图像丢弃率" description="image-CFG dropout。">
            <FloatInput
              value={v.easycontrol?.dropP}
              onChange={(n) =>
                set(["backend", "animaLora", "easycontrol", "dropP"], n)
              }
              placeholder="0.1"
              step={0.01}
              min={0}
              max={1}
            />
          </Row>
        </Section>
      )}

      {/* === Method = ip_adapter 子配置 === */}
      {method === "ip_adapter" && (
        <Section title="method=ip_adapter 子配置" defaultOpen>
          <Row label="编码器">
            <EnumSelect
              value={v.ipAdapter?.encoder ?? "PE-Core-L14-336"}
              onChange={(s) =>
                set(["backend", "animaLora", "ipAdapter", "encoder"], s)
              }
              options={[
                { value: "PE-Core-L14-336", label: "PE-Core-L14-336 · 默认" },
                { value: "PE-Core-G14-448", label: "PE-Core-G14-448" },
              ]}
            />
          </Row>
          <Row label="IP 缩放系数">
            <FloatInput
              value={v.ipAdapter?.ipScale}
              onChange={(n) =>
                set(["backend", "animaLora", "ipAdapter", "ipScale"], n)
              }
              placeholder="1.0"
              step={0.1}
            />
          </Row>
          <Row label="门学习率" description="门 LR,通常 10× 全局 LR。">
            <FloatInput
              value={v.ipAdapter?.gateLr}
              onChange={(n) =>
                set(["backend", "animaLora", "ipAdapter", "gateLr"], n)
              }
              placeholder="1e-3"
              step={1e-4}
            />
          </Row>
          <Row label="图像丢弃率">
            <FloatInput
              value={v.ipAdapter?.imageDropP}
              onChange={(n) =>
                set(["backend", "animaLora", "ipAdapter", "imageDropP"], n)
              }
              placeholder="0.05"
              step={0.01}
              min={0}
              max={1}
            />
          </Row>
        </Section>
      )}

      {/* === 上游默认 / 锁定字段 (B5 cut-locks) === */}
      <Section
        icon={<Lock className="size-3.5" />}
        title="上游默认 / 锁定字段"
        subtitle="anima_lora base.toml 写死的字段。带 🔒 是 upstream 无法 override 的;带 ⚠️ 可改但有副作用。"
      >
        <Row
          label="Masked Loss"
          labelBadge={lockBadgeFor("maskedLoss")}
          description="Anima 训练管线硬依赖,关掉是无效操作。"
        >
          <ToggleSwitch
            checked={v.maskedLoss ?? true}
            onCheckedChange={(c) => set(["backend", "animaLora", "maskedLoss"], c)}
          />
        </Row>
        <Row
          label="torch.compile"
          labelBadge={lockBadgeFor("torchCompile")}
          description="static_token_count 性能收益的前提,upstream 训练循环假定开启。"
        >
          <ToggleSwitch
            checked={v.torchCompile ?? true}
            onCheckedChange={(c) => set(["backend", "animaLora", "torchCompile"], c)}
          />
        </Row>
        <Row
          label="跳过缓存校验"
          labelBadge={lockBadgeFor("skipCacheCheck")}
          description="跳过缓存哈希校验,只影响启动速度。"
        >
          <ToggleSwitch
            checked={v.skipCacheCheck ?? true}
            onCheckedChange={(c) => set(["backend", "animaLora", "skipCacheCheck"], c)}
          />
        </Row>
        <Row
          label="DataLoader pin_memory"
          labelBadge={lockBadgeFor("dataloaderPinMemory")}
          description="DataLoader pin_memory 一直开;upstream 没提供反向 flag。"
        >
          <ToggleSwitch
            checked={v.dataloaderPinMemory ?? true}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "dataloaderPinMemory"], c)
            }
          />
        </Row>
        <Row
          label="持久化 DataLoader workers"
          labelBadge={lockBadgeFor("persistentDataLoaderWorkers")}
          description="减少 epoch 边界 stall,但长跑可能泄漏 file handle。"
        >
          <ToggleSwitch
            checked={v.persistentDataLoaderWorkers ?? false}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "persistentDataLoaderWorkers"], c)
            }
          />
        </Row>
        <Row
          label="裁剪交叉注意力 KV"
          labelBadge={lockBadgeFor("trimCrossattnKv")}
          description="启用 KV trimming · 短 caption 加速约 10–15 %。"
        >
          <ToggleSwitch
            checked={v.trimCrossattnKv ?? false}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "trimCrossattnKv"], c)
            }
          />
        </Row>
        <Row
          label="半精度 VAE"
          labelBadge={lockBadgeFor("noHalfVae")}
          description="true 半精度 VAE 省显存,但偶尔在边缘数据集产生 NaN。"
        >
          <ToggleSwitch
            checked={v.noHalfVae ?? false}
            onCheckedChange={(c) => set(["backend", "animaLora", "noHalfVae"], c)}
          />
        </Row>
        <Row
          label="保存精度"
          labelBadge={lockBadgeFor("savePrecision")}
          description="bf16 是 upstream 默认且匹配训练 dtype。"
        >
          <EnumSelect
            value={v.savePrecision ?? "bf16"}
            onChange={(s) => set(["backend", "animaLora", "savePrecision"], s)}
            options={[
              { value: "bf16", label: "bf16 · 默认" },
              { value: "fp16", label: "fp16" },
              { value: "fp32", label: "fp32 · 2× 体积，无质量收益" },
            ]}
          />
        </Row>
        <Row
          label="保存格式"
          labelBadge={lockBadgeFor("saveModelAs")}
          description="Anima 只能加载 safetensors。"
        >
          <EnumSelect
            value={v.saveModelAs ?? "safetensors"}
            onChange={(s) => set(["backend", "animaLora", "saveModelAs"], s)}
            options={[{ value: "safetensors", label: "safetensors · 锁定" }]}
          />
        </Row>
        <Row label="日志记录步数" description="每 N 步记录一次训练日志。">
          <FloatInput
            value={v.logEveryNSteps}
            onChange={(n) => set(["backend", "animaLora", "logEveryNSteps"], n)}
            placeholder="2"
            min={1}
          />
        </Row>

        {/* — Dataset blueprint 字段(写在 [[datasets]] / [general] 段) — */}
        <Row
          label="保留 token 数"
          labelBadge={lockBadgeFor("keepTokens")}
          description="caption shuffle 保前 N 个 tag。改 < 3 trigger word 不再可靠。"
        >
          <FloatInput
            value={v.keepTokens}
            onChange={(n) => set(["backend", "animaLora", "keepTokens"], n)}
            placeholder="3"
            min={0}
          />
        </Row>
        <Row
          label="caption 文件后缀"
          labelBadge={lockBadgeFor("captionExtension")}
          description="caption 文件后缀。改了所有图片会被跳过。"
        >
          <PathInput
            value={v.captionExtension ?? ""}
            onChange={(s) =>
              set(["backend", "animaLora", "captionExtension"], s || ".txt")
            }
            placeholder=".txt"
          />
        </Row>
        <Row
          label="验证集大小"
          labelBadge={lockBadgeFor("validationSplitNum")}
          description="留出验证集大小;0 = 关 CMMD 验证。"
        >
          <FloatInput
            value={v.validationSplitNum}
            onChange={(n) => set(["backend", "animaLora", "validationSplitNum"], n)}
            placeholder="16"
            min={0}
          />
        </Row>
        <Row
          label="多分辨率分桶"
          labelBadge={lockBadgeFor("enableBucket")}
          description="多分辨率 bucketing,Anima static-shape compile 硬约束。"
        >
          <ToggleSwitch
            checked={v.enableBucket ?? true}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "enableBucket"], c)
            }
          />
        </Row>
        <Row
          label="路径匹配模式"
          description="fnmatch 模式;* 全部图,char_a/*|char_b/* OR-合并子文件夹。"
        >
          <PathInput
            value={v.pathPattern ?? ""}
            onChange={(s) =>
              set(["backend", "animaLora", "pathPattern"], s || "*")
            }
            placeholder="*"
          />
        </Row>
      </Section>

      {/* === Turbo 蒸馏(独立路径) === */}
      <Section
        title="Turbo / DMD 蒸馏"
        subtitle="开启后会切换到 distill_turbo.py 路径(忽略 method/preset),输出 4-step LoRA"
      >
        <Row
          label="启用 turbo 蒸馏"
          description="勾选后下方字段才会写入 turbo 子配置;否则该字段为 null,保持普通训练。"
        >
          <ToggleSwitch
            checked={!!v.turbo}
            onCheckedChange={(c) =>
              set(
                ["backend", "animaLora", "turbo"],
                c
                  ? {
                      iterations: 1000,
                      studentRank: 48,
                      studentAlpha: 48,
                      studentSteps: 4,
                      teacherCfg: 4,
                    }
                  : undefined,
              )
            }
          />
        </Row>
        {v.turbo && (
          <>
            <Row label="迭代次数">
              <FloatInput
                value={v.turbo.iterations}
                onChange={(n) =>
                  set(["backend", "animaLora", "turbo", "iterations"], n)
                }
                placeholder="1000"
                min={1}
              />
            </Row>
            <Row label="学生 rank / alpha" description="学生 LoRA 容量。">
              <div className="flex gap-2 items-center">
                <FloatInput
                  value={v.turbo.studentRank}
                  onChange={(n) =>
                    set(["backend", "animaLora", "turbo", "studentRank"], n)
                  }
                  placeholder="48"
                  min={1}
                />
                <span className="text-muted-foreground">/</span>
                <FloatInput
                  value={v.turbo.studentAlpha}
                  onChange={(n) =>
                    set(["backend", "animaLora", "turbo", "studentAlpha"], n)
                  }
                  placeholder="48"
                  min={1}
                />
              </div>
            </Row>
            <Row label="学生推理步数" description="蒸馏后用 --infer_steps N。">
              <FloatInput
                value={v.turbo.studentSteps}
                onChange={(n) =>
                  set(["backend", "animaLora", "turbo", "studentSteps"], n)
                }
                placeholder="4"
                min={1}
              />
            </Row>
            <Row label="教师 CFG" description="教师 CFG,会被烤进学生（推理时 --cfg 1.0）。">
              <FloatInput
                value={v.turbo.teacherCfg}
                onChange={(n) =>
                  set(["backend", "animaLora", "turbo", "teacherCfg"], n)
                }
                placeholder="4"
                step={0.5}
              />
            </Row>
            <Row label="学生学习率">
              <FloatInput
                value={v.turbo.studentLr}
                onChange={(n) =>
                  set(["backend", "animaLora", "turbo", "studentLr"], n)
                }
                placeholder="5e-6"
                step={1e-7}
              />
            </Row>
            <Row label="Fake 学习率">
              <FloatInput
                value={v.turbo.fakeLr}
                onChange={(n) =>
                  set(["backend", "animaLora", "turbo", "fakeLr"], n)
                }
                placeholder="5e-5"
                step={1e-6}
              />
            </Row>
            <Row label="保存间隔" description="每 N 次迭代保存一次。">
              <FloatInput
                value={v.turbo.saveEvery}
                onChange={(n) =>
                  set(["backend", "animaLora", "turbo", "saveEvery"], n)
                }
                placeholder="250"
                min={1}
              />
            </Row>
          </>
        )}
      </Section>
    </>
  )
})
