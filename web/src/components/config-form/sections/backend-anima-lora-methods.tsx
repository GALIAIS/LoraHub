import { Sparkles } from "lucide-react"
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

type AnimaLoraValue = NonNullable<
  NonNullable<ConfigFormValue["backend"]>["animaLora"]
>
type AnimaLoraMethod = NonNullable<AnimaLoraValue["method"]>
type LoraAlgorithm = NonNullable<NonNullable<AnimaLoraValue["lora"]>["algorithm"]>

const LORA_ALGORITHM_OPTIONS = [
  { value: "ortho", label: "OrthoLoRA · 默认" },
  { value: "lora", label: "LoRA · 经典低秩 ΔW = BA" },
  { value: "tlora", label: "T-LoRA · 时间步动态 rank" },
  { value: "asr_tlora", label: "ASR T-LoRA · 逐样本时间步 rank" },
  { value: "dora", label: "DoRA · 方向 / 幅度分离" },
  { value: "dylora", label: "DyLoRA · 训练随机 rank 截断" },
  { value: "glora", label: "GLoRA · LoRA + 每秩对角 gate" },
  { value: "ia3", label: "IA³ · 每输出通道缩放" },
  { value: "lokr", label: "LoKr · Kronecker 积分解" },
  { value: "lokr_factorized", label: "LoKr factorized · 不 materialize ΔW" },
  { value: "loha", label: "LoHA · Hadamard 积, r² 表达力" },
  { value: "diag_oft", label: "Diag-OFT · 块对角正交 (保 hyperspherical)" },
  { value: "boft", label: "BOFT · 蝴蝶级联正交" },
  { value: "vera", label: "VeRA · 冻结随机投影 + 缩放向量" },
  { value: "full", label: "Full · 自由 ΔW Parameter (baseline)" },
  { value: "lycoris_locon", label: "LyCORIS LoCon · LoRA/LoCon 兼容" },
  { value: "lycoris_tlora", label: "LyCORIS T-LoRA · 时间步感知 LoRA" },
  { value: "lycoris_loha", label: "LyCORIS LoHa · Hadamard 分解" },
  { value: "lycoris_lokr", label: "LyCORIS LoKr · Kronecker 分解" },
  { value: "lycoris_lokr_factorized", label: "LyCORIS LoKr factorized" },
  { value: "lycoris_ia3", label: "LyCORIS IA³ · 通道缩放" },
  { value: "lycoris_dylora", label: "LyCORIS DyLoRA · 动态 rank" },
  { value: "lycoris_diag_oft", label: "LyCORIS Diag-OFT · 块正交" },
  { value: "lycoris_boft", label: "LyCORIS BOFT · 蝴蝶正交" },
  { value: "lycoris_glora", label: "LyCORIS GLoRA · gated LoRA" },
  { value: "lycoris_full", label: "LyCORIS Full · Native fine-tuning" },
] as const

function canonicalAlgorithm(algorithm: LoraAlgorithm) {
  switch (algorithm) {
    case "locon":
    case "lycoris_lora":
    case "lycoris_locon":
      return "lora"
    case "lycoris_tlora":
      return "tlora"
    case "lycoris_ia3":
      return "ia3"
    case "lycoris_lokr":
      return "lokr"
    case "lycoris_lokr_factorized":
      return "lokr_factorized"
    case "lycoris_loha":
      return "loha"
    case "lycoris_dylora":
      return "dylora"
    case "lycoris_full":
      return "full"
    case "diag-oft":
    case "lycoris_diag_oft":
    case "lycoris_diag-oft":
      return "diag_oft"
    case "lycoris_boft":
      return "boft"
    case "lycoris_glora":
      return "glora"
    default:
      return algorithm
  }
}

function usesTimestepMaskControls(algorithm: LoraAlgorithm) {
  const canonical = canonicalAlgorithm(algorithm)
  return (
    canonical === "lora" ||
    canonical === "tlora" ||
    canonical === "asr_tlora" ||
    canonical === "ortho" ||
    canonical === "dora" ||
    canonical === "dylora" ||
    canonical === "glora"
  )
}

export function AnimaLoraMethodConfig({
  method,
  value,
  set,
  errorMap,
}: {
  method: AnimaLoraMethod
  value: AnimaLoraValue
  set: Setter
  errorMap: ErrorMap
}) {
  if (method === "lora") {
    return <LoraMethodConfig value={value} set={set} errorMap={errorMap} />
  }
  if (method === "postfix") {
    return <PostfixMethodConfig value={value} set={set} />
  }
  if (method === "chimera") {
    return <ChimeraMethodConfig value={value} set={set} />
  }
  if (method === "easycontrol") {
    return <EasyControlMethodConfig value={value} set={set} />
  }
  if (method === "full_finetune") {
    return null
  }
  return <IpAdapterMethodConfig value={value} set={set} />
}

function LoraMethodConfig({
  value,
  set,
  errorMap,
}: {
  value: AnimaLoraValue
  set: Setter
  errorMap: ErrorMap
}) {
  const algorithm = value.lora?.algorithm ?? "ortho"
  const canonical = canonicalAlgorithm(algorithm)
  const showTimestepMaskControls = usesTimestepMaskControls(algorithm)
  const isTlora = canonical === "tlora" || canonical === "asr_tlora"
  const timestepMaskEnabled =
    isTlora || (showTimestepMaskControls && (value.lora?.useTimestepMask ?? true))
  const setAlgorithm = (next: string) => {
    set(["backend", "animaLora", "lora", "algorithm"], next)
  }

  return (
    <Section
      icon={<Sparkles className="size-3.5" />}
      title="method=lora 子配置"
      subtitle="LoRA 算法与时间步 mask"
    >
      <Row
        label="算法"
        description="选择 adapter 参数化方式。"
      >
        <EnumSelect
          value={algorithm}
          onChange={setAlgorithm}
          options={LORA_ALGORITHM_OPTIONS}
        />
      </Row>
      {["lora", "tlora", "asr_tlora"].includes(canonical) && (
        <Row label="下投影初始化" description="Kaiming 随机初始化，或从基础权重 SVD 初始化。">
          <EnumSelect
            value={value.lora?.downInit ?? "kaiming"}
            onChange={(next) => set(["backend", "animaLora", "lora", "downInit"], next)}
            options={[
              { value: "kaiming", label: "Kaiming · 默认" },
              { value: "weight_svd", label: "基础权重 SVD" },
            ]}
          />
        </Row>
      )}
      {(canonical === "lokr" || canonical === "lokr_factorized") && (
        <Row
          label="LoKr factor"
          description="LoKr Kronecker 分解因子。"
        >
          <IntInput
            value={value.lora?.lokrFactor ?? 8}
            onChange={(n) =>
              set(["backend", "animaLora", "lora", "lokrFactor"], n ?? 8)
            }
            min={1}
            placeholder="8"
          />
        </Row>
      )}
      {canonical === "boft" && (
        <Row
          label="BOFT 蝴蝶层数"
          description="BOFT butterfly factor 数量。"
        >
          <IntInput
            value={value.lora?.boftFactors ?? 4}
            onChange={(n) =>
              set(["backend", "animaLora", "lora", "boftFactors"], n ?? 4)
            }
            min={1}
            placeholder="4"
          />
        </Row>
      )}
      {showTimestepMaskControls && (
        <Row
          label="启用时间步 mask"
          description={
            isTlora
              ? "T-LoRA 强制启用。"
              : "启用 timestep rank mask。"
          }
        >
          <ToggleSwitch
            checked={isTlora ? true : (value.lora?.useTimestepMask ?? true)}
            onCheckedChange={(c) =>
              set(["backend", "animaLora", "lora", "useTimestepMask"], c)
            }
            disabled={isTlora}
          />
        </Row>
      )}
      {timestepMaskEnabled && (
        <Row label="逐样本时间步 mask" description="批内每个样本按自身时间步选择有效 rank。">
          <ToggleSwitch
            checked={value.lora?.perSampleTimestepMask ?? canonical === "asr_tlora"}
            onCheckedChange={(checked) => set(["backend", "animaLora", "lora", "perSampleTimestepMask"], checked)}
          />
        </Row>
      )}
      {timestepMaskEnabled && (
        <Row label="T-mask 最小 rank">
          <IntInput
            value={value.lora?.minRank ?? 16}
            onChange={(n) =>
              set(["backend", "animaLora", "lora", "minRank"], n ?? 16)
            }
            placeholder="16"
            min={1}
          />
        </Row>
      )}
      {timestepMaskEnabled && (
        <Row label="alpha/rank 缩放">
          <FloatInput
            value={value.lora?.alphaRankScale}
            onChange={(n) =>
              set(["backend", "animaLora", "lora", "alphaRankScale"], n)
            }
            placeholder="1.0"
            step={0.1}
          />
        </Row>
      )}
      {["lora", "tlora", "asr_tlora", "ortho", "dora", "dylora", "glora"].includes(
        canonical,
      ) && (
        <Row
          label="channel scaling α"
          description="LoRA 输出 channel-wise 缩放系数。"
          errors={errorMap.get("backend.animaLora.channelScalingAlpha")}
        >
          <FloatInput
            value={value.channelScalingAlpha}
            onChange={(n) =>
              set(
                ["backend", "animaLora", "channelScalingAlpha"],
                n ?? 0.5,
              )
            }
            placeholder="0.5"
            step={0.05}
            min={0}
            max={1}
          />
        </Row>
      )}
    </Section>
  )
}

function PostfixMethodConfig({
  value,
  set,
}: {
  value: AnimaLoraValue
  set: Setter
}) {
  return (
    <Section title="postfix 配置">
      <Row label="模式">
        <EnumSelect
          value={value.postfix?.mode ?? "cond"}
          onChange={(s) => set(["backend", "animaLora", "postfix", "mode"], s)}
          options={[
            { value: "cond", label: "Conditional · 受 caption 调制" },
            { value: "postfix", label: "Postfix · 自由 K×D 张量" },
          ]}
        />
      </Row>
      <Row label="条件隐藏维度">
        <IntInput
          value={value.postfix?.condHiddenDim}
          onChange={(n) =>
            set(["backend", "animaLora", "postfix", "condHiddenDim"], n)
          }
          placeholder="1024"
          min={1}
        />
      </Row>
      <Row label="拼接位置">
        <EnumSelect
          value={value.postfix?.splicePosition ?? "front_of_padding"}
          onChange={(next) => set(["backend", "animaLora", "postfix", "splicePosition"], next)}
          options={[
            { value: "front_of_padding", label: "Padding 前" },
            { value: "after_padding", label: "Padding 后" },
          ]}
        />
      </Row>
      <Row label="正交基">
        <EnumSelect
          value={value.postfix?.orthoBasis ?? "svd_te"}
          onChange={(next) => set(["backend", "animaLora", "postfix", "orthoBasis"], next)}
          options={[
            { value: "svd_te", label: "文本编码器 SVD" },
            { value: "random", label: "随机正交基" },
            { value: "identity", label: "单位基" },
          ]}
        />
      </Row>
      <Row label="lambda 初始值">
        <FloatInput
          value={value.postfix?.lambdaInit}
          onChange={(n) =>
            set(["backend", "animaLora", "postfix", "lambdaInit"], n)
          }
          placeholder="0.3"
          step={0.05}
        />
      </Row>
      <Row label="TE 缓存目录">
        <PathInput
          value={value.postfix?.teCacheDir ?? ""}
          onChange={(s) =>
            set(["backend", "animaLora", "postfix", "teCacheDir"], s || null)
          }
          placeholder="post_image_dataset/lora"
        />
      </Row>
      {(value.postfix?.orthoBasis ?? "svd_te") === "svd_te" && (
        <Row label="SVD 文件数">
          <IntInput min={1} value={value.postfix?.svdNumFiles ?? 1024} onChange={(n) => set(["backend", "animaLora", "postfix", "svdNumFiles"], n ?? 1024)} />
        </Row>
      )}
      <Row label="正交基随机种子">
        <IntInput value={value.postfix?.orthoBasisSeed ?? 0} onChange={(n) => set(["backend", "animaLora", "postfix", "orthoBasisSeed"], n ?? 0)} />
      </Row>
    </Section>
  )
}

function ChimeraMethodConfig({
  value,
  set,
}: {
  value: AnimaLoraValue
  set: Setter
}) {
  return (
    <Section title="chimera 配置">
      <Row label="FEI 特征维度">
        <IntInput min={1} value={value.chimera?.feiFeatureDim ?? 2} onChange={(n) => set(["backend", "animaLora", "chimera", "feiFeatureDim"], n ?? 2)} />
      </Row>
      <Row label="Sigma 特征维度">
        <IntInput min={1} value={value.chimera?.sigmaFeatureDim ?? 16} onChange={(n) => set(["backend", "animaLora", "chimera", "sigmaFeatureDim"], n ?? 16)} />
      </Row>
      <Row label="内容平衡权重">
        <FloatInput
          value={value.chimera?.balanceWContent}
          onChange={(n) =>
            set(["backend", "animaLora", "chimera", "balanceWContent"], n)
          }
          placeholder="2e-7"
          step={1e-8}
        />
      </Row>
      <Row label="频率平衡权重">
        <FloatInput
          value={value.chimera?.balanceWFreq}
          onChange={(n) =>
            set(["backend", "animaLora", "chimera", "balanceWFreq"], n)
          }
          placeholder="5e-7"
          step={1e-8}
        />
      </Row>
      <Row label="平衡损失预热比例">
        <FloatInput
          value={value.chimera?.balanceLossWarmupRatio}
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
  )
}

function EasyControlMethodConfig({
  value,
  set,
}: {
  value: AnimaLoraValue
  set: Setter
}) {
  return (
    <Section title="easycontrol 配置">
      <Row
        label="条件门初始值"
        description="softmax gate 初始值。"
      >
        <FloatInput
          value={value.easycontrol?.bCondInit}
          onChange={(n) =>
            set(["backend", "animaLora", "easycontrol", "bCondInit"], n)
          }
          placeholder="-10"
          step={0.5}
        />
      </Row>
      <Row label="条件缩放">
        <FloatInput value={value.easycontrol?.condScale ?? 1} onChange={(n) => set(["backend", "animaLora", "easycontrol", "condScale"], n ?? 1)} min={0} step={0.1} />
      </Row>
      <Row label="条件 token 数">
        <IntInput
          value={value.easycontrol?.condTokenCount}
          onChange={(n) =>
            set(["backend", "animaLora", "easycontrol", "condTokenCount"], n)
          }
          placeholder="4096"
          min={1}
        />
      </Row>
      <Row label="应用 FFN LoRA">
        <ToggleSwitch
          checked={value.easycontrol?.applyFfnLora ?? true}
          onCheckedChange={(c) =>
            set(["backend", "animaLora", "easycontrol", "applyFfnLora"], c)
          }
        />
      </Row>
      <Row label="图像丢弃率" description="image-CFG dropout 概率。">
        <FloatInput
          value={value.easycontrol?.dropP}
          onChange={(n) =>
            set(["backend", "animaLora", "easycontrol", "dropP"], n)
          }
          placeholder="0.1"
          step={0.01}
          min={0}
          max={1}
        />
      </Row>
      <Row label="条件噪声上限" description="条件图像注入的最大噪声强度。">
        <FloatInput value={value.easycontrol?.condNoiseMax ?? 0.3} onChange={(n) => set(["backend", "animaLora", "easycontrol", "condNoiseMax"], n ?? 0.3)} min={0} step={0.01} />
      </Row>
    </Section>
  )
}

function IpAdapterMethodConfig({
  value,
  set,
}: {
  value: AnimaLoraValue
  set: Setter
}) {
  return (
    <Section title="ip_adapter 配置">
      <Row label="编码器">
        <EnumSelect
          value={value.ipAdapter?.encoder ?? "PE-Core-L14-336"}
          onChange={(s) =>
            set(["backend", "animaLora", "ipAdapter", "encoder"], s)
          }
          options={[
            { value: "PE-Core-L14-336", label: "PE-Core-L14-336 · 默认" },
            { value: "PE-Core-G14-448", label: "PE-Core-G14-448" },
          ]}
        />
      </Row>
      <Row label="Resampler 层数">
        <IntInput min={1} value={value.ipAdapter?.resamplerLayers ?? 2} onChange={(n) => set(["backend", "animaLora", "ipAdapter", "resamplerLayers"], n ?? 2)} />
      </Row>
      <Row label="Resampler 注意力头数">
        <IntInput min={1} value={value.ipAdapter?.resamplerHeads ?? 8} onChange={(n) => set(["backend", "animaLora", "ipAdapter", "resamplerHeads"], n ?? 8)} />
      </Row>
      <Row label="IP 缩放系数">
        <FloatInput
          value={value.ipAdapter?.ipScale}
          onChange={(n) =>
            set(["backend", "animaLora", "ipAdapter", "ipScale"], n)
          }
          placeholder="1.0"
          step={0.1}
        />
      </Row>
      <Row label="门学习率" description="gate learning rate。">
        <FloatInput
          value={value.ipAdapter?.gateLr}
          onChange={(n) =>
            set(["backend", "animaLora", "ipAdapter", "gateLr"], n)
          }
          placeholder="1e-3"
          step={1e-4}
        />
      </Row>
      <Row label="图像丢弃率">
        <FloatInput
          value={value.ipAdapter?.imageDropP}
          onChange={(n) =>
            set(["backend", "animaLora", "ipAdapter", "imageDropP"], n)
          }
          placeholder="0.05"
          step={0.01}
          min={0}
          max={1}
        />
      </Row>
      <Row label="特征缓存写盘" description="将图像编码器特征缓存到磁盘。">
        <ToggleSwitch checked={value.ipAdapter?.featuresCacheToDisk ?? true} onCheckedChange={(checked) => set(["backend", "animaLora", "ipAdapter", "featuresCacheToDisk"], checked)} />
      </Row>
    </Section>
  )
}
