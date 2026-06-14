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
type LoraAlgorithm = NonNullable<AnimaLoraValue["lora"]>["algorithm"]

const LORA_ALGORITHM_OPTIONS = [
  { value: "ortho", label: "OrthoLoRA · 默认" },
  { value: "lora", label: "LoRA · 经典低秩 ΔW = BA" },
  { value: "tlora", label: "T-LoRA · 时间步动态 rank" },
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

function usesTimestepMaskControls(algorithm: LoraAlgorithm) {
  return (
    algorithm === "lora" ||
    algorithm === "tlora" ||
    algorithm === "ortho" ||
    algorithm === "dora" ||
    algorithm === "dylora" ||
    algorithm === "glora"
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
  const showTimestepMaskControls = usesTimestepMaskControls(algorithm)
  const isTlora = algorithm === "tlora"
  const setAlgorithm = (next: string) => {
    set(["backend", "animaLora", "lora", "algorithm"], next)
    if (next === "tlora") {
      set(["backend", "animaLora", "lora", "useTimestepMask"], true)
    }
  }

  return (
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
          value={algorithm}
          onChange={setAlgorithm}
          options={LORA_ALGORITHM_OPTIONS}
        />
      </Row>
      {algorithm === "lokr" && (
        <Row
          label="LoKr factor"
          description="将 (out, in) 拆为 (a×c, b×d) 的最大块大小。8 是 LyCORIS 默认；越大 W₁ 表达力越强，LoRA 边项越小。"
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
      {algorithm === "boft" && (
        <Row
          label="BOFT 蝴蝶层数"
          description="蝴蝶旋转的级联深度；m ≥ log₂(out_dim) 即可张满 SO(out_dim)。默认 4 已足够。"
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
              ? "T-LoRA 固定启用时间步相关 rank mask；如需关闭，请选择 LoRA。"
              : "T-LoRA 时间步相关 rank mask，可叠在带 LoRA legs 的算法上。"
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
      {showTimestepMaskControls && (
        <Row label="T-mask 最小 rank">
          <IntInput
            value={value.lora?.minRank}
            onChange={(n) =>
              set(["backend", "animaLora", "lora", "minRank"], n ?? 8)
            }
            placeholder="8"
            min={1}
          />
        </Row>
      )}
      {showTimestepMaskControls && (
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
      <Row
        label="channel scaling α"
        description="OrthoLoRA / LoRA 输出 channel-wise 缩放系数，上游默认 0.5。降低可缓解梯度震荡。"
        errors={errorMap.get("backend.animaLora.lora.channelScalingAlpha")}
      >
        <FloatInput
          value={value.lora?.channelScalingAlpha}
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
    <Section title="method=postfix 子配置" defaultOpen>
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
        <FloatInput
          value={value.postfix?.condHiddenDim}
          onChange={(n) =>
            set(["backend", "animaLora", "postfix", "condHiddenDim"], n)
          }
          placeholder="1024"
          min={1}
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
    <Section title="method=chimera 子配置" defaultOpen>
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
    <Section title="method=easycontrol 子配置" defaultOpen>
      <Row
        label="条件门初始值"
        description="softmax gate 初始值 · -10 让 step 0 等同 baseline DiT。"
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
      <Row label="条件 token 数">
        <FloatInput
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
      <Row label="图像丢弃率" description="image-CFG dropout。">
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
    <Section title="method=ip_adapter 子配置" defaultOpen>
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
      <Row label="门学习率" description="门 LR,通常 10× 全局 LR。">
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
    </Section>
  )
}
