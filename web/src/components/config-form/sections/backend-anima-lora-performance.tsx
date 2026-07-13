import type { ReactNode } from "react"
import type { ConfigFormValue, Setter } from "../types"
import {
  EnumSelect,
  FloatInput,
  IntInput,
  Row,
  Section,
  ToggleSwitch,
} from "../widgets"
import {
  ATTN_OPTIONS,
  BUCKET_TABLE_OPTIONS,
  COMPILE_INDUCTOR_OPTIONS,
  COMPILE_MODE_OPTIONS,
  MIXED_PRECISION_OPTIONS,
} from "./backend-anima-lora-options"

type AnimaLoraValue = NonNullable<
  NonNullable<ConfigFormValue["backend"]>["animaLora"]
>

export function AnimaLoraCacheSection({
  value,
  set,
  lockBadgeFor,
}: {
  value: AnimaLoraValue
  set: Setter
  lockBadgeFor: (field: string) => ReactNode
}) {
  return (
    <Section title="缓存" subtitle="latent / TE / LLM adapter">
      <Row
        label="缓存潜变量"
        labelBadge={lockBadgeFor("cacheLatents")}
        description="预计算 VAE latent。"
      >
        <ToggleSwitch
          checked={value.cacheLatents ?? true}
          onCheckedChange={(checked) =>
            set(["backend", "animaLora", "cacheLatents"], checked)
          }
        />
      </Row>
      <Row
        label="潜变量缓存写盘"
        labelBadge={lockBadgeFor("cacheLatentsToDisk")}
        description="将 latent 缓存写入磁盘。"
      >
        <ToggleSwitch
          checked={value.cacheLatentsToDisk ?? true}
          onCheckedChange={(checked) =>
            set(["backend", "animaLora", "cacheLatentsToDisk"], checked)
          }
        />
      </Row>
      <Row
        label="缓存文本编码器输出"
        labelBadge={lockBadgeFor("cacheTextEncoderOutputs")}
        description="预计算 text encoder 输出。"
      >
        <ToggleSwitch
          checked={value.cacheTextEncoderOutputs ?? true}
          onCheckedChange={(checked) =>
            set(["backend", "animaLora", "cacheTextEncoderOutputs"], checked)
          }
        />
      </Row>
      <Row
        label="TE 缓存写盘"
        labelBadge={lockBadgeFor("cacheTextEncoderOutputsToDisk")}
        description="将 text encoder 缓存写入磁盘。"
      >
        <ToggleSwitch
          checked={value.cacheTextEncoderOutputsToDisk ?? true}
          onCheckedChange={(checked) =>
            set(
              ["backend", "animaLora", "cacheTextEncoderOutputsToDisk"],
              checked,
            )
          }
        />
      </Row>
      <Row
        label="缓存 LLM Adapter 输出"
        labelBadge={lockBadgeFor("cacheLlmAdapterOutputs")}
        description="预计算 LLM adapter 输出并缓存。"
      >
        <ToggleSwitch
          checked={value.cacheLlmAdapterOutputs ?? true}
          onCheckedChange={(checked) =>
            set(["backend", "animaLora", "cacheLlmAdapterOutputs"], checked)
          }
        />
      </Row>
      <Row
        label="打乱 caption 变体"
        description="训练时使用 caption shuffle 变体。"
      >
        <ToggleSwitch
          checked={value.useShuffledCaptionVariants ?? true}
          onCheckedChange={(checked) =>
            set(["backend", "animaLora", "useShuffledCaptionVariants"], checked)
          }
        />
      </Row>
      <Row label="数据抽样比例" description="仅使用数据集的一部分；留空表示使用全部数据。">
        <FloatInput
          min={0.0001}
          max={1}
          step={0.05}
          value={value.sampleRatio ?? null}
          onChange={(next) => set(["backend", "animaLora", "sampleRatio"], next)}
          placeholder="（全部）"
        />
      </Row>
      <Row
        label="静态 token 数"
        labelBadge={lockBadgeFor("staticTokenCount")}
        description="static_token_count。native-flatten 开启时不读取。"
      >
        <IntInput
          value={value.staticTokenCount}
          onChange={(next) =>
            set(["backend", "animaLora", "staticTokenCount"], next)
          }
          placeholder="4096"
          min={1}
        />
      </Row>
      <Row
        label="VAE 分块大小"
        labelBadge={lockBadgeFor("vaeChunkSize")}
        description="VAE chunk size。"
      >
        <IntInput
          value={value.vaeChunkSize}
          onChange={(next) =>
            set(["backend", "animaLora", "vaeChunkSize"], next)
          }
          placeholder="64"
          min={1}
        />
      </Row>
      <Row
        label="禁用 VAE 缓存"
        labelBadge={lockBadgeFor("vaeDisableCache")}
        description="关闭 VAE 内部缓存。"
      >
        <ToggleSwitch
          checked={value.vaeDisableCache ?? false}
          onCheckedChange={(checked) =>
            set(["backend", "animaLora", "vaeDisableCache"], checked)
          }
        />
      </Row>
    </Section>
  )
}

export function AnimaLoraCompileSection({
  value,
  set,
}: {
  value: AnimaLoraValue
  set: Setter
}) {
  return (
    <Section title="注意力 / torch.compile" subtitle="attention backend 与编译模式">
      <Row label="注意力模式" description="attention backend。">
        <EnumSelect
          value={value.attnMode ?? "flash"}
          onChange={(next) => set(["backend", "animaLora", "attnMode"], next)}
          options={ATTN_OPTIONS}
        />
      </Row>
      {value.attnMode === "xformers" && (
        <Row label="启用 xFormers" description="向训练脚本显式传递 xformers 开关。">
          <ToggleSwitch checked={value.xformers ?? false} onCheckedChange={(checked) => set(["backend", "animaLora", "xformers"], checked)} />
        </Row>
      )}
      <Row label="拆分注意力" description="分块计算注意力以降低峰值显存。">
        <ToggleSwitch checked={value.splitAttn ?? false} onCheckedChange={(checked) => set(["backend", "animaLora", "splitAttn"], checked)} />
      </Row>
      <Row
        label="编译模式"
        description="torch.compile 模式。"
      >
        <EnumSelect
          value={value.compileMode ?? ""}
          onChange={(next) =>
            set(["backend", "animaLora", "compileMode"], next ? next : null)
          }
          options={COMPILE_MODE_OPTIONS}
        />
      </Row>
      <Row label="Inductor 模式">
        <EnumSelect
          value={value.compileInductorMode ?? ""}
          onChange={(next) =>
            set(
              ["backend", "animaLora", "compileInductorMode"],
              next ? next : null,
            )
          }
          options={COMPILE_INDUCTOR_OPTIONS}
        />
      </Row>
      <Row
        label="启用 native-flatten"
        description="启用 native-flatten bucket 路径。与 staticTokenCount 互斥。"
      >
        <ToggleSwitch
          checked={value.enableNativeFlatten ?? false}
          onCheckedChange={(checked) =>
            set(["backend", "animaLora", "enableNativeFlatten"], checked)
          }
        />
      </Row>
      <Row
        label="Bucket 表"
        description="native-flatten / staticTokenCount 使用的 bucket 表。"
      >
        <EnumSelect
          value={value.bucketTable ?? ""}
          onChange={(next) =>
            set(["backend", "animaLora", "bucketTable"], next ? next : null)
          }
          options={BUCKET_TABLE_OPTIONS}
        />
      </Row>
      <Row label="自定义 autograd" description="启用自定义 autograd 路径。">
        <ToggleSwitch
          checked={value.useCustomDownAutograd ?? true}
          onCheckedChange={(checked) =>
            set(["backend", "animaLora", "useCustomDownAutograd"], checked)
          }
        />
      </Row>
    </Section>
  )
}

export function AnimaLoraMemorySection({
  value,
  set,
}: {
  value: AnimaLoraValue
  set: Setter
}) {
  return (
    <Section title="显存 / offload" subtitle="block swap 与 checkpointing">
      <Row label="块交换数" description="blocks_to_swap。0 表示关闭。">
        <IntInput
          value={value.blocksToSwap}
          onChange={(next) =>
            set(["backend", "animaLora", "blocksToSwap"], next)
          }
          placeholder="0"
          min={0}
        />
      </Row>
      <Row label="梯度检查点">
        <ToggleSwitch
          checked={value.gradientCheckpointing ?? false}
          onCheckedChange={(checked) =>
            set(["backend", "animaLora", "gradientCheckpointing"], checked)
          }
        />
      </Row>
      <Row label="Unsloth offload 检查点" description="启用 Unsloth offload checkpointing。">
        <ToggleSwitch
          checked={value.unslothOffloadCheckpointing ?? false}
          onCheckedChange={(checked) =>
            set(
              ["backend", "animaLora", "unslothOffloadCheckpointing"],
              checked,
            )
          }
        />
      </Row>
      <Row label="CPU offload 检查点">
        <ToggleSwitch
          checked={value.cpuOffloadCheckpointing ?? false}
          onCheckedChange={(checked) =>
            set(["backend", "animaLora", "cpuOffloadCheckpointing"], checked)
          }
        />
      </Row>
      <Row label="混合精度">
        <EnumSelect
          value={value.mixedPrecision ?? "bf16"}
          onChange={(next) =>
            set(["backend", "animaLora", "mixedPrecision"], next)
          }
          options={MIXED_PRECISION_OPTIONS}
        />
      </Row>
    </Section>
  )
}
