import type { ReactNode } from "react"
import type { ConfigFormValue, Setter } from "../types"
import { EnumSelect, FloatInput, Row, Section, ToggleSwitch } from "../widgets"
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
    <Section title="缓存" subtitle="latent / TE / LLM adapter 输出落盘">
      <Row
        label="缓存潜变量"
        labelBadge={lockBadgeFor("cacheLatents")}
        description="提前用 VAE 编码图片并存盘，大幅提速但占用额外硬盘空间。"
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
        description="将潜变量缓存写到磁盘，释放内存。"
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
        description="预计算 TE 输出并缓存,避免训练时重复 forward。"
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
        description="将文本编码器缓存写到磁盘，释放内存。"
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
        description="每 epoch 使用不同的 caption shuffle 变体,增加数据多样性。"
      >
        <ToggleSwitch
          checked={value.useShuffledCaptionVariants ?? true}
          onCheckedChange={(checked) =>
            set(["backend", "animaLora", "useShuffledCaptionVariants"], checked)
          }
        />
      </Row>
      <Row
        label="静态 token 数"
        labelBadge={lockBadgeFor("staticTokenCount")}
        description="默认 4096 适配 1024² 训练。1536² 训练设 9240 + Bucket 表选 1536。开启 native-flatten 时本字段会被忽略（两条 bucket 路径互斥）。"
      >
        <FloatInput
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
        description="QwenImage VAE memory layout 锁死 64。"
      >
        <FloatInput
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
        description="关闭 VAE 内部 KV 缓存,拖慢 ~30% 但与官方行为一致。"
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
    <Section title="注意力 / torch.compile" subtitle="性能权衡旋钮">
      <Row label="注意力模式" description="不装 flash-attn 时选 torch。">
        <EnumSelect
          value={value.attnMode ?? "flash"}
          onChange={(next) => set(["backend", "animaLora", "attnMode"], next)}
          options={ATTN_OPTIONS}
        />
      </Row>
      <Row
        label="编译模式"
        description="full 与 gradient_checkpointing / blocks_to_swap 互斥，LoraHub 在编译期会校验。"
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
        description="走 4032+4200 双家族 bucket 表 + 不 padding 的 fake-5D 展平，block 栈编译为 2 张图（vs 现状 ~24 张）。在 RTX Pro 6000 / 4090 类卡上提速约 2×。与 staticTokenCount 互斥；切换后需要重做 dataset 缓存。"
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
        description="默认 · 按 native-flatten / staticTokenCount 自动选择（4032+4200 或 4096）。1536 · 9216+9240 双家族，用于 Anima v1.0 native 1536² 训练，12 个 entry 覆盖 ar 0.44–2.25。选 1536 时必须同时启用 native-flatten，或将 staticTokenCount 提至 9240 及以上。"
      >
        <EnumSelect
          value={value.bucketTable ?? ""}
          onChange={(next) =>
            set(["backend", "animaLora", "bucketTable"], next ? next : null)
          }
          options={BUCKET_TABLE_OPTIONS}
        />
      </Row>
      <Row label="自定义 autograd" description="anima_lora 自定义内存优化 autograd · 默认开。">
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
    <Section title="显存 / offload" subtitle="低显存训练相关">
      <Row label="块交换数" description="0=关。graft preset 默认 20。">
        <FloatInput
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
      <Row label="Unsloth offload 检查点" description="低显存预设的杀手锏。">
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
