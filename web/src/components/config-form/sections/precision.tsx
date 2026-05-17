import { memo } from "react"
import { PRECISION_OPTIONS } from "../options"
import type { ConfigFormValue, ErrorMap, Setter } from "../types"
import { EnumSelect, Row, ToggleSwitch } from "../widgets"

export const PrecisionFields = memo(function PrecisionFields({
  value,
  set,
}: {
  value: ConfigFormValue
  set: Setter
  errorMap: ErrorMap
}) {
  return (
    <>
      <Row label="精度" description="bf16 需要 Ampere 及以上（RTX 30/40、A100、H100）。">
        <EnumSelect
          value={value.precision ?? "bf16"}
          onChange={(p) => set(["precision"], p)}
          options={PRECISION_OPTIONS}
        />
      </Row>
      <Row
        label="梯度检查点"
        description="以约 20% 吞吐为代价节省显存。8GB 显卡几乎必开。"
      >
        <ToggleSwitch
          checked={value.gradientCheckpointing ?? true}
          onCheckedChange={(b) => set(["gradientCheckpointing"], b)}
        />
      </Row>
      <Row
        label="缓存潜变量"
        description="提前用 VAE 编码图片并存盘，大幅提速但占额外硬盘。"
      >
        <ToggleSwitch
          checked={value.cacheLatents ?? true}
          onCheckedChange={(b) => set(["cacheLatents"], b)}
        />
      </Row>
      <Row
        label="cacheLatentsToDisk"
        description="把潜变量缓存写到磁盘（释放内存）。"
      >
        <ToggleSwitch
          checked={value.cacheLatentsToDisk ?? false}
          onCheckedChange={(b) => set(["cacheLatentsToDisk"], b)}
        />
      </Row>
      <Row
        label="skipCacheCheck"
        description="跳过缓存一致性检查，加速冷启动。"
      >
        <ToggleSwitch
          checked={value.skipCacheCheck ?? false}
          onCheckedChange={(b) => set(["skipCacheCheck"], b)}
        />
      </Row>
      <Row
        label="cacheInfo"
        description="把缓存元信息单独写出（kohya 调试用）。"
      >
        <ToggleSwitch
          checked={value.cacheInfo ?? false}
          onCheckedChange={(b) => set(["cacheInfo"], b)}
        />
      </Row>
      <Row
        label="trainInpainting"
        description="启用 inpainting 训练目标。"
      >
        <ToggleSwitch
          checked={value.trainInpainting ?? false}
          onCheckedChange={(b) => set(["trainInpainting"], b)}
        />
      </Row>
    </>
  )
})
