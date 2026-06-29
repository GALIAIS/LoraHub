import { memo } from "react"
import { PRECISION_OPTIONS } from "../options"
import type { ConfigFormValue, ErrorMap, Setter } from "../types"
import { EnumSelect, Row, ToggleSwitch } from "../widgets"

export const PrecisionFields = memo(function PrecisionFields({
  value,
  set,
  backendType,
}: {
  value: ConfigFormValue
  set: Setter
  errorMap: ErrorMap
  backendType?: "kohya" | "diffusion-pipe" | "anima_lora" | "ai_toolkit"
}) {
  const isAiToolkit = backendType === "ai_toolkit"
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
        description="用重算换取更低显存占用。"
      >
        <ToggleSwitch
          checked={value.gradientCheckpointing ?? true}
          onCheckedChange={(b) => set(["gradientCheckpointing"], b)}
        />
      </Row>
      {!isAiToolkit && (
        <Row
          label="缓存潜变量"
          description="提前用 VAE 编码图片并缓存。"
        >
          <ToggleSwitch
            checked={value.cacheLatents ?? true}
            onCheckedChange={(b) => set(["cacheLatents"], b)}
          />
        </Row>
      )}
      <Row
        label="潜变量缓存到磁盘"
        description="把 VAE 潜变量缓存写到磁盘。"
      >
        <ToggleSwitch
          checked={value.cacheLatentsToDisk ?? false}
          onCheckedChange={(b) => set(["cacheLatentsToDisk"], b)}
        />
      </Row>
      {!isAiToolkit && (
        <>
          <Row
            label="跳过缓存检查"
            description="跳过缓存一致性检查。"
          >
            <ToggleSwitch
              checked={value.skipCacheCheck ?? false}
              onCheckedChange={(b) => set(["skipCacheCheck"], b)}
            />
          </Row>
          <Row
            label="缓存信息"
            description="写出缓存元信息。"
          >
            <ToggleSwitch
              checked={value.cacheInfo ?? false}
              onCheckedChange={(b) => set(["cacheInfo"], b)}
            />
          </Row>
          <Row
            label="Inpainting 训练"
            description="启用 inpainting 训练目标。"
          >
            <ToggleSwitch
              checked={value.trainInpainting ?? false}
              onCheckedChange={(b) => set(["trainInpainting"], b)}
            />
          </Row>
        </>
      )}
    </>
  )
})
