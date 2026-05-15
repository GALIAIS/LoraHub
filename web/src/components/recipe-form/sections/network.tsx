import { memo } from "react"
import { Switch } from "@/components/ui/switch"
import { NETWORK_TYPE_OPTIONS } from "../options"
import type { ErrorMap, RecipeFormValue, Setter } from "../types"
import { EnumSelect, IntInput, Row } from "../widgets"

export const NetworkFields = memo(function NetworkFields({
  value = {},
  set,
  errorMap,
}: {
  value: RecipeFormValue["network"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  return (
    <>
      <Row label="网络类型">
        <EnumSelect
          value={v.type ?? "lora"}
          onChange={(t) => set(["network", "type"], t)}
          options={NETWORK_TYPE_OPTIONS}
        />
      </Row>
      <Row
        label="Rank（秩）"
        description="越高容量越大，显存占用也越大。SDXL 角色推荐 32。"
        errors={errorMap.get("network.rank")}
      >
        <IntInput
          min={1}
          max={512}
          value={v.rank ?? 32}
          onChange={(n) => set(["network", "rank"], n ?? 32)}
        />
      </Row>
      <Row
        label="Alpha（缩放）"
        description="实际学习率缩放因子。常见做法是 alpha = rank / 2。"
        errors={errorMap.get("network.alpha")}
      >
        <IntInput
          min={1}
          value={v.alpha ?? 16}
          onChange={(n) => set(["network", "alpha"], n ?? 16)}
        />
      </Row>
      <Row label="训练 U-Net" description="训练 U-Net（视觉变化所必需）。">
        <Switch
          checked={v.target_unet ?? true}
          onCheckedChange={(b) => set(["network", "target_unet"], b)}
        />
      </Row>
      <Row
        label="训练文本编码器"
        description="一并训练文本编码器，速度更慢；有助于风格 / 概念的泛化。"
      >
        <Switch
          checked={v.target_text_encoder ?? false}
          onCheckedChange={(b) => set(["network", "target_text_encoder"], b)}
        />
      </Row>
    </>
  )
})
