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
      <Row label="Type">
        <EnumSelect
          value={v.type ?? "lora"}
          onChange={(t) => set(["network", "type"], t)}
          options={NETWORK_TYPE_OPTIONS}
        />
      </Row>
      <Row
        label="Rank"
        description="Higher = more capacity, more VRAM. 32 is a strong default for SDXL characters."
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
        label="Alpha"
        description="Effective LR scaler. Many people set alpha = rank/2."
        errors={errorMap.get("network.alpha")}
      >
        <IntInput
          min={1}
          value={v.alpha ?? 16}
          onChange={(n) => set(["network", "alpha"], n ?? 16)}
        />
      </Row>
      <Row label="Target U-Net" description="Train the U-Net (required for visual change).">
        <Switch
          checked={v.target_unet ?? true}
          onCheckedChange={(b) => set(["network", "target_unet"], b)}
        />
      </Row>
      <Row
        label="Target text encoder"
        description="Train the text encoder too. Slower; helps style/concept generalization."
      >
        <Switch
          checked={v.target_text_encoder ?? false}
          onCheckedChange={(b) => set(["network", "target_text_encoder"], b)}
        />
      </Row>
    </>
  )
})
