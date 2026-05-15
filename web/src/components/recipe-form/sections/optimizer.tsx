import { memo } from "react"
import { LR_SCHEDULE_OPTIONS, OPTIMIZER_OPTIONS } from "../options"
import type { ErrorMap, RecipeFormValue, Setter } from "../types"
import { EnumSelect, FloatInput, IntInput, Row } from "../widgets"

export const OptimizerFields = memo(function OptimizerFields({
  value = {},
  set,
  errorMap,
}: {
  value: RecipeFormValue["optimizer"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  const lr = v.lr ?? {}
  return (
    <>
      <Row label="Optimizer">
        <EnumSelect
          value={v.type ?? "adamw8bit"}
          onChange={(t) => set(["optimizer", "type"], t)}
          options={OPTIMIZER_OPTIONS}
        />
      </Row>
      <Row label="LR — U-Net" description="Typical SDXL char LoRA: 1e-4." errors={errorMap.get("optimizer.lr.unet")}>
        <FloatInput
          step={0.00001}
          value={lr.unet ?? 1e-4}
          onChange={(n) => set(["optimizer", "lr", "unet"], n ?? 1e-4)}
        />
      </Row>
      <Row label="LR — text encoder" errors={errorMap.get("optimizer.lr.text_encoder")}>
        <FloatInput
          step={0.00001}
          value={lr.text_encoder ?? 5e-5}
          onChange={(n) => set(["optimizer", "lr", "text_encoder"], n ?? 5e-5)}
        />
      </Row>
      <Row label="Schedule">
        <EnumSelect
          value={v.schedule ?? "cosine_with_restarts"}
          onChange={(s) => set(["optimizer", "schedule"], s)}
          options={LR_SCHEDULE_OPTIONS}
        />
      </Row>
      <Row label="Warmup steps">
        <IntInput
          min={0}
          value={v.warmup_steps ?? 100}
          onChange={(n) => set(["optimizer", "warmup_steps"], n ?? 0)}
        />
      </Row>
    </>
  )
})
