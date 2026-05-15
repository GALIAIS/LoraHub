import { memo } from "react"
import { ARCH_OPTIONS } from "../options"
import type { ErrorMap, RecipeFormValue, Setter } from "../types"
import { EnumSelect, PathInput, Row } from "../widgets"

export const BaseModelFields = memo(function BaseModelFields({
  value,
  set,
  errorMap,
}: {
  value: RecipeFormValue["base_model"]
  set: Setter
  errorMap: ErrorMap
}) {
  return (
    <>
      <Row label="Architecture" required errors={errorMap.get("base_model.arch")}>
        <EnumSelect
          value={value.arch}
          onChange={(v) => set(["base_model", "arch"], v)}
          options={ARCH_OPTIONS}
        />
      </Row>
      <Row
        label="Checkpoint"
        required
        description="Absolute path to the base model .safetensors file."
        errors={errorMap.get("base_model.checkpoint")}
      >
        <PathInput
          value={value.checkpoint}
          onChange={(v) => set(["base_model", "checkpoint"], v)}
          placeholder="./models/sdxl_base_1.0.safetensors"
        />
      </Row>
      <Row
        label="VAE override"
        description="Optional. Use a custom VAE instead of the one bundled with the checkpoint."
        errors={errorMap.get("base_model.vae")}
      >
        <PathInput
          value={value.vae ?? ""}
          onChange={(v) => set(["base_model", "vae"], v || null)}
          placeholder="./models/sdxl_vae.safetensors (optional)"
        />
      </Row>
    </>
  )
})
