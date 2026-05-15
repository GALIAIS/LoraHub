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
      <Row label="架构" required errors={errorMap.get("base_model.arch")}>
        <EnumSelect
          value={value.arch}
          onChange={(v) => set(["base_model", "arch"], v)}
          options={ARCH_OPTIONS}
        />
      </Row>
      <Row
        label="基础模型"
        required
        description="基础模型 .safetensors 文件的绝对路径。"
        errors={errorMap.get("base_model.checkpoint")}
      >
        <PathInput
          value={value.checkpoint}
          onChange={(v) => set(["base_model", "checkpoint"], v)}
          placeholder="./models/sdxl_base_1.0.safetensors"
        />
      </Row>
      <Row
        label="VAE 覆盖"
        description="可选。使用自定义 VAE 替代基础模型自带的。"
        errors={errorMap.get("base_model.vae")}
      >
        <PathInput
          value={value.vae ?? ""}
          onChange={(v) => set(["base_model", "vae"], v || null)}
          placeholder="./models/sdxl_vae.safetensors（可选）"
        />
      </Row>
    </>
  )
})
