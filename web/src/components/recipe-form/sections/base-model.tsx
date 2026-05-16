import { memo, useCallback } from "react"
import { ARCH_OPTIONS, ARCH_VARIANT_OPTIONS } from "../options"
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
  // arch_variant only applies to SDXL; switching to anything else must clear
  // the variant or backend schema validation rejects the recipe. Both updates
  // are dispatched in one set() — chaining two calls would race because the
  // form's set captures `value` in a closure and the second call would not
  // see the first call's state change.
  const onArchChange = useCallback(
    (next: string) => {
      if (next !== "sdxl" && value.arch_variant) {
        set(["base_model"], { ...value, arch: next, arch_variant: "" })
      } else {
        set(["base_model", "arch"], next)
      }
    },
    [set, value],
  )

  return (
    <>
      <Row label="架构" required errors={errorMap.get("base_model.arch")}>
        <EnumSelect
          value={value.arch}
          onChange={onArchChange}
          options={ARCH_OPTIONS}
        />
      </Row>
      {value.arch === "sdxl" && (
        <Row
          label="SDXL 子族"
          description="选择对应 finetune 谱系会调整默认学习率与几个 CLI 标志。"
          errors={errorMap.get("base_model.arch_variant")}
        >
          <EnumSelect
            value={value.arch_variant ?? ""}
            onChange={(v) => set(["base_model", "arch_variant"], v)}
            options={ARCH_VARIANT_OPTIONS}
          />
        </Row>
      )}
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
