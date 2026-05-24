import { memo, useCallback, useMemo } from "react"
import type { BackendId } from "@/lib/api"
import { ARCH_OPTIONS, ARCH_VARIANT_OPTIONS } from "../options"
import { SUPPORTED_ARCHS_BY_BACKEND } from "../backend-meta"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
import { EnumSelect, Row } from "../widgets"
import { ModelPathPicker } from "../widgets-model-picker"

export const BaseModelFields = memo(function BaseModelFields({
  value,
  set,
  errorMap,
  backendType,
}: {
  value: ConfigFormValue["baseModel"]
  set: Setter
  errorMap: ErrorMap
  /**
   * Currently selected backend. Drives the arch dropdown filter so users
   * can't pick an arch the active backend doesn't actually train (kohya
   * doesn't ship a flux2/wan trainer; anima_lora is anima-only). When
   * undefined the editor falls back to showing every arch — used by the
   * read-only preview where filtering would just hide info.
   */
  backendType?: BackendId | undefined
}) {
  // archVariant only applies to SDXL; switching to anything else must clear
  // the variant or backend schema validation rejects the config. Both updates
  // are dispatched in one set() — chaining two calls would race because the
  // form's set captures `value` in a closure and the second call would not
  // see the first call's state change.
  const onArchChange = useCallback(
    (next: string) => {
      if (next !== "sdxl" && value.archVariant) {
        set(["baseModel"], { ...value, arch: next, archVariant: "" })
      } else {
        set(["baseModel", "arch"], next)
      }
    },
    [set, value],
  )

  // Filter the arch list to only what the current backend supports. We
  // keep the full ARCH_OPTIONS order (curated 常用 → 视频 → 实验) so the
  // user-facing sort doesn't shuffle when they switch backends. If
  // backendType is missing (preview / unconfigured) we surface every
  // arch — the schema validator on save will catch any mismatch.
  const filteredArchOptions = useMemo(() => {
    if (!backendType) return ARCH_OPTIONS
    const supported = SUPPORTED_ARCHS_BY_BACKEND[backendType]
    if (!supported) return ARCH_OPTIONS
    return ARCH_OPTIONS.filter((opt) =>
      supported.has(opt.value as never),
    )
  }, [backendType])

  return (
    <>
      <Row label="架构" required errors={errorMap.get("baseModel.arch")}>
        <EnumSelect
          value={value.arch}
          onChange={onArchChange}
          options={filteredArchOptions}
        />
      </Row>
      {value.arch === "sdxl" && (
        <Row
          label="SDXL 子族"
          description="选择对应 finetune 谱系会调整默认学习率与几个 CLI 标志。"
          errors={errorMap.get("baseModel.archVariant")}
        >
          <EnumSelect
            value={value.archVariant ?? ""}
            onChange={(v) => set(["baseModel", "archVariant"], v)}
            options={ARCH_VARIANT_OPTIONS}
          />
        </Row>
      )}
      <Row
        label="基础模型"
        required
        description="基础模型 .safetensors 文件路径。可从 models/ 目录扫描结果下拉选择，或手动输入绝对/相对路径。"
        errors={errorMap.get("baseModel.checkpoint")}
      >
        <ModelPathPicker
          value={value.checkpoint}
          onChange={(v) => set(["baseModel", "checkpoint"], v)}
          placeholder="./models/sdxl_base_1.0.safetensors"
        />
      </Row>
      <Row
        label="VAE 覆盖"
        description="可选。使用自定义 VAE 替代基础模型自带的。"
        errors={errorMap.get("baseModel.vae")}
      >
        <ModelPathPicker
          value={value.vae ?? ""}
          onChange={(v) => set(["baseModel", "vae"], v || null)}
          placeholder="./models/sdxl_vae.safetensors（可选）"
        />
      </Row>
    </>
  )
})
