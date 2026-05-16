import { memo } from "react"
import { Input } from "@/components/ui/input"
import { SAVE_DTYPE_OPTIONS } from "../options"
import type { ErrorMap, ConfigFormValue, Setter } from "../types"
import { EnumSelect, IntInput, PathInput, Row } from "../widgets"

export const OutputFields = memo(function OutputFields({
  value = {},
  set,
  errorMap,
}: {
  value: ConfigFormValue["output"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  return (
    <>
      <Row label="名称" description="作为 LoRA 文件名和任务标识。">
        <Input
          value={v.name ?? ""}
          className="font-mono w-64"
          onChange={(e) => set(["output", "name"], e.target.value)}
          placeholder="my_character"
        />
      </Row>
      <Row label="每 N 回合保存一次">
        <IntInput
          min={1}
          value={v.save_every_n_epochs ?? 1}
          onChange={(n) => set(["output", "save_every_n_epochs"], n ?? 1)}
        />
      </Row>
      <Row label="保存精度" description="fp16 文件更小；bf16 需要 Ampere 及以上。">
        <EnumSelect
          value={v.save_dtype ?? "fp16"}
          onChange={(d) => set(["output", "save_dtype"], d)}
          options={SAVE_DTYPE_OPTIONS}
        />
      </Row>
      <Row label="输出目录" description="默认 <workspace>/output。" errors={errorMap.get("output.output_dir")}>
        <PathInput
          value={v.output_dir ?? ""}
          onChange={(s) => set(["output", "output_dir"], s || null)}
          placeholder="（默认 workspace/output）"
        />
      </Row>
    </>
  )
})
