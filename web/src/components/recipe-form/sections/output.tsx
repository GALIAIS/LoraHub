import { memo } from "react"
import { Input } from "@/components/ui/input"
import { SAVE_DTYPE_OPTIONS } from "../options"
import type { ErrorMap, RecipeFormValue, Setter } from "../types"
import { EnumSelect, IntInput, PathInput, Row } from "../widgets"

export const OutputFields = memo(function OutputFields({
  value = {},
  set,
  errorMap,
}: {
  value: RecipeFormValue["output"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  return (
    <>
      <Row label="Name" description="Used as the LoRA filename and run identifier.">
        <Input
          value={v.name ?? ""}
          className="font-mono w-64"
          onChange={(e) => set(["output", "name"], e.target.value)}
          placeholder="my_character"
        />
      </Row>
      <Row label="Save every N epochs">
        <IntInput
          min={1}
          value={v.save_every_n_epochs ?? 1}
          onChange={(n) => set(["output", "save_every_n_epochs"], n ?? 1)}
        />
      </Row>
      <Row label="Save dtype" description="fp16 keeps file size small; bf16 needs Ampere+.">
        <EnumSelect
          value={v.save_dtype ?? "fp16"}
          onChange={(d) => set(["output", "save_dtype"], d)}
          options={SAVE_DTYPE_OPTIONS}
        />
      </Row>
      <Row label="Output dir" description="Defaults to <workspace>/output." errors={errorMap.get("output.output_dir")}>
        <PathInput
          value={v.output_dir ?? ""}
          onChange={(s) => set(["output", "output_dir"], s || null)}
          placeholder="(default: workspace/output)"
        />
      </Row>
    </>
  )
})
