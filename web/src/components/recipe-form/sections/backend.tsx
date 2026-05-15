import { memo } from "react"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { BACKEND_OPTIONS } from "../options"
import type { ErrorMap, RecipeFormValue, Setter } from "../types"
import { EnumSelect, PathInput, Row } from "../widgets"

export const BackendFields = memo(function BackendFields({
  value = {},
  set,
  errorMap,
}: {
  value: RecipeFormValue["backend"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  return (
    <>
      <Row
        label="Backend"
        description={
          <>
            Settings &gt; Kohya backend handles the workspace-wide default; this overrides per recipe.
          </>
        }
      >
        <div className="flex items-center gap-2">
          <EnumSelect
            value={v.type ?? "kohya"}
            onChange={(t) => set(["backend", "type"], t)}
            options={BACKEND_OPTIONS}
          />
          {v.type === "diffusers" && (
            <Badge variant="outline" className="rounded-[2px] uppercase text-[10px]">
              v0.3
            </Badge>
          )}
        </div>
      </Row>
      <Row label="sd-scripts path" errors={errorMap.get("backend.sd_scripts_path")}>
        <PathInput
          value={v.sd_scripts_path ?? ""}
          onChange={(s) => set(["backend", "sd_scripts_path"], s || null)}
          placeholder="(use Settings default)"
        />
      </Row>
      <Row label="Python executable" errors={errorMap.get("backend.python_executable")}>
        <PathInput
          value={v.python_executable ?? ""}
          onChange={(s) => set(["backend", "python_executable"], s || null)}
          placeholder="(use Settings default)"
        />
      </Row>
      <Row label="Pin version" description="Optional git ref / tag of sd-scripts to lock to.">
        <Input
          value={v.pin_version ?? ""}
          className="font-mono w-64"
          onChange={(e) => set(["backend", "pin_version"], e.target.value || null)}
          placeholder="e.g. main, sdxl, 0.8.4"
        />
      </Row>
    </>
  )
})
