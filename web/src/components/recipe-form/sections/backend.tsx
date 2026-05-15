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
        label="后端"
        description={
          <>
            「设置 &gt; Kohya 后端」管理工作区级默认值；此处按配方覆盖。
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
      <Row label="sd-scripts 路径" errors={errorMap.get("backend.sd_scripts_path")}>
        <PathInput
          value={v.sd_scripts_path ?? ""}
          onChange={(s) => set(["backend", "sd_scripts_path"], s || null)}
          placeholder="（使用设置中的默认值）"
        />
      </Row>
      <Row label="Python 解释器" errors={errorMap.get("backend.python_executable")}>
        <PathInput
          value={v.python_executable ?? ""}
          onChange={(s) => set(["backend", "python_executable"], s || null)}
          placeholder="（使用设置中的默认值）"
        />
      </Row>
      <Row label="锁定版本" description="可选。锁定 sd-scripts 的 git ref / tag。">
        <Input
          value={v.pin_version ?? ""}
          className="font-mono w-64"
          onChange={(e) => set(["backend", "pin_version"], e.target.value || null)}
          placeholder="例如 main、sdxl、0.8.4"
        />
      </Row>
    </>
  )
})
