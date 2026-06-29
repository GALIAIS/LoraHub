/**
 * XY 轴编辑器与预设按钮。
 *
 * AxisEditor 选轴字段并按字段类型给输入框（逗号或换行分隔）；
 * PresetButton 是效果矩阵预设的轻量按钮封装。
 */
import type { LoraTestAxisInput } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { AXIS_FIELDS } from "./types"

export function AxisEditor({
  title,
  field,
  values,
  onFieldChange,
  onValuesChange,
}: {
  title: string
  field: LoraTestAxisInput["field"]
  values: string
  onFieldChange: (field: LoraTestAxisInput["field"]) => void
  onValuesChange: (values: string) => void
}) {
  return (
    <div className="rounded-[6px] border border-border/60 bg-muted/20 p-3">
      <div className="mb-2 text-xs font-medium">{title}</div>
      <div className="grid gap-2">
        <Select
          value={field}
          onValueChange={(value) => onFieldChange(value as LoraTestAxisInput["field"])}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {AXIS_FIELDS.map((item) => (
              <SelectItem key={item.value} value={item.value}>
                {item.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {field === "prompt" || field === "negative_prompt" || field === "checkpoint" ? (
          <Textarea
            value={values}
            onChange={(e) => onValuesChange(e.target.value)}
            placeholder={axisPlaceholder(field)}
            className="min-h-24 font-mono text-xs"
          />
        ) : (
          <Input
            value={values}
            onChange={(e) => onValuesChange(e.target.value)}
            placeholder={axisPlaceholder(field)}
            className="font-mono"
          />
        )}
      </div>
    </div>
  )
}

export function PresetButton({
  label,
  onClick,
}: {
  label: string
  onClick: () => void
}) {
  return (
    <Button type="button" variant="outline" size="sm" onClick={onClick}>
      {label}
    </Button>
  )
}

function axisPlaceholder(field: LoraTestAxisInput["field"]): string {
  if (field === "checkpoint") return "每行一个 checkpoint 相对路径"
  if (field === "variant") return "base, lora"
  if (field === "prompt") return "每行一个 prompt"
  if (field === "negative_prompt") return "每行一个 negative；empty 表示空负面词"
  if (field === "size") return "768x1344, 896x1632, 1024x1024"
  return "0.6, 0.8, 1.0, 1.2"
}
