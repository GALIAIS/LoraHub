import { Play } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import type { LaunchOverrides } from "../types"

export function LaunchOverrideDialog({
  open,
  onOpenChange,
  recipeName,
  overrides,
  setOverrides,
  defaults,
  launching,
  errorMessage,
  onSubmit,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  recipeName: string
  overrides: LaunchOverrides
  setOverrides: (next: LaunchOverrides) => void
  defaults: LaunchOverrides
  launching: boolean
  errorMessage: string | null
  onSubmit: () => void
}) {
  const update = <K extends keyof LaunchOverrides>(key: K, value: LaunchOverrides[K]) => {
    setOverrides({ ...overrides, [key]: value })
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[min(calc(100%-2rem),34rem)]">
        <DialogHeader>
          <DialogTitle>启动时覆盖</DialogTitle>
          <DialogDescription>
            仅对本次启动生效，留空则使用配置原值。磁盘上的配置文件不会被修改。
          </DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-1 gap-3">
          <OverrideField
            label="dataset.source"
            placeholder={defaults.datasetSource || "./datasets/my_character"}
            value={overrides.datasetSource}
            onChange={(v) => update("datasetSource", v)}
            description="本次训练使用的图片目录。"
          />
          <OverrideField
            label="output.name"
            placeholder={defaults.outputName || "my_character_v1"}
            value={overrides.outputName}
            onChange={(v) => update("outputName", v)}
            description="生成 LoRA 文件的名称前缀。"
          />
          <div className="grid grid-cols-3 gap-3">
            <OverrideField
              label="batch_size"
              placeholder={defaults.batchSize || "1"}
              value={overrides.batchSize}
              onChange={(v) => update("batchSize", v)}
              type="number"
              min={1}
            />
            <OverrideField
              label="epochs"
              placeholder={defaults.epochs || "10"}
              value={overrides.epochs}
              onChange={(v) => update("epochs", v)}
              type="number"
              min={1}
            />
            <OverrideField
              label="max_steps"
              placeholder={defaults.maxSteps || "(不限)"}
              value={overrides.maxSteps}
              onChange={(v) => update("maxSteps", v)}
              type="number"
              min={1}
            />
          </div>
        </div>

        {errorMessage && (
          <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs font-mono text-destructive whitespace-pre-wrap break-words">
            {errorMessage}
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={launching}>
            取消
          </Button>
          <Button onClick={onSubmit} disabled={launching}>
            <Play className="size-3" />
            {launching ? "启动中…" : `训练 ${recipeName}`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function OverrideField({
  label,
  value,
  onChange,
  placeholder,
  description,
  type = "text",
  min,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  placeholder?: string
  description?: string
  type?: "text" | "number"
  min?: number
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <Label className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </Label>
      <Input
        type={type}
        value={value}
        placeholder={placeholder}
        min={min}
        onChange={(event) => onChange(event.target.value)}
        className="font-mono"
      />
      {description && (
        <span className="text-[11px] text-muted-foreground">{description}</span>
      )}
    </div>
  )
}
