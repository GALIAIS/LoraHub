/**
 * LoRA 测试台共用的小型表单控件。
 *
 * Field / NumberField / Param — 多个 panel 都要用，抽出来避免重复。
 */
import type { ReactNode } from "react"
import { Input } from "@/components/ui/input"

export function Field({
  label,
  children,
}: {
  label: string
  children: ReactNode
}) {
  return (
    <label className="flex flex-col gap-1.5 text-xs font-medium">
      {label}
      {children}
    </label>
  )
}

export function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (value: number) => void
}) {
  return (
    <Field label={label}>
      <Input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const next = Number(e.target.value)
          if (Number.isFinite(next)) onChange(next)
        }}
      />
    </Field>
  )
}

export function Param({
  label,
  value,
  block = false,
}: {
  label: string
  value: string
  block?: boolean
}) {
  return (
    <div className="rounded-[6px] border border-border/60 bg-muted/25 p-2">
      <div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </div>
      <div className={block ? "mt-1 whitespace-pre-wrap" : "mt-1 break-all font-mono"}>
        {value}
      </div>
    </div>
  )
}
