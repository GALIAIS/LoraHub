/**
 * Reusable layout and input widgets shared across recipe form sections.
 *
 * Each widget is memoized so identity is stable across parent rerenders when
 * its props don't change.
 */
import { memo } from "react"
import { ChevronDown } from "lucide-react"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { cn } from "@/lib/utils"

// ============================================================= Section ====

interface SectionProps {
  icon: React.ReactNode
  title: string
  subtitle?: string
  defaultOpen?: boolean
  children: React.ReactNode
}

export const Section = memo(function Section({
  icon,
  title,
  subtitle,
  defaultOpen,
  children,
}: SectionProps) {
  return (
    <details
      open={defaultOpen}
      className="group rounded-[6px] border border-border/60 bg-card/40 shadow-[var(--panel-shadow)] open:bg-card/60 transition-colors"
    >
      <summary className="flex items-center gap-3 px-4 py-3 cursor-pointer select-none list-none [&::-webkit-details-marker]:hidden">
        <span className="grid place-items-center size-7 rounded-[4px] bg-muted/50 text-muted-foreground">
          {icon}
        </span>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold tracking-tight">{title}</div>
          {subtitle && (
            <div className="text-[11px] text-muted-foreground truncate">{subtitle}</div>
          )}
        </div>
        <ChevronDown className="size-4 text-muted-foreground transition-transform group-open:rotate-180" />
      </summary>
      <div className="px-4 pb-4 pt-1 space-y-3.5 border-t border-border/40">
        {children}
      </div>
    </details>
  )
})

// ============================================================= Row =========

interface RowProps {
  label: string
  description?: React.ReactNode
  required?: boolean
  errors?: string[]
  children: React.ReactNode
  htmlFor?: string
}

export const Row = memo(function Row({
  label,
  description,
  required,
  errors,
  children,
  htmlFor,
}: RowProps) {
  return (
    <div className="grid grid-cols-[10rem_1fr] gap-x-4 items-start">
      <Label htmlFor={htmlFor} className="text-xs pt-2 leading-tight">
        {label}
        {required && <span className="ml-1 text-destructive/80">*</span>}
      </Label>
      <div className="min-w-0">
        {children}
        {description && (
          <p className="text-[11px] text-muted-foreground/80 mt-1">{description}</p>
        )}
        {errors && errors.length > 0 && (
          <ul className="mt-1 text-[11px] text-destructive">
            {errors.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
})

// ===================================================== shared widgets =====

interface IntInputProps {
  id?: string
  value: number | undefined | null
  onChange: (v: number | null) => void
  min?: number
  max?: number
  step?: number
  placeholder?: string
  className?: string
}

export const IntInput = memo(function IntInput({
  id,
  value,
  onChange,
  min,
  max,
  step = 1,
  placeholder,
  className,
}: IntInputProps) {
  return (
    <Input
      id={id}
      type="number"
      step={step}
      min={min}
      max={max}
      value={value === null || value === undefined ? "" : String(value)}
      placeholder={placeholder}
      className={cn("font-mono w-32", className)}
      onChange={(e) => {
        const raw = e.target.value
        if (raw === "") {
          onChange(null)
          return
        }
        const n = parseInt(raw, 10)
        onChange(Number.isNaN(n) ? null : n)
      }}
    />
  )
})

interface FloatInputProps {
  id?: string
  value: number | undefined | null
  onChange: (v: number | null) => void
  step?: number
  className?: string
  placeholder?: string
}

export const FloatInput = memo(function FloatInput({
  id,
  value,
  onChange,
  step,
  className,
  placeholder,
}: FloatInputProps) {
  return (
    <Input
      id={id}
      type="number"
      step={step ?? "any"}
      value={value === null || value === undefined ? "" : String(value)}
      placeholder={placeholder}
      className={cn("font-mono w-40", className)}
      onChange={(e) => {
        const raw = e.target.value
        if (raw === "") {
          onChange(null)
          return
        }
        const n = parseFloat(raw)
        onChange(Number.isNaN(n) ? null : n)
      }}
    />
  )
})

interface PathInputProps {
  id?: string
  value: string | undefined | null
  onChange: (v: string) => void
  placeholder?: string
}

export const PathInput = memo(function PathInput({
  id,
  value,
  onChange,
  placeholder,
}: PathInputProps) {
  return (
    <Input
      id={id}
      value={value ?? ""}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
      className="font-mono w-full max-w-2xl"
    />
  )
})

interface ResolutionInputProps {
  value: number[] | undefined
  onChange: (next: [number, number]) => void
}

export const ResolutionInput = memo(function ResolutionInput({
  value,
  onChange,
}: ResolutionInputProps) {
  const [w = 1024, h = 1024] = value ?? []
  return (
    <div className="flex items-center gap-2">
      <Input
        type="number"
        value={String(w)}
        className="font-mono w-24"
        onChange={(e) => onChange([parseInt(e.target.value, 10) || 0, h])}
      />
      <span className="text-muted-foreground text-xs">×</span>
      <Input
        type="number"
        value={String(h)}
        className="font-mono w-24"
        onChange={(e) => onChange([w, parseInt(e.target.value, 10) || 0])}
      />
    </div>
  )
})

interface EnumSelectProps {
  value: string | undefined
  onChange: (v: string) => void
  options: ReadonlyArray<{ value: string; label: string }>
  placeholder?: string
}

export const EnumSelect = memo(function EnumSelect({
  value,
  onChange,
  options,
  placeholder,
}: EnumSelectProps) {
  return (
    <Select value={value ?? ""} onValueChange={(v) => onChange(v ?? "")}>
      <SelectTrigger className="w-64">
        <SelectValue placeholder={placeholder ?? "Select…"} />
      </SelectTrigger>
      <SelectContent>
        {options.map((o) => (
          <SelectItem key={o.value} value={o.value}>
            {o.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
})
