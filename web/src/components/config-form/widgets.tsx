/**
 * Reusable layout and input widgets shared across config form sections.
 *
 * Each widget is memoized so identity is stable across parent rerenders when
 * its props don't change.
 */
import { createContext, memo, useContext, useEffect, useRef, useState } from "react"
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
import { Switch as RawSwitch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"

// Read-only mode is propagated via context so individual widgets don't need to
// thread a `readOnly` prop through every section. The form-level provider sets
// this once; each input checks it and disables itself accordingly.
const ReadOnlyContext = createContext(false)

export const ReadOnlyProvider = ReadOnlyContext.Provider
export const useReadOnly = () => useContext(ReadOnlyContext)

// ============================================================= Section ====

interface SectionProps {
  /** Optional. When omitted the icon slot collapses so titles align flush. */
  icon?: React.ReactNode
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
        {icon && (
          <span className="grid place-items-center size-7 rounded-[4px] bg-muted/50 text-muted-foreground">
            {icon}
          </span>
        )}
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
  /** Optional badge / icon rendered inline next to the label. Used by the
   *  anima_lora 🔒 / ⚠️ markers without forcing every Row to take ReactNode. */
  labelBadge?: React.ReactNode
  description?: React.ReactNode
  required?: boolean
  errors?: string[]
  children: React.ReactNode
  htmlFor?: string
}

export const Row = memo(function Row({
  label,
  labelBadge,
  description,
  required,
  errors,
  children,
  htmlFor,
}: RowProps) {
  return (
    <div className="grid grid-cols-[10rem_1fr] gap-x-4 items-start">
      <Label htmlFor={htmlFor} className="text-xs pt-2 leading-tight">
        <span className="inline-flex items-center gap-1.5 flex-wrap">
          <span>{label}</span>
          {labelBadge}
          {required && <span className="ml-1 text-destructive/80">*</span>}
        </span>
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
  const readOnly = useReadOnly()
  return (
    <Input
      id={id}
      type="number"
      step={step}
      min={min}
      max={max}
      disabled={readOnly}
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
  min?: number
  max?: number
  className?: string
  placeholder?: string
}

export const FloatInput = memo(function FloatInput({
  id,
  value,
  onChange,
  step,
  min,
  max,
  className,
  placeholder,
}: FloatInputProps) {
  const readOnly = useReadOnly()
  return (
    <Input
      id={id}
      type="number"
      step={step ?? "any"}
      min={min}
      max={max}
      disabled={readOnly}
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

// Upper bound for randomly-drawn seeds — matches ComfyUI / rgthree's
// 2^50 cap so a workflow seed stays interchangeable between the two
// tools.
const SEED_MAX = 1_125_899_906_842_624

interface SeedInputProps {
  /** Current seed. -1 is the ComfyUI sentinel: "draw a fresh seed at run time". */
  value: number | null | undefined
  onChange: (v: number) => void
  className?: string
}

/**
 * Numeric input + "🎲" button + "随机" toggle.
 *
 * - Typing a number locks that exact seed for every run.
 * - Clicking 🎲 drops a fresh random integer into the field; the YAML
 *   then records that specific value (great for "I liked the 9th
 *   roll, freeze it").
 * - The randomize pill flips the value to ``-1`` — the launcher draws
 *   a fresh seed at run-start so each queue press differs.
 *
 * The widget treats ``null`` / ``undefined`` as "default to randomise"
 * so importing an old yaml that never set a seed gets the new
 * behaviour automatically.
 */
export const SeedInput = memo(function SeedInput({
  value,
  onChange,
  className,
}: SeedInputProps) {
  const readOnly = useReadOnly()
  const isRandom = value === -1 || value === null || value === undefined
  const display = isRandom ? "" : String(value)

  function rollNew() {
    if (readOnly) return
    onChange(Math.floor(Math.random() * SEED_MAX))
  }
  function setRandomSentinel() {
    if (readOnly) return
    onChange(-1)
  }

  return (
    <div className={cn("flex items-center gap-1.5", className)}>
      <Input
        type="number"
        min={-1}
        max={SEED_MAX}
        step={1}
        disabled={readOnly}
        value={display}
        placeholder={isRandom ? "运行时随机 (-1)" : ""}
        className="font-mono w-56 tabular-nums"
        onChange={(e) => {
          const raw = e.target.value.trim()
          if (raw === "" || raw === "-1") {
            onChange(-1)
            return
          }
          const n = parseInt(raw, 10)
          if (Number.isNaN(n)) return
          onChange(Math.max(-1, Math.min(SEED_MAX, n)))
        }}
      />
      <button
        type="button"
        onClick={rollNew}
        disabled={readOnly}
        title="掷骰子（生成新种子并固定）"
        aria-label="生成新种子"
        className={cn(
          "inline-flex h-9 w-9 items-center justify-center rounded-[4px] border border-border/50",
          "bg-background hover:bg-muted/40 text-base disabled:opacity-50",
        )}
      >
        🎲
      </button>
      <button
        type="button"
        onClick={setRandomSentinel}
        disabled={readOnly}
        title="每次运行时由后端随机抽取（写入 -1）"
        className={cn(
          "h-9 px-2 rounded-[4px] border text-[11px] font-medium",
          isRandom
            ? "border-primary/40 bg-primary/15 text-foreground"
            : "border-border/50 bg-background hover:bg-muted/40 text-muted-foreground",
          readOnly && "opacity-50 cursor-not-allowed",
        )}
      >
        随机
      </button>
    </div>
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
  const readOnly = useReadOnly()
  return (
    <Input
      id={id}
      value={value ?? ""}
      placeholder={placeholder}
      disabled={readOnly}
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
  const readOnly = useReadOnly()
  const [w = 1024, h = 1024] = value ?? []
  return (
    <div className="flex items-center gap-2">
      <Input
        type="number"
        value={String(w)}
        disabled={readOnly}
        className="font-mono w-24"
        onChange={(e) => onChange([parseInt(e.target.value, 10) || 0, h])}
      />
      <span className="text-muted-foreground text-xs">×</span>
      <Input
        type="number"
        value={String(h)}
        disabled={readOnly}
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
  const readOnly = useReadOnly()
  return (
    <Select items={options} value={value ?? ""} onValueChange={(v) => onChange(v ?? "")} disabled={readOnly}>
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

// ====================================================== ToggleSwitch ========

interface ToggleProps {
  checked: boolean
  onCheckedChange: (next: boolean) => void
}

/**
 * Switch wrapper that picks up the read-only context the rest of the widgets
 * use. Sections import this instead of the raw Switch primitive so the
 * preview pane can render every toggle as inert without per-section plumbing.
 */
export const ToggleSwitch = memo(function ToggleSwitch({
  checked,
  onCheckedChange,
}: ToggleProps) {
  const readOnly = useReadOnly()
  return (
    <RawSwitch checked={checked} onCheckedChange={onCheckedChange} disabled={readOnly} />
  )
})

// ============================================================ TextInput ====

interface TextInputProps {
  value: string
  onChange: (v: string) => void
  className?: string
  placeholder?: string
}

/**
 * Plain string input, mirrors PathInput but for non-path values (caption ext,
 * pin_version, output name, …). Picks up read-only mode for free.
 */
export const TextInput = memo(function TextInput({
  value,
  onChange,
  className,
  placeholder,
}: TextInputProps) {
  const readOnly = useReadOnly()
  return (
    <Input
      value={value}
      placeholder={placeholder}
      disabled={readOnly}
      onChange={(e) => onChange(e.target.value)}
      className={cn("font-mono", className)}
    />
  )
})

// ====================================================== KeyValueTextArea ====

interface KeyValueTextAreaProps {
  value: Record<string, string> | undefined
  onChange: (next: Record<string, string>) => void
  placeholder?: string
  rows?: number
}

/**
 * Free-form `key=value` editor backed by a textarea. One pair per line;
 * blank lines and lines without `=` are ignored. The textarea keeps its own
 * draft text so partially-typed lines don't get clobbered as the user types,
 * and only emits a fresh map upstream when parsing succeeds. External value
 * changes (e.g. loading a different config) reset the draft.
 *
 * Used by:
 *   - optimizer.optimizer_args (Record<str, str>)
 *   - backend.diffusion_pipe.model_paths (Record<str, str>)
 *
 * Kept deliberately simple: per the spec we don't need a chip / table UI here.
 */
export const KeyValueTextArea = memo(function KeyValueTextArea({
  value,
  onChange,
  placeholder,
  rows = 4,
}: KeyValueTextAreaProps) {
  const readOnly = useReadOnly()
  // Track the canonical serialization of the upstream value so we only reset
  // the draft when the *external* value really changes (preventing the user's
  // unfinished line from snapping back after a parent rerender).
  const upstream = serialize(value ?? {})
  const lastUpstream = useRef(upstream)
  const [draft, setDraft] = useState(upstream)
  useEffect(() => {
    if (upstream !== lastUpstream.current) {
      lastUpstream.current = upstream
      setDraft(upstream)
    }
  }, [upstream])
  return (
    <textarea
      value={draft}
      placeholder={placeholder}
      disabled={readOnly}
      rows={rows}
      onChange={(e) => {
        const next = e.target.value
        setDraft(next)
        const parsed = parse(next)
        const serialized = serialize(parsed)
        // Only push upstream when the parsed map's canonical form differs
        // from what we last received — keeps redundant onChange calls out
        // of the parent's reducer.
        if (serialized !== lastUpstream.current) {
          lastUpstream.current = serialized
          onChange(parsed)
        }
      }}
      className={cn(
        // Mirror Input visuals; textarea is plain HTML so we restate base
        // classes rather than depend on the Input primitive (which is
        // single-line). Width spans the row so long paths stay legible.
        "font-mono w-full max-w-2xl rounded-[4px] border border-input bg-transparent px-3 py-2 text-sm shadow-xs outline-none",
        "placeholder:text-muted-foreground/60",
        "focus-visible:ring-1 focus-visible:ring-ring",
        "disabled:cursor-not-allowed disabled:opacity-60",
      )}
    />
  )
})

function serialize(map: Record<string, string>): string {
  return Object.entries(map)
    .map(([k, v]) => `${k} = ${v}`)
    .join("\n")
}

function parse(text: string): Record<string, string> {
  const out: Record<string, string> = {}
  for (const raw of text.split("\n")) {
    const line = raw.trim()
    if (!line) continue
    const eq = line.indexOf("=")
    if (eq <= 0) continue
    const key = line.slice(0, eq).trim()
    const val = line.slice(eq + 1).trim()
    if (!key) continue
    out[key] = val
  }
  return out
}
