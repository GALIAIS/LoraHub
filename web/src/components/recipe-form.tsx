/**
 * Visual recipe editor — covers every field that influences training.
 *
 * Built directly against RecipeConfig (lorahub/core/config/schema.py) so each
 * widget knows its semantics; the form is collapsible per section, validates
 * locally, and surfaces server validation errors next to the offending field.
 */
import { memo, useCallback, useMemo } from "react"
import {
  ChevronDown,
  Cpu,
  FileImage,
  Folder,
  Image,
  Layers,
  PaintBucket,
  Settings2,
  SlidersHorizontal,
  Wand2,
} from "lucide-react"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import type { ValidationFieldError } from "@/lib/api"

// ---------------------------------------------------------------- types -----

export interface RecipeFormValue {
  schema_version?: string
  base_model: {
    arch: string
    checkpoint: string
    vae?: string | null
  }
  dataset: {
    source: string
    resolution: [number, number] | number[]
    bucket?: {
      enabled?: boolean
      min?: number
      max?: number
      step?: number
    }
    caption?: {
      strategy?: string
      ext?: string
      shuffle?: boolean
      drop_rate?: number
    }
    num_repeats?: number
  }
  network?: {
    type?: string
    rank?: number
    alpha?: number
    target_unet?: boolean
    target_text_encoder?: boolean
  }
  optimizer?: {
    type?: string
    lr?: { unet?: number; text_encoder?: number }
    schedule?: string
    warmup_steps?: number
  }
  schedule?: {
    epochs?: number
    batch_size?: number
    grad_accum?: number
    max_steps?: number | null
  }
  precision?: string
  gradient_checkpointing?: boolean
  cache_latents?: boolean
  sampling?: {
    enabled?: boolean
    every_n_epochs?: number
    prompts_file?: string | null
    resolution?: [number, number] | number[]
    seed?: number
  }
  output?: {
    name?: string
    save_every_n_epochs?: number
    save_dtype?: string
    output_dir?: string | null
  }
  backend?: {
    type?: string
    pin_version?: string | null
    sd_scripts_path?: string | null
    python_executable?: string | null
    extra_args?: Record<string, unknown>
  }
  [k: string]: unknown
}

interface RecipeFormProps {
  value: RecipeFormValue
  onChange: (next: RecipeFormValue) => void
  errors?: ValidationFieldError[]
}

// ----------------------------------------------- module-level option lists --
// Hoisted so they're allocated once (rerender-defer-reads, js-cache-storage).

const ARCH_OPTIONS = [
  { value: "sdxl", label: "SDXL (1024)" },
  { value: "sd15", label: "SD 1.5 (512/768)" },
  { value: "flux", label: "Flux" },
  { value: "sd3", label: "SD 3" },
] as const

const NETWORK_TYPE_OPTIONS = [
  { value: "lora", label: "LoRA" },
  { value: "locon", label: "LoCon" },
  { value: "loha", label: "LoHA" },
  { value: "dora", label: "DoRA" },
] as const

const OPTIMIZER_OPTIONS = [
  { value: "adamw8bit", label: "AdamW 8bit (bitsandbytes)" },
  { value: "adamw", label: "AdamW" },
  { value: "lion", label: "Lion" },
  { value: "lion8bit", label: "Lion 8bit" },
  { value: "prodigy", label: "Prodigy" },
  { value: "dadaptation", label: "D-Adaptation" },
] as const

const LR_SCHEDULE_OPTIONS = [
  { value: "cosine_with_restarts", label: "cosine_with_restarts" },
  { value: "cosine", label: "cosine" },
  { value: "linear", label: "linear" },
  { value: "constant", label: "constant" },
  { value: "constant_with_warmup", label: "constant_with_warmup" },
  { value: "polynomial", label: "polynomial" },
] as const

const PRECISION_OPTIONS = [
  { value: "bf16", label: "bf16 (Ampere+)" },
  { value: "fp16", label: "fp16" },
  { value: "fp32", label: "fp32" },
] as const

const SAVE_DTYPE_OPTIONS = [
  { value: "fp16", label: "fp16" },
  { value: "bf16", label: "bf16" },
  { value: "float", label: "float32" },
] as const

const CAPTION_STRATEGY_OPTIONS = [
  { value: "tag_file", label: "tag_file (.txt next to images)" },
  { value: "filename", label: "filename" },
  { value: "none", label: "none" },
] as const

const BACKEND_OPTIONS = [
  { value: "kohya", label: "kohya-ss/sd-scripts" },
  { value: "diffusers", label: "🤗 diffusers (planned)" },
] as const

// ------------------------------------------------------ pure update helpers

function setIn<T extends object>(obj: T, path: ReadonlyArray<string | number>, value: unknown): T {
  if (path.length === 0) return value as T
  const cloned: any = Array.isArray(obj) ? [...(obj as any)] : { ...(obj as any) }
  const [head, ...rest] = path
  cloned[head as any] = setIn(cloned[head as any] ?? (typeof rest[0] === "number" ? [] : {}), rest, value)
  return cloned
}

function buildErrorMap(errors: ValidationFieldError[] | undefined) {
  const m = new Map<string, string[]>()
  if (!errors) return m
  for (const e of errors) {
    const key = e.loc.join(".")
    const arr = m.get(key) ?? []
    arr.push(e.msg)
    m.set(key, arr)
  }
  return m
}

// =========================================================== main form =====

export function RecipeForm({ value, onChange, errors }: RecipeFormProps) {
  const errorMap = useMemo(() => buildErrorMap(errors), [errors])

  // Stable updater factory to keep child callbacks identity-stable when
  // value/onChange themselves are stable.
  const set = useCallback(
    (path: ReadonlyArray<string | number>, next: unknown) => {
      onChange(setIn(value, path, next))
    },
    [value, onChange],
  )

  return (
    <div className="space-y-3">
      <Section
        icon={<Cpu className="size-3.5" />}
        title="Base model"
        subtitle="Architecture and checkpoint to fine-tune"
        defaultOpen
      >
        <BaseModelFields value={value.base_model} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<FileImage className="size-3.5" />}
        title="Dataset"
        subtitle="Where the training images live and how they're loaded"
        defaultOpen
      >
        <DatasetFields value={value.dataset} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<Layers className="size-3.5" />}
        title="Network"
        subtitle="LoRA structure: rank, alpha, target modules"
      >
        <NetworkFields value={value.network} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<SlidersHorizontal className="size-3.5" />}
        title="Optimizer & learning rate"
        subtitle="How weights move during training"
      >
        <OptimizerFields value={value.optimizer} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<Settings2 className="size-3.5" />}
        title="Schedule"
        subtitle="Epochs, batch size, gradient accumulation"
        defaultOpen
      >
        <ScheduleFields value={value.schedule} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<PaintBucket className="size-3.5" />}
        title="Precision & memory"
        subtitle="Mixed precision, gradient checkpointing, latent cache"
      >
        <PrecisionFields value={value} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<Image className="size-3.5" />}
        title="Sampling"
        subtitle="Generate preview images during training"
      >
        <SamplingFields value={value.sampling} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<Folder className="size-3.5" />}
        title="Output"
        subtitle="Filename, save cadence, dtype"
      >
        <OutputFields value={value.output} set={set} errorMap={errorMap} />
      </Section>

      <Section
        icon={<Wand2 className="size-3.5" />}
        title="Backend"
        subtitle="Override the kohya checkout / Python on a per-recipe basis"
      >
        <BackendFields value={value.backend} set={set} errorMap={errorMap} />
      </Section>
    </div>
  )
}

// ============================================================= Section ====

type ErrorMap = Map<string, string[]>
type Setter = (path: ReadonlyArray<string | number>, next: unknown) => void

interface SectionProps {
  icon: React.ReactNode
  title: string
  subtitle?: string
  defaultOpen?: boolean
  children: React.ReactNode
}

const Section = memo(function Section({
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

const Row = memo(function Row({
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

const IntInput = memo(function IntInput({
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

const FloatInput = memo(function FloatInput({
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

const PathInput = memo(function PathInput({ id, value, onChange, placeholder }: PathInputProps) {
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

const ResolutionInput = memo(function ResolutionInput({ value, onChange }: ResolutionInputProps) {
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

const EnumSelect = memo(function EnumSelect({
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

// ============================================================= Sections ===

const BaseModelFields = memo(function BaseModelFields({
  value,
  set,
  errorMap,
}: {
  value: RecipeFormValue["base_model"]
  set: Setter
  errorMap: ErrorMap
}) {
  return (
    <>
      <Row label="Architecture" required errors={errorMap.get("base_model.arch")}>
        <EnumSelect
          value={value.arch}
          onChange={(v) => set(["base_model", "arch"], v)}
          options={ARCH_OPTIONS}
        />
      </Row>
      <Row
        label="Checkpoint"
        required
        description="Absolute path to the base model .safetensors file."
        errors={errorMap.get("base_model.checkpoint")}
      >
        <PathInput
          value={value.checkpoint}
          onChange={(v) => set(["base_model", "checkpoint"], v)}
          placeholder="./models/sdxl_base_1.0.safetensors"
        />
      </Row>
      <Row
        label="VAE override"
        description="Optional. Use a custom VAE instead of the one bundled with the checkpoint."
        errors={errorMap.get("base_model.vae")}
      >
        <PathInput
          value={value.vae ?? ""}
          onChange={(v) => set(["base_model", "vae"], v || null)}
          placeholder="./models/sdxl_vae.safetensors (optional)"
        />
      </Row>
    </>
  )
})

const DatasetFields = memo(function DatasetFields({
  value,
  set,
  errorMap,
}: {
  value: RecipeFormValue["dataset"]
  set: Setter
  errorMap: ErrorMap
}) {
  const bucket = value.bucket ?? {}
  const caption = value.caption ?? {}
  return (
    <>
      <Row label="Source" required errors={errorMap.get("dataset.source")}>
        <PathInput
          value={value.source}
          onChange={(v) => set(["dataset", "source"], v)}
          placeholder="./datasets/my_character"
        />
      </Row>
      <Row label="Resolution" errors={errorMap.get("dataset.resolution")}>
        <ResolutionInput
          value={value.resolution}
          onChange={(v) => set(["dataset", "resolution"], v)}
        />
      </Row>
      <Row label="Repeats" description="How many times each image is seen per epoch.">
        <IntInput
          min={1}
          value={value.num_repeats}
          onChange={(v) => set(["dataset", "num_repeats"], v ?? 1)}
        />
      </Row>

      <div className="rounded-[4px] border border-border/40 bg-muted/20 p-3 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-muted-foreground uppercase tracking-[0.18em]">
            Bucket
          </span>
          <Switch
            checked={bucket.enabled ?? true}
            onCheckedChange={(v) => set(["dataset", "bucket", "enabled"], v)}
          />
        </div>
        {bucket.enabled !== false && (
          <div className="grid grid-cols-3 gap-3">
            <div>
              <Label className="text-[11px] text-muted-foreground">min</Label>
              <IntInput
                min={64}
                value={bucket.min ?? 256}
                onChange={(v) => set(["dataset", "bucket", "min"], v ?? 256)}
              />
            </div>
            <div>
              <Label className="text-[11px] text-muted-foreground">max</Label>
              <IntInput
                min={64}
                value={bucket.max ?? 2048}
                onChange={(v) => set(["dataset", "bucket", "max"], v ?? 2048)}
              />
            </div>
            <div>
              <Label className="text-[11px] text-muted-foreground">step</Label>
              <IntInput
                min={8}
                value={bucket.step ?? 64}
                onChange={(v) => set(["dataset", "bucket", "step"], v ?? 64)}
              />
            </div>
          </div>
        )}
      </div>

      <div className="rounded-[4px] border border-border/40 bg-muted/20 p-3 space-y-3">
        <div className="text-xs font-semibold text-muted-foreground uppercase tracking-[0.18em]">
          Caption
        </div>
        <Row label="Strategy">
          <EnumSelect
            value={caption.strategy ?? "tag_file"}
            onChange={(v) => set(["dataset", "caption", "strategy"], v)}
            options={CAPTION_STRATEGY_OPTIONS}
          />
        </Row>
        <Row label="Extension" description=".txt is the kohya default.">
          <Input
            value={caption.ext ?? ".txt"}
            className="font-mono w-32"
            onChange={(e) => set(["dataset", "caption", "ext"], e.target.value)}
          />
        </Row>
        <Row label="Shuffle tags" description="Randomize comma-separated tags each step.">
          <Switch
            checked={caption.shuffle ?? true}
            onCheckedChange={(v) => set(["dataset", "caption", "shuffle"], v)}
          />
        </Row>
        <Row
          label="Drop rate"
          description="Probability (0-1) that a tag is dropped per step."
          errors={errorMap.get("dataset.caption.drop_rate")}
        >
          <FloatInput
            step={0.05}
            value={caption.drop_rate ?? 0}
            onChange={(v) => set(["dataset", "caption", "drop_rate"], v ?? 0)}
          />
        </Row>
      </div>
    </>
  )
})

const NetworkFields = memo(function NetworkFields({
  value = {},
  set,
  errorMap,
}: {
  value: RecipeFormValue["network"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  return (
    <>
      <Row label="Type">
        <EnumSelect
          value={v.type ?? "lora"}
          onChange={(t) => set(["network", "type"], t)}
          options={NETWORK_TYPE_OPTIONS}
        />
      </Row>
      <Row
        label="Rank"
        description="Higher = more capacity, more VRAM. 32 is a strong default for SDXL characters."
        errors={errorMap.get("network.rank")}
      >
        <IntInput
          min={1}
          max={512}
          value={v.rank ?? 32}
          onChange={(n) => set(["network", "rank"], n ?? 32)}
        />
      </Row>
      <Row
        label="Alpha"
        description="Effective LR scaler. Many people set alpha = rank/2."
        errors={errorMap.get("network.alpha")}
      >
        <IntInput
          min={1}
          value={v.alpha ?? 16}
          onChange={(n) => set(["network", "alpha"], n ?? 16)}
        />
      </Row>
      <Row label="Target U-Net" description="Train the U-Net (required for visual change).">
        <Switch
          checked={v.target_unet ?? true}
          onCheckedChange={(b) => set(["network", "target_unet"], b)}
        />
      </Row>
      <Row
        label="Target text encoder"
        description="Train the text encoder too. Slower; helps style/concept generalization."
      >
        <Switch
          checked={v.target_text_encoder ?? false}
          onCheckedChange={(b) => set(["network", "target_text_encoder"], b)}
        />
      </Row>
    </>
  )
})

const OptimizerFields = memo(function OptimizerFields({
  value = {},
  set,
  errorMap,
}: {
  value: RecipeFormValue["optimizer"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  const lr = v.lr ?? {}
  return (
    <>
      <Row label="Optimizer">
        <EnumSelect
          value={v.type ?? "adamw8bit"}
          onChange={(t) => set(["optimizer", "type"], t)}
          options={OPTIMIZER_OPTIONS}
        />
      </Row>
      <Row label="LR — U-Net" description="Typical SDXL char LoRA: 1e-4." errors={errorMap.get("optimizer.lr.unet")}>
        <FloatInput
          step={0.00001}
          value={lr.unet ?? 1e-4}
          onChange={(n) => set(["optimizer", "lr", "unet"], n ?? 1e-4)}
        />
      </Row>
      <Row label="LR — text encoder" errors={errorMap.get("optimizer.lr.text_encoder")}>
        <FloatInput
          step={0.00001}
          value={lr.text_encoder ?? 5e-5}
          onChange={(n) => set(["optimizer", "lr", "text_encoder"], n ?? 5e-5)}
        />
      </Row>
      <Row label="Schedule">
        <EnumSelect
          value={v.schedule ?? "cosine_with_restarts"}
          onChange={(s) => set(["optimizer", "schedule"], s)}
          options={LR_SCHEDULE_OPTIONS}
        />
      </Row>
      <Row label="Warmup steps">
        <IntInput
          min={0}
          value={v.warmup_steps ?? 100}
          onChange={(n) => set(["optimizer", "warmup_steps"], n ?? 0)}
        />
      </Row>
    </>
  )
})

const ScheduleFields = memo(function ScheduleFields({
  value = {},
  set,
  errorMap,
}: {
  value: RecipeFormValue["schedule"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  return (
    <>
      <Row label="Epochs" errors={errorMap.get("schedule.epochs")}>
        <IntInput min={1} value={v.epochs ?? 10} onChange={(n) => set(["schedule", "epochs"], n ?? 1)} />
      </Row>
      <Row label="Batch size" errors={errorMap.get("schedule.batch_size")}>
        <IntInput min={1} value={v.batch_size ?? 1} onChange={(n) => set(["schedule", "batch_size"], n ?? 1)} />
      </Row>
      <Row label="Grad accumulation" description="Effective batch = batch × grad_accum.">
        <IntInput min={1} value={v.grad_accum ?? 2} onChange={(n) => set(["schedule", "grad_accum"], n ?? 1)} />
      </Row>
      <Row label="Max steps" description="Optional hard cap; leave empty to run all epochs.">
        <IntInput
          min={1}
          value={v.max_steps ?? null}
          onChange={(n) => set(["schedule", "max_steps"], n)}
          placeholder="(unbounded)"
        />
      </Row>
    </>
  )
})

const PrecisionFields = memo(function PrecisionFields({
  value,
  set,
}: {
  value: RecipeFormValue
  set: Setter
  errorMap: ErrorMap
}) {
  return (
    <>
      <Row label="Precision" description="bf16 needs Ampere+ (RTX 30/40, A100, H100).">
        <EnumSelect
          value={value.precision ?? "bf16"}
          onChange={(p) => set(["precision"], p)}
          options={PRECISION_OPTIONS}
        />
      </Row>
      <Row
        label="Gradient checkpointing"
        description="Cuts VRAM at the cost of ~20% throughput. Almost always on for 8GB cards."
      >
        <Switch
          checked={value.gradient_checkpointing ?? true}
          onCheckedChange={(b) => set(["gradient_checkpointing"], b)}
        />
      </Row>
      <Row
        label="Cache latents"
        description="Pre-encode images via VAE, store on disk. Big speedup; needs disk space."
      >
        <Switch
          checked={value.cache_latents ?? true}
          onCheckedChange={(b) => set(["cache_latents"], b)}
        />
      </Row>
    </>
  )
})

const SamplingFields = memo(function SamplingFields({
  value = {},
  set,
  errorMap,
}: {
  value: RecipeFormValue["sampling"]
  set: Setter
  errorMap: ErrorMap
}) {
  const v = value ?? {}
  const enabled = v.enabled ?? true
  return (
    <>
      <Row label="Enabled" description="Render preview images during training.">
        <Switch checked={enabled} onCheckedChange={(b) => set(["sampling", "enabled"], b)} />
      </Row>
      {enabled && (
        <>
          <Row label="Every N epochs">
            <IntInput
              min={1}
              value={v.every_n_epochs ?? 1}
              onChange={(n) => set(["sampling", "every_n_epochs"], n ?? 1)}
            />
          </Row>
          <Row label="Prompts file" description="Plain text file, one prompt per line." errors={errorMap.get("sampling.prompts_file")}>
            <PathInput
              value={v.prompts_file ?? ""}
              onChange={(s) => set(["sampling", "prompts_file"], s || null)}
              placeholder="./prompts.txt"
            />
          </Row>
          <Row label="Resolution">
            <ResolutionInput
              value={v.resolution}
              onChange={(r) => set(["sampling", "resolution"], r)}
            />
          </Row>
          <Row label="Seed">
            <IntInput value={v.seed ?? 42} onChange={(n) => set(["sampling", "seed"], n ?? 42)} />
          </Row>
        </>
      )}
    </>
  )
})

const OutputFields = memo(function OutputFields({
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

const BackendFields = memo(function BackendFields({
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
