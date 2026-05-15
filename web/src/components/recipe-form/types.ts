/**
 * Shared types and pure update helpers for the recipe form.
 *
 * Kept in a dedicated module so each section file can import only the symbols
 * it needs without dragging in widgets or option lists.
 */
import type { ValidationFieldError } from "@/lib/api"

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

export type ErrorMap = Map<string, string[]>
export type Setter = (path: ReadonlyArray<string | number>, next: unknown) => void

// ------------------------------------------------------ pure update helpers

export function setIn<T extends object>(
  obj: T,
  path: ReadonlyArray<string | number>,
  value: unknown,
): T {
  if (path.length === 0) return value as T
  const cloned: any = Array.isArray(obj) ? [...(obj as any)] : { ...(obj as any) }
  const [head, ...rest] = path
  cloned[head as any] = setIn(
    cloned[head as any] ?? (typeof rest[0] === "number" ? [] : {}),
    rest,
    value,
  )
  return cloned
}

export function buildErrorMap(errors: ValidationFieldError[] | undefined): ErrorMap {
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
