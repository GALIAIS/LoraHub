import type { RecipeFormValue } from "@/components/recipe-form"
import type { LaunchOverrides } from "./types"

export function buildDefaults(): RecipeFormValue {
  // Minimal valid skeleton — enough that the form renders with sensible
  // starting values; the user only has to fill in the two paths.
  return {
    schema_version: "1.0",
    base_model: { arch: "sdxl", checkpoint: "" },
    dataset: { source: "", resolution: [1024, 1024] },
  }
}

export function shortenPath(p: string): string {
  const idx = p.toLowerCase().lastIndexOf("recipes")
  return idx >= 0 ? p.slice(idx) : p
}

export function emptyOverrides(): LaunchOverrides {
  return {
    datasetSource: "",
    outputName: "",
    batchSize: "",
    epochs: "",
    maxSteps: "",
  }
}

/**
 * Pull current values out of a parsed recipe so the dialog can prefill the
 * inputs (or, when only used for placeholders, show what is currently in the
 * recipe). When `pendingDataset` is provided it wins over the recipe value
 * for `dataset.source` — that is how the Datasets page hands a freshly
 * scanned folder to the launch dialog.
 */
export function extractOverrides(
  parsed: Record<string, unknown> | null,
  pendingDataset: string | null,
): LaunchOverrides {
  const dataset = (parsed?.dataset as Record<string, unknown> | undefined) ?? {}
  const output = (parsed?.output as Record<string, unknown> | undefined) ?? {}
  const schedule = (parsed?.schedule as Record<string, unknown> | undefined) ?? {}

  const datasetSource =
    pendingDataset && pendingDataset.trim().length > 0
      ? pendingDataset
      : asString(dataset.source)

  return {
    datasetSource,
    outputName: asString(output.name),
    batchSize: asString(schedule.batch_size),
    epochs: asString(schedule.epochs),
    maxSteps: asString(schedule.max_steps),
  }
}

export function asString(value: unknown): string {
  if (value === null || value === undefined) return ""
  if (typeof value === "string") return value
  if (typeof value === "number" || typeof value === "boolean") return String(value)
  return ""
}

/**
 * Deep clone the recipe and write back any non-empty overrides at the right
 * paths. Numeric fields are coerced; invalid input is silently dropped so the
 * backend still sees a valid value rather than NaN.
 */
export function applyOverrides(
  recipe: Record<string, unknown>,
  overrides: LaunchOverrides,
): Record<string, unknown> {
  const cloned = structuredClone(recipe) as Record<string, unknown>
  const trim = (v: string) => v.trim()

  if (trim(overrides.datasetSource)) {
    setIn(cloned, ["dataset", "source"], trim(overrides.datasetSource))
  }
  if (trim(overrides.outputName)) {
    setIn(cloned, ["output", "name"], trim(overrides.outputName))
  }

  const batchSize = parsePositiveInt(overrides.batchSize)
  if (batchSize !== null) {
    setIn(cloned, ["schedule", "batch_size"], batchSize)
  }
  const epochs = parsePositiveInt(overrides.epochs)
  if (epochs !== null) {
    setIn(cloned, ["schedule", "epochs"], epochs)
  }
  const maxSteps = parsePositiveInt(overrides.maxSteps)
  if (maxSteps !== null) {
    setIn(cloned, ["schedule", "max_steps"], maxSteps)
  }

  return cloned
}

export function parsePositiveInt(raw: string): number | null {
  const trimmed = raw.trim()
  if (trimmed === "") return null
  const num = Number(trimmed)
  if (!Number.isFinite(num)) return null
  const int = Math.trunc(num)
  if (int < 1) return null
  return int
}

/** Lodash-style setIn: walks/creates nested object keys and writes the leaf. */
export function setIn(
  target: Record<string, unknown>,
  path: string[],
  value: unknown,
): void {
  let cursor: Record<string, unknown> = target
  for (let i = 0; i < path.length - 1; i++) {
    const key = path[i]
    const next = cursor[key]
    if (next === null || typeof next !== "object" || Array.isArray(next)) {
      const created: Record<string, unknown> = {}
      cursor[key] = created
      cursor = created
    } else {
      cursor = next as Record<string, unknown>
    }
  }
  cursor[path[path.length - 1]] = value
}
