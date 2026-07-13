import type { ConfigFormValue } from "@/components/config-form"
import { defaultArchFor } from "@/components/config-form/backend-meta"
import type { BackendId } from "@/lib/api"
import type { LaunchOverrides } from "./types"

/**
 * Build the seed config a fresh "新建" creates.
 *
 * When called with the user's currently-default backend, the seed is
 * pre-wired for that backend: arch picks a sensible default the
 * backend supports (e.g. anima_lora → "anima"), and any backend-
 * specific options block (animaLora, diffusionPipe) gets a stub so
 * the editor opens with the right side-section already visible.
 *
 * Falls back to "kohya / sdxl" when no backend is provided — the
 * legacy behaviour that pre-dates the multi-backend UI work.
 */
export function buildDefaults(backend?: BackendId): ConfigFormValue {
  const effective = backend ?? "kohya"
  const arch = defaultArchFor(effective)
  const base: ConfigFormValue = {
    schemaVersion: "1.0",
    baseModel: { arch, checkpoint: "" },
    dataset: { source: "", resolution: [1024, 1024] },
    backend: { type: effective },
  }
  // Seed the per-backend options block so the editor opens with the
  // right sub-section visible from the first paint, instead of forcing
  // the user to flip the type field once before the section appears.
  if (effective === "anima_lora") {
    return {
      ...base,
      backend: {
        ...base.backend,
        animaLora: {
          method: "lora",
          preset: "default",
          // OrthoLoRA + T-LoRA stack — matches upstream lora.toml.
          lora: {
            useOrtho: true,
            useTimestepMask: true,
            minRank: 8,
            alphaRankScale: 1.0,
          },
        },
      },
    }
  }
  if (effective === "diffusion-pipe") {
    return {
      ...base,
      backend: {
        ...base.backend,
        diffusionPipe: {},
      },
    }
  }
  if (effective === "ai_toolkit") {
    return {
      ...base,
      baseModel: {
        ...base.baseModel,
        checkpoint: "krea/Krea-2-Raw",
      },
      network: {
        type: "lora",
        rank: 16,
        alpha: 16,
        targetUnet: true,
        targetTextEncoder: false,
      },
      optimizer: {
        type: "adamw8bit",
        lr: { unet: 1e-4, textEncoder: 5e-5 },
      },
      schedule: {
        epochs: 10,
        maxSteps: null,
        batchSize: 1,
        gradAccum: 1,
      },
      precision: "bf16",
      gradientCheckpointing: true,
      sampling: {
        enabled: true,
        everyNEpochs: 1,
        everyNSteps: null,
        resolution: [1024, 1024],
        seed: 42,
        inferenceSteps: 28,
        inferenceCfg: 4.5,
        prompts: [{ prompt: "a high quality image" }],
      },
      output: {
        name: "krea2_lora",
        saveEveryNEpochs: 1,
        saveEveryNSteps: null,
        saveLastNSteps: 4,
        saveDtype: "fp16",
      },
      backend: {
        ...base.backend,
        gpuDispatch: { mode: "one-job-per-gpu" },
        aiToolkit: {
          dataset: { resolutions: [1024] },
          train: { lrScheduler: "constant" },
        },
      },
    }
  }
  return base
}

export function shortenPath(p: string): string {
  const idx = p.toLowerCase().lastIndexOf("configs")
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
 * Pull current values out of a parsed config so the dialog can prefill the
 * inputs (or, when only used for placeholders, show what is currently in the
 * config). When `pendingDataset` is provided it wins over the config value
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
    batchSize: asString(schedule.batchSize),
    epochs: asString(schedule.epochs),
    maxSteps: asString(schedule.maxSteps),
  }
}

export function asString(value: unknown): string {
  if (value === null || value === undefined) return ""
  if (typeof value === "string") return value
  if (typeof value === "number" || typeof value === "boolean") return String(value)
  return ""
}

/**
 * Deep clone the config and write back any non-empty overrides at the right
 * paths. Numeric fields are coerced; invalid input is silently dropped so the
 * backend still sees a valid value rather than NaN.
 */
export function applyOverrides(
  config: Record<string, unknown>,
  overrides: LaunchOverrides,
): Record<string, unknown> {
  const cloned = structuredClone(config) as Record<string, unknown>
  const trim = (v: string) => v.trim()

  if (trim(overrides.datasetSource)) {
    setIn(cloned, ["dataset", "source"], trim(overrides.datasetSource))
  }
  if (trim(overrides.outputName)) {
    setIn(cloned, ["output", "name"], trim(overrides.outputName))
  }

  const batchSize = parsePositiveInt(overrides.batchSize)
  if (batchSize !== null) {
    setIn(cloned, ["schedule", "batchSize"], batchSize)
  }
  const epochs = parsePositiveInt(overrides.epochs)
  if (epochs !== null) {
    setIn(cloned, ["schedule", "epochs"], epochs)
  }
  const maxSteps = parsePositiveInt(overrides.maxSteps)
  if (maxSteps !== null) {
    setIn(cloned, ["schedule", "maxSteps"], maxSteps)
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
