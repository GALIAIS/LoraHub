import type { BackendId, JobDetail } from "@/lib/api"

export interface AnalysisBackendInfo {
  type: BackendId | null
  label: string
  supportsValidation: boolean | null
  validationConfigured: boolean | null
  configuredLr: number | null
  gradAccum: number | null
  minSnrGamma: number | null
  flowShift: number | null
  sampler: string | null
  sampleSteps: number | null
  sampleCfg: number | null
}

const BACKEND_LABELS: Record<BackendId, string> = {
  kohya: "kohya",
  "diffusion-pipe": "diffusion-pipe",
  anima_lora: "anima_lora",
  ai_toolkit: "ai_toolkit",
}

export function deriveAnalysisBackendInfo(
  detail: JobDetail | null | undefined,
): AnalysisBackendInfo {
  const config = asRecord(detail?.config_snapshot)
  const backend = asRecord(config.backend)
  const baseModel = asRecord(readValue(config, "baseModel", "base_model"))
  const rawType = readString(backend, "type")
  const type = isBackendId(rawType) ? rawType : null
  const arch = readString(baseModel, "arch")
  const anima = asRecord(readValue(backend, "animaLora", "anima_lora"))
  const aiToolkit = asRecord(readValue(backend, "aiToolkit", "ai_toolkit"))
  const aiTrain = asRecord(readValue(aiToolkit, "train"))
  const schedule = asRecord(readValue(config, "schedule"))
  const sampling = asRecord(readValue(config, "sampling"))
  const optimizer = asRecord(readValue(config, "optimizer"))
  const loss = asRecord(readValue(config, "loss"))
  const optimizerLr = readValue(optimizer, "lr")
  const lrGroup = asRecord(optimizerLr)
  const prompts = Array.isArray(sampling.prompts) ? sampling.prompts : []
  const firstPrompt = asRecord(prompts[0])

  const configuredLr = firstNumber(
    readValue(anima, "learningRate", "learning_rate"),
    readValue(lrGroup, "unet"),
    optimizerLr,
  )
  const validationSplit = firstNumber(
    readValue(anima, "validationSplitNum", "validation_split_num"),
  )

  return {
    type,
    label:
      type != null
        ? `${BACKEND_LABELS[type]}${type === "ai_toolkit" && arch ? ` · ${arch}` : ""}`
        : "后端未知",
    supportsValidation:
      type === "ai_toolkit" ? false : type === "anima_lora" ? true : null,
    validationConfigured:
      type === "ai_toolkit"
        ? false
        : type === "anima_lora"
          ? (validationSplit ?? 0) > 0
          : null,
    configuredLr,
    gradAccum: firstNumber(readValue(schedule, "gradAccum", "grad_accum")),
    minSnrGamma: firstNumber(
      readValue(aiTrain, "minSnrGamma", "min_snr_gamma"),
      readValue(anima, "minSnrGamma", "min_snr_gamma"),
      readValue(loss, "minSnrGamma", "min_snr_gamma"),
    ),
    flowShift: firstNumber(
      readValue(firstPrompt, "flowShift", "flow_shift"),
      readValue(anima, "discreteFlowShift", "discrete_flow_shift"),
      readValue(sampling, "flowShift", "flow_shift"),
    ),
    sampler:
      readString(firstPrompt, "sampler") ??
      readString(sampling, "sampleSampler", "sample_sampler", "sampler"),
    sampleSteps: firstNumber(
      readValue(firstPrompt, "steps"),
      readValue(sampling, "inferenceSteps", "inference_steps"),
    ),
    sampleCfg: firstNumber(
      readValue(firstPrompt, "cfg"),
      readValue(sampling, "inferenceCfg", "inference_cfg"),
    ),
  }
}

function isBackendId(value: string | null): value is BackendId {
  return (
    value === "kohya" ||
    value === "diffusion-pipe" ||
    value === "anima_lora" ||
    value === "ai_toolkit"
  )
}

function asRecord(value: unknown): Record<string, unknown> {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}

function readValue(
  source: Record<string, unknown>,
  ...keys: string[]
): unknown {
  for (const key of keys) {
    if (source[key] != null) return source[key]
  }
  return null
}

function readString(
  source: Record<string, unknown>,
  ...keys: string[]
): string | null {
  const value = readValue(source, ...keys)
  return typeof value === "string" && value.trim() ? value.trim() : null
}

function firstNumber(...values: unknown[]): number | null {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) return value
  }
  return null
}
