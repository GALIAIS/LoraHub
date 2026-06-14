import type { JobDetail, JobMetricsResponse } from "@/lib/api"

export type Trend = "improving" | "flat" | "diverging" | "unknown"
export type Overfit = null | "warn" | "danger"

export interface Summary {
  step: number | null
  totalSteps: number | null
  percent: number | null
  etaSec: number | null
  wallSec: number | null
  lossStart: number | null
  lossLatest: number | null
  lossDropPct: number | null
  trend: Trend
  valGap: number | null
  overfit: Overfit
  hparams: {
    lr: string | null
    rank: string | null
    alpha: string | null
    dropout: string | null
    numRepeats: string | null
    batchAccum: string | null
    epochs: string | null
    optimizer: string | null
  }
}

export function deriveSummary(
  job: JobDetail | undefined,
  metrics: JobMetricsResponse | undefined,
  fallbackTotalSteps: number | null,
): Summary | null {
  const cfg = (job?.config_snapshot ?? {}) as Record<string, unknown>
  if (Object.keys(cfg).length === 0 && !metrics) return null

  // Loss series (cleaned of non-finite entries)
  const points = (metrics?.loss ?? []).filter(
    (p): p is { step: number; loss: number; ts: number } =>
      typeof p.loss === "number" && Number.isFinite(p.loss),
  )

  const lossStart = points.length > 0 ? points[0].loss : null
  const lossLatest = points.length > 0 ? points[points.length - 1].loss : null
  const lossDropPct =
    lossStart != null && lossLatest != null && lossStart > 0
      ? ((lossStart - lossLatest) / lossStart) * 100
      : null

  // Trend: compare last 20% vs prior 20% of points to classify shape.
  const trend = classifyTrend(points)

  // Validation gap (latest val_loss - latest train loss in the same epoch).
  const vals = metrics?.val_loss ?? []
  let valGap: number | null = null
  let overfit: Overfit = null
  if (vals.length > 0 && lossLatest != null) {
    const latestVal = vals[vals.length - 1].val_loss
    if (typeof latestVal === "number" && Number.isFinite(latestVal)) {
      valGap = latestVal - lossLatest
      if (valGap > 0.5 * Math.max(lossLatest, 1e-3)) overfit = "danger"
      else if (valGap > 0.2 * Math.max(lossLatest, 1e-3)) overfit = "warn"
    }
  }

  // Progress
  const step = lossLatest != null ? points[points.length - 1].step : null
  const totalSteps = pickTotalSteps(metrics, cfg, fallbackTotalSteps)
  const percent =
    step != null && totalSteps != null && totalSteps > 0
      ? (step / totalSteps) * 100
      : null
  const wallSec =
    metrics?.first_step_ts != null && metrics?.last_step_ts != null
      ? metrics.last_step_ts - metrics.first_step_ts
      : null
  const etaSec =
    wallSec != null && step != null && totalSteps != null && step > 0
      ? (wallSec / step) * (totalSteps - step)
      : null

  // Hparams snapshot — accept both camelCase (new YAMLs) and snake_case
  // (legacy snapshots persisted before the schema migration). Pull each
  // field through a tolerant helper.
  const network = pluckObj(cfg, "network") ?? {}
  const optimizer = pluckObj(cfg, "optimizer") ?? {}
  const schedule = pluckObj(cfg, "schedule") ?? {}
  const dataset = pluckObj(cfg, "dataset") ?? {}
  const backend = pluckObj(cfg, "backend") ?? {}
  const backendType = stringOrNull(pluck(backend, "type"))
  // anima_lora reads learning rate from backend.animaLora.learningRate;
  // kohya / diffusion-pipe read it from optimizer.lr.unet. Mirror the
  // compiler's source-of-truth field per backend so the displayed value
  // matches what the trainer actually consumes.
  const animaLora = pluckObj(backend, "animaLora") ?? pluckObj(backend, "anima_lora") ?? {}
  const lr =
    backendType === "anima_lora"
      ? pluckOne(animaLora, "learningRate", "learning_rate")
      : pluck(pluckObj(optimizer, "lr") ?? {}, "unet")
  const optimizerType =
    backendType === "anima_lora"
      ? pluckOne(animaLora, "optimizerType", "optimizer_type")
      : pluck(optimizer, "type")
  const batchSize = pluckOne(schedule, "batchSize", "batch_size")
  const gradAccum = pluckOne(schedule, "gradAccum", "grad_accum")

  // anima_lora reads rank/alpha/dropout from backend.animaLora.*
  // (networkDim / networkAlpha / lora.networkDropout); kohya / dp
  // read from the top-level network section. Mirror the lr / optimizer
  // routing above so the summary always reflects what train.py actually
  // consumed.
  const animaLoraLora = pluckObj(animaLora, "lora") ?? {}
  const rank =
    backendType === "anima_lora"
      ? pluckOne(animaLora, "networkDim", "network_dim")
      : pluck(network, "rank")
  const alpha =
    backendType === "anima_lora"
      ? pluckOne(animaLora, "networkAlpha", "network_alpha")
      : pluck(network, "alpha")
  const networkDropout =
    backendType === "anima_lora"
      ? pluckOne(animaLoraLora, "networkDropout", "network_dropout")
      : pluckOne(network, "networkDropout", "network_dropout")
  const hparams = {
    lr: typeof lr === "number" ? scientific(lr) : null,
    rank: stringOrNull(rank),
    alpha: stringOrNull(alpha),
    dropout: numberOrNull(networkDropout),
    numRepeats: stringOrNull(pluckOne(dataset, "numRepeats", "num_repeats")),
    batchAccum:
      batchSize != null && gradAccum != null
        ? `${batchSize}×${gradAccum}`
        : stringOrNull(batchSize),
    epochs: stringOrNull(pluck(schedule, "epochs")),
    optimizer: stringOrNull(optimizerType),
  }

  return {
    step,
    totalSteps,
    percent,
    etaSec,
    wallSec,
    lossStart,
    lossLatest,
    lossDropPct,
    trend,
    valGap,
    overfit,
    hparams,
  }
}

function classifyTrend(
  points: Array<{ step: number; loss: number }>,
): Trend {
  if (points.length < 6) return "unknown"
  const tail = points.slice(Math.floor(points.length * 0.8))
  const head = points.slice(
    Math.floor(points.length * 0.6),
    Math.floor(points.length * 0.8),
  )
  if (tail.length === 0 || head.length === 0) return "unknown"
  const tailAvg = tail.reduce((s, p) => s + p.loss, 0) / tail.length
  const headAvg = head.reduce((s, p) => s + p.loss, 0) / head.length
  const delta = (tailAvg - headAvg) / Math.max(headAvg, 1e-6)
  if (delta < -0.02) return "improving"
  if (delta > 0.05) return "diverging"
  return "flat"
}

export function trendLabel(t: Trend): string {
  switch (t) {
    case "improving":
      return "下降中"
    case "flat":
      return "平台期"
    case "diverging":
      return "上升发散"
    default:
      return "—"
  }
}

export function trendTone(t: Trend): string {
  switch (t) {
    case "improving":
      return "text-emerald-600 dark:text-emerald-400"
    case "flat":
      return "text-amber-700 dark:text-amber-400"
    case "diverging":
      return "text-red-600 dark:text-red-400"
    default:
      return "text-foreground/85"
  }
}

function pluck(obj: unknown, key: string): unknown {
  if (obj && typeof obj === "object" && key in (obj as object)) {
    return (obj as Record<string, unknown>)[key]
  }
  return null
}

function pluckObj(obj: unknown, key: string): Record<string, unknown> | null {
  const v = pluck(obj, key)
  return v && typeof v === "object" ? (v as Record<string, unknown>) : null
}

function pluckOne(obj: unknown, ...keys: string[]): unknown {
  if (!obj || typeof obj !== "object") return undefined
  const o = obj as Record<string, unknown>
  for (const k of keys) {
    if (k in o && o[k] != null) return o[k]
  }
  return undefined
}

function pickTotalSteps(
  metrics: JobMetricsResponse | undefined,
  cfg: Record<string, unknown>,
  fallback: number | null,
): number | null {
  // Trainer-reported total wins — it's the only number that survives
  // mid-run schedule changes (warmup steps rolling in, sample_ratio
  // truncating the dataset, …). Falls back to schedule.maxSteps in the
  // config, then to the config-derived estimate (epochs × repeats ×
  // images / (batch × accum)) the parent computes off the dataset
  // scan. This is the same priority order overview-tab uses, so the
  // three tabs now agree.
  const fromMetrics = metrics?.total_steps
  if (typeof fromMetrics === "number" && fromMetrics > 0) return fromMetrics
  const sched = pluckObj(cfg, "schedule") ?? {}
  const explicit = pluckOne(sched, "maxSteps", "max_steps")
  if (typeof explicit === "number" && explicit > 0) return explicit
  return fallback
}

function scientific(n: number): string {
  if (!Number.isFinite(n)) return "—"
  if (n === 0) return "0"
  const abs = Math.abs(n)
  if (abs >= 1e-3 && abs < 1e3) return n.toString()
  return n.toExponential(2)
}

function stringOrNull(v: unknown): string | null {
  if (v == null) return null
  return String(v)
}

function numberOrNull(v: unknown): string | null {
  if (typeof v !== "number" || !Number.isFinite(v)) return null
  return v.toFixed(3).replace(/\.?0+$/, "")
}
