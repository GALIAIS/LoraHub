// Shared constants and helpers for the Jobs page.

export const TERMINAL_STATES = new Set([
  "succeeded",
  "failed",
  "canceled",
  "interrupted",
])

export const EVENT_TYPE_LABELS: Record<string, string> = {
  step: "训练步",
  epoch_end: "回合结束",
  checkpoint_saved: "保存检查点",
  sample_ready: "样本生成",
  validation: "验证",
  cache_progress: "缓存进度",
  oom: "显存不足",
  gpu_sample: "GPU 采样",
  done: "完成",
  error: "错误",
  log: "日志",
  start: "启动",
  cancel: "取消",
  diagnostic_warning: "诊断告警",
  preview_unavailable: "预览不可用",
}

export const STATE_LABELS: Record<string, string> = {
  running: "运行中",
  preparing: "准备中",
  succeeded: "已完成",
  failed: "失败",
  canceled: "已取消",
  canceling: "取消中",
  queued: "排队中",
  interrupted: "已中断",
}

// States during which the worker slot is held but the trainer
// subprocess hasn't necessarily started yet. Used by the UI to
// pick "live realtime tiles" vs "queued placeholder".
export const ACTIVE_STATES = new Set(["preparing", "running"])

export function stateLabel(state: string): string {
  return STATE_LABELS[state] ?? state
}

export type StatusFilter =
  | "all"
  | "running"
  | "succeeded"
  | "failed"
  | "canceled"

export const STATUS_FILTER_OPTIONS: { value: StatusFilter; label: string }[] = [
  { value: "all", label: "全部状态" },
  { value: "running", label: "运行中" },
  { value: "succeeded", label: "已完成" },
  { value: "failed", label: "失败" },
  { value: "canceled", label: "已取消" },
]

export const COMPARE_LIMIT = 4

// Distinct stroke tokens for overlaying multiple loss series on the same chart.
export const COMPARE_COLORS = [
  "var(--primary)",
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
] as const

export function fmtBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0 B"
  const units = ["B", "KB", "MB", "GB", "TB", "PB"]
  let i = 0
  let v = n
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i += 1
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : v >= 10 ? 1 : 2)} ${units[i]}`
}

export function fmtUnixSeconds(ts: number): string {
  if (!Number.isFinite(ts) || ts <= 0) return "—"
  return new Date(ts * 1000).toLocaleString()
}

export function fmtDuration(secs: number | null | undefined): string {
  if (secs == null || !Number.isFinite(secs) || secs < 0) return "—"
  const total = Math.round(secs)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  if (h > 0) return `${h}h ${m}m ${s}s`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

// Even-stride downsample to keep large series rendering cheap. Always keeps
// the first and last point so the chart endpoints don't drift.
export function downsamplePoints<T>(points: T[], maxPoints: number): T[] {
  if (points.length <= maxPoints) return points
  const out: T[] = []
  const step = (points.length - 1) / (maxPoints - 1)
  for (let i = 0; i < maxPoints; i += 1) {
    out.push(points[Math.round(i * step)])
  }
  return out
}

/**
 * Derive an expected `total_steps` from a job's config_snapshot.
 *
 * Priority:
 *   1. `schedule.maxSteps` (or `max_steps`) if explicitly set in the config.
 *   2. `epochs * dataset.numRepeats * image_files / (batchSize * gradAccum)`
 *      if we have an image count from the latest preflight.
 *   3. `null` (fall back to `?`) — backend events with `total_steps` will
 *      override this anyway as soon as the trainer reports them.
 *
 * Reads camelCase **and** snake_case keys: the snapshot gets dumped from
 * pydantic with `mode="json"` (no `by_alias`), so we'd see snake_case in
 * practice today, but the schema is alias-aware, so newer dumps may flip
 * to camelCase. Reading both keeps us future-proof.
 *
 * The result is intentionally an over-estimate when bucketing is on, but the
 * UI just needs *a* denominator so progress bars and "第 X/N 步" tags don't
 * render `undefined`.
 */
export function expectedTotalSteps(
  configSnapshot: Record<string, unknown> | null | undefined,
  imageFiles?: number | null,
): number | null {
  if (!configSnapshot || typeof configSnapshot !== "object") return null
  const schedule = (configSnapshot as Record<string, unknown>)["schedule"]
  const dataset = (configSnapshot as Record<string, unknown>)["dataset"]

  function pickNum(
    obj: Record<string, unknown> | null | undefined,
    ...keys: string[]
  ): number | null {
    if (!obj) return null
    for (const k of keys) {
      const v = obj[k]
      if (typeof v === "number" && Number.isFinite(v)) return v
    }
    return null
  }

  const sched =
    schedule && typeof schedule === "object"
      ? (schedule as Record<string, unknown>)
      : null
  const ds =
    dataset && typeof dataset === "object"
      ? (dataset as Record<string, unknown>)
      : null

  // 1) explicit cap wins
  const maxSteps = pickNum(sched, "maxSteps", "max_steps")
  if (maxSteps != null && maxSteps > 0) return maxSteps

  // 2) derived from dataset size
  if (sched && ds && typeof imageFiles === "number" && imageFiles > 0) {
    const epochs = pickNum(sched, "epochs") ?? 1
    const batch = pickNum(sched, "batchSize", "batch_size") ?? 1
    const accum = pickNum(sched, "gradAccum", "grad_accum") ?? 1
    const repeats = pickNum(ds, "numRepeats", "num_repeats") ?? 1
    const denom = Math.max(1, batch * accum)
    return Math.max(1, Math.ceil((epochs * repeats * imageFiles) / denom))
  }
  return null
}
