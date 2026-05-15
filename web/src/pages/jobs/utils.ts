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
  done: "完成",
  error: "错误",
  log: "日志",
  start: "启动",
  cancel: "取消",
}

export const STATE_LABELS: Record<string, string> = {
  running: "运行中",
  succeeded: "已完成",
  failed: "失败",
  canceled: "已取消",
  canceling: "取消中",
  queued: "排队中",
  interrupted: "已中断",
}

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
