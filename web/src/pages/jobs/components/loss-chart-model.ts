import type { OverfitSignal } from "@/lib/api"

export interface LossSeries {
  id: string
  label: string
  color: string
  // `dashed` is reserved for derived overlays (EMA smoothing, baselines)
  // so users can tell them apart from primary measurements at a glance.
  dashed?: boolean
  points: { step: number; loss: number }[]
}

// Optional vertical guides — used by the metrics tab to mark every
// checkpoint save on the loss chart so the user can correlate
// loss inflections with what artefact was written at that step.
export interface ChartMarker {
  step: number
  label?: string
  color?: string
}

// Optional confidence band drawn behind the primary series. Used by the
// analysis workbench to render a rolling IQR (Q25..Q75) underneath the
// median line so high-variance diffusion losses don't read as a single
// line. The band itself isn't a series — it can't be toggled in the
// legend and doesn't carry tooltip values.
export interface ChartBand {
  id: string
  label?: string
  color: string
  /** Same step axis as the series. lo/hi are absolute loss values. */
  points: { step: number; lo: number; hi: number }[]
}

export interface LossChartProps {
  series: LossSeries[]
  className?: string
  emptyHint?: string
  overfitSignal?: OverfitSignal | null
  markers?: ChartMarker[]
  /** Optional confidence band(s) drawn behind the primary series. */
  bands?: ChartBand[]
  /**
   * Label rendered next to the X-axis ticks. Defaults to "step" — the
   * analysis workbench overrides this when the user toggles the X
   * axis to epoch or wallclock-seconds.
   */
  xLabel?: string
  /**
   * Custom formatter for X-axis tick values. Defaults to integer
   * rendering; the workbench passes a duration formatter when the
   * X axis is wallclock-seconds.
   */
  xTickFormat?: (v: number) => string
  /**
   * Stable key used to persist the user's view (zoom range, log toggle)
   * across re-renders within a session. Pass the active job id when the
   * chart shows one job's loss; pass `null` to skip persistence.
   */
  persistKey?: string | null
  /** Internally toggled; do not pass from the outside. */
  fullscreen?: boolean
}

export const VIEW_W = 800
export const VIEW_H = 300
export const PAD_LEFT = 52
export const PAD_RIGHT = 16
export const PAD_TOP = 14
export const PAD_BOTTOM = 28
export const MAX_POINTS = 1500

export function formatLoss(v: number): string {
  if (!Number.isFinite(v)) return "—"
  if (Math.abs(v) >= 100) return v.toFixed(1)
  if (Math.abs(v) >= 1) return v.toFixed(3)
  return v.toFixed(4)
}

export function trendCopy(trend: OverfitSignal["trend"]): {
  label: string
  tone: "ok" | "muted" | "danger"
} | null {
  switch (trend) {
    case "improving":
      return { label: "持续下降", tone: "ok" }
    case "flat":
      return { label: "已平台", tone: "muted" }
    case "overfitting":
      return { label: "疑似过拟合", tone: "danger" }
    default:
      return null
  }
}
