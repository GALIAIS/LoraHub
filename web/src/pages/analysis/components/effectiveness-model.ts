import type { JobMetricsResponse, OverfitTrend } from "@/lib/api"
import {
  defaultRollingWindow,
  rollingQuartiles,
  trailingSlope,
} from "./loss-stats"

export interface ConvergenceVerdict {
  /** Percentage of loss reduction from the early window mean to the late window mean. */
  dropPct: number
  earlyMean: number
  lateMean: number
  /** "improving" -> still going down meaningfully; "plateau" -> small/no change; "diverging" -> going up. */
  state: "improving" | "plateau" | "diverging"
  /** Number of training-loss samples used. */
  samples: number
}

export interface StabilityVerdict {
  /**
   * Slope-to-spread ratio: |slope*windowSpan| / IQR. >= 1 means the
   * mean trajectory has moved more than one IQR over the window.
   * Negative `slope` means loss is still falling.
   */
  snr: number
  /** OLS slope of loss vs step on the trailing window. */
  slope: number
  /** IQR of the trailing window (Q75 - Q25), used as the spread proxy. */
  iqr: number
  /** R2 of the OLS fit on the same window: a directional confidence. */
  rSquared: number
  /** Coefficient of variation, kept as a secondary descriptor. */
  cov: number
  /**
   * "progressing" -> SNR >= 1 with negative slope (still descending);
   * "stalled"      -> |SNR| < 1 (movement smaller than within-window noise);
   * "diverging"    -> SNR >= 1 with positive slope.
   */
  state: "progressing" | "stalled" | "diverging"
  windowSamples: number
}

export interface OverfitVerdict {
  trainLatest: number | null
  valLatest: number | null
  gap: number | null
  trend: OverfitTrend | null
  /** Synthesised severity used to drive the bar fill + colour. */
  severity: "ok" | "watch" | "warn"
}

export interface LrDropEvent {
  /** Index in the input series where the drop was detected. */
  index: number
  step: number
  lrBefore: number
  lrAfter: number
  /** Mean loss in the N-step pre-window (excluding the drop point itself). */
  preLoss: number
  /** Mean loss in the N-step post-window. */
  postLoss: number
  /** Loss improvement as a fraction of preLoss; >= 0 means improved. */
  improvementPct: number
}

export interface LrResponseVerdict {
  /** All detected LR drop events, ordered chronologically. */
  events: LrDropEvent[]
  /** Mean improvement (%) across every event with a usable post-window. */
  meanImprovementPct: number
  /** Number of events that produced a noticeable (> 0.5%) improvement. */
  responsiveEvents: number
  /**
   * "responsive" -> >= 60% of drops produced > 0.5% improvement;
   * "weak"        -> at least one drop but most ineffective;
   * "no-events"   -> LR series has no detected drop yet.
   */
  state: "responsive" | "weak" | "no-events"
}

export interface ForgettingVerdict {
  /** Latest preservation in [0..1] across every neutral prompt. */
  latest: number | null
  /** Slope of preserved over time (negative = drifting away from base). */
  trend: number | null
  /** Number of neutral-prompt samples observed so far. */
  samples: number
  /**
   * "stable"     -> latest >= 0.85, no significant negative trend;
   * "drifting"   -> latest 0.65-0.85 OR negative trend;
   * "forgetting" -> latest < 0.65;
   * "no-data"    -> no neutral prompts probed yet.
   */
  state: "stable" | "drifting" | "forgetting" | "no-data"
}

export type StageKey =
  | "warmup"
  | "converging"
  | "plateau"
  | "diverging"
  | "unknown"

export const STAGE_LABELS: Record<StageKey, string> = {
  warmup: "热身阶段",
  converging: "收敛中",
  plateau: "已平台",
  diverging: "发散风险",
  unknown: "数据不足",
}

export const STAGE_TONES: Record<StageKey, string> = {
  warmup: "text-sky-700 dark:text-sky-300",
  converging: "text-emerald-700 dark:text-emerald-300",
  plateau: "text-amber-700 dark:text-amber-300",
  diverging: "text-red-600 dark:text-red-400",
  unknown: "text-muted-foreground",
}

export const STAGE_BG: Record<StageKey, string> = {
  warmup: "from-sky-500/15 to-sky-500/0",
  converging: "from-emerald-500/15 to-emerald-500/0",
  plateau: "from-amber-500/15 to-amber-500/0",
  diverging: "from-red-500/15 to-red-500/0",
  unknown: "from-muted/40 to-muted/0",
}

export function describeConvergence(
  losses: number[],
): ConvergenceVerdict | null {
  if (losses.length < 6) return null
  // Use leading 20% as the "early" window and trailing 20% as the "late"
  // window, bounded so tiny and very long runs stay stable.
  const winSize = Math.min(Math.max(3, Math.floor(losses.length * 0.2)), 100)
  const early = losses.slice(0, winSize)
  const late = losses.slice(-winSize)
  const earlyMean = early.reduce((a, b) => a + b, 0) / early.length
  const lateMean = late.reduce((a, b) => a + b, 0) / late.length
  const dropPct =
    earlyMean > 0 ? ((earlyMean - lateMean) / earlyMean) * 100 : 0
  let state: ConvergenceVerdict["state"]
  if (dropPct < -1.5) state = "diverging"
  else if (dropPct < 3) state = "plateau"
  else state = "improving"
  return { dropPct, earlyMean, lateMean, state, samples: losses.length }
}

export function describeStability(
  points: { step: number; loss: number }[],
): StabilityVerdict | null {
  if (points.length < 8) return null
  const window = defaultRollingWindow(points.length)
  const slope = trailingSlope(points, window)
  if (!slope) return null
  const trailing = points.slice(-window)
  const losses = trailing.map((p) => p.loss)
  const spread = (() => {
    const { band } = rollingQuartiles(trailing, window)
    if (band.length === 0) return 0
    const last = band[band.length - 1]
    return Math.max(last.hi - last.lo, 1e-9)
  })()
  const stepSpan =
    trailing.length > 1
      ? trailing[trailing.length - 1].step - trailing[0].step
      : 1
  const totalChange = slope.slope * (stepSpan || 1)
  const snr = totalChange / spread
  const mean = losses.reduce((a, b) => a + b, 0) / losses.length
  const variance =
    losses.reduce((acc, v) => acc + (v - mean) ** 2, 0) / losses.length
  const cov = Math.abs(mean) > 1e-9 ? Math.sqrt(variance) / Math.abs(mean) : 0
  let state: StabilityVerdict["state"]
  if (Math.abs(snr) < 1) state = "stalled"
  else if (snr < 0) state = "progressing"
  else state = "diverging"
  return {
    snr,
    slope: slope.slope,
    iqr: spread,
    rSquared: slope.rSquared,
    cov,
    state,
    windowSamples: trailing.length,
  }
}

export function describeOverfit(
  m: JobMetricsResponse | null,
): OverfitVerdict {
  const o = m?.overfit_signal
  const gap = o?.gap ?? null
  const trend = o?.trend ?? null
  let severity: OverfitVerdict["severity"] = "ok"
  if (trend === "overfitting") severity = "warn"
  else if (gap != null && gap > 0.05) severity = "watch"
  return {
    trainLatest: o?.latest_train ?? null,
    valLatest: o?.latest_val ?? null,
    gap,
    trend,
    severity,
  }
}

export function describeLrResponse(
  m: JobMetricsResponse | null,
): LrResponseVerdict {
  const points = (m?.loss ?? []).filter(
    (p): p is { step: number; loss: number; lr: number; ts: number } =>
      typeof p.loss === "number" &&
      Number.isFinite(p.loss) &&
      typeof p.lr === "number" &&
      Number.isFinite(p.lr),
  )
  if (points.length < 8) {
    return {
      events: [],
      meanImprovementPct: 0,
      responsiveEvents: 0,
      state: "no-events",
    }
  }
  const events: LrDropEvent[] = []
  const win = Math.min(16, Math.max(4, Math.floor(points.length * 0.05)))
  for (let i = 1; i < points.length - 1; i += 1) {
    const prevLr = points[i - 1].lr
    const curLr = points[i].lr
    if (prevLr <= 0 || curLr <= 0) continue
    const relDrop = (prevLr - curLr) / prevLr
    if (relDrop < 0.1) continue
    const preStart = Math.max(0, i - win)
    const preSlice = points.slice(preStart, i)
    if (preSlice.length < 3) continue
    const postEnd = Math.min(points.length, i + 1 + win)
    const postSlice = points.slice(i + 1, postEnd)
    if (postSlice.length < 3) continue
    const preLoss =
      preSlice.reduce((a, p) => a + p.loss, 0) / preSlice.length
    const postLoss =
      postSlice.reduce((a, p) => a + p.loss, 0) / postSlice.length
    const improvementPct =
      preLoss > 0 ? ((preLoss - postLoss) / preLoss) * 100 : 0
    events.push({
      index: i,
      step: points[i].step,
      lrBefore: prevLr,
      lrAfter: curLr,
      preLoss,
      postLoss,
      improvementPct,
    })
  }
  if (events.length === 0) {
    return {
      events: [],
      meanImprovementPct: 0,
      responsiveEvents: 0,
      state: "no-events",
    }
  }
  const meanImprovementPct =
    events.reduce((a, e) => a + e.improvementPct, 0) / events.length
  const responsiveEvents = events.filter(
    (e) => e.improvementPct > 0.5,
  ).length
  const responsiveRatio = responsiveEvents / events.length
  const state: LrResponseVerdict["state"] =
    responsiveRatio >= 0.6 ? "responsive" : "weak"
  return { events, meanImprovementPct, responsiveEvents, state }
}

export function describeForgetting(
  m: JobMetricsResponse | null,
): ForgettingVerdict {
  const probe = (m?.forgetting_probe ?? []).filter(
    (p): p is { step: number; preserved: number } & typeof p =>
      typeof p.preserved === "number" &&
      Number.isFinite(p.preserved) &&
      typeof p.step === "number",
  )
  if (probe.length === 0) {
    return { latest: null, trend: null, samples: 0, state: "no-data" }
  }
  const latest = probe[probe.length - 1].preserved as number
  let trend: number | null = null
  if (probe.length >= 3) {
    const meanX =
      probe.reduce((a, p) => a + (p.step as number), 0) / probe.length
    const meanY =
      probe.reduce((a, p) => a + (p.preserved as number), 0) / probe.length
    let num = 0
    let den = 0
    for (const p of probe) {
      const dx = (p.step as number) - meanX
      num += dx * ((p.preserved as number) - meanY)
      den += dx * dx
    }
    trend = den === 0 ? 0 : num / den
  }
  let state: ForgettingVerdict["state"]
  if (latest < 0.65) state = "forgetting"
  else if (latest < 0.85 || (trend != null && trend < -1e-5)) {
    state = "drifting"
  } else state = "stable"
  return {
    latest,
    trend,
    samples: probe.length,
    state,
  }
}

export function classifyStage(
  conv: ConvergenceVerdict | null,
  stab: StabilityVerdict | null,
  losses: number[],
): { key: StageKey; reason: string } {
  if (!conv) return { key: "unknown", reason: "训练步数不足以判断" }
  if (conv.state === "diverging") {
    return {
      key: "diverging",
      reason: `近 ${pctFormat(Math.abs(conv.dropPct))} 损失反向上升`,
    }
  }
  const idxFraction =
    losses.length > 0 ? losses.length / (losses.length + 0) : 0
  void idxFraction
  if (conv.state === "improving" && losses.length < 32) {
    return { key: "warmup", reason: "样本仍在快速下降, 热身阶段" }
  }
  if (conv.state === "improving") {
    return {
      key: "converging",
      reason: `早→晚均值降幅 ${pctFormat(conv.dropPct)}`,
    }
  }
  if (conv.state === "plateau") {
    if (stab && stab.state === "diverging") {
      return {
        key: "diverging",
        reason: `平台期但近窗回升 (SNR ${stab.snr.toFixed(2)})`,
      }
    }
    return {
      key: "plateau",
      reason: stab
        ? `近窗均值变化 < 3%, 趋势 SNR ${stab.snr.toFixed(2)}`
        : "近窗均值变化 < 3%",
    }
  }
  return { key: "unknown", reason: "" }
}

function pctFormat(v: number): string {
  return `${v.toFixed(1)}%`
}
