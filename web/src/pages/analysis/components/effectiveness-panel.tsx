/**
 * EffectivenessPanel — "is this run actually working?"
 *
 * Four cards arranged in a responsive grid, each speaking to one
 * dimension of training effectiveness:
 *
 *   1. 收敛趋势   — early-window loss vs late-window loss, with a
 *                   visual drop bar + "still falling / plateau /
 *                   diverging" verdict.
 *   2. 平稳度     — coefficient of variation on the late window;
 *                   helps the user spot oscillating runs that look
 *                   "fine on average" but actually thrash.
 *   3. 过拟合风险 — train↔val gap, gap-vs-baseline trend, and the
 *                   backend's overfit_signal verdict shown together
 *                   so the user doesn't have to reconcile them.
 *   4. 训练阶段   — derived stage badge (warm-up / converging /
 *                   plateau / diverging) plus the heuristic the
 *                   classifier used.
 *
 * Pure reductions over `metrics.loss` / `metrics.val_loss` —
 * everything is recomputed in O(N) on the client; no extra API
 * round-trip required. Cards animate in with a short staggered
 * fade so a "training started, first metrics arrived" event reads
 * as fluid rather than abrupt.
 */
import { useMemo } from "react"
import {
  Activity,
  Gauge,
  ShieldAlert,
  Sparkles,
  Zap,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { JobMetricsResponse, OverfitTrend } from "@/lib/api"
import {
  defaultRollingWindow,
  rollingQuartiles,
  trailingSlope,
} from "./loss-stats"

interface Props {
  metrics: JobMetricsResponse | null
}

interface ConvergenceVerdict {
  /** Percentage of loss reduction from the early window mean to the late window mean. */
  dropPct: number
  earlyMean: number
  lateMean: number
  /** "improving" → still going down meaningfully; "plateau" → small/no change; "diverging" → going up. */
  state: "improving" | "plateau" | "diverging"
  /** Number of training-loss samples used. */
  samples: number
}

interface StabilityVerdict {
  /**
   * Slope-to-spread ratio: |slope·windowSpan| / IQR. ≥ 1 means the
   * mean trajectory has moved more than one IQR over the window —
   * a signal that the run is actively progressing rather than
   * thrashing in place. Negative `slope` means loss is still falling.
   */
  snr: number
  /** OLS slope of loss vs step on the trailing window. */
  slope: number
  /** IQR of the trailing window (Q75 − Q25), used as the spread proxy. */
  iqr: number
  /** R² of the OLS fit on the same window — a directional confidence. */
  rSquared: number
  /** Coefficient of variation, kept as a secondary descriptor. */
  cov: number
  /**
   * "progressing" → SNR ≥ 1 with negative slope (still descending);
   * "stalled"      → |SNR| < 1 (movement smaller than within-window noise);
   * "diverging"    → SNR ≥ 1 with positive slope.
   */
  state: "progressing" | "stalled" | "diverging"
  windowSamples: number
}

interface OverfitVerdict {
  trainLatest: number | null
  valLatest: number | null
  gap: number | null
  trend: OverfitTrend | null
  /** Synthesised severity used to drive the bar fill + colour. */
  severity: "ok" | "watch" | "warn"
}

interface LrDropEvent {
  /** Index in the input series where the drop was detected. */
  index: number
  step: number
  lrBefore: number
  lrAfter: number
  /** Mean loss in the N-step pre-window (excluding the drop point itself). */
  preLoss: number
  /** Mean loss in the N-step post-window. */
  postLoss: number
  /** Loss improvement as a fraction of preLoss; ≥ 0 means improved. */
  improvementPct: number
}

interface LrResponseVerdict {
  /** All detected LR drop events, ordered chronologically. */
  events: LrDropEvent[]
  /** Mean improvement (%) across every event with a usable post-window. */
  meanImprovementPct: number
  /** Number of events that produced a noticeable (> 0.5%) improvement. */
  responsiveEvents: number
  /**
   * "responsive" → ≥ 60% of drops produced > 0.5% improvement;
   * "weak"        → at least one drop but most ineffective;
   * "no-events"   → LR series has no detected drop yet.
   */
  state: "responsive" | "weak" | "no-events"
}

type StageKey = "warmup" | "converging" | "plateau" | "diverging" | "unknown"

const STAGE_LABELS: Record<StageKey, string> = {
  warmup: "热身阶段",
  converging: "收敛中",
  plateau: "已平台",
  diverging: "发散风险",
  unknown: "数据不足",
}

const STAGE_TONES: Record<StageKey, string> = {
  warmup: "text-sky-700 dark:text-sky-300",
  converging: "text-emerald-700 dark:text-emerald-300",
  plateau: "text-amber-700 dark:text-amber-300",
  diverging: "text-red-600 dark:text-red-400",
  unknown: "text-muted-foreground",
}

const STAGE_BG: Record<StageKey, string> = {
  warmup: "from-sky-500/15 to-sky-500/0",
  converging: "from-emerald-500/15 to-emerald-500/0",
  plateau: "from-amber-500/15 to-amber-500/0",
  diverging: "from-red-500/15 to-red-500/0",
  unknown: "from-muted/40 to-muted/0",
}

/* ---------------------------------------------------------------------- */
/* derivations                                                            */
/* ---------------------------------------------------------------------- */

function describeConvergence(losses: number[]): ConvergenceVerdict | null {
  if (losses.length < 6) return null
  // Use leading 20% as the "early" window and trailing 20% as the "late"
  // window — bounds at min 3 / max 100 samples each so we don't compare
  // a single point to a single point on tiny runs and don't pay O(N) on
  // very long runs.
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

function describeStability(
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
  // CoV kept for the caption — an at-a-glance "is this curve quiet or
  // jagged" descriptor that complements SNR's directional signal.
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

function describeOverfit(
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

function describeLrResponse(
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
  // A "drop" is any step where lr fell by more than 10% relative to the
  // immediately-preceding lr. We tolerate small-noise drift (some
  // schedulers wobble within a step's worth of FP error) by demanding
  // a relative drop ≥ 0.10.
  const events: LrDropEvent[] = []
  // Smaller of "5% of total samples" and "16 steps" — the post-window
  // needs to be short enough that a second LR drop doesn't contaminate
  // it, long enough to average out per-step noise.
  const win = Math.min(16, Math.max(4, Math.floor(points.length * 0.05)))
  for (let i = 1; i < points.length - 1; i += 1) {
    const prevLr = points[i - 1].lr
    const curLr = points[i].lr
    if (prevLr <= 0 || curLr <= 0) continue
    const relDrop = (prevLr - curLr) / prevLr
    if (relDrop < 0.1) continue
    // pre-window: [i - win, i)
    const preStart = Math.max(0, i - win)
    const preSlice = points.slice(preStart, i)
    if (preSlice.length < 3) continue
    // post-window: (i, i + win] — start one step *after* the drop so
    // the inflection itself isn't inside the average.
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

function classifyStage(
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
  // First 25% of the run is treated as warm-up so a still-falling early
  // schedule isn't mislabelled as "converged".
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

/* ---------------------------------------------------------------------- */
/* component                                                              */
/* ---------------------------------------------------------------------- */

export function EffectivenessPanel({ metrics }: Props) {
  const points = useMemo<{ step: number; loss: number }[]>(() => {
    return (metrics?.loss ?? [])
      .filter(
        (p): p is { step: number; loss: number; ts: number } =>
          typeof p.loss === "number" && Number.isFinite(p.loss),
      )
      .map((p) => ({ step: p.step, loss: p.loss }))
  }, [metrics])
  const losses = useMemo(() => points.map((p) => p.loss), [points])

  const convergence = useMemo(() => describeConvergence(losses), [losses])
  const stability = useMemo(() => describeStability(points), [points])
  const overfit = useMemo(() => describeOverfit(metrics), [metrics])
  const lrResponse = useMemo(() => describeLrResponse(metrics), [metrics])
  const stage = useMemo(
    () => classifyStage(convergence, stability, losses),
    [convergence, stability, losses],
  )

  const dropPct = convergence?.dropPct ?? 0
  // Map -10%..+60% drop onto a 0..1 fill so users can read it as a bar.
  const dropFill = clamp((dropPct - -10) / 70, 0, 1)
  // Stability bar: |SNR| capped at 3 then mapped to [0..1]; visualises
  // "how much the trajectory has actually moved relative to within-
  // window dispersion" rather than raw variance.
  const stabFill = stability ? clamp(Math.min(Math.abs(stability.snr), 3) / 3, 0, 1) : 0
  const overfitFill =
    overfit.gap == null ? 0 : clamp(1 - overfit.gap / 0.2, 0, 1)
  // LR-response bar: mean improvement % capped at 10% maps to 1.0.
  const lrFill =
    lrResponse.events.length === 0
      ? 0
      : clamp(lrResponse.meanImprovementPct / 10, 0, 1)

  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80 mb-2 px-0.5">
        训练有效性洞察
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-3">
        <InsightCard
          icon={<Activity className="size-3.5" />}
          title="收敛趋势"
          tone={
            convergence?.state === "improving"
              ? "positive"
              : convergence?.state === "diverging"
                ? "negative"
                : "neutral"
          }
          headline={
            convergence
              ? `${convergence.state === "diverging" ? "+" : "−"}${Math.abs(
                  convergence.dropPct,
                ).toFixed(1)}%`
              : "—"
          }
          caption={
            convergence
              ? `早窗均值 ${fmtFloat(convergence.earlyMean)} → 晚窗均值 ${fmtFloat(convergence.lateMean)}`
              : "样本不足 6 点"
          }
          fill={dropFill}
          stagger={0}
          lowConfidence={!!convergence && convergence.samples < 24}
          rationale={
            convergence
              ? [
                  `样本: ${convergence.samples} 个 loss 点`,
                  `窗口: 头/尾各 ${Math.min(Math.max(3, Math.floor(convergence.samples * 0.2)), 100)} 个`,
                  `规则: 降幅 < −1.5% 视为发散; ≥ +3% 视为下降; 之间为平台`,
                  `结果: ${convergence.state === "improving" ? "下降中" : convergence.state === "diverging" ? "反向上升" : "平台"}`,
                ]
              : undefined
          }
        />
        <InsightCard
          icon={<Gauge className="size-3.5" />}
          title="趋势进展"
          tone={
            stability?.state === "progressing"
              ? "positive"
              : stability?.state === "diverging"
                ? "negative"
                : "neutral"
          }
          headline={
            stability
              ? `SNR ${stability.snr >= 0 ? "+" : ""}${stability.snr.toFixed(2)}`
              : "—"
          }
          caption={
            stability
              ? `${stabilityLabel(stability.state)} · 斜率 ${stability.slope.toExponential(1)} · IQR ${fmtFloat(stability.iqr)} · 近 ${stability.windowSamples} 步 · CoV ${stability.cov.toFixed(2)}`
              : "样本不足 8 点"
          }
          fill={stabFill}
          stagger={80}
          lowConfidence={!!stability && stability.windowSamples < 16}
          rationale={
            stability
              ? [
                  `窗口: 末尾 ${stability.windowSamples} 个 loss 点 (默认按总数 5% 取窗)`,
                  `斜率: ${stability.slope.toExponential(2)}/step (OLS, R² ${stability.rSquared.toFixed(3)})`,
                  `IQR: ${stability.iqr.toExponential(2)} (Q75 − Q25)`,
                  `SNR: |slope·windowSpan| / IQR = ${Math.abs(stability.snr).toFixed(2)}`,
                  `规则: |SNR| < 1 为停滞; SNR < 0 为下降; SNR > 0 为回升`,
                ]
              : undefined
          }
        />
        <InsightCard
          icon={<ShieldAlert className="size-3.5" />}
          title="过拟合风险"
          tone={
            overfit.severity === "warn"
              ? "negative"
              : overfit.severity === "watch"
                ? "neutral"
                : "positive"
          }
          headline={
            overfit.gap != null
              ? `gap ${overfit.gap >= 0 ? "+" : ""}${overfit.gap.toFixed(4)}`
              : "—"
          }
          caption={
            overfit.trainLatest != null && overfit.valLatest != null
              ? `train ${fmtFloat(overfit.trainLatest)} · val ${fmtFloat(overfit.valLatest)}${
                  overfit.trend ? ` · ${overfitTrendLabel(overfit.trend)}` : ""
                }`
              : "尚未产生验证 loss"
          }
          fill={overfitFill}
          stagger={160}
          lowConfidence={overfit.gap == null || overfit.trend == null}
          rationale={[
            overfit.trainLatest != null
              ? `最新 train: ${fmtFloat(overfit.trainLatest)}`
              : "暂无 train 损失",
            overfit.valLatest != null
              ? `最新 val: ${fmtFloat(overfit.valLatest)}`
              : "暂无 val 损失 — 请在配置中开启验证集",
            overfit.gap != null
              ? `gap = val − train = ${overfit.gap.toFixed(4)}`
              : "无法计算 gap",
            overfit.trend
              ? `后端趋势判定: ${overfitTrendLabel(overfit.trend)} (依据 train/val 同窗对比)`
              : "趋势依据不足",
            "阈值: gap > 0.05 视为 watch; trend=overfitting 视为 warn",
          ]}
        />
        <InsightCard
          icon={<Zap className="size-3.5" />}
          title="LR 响应度"
          tone={
            lrResponse.state === "responsive"
              ? "positive"
              : lrResponse.state === "weak"
                ? "negative"
                : "neutral"
          }
          headline={
            lrResponse.events.length === 0
              ? "无下降事件"
              : `${lrResponse.meanImprovementPct >= 0 ? "+" : ""}${lrResponse.meanImprovementPct.toFixed(1)}%`
          }
          caption={
            lrResponse.events.length === 0
              ? "学习率尚未发生显著下降, 或后端未上报 lr"
              : `${lrResponse.responsiveEvents}/${lrResponse.events.length} 次下降产生改善 · ${lrResponseLabel(lrResponse.state)}`
          }
          fill={lrFill}
          stagger={240}
          lowConfidence={lrResponse.events.length > 0 && lrResponse.events.length < 2}
          rationale={
            lrResponse.events.length === 0
              ? [
                  "未识别到 lr 相对下降 ≥ 10% 的步数",
                  "若使用 cosine 衰减且 step 间下降幅度 < 10%, 本卡仅在分段调度时有数据",
                ]
              : [
                  `识别 ${lrResponse.events.length} 次相对下降 ≥ 10% 的事件`,
                  `每次下降的前/后 ≤ 16 步窗口内取 loss 均值, 计算改善百分比`,
                  `事件平均改善: ${lrResponse.meanImprovementPct.toFixed(2)}%`,
                  `阈值: 单次改善 > 0.5% 视为有效; 有效率 ≥ 60% 视为敏感`,
                  ...lrResponse.events.slice(-3).map(
                    (e) =>
                      `step ${e.step}: ${e.lrBefore.toExponential(1)} → ${e.lrAfter.toExponential(1)}, Δloss ${e.improvementPct >= 0 ? "+" : ""}${e.improvementPct.toFixed(2)}%`,
                  ),
                ]
          }
        />
        <StageCard
          stage={stage.key}
          reason={stage.reason}
          stagger={320}
          rationale={[
            `阶段判定参考: 收敛趋势 + 趋势进展`,
            convergence
              ? `收敛: ${convergence.state}, 降幅 ${convergence.dropPct.toFixed(1)}%`
              : "收敛: 数据不足",
            stability
              ? `进展: ${stability.state}, SNR ${stability.snr.toFixed(2)}`
              : "进展: 数据不足",
            losses.length < 32
              ? `当前样本 ${losses.length} 点 (< 32) 时下降态默认为热身阶段`
              : `当前样本 ${losses.length} 点`,
          ]}
        />
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------------- */
/* card primitives                                                        */
/* ---------------------------------------------------------------------- */

type Tone = "positive" | "neutral" | "negative"

const TONE_FILL: Record<Tone, string> = {
  positive: "bg-emerald-500/70",
  neutral: "bg-amber-500/70",
  negative: "bg-red-500/70",
}

const TONE_TEXT: Record<Tone, string> = {
  positive: "text-emerald-700 dark:text-emerald-300",
  neutral: "text-amber-700 dark:text-amber-300",
  negative: "text-red-600 dark:text-red-400",
}

function InsightCard({
  icon,
  title,
  tone,
  headline,
  caption,
  fill,
  stagger,
  rationale,
  lowConfidence,
}: {
  icon: React.ReactNode
  title: string
  tone: Tone
  headline: string
  caption: string
  fill: number
  stagger: number
  /** Bullet-style strings shown under the "判定依据" disclosure. */
  rationale?: string[]
  /**
   * Optional flag rendered as a "低置信度" pill. Use it when the
   * verdict is computed from too few samples or noisy partial data so
   * the user knows to weigh it accordingly.
   */
  lowConfidence?: boolean
}) {
  return (
    <Card
      className="analysis-fade-in-stagger overflow-hidden"
      style={{ ["--stagger-delay" as string]: `${stagger}ms` }}
    >
      <CardHeader className="py-2 px-3.5 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
        <CardTitle className="text-[10.5px] tracking-[0.16em] text-foreground/85 font-mono inline-flex items-center gap-1.5">
          <span className={cn("opacity-80", TONE_TEXT[tone])}>{icon}</span>
          {title}
        </CardTitle>
        {lowConfidence && (
          <span className="rounded-[3px] border border-amber-500/40 bg-amber-500/10 px-1.5 py-0 text-[9.5px] uppercase tracking-[0.14em] text-amber-700 dark:text-amber-300">
            低置信
          </span>
        )}
      </CardHeader>
      <CardContent className="p-3.5 space-y-2">
        <div className={cn("text-[18px] font-semibold tracking-tight tabular-nums", TONE_TEXT[tone])}>
          {headline}
        </div>
        <div className="h-1.5 rounded-full bg-muted/60 overflow-hidden">
          <div
            className={cn("analysis-bar-fill h-full rounded-full", TONE_FILL[tone])}
            style={{ width: `${(fill * 100).toFixed(1)}%` }}
          />
        </div>
        <div className="text-[11px] text-muted-foreground leading-relaxed">
          {caption}
        </div>
        {rationale && rationale.length > 0 && (
          <details className="group text-[10.5px] text-muted-foreground/85 mt-1.5">
            <summary className="cursor-pointer select-none inline-flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors">
              <span className="inline-block transition-transform group-open:rotate-90">
                ▸
              </span>
              判定依据
            </summary>
            <ul className="mt-1 space-y-0.5 pl-3 leading-relaxed">
              {rationale.map((r, i) => (
                <li key={i} className="font-mono">
                  · {r}
                </li>
              ))}
            </ul>
          </details>
        )}
      </CardContent>
    </Card>
  )
}

function StageCard({
  stage,
  reason,
  stagger,
  rationale,
}: {
  stage: StageKey
  reason: string
  stagger: number
  rationale?: string[]
}) {
  return (
    <Card
      className="analysis-fade-in-stagger overflow-hidden relative"
      style={{ ["--stagger-delay" as string]: `${stagger}ms` }}
    >
      <div
        className={cn(
          "pointer-events-none absolute inset-0 bg-gradient-to-br opacity-70",
          STAGE_BG[stage],
        )}
        aria-hidden
      />
      <CardHeader className="py-2 px-3.5 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2 relative">
        <CardTitle className="text-[10.5px] tracking-[0.16em] text-foreground/85 font-mono inline-flex items-center gap-1.5">
          <span className={cn("opacity-80", STAGE_TONES[stage])}>
            <Sparkles className="size-3.5" />
          </span>
          训练阶段
        </CardTitle>
      </CardHeader>
      <CardContent className="p-3.5 space-y-2 relative">
        <div className={cn("text-[18px] font-semibold tracking-tight", STAGE_TONES[stage])}>
          {STAGE_LABELS[stage]}
        </div>
        <StageDots stage={stage} />
        <div className="text-[11px] text-muted-foreground leading-relaxed min-h-[1.2em]">
          {reason || "等待更多训练数据"}
        </div>
        {rationale && rationale.length > 0 && (
          <details className="group text-[10.5px] text-muted-foreground/85 mt-1">
            <summary className="cursor-pointer select-none inline-flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors">
              <span className="inline-block transition-transform group-open:rotate-90">▸</span>
              判定依据
            </summary>
            <ul className="mt-1 space-y-0.5 pl-3 leading-relaxed">
              {rationale.map((r, i) => (
                <li key={i} className="font-mono">
                  · {r}
                </li>
              ))}
            </ul>
          </details>
        )}
      </CardContent>
    </Card>
  )
}

const STAGE_ORDER: StageKey[] = ["warmup", "converging", "plateau", "diverging"]

function StageDots({ stage }: { stage: StageKey }) {
  const idx = STAGE_ORDER.indexOf(stage)
  return (
    <div className="flex items-center gap-1.5" aria-label={`阶段: ${STAGE_LABELS[stage]}`}>
      {STAGE_ORDER.map((s, i) => {
        const isActive = s === stage
        const isPassed = idx >= 0 && i < idx
        return (
          <span
            key={s}
            className={cn(
              "inline-block h-1.5 flex-1 rounded-full transition-all duration-500",
              isActive
                ? "bg-foreground/85 scale-y-[1.4]"
                : isPassed
                  ? "bg-foreground/40"
                  : "bg-muted/60",
            )}
            aria-hidden
          />
        )
      })}
    </div>
  )
}

/* ---------------------------------------------------------------------- */
/* helpers                                                                */
/* ---------------------------------------------------------------------- */

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

function fmtFloat(v: number): string {
  if (!Number.isFinite(v)) return "—"
  if (Math.abs(v) >= 100) return v.toFixed(1)
  if (Math.abs(v) >= 1) return v.toFixed(3)
  if (Math.abs(v) >= 0.01) return v.toFixed(4)
  if (v === 0) return "0"
  return v.toExponential(2)
}

function pctFormat(v: number): string {
  return `${v.toFixed(1)}%`
}

function stabilityLabel(s: StabilityVerdict["state"]): string {
  return s === "progressing" ? "仍在显著下降" : s === "diverging" ? "近窗回升" : "近窗停滞"
}

function lrResponseLabel(s: LrResponseVerdict["state"]): string {
  return s === "responsive"
    ? "对 LR 敏感"
    : s === "weak"
      ? "对 LR 不敏感"
      : "暂无可分析事件"
}

function overfitTrendLabel(t: OverfitTrend): string {
  return t === "improving" ? "持续改进" : t === "flat" ? "趋于平稳" : "出现过拟合"
}
