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
} from "lucide-react"
import { cn } from "@/lib/utils"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { JobMetricsResponse, OverfitTrend } from "@/lib/api"

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
  /** Coefficient of variation on the late window (std / |mean|). */
  cov: number
  /** "stable" if cov < 0.05, "noisy" if 0.05–0.15, "thrashing" beyond. */
  state: "stable" | "noisy" | "thrashing"
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

function describeStability(losses: number[]): StabilityVerdict | null {
  if (losses.length < 8) return null
  const winSize = Math.min(Math.max(8, Math.floor(losses.length * 0.3)), 200)
  const window = losses.slice(-winSize)
  const mean = window.reduce((a, b) => a + b, 0) / window.length
  if (Math.abs(mean) < 1e-9) return null
  const variance =
    window.reduce((acc, v) => acc + (v - mean) ** 2, 0) / window.length
  const cov = Math.sqrt(variance) / Math.abs(mean)
  let state: StabilityVerdict["state"]
  if (cov < 0.05) state = "stable"
  else if (cov < 0.15) state = "noisy"
  else state = "thrashing"
  return { cov, state, windowSamples: window.length }
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
    if (stab && stab.state === "thrashing") {
      return {
        key: "diverging",
        reason: `平台期但波动剧烈 (CoV ${stab.cov.toFixed(2)})`,
      }
    }
    return {
      key: "plateau",
      reason: stab
        ? `近窗均值变化 < 3%, CoV ${stab.cov.toFixed(2)}`
        : "近窗均值变化 < 3%",
    }
  }
  return { key: "unknown", reason: "" }
}

/* ---------------------------------------------------------------------- */
/* component                                                              */
/* ---------------------------------------------------------------------- */

export function EffectivenessPanel({ metrics }: Props) {
  const losses = useMemo<number[]>(() => {
    return (metrics?.loss ?? [])
      .map((p) => p.loss)
      .filter((v): v is number => typeof v === "number" && Number.isFinite(v))
  }, [metrics])

  const convergence = useMemo(() => describeConvergence(losses), [losses])
  const stability = useMemo(() => describeStability(losses), [losses])
  const overfit = useMemo(() => describeOverfit(metrics), [metrics])
  const stage = useMemo(
    () => classifyStage(convergence, stability, losses),
    [convergence, stability, losses],
  )

  const dropPct = convergence?.dropPct ?? 0
  // Map -10%..+60% drop onto a 0..1 fill so users can read it as a bar.
  const dropFill = clamp((dropPct - -10) / 70, 0, 1)
  const stabFill = stability ? clamp(1 - stability.cov / 0.3, 0, 1) : 0
  const overfitFill =
    overfit.gap == null ? 0 : clamp(1 - overfit.gap / 0.2, 0, 1)

  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80 mb-2 px-0.5">
        训练有效性洞察
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
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
        />
        <InsightCard
          icon={<Gauge className="size-3.5" />}
          title="平稳度"
          tone={
            stability?.state === "stable"
              ? "positive"
              : stability?.state === "thrashing"
                ? "negative"
                : "neutral"
          }
          headline={stability ? `CoV ${stability.cov.toFixed(3)}` : "—"}
          caption={
            stability
              ? `${stabilityLabel(stability.state)} · 近 ${stability.windowSamples} 步`
              : "样本不足 8 点"
          }
          fill={stabFill}
          stagger={80}
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
        />
        <StageCard
          stage={stage.key}
          reason={stage.reason}
          stagger={240}
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
}: {
  icon: React.ReactNode
  title: string
  tone: Tone
  headline: string
  caption: string
  fill: number
  stagger: number
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
      </CardContent>
    </Card>
  )
}

function StageCard({
  stage,
  reason,
  stagger,
}: {
  stage: StageKey
  reason: string
  stagger: number
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
  return s === "stable" ? "稳定" : s === "noisy" ? "存在抖动" : "波动剧烈"
}

function overfitTrendLabel(t: OverfitTrend): string {
  return t === "improving" ? "持续改进" : t === "flat" ? "趋于平稳" : "出现过拟合"
}
