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
  Layers,
  ShieldAlert,
  Zap,
} from "lucide-react"
import type { JobMetricsResponse, OverfitTrend } from "@/lib/api"
import {
  classifyStage,
  describeConvergence,
  describeForgetting,
  describeLrResponse,
  describeOverfit,
  describeStability,
  type ForgettingVerdict,
  type LrResponseVerdict,
  type StabilityVerdict,
} from "./effectiveness-model"
import { InsightCard, StageCard, type Tone } from "./effectiveness-cards"
import type { AnalysisBackendInfo } from "./analysis-backend"

interface Props {
  metrics: JobMetricsResponse | null
  /**
   * Estimated training progress in [0..1]. Used to make the
   * convergence + stage tones context-aware: a plateau in the last
   * 30% of a run is a good thing (the model has settled); a plateau
   * in the first 30% is bad (nothing is being learned). Pass null
   * when total steps is unknown — colours fall back to a neutral
   * baseline.
   */
  progress?: number | null
  backend: AnalysisBackendInfo
}

/* ---------------------------------------------------------------------- */
/* component                                                              */
/* ---------------------------------------------------------------------- */

export function EffectivenessPanel({ metrics, progress, backend }: Props) {
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
  const forgetting = useMemo(() => describeForgetting(metrics), [metrics])
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
  // Forgetting bar: preserved in [0..1] is itself the fill.
  const forgetFill = forgetting.latest == null ? 0 : clamp(forgetting.latest, 0, 1)

  // Context-aware tones for convergence + stage. A plateau in the
  // last 30% of a run is a *positive* signal (the model converged);
  // the same plateau in the first 30% is a *warning* (nothing is
  // being learned). When progress is unknown, fall back to the
  // state-machine defaults.
  const convergenceTone: Tone = (() => {
    if (!convergence) return "neutral"
    if (convergence.state === "improving") return "positive"
    if (convergence.state === "diverging") return "negative"
    // plateau
    if (progress != null && progress >= 0.7) return "positive"
    if (progress != null && progress < 0.3) return "negative"
    return "neutral"
  })()
  const stageTone: Tone = (() => {
    if (stage.key === "diverging") return "negative"
    if (stage.key === "warmup" || stage.key === "converging") return "positive"
    if (stage.key === "plateau") {
      if (progress != null && progress >= 0.7) return "positive"
      if (progress != null && progress < 0.3) return "negative"
      return "neutral"
    }
    return "neutral"
  })()

  return (
    <div>
      <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80 mb-2 px-0.5">
        训练有效性洞察
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
        <InsightCard
          icon={<Activity className="size-3.5" />}
          title="收敛趋势"
          tone={convergenceTone}
          headline={
            convergence
              ? `${convergence.state === "diverging" ? "+" : "−"}${Math.abs(
                  convergence.dropPct,
                ).toFixed(1)}%`
              : "—"
          }
          headlineNumber={convergence?.dropPct ?? null}
          formatHeadline={(v) =>
            `${v < 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}%`
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
          headlineNumber={stability?.snr ?? null}
          formatHeadline={(v) => `SNR ${v >= 0 ? "+" : ""}${v.toFixed(2)}`}
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
            overfit.gap == null
              ? "neutral"
              : overfit.severity === "warn"
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
          headlineNumber={overfit.gap ?? null}
          formatHeadline={(v) =>
            `gap ${v >= 0 ? "+" : ""}${v.toFixed(4)}`
          }
          caption={
            overfit.trainLatest != null && overfit.valLatest != null
              ? `train ${fmtFloat(overfit.trainLatest)} · val ${fmtFloat(overfit.valLatest)}${
                  overfit.trend ? ` · ${overfitTrendLabel(overfit.trend)}` : ""
                }`
              : backend.supportsValidation === false
                ? "单曲线模式，ai_toolkit 未上报验证 loss"
                : backend.validationConfigured === false
                  ? "单曲线模式，当前配置未启用验证集"
                  : backend.validationConfigured === true
                    ? "等待验证 loss，暂时无法计算 train–val gap"
                    : "后端未上报验证 loss，暂时采用单曲线模式"
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
              : backend.supportsValidation === false
                ? "ai_toolkit 当前没有 val_loss 数据源"
                : backend.validationConfigured === false
                  ? "当前配置未启用验证集"
                  : backend.validationConfigured === true
                    ? "验证已配置，等待后端上报"
                    : "后端未上报验证损失",
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
          headlineNumber={
            lrResponse.events.length === 0
              ? null
              : lrResponse.meanImprovementPct
          }
          formatHeadline={(v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`}
          caption={
            lrResponse.events.length === 0
              ? "未识别到学习率下降事件，或后端未上报 lr"
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
        <InsightCard
          icon={<Layers className="size-3.5" />}
          title="基模型保留度"
          tone={
            forgetting.state === "stable"
              ? "positive"
              : forgetting.state === "forgetting"
                ? "negative"
                : forgetting.state === "drifting"
                  ? "neutral"
                  : "neutral"
          }
          headline={
            forgetting.latest == null
              ? "未配置"
              : `${(forgetting.latest * 100).toFixed(0)}%`
          }
          headlineNumber={
            forgetting.latest == null ? null : forgetting.latest * 100
          }
          formatHeadline={(v) => `${v.toFixed(0)}%`}
          caption={
            forgetting.state === "no-data"
              ? "在 sample_prompts 中给中性 prompt 加 forget/neutral/preserve 标记后启用"
              : `${forgettingLabel(forgetting.state)} · 已采样 ${forgetting.samples} 次${
                  forgetting.trend != null
                    ? ` · 趋势 ${forgetting.trend >= 0 ? "+" : ""}${forgetting.trend.toExponential(1)}/step`
                    : ""
                }`
          }
          fill={forgetFill}
          stagger={400}
          lowConfidence={
            forgetting.state !== "no-data" && forgetting.samples < 3
          }
          rationale={
            forgetting.state === "no-data"
              ? [
                  "未识别到中性 prompt 样本",
                  "在 sample_prompts.txt 中将提示词文件名/索引标记为 forget/neutral/preserve",
                  "比较算法: 8x8 dHash + 64-bit Hamming 距离, 每张图 < 5ms",
                ]
              : [
                  `已采样 ${forgetting.samples} 张中性 prompt 输出`,
                  `相似度计算: 与该 prompt 最早一张样本的 dHash 比较`,
                  `阈值: < 0.65 视为遗忘; 0.65-0.85 或趋势负 视为漂移; ≥ 0.85 稳定`,
                  forgetting.trend != null
                    ? `趋势 OLS 斜率: ${forgetting.trend.toExponential(2)}/step`
                    : "样本不足以拟合趋势",
                ]
          }
        />
        <StageCard
          stage={stage.key}
          stageTone={stageTone}
          reason={stage.reason}
          stagger={480}
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
            progress != null
              ? `训练进度: ${(progress * 100).toFixed(0)}%, 用于上下文感知配色`
              : "训练进度未知, 颜色按状态默认值",
          ]}
        />
      </div>
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

function stabilityLabel(s: StabilityVerdict["state"]): string {
  return s === "progressing" ? "近窗下降" : s === "diverging" ? "近窗回升" : "近窗停滞"
}

function lrResponseLabel(s: LrResponseVerdict["state"]): string {
  return s === "responsive"
    ? "对 LR 敏感"
    : s === "weak"
      ? "对 LR 不敏感"
      : "暂无可分析事件"
}

function forgettingLabel(s: ForgettingVerdict["state"]): string {
  return s === "stable"
    ? "稳定"
    : s === "drifting"
      ? "漂移中"
      : s === "forgetting"
        ? "遗忘明显"
        : "未配置"
}

function overfitTrendLabel(t: OverfitTrend): string {
  return t === "improving" ? "持续改进" : t === "flat" ? "趋于平稳" : "出现过拟合"
}
