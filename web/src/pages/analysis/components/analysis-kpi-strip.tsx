/**
 * KPI strip — single dense row of headline numbers shown directly below
 * the analysis page header. Replaces the wider `<JobSummaryStrip>` and
 * the collapsed-state digest of `RunSummaryCard`, so the user has one
 * place to glance at "is this run healthy" without expanding cards or
 * jumping back to the jobs page.
 *
 * Numbers come from the same `/jobs/{id}` + `/jobs/{id}/metrics`
 * endpoints the rest of the workbench polls; data freshness is
 * uniform with the metric tabs (4 s while running, off when terminal).
 */
import { useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import {
  AlertTriangle,
  ArrowDownRight,
  Hourglass,
  ListChecks,
  Save,
  Sparkles,
  TrendingDown,
} from "lucide-react"
import { api } from "@/lib/api"
import type { JobMetricsResponse, JobSummary } from "@/lib/api"
import { fmtDuration, TERMINAL_STATES } from "../../jobs/utils"
import { cn } from "@/lib/utils"
import type { AnalysisBackendInfo } from "./analysis-backend"

interface Props {
  job: JobSummary | undefined
  fallbackTotalSteps: number | null
  backend: AnalysisBackendInfo
}

export function AnalysisKpiStrip({ job, fallbackTotalSteps, backend }: Props) {
  const isTerminal = job ? TERMINAL_STATES.has(job.state) : false
  const metrics = useQuery({
    queryKey: ["job-metrics", job?.id],
    queryFn: () => api.getJobMetrics(job!.id),
    enabled: !!job?.id,
    refetchInterval: isTerminal ? false : 4000,
    staleTime: 2_000,
  })

  const summary = useMemo(
    () => deriveKpis(metrics.data, fallbackTotalSteps),
    [metrics.data, fallbackTotalSteps],
  )

  const overfit = metrics.data?.overfit_signal
  const overfitWarn = overfit?.trend === "overfitting"

  return (
    <div className="px-7 py-2.5 border-b border-border/60 bg-muted/25">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 text-[11px] tabular-nums text-foreground/85">
        <Stat icon={<Hourglass className="size-3" />} label="进度">
          {summary.step != null ? (
            <span className="font-semibold">{summary.step}</span>
          ) : (
            <Missing reason="尚未收到 step 事件" label="—" />
          )}
          <span className="text-muted-foreground/80">
            {" "}
            /{" "}
            {summary.totalSteps != null ? (
              summary.totalSteps
            ) : (
              <Missing
                reason="后端未上报 total_steps, 配置中也无法推导"
                label="?"
              />
            )}
            {summary.percent != null && (
              <span className="ml-1 text-foreground/70">
                {summary.percent.toFixed(0)}%
              </span>
            )}
          </span>
        </Stat>
        <Stat icon={<TrendingDown className="size-3" />} label="损失">
          {summary.lossLatest != null ? (
            <>
              <span className="font-semibold">
                {summary.lossLatest.toFixed(4)}
              </span>
              {summary.lossDropPct != null && (
                <span
                  className={cn(
                    "ml-1.5",
                    summary.lossDropPct > 0
                      ? "text-emerald-600 dark:text-emerald-400"
                      : "text-amber-700 dark:text-amber-400",
                  )}
                >
                  {summary.lossDropPct > 0 ? "↓" : "↑"}{" "}
                  {Math.abs(summary.lossDropPct).toFixed(1)}%
                </span>
              )}
            </>
          ) : (
            <Missing
              reason={
                summary.nonfiniteLoss
                  ? "后端上报 loss=NaN/Inf, 曲线会跳过该点"
                  : "尚未收到任何 loss 采样"
              }
              label={summary.nonfiniteLoss ? "NaN" : "—"}
            />
          )}
        </Stat>
        <Stat icon={<ArrowDownRight className="size-3" />} label="验证">
          {summary.valLatest != null ? (
            <>
              <span className="font-semibold">
                {summary.valLatest.toFixed(4)}
              </span>
              {summary.valGap != null && (
                <span
                  className={cn(
                    "ml-1.5",
                    overfitWarn
                      ? "text-red-600 dark:text-red-400"
                      : "text-muted-foreground",
                  )}
                >
                  Δ {summary.valGap > 0 ? "+" : ""}
                  {summary.valGap.toFixed(4)}
                </span>
              )}
            </>
          ) : (
            <Missing
              reason={
                summary.nonfiniteValLoss
                  ? "后端上报 val_loss=NaN/Inf, 曲线会跳过该点"
                  : backend.supportsValidation === false
                    ? "ai_toolkit 当前未提供验证损失，采用单曲线分析"
                    : backend.validationConfigured === false
                      ? "当前配置未启用验证集"
                      : backend.validationConfigured === true
                        ? "验证已配置，但后端尚未上报 val_loss"
                        : "后端尚未上报验证损失"
              }
              label={
                summary.nonfiniteValLoss
                  ? "NaN"
                  : backend.supportsValidation === false
                    ? "未提供"
                    : backend.validationConfigured === false
                      ? "未启用"
                      : backend.validationConfigured === true
                        ? "等待"
                        : "未上报"
              }
            />
          )}
        </Stat>
        <Stat icon={<Save className="size-3" />} label="检查点">
          {summary.checkpoints}
        </Stat>
        <Stat icon={<Sparkles className="size-3" />} label="样本">
          {summary.samples}
        </Stat>
        <Stat icon={<ListChecks className="size-3" />} label="用时">
          {summary.wallSec != null ? (
            fmtDuration(summary.wallSec)
          ) : (
            <Missing reason="任务尚未产生第一步, 无法计算用时" label="—" />
          )}
          {summary.etaSec != null && summary.etaSec > 0 && (
            <span className="text-muted-foreground/80">
              {" "}
              · ETA {fmtDuration(summary.etaSec)}
            </span>
          )}
        </Stat>
        {overfitWarn && (
          <span className="ml-auto inline-flex items-center gap-1 rounded-[3px] border border-red-500/40 bg-red-500/10 px-1.5 py-0.5 text-red-600 dark:text-red-400">
            <AlertTriangle className="size-3" />
            疑似过拟合
          </span>
        )}
      </div>
    </div>
  )
}

function Stat({
  icon,
  label,
  children,
}: {
  icon: React.ReactNode
  label: string
  children: React.ReactNode
}) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="text-muted-foreground/70" aria-hidden>
        {icon}
      </span>
      <span className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground/80">
        {label}
      </span>
      <span className="text-foreground/90">{children}</span>
    </span>
  )
}

/**
 * Compact placeholder for missing values. Carries a `title` tooltip
 * with the reason so the user knows whether the data is genuinely
 * absent (not yet sampled), unsupported (backend doesn't emit it) or
 * filtered out.
 */
function Missing({
  reason,
  label = "—",
}: {
  reason: string
  label?: string
}) {
  return (
    <span
      className="cursor-help text-muted-foreground/70 underline decoration-dotted decoration-muted-foreground/40 underline-offset-2"
      title={reason}
    >
      {label}
    </span>
  )
}

interface Kpis {
  step: number | null
  totalSteps: number | null
  percent: number | null
  lossLatest: number | null
  lossDropPct: number | null
  valLatest: number | null
  valGap: number | null
  checkpoints: number
  samples: number
  wallSec: number | null
  etaSec: number | null
  nonfiniteLoss: boolean
  nonfiniteValLoss: boolean
}

function deriveKpis(
  m: JobMetricsResponse | undefined,
  fallbackTotalSteps: number | null,
): Kpis {
  const points = (m?.loss ?? []).filter(
    (p): p is { step: number; loss: number; ts: number } =>
      typeof p.loss === "number" && Number.isFinite(p.loss),
  )
  const lossStart = points.length > 0 ? points[0].loss : null
  const lossLatest = points.length > 0 ? points[points.length - 1].loss : null
  const lossDropPct =
    lossStart != null && lossLatest != null && lossStart > 0
      ? ((lossStart - lossLatest) / lossStart) * 100
      : null
  const step = points.length > 0 ? points[points.length - 1].step : null
  const lastStep =
    typeof m?.last_step === "number" && Number.isFinite(m.last_step)
      ? m.last_step
      : step
  // Trainer-reported total_steps wins (kohya / anima emit it on every
  // step event); fall back to the parent-supplied config-derived
  // estimate when dp doesn't emit one. Same priority as the overview
  // tab so the two screens agree.
  const totalSteps =
    typeof m?.total_steps === "number" && m.total_steps > 0
      ? m.total_steps
      : (fallbackTotalSteps ?? null)
  const percent =
    lastStep != null && totalSteps != null && totalSteps > 0
      ? (lastStep / totalSteps) * 100
      : null
  const wallSec =
    m?.first_step_ts != null && m?.last_step_ts != null
      ? m.last_step_ts - m.first_step_ts
      : null
  const reportedEta = [...(m?.loss ?? [])]
    .reverse()
    .find((point) => typeof point.eta_s === "number")?.eta_s
  const etaSec =
    typeof reportedEta === "number"
      ? reportedEta
      : wallSec != null && lastStep != null && totalSteps != null && lastStep > 0
        ? (wallSec / lastStep) * Math.max(0, totalSteps - lastStep)
        : null
  const vals = m?.val_loss ?? []
  const valLatest =
    vals.length > 0
      ? typeof vals[vals.length - 1].val_loss === "number"
        ? vals[vals.length - 1].val_loss
        : null
      : null
  const valGap =
    valLatest != null && lossLatest != null ? valLatest - lossLatest : null
  return {
    step: lastStep,
    totalSteps,
    percent,
    lossLatest,
    lossDropPct,
    valLatest,
    valGap,
    checkpoints: m?.checkpoints?.length ?? 0,
    samples: m?.samples?.length ?? 0,
    wallSec,
    etaSec,
    nonfiniteLoss: !!m?.last_nonfinite_loss,
    nonfiniteValLoss: !!m?.last_nonfinite_val_loss,
  }
}
