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

interface Props {
  job: JobSummary | undefined
  fallbackTotalSteps: number | null
}

export function AnalysisKpiStrip({ job, fallbackTotalSteps }: Props) {
  const isTerminal = job ? TERMINAL_STATES.has(job.state) : false
  const metrics = useQuery({
    queryKey: ["job-metrics", job?.id],
    queryFn: () => api.getJobMetrics(job!.id),
    enabled: !!job?.id,
    refetchInterval: isTerminal ? false : 4000,
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
          <span className="font-semibold">{summary.step ?? "—"}</span>
          <span className="text-muted-foreground/80">
            {" "}
            / {summary.totalSteps ?? "?"}
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
            "—"
          )}
        </Stat>
        {summary.valLatest != null && (
          <Stat icon={<ArrowDownRight className="size-3" />} label="验证">
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
          </Stat>
        )}
        <Stat icon={<Save className="size-3" />} label="检查点">
          {summary.checkpoints}
        </Stat>
        <Stat icon={<Sparkles className="size-3" />} label="样本">
          {summary.samples}
        </Stat>
        <Stat icon={<ListChecks className="size-3" />} label="用时">
          {fmtDuration(summary.wallSec)}
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
  // Trainer-reported total_steps wins (kohya / anima emit it on every
  // step event); fall back to the parent-supplied config-derived
  // estimate when dp doesn't emit one. Same priority as the overview
  // tab so the two screens agree.
  const totalSteps =
    typeof m?.total_steps === "number" && m.total_steps > 0
      ? m.total_steps
      : (fallbackTotalSteps ?? null)
  const percent =
    step != null && totalSteps != null && totalSteps > 0
      ? (step / totalSteps) * 100
      : null
  const wallSec =
    m?.first_step_ts != null && m?.last_step_ts != null
      ? m.last_step_ts - m.first_step_ts
      : null
  const etaSec =
    wallSec != null && step != null && totalSteps != null && step > 0
      ? (wallSec / step) * Math.max(0, totalSteps - step)
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
    step,
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
  }
}
