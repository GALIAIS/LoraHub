/**
 * Slim summary strip rendered under the analysis header. Keeps the user
 * situated — they don't need to flip back to the job-detail page just
 * to remember progress or wall time. Pulls everything from existing
 * endpoints (`/jobs/{id}` for run state, `/jobs/{id}/metrics` for
 * the live counters).
 */
import { useQuery } from "@tanstack/react-query"
import { Hourglass, Layers, ListChecks, Save, TrendingDown } from "lucide-react"
import { api } from "@/lib/api"
import { fmtDuration, TERMINAL_STATES } from "../../jobs/utils"
import { cn } from "@/lib/utils"

export function JobSummaryStrip({ jobId }: { jobId: string }) {
  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.getJob(jobId),
    refetchInterval: 4000,
  })
  const isTerminal = job.data ? TERMINAL_STATES.has(job.data.state) : false

  const metrics = useQuery({
    queryKey: ["job-metrics", jobId],
    queryFn: () => api.getJobMetrics(jobId),
    refetchInterval: isTerminal ? false : 4000,
  })

  const m = metrics.data
  const lastStep =
    m && m.loss.length > 0 ? m.loss[m.loss.length - 1].step : null
  const lastLoss = m && m.loss.length > 0 ? m.loss[m.loss.length - 1].loss : null
  const checkpoints = m?.checkpoints?.length ?? 0
  const samples = m?.samples?.length ?? 0
  const wallSec =
    m?.first_step_ts != null && m?.last_step_ts != null
      ? m.last_step_ts - m.first_step_ts
      : null

  return (
    <div className="px-7 py-2 border-b border-border/60 bg-muted/30">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 text-[11px] tabular-nums text-foreground/85">
        <Stat icon={<Hourglass className="size-3" />} label="进度">
          {lastStep != null ? lastStep : "—"}
        </Stat>
        <Stat icon={<TrendingDown className="size-3" />} label="最新损失">
          {typeof lastLoss === "number" ? lastLoss.toFixed(4) : "—"}
        </Stat>
        <Stat icon={<Save className="size-3" />} label="检查点">
          {checkpoints}
        </Stat>
        <Stat icon={<Layers className="size-3" />} label="样本">
          {samples}
        </Stat>
        <Stat icon={<ListChecks className="size-3" />} label="用时">
          {fmtDuration(wallSec)}
        </Stat>
      </div>
    </div>
  )
}

function Stat({
  icon,
  label,
  children,
  tone,
}: {
  icon: React.ReactNode
  label: string
  children: React.ReactNode
  tone?: string
}) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="text-muted-foreground/70" aria-hidden>
        {icon}
      </span>
      <span className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground/80">
        {label}
      </span>
      <span className={cn("text-foreground/90", tone)}>{children}</span>
    </span>
  )
}
