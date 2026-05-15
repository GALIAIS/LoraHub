import { useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import { api, type JobSummary, type TrainingEvent } from "@/lib/api"
import { Stat } from "./stat"
import { fmtDuration, fmtUnixSeconds, stateLabel, TERMINAL_STATES } from "../utils"

export function OverviewTab({
  jobId,
  job,
  events,
}: {
  jobId: string
  job: JobSummary | undefined
  events: TrainingEvent[]
}) {
  const lastStep = useMemo(
    () => [...events].reverse().find((e) => e.type === "step"),
    [events],
  )
  const isTerminal = job ? TERMINAL_STATES.has(job.state) : false

  // Metrics (first/last step ts, duration) are only meaningful once at least
  // one step has been recorded. Refetch while live so the panel stays fresh.
  const metrics = useQuery({
    queryKey: ["job-metrics", jobId],
    queryFn: () => api.getJobMetrics(jobId),
    refetchInterval: isTerminal ? false : 4000,
  })

  const m = metrics.data

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-3 gap-3">
        <Stat label="状态" value={job?.state ? stateLabel(job.state) : "—"} />
        <Stat
          label="进度"
          value={
            lastStep
              ? `${lastStep.payload.step ?? "?"} / ${
                  lastStep.payload.total_steps ?? "?"
                }`
              : "—"
          }
        />
        <Stat
          label="最新损失"
          value={
            typeof lastStep?.payload.loss === "number"
              ? (lastStep.payload.loss as number).toFixed(4)
              : "—"
          }
        />
      </div>

      <div>
        <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80 mb-2">
          指标摘要
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Stat
            label="首步时间"
            value={m?.first_step_ts ? fmtUnixSeconds(m.first_step_ts) : "—"}
          />
          <Stat
            label="最新步时间"
            value={m?.last_step_ts ? fmtUnixSeconds(m.last_step_ts) : "—"}
          />
          <Stat label="累计用时" value={fmtDuration(m?.duration_s)} />
          <Stat
            label="已采样步数"
            value={m ? String(m.loss.length) : "—"}
          />
        </div>
      </div>

      <div>
        <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80 mb-2">
          产物计数
        </div>
        <div className="grid grid-cols-3 gap-3">
          <Stat label="检查点" value={m ? String(m.checkpoints.length) : "—"} />
          <Stat label="样本" value={m ? String(m.samples.length) : "—"} />
          <Stat label="回合事件" value={m ? String(m.epochs.length) : "—"} />
        </div>
      </div>

      {job && (
        <div>
          <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80 mb-2">
            任务时间
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            <Stat
              label="创建于"
              value={
                job.created_at ? new Date(job.created_at).toLocaleString() : "—"
              }
            />
            <Stat
              label="启动于"
              value={
                job.started_at ? new Date(job.started_at).toLocaleString() : "—"
              }
            />
            <Stat
              label="完成于"
              value={
                job.finished_at
                  ? new Date(job.finished_at).toLocaleString()
                  : "—"
              }
            />
          </div>
        </div>
      )}
    </div>
  )
}
