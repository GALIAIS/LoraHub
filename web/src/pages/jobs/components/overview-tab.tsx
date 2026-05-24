import { useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import {
  api,
  useSystemStream,
  type JobSummary,
  type TrainingEvent,
} from "@/lib/api"
import { Stat } from "./stat"
import {
  EtaTile,
  GpuLiveTile,
  LossTrendTile,
  ThroughputSwitcher,
  ThroughputTile,
} from "./realtime-tile"
import { fmtDuration, fmtUnixSeconds, stateLabel, TERMINAL_STATES, ACTIVE_STATES } from "../utils"
import { TrainingFeatureBadges } from "./training-feature-badges"

const THROUGHPUT_WINDOW = 60
const LOSS_WINDOW = 100

interface StepSample {
  step: number
  ts: number
  totalSteps: number | null
  loss: number | null
}

function extractStepSamples(events: TrainingEvent[]): StepSample[] {
  const out: StepSample[] = []
  for (const e of events) {
    if (e.type !== "step") continue
    const p = e.payload
    const step = typeof p.step === "number" ? p.step : null
    if (step === null) continue
    const total = typeof p.total_steps === "number" ? p.total_steps : null
    const loss = typeof p.loss === "number" ? p.loss : null
    out.push({ step, ts: e.timestamp, totalSteps: total, loss })
  }
  return out
}

function computeItPerSec(samples: StepSample[]): number | null {
  if (samples.length < 2) return null
  const first = samples[0]
  const last = samples[samples.length - 1]
  const dSteps = last.step - first.step
  const dT = last.ts - first.ts
  if (dT <= 0 || dSteps <= 0) return null
  return dSteps / dT
}

export function OverviewTab({
  jobId,
  job,
  events,
  fallbackTotalSteps = null,
}: {
  jobId: string
  job: (JobSummary & { config_snapshot?: Record<string, unknown> }) | undefined
  events: TrainingEvent[]
  fallbackTotalSteps?: number | null
}) {
  const isTerminal = job ? TERMINAL_STATES.has(job.state) : false
  // "Active" = worker slot is held (preparing or running). The realtime
  // tiles need to show during anima_lora's preprocess phase too — that
  // can take 1-2 minutes and currently leaves the user staring at "排队
  // 中" while the GPU is actually busy resizing / caching latents.
  const isActive = job ? ACTIVE_STATES.has(job.state) : false

  // Telemetry stream stays open whenever the user is on this tab so the cards
  // refresh without waiting for a poll. Only opening it for live jobs would
  // cost us a fresh handshake every time the job restarts.
  const system = useSystemStream(true)

  const lastStep = useMemo(
    () => [...events].reverse().find((e) => e.type === "step"),
    [events],
  )

  // Metrics (first/last step ts, duration) are only meaningful once at least
  // one step has been recorded. Refetch while live so the panel stays fresh.
  const metrics = useQuery({
    queryKey: ["job-metrics", jobId],
    queryFn: () => api.getJobMetrics(jobId),
    refetchInterval: isTerminal ? false : 4000,
    staleTime: 2_000,
  })

  const m = metrics.data

  const stepSamples = useMemo(() => extractStepSamples(events), [events])
  const recentStepSamples = useMemo(
    () =>
      stepSamples.length <= THROUGHPUT_WINDOW
        ? stepSamples
        : stepSamples.slice(stepSamples.length - THROUGHPUT_WINDOW),
    [stepSamples],
  )

  // Compute per-window it/s by sliding across the recent samples; this gives
  // the sparkline some shape rather than a flat line at the running average.
  const throughputHistory = useMemo(() => {
    const out: number[] = []
    if (recentStepSamples.length < 4) return out
    const span = 4
    for (let i = span; i < recentStepSamples.length; i += 1) {
      const slice = recentStepSamples.slice(i - span, i + 1)
      const v = computeItPerSec(slice)
      if (v !== null) out.push(v)
    }
    return out
  }, [recentStepSamples])

  const itPerSecRecent = computeItPerSec(recentStepSamples)
  const itPerSecAvg = computeItPerSec(stepSamples)

  // ETA only makes sense once we know a total, otherwise we'd be dividing
  // by zero. Prefer the most recent payload-side total (event truth) and
  // fall back to a config-derived estimate (provided by the parent) when
  // the trainer hasn't reported one yet — dp's parser doesn't emit
  // total_steps, so without the fallback this number stays "?".
  const totalSteps = useMemo(() => {
    for (let i = stepSamples.length - 1; i >= 0; i -= 1) {
      const v = stepSamples[i].totalSteps
      if (typeof v === "number" && v > 0) return v
    }
    return fallbackTotalSteps
  }, [stepSamples, fallbackTotalSteps])

  const currentStep =
    stepSamples.length > 0 ? stepSamples[stepSamples.length - 1].step : null

  const etaSeconds = useMemo(() => {
    if (!isActive) return null
    if (itPerSecRecent === null || itPerSecRecent <= 0) return null
    if (currentStep === null || totalSteps === null) return null
    const remaining = totalSteps - currentStep
    if (remaining <= 0) return 0
    return remaining / itPerSecRecent
  }, [isActive, itPerSecRecent, currentStep, totalSteps])

  const lossHistory = useMemo(() => {
    const all = stepSamples
      .map((s) => s.loss)
      .filter((v): v is number => typeof v === "number" && Number.isFinite(v))
    return all.length <= LOSS_WINDOW
      ? all
      : all.slice(all.length - LOSS_WINDOW)
  }, [stepSamples])

  const latestLoss = lossHistory.length
    ? lossHistory[lossHistory.length - 1]
    : null

  const liveGpu = system.snapshot?.gpus?.[0] ?? null

  // Final-stats tile values are derived once the run is done; they intentionally
  // ignore the live stream so a re-render doesn't perturb completed numbers.
  const finalDuration = m?.duration_s ?? null
  const finalThroughput = useMemo(() => {
    if (
      m?.first_step_ts &&
      m?.last_step_ts &&
      m.last_step_ts > m.first_step_ts &&
      m.loss.length > 1
    ) {
      const dT = m.last_step_ts - m.first_step_ts
      const dSteps = m.loss[m.loss.length - 1].step - m.loss[0].step
      if (dT > 0 && dSteps > 0) return dSteps / dT
    }
    return null
  }, [m])
  const finalLoss = useMemo(() => {
    if (!m || m.loss.length === 0) return null
    const last = m.loss[m.loss.length - 1].loss
    return typeof last === "number" ? last : null
  }, [m])

  return (
    <div className="space-y-5">
      <TrainingFeatureBadges configSnapshot={job?.config_snapshot} />
      <div className="grid grid-cols-3 gap-3">
        <Stat label="状态" value={job?.state ? stateLabel(job.state) : "—"} />
        <Stat
          label="进度"
          value={
            lastStep
              ? `${lastStep.payload.step ?? "?"} / ${
                  (typeof lastStep.payload.total_steps === "number" &&
                  lastStep.payload.total_steps > 0
                    ? lastStep.payload.total_steps
                    : totalSteps) ?? "?"
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

      {isActive ? (
        <div>
          <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80 mb-2">
            实时
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
            <GpuLiveTile gpu={liveGpu} active={isActive} />
            <ThroughputTile
              itPerSecRecent={itPerSecRecent}
              itPerSecAvg={itPerSecAvg}
              history={throughputHistory}
            />
            <EtaTile
              etaSeconds={etaSeconds}
              step={currentStep}
              totalSteps={totalSteps}
            />
            <LossTrendTile history={lossHistory} latest={latestLoss} />
          </div>
        </div>
      ) : isTerminal ? (
        <div>
          <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80 mb-2">
            最终统计
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            <Stat label="总用时" value={fmtDuration(finalDuration)} />
            <Stat
              label="平均吞吐"
              value={
                finalThroughput !== null && finalThroughput > 0 ? (
                  <ThroughputSwitcher itPerSec={finalThroughput} />
                ) : (
                  "—"
                )
              }
            />
            <Stat
              label="最终损失"
              value={
                typeof finalLoss === "number" ? finalLoss.toFixed(4) : "—"
              }
            />
          </div>
        </div>
      ) : null}

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
