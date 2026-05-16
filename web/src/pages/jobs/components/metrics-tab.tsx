import { useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { LossChart, type LossSeries } from "./loss-chart"
import { TERMINAL_STATES } from "../utils"

export function MetricsTab({
  jobId,
  jobState,
}: {
  jobId: string
  jobState: string | undefined
}) {
  const isTerminal = jobState ? TERMINAL_STATES.has(jobState) : false
  const metrics = useQuery({
    queryKey: ["job-metrics", jobId],
    queryFn: () => api.getJobMetrics(jobId),
    refetchInterval: isTerminal ? false : 4000,
  })

  // Train loss series uses the existing color slot; the validation overlay
  // gets a contrasting amber so the gap between the two curves is the
  // visual cue users actually look at.
  const series: LossSeries[] = useMemo(() => {
    const trainPoints = (metrics.data?.loss ?? [])
      .filter(
        (p): p is { step: number; loss: number; epoch?: number | null; ts: number } =>
          typeof p.loss === "number" && Number.isFinite(p.loss),
      )
      .map((p) => ({ step: p.step, loss: p.loss }))

    const out: LossSeries[] = []
    if (trainPoints.length > 0) {
      out.push({
        id: `${jobId}-train`,
        label: "训练 loss",
        color: "var(--chart-1)",
        points: trainPoints,
      })
    }

    // Validation events are reported per epoch — translate to step using the
    // mean train-loss step in that epoch so both curves share an x-axis. If
    // the train loss series is missing or epoch info is absent, fall back
    // to (epoch * total_steps_per_epoch_estimate) using the last-seen step.
    const valPoints = (metrics.data?.val_loss ?? []).filter(
      (p): p is { epoch: number; val_loss: number; step?: number | null; ts: number } =>
        typeof p.val_loss === "number" && Number.isFinite(p.val_loss),
    )
    if (valPoints.length > 0) {
      // Map epoch -> last train step seen during that epoch.
      const epochToStep = new Map<number, number>()
      for (const p of metrics.data?.loss ?? []) {
        if (typeof p.epoch === "number" && typeof p.step === "number") {
          epochToStep.set(p.epoch, p.step)
        }
      }
      const lastTrainStep = trainPoints.length
        ? trainPoints[trainPoints.length - 1].step
        : 0

      const mapped = valPoints.map((p) => {
        const x =
          typeof p.step === "number"
            ? p.step
            : (epochToStep.get(p.epoch) ?? (lastTrainStep || p.epoch))
        return { step: x, loss: p.val_loss }
      })
      out.push({
        id: `${jobId}-val`,
        label: "验证 loss",
        color: "var(--chart-2)",
        points: mapped,
      })
    }

    return out
  }, [metrics.data, jobId])

  const totalPoints = metrics.data?.loss?.length ?? 0
  const overfitSignal = metrics.data?.overfit_signal

  return (
    <div className="space-y-4">
      <Card className="rounded-[6px] border-border/60 shadow-[var(--panel-shadow)]">
        <CardHeader className="py-3 px-4 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
          <CardTitle className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
            损失曲线
          </CardTitle>
          <span className="text-[10px] text-muted-foreground/70">
            {metrics.isLoading
              ? "加载中…"
              : metrics.isError
                ? "加载失败"
                : `共 ${totalPoints} 个采样点${totalPoints > 1000 ? "（已下采样到 1000）" : ""}`}
          </span>
        </CardHeader>
        <CardContent className="p-4">
          <LossChart series={series} overfitSignal={overfitSignal ?? null} />
        </CardContent>
      </Card>

      <Card className="rounded-[6px] border-border/60 shadow-[var(--panel-shadow)]">
        <CardHeader className="py-3 px-4 border-b border-border/60 bg-muted/40">
          <CardTitle className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
            资源趋势
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4 text-xs text-muted-foreground leading-relaxed">
          实时硬件指标已在数据面板呈现；任务级别的 GPU 历史将在 v0.3 加入。
        </CardContent>
      </Card>
    </div>
  )
}
