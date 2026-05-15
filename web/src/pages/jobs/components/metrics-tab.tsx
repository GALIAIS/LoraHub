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

  const series: LossSeries[] = useMemo(() => {
    const points = (metrics.data?.loss ?? [])
      .filter(
        (p): p is { step: number; loss: number; epoch?: number | null; ts: number } =>
          typeof p.loss === "number" && Number.isFinite(p.loss),
      )
      .map((p) => ({ step: p.step, loss: p.loss }))
    if (points.length === 0) return []
    return [
      {
        id: jobId,
        label: jobId.slice(-8),
        color: "var(--primary)",
        points,
      },
    ]
  }, [metrics.data, jobId])

  const totalPoints = metrics.data?.loss?.length ?? 0

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
          <LossChart series={series} />
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
