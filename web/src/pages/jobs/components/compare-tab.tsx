import { useMemo } from "react"
import { useQueries } from "@tanstack/react-query"
import { api } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { LossChart, type LossSeries } from "./loss-chart"
import { COMPARE_COLORS } from "../utils"

export function CompareTab({ compareIds }: { compareIds: string[] }) {
  const results = useQueries({
    queries: compareIds.map((id) => ({
      queryKey: ["job-metrics", id],
      queryFn: () => api.getJobMetrics(id),
    })),
  })

  const series: LossSeries[] = useMemo(() => {
    return compareIds.map((id, idx) => {
      const data = results[idx]?.data
      const points = (data?.loss ?? [])
        .filter(
          (p): p is { step: number; loss: number; epoch?: number | null; ts: number } =>
            typeof p.loss === "number" && Number.isFinite(p.loss),
        )
        .map((p) => ({ step: p.step, loss: p.loss }))
      return {
        id,
        label: id.slice(-8),
        color: COMPARE_COLORS[idx % COMPARE_COLORS.length],
        points,
      }
    })
  }, [compareIds, results])

  const loading = results.some((r) => r.isLoading)
  const errored = results.some((r) => r.isError)

  if (compareIds.length < 2) {
    return (
      <Card className="rounded-[6px] border-border/60 shadow-[var(--panel-shadow)]">
        <CardContent className="p-6 text-sm text-muted-foreground text-center">
          请在左侧列表勾选至少 2 个任务以进行对比。
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="rounded-[6px] border-border/60 shadow-[var(--panel-shadow)]">
      <CardHeader className="py-3 px-4 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
        <CardTitle className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
          损失对比 · {compareIds.length} 个任务
        </CardTitle>
        <span className="text-[10px] text-muted-foreground/70">
          {loading ? "加载中…" : errored ? "部分加载失败" : "X 轴对齐到训练步"}
        </span>
      </CardHeader>
      <CardContent className="p-4 space-y-3">
        <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-[11px]">
          {series.map((s) => (
            <span key={s.id} className="inline-flex items-center gap-1.5 font-mono">
              <span
                className="inline-block size-2.5 rounded-[2px]"
                style={{ background: s.color }}
              />
              <span>{s.label}</span>
              <span className="text-muted-foreground/70">
                ({s.points.length} 点)
              </span>
            </span>
          ))}
        </div>
        <LossChart series={series} emptyHint="所选任务暂无可对比的损失数据。" />
      </CardContent>
    </Card>
  )
}
