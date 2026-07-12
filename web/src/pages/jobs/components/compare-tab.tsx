import { useMemo } from "react"
import { useQueries } from "@tanstack/react-query"
import { api } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { LossChart, type LossSeries } from "./loss-chart"
import { COMPARE_COLORS } from "../utils"
import { deriveAnalysisBackendInfo } from "../../analysis/components/analysis-backend"

export function CompareTab({ compareIds }: { compareIds: string[] }) {
  const results = useQueries({
    queries: compareIds.map((id) => ({
      queryKey: ["job-metrics", id],
      queryFn: () => api.getJobMetrics(id),
      // Compare 视图打开后立刻并发拉一批 metrics,2s 内被其他面板
      // (overview / job-detail / analysis)再次打开时直接复用缓存。
      staleTime: 2_000,
    })),
  })
  const detailResults = useQueries({
    queries: compareIds.map((id) => ({
      queryKey: ["job", id],
      queryFn: () => api.getJob(id),
      staleTime: 60_000,
    })),
  })
  const backendTypes = detailResults.map(
    (result) => deriveAnalysisBackendInfo(result.data).type,
  )
  const knownBackends = new Set(backendTypes.filter(Boolean))
  const crossBackend = knownBackends.size > 1

  const series: LossSeries[] = useMemo(() => {
    return compareIds.flatMap((id, idx) => {
      const data = results[idx]?.data
      const raw = (data?.loss ?? [])
        .filter(
          (p): p is { step: number; loss: number; epoch?: number | null; ts: number } =>
            typeof p.loss === "number" && Number.isFinite(p.loss),
        )
      const first = raw[0]
      const last = raw[raw.length - 1]
      const points = crossBackend
        ? raw.map((point) => ({
            step:
              first && last && last.step > first.step
                ? ((point.step - first.step) / (last.step - first.step)) * 100
                : point.step,
            loss:
              first && first.loss !== 0
                ? point.loss / first.loss
                : point.loss,
          }))
        : raw.map((point) => ({ step: point.step, loss: point.loss }))
      const backend = backendTypes[idx]
      const out: LossSeries[] = [
        {
          id: `${id}-train`,
          label: `${id.slice(-8)} · ${backend ?? "未知后端"}`,
          color: COMPARE_COLORS[idx % COMPARE_COLORS.length],
          points,
        },
      ]
      if (!crossBackend) {
        const epochToStep = new Map<number, number>()
        for (const point of raw) {
          if (typeof point.epoch === "number") {
            epochToStep.set(point.epoch, point.step)
          }
        }
        const valPoints = (data?.val_loss ?? [])
          .filter((point) => Number.isFinite(point.val_loss))
          .map((point) => ({
            step:
              typeof point.step === "number"
                ? point.step
                : (epochToStep.get(point.epoch) ?? point.epoch),
            loss: point.val_loss,
          }))
        if (valPoints.length > 0) {
          out.push({
            id: `${id}-val`,
            label: `${id.slice(-8)} · 验证`,
            color: COMPARE_COLORS[idx % COMPARE_COLORS.length],
            dashed: true,
            points: valPoints,
          })
        }
      }
      return out
    })
  }, [backendTypes, compareIds, crossBackend, results])

  const loading =
    results.some((result) => result.isLoading) ||
    detailResults.some((result) => result.isLoading)
  const errored =
    results.some((result) => result.isError) ||
    detailResults.some((result) => result.isError)

  if (compareIds.length < 2) {
    return (
      <Card>
        <CardContent className="p-6 text-sm text-muted-foreground text-center">
          请在左侧列表勾选至少 2 个任务以进行对比。
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader className="py-3 px-4 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
        <CardTitle className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
          {crossBackend ? "跨后端相对损失形态" : "同后端损失对比"} ·{" "}
          {compareIds.length} 个任务
        </CardTitle>
        <span className="text-[10px] text-muted-foreground/70">
          {loading
            ? "加载中…"
            : errored
              ? "部分加载失败"
              : crossBackend
                ? "进度归一化，首个 loss = 1.0"
                : "X 轴对齐到训练步，可比较验证曲线"}
        </span>
      </CardHeader>
      <CardContent className="p-4 space-y-3">
        {crossBackend && (
          <div className="border-l-2 border-amber-500/70 bg-amber-500/8 px-3 py-2 text-[11px] leading-5 text-muted-foreground">
            不同后端的损失目标与绝对量级不同。图中仅比较归一化后的下降形态；验证损失和 epoch 均值不做跨后端叠加。
          </div>
        )}
        <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-[11px]">
          {compareIds.map((id, index) => {
            const data = results[index]?.data
            return (
              <span
                key={id}
                className="inline-flex items-center gap-1.5 font-mono"
              >
                <span
                  className="inline-block size-2.5 rounded-[2px]"
                  style={{
                    background:
                      COMPARE_COLORS[index % COMPARE_COLORS.length],
                  }}
                />
                <span>
                  {id.slice(-8)} · {backendTypes[index] ?? "未知后端"}
                </span>
                <span className="text-muted-foreground/70">
                  {data?.loss?.length ?? 0} 点 ·{" "}
                  {formatDuration(data?.duration_s)} ·{" "}
                  {data?.checkpoints?.length ?? 0} 个检查点
                </span>
              </span>
            )
          })}
        </div>
        <LossChart
          series={series}
          xLabel={crossBackend ? "训练进度 (%)" : "step"}
          emptyHint="所选任务暂无可对比的损失数据。"
        />
      </CardContent>
    </Card>
  )
}

function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "用时未知"
  if (seconds < 60) return `${Math.round(seconds)} 秒`
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  return hours > 0 ? `${hours} 小时 ${minutes} 分` : `${minutes} 分钟`
}
