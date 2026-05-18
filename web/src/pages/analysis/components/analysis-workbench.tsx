/**
 * AnalysisWorkbench — single-job analytical view.
 *
 * Layout (top to bottom):
 *   1. KPI strip   — six headline metrics, one row, no card chrome.
 *   2. Loss panel  — main chart, full width. The visual focus.
 *   3. Secondary   — two-column grid: "LR + iter time + samples/sec"
 *                    (left, dual axis) and "GPU util + VRAM% + temp"
 *                    (right, dual axis).
 *   4. Tabs        — series stats / metrics table / samples / AI
 *                    analysis. Only one is rendered at a time so the
 *                    page never feels like an information avalanche.
 */
import { useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { api, type JobSummary } from "@/lib/api"
import {
  LossChart,
  type ChartMarker,
  type LossSeries,
} from "../../jobs/components/loss-chart"
import { TERMINAL_STATES } from "../../jobs/utils"
import { AnalysisKpiStrip } from "./analysis-kpi-strip"
import { SecondaryCharts } from "./secondary-charts"
import { AICard } from "../panels/ai-card"
import { MetricsTable } from "../panels/metrics-table"
import { SamplesGallery } from "../panels/samples-gallery"
import { SeriesStatsCard } from "../panels/series-stats"

type BottomTabKey = "stats" | "table" | "samples" | "ai"

const EMA_ALPHA = 0.1

export function AnalysisWorkbench({
  job,
  fallbackTotalSteps,
}: {
  job: JobSummary
  fallbackTotalSteps: number | null
}) {
  const isTerminal = TERMINAL_STATES.has(job.state)
  const metrics = useQuery({
    queryKey: ["job-metrics", job.id],
    queryFn: () => api.getJobMetrics(job.id),
    refetchInterval: isTerminal ? false : 4000,
  })
  const files = useQuery({
    queryKey: ["job-files", job.id],
    queryFn: () => api.getJobFiles(job.id),
    refetchInterval: isTerminal ? false : 8000,
  })
  const aiCache = useQuery({
    queryKey: ["job-analysis", job.id],
    queryFn: () => api.getJobAnalysis(job.id),
    refetchInterval: false,
  })

  const lossSeries: LossSeries[] = useMemo(() => {
    const points = (metrics.data?.loss ?? []).filter(
      (p): p is { step: number; loss: number; epoch?: number | null; ts: number } =>
        typeof p.loss === "number" && Number.isFinite(p.loss),
    )
    const trainPoints = points.map((p) => ({ step: p.step, loss: p.loss }))
    const out: LossSeries[] = []
    if (trainPoints.length > 0) {
      out.push({
        id: `${job.id}-train`,
        label: "训练 loss",
        color: "var(--chart-1)",
        points: trainPoints,
      })
      if (trainPoints.length >= 6) {
        let acc = trainPoints[0].loss
        const ema = trainPoints.map((p) => {
          acc = EMA_ALPHA * p.loss + (1 - EMA_ALPHA) * acc
          return { step: p.step, loss: acc }
        })
        out.push({
          id: `${job.id}-train-ema`,
          label: `EMA α=${EMA_ALPHA}`,
          color: "var(--chart-1)",
          dashed: true,
          points: ema,
        })
      }
    }
    const valPoints = (metrics.data?.val_loss ?? []).filter(
      (p): p is { epoch: number; val_loss: number; step?: number | null; ts: number } =>
        typeof p.val_loss === "number" && Number.isFinite(p.val_loss),
    )
    if (valPoints.length > 0) {
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
        id: `${job.id}-val`,
        label: "验证 loss",
        color: "var(--chart-2)",
        points: mapped,
      })
    }
    return out
  }, [metrics.data, job.id])

  const checkpointMarkers: ChartMarker[] = useMemo(() => {
    return (metrics.data?.checkpoints ?? [])
      .filter((c): c is { path: string; step: number; ts: number } =>
        typeof c.step === "number" && Number.isFinite(c.step),
      )
      .map((c) => ({
        step: c.step,
        label: c.path,
        color: "var(--chart-3)",
      }))
  }, [metrics.data])

  const totalPoints = metrics.data?.loss?.length ?? 0
  const overfit = metrics.data?.overfit_signal

  // Bottom tabs default + counts (used in tab labels).
  const [bottomTab, setBottomTab] = useState<BottomTabKey>("stats")
  const samplesCount = files.data?.samples?.length ?? 0
  const tableRowCount = totalPoints
  const seriesCount = lossSeries.length
  const aiState: "ready" | "missing" = aiCache.data?.analysis ? "ready" : "missing"

  return (
    <div className="flex flex-col min-h-0">
      <AnalysisKpiStrip job={job} fallbackTotalSteps={fallbackTotalSteps} />

      <div className="px-7 py-4 space-y-3">
        {/* Loss panel — visual focus of the page. */}
        <Card className="rounded-[6px] border-border/60">
          <CardHeader className="py-2 px-3.5 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
            <CardTitle className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
              损失曲线
            </CardTitle>
            <span className="text-[10px] text-muted-foreground/70">
              {metrics.isLoading
                ? "加载中…"
                : metrics.isError
                  ? "加载失败"
                  : `${totalPoints} 个采样点${
                      totalPoints > 1000 ? "（已下采样到 1000）" : ""
                    }${
                      checkpointMarkers.length > 0
                        ? ` · ${checkpointMarkers.length} 个检查点`
                        : ""
                    }`}
            </span>
          </CardHeader>
          <CardContent className="p-3.5">
            <LossChart
              series={lossSeries}
              overfitSignal={overfit ?? null}
              markers={checkpointMarkers}
              persistKey={job.id}
            />
          </CardContent>
        </Card>

        {/* Secondary charts — two columns. */}
        <SecondaryCharts
          metrics={metrics.data ?? null}
          loading={metrics.isLoading}
          jobId={job.id}
        />

        {/* Bottom tabs — stats / table / samples / AI. */}
        <Tabs
          value={bottomTab}
          onValueChange={(v) => setBottomTab(v as BottomTabKey)}
        >
          <div className="border-y border-border/60 bg-muted/40 px-3.5 py-1.5 rounded-t-[6px]">
            <TabsList variant="line" className="gap-3">
              <TabsTrigger value="stats" className="text-[11.5px]">
                序列统计{" "}
                <span className="ml-1 tabular-nums text-muted-foreground/80">
                  {seriesCount}
                </span>
              </TabsTrigger>
              <TabsTrigger value="table" className="text-[11.5px]">
                指标表格{" "}
                <span className="ml-1 tabular-nums text-muted-foreground/80">
                  {tableRowCount}
                </span>
              </TabsTrigger>
              <TabsTrigger value="samples" className="text-[11.5px]">
                样本{" "}
                <span className="ml-1 tabular-nums text-muted-foreground/80">
                  {samplesCount}
                </span>
              </TabsTrigger>
              <TabsTrigger value="ai" className="text-[11.5px]">
                AI 分析{" "}
                <span
                  className={
                    aiState === "ready"
                      ? "ml-1 text-emerald-600 dark:text-emerald-400"
                      : "ml-1 text-muted-foreground/80"
                  }
                >
                  {aiState === "ready" ? "✓" : "未生成"}
                </span>
              </TabsTrigger>
            </TabsList>
          </div>
          <TabsContent value="stats" className="m-0">
            <SeriesStatsCard
              series={lossSeries}
              metrics={metrics.data ?? null}
            />
          </TabsContent>
          <TabsContent value="table" className="m-0">
            <MetricsTable
              loss={metrics.data?.loss ?? []}
              valLoss={metrics.data?.val_loss ?? []}
              loading={metrics.isLoading}
            />
          </TabsContent>
          <TabsContent value="samples" className="m-0">
            <SamplesGallery
              jobId={job.id}
              samples={files.data?.samples ?? []}
              loading={files.isLoading}
            />
          </TabsContent>
          <TabsContent value="ai" className="m-0">
            <AICard
              jobId={job.id}
              cached={aiCache.data?.analysis ?? null}
              loading={aiCache.isLoading}
              canRun={isTerminal || (metrics.data?.loss?.length ?? 0) > 0}
            />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
