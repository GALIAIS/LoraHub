/**
 * Two-card secondary chart row for the analysis workbench: training
 * throughput on the left, GPU resources on the right. Each card uses a
 * single multi-line chart instead of stacking three rows of
 * `<SeriesLineChart>` so vertical density stays under control while
 * losing nothing important.
 */
import { useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { JobMetricsResponse } from "@/lib/api"
import { MultiLineChart, type MultiLineSeries } from "./multi-line-chart"

export function SecondaryCharts({
  metrics,
  loading,
  jobId,
}: {
  metrics: JobMetricsResponse | null
  loading: boolean
  jobId?: string | null
}) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
      <ThroughputCard metrics={metrics} loading={loading} jobId={jobId} />
      <ResourcesCard metrics={metrics} loading={loading} jobId={jobId} />
    </div>
  )
}

function ThroughputCard({
  metrics,
  loading,
  jobId,
}: {
  metrics: JobMetricsResponse | null
  loading: boolean
  jobId?: string | null
}) {
  const series: MultiLineSeries[] = useMemo(() => {
    const points = metrics?.loss ?? []
    const out: MultiLineSeries[] = []
    const lr = points
      .filter((p) => typeof p.lr === "number" && Number.isFinite(p.lr))
      .map((p) => ({ x: p.step, y: p.lr as number }))
    const it = points
      .filter(
        (p) =>
          typeof p.iter_time_s === "number" && Number.isFinite(p.iter_time_s),
      )
      .map((p) => ({ x: p.step, y: p.iter_time_s as number }))
    const sps = points
      .filter(
        (p) =>
          typeof p.samples_per_sec === "number" &&
          Number.isFinite(p.samples_per_sec),
      )
      .map((p) => ({ x: p.step, y: p.samples_per_sec as number }))
    if (lr.length > 0)
      out.push({
        id: "lr",
        label: "学习率",
        color: "var(--chart-1)",
        axis: "left",
        points: lr,
      })
    if (it.length > 0)
      out.push({
        id: "iter",
        label: "迭代时长",
        color: "var(--chart-3)",
        axis: "right",
        unit: "s",
        points: it,
      })
    if (sps.length > 0)
      out.push({
        id: "sps",
        label: "样本/秒",
        color: "var(--chart-4)",
        axis: "right",
        points: sps,
      })
    return out
  }, [metrics])

  return (
    <Card className="rounded-[6px] border-border/60">
      <CardHeader className="py-2 px-3.5 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
        <CardTitle className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          学习率与吞吐
        </CardTitle>
        <span className="text-[10px] text-muted-foreground/70">
          {loading
            ? "加载中…"
            : series.length === 0
              ? "未采集到数据"
              : "横轴：训练步"}
        </span>
      </CardHeader>
      <CardContent className="p-3">
        {series.length === 0 ? (
          <div className="text-xs text-muted-foreground leading-relaxed py-4">
            训练后端尚未输出学习率 / 迭代时长 / 吞吐数据。
            diffusion-pipe 在产出第一行 <code className="font-mono">steps: N loss: …</code> 后会自动出现这些曲线。
          </div>
        ) : (
          <MultiLineChart
            series={series}
            xLabel="step"
            persistKey={jobId ? `${jobId}.throughput` : null}
            title="学习率与吞吐"
          />
        )}
      </CardContent>
    </Card>
  )
}

function ResourcesCard({
  metrics,
  loading,
  jobId,
}: {
  metrics: JobMetricsResponse | null
  loading: boolean
  jobId?: string | null
}) {
  const samples = metrics?.gpu_samples ?? []

  const series: MultiLineSeries[] = useMemo(() => {
    if (samples.length === 0) return []
    const t0 = samples[0].ts
    const util: { x: number; y: number | null }[] = []
    const vramPct: { x: number; y: number | null }[] = []
    const temp: { x: number; y: number | null }[] = []
    for (const s of samples) {
      const tMin = (s.ts - t0) / 60
      util.push({ x: tMin, y: s.util_percent })
      vramPct.push({
        x: tMin,
        y:
          s.vram_used_mib != null && s.vram_total_mib && s.vram_total_mib > 0
            ? (s.vram_used_mib / s.vram_total_mib) * 100
            : null,
      })
      temp.push({ x: tMin, y: s.temperature_c })
    }
    const out: MultiLineSeries[] = []
    if (util.some((p) => p.y != null))
      out.push({
        id: "util",
        label: "GPU 利用率",
        unit: "%",
        color: "var(--chart-1)",
        axis: "left",
        points: util,
      })
    if (vramPct.some((p) => p.y != null))
      out.push({
        id: "vram",
        label: "显存占用",
        unit: "%",
        color: "var(--chart-2)",
        axis: "left",
        points: vramPct,
      })
    if (temp.some((p) => p.y != null))
      out.push({
        id: "temp",
        label: "温度",
        unit: "°C",
        color: "var(--chart-3)",
        axis: "right",
        points: temp,
      })
    return out
  }, [samples])

  const durationMin =
    samples.length > 0
      ? (samples[samples.length - 1].ts - samples[0].ts) / 60
      : 0

  return (
    <Card className="rounded-[6px] border-border/60">
      <CardHeader className="py-2 px-3.5 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
        <CardTitle className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          GPU 资源
        </CardTitle>
        <span className="text-[10px] text-muted-foreground/70">
          {loading
            ? "加载中…"
            : samples.length === 0
              ? "未采集到数据"
              : `${samples.length} 采样 · ${durationMin.toFixed(1)} min`}
        </span>
      </CardHeader>
      <CardContent className="p-3">
        {series.length === 0 ? (
          <div className="text-xs text-muted-foreground leading-relaxed py-4">
            训练运行期间每 5 秒采集一次 GPU 利用率 / 显存占用 / 温度。当前没有采样数据。
          </div>
        ) : (
          <MultiLineChart
            series={series}
            xLabel="分钟"
            persistKey={jobId ? `${jobId}.resources` : null}
            title="GPU 资源"
          />
        )}
      </CardContent>
    </Card>
  )
}
