import { useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import { api } from "@/lib/api"
import type { JobMetricsResponse } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { LossChart, type ChartMarker, type LossSeries } from "./loss-chart"
import { SeriesLineChart } from "./series-line-chart"
import { TERMINAL_STATES } from "../utils"

// Coefficient of the EMA smoothing applied to train loss. 0.1 was picked
// after eyeballing dp logs — large enough to flatten step-to-step noise
// without lagging visible regressions by more than ~1 epoch.
const EMA_ALPHA = 0.1

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

  // ---- Loss series: train + EMA(train) + val (if any). ---------------------
  const lossSeries: LossSeries[] = useMemo(() => {
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
      // EMA smoothing — only when there are enough points for the curve
      // to mean something. Five raw points isn't enough to tell jitter
      // from trend; below that we skip the overlay.
      if (trainPoints.length >= 6) {
        let acc = trainPoints[0].loss
        const ema = trainPoints.map((p) => {
          acc = EMA_ALPHA * p.loss + (1 - EMA_ALPHA) * acc
          return { step: p.step, loss: acc }
        })
        out.push({
          id: `${jobId}-train-ema`,
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
        id: `${jobId}-val`,
        label: "验证 loss",
        color: "var(--chart-2)",
        points: mapped,
      })
    }
    return out
  }, [metrics.data, jobId])

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
                : `共 ${totalPoints} 个采样点${totalPoints > 1000 ? "（已下采样到 1000）" : ""}${
                    checkpointMarkers.length > 0
                      ? ` · ${checkpointMarkers.length} 个检查点`
                      : ""
                  }`}
          </span>
        </CardHeader>
        <CardContent className="p-4">
          <LossChart
            series={lossSeries}
            overfitSignal={overfitSignal ?? null}
            markers={checkpointMarkers}
          />
        </CardContent>
      </Card>

      <LrThroughputCard metrics={metrics.data ?? null} loading={metrics.isLoading} />

      <ResourceTrendCard metrics={metrics.data ?? null} loading={metrics.isLoading} />

      <SeriesStatsCard
        series={lossSeries}
        metrics={metrics.data ?? null}
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Learning rate + throughput
// ---------------------------------------------------------------------------

function LrThroughputCard({
  metrics,
  loading,
}: {
  metrics: JobMetricsResponse | null
  loading: boolean
}) {
  const points = metrics?.loss ?? []
  const lrPoints = points
    .filter((p) => typeof p.lr === "number" && Number.isFinite(p.lr))
    .map((p) => ({ x: p.step, y: p.lr as number }))
  const itPoints = points
    .filter(
      (p) =>
        typeof p.iter_time_s === "number" && Number.isFinite(p.iter_time_s),
    )
    .map((p) => ({ x: p.step, y: p.iter_time_s as number }))
  const sampPoints = points
    .filter(
      (p) =>
        typeof p.samples_per_sec === "number" &&
        Number.isFinite(p.samples_per_sec),
    )
    .map((p) => ({ x: p.step, y: p.samples_per_sec as number }))

  const hasAny = lrPoints.length + itPoints.length + sampPoints.length > 0

  return (
    <Card className="rounded-[6px] border-border/60 shadow-[var(--panel-shadow)]">
      <CardHeader className="py-3 px-4 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
        <CardTitle className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
          学习率与吞吐
        </CardTitle>
        <span className="text-[10px] text-muted-foreground/70">
          {loading ? "加载中…" : hasAny ? "横轴：训练步" : "未采集到数据"}
        </span>
      </CardHeader>
      <CardContent className="p-4 space-y-3">
        {!hasAny && (
          <div className="text-xs text-muted-foreground leading-relaxed">
            训练后端尚未输出学习率 / 迭代时间 / 吞吐数据。diffusion-pipe 在产出第一行
            <code className="font-mono mx-1">steps: N loss: …</code>
            后会自动出现这些曲线。
          </div>
        )}
        {hasAny && (
          <>
            <SeriesLineChart
              label="学习率"
              unit=""
              points={lrPoints}
              color="var(--chart-1)"
              hint="无学习率数据"
            />
            <SeriesLineChart
              label="迭代时长"
              unit="s"
              points={itPoints}
              color="var(--chart-3)"
              hint="无迭代时长数据"
            />
            <SeriesLineChart
              label="样本/秒"
              unit=""
              points={sampPoints}
              color="var(--chart-4)"
              hint="无吞吐数据"
            />
          </>
        )}
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Resource trend (GPU util / VRAM / temp)
// ---------------------------------------------------------------------------

function ResourceTrendCard({
  metrics,
  loading,
}: {
  metrics: JobMetricsResponse | null
  loading: boolean
}) {
  const samples = metrics?.gpu_samples ?? []
  const series = useMemo(() => {
    if (samples.length === 0) return null
    const t0 = samples[0].ts
    const points = samples.map((s) => ({
      tMin: (s.ts - t0) / 60,
      util: s.util_percent,
      vramPct:
        s.vram_used_mib != null && s.vram_total_mib && s.vram_total_mib > 0
          ? (s.vram_used_mib / s.vram_total_mib) * 100
          : null,
      vramMib: s.vram_used_mib,
      temp: s.temperature_c,
    }))
    return { points, durationMin: points[points.length - 1].tMin }
  }, [samples])

  return (
    <Card className="rounded-[6px] border-border/60 shadow-[var(--panel-shadow)]">
      <CardHeader className="py-3 px-4 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
        <CardTitle className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
          资源趋势
        </CardTitle>
        <span className="text-[10px] text-muted-foreground/70">
          {loading ? "加载中…" : `${samples.length} 个采样点`}
        </span>
      </CardHeader>
      <CardContent className="p-4">
        {!series && (
          <div className="text-xs text-muted-foreground leading-relaxed">
            训练运行期间每 5 秒采集一次 GPU 利用率 / 显存占用 / 温度。当前没有采样数据。
          </div>
        )}
        {series && (
          <div className="space-y-3">
            <SeriesLineChart
              label="GPU 利用率"
              unit="%"
              points={series.points.map((p) => ({ x: p.tMin, y: p.util }))}
              color="var(--chart-1)"
              yMax={100}
            />
            <SeriesLineChart
              label="显存占用"
              unit="%"
              points={series.points.map((p) => ({ x: p.tMin, y: p.vramPct }))}
              color="var(--chart-2)"
              yMax={100}
            />
            <SeriesLineChart
              label="显存(MiB)"
              unit=""
              points={series.points.map((p) => ({ x: p.tMin, y: p.vramMib }))}
              color="var(--chart-4)"
            />
            <SeriesLineChart
              label="温度"
              unit="°C"
              points={series.points.map((p) => ({ x: p.tMin, y: p.temp }))}
              color="var(--chart-3)"
            />
            <div className="text-[10px] text-muted-foreground/70 text-right">
              横轴：分钟（共 {series.durationMin.toFixed(1)} min）
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Series stats table
// ---------------------------------------------------------------------------

interface StatRow {
  id: string
  label: string
  color: string
  count: number
  min: number
  max: number
  avg: number
  std: number
  latest: number
}

function describeSeries(
  id: string,
  label: string,
  color: string,
  values: number[],
): StatRow | null {
  const filtered = values.filter((v) => Number.isFinite(v))
  if (filtered.length === 0) return null
  const min = Math.min(...filtered)
  const max = Math.max(...filtered)
  const avg = filtered.reduce((a, b) => a + b, 0) / filtered.length
  const variance =
    filtered.reduce((a, b) => a + (b - avg) ** 2, 0) / filtered.length
  const std = Math.sqrt(variance)
  return {
    id,
    label,
    color,
    count: filtered.length,
    min,
    max,
    avg,
    std,
    latest: filtered[filtered.length - 1],
  }
}

function fmtVal(v: number): string {
  if (!Number.isFinite(v)) return "—"
  if (Math.abs(v) >= 100) return v.toFixed(1)
  if (Math.abs(v) >= 1) return v.toFixed(3)
  if (Math.abs(v) >= 0.01) return v.toFixed(4)
  if (v === 0) return "0"
  return v.toExponential(2)
}

function SeriesStatsCard({
  series,
  metrics,
}: {
  series: LossSeries[]
  metrics: JobMetricsResponse | null
}) {
  const rows: StatRow[] = useMemo(() => {
    const out: StatRow[] = []
    for (const s of series) {
      const r = describeSeries(
        s.id,
        s.label,
        s.color,
        s.points.map((p) => p.loss),
      )
      if (r) out.push(r)
    }
    const lr = metrics?.loss
      ?.filter((p) => typeof p.lr === "number" && Number.isFinite(p.lr))
      .map((p) => p.lr as number) ?? []
    const r1 = describeSeries("lr", "学习率", "var(--chart-1)", lr)
    if (r1) out.push(r1)
    const it =
      metrics?.loss
        ?.filter(
          (p) =>
            typeof p.iter_time_s === "number" &&
            Number.isFinite(p.iter_time_s),
        )
        .map((p) => p.iter_time_s as number) ?? []
    const r2 = describeSeries("it", "迭代时长 s", "var(--chart-3)", it)
    if (r2) out.push(r2)
    const sps =
      metrics?.loss
        ?.filter(
          (p) =>
            typeof p.samples_per_sec === "number" &&
            Number.isFinite(p.samples_per_sec),
        )
        .map((p) => p.samples_per_sec as number) ?? []
    const r3 = describeSeries("sps", "样本/秒", "var(--chart-4)", sps)
    if (r3) out.push(r3)
    return out
  }, [series, metrics])

  return (
    <Card className="rounded-[6px] border-border/60 shadow-[var(--panel-shadow)]">
      <CardHeader className="py-3 px-4 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
        <CardTitle className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
          序列统计
        </CardTitle>
        <span className="text-[10px] text-muted-foreground/70">
          {rows.length === 0 ? "无数据" : `共 ${rows.length} 条曲线`}
        </span>
      </CardHeader>
      <CardContent className="p-0">
        {rows.length === 0 ? (
          <div className="px-4 py-6 text-xs text-muted-foreground">
            尚无可统计的曲线。
          </div>
        ) : (
          <table className="w-full text-[11px] tabular-nums">
            <thead className="border-b border-border/60 text-muted-foreground">
              <tr className="text-left">
                <th className="px-3 py-2 font-medium text-[10px] uppercase tracking-[0.12em]">
                  曲线
                </th>
                <th className="px-3 py-2 text-right font-medium text-[10px] uppercase tracking-[0.12em]">
                  采样数
                </th>
                <th className="px-3 py-2 text-right font-medium text-[10px] uppercase tracking-[0.12em]">
                  最新
                </th>
                <th className="px-3 py-2 text-right font-medium text-[10px] uppercase tracking-[0.12em]">
                  均值
                </th>
                <th className="px-3 py-2 text-right font-medium text-[10px] uppercase tracking-[0.12em]">
                  标准差
                </th>
                <th className="px-3 py-2 text-right font-medium text-[10px] uppercase tracking-[0.12em]">
                  峰值
                </th>
                <th className="px-3 py-2 text-right font-medium text-[10px] uppercase tracking-[0.12em]">
                  谷值
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr
                  key={r.id}
                  className="border-b border-border/30 last:border-b-0"
                >
                  <td className="px-3 py-1.5">
                    <span className="inline-flex items-center gap-2">
                      <span
                        className="inline-block size-2 rounded-full"
                        style={{ background: r.color }}
                        aria-hidden
                      />
                      <span className="text-foreground/85">{r.label}</span>
                    </span>
                  </td>
                  <td className="px-3 py-1.5 text-right text-muted-foreground">
                    {r.count}
                  </td>
                  <td className="px-3 py-1.5 text-right">{fmtVal(r.latest)}</td>
                  <td className="px-3 py-1.5 text-right">{fmtVal(r.avg)}</td>
                  <td className="px-3 py-1.5 text-right text-muted-foreground">
                    {fmtVal(r.std)}
                  </td>
                  <td className="px-3 py-1.5 text-right">{fmtVal(r.max)}</td>
                  <td className="px-3 py-1.5 text-right">{fmtVal(r.min)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </CardContent>
    </Card>
  )
}
