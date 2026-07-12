import { useEffect, useMemo, useState } from "react"
import type { CSSProperties, WheelEvent } from "react"
import { ChartTooltip } from "@/components/charts/tooltip"
import { decimateTimeSeries } from "@/components/charts/decimate-time-series"
import { Grid } from "@/components/charts/grid"
import { Line } from "@/components/charts/line"
import { LineChart as BklitLineChart } from "@/components/charts/line-chart"
import { chartCssVars, useChartStable } from "@/components/charts/chart-context"
import {
  ChartNumericXAxis,
  ChartNumericYAxis,
} from "@/components/charts/numeric-axis"
import { cn } from "@/lib/utils"
import { FullscreenModal } from "./loss-chart-fullscreen"
import { LegendRow, TrendBadge } from "./loss-chart-widgets"
import { ChartToolbar } from "./chart-toolbar"
import {
  MAX_POINTS,
  formatLoss,
  trendCopy,
  type LossChartProps,
} from "./loss-chart-model"

export type { ChartBand, ChartMarker, LossSeries } from "./loss-chart-model"

const STEP_MS = 1000
const ZOOM_STEP = 1.25

export function LossChart(props: LossChartProps) {
  const [fullscreen, setFullscreen] = useState(false)
  return (
    <>
      <LossChartCore
        {...props}
        fullscreen={false}
        onFullscreen={() => setFullscreen(true)}
      />
      {fullscreen && (
        <FullscreenModal onClose={() => setFullscreen(false)}>
          <LossChartCore {...props} fullscreen onFullscreen={undefined} />
        </FullscreenModal>
      )}
    </>
  )
}

interface CoreProps extends LossChartProps {
  fullscreen: boolean
  onFullscreen?: () => void
}

function LossChartCore({
  series,
  className,
  emptyHint = "暂无损失数据。",
  overfitSignal,
  markers = [],
  bands = [],
  xLabel,
  xTickFormat,
  fullscreen,
  onFullscreen,
}: CoreProps) {
  const [hidden, setHidden] = useState<Record<string, boolean>>({})
  const [viewDomain, setViewDomain] = useState<[number, number] | null>(null)
  useEffect(() => {
    setHidden((prev) => {
      const valid = new Set(series.map((s) => s.id))
      const next: Record<string, boolean> = {}
      for (const key of Object.keys(prev)) {
        if (valid.has(key)) next[key] = prev[key]
      }
      return next
    })
  }, [series])

  const { prepared, preparedBands } = useMemo(() => {
    const anchor = series[0]?.points ?? []
    const sampledAnchor = decimateTimeSeries(anchor, MAX_POINTS, ["loss"])
    const indexByPoint = new Map(anchor.map((point, index) => [point, index]))
    const sampledIndices = sampledAnchor
      .map((point) => indexByPoint.get(point))
      .filter((index): index is number => index != null)
    const sample = <T extends { step: number }>(
      points: T[],
      valueKeys: string[],
    ) => {
      if (points.length <= MAX_POINTS) return points
      if (sameStepGrid(anchor, points)) {
        return sampledIndices.map((index) => points[index]).filter(Boolean)
      }
      return decimateTimeSeries(points, MAX_POINTS, valueKeys)
    }
    return {
      prepared: series.map((item) => ({
        ...item,
        points: sample(item.points, ["loss"]),
      })),
      preparedBands: bands.map((band) => ({
        ...band,
        points: sample(band.points, ["lo", "hi"]),
      })),
    }
  }, [bands, series])
  const visibleSeries = prepared.filter((s) => !hidden[s.id])
  const data = useMemo(
    () => mergeLossData(visibleSeries, preparedBands),
    [preparedBands, visibleSeries],
  )
  const fullExtent = useMemo(() => {
    if (data.length === 0) return null
    const steps = data
      .map((row) => row.step)
      .filter((step): step is number => typeof step === "number")
    if (steps.length === 0) return null
    const min = Math.min(...steps)
    const max = Math.max(...steps)
    return min === max ? { min, max: min + 1 } : { min, max }
  }, [data])
  useEffect(() => {
    if (!fullExtent || !viewDomain) return
    if (viewDomain[1] <= fullExtent.min || viewDomain[0] >= fullExtent.max) {
      setViewDomain(null)
    }
  }, [fullExtent, viewDomain])
  const trend = overfitSignal ? trendCopy(overfitSignal.trend) : null
  const dataSteps = useMemo(() => new Set(data.map((row) => row.step)), [data])
  const visibleMarkers = useMemo(
    () => markers.filter((marker) => dataSteps.has(marker.step)),
    [dataSteps, markers],
  )
  const heavyChart =
    data.length > 500 ||
    prepared.reduce((sum, item) => sum + item.points.length, 0) > 1_000

  function toggleSeries(id: string) {
    setHidden((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  function downloadCsv() {
    const lines = ["series,step,loss"]
    for (const s of prepared) {
      for (const p of s.points) lines.push(`${s.label},${p.step},${p.loss}`)
    }
    const blob = new Blob([lines.join("\n")], {
      type: "text/csv;charset=utf-8",
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "loss.csv"
    a.click()
    URL.revokeObjectURL(url)
  }

  function zoomBy(factor: number, anchor?: number) {
    if (!fullExtent) return
    setViewDomain((current) => {
      const min = fullExtent.min
      const max = fullExtent.max
      const span = max - min
      const from = current ?? [min, max]
      const fromSpan = from[1] - from[0]
      const nextSpan = Math.max(1, Math.min(span, fromSpan / factor))
      if (nextSpan >= span) return null
      const pivot = anchor ?? (from[0] + from[1]) / 2
      const ratio = (pivot - from[0]) / fromSpan
      let nextMin = pivot - ratio * nextSpan
      let nextMax = nextMin + nextSpan
      if (nextMin < min) {
        nextMin = min
        nextMax = min + nextSpan
      }
      if (nextMax > max) {
        nextMax = max
        nextMin = max - nextSpan
      }
      return [nextMin, nextMax]
    })
  }

  function handleWheel(event: WheelEvent<HTMLDivElement>) {
    if (!fullExtent) return
    event.preventDefault()
    event.stopPropagation()
    const rect = event.currentTarget.getBoundingClientRect()
    const current = viewDomain ?? [fullExtent.min, fullExtent.max]
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width))
    const anchor = current[0] + (current[1] - current[0]) * ratio
    zoomBy(event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP, anchor)
  }

  const xDomain =
    viewDomain != null
      ? ([new Date(viewDomain[0] * STEP_MS), new Date(viewDomain[1] * STEP_MS)] as [
          Date,
          Date,
        ])
      : undefined
  const zoomedIn = viewDomain != null

  return (
    <div
      className={cn("relative w-full", className)}
      onDoubleClick={() => setViewDomain(null)}
      onWheelCapture={handleWheel}
    >
      <div className="absolute right-2 top-2 z-10 flex items-start gap-2">
        <TrendBadge trend={trend} gap={overfitSignal?.gap} />
        <ChartToolbar
          zoomedIn={zoomedIn}
          onZoomIn={() => zoomBy(1.5)}
          onZoomOut={() => zoomBy(1 / 1.5)}
          onReset={() => setViewDomain(null)}
          onFullscreen={onFullscreen}
          onDownload={downloadCsv}
        />
      </div>

      {data.length === 0 ? (
        <div className="grid h-[260px] place-items-center text-sm text-muted-foreground">
          {emptyHint}
        </div>
      ) : (
        <BklitLineChart
          data={data}
          xDataKey="date"
          animationDuration={heavyChart ? 0 : 900}
          aspectRatio={fullscreen ? undefined : "16 / 6"}
          className={fullscreen ? "h-[70vh]" : "min-h-[260px]"}
          margin={{ top: 46, right: 28, bottom: 42, left: 56 }}
          style={fullscreen ? ({ height: "70vh" } as CSSProperties) : undefined}
          tweenYDomainOnXDomainChange
          xDomain={xDomain}
          xDomainSlotCount={data.length}
          yDomainTween={!heavyChart}
        >
          <Grid horizontal vertical={false} hideHorizontalEdgeLines />
          <LossBandLayer bands={preparedBands} />
          <MarkerLayer markers={visibleMarkers} />
          {visibleSeries.map((s, index) => (
            <Line
              key={s.id}
              dataKey={s.id}
              stroke={s.color || `var(--chart-${(index % 5) + 1})`}
              strokeWidth={s.dashed ? 2 : 1.6}
              dashFromIndex={s.dashed ? 0 : undefined}
              dashArray={s.dashed ? "5,4" : undefined}
              animate={!heavyChart}
              fadeEdges={!heavyChart}
              showHighlight={!heavyChart}
              showMarkers={s.points.length <= 80}
              markers={{
                inactiveBlur: 0,
                radius: 2,
                ringGap: 0,
                showActiveHighlight: false,
                strokeWidth: 1,
              }}
            />
          ))}
          <ChartNumericYAxis format={formatLoss} />
          <ChartNumericXAxis format={xTickFormat} label={xLabel} />
          <ChartTooltip
            backgroundColor={chartCssVars.tooltipBackground}
            showDatePill={false}
            rows={(point) =>
              visibleSeries
                .map((s) => {
                  const value = point[s.id]
                  return typeof value === "number"
                    ? { color: s.color, label: s.label, value: formatLoss(value) }
                    : null
                })
                .filter((row): row is { color: string; label: string; value: string } =>
                  row != null,
                )
            }
            content={({ point }) => (
              <div className="px-3 py-2 text-xs">
                <div className="mb-2 text-[10px] uppercase tracking-[0.18em] text-chart-tooltip-muted">
                  {xLabel ?? "step"} {formatAxisValue(point.step, xTickFormat)}
                </div>
                <div className="space-y-1.5">
                  {visibleSeries.map((s) => {
                    const value = point[s.id]
                    if (typeof value !== "number") return null
                    return (
                      <div
                        key={s.id}
                        className="flex items-center justify-between gap-4"
                      >
                        <span className="flex items-center gap-2 text-chart-tooltip-muted">
                          <span
                            className="size-2.5 rounded-full"
                            style={{ background: s.color }}
                          />
                          {s.label}
                        </span>
                        <span className="font-medium tabular-nums text-chart-tooltip-foreground">
                          {formatLoss(value)}
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          />
        </BklitLineChart>
      )}

      <LegendRow
        bands={preparedBands}
        series={prepared}
        hidden={hidden}
        markersCount={markers.length}
        xLabel={xLabel}
        zoomedIn={zoomedIn}
        xMin={viewDomain?.[0] ?? 0}
        xMax={viewDomain?.[1] ?? 0}
        fullXMax={fullExtent?.max}
        onToggleSeries={toggleSeries}
        onReset={() => setViewDomain(null)}
      />
    </div>
  )
}

function formatAxisValue(
  value: unknown,
  formatter?: (value: number) => string,
) {
  if (typeof value !== "number") return "?"
  return formatter ? formatter(value) : String(Math.round(value))
}

function sameStepGrid(
  anchor: { step: number }[],
  points: { step: number }[],
) {
  return (
    anchor.length > 0 &&
    anchor.length === points.length &&
    anchor.every((point, index) => point.step === points[index].step)
  )
}

function mergeLossData(
  series: { id: string; points: { step: number; loss: number }[] }[],
  bands: { id: string; points: { step: number; lo: number; hi: number }[] }[],
) {
  const byStep = new Map<number, Record<string, unknown>>()
  const rowForStep = (step: number) => {
    const row =
      byStep.get(step) ??
      ({
        step,
        date: new Date(step * STEP_MS),
      } satisfies Record<string, unknown>)
    byStep.set(step, row)
    return row
  }

  for (const s of series) {
    for (const p of s.points) {
      const row = rowForStep(p.step)
      row[s.id] = p.loss
    }
  }
  for (const b of bands) {
    for (const p of b.points) {
      const row = rowForStep(p.step)
      row[`${b.id}:lo`] = p.lo
      row[`${b.id}:hi`] = p.hi
    }
  }
  return [...byStep.values()].sort(
    (a, b) => Number(a.step) - Number(b.step),
  )
}

function LossBandLayer({
  bands,
}: {
  bands: { id: string; color: string; points: { step: number; lo: number; hi: number }[] }[]
}) {
  const { data, innerHeight, xAccessor, xScale, yScale } = useChartStable()
  const rowsByStep = useMemo(
    () => new Map(data.map((row) => [row.step, row])),
    [data],
  )
  if (bands.length === 0 || data.length === 0) return null

  return (
    <g className="loss-chart-bands" pointerEvents="none">
      {bands.map((band) => {
        const points = band.points
          .map((point) => {
            const row = rowsByStep.get(point.step)
            if (!row) return null
            return {
              x: xScale(xAccessor(row)) ?? 0,
              yHi: yScale(point.hi) ?? 0,
              yLo: yScale(point.lo) ?? innerHeight,
            }
          })
          .filter(
            (point): point is { x: number; yHi: number; yLo: number } =>
              point !== null,
          )
        if (points.length < 2) return null
        const polygon = [
          ...points.map((point) => `${point.x},${point.yHi}`),
          ...[...points].reverse().map((point) => `${point.x},${point.yLo}`),
        ].join(" ")
        return <polygon fill={band.color} key={band.id} points={polygon} />
      })}
    </g>
  )
}

function MarkerLayer({
  markers,
}: {
  markers: { step: number; color?: string }[]
}) {
  const { data, innerHeight, xAccessor, xScale } = useChartStable()
  const rowsByStep = useMemo(
    () => new Map(data.map((row) => [row.step, row])),
    [data],
  )
  if (markers.length === 0 || data.length === 0) return null

  return (
    <g className="loss-chart-markers" pointerEvents="none">
      {markers.map((marker, index) => {
        const row = rowsByStep.get(marker.step)
        if (!row) return null
        const x = xScale(xAccessor(row)) ?? 0
        const color = marker.color ?? chartCssVars.foregroundMuted
        return (
          <g key={`${marker.step}-${index}`}>
            <line
              stroke={color}
              strokeDasharray="4,4"
              strokeOpacity={0.5}
              x1={x}
              x2={x}
              y1={0}
              y2={innerHeight}
            />
            <circle cx={x} cy={4} fill={color} r={2.5} />
          </g>
        )
      })}
    </g>
  )
}
