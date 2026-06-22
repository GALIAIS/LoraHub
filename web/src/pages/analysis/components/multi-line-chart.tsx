import { useEffect, useMemo, useState } from "react"
import type { CSSProperties, WheelEvent } from "react"
import { createPortal } from "react-dom"
import { EyeOff, X } from "lucide-react"
import { ChartTooltip } from "@/components/charts/tooltip"
import { chartCssVars } from "@/components/charts/chart-context"
import { Grid } from "@/components/charts/grid"
import { Line } from "@/components/charts/line"
import { LineChart as BklitLineChart } from "@/components/charts/line-chart"
import {
  ChartNumericXAxis,
  ChartNumericYAxis,
} from "@/components/charts/numeric-axis"
import { cn } from "@/lib/utils"
import { ChartToolbar } from "../../jobs/components/chart-toolbar"

export interface MultiLinePoint {
  x: number
  y: number | null
}

export interface MultiLineSeries {
  id: string
  label: string
  color: string
  unit?: string
  axis?: "left" | "right"
  points: MultiLinePoint[]
}

const STEP_MS = 1000
const ZOOM_STEP = 1.25

function fmtNum(v: number): string {
  if (!Number.isFinite(v)) return "-"
  if (v === 0) return "0"
  const abs = Math.abs(v)
  if (abs >= 100) return v.toFixed(1)
  if (abs >= 1) return v.toFixed(2)
  if (abs >= 0.01) return v.toFixed(3)
  return v.toExponential(2)
}

interface Props {
  series: MultiLineSeries[]
  xLabel?: string
  emptyHint?: string
  className?: string
  persistKey?: string | null
  title?: string
}

export function MultiLineChart(props: Props) {
  const [fullscreen, setFullscreen] = useState(false)
  return (
    <>
      <Core {...props} fullscreen={false} onFullscreen={() => setFullscreen(true)} />
      {fullscreen && (
        <FullscreenModal title={props.title} onClose={() => setFullscreen(false)}>
          <Core {...props} fullscreen onFullscreen={undefined} />
        </FullscreenModal>
      )}
    </>
  )
}

interface CoreProps extends Props {
  fullscreen: boolean
  onFullscreen?: () => void
}

function Core({
  series,
  xLabel,
  emptyHint = "暂无数据",
  className,
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

  const visible = useMemo(
    () => series.filter((s) => !hidden[s.id]),
    [hidden, series],
  )
  const data = useMemo(() => mergeSeriesData(visible), [visible])
  const fullExtent = useMemo(() => {
    if (data.length === 0) return null
    const xs = data
      .map((row) => row.x)
      .filter((x): x is number => typeof x === "number")
    if (xs.length === 0) return null
    const min = Math.min(...xs)
    const max = Math.max(...xs)
    return min === max ? { min, max: min + 1 } : { min, max }
  }, [data])
  useEffect(() => {
    if (!fullExtent || !viewDomain) return
    if (viewDomain[1] <= fullExtent.min || viewDomain[0] >= fullExtent.max) {
      setViewDomain(null)
    }
  }, [fullExtent, viewDomain])
  const hasData = data.length > 0

  function toggle(id: string) {
    setHidden((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  function downloadCsv() {
    const lines = ["series,x,y"]
    for (const s of series) {
      for (const p of s.points) {
        if (p.y == null || !Number.isFinite(p.y)) continue
        lines.push(`${s.label},${p.x},${p.y}`)
      }
    }
    const blob = new Blob([lines.join("\n")], {
      type: "text/csv;charset=utf-8",
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "series.csv"
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
      className={cn("relative flex flex-col gap-2", className)}
      onDoubleClick={() => setViewDomain(null)}
      onWheelCapture={handleWheel}
    >
      <div className="absolute right-1 top-1 z-10 flex items-start gap-1.5">
        <ChartToolbar
          zoomedIn={zoomedIn}
          onZoomIn={() => zoomBy(1.5)}
          onZoomOut={() => zoomBy(1 / 1.5)}
          onReset={() => setViewDomain(null)}
          onFullscreen={onFullscreen}
          onDownload={downloadCsv}
        />
      </div>

      {hasData ? (
        <BklitLineChart
          data={data}
          xDataKey="date"
          animationDuration={850}
          aspectRatio={fullscreen ? undefined : "16 / 6"}
          className={fullscreen ? "h-[60vh]" : "min-h-[210px]"}
          margin={{ top: 34, right: 18, bottom: 44, left: 52 }}
          style={fullscreen ? ({ height: "60vh" } as CSSProperties) : undefined}
          tweenYDomainOnXDomainChange
          xDomain={xDomain}
          xDomainSlotCount={data.length}
        >
          <Grid horizontal vertical={false} hideHorizontalEdgeLines />
          {visible.map((s, index) => (
            <Line
              dataKey={s.id}
              key={s.id}
              showMarkers={s.points.length <= 40}
              markers={{
                inactiveBlur: 0,
                radius: 2,
                ringGap: 0,
                showActiveHighlight: false,
                strokeWidth: 1,
              }}
              stroke={s.color || `var(--chart-${(index % 5) + 1})`}
              strokeWidth={1.8}
            />
          ))}
          <ChartNumericYAxis format={fmtNum} />
          <ChartNumericXAxis label={xLabel} />
          <ChartTooltip
            backgroundColor={chartCssVars.tooltipBackground}
            showDatePill={false}
            rows={(point) =>
              visible
                .map((s) => {
                  const value = point[s.id]
                  return typeof value === "number"
                    ? {
                        color: s.color,
                        label: s.label,
                        value: `${fmtNum(value)}${s.unit ?? ""}`,
                      }
                    : null
                })
                .filter((row): row is { color: string; label: string; value: string } =>
                  row != null,
                )
            }
            content={({ point }) => (
              <div className="px-3 py-2 text-xs">
                <div className="mb-2 text-[10px] uppercase tracking-[0.16em] text-chart-tooltip-muted">
                  x = {fmtNum(typeof point.x === "number" ? point.x : Number.NaN)}
                </div>
                <div className="space-y-1.5">
                  {visible.map((s) => {
                    const value = point[s.id]
                    if (typeof value !== "number") return null
                    return (
                      <div
                        className="flex items-center justify-between gap-4"
                        key={s.id}
                      >
                        <span className="flex min-w-0 items-center gap-2 text-chart-tooltip-muted">
                          <span
                            aria-hidden
                            className="size-2.5 shrink-0 rounded-full"
                            style={{ background: s.color }}
                          />
                          <span className="truncate">{s.label}</span>
                        </span>
                        <span className="font-medium tabular-nums text-chart-tooltip-foreground">
                          {fmtNum(value)}
                          {s.unit ?? ""}
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          />
        </BklitLineChart>
      ) : (
        <div className="grid min-h-[210px] place-items-center text-xs text-muted-foreground">
          {emptyHint}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]">
        {series.map((s) => {
          const off = !!hidden[s.id]
          const last = [...s.points]
            .reverse()
            .find((p): p is { x: number; y: number } => typeof p.y === "number")
          return (
            <button
              className={cn(
                "group inline-flex max-w-full items-center gap-1.5 rounded-[3px] border px-1.5 py-0.5 transition-colors",
                off
                  ? "border-border/40 text-muted-foreground/70"
                  : "border-border/60 bg-muted/40 text-foreground/85",
              )}
              key={s.id}
              onClick={() => toggle(s.id)}
              title={off ? "点击显示" : "点击隐藏"}
              type="button"
            >
              {off ? (
                <EyeOff className="size-3 shrink-0" />
              ) : (
                <span
                  aria-hidden
                  className="inline-block h-[2px] w-3 shrink-0 align-middle"
                  style={{ background: s.color }}
                />
              )}
              <span className={cn("truncate", off && "line-through")}>
                {s.label}
                {s.axis === "right" ? " (右)" : ""}
              </span>
              {!off && last && (
                <span className="shrink-0 text-muted-foreground/70 tabular-nums">
                  {fmtNum(last.y)}
                  {s.unit ?? ""}
                </span>
              )}
            </button>
          )
        })}
        {xLabel && (
          <span className="ml-auto text-[10px] text-muted-foreground/70">
            {zoomedIn && viewDomain
              ? `${Math.round(viewDomain[0])}-${Math.round(viewDomain[1])} · `
              : ""}
            {xLabel}
          </span>
        )}
      </div>
    </div>
  )
}

function mergeSeriesData(series: MultiLineSeries[]) {
  const byX = new Map<number, Record<string, unknown>>()
  for (const s of series) {
    for (const p of s.points) {
      if (p.y == null || !Number.isFinite(p.y)) continue
      const row =
        byX.get(p.x) ??
        ({
          x: p.x,
          step: p.x,
          date: new Date(p.x * STEP_MS),
        } satisfies Record<string, unknown>)
      row[s.id] = p.y
      byX.set(p.x, row)
    }
  }
  return [...byX.values()].sort((a, b) => Number(a.x) - Number(b.x))
}

function FullscreenModal({
  children,
  onClose,
  title,
}: {
  children: React.ReactNode
  onClose: () => void
  title?: string
}) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onClose])

  if (typeof document === "undefined") return null
  return createPortal(
    <div
      className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/60 p-6"
      onClick={onClose}
    >
      <div
        className="relative w-[90vw] max-w-[1280px] rounded-[6px] border border-border/60 bg-background p-4 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-2 flex items-center justify-between">
          <span className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
            {title ?? "图表"}
          </span>
          <button
            aria-label="关闭全屏"
            className="inline-flex size-7 items-center justify-center rounded-[3px] text-muted-foreground hover:bg-muted hover:text-foreground"
            onClick={onClose}
            type="button"
          >
            <X className="size-4" />
          </button>
        </div>
        {children}
      </div>
    </div>,
    document.body,
  )
}
