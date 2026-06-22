/**
 * LossChart — interactive multi-series SVG plot used by the loss panel.
 *
 * Features:
 *   - Pan / zoom on X via wheel + drag; double-click to reset.
 *   - Box-select zoom toggled from the toolbar.
 *   - Linear / log Y axis toggle.
 *   - EMA smoothing slider applied at render time on top of upstream
 *     dashed series (the parent already produces an EMA series by
 *     default; the slider lets the user retune α without round-tripping
 *     through the metrics endpoint).
 *   - Multi-series crosshair tooltip — every visible series reports its
 *     value at the hovered X.
 *   - Optional checkpoint markers (vertical dashed lines) and a
 *     train/val gap band when both curves are present.
 *   - Fullscreen modal for detailed inspection.
 */
import { useCallback, useEffect, useMemo, useState } from "react"
import type { CSSProperties } from "react"
import type { LossTooltip } from "./loss-chart-widgets"
import { cn } from "@/lib/utils"
import { downsamplePoints } from "../utils"
import { FullscreenModal } from "./loss-chart-fullscreen"
import { LegendRow, TooltipCard, TrendBadge } from "./loss-chart-widgets"
import { ChartToolbar } from "./chart-toolbar"
import {
  MAX_POINTS,
  PAD_BOTTOM,
  PAD_LEFT,
  PAD_RIGHT,
  PAD_TOP,
  VIEW_H,
  VIEW_W,
  formatLoss,
  trendCopy,
  type LossChartProps,
} from "./loss-chart-model"
import { useLossChartInteraction } from "./loss-chart-interaction"
export type { ChartBand, ChartMarker, LossSeries } from "./loss-chart-model"

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
  persistKey,
  fullscreen,
  onFullscreen,
}: CoreProps) {
  // ----- Series visibility (legend toggles) --------------------------------
  const [hidden, setHidden] = useState<Record<string, boolean>>({})
  useEffect(() => {
    setHidden((prev) => {
      const valid = new Set(series.map((s) => s.id))
      const next: Record<string, boolean> = {}
      for (const k of Object.keys(prev)) if (valid.has(k)) next[k] = prev[k]
      return next
    })
  }, [series])

  // ----- Resampled series (independent of view) ---------------------------
  const prepared = useMemo(() => {
    return series.map((s) => ({
      ...s,
      points: downsamplePoints(s.points, MAX_POINTS),
    }))
  }, [series])

  const visibleSeries = useMemo(
    () => prepared.filter((s) => !hidden[s.id]),
    [prepared, hidden],
  )

  // Full data extent (used for "reset" + when no zoom is active).
  const fullExtent = useMemo(() => {
    let xMin = Infinity
    let xMax = -Infinity
    for (const s of visibleSeries) {
      for (const p of s.points) {
        if (p.step < xMin) xMin = p.step
        if (p.step > xMax) xMax = p.step
      }
    }
    if (!Number.isFinite(xMin) || !Number.isFinite(xMax)) return null
    if (xMin === xMax) xMax = xMin + 1
    return { xMin, xMax }
  }, [visibleSeries])

  const innerW = VIEW_W - PAD_LEFT - PAD_RIGHT
  const innerH = VIEW_H - PAD_TOP - PAD_BOTTOM

  const baseRange = fullExtent
    ? ([fullExtent.xMin, fullExtent.xMax] as [number, number])
    : ([0, 1] as [number, number])

  const {
    hoverX,
    onDoubleClick,
    onPointerDown,
    onPointerLeave,
    onPointerMove,
    onPointerUp,
    onWheel,
    reset,
    selectMode,
    selectRect,
    setSelectMode,
    setYLog,
    svgRef,
    viewRange,
    viewXMax,
    viewXMin,
    yLog,
    zoomBy,
  } = useLossChartInteraction({
    persistKey,
    fullExtent,
    xMin: baseRange[0],
    xMax: baseRange[1],
    innerW,
  })

  const viewInverseX = useCallback(
    (px: number) =>
      viewXMin + ((px - PAD_LEFT) / innerW) * (viewXMax - viewXMin),
    [viewXMin, viewXMax, innerW],
  )

  // Y extent depends on the X clip (so zooming X recomputes Y).
  const yExtent = useMemo(() => {
    let lo = Infinity
    let hi = -Infinity
    for (const s of visibleSeries) {
      for (const p of s.points) {
        if (p.step < viewXMin || p.step > viewXMax) continue
        if (yLog && p.loss <= 0) continue
        if (p.loss < lo) lo = p.loss
        if (p.loss > hi) hi = p.loss
      }
    }
    // Bands need to fit within the y-window too — otherwise an IQR
    // drawn behind the series gets clipped at the top/bottom edge.
    for (const b of bands) {
      for (const p of b.points) {
        if (p.step < viewXMin || p.step > viewXMax) continue
        if (yLog && (p.lo <= 0 || p.hi <= 0)) continue
        if (p.lo < lo) lo = p.lo
        if (p.hi > hi) hi = p.hi
      }
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
      return { lo: 0, hi: 1 }
    }
    if (lo === hi) {
      const pad = Math.max(0.001, Math.abs(lo) * 0.05)
      return { lo: lo - pad, hi: hi + pad }
    }
    if (yLog) {
      const padLog = (Math.log10(hi) - Math.log10(lo)) * 0.06
      return { lo: lo / 10 ** padLog, hi: hi * 10 ** padLog }
    }
    const pad = (hi - lo) * 0.08
    return { lo: lo - pad, hi: hi + pad }
  }, [visibleSeries, bands, viewXMin, viewXMax, yLog])

  const xScale = useCallback(
    (step: number) =>
      PAD_LEFT + ((step - viewXMin) / (viewXMax - viewXMin || 1)) * innerW,
    [viewXMin, viewXMax, innerW],
  )
  const yScale = useCallback(
    (loss: number) => {
      if (yLog) {
        if (loss <= 0) return PAD_TOP + innerH
        const lLo = Math.log10(yExtent.lo)
        const lHi = Math.log10(yExtent.hi)
        return PAD_TOP + (1 - (Math.log10(loss) - lLo) / (lHi - lLo || 1)) * innerH
      }
      return (
        PAD_TOP +
        (1 - (loss - yExtent.lo) / (yExtent.hi - yExtent.lo || 1)) * innerH
      )
    },
    [yExtent, yLog, innerH],
  )

  // ----- Tick generation ---------------------------------------------------
  const yTicks = useMemo(() => {
    const out: number[] = []
    if (yLog) {
      const lLo = Math.log10(yExtent.lo)
      const lHi = Math.log10(yExtent.hi)
      const step = (lHi - lLo) / 4
      for (let i = 0; i <= 4; i += 1) out.push(10 ** (lHi - step * i))
    } else {
      for (let i = 0; i <= 4; i += 1)
        out.push(yExtent.hi - ((yExtent.hi - yExtent.lo) * i) / 4)
    }
    return out
  }, [yExtent, yLog])
  const xTicks = useMemo(() => {
    const out: number[] = []
    for (let i = 0; i <= 4; i += 1)
      out.push(viewXMin + ((viewXMax - viewXMin) * i) / 4)
    return out
  }, [viewXMin, viewXMax])

  // ----- Crosshair / tooltip data point picking ---------------------------
  const tooltip = useMemo<LossTooltip | null>(() => {
    if (hoverX == null) return null
    if (hoverX < PAD_LEFT || hoverX > VIEW_W - PAD_RIGHT) return null
    const targetStep = viewInverseX(hoverX)
    const items: Array<{
      seriesId: string
      label: string
      color: string
      step: number
      loss: number
      cy: number
    }> = []
    for (const s of visibleSeries) {
      let cand: { step: number; loss: number } | null = null
      let candDist = Infinity
      for (const p of s.points) {
        const d = Math.abs(p.step - targetStep)
        if (d < candDist) {
          candDist = d
          cand = p
        }
      }
      if (!cand) continue
      items.push({
        seriesId: s.id,
        label: s.label,
        color: s.color,
        step: cand.step,
        loss: cand.loss,
        cy: yScale(cand.loss),
      })
    }
    if (items.length === 0) return null
    // All items snap to the same nearest step on the densest series, so
    // the tooltip header just reads "step N" once.
    const step = items.reduce(
      (best, it) =>
        Math.abs(it.step - targetStep) < Math.abs(best.step - targetStep)
          ? it
          : best,
      items[0],
    ).step
    return { step, items, anchorX: xScale(step) }
  }, [hoverX, visibleSeries, viewInverseX, xScale, yScale])

  // ----- Train / val gap band --------------------------------------------
  const gapBand = useMemo(() => {
    if (visibleSeries.length < 2) return null
    const train = visibleSeries.find((s) => !s.dashed && s.id.endsWith("-train"))
    const val = visibleSeries.find((s) => s.id.endsWith("-val"))
    if (!train || !val || !train.points.length || !val.points.length) return null
    return val.points.map((vp) => {
      let bestLoss = train.points[0].loss
      let bestDist = Math.abs(train.points[0].step - vp.step)
      for (const tp of train.points) {
        const d = Math.abs(tp.step - vp.step)
        if (d < bestDist) {
          bestDist = d
          bestLoss = tp.loss
        }
      }
      return { step: vp.step, train: bestLoss, val: vp.loss }
    })
  }, [visibleSeries])

  function toggleSeries(id: string) {
    setHidden((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  function downloadCsv() {
    const lines: string[] = ["series,step,loss"]
    for (const s of prepared) {
      for (const p of s.points)
        lines.push(`${s.label},${p.step},${p.loss}`)
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

  const trend = overfitSignal ? trendCopy(overfitSignal.trend) : null
  const zoomedIn = viewRange !== null
  const hasData = !!fullExtent

  // Height in px for the SVG; fullscreen pushes the canvas taller.
  const heightStyle = fullscreen ? "h-[70vh]" : "h-auto"

  return (
    <div className={cn("relative w-full", className)}>
      <div className="absolute right-2 top-2 z-10 flex items-start gap-2">
        <TrendBadge trend={trend} gap={overfitSignal?.gap} />
        <button
          type="button"
          onClick={() => setYLog((v) => !v)}
          className={cn(
            "h-6 rounded-[3px] border px-1.5 text-[10.5px] font-medium transition-colors",
            yLog
              ? "border-primary/40 bg-primary/15 text-primary"
              : "border-border/60 bg-background/85 text-muted-foreground hover:text-foreground",
          )}
          title="切换线性 / 对数 Y 轴"
        >
          Y · {yLog ? "log" : "lin"}
        </button>
        <ChartToolbar
          zoomedIn={zoomedIn}
          selectMode={selectMode}
          onZoomIn={() => zoomBy(1.5)}
          onZoomOut={() => zoomBy(1 / 1.5)}
          onToggleSelect={() => setSelectMode((v) => !v)}
          onReset={reset}
          onFullscreen={onFullscreen}
          onDownload={downloadCsv}
        />
      </div>

      <div className="w-full overflow-hidden">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
          className={cn("block w-full select-none", heightStyle)}
          style={{ color: "var(--primary)" } as CSSProperties}
          onWheel={onWheel}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={onPointerLeave}
          onDoubleClick={onDoubleClick}
        >
          <rect
            x={PAD_LEFT}
            y={PAD_TOP}
            width={innerW}
            height={innerH}
            fill="transparent"
            stroke="currentColor"
            strokeOpacity={0.08}
          />
          {/* Y ticks */}
          {yTicks.map((v, i) => {
            const y = yScale(v)
            return (
              <g key={`y${i}`}>
                <line
                  x1={PAD_LEFT}
                  x2={VIEW_W - PAD_RIGHT}
                  y1={y}
                  y2={y}
                  stroke="currentColor"
                  strokeOpacity={0.07}
                  strokeDasharray="3 4"
                />
                <text
                  x={PAD_LEFT - 6}
                  y={y}
                  textAnchor="end"
                  dominantBaseline="middle"
                  fontSize={10}
                  fill="currentColor"
                  opacity={0.55}
                >
                  {formatLoss(v)}
                </text>
              </g>
            )
          })}
          {/* X ticks */}
          {xTicks.map((v, i) => {
            const x = xScale(v)
            return (
              <g key={`x${i}`}>
                <line
                  x1={x}
                  x2={x}
                  y1={VIEW_H - PAD_BOTTOM}
                  y2={VIEW_H - PAD_BOTTOM + 4}
                  stroke="currentColor"
                  strokeOpacity={0.3}
                />
                <text
                  x={x}
                  y={VIEW_H - PAD_BOTTOM + 16}
                  textAnchor="middle"
                  fontSize={10}
                  fill="currentColor"
                  opacity={0.55}
                >
                  {xTickFormat ? xTickFormat(v) : Math.round(v)}
                </text>
              </g>
            )
          })}
          {/* Confidence bands (e.g. rolling IQR around the median) */}
          {hasData &&
            bands.map((b) => {
              const pts = b.points.filter(
                (p) => p.step >= viewXMin && p.step <= viewXMax,
              )
              if (pts.length < 2) return null
              const polyPoints = [
                ...pts.map((p) => `${xScale(p.step)},${yScale(p.hi)}`),
                ...[...pts]
                  .reverse()
                  .map((p) => `${xScale(p.step)},${yScale(p.lo)}`),
              ].join(" ")
              return (
                <polygon
                  key={`band-${b.id}`}
                  points={polyPoints}
                  fill={b.color}
                  stroke="none"
                  pointerEvents="none"
                />
              )
            })}
          {/* Gap band */}
          {hasData && gapBand && gapBand.length >= 2 && (
            <polygon
              points={[
                ...gapBand.map((g) => `${xScale(g.step)},${yScale(g.val)}`),
                ...[...gapBand]
                  .reverse()
                  .map((g) => `${xScale(g.step)},${yScale(g.train)}`),
              ].join(" ")}
              fill={
                overfitSignal?.trend === "overfitting"
                  ? "color-mix(in oklch, var(--destructive) 18%, transparent)"
                  : "color-mix(in oklch, var(--chart-2) 14%, transparent)"
              }
              stroke="none"
            />
          )}
          {/* Markers */}
          {hasData &&
            markers
              .filter((m) => m.step >= viewXMin && m.step <= viewXMax)
              .map((m, i) => {
                const x = xScale(m.step)
                const stroke = m.color ?? "var(--chart-3)"
                return (
                  <g key={`mk-${i}`} pointerEvents="none">
                    <line
                      x1={x}
                      x2={x}
                      y1={PAD_TOP}
                      y2={VIEW_H - PAD_BOTTOM}
                      stroke={stroke}
                      strokeOpacity={0.55}
                      strokeWidth={1}
                      strokeDasharray="3 3"
                    />
                    <circle cx={x} cy={PAD_TOP + 3} r={2.5} fill={stroke} />
                  </g>
                )
              })}
          {/* Polylines */}
          {hasData ? (
            visibleSeries.map((s) => (
              <polyline
                key={s.id}
                fill="none"
                stroke={s.color}
                strokeWidth={s.dashed ? 1.25 : 1.5}
                strokeDasharray={s.dashed ? "4 3" : undefined}
                strokeOpacity={s.dashed ? 0.85 : 1}
                strokeLinejoin="round"
                strokeLinecap="round"
                points={s.points
                  .filter((p) => p.step >= viewXMin && p.step <= viewXMax)
                  .map((p) => `${xScale(p.step)},${yScale(p.loss)}`)
                  .join(" ")}
              />
            ))
          ) : (
            <text
              x={VIEW_W / 2}
              y={VIEW_H / 2}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize={12}
              fill="currentColor"
              opacity={0.55}
            >
              {emptyHint}
            </text>
          )}
          {/* Box-select rectangle */}
          {selectRect && (
            <rect
              x={Math.min(selectRect.x0, selectRect.x1)}
              y={PAD_TOP}
              width={Math.abs(selectRect.x1 - selectRect.x0)}
              height={innerH}
              fill="color-mix(in oklch, var(--primary) 14%, transparent)"
              stroke="var(--primary)"
              strokeOpacity={0.5}
              strokeDasharray="3 3"
            />
          )}
          {/* Crosshair + dots */}
          {tooltip && (
            <g pointerEvents="none">
              <line
                x1={tooltip.anchorX}
                x2={tooltip.anchorX}
                y1={PAD_TOP}
                y2={VIEW_H - PAD_BOTTOM}
                stroke="currentColor"
                strokeOpacity={0.25}
                strokeDasharray="2 4"
              />
              {tooltip.items.map((it) => (
                <circle
                  key={it.seriesId}
                  cx={tooltip.anchorX}
                  cy={it.cy}
                  r={3.25}
                  fill={it.color}
                />
              ))}
            </g>
          )}
        </svg>
      </div>

      <TooltipCard tooltip={tooltip} />

      {/* Legend / range chip row */}
      <LegendRow
        series={prepared}
        hidden={hidden}
        markersCount={markers.length}
        xLabel={xLabel}
        zoomedIn={zoomedIn}
        xMin={viewXMin}
        xMax={viewXMax}
        fullXMax={fullExtent?.xMax}
        onToggleSeries={toggleSeries}
        onReset={reset}
      />
    </div>
  )
}
