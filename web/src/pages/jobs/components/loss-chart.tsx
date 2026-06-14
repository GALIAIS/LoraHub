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
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
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

  // ----- View state: x range, log/linear Y axis, gesture mode --------------
  // We hydrate from sessionStorage on first mount when persistKey is set,
  // so reloading / re-entering the workbench keeps the user's zoom + log
  // toggle without polluting the URL.
  const storageKey = persistKey ? `lorahub.loss.${persistKey}` : null
  const [yLog, setYLog] = useState<boolean>(() => {
    if (!storageKey) return false
    try {
      const raw = window.sessionStorage.getItem(storageKey)
      if (raw) return JSON.parse(raw)?.yLog === true
    } catch {
      // Ignore corrupt storage.
    }
    return false
  })
  const [selectMode, setSelectMode] = useState(false)
  // Null = auto extent. Setting a range puts the chart into "user-zoomed"
  // mode; live data appended afterwards no longer rescales the view.
  const [viewRange, setViewRange] = useState<[number, number] | null>(() => {
    if (!storageKey) return null
    try {
      const raw = window.sessionStorage.getItem(storageKey)
      if (!raw) return null
      const parsed = JSON.parse(raw)
      const xr = parsed?.xRange
      if (Array.isArray(xr) && xr.length === 2 && xr.every(Number.isFinite))
        return [xr[0], xr[1]]
    } catch {
      // Ignore corrupt storage.
    }
    return null
  })

  // Persist on change.
  useEffect(() => {
    if (!storageKey) return
    try {
      window.sessionStorage.setItem(
        storageKey,
        JSON.stringify({
          xRange: viewRange,
          yLog,
        }),
      )
    } catch {
      // Quota exceeded or disabled — silently skip.
    }
  }, [storageKey, viewRange, yLog])

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

  const effectiveX = viewRange ?? (fullExtent ? [fullExtent.xMin, fullExtent.xMax] : [0, 1])
  const xMin = effectiveX[0]
  const xMax = effectiveX[1]

  // Y extent depends on the X clip (so zooming X recomputes Y).
  const yExtent = useMemo(() => {
    let lo = Infinity
    let hi = -Infinity
    for (const s of visibleSeries) {
      for (const p of s.points) {
        if (p.step < xMin || p.step > xMax) continue
        if (yLog && p.loss <= 0) continue
        if (p.loss < lo) lo = p.loss
        if (p.loss > hi) hi = p.loss
      }
    }
    // Bands need to fit within the y-window too — otherwise an IQR
    // drawn behind the series gets clipped at the top/bottom edge.
    for (const b of bands) {
      for (const p of b.points) {
        if (p.step < xMin || p.step > xMax) continue
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
  }, [visibleSeries, bands, xMin, xMax, yLog])

  const innerW = VIEW_W - PAD_LEFT - PAD_RIGHT
  const innerH = VIEW_H - PAD_TOP - PAD_BOTTOM

  const xScale = useCallback(
    (step: number) =>
      PAD_LEFT + ((step - xMin) / (xMax - xMin || 1)) * innerW,
    [xMin, xMax, innerW],
  )
  const inverseX = useCallback(
    (px: number) => xMin + ((px - PAD_LEFT) / innerW) * (xMax - xMin),
    [xMin, xMax, innerW],
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
      out.push(xMin + ((xMax - xMin) * i) / 4)
    return out
  }, [xMin, xMax])

  // ----- Pointer / gesture handling ---------------------------------------
  const svgRef = useRef<SVGSVGElement | null>(null)
  const [hoverX, setHoverX] = useState<number | null>(null)
  // Pan in progress when set; tracks last pointer X in viewBox coords.
  const panRef = useRef<{ lastVX: number } | null>(null)
  // Box-select rectangle in viewBox coords.
  const [selectRect, setSelectRect] = useState<
    { x0: number; x1: number } | null
  >(null)

  function clientToViewBox(e: React.PointerEvent | React.WheelEvent): number {
    const svg = svgRef.current
    if (!svg) return 0
    const rect = svg.getBoundingClientRect()
    return ((e.clientX - rect.left) / rect.width) * VIEW_W
  }

  function setRangeClamped(lo: number, hi: number) {
    if (!fullExtent) return
    const span = hi - lo
    if (span <= 0) return
    // Clamp to full extent; refuse to zoom in narrower than 0.5% of full
    // range — beyond that the chart becomes useless and the user has
    // no way to read tick labels.
    const fullSpan = fullExtent.xMax - fullExtent.xMin
    const minSpan = fullSpan * 0.005
    if (span < minSpan) return
    let nlo = Math.max(fullExtent.xMin, lo)
    let nhi = Math.min(fullExtent.xMax, hi)
    if (nhi - nlo < minSpan) {
      const center = (nlo + nhi) / 2
      nlo = center - minSpan / 2
      nhi = center + minSpan / 2
    }
    if (nlo === fullExtent.xMin && nhi === fullExtent.xMax) {
      setViewRange(null)
    } else {
      setViewRange([nlo, nhi])
    }
  }

  function zoomBy(factor: number, anchorVX?: number) {
    const lo = xMin
    const hi = xMax
    const anchor =
      anchorVX != null ? inverseX(anchorVX) : (lo + hi) / 2
    const span = (hi - lo) / factor
    setRangeClamped(anchor - span * ((anchor - lo) / (hi - lo || 1)), anchor + span * ((hi - anchor) / (hi - lo || 1)))
  }

  function reset() {
    setViewRange(null)
    setSelectRect(null)
  }

  function onWheel(e: React.WheelEvent<SVGSVGElement>) {
    if (!fullExtent) return
    e.preventDefault()
    const factor = e.deltaY < 0 ? 1.25 : 1 / 1.25
    zoomBy(factor, clientToViewBox(e))
  }

  function onPointerDown(e: React.PointerEvent<SVGSVGElement>) {
    if (e.button !== 0) return
    const vx = clientToViewBox(e)
    const insideChart = vx >= PAD_LEFT && vx <= VIEW_W - PAD_RIGHT
    if (!insideChart) return
    if (selectMode || e.shiftKey) {
      setSelectRect({ x0: vx, x1: vx })
      ;(e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId)
      return
    }
    panRef.current = { lastVX: vx }
    ;(e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId)
  }

  function onPointerMove(e: React.PointerEvent<SVGSVGElement>) {
    const vx = clientToViewBox(e)
    setHoverX(vx)
    if (selectRect) {
      setSelectRect({ x0: selectRect.x0, x1: vx })
      return
    }
    if (panRef.current) {
      const dx = vx - panRef.current.lastVX
      panRef.current.lastVX = vx
      // Convert pixel dx to data dx and pan.
      const dataDx = -(dx / innerW) * (xMax - xMin)
      setRangeClamped(xMin + dataDx, xMax + dataDx)
    }
  }

  function onPointerUp(e: React.PointerEvent<SVGSVGElement>) {
    panRef.current = null
    if (selectRect) {
      const x0 = Math.min(selectRect.x0, selectRect.x1)
      const x1 = Math.max(selectRect.x0, selectRect.x1)
      // Only commit if the user actually dragged some distance.
      if (Math.abs(x1 - x0) > 4) {
        setRangeClamped(inverseX(x0), inverseX(x1))
      }
      setSelectRect(null)
    }
    ;(e.currentTarget as SVGSVGElement).releasePointerCapture(e.pointerId)
  }

  function onPointerLeave() {
    setHoverX(null)
    panRef.current = null
  }

  function onDoubleClick() {
    reset()
  }

  // ----- Crosshair / tooltip data point picking ---------------------------
  const tooltip = useMemo<LossTooltip | null>(() => {
    if (hoverX == null) return null
    if (hoverX < PAD_LEFT || hoverX > VIEW_W - PAD_RIGHT) return null
    const targetStep = inverseX(hoverX)
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
  }, [hoverX, visibleSeries, inverseX, xScale, yScale])

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
                (p) => p.step >= xMin && p.step <= xMax,
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
              .filter((m) => m.step >= xMin && m.step <= xMax)
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
                  .filter((p) => p.step >= xMin && p.step <= xMax)
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
        xMin={xMin}
        xMax={xMax}
        fullXMax={fullExtent?.xMax}
        onToggleSeries={toggleSeries}
        onReset={reset}
      />
    </div>
  )
}
