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
import { createPortal } from "react-dom"
import { AlertTriangle, Eye, EyeOff, X } from "lucide-react"
import { cn } from "@/lib/utils"
import type { OverfitSignal } from "@/lib/api"
import { downsamplePoints } from "../utils"
import { ChartToolbar } from "./chart-toolbar"

export interface LossSeries {
  id: string
  label: string
  color: string
  // `dashed` is reserved for derived overlays (EMA smoothing, baselines)
  // so users can tell them apart from primary measurements at a glance.
  dashed?: boolean
  points: { step: number; loss: number }[]
}

// Optional vertical guides — used by the metrics tab to mark every
// checkpoint save on the loss chart so the user can correlate
// loss inflections with what artefact was written at that step.
export interface ChartMarker {
  step: number
  label?: string
  color?: string
}

// Optional confidence band drawn behind the primary series. Used by the
// analysis workbench to render a rolling IQR (Q25..Q75) underneath the
// median line so high-variance diffusion losses don't read as a single
// line. The band itself isn't a series — it can't be toggled in the
// legend and doesn't carry tooltip values.
export interface ChartBand {
  id: string
  color: string
  /** Same step axis as the series. lo/hi are absolute loss values. */
  points: { step: number; lo: number; hi: number }[]
}

const VIEW_W = 800
const VIEW_H = 300
const PAD_LEFT = 52
const PAD_RIGHT = 16
const PAD_TOP = 14
const PAD_BOTTOM = 28
const MAX_POINTS = 1500

function formatLoss(v: number): string {
  if (!Number.isFinite(v)) return "—"
  if (Math.abs(v) >= 100) return v.toFixed(1)
  if (Math.abs(v) >= 1) return v.toFixed(3)
  return v.toFixed(4)
}

function trendCopy(trend: OverfitSignal["trend"]): {
  label: string
  tone: "ok" | "muted" | "danger"
} | null {
  switch (trend) {
    case "improving":
      return { label: "持续下降", tone: "ok" }
    case "flat":
      return { label: "已平台", tone: "muted" }
    case "overfitting":
      return { label: "疑似过拟合", tone: "danger" }
    default:
      return null
  }
}

interface LossChartProps {
  series: LossSeries[]
  className?: string
  emptyHint?: string
  overfitSignal?: OverfitSignal | null
  markers?: ChartMarker[]
  /** Optional confidence band(s) drawn behind the primary series. */
  bands?: ChartBand[]
  /**
   * Stable key used to persist the user's view (zoom range, log toggle)
   * across re-renders within a session. Pass the active job id when the
   * chart shows one job's loss; pass `null` to skip persistence.
   */
  persistKey?: string | null
  /** Internally toggled; do not pass from the outside. */
  fullscreen?: boolean
}

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
  const tooltip = useMemo(() => {
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
        {trend && (
          <span
            className={cn(
              "inline-flex items-center gap-1 rounded-[3px] border px-1.5 py-0.5 text-[10.5px] font-medium",
              trend.tone === "danger"
                ? "border-destructive/40 bg-destructive/10 text-destructive"
                : trend.tone === "ok"
                  ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                  : "border-border bg-muted text-muted-foreground",
            )}
          >
            {trend.tone === "danger" && (
              <AlertTriangle className="size-3" aria-hidden />
            )}
            {trend.label}
            {overfitSignal?.gap != null && (
              <span className="ml-1 tabular-nums text-muted-foreground/80">
                Δ{formatLoss(overfitSignal.gap)}
              </span>
            )}
          </span>
        )}
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
                  {Math.round(v)}
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

      {/* Tooltip card */}
      {tooltip && (
        <div className="pointer-events-none absolute right-2 bottom-12 max-w-[260px] rounded-[5px] border border-border/60 bg-background/95 backdrop-blur-sm shadow-[var(--panel-shadow)] px-2.5 py-1.5 text-[11px] tabular-nums">
          <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            step {tooltip.step}
          </div>
          <ul className="mt-0.5 space-y-0.5">
            {tooltip.items.map((it) => (
              <li
                key={it.seriesId}
                className="flex items-center gap-1.5 whitespace-nowrap"
              >
                <span
                  className="inline-block size-2 rounded-full"
                  style={{ background: it.color }}
                />
                <span className="text-foreground/85">{it.label}</span>
                <span className="ml-auto text-foreground/95">
                  {formatLoss(it.loss)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Legend / range chip row */}
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5 px-1 text-[11px]">
        {prepared.map((s) => {
          const off = !!hidden[s.id]
          return (
            <button
              key={s.id}
              type="button"
              onClick={() => toggleSeries(s.id)}
              className={cn(
                "group inline-flex items-center gap-1.5 rounded-[3px] border px-1.5 py-0.5 transition-colors",
                off
                  ? "border-border/40 text-muted-foreground/70"
                  : "border-border/60 bg-muted/40 text-foreground/85",
              )}
              title={off ? "点击显示" : "点击隐藏"}
            >
              {off ? (
                <EyeOff className="size-3" />
              ) : (
                <span
                  className="inline-block h-[2px] w-3 align-middle"
                  style={{ background: s.color }}
                  aria-hidden
                />
              )}
              <span className={cn(off && "line-through")}>{s.label}</span>
            </button>
          )
        })}
        {markers.length > 0 && (
          <span className="text-[10px] text-muted-foreground/70">
            · {markers.length} 个检查点标记
          </span>
        )}
        {zoomedIn && (
          <span className="ml-auto inline-flex items-center gap-1 text-[10.5px] text-muted-foreground">
            <Eye className="size-3" />
            视图 {Math.round(xMin)} – {Math.round(xMax)}
          </span>
        )}
        {zoomedIn && fullExtent && fullExtent.xMax > xMax && (
          <button
            type="button"
            onClick={reset}
            className="inline-flex items-center gap-1 rounded-[3px] border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 text-[10.5px] text-amber-700 dark:text-amber-300 hover:bg-amber-500/20"
            title="新数据已超出视图,点击跟随到最新"
          >
            <span className="size-1.5 rounded-full bg-amber-500 animate-pulse" />
            +{Math.round(fullExtent.xMax - xMax)} 步未显示 · 跟随
          </button>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Fullscreen modal
// ---------------------------------------------------------------------------

function FullscreenModal({
  children,
  onClose,
}: {
  children: React.ReactNode
  onClose: () => void
}) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onClose])

  // Portal to <body> so the modal escapes any ancestor stacking context
  // (parent Cards / Tabs panels often create one via transform / isolate
  // / will-change). Without this a sibling chart Card rendered later in
  // the DOM can paint on top of our "fullscreen" view.
  if (typeof document === "undefined") return null
  return createPortal(
    <div
      className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/60 p-6"
      onClick={onClose}
    >
      <div
        className="relative w-[92vw] max-w-[1400px] rounded-[6px] border border-border/60 bg-background p-4 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          onClick={onClose}
          className="absolute right-3 top-3 z-20 inline-flex size-7 items-center justify-center rounded-[3px] text-muted-foreground hover:bg-muted hover:text-foreground"
          aria-label="关闭全屏"
        >
          <X className="size-4" />
        </button>
        {children}
      </div>
    </div>,
    document.body,
  )
}
