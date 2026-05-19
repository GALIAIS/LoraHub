/**
 * MultiLineChart — interactive multi-series SVG chart with optional dual
 * Y axis. Used by the analysis workbench's secondary panels.
 *
 * Features:
 *   - Per-series visibility chips (legend doubles as a control bar).
 *   - Wheel zoom + drag pan + double-click reset on the X axis. Zoom
 *     state survives live data appends (we don't rescale once the user
 *     has narrowed the view).
 *   - Multi-series crosshair tooltip — every visible series reports
 *     its value at the hovered X.
 *   - Floating toolbar (zoom in/out, reset, fullscreen, CSV download).
 *   - Optional sessionStorage persistence keyed by `persistKey`.
 *
 * Heavy lifting matches `<LossChart>` so users only have to learn one
 * gesture vocabulary across the workbench.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import type { CSSProperties } from "react"
import { createPortal } from "react-dom"
import { Eye, EyeOff, X } from "lucide-react"
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

const VIEW_W = 700
const VIEW_H = 230
const PAD_LEFT = 50
const PAD_RIGHT = 50
const PAD_TOP = 12
const PAD_BOTTOM = 28

function fmtNum(v: number): string {
  if (!Number.isFinite(v)) return "—"
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
  /** Stable id for sessionStorage persistence; null disables. */
  persistKey?: string | null
  /** Optional title rendered inside the fullscreen modal. */
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
  persistKey,
  fullscreen,
  onFullscreen,
}: CoreProps) {
  // ----- Per-series visibility -------------------------------------------
  const [hidden, setHidden] = useState<Record<string, boolean>>({})
  useEffect(() => {
    setHidden((prev) => {
      const valid = new Set(series.map((s) => s.id))
      const next: Record<string, boolean> = {}
      for (const k of Object.keys(prev)) if (valid.has(k)) next[k] = prev[k]
      return next
    })
  }, [series])

  const visible = useMemo(
    () => series.filter((s) => !hidden[s.id]),
    [series, hidden],
  )

  // ----- View state with sessionStorage hydration -------------------------
  const storageKey = persistKey ? `lorahub.multi.${persistKey}` : null
  const [viewRange, setViewRange] = useState<[number, number] | null>(() => {
    if (!storageKey) return null
    try {
      const raw = window.sessionStorage.getItem(storageKey)
      if (!raw) return null
      const parsed = JSON.parse(raw)
      if (
        Array.isArray(parsed?.xRange) &&
        parsed.xRange.length === 2 &&
        parsed.xRange.every(Number.isFinite)
      )
        return [parsed.xRange[0], parsed.xRange[1]]
    } catch {
      // ignore corrupt storage
    }
    return null
  })
  useEffect(() => {
    if (!storageKey) return
    try {
      window.sessionStorage.setItem(
        storageKey,
        JSON.stringify({ xRange: viewRange }),
      )
    } catch {
      // quota or disabled — silently skip
    }
  }, [storageKey, viewRange])

  // ----- Extents ----------------------------------------------------------
  const fullExtent = useMemo(() => {
    let xMin = Infinity
    let xMax = -Infinity
    for (const s of visible) {
      for (const p of s.points) {
        if (p.y == null || !Number.isFinite(p.y)) continue
        if (p.x < xMin) xMin = p.x
        if (p.x > xMax) xMax = p.x
      }
    }
    if (!Number.isFinite(xMin) || !Number.isFinite(xMax)) return null
    if (xMin === xMax) xMax = xMin + 1
    return { xMin, xMax }
  }, [visible])

  const effectiveX = viewRange ?? (fullExtent ? [fullExtent.xMin, fullExtent.xMax] : [0, 1])
  const xMin = effectiveX[0]
  const xMax = effectiveX[1]
  const xSpan = xMax - xMin || 1

  const axisStats = useMemo(() => {
    const left: number[] = []
    const right: number[] = []
    for (const s of visible) {
      const axis = s.axis ?? "left"
      const bucket = axis === "right" ? right : left
      for (const p of s.points) {
        if (p.x < xMin || p.x > xMax) continue
        if (p.y == null || !Number.isFinite(p.y)) continue
        bucket.push(p.y)
      }
    }
    return { left, right }
  }, [visible, xMin, xMax])

  function axisRange(arr: number[]): { lo: number; hi: number } {
    if (arr.length === 0) return { lo: 0, hi: 1 }
    let lo = Math.min(...arr)
    let hi = Math.max(...arr)
    if (lo === hi) {
      const pad = Math.max(Math.abs(lo) * 0.05, 0.001)
      lo -= pad
      hi += pad
    } else {
      const pad = (hi - lo) * 0.08
      lo -= pad
      hi += pad
    }
    return { lo, hi }
  }
  const leftR = axisRange(axisStats.left)
  const rightR = axisRange(axisStats.right)
  const hasLeft = axisStats.left.length > 0
  const hasRight = axisStats.right.length > 0

  // ----- Scales -----------------------------------------------------------
  const innerW = VIEW_W - PAD_LEFT - PAD_RIGHT
  const innerH = VIEW_H - PAD_TOP - PAD_BOTTOM

  const xScale = useCallback(
    (x: number) => PAD_LEFT + ((x - xMin) / xSpan) * innerW,
    [xMin, xSpan, innerW],
  )
  const inverseX = useCallback(
    (px: number) => xMin + ((px - PAD_LEFT) / innerW) * xSpan,
    [xMin, xSpan, innerW],
  )
  const yScale = useCallback(
    (y: number, axis: "left" | "right") => {
      const r = axis === "right" ? rightR : leftR
      const span = r.hi - r.lo || 1
      return PAD_TOP + (1 - (y - r.lo) / span) * innerH
    },
    [leftR, rightR, innerH],
  )

  const yTicks = (axis: "left" | "right") => {
    const r = axis === "right" ? rightR : leftR
    const out: number[] = []
    for (let i = 0; i <= 4; i += 1) out.push(r.hi - ((r.hi - r.lo) * i) / 4)
    return out
  }
  const xTicks = useMemo(() => {
    const out: number[] = []
    for (let i = 0; i <= 4; i += 1) out.push(xMin + (xSpan * i) / 4)
    return out
  }, [xMin, xSpan])

  const hasData = !!fullExtent

  // ----- Pointer / gestures ----------------------------------------------
  const svgRef = useRef<SVGSVGElement | null>(null)
  const panRef = useRef<{ lastVX: number } | null>(null)
  const [hoverX, setHoverX] = useState<number | null>(null)

  function clientToVB(e: React.PointerEvent | React.WheelEvent): number {
    const svg = svgRef.current
    if (!svg) return 0
    const rect = svg.getBoundingClientRect()
    return ((e.clientX - rect.left) / rect.width) * VIEW_W
  }

  function setRangeClamped(lo: number, hi: number) {
    if (!fullExtent) return
    const span = hi - lo
    if (span <= 0) return
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
    const anchor = anchorVX != null ? inverseX(anchorVX) : (lo + hi) / 2
    const span = (hi - lo) / factor
    setRangeClamped(
      anchor - span * ((anchor - lo) / (hi - lo || 1)),
      anchor + span * ((hi - anchor) / (hi - lo || 1)),
    )
  }

  function reset() {
    setViewRange(null)
  }

  function onWheel(e: React.WheelEvent<SVGSVGElement>) {
    if (!fullExtent) return
    e.preventDefault()
    const factor = e.deltaY < 0 ? 1.25 : 1 / 1.25
    zoomBy(factor, clientToVB(e))
  }

  function onPointerDown(e: React.PointerEvent<SVGSVGElement>) {
    if (e.button !== 0) return
    const vx = clientToVB(e)
    if (vx < PAD_LEFT || vx > VIEW_W - PAD_RIGHT) return
    panRef.current = { lastVX: vx }
    ;(e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId)
  }
  function onPointerMove(e: React.PointerEvent<SVGSVGElement>) {
    const vx = clientToVB(e)
    setHoverX(vx)
    if (panRef.current) {
      const dx = vx - panRef.current.lastVX
      panRef.current.lastVX = vx
      const dataDx = -(dx / innerW) * xSpan
      setRangeClamped(xMin + dataDx, xMax + dataDx)
    }
  }
  function onPointerUp(e: React.PointerEvent<SVGSVGElement>) {
    panRef.current = null
    ;(e.currentTarget as SVGSVGElement).releasePointerCapture(e.pointerId)
  }
  function onPointerLeave() {
    panRef.current = null
    setHoverX(null)
  }
  function onDoubleClick() {
    reset()
  }

  // ----- Tooltip / crosshair ---------------------------------------------
  const tooltip = useMemo(() => {
    if (hoverX == null || !hasData) return null
    if (hoverX < PAD_LEFT || hoverX > VIEW_W - PAD_RIGHT) return null
    const targetX = inverseX(hoverX)
    const items: Array<{
      seriesId: string
      label: string
      color: string
      unit?: string
      axis: "left" | "right"
      x: number
      y: number
      cy: number
    }> = []
    for (const s of visible) {
      let cand: { x: number; y: number } | null = null
      let candDist = Infinity
      for (const p of s.points) {
        if (p.y == null || !Number.isFinite(p.y)) continue
        const d = Math.abs(p.x - targetX)
        if (d < candDist) {
          candDist = d
          cand = { x: p.x, y: p.y }
        }
      }
      if (!cand) continue
      const axis = s.axis ?? "left"
      items.push({
        seriesId: s.id,
        label: s.label,
        color: s.color,
        unit: s.unit,
        axis,
        x: cand.x,
        y: cand.y,
        cy: yScale(cand.y, axis),
      })
    }
    if (items.length === 0) return null
    const anchorX = xScale(items[0].x)
    return { anchorX, items, x: items[0].x }
  }, [hoverX, visible, hasData, inverseX, xScale, yScale])

  function toggle(id: string) {
    setHidden((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  function downloadCsv() {
    const lines: string[] = ["series,x,y"]
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

  const zoomedIn = viewRange !== null
  const heightStyle = fullscreen ? "h-[60vh]" : "h-auto"

  return (
    <div className={cn("relative flex flex-col gap-2", className)}>
      <div className="absolute right-1 top-1 z-10 flex items-start gap-1.5">
        <ChartToolbar
          zoomedIn={zoomedIn}
          onZoomIn={() => zoomBy(1.5)}
          onZoomOut={() => zoomBy(1 / 1.5)}
          onReset={reset}
          onFullscreen={onFullscreen}
          onDownload={downloadCsv}
        />
      </div>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        className={cn("block w-full select-none", heightStyle)}
        style={{ color: "var(--foreground)" } as CSSProperties}
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
        {hasLeft &&
          yTicks("left").map((v, i) => {
            const y = yScale(v, "left")
            return (
              <g key={`yL${i}`}>
                <line
                  x1={PAD_LEFT}
                  x2={VIEW_W - PAD_RIGHT}
                  y1={y}
                  y2={y}
                  stroke="currentColor"
                  strokeOpacity={0.06}
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
                  {fmtNum(v)}
                </text>
              </g>
            )
          })}
        {hasRight &&
          yTicks("right").map((v, i) => {
            const y = yScale(v, "right")
            return (
              <text
                key={`yR${i}`}
                x={VIEW_W - PAD_RIGHT + 6}
                y={y}
                textAnchor="start"
                dominantBaseline="middle"
                fontSize={10}
                fill="currentColor"
                opacity={0.5}
              >
                {fmtNum(v)}
              </text>
            )
          })}
        {hasData &&
          xTicks.map((v, i) => {
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
        {hasData ? (
          visible.map((s) => {
            const filtered = s.points.filter(
              (p): p is { x: number; y: number } =>
                typeof p.y === "number" && Number.isFinite(p.y),
            )
            if (filtered.length === 0) return null
            const axis = s.axis ?? "left"
            return (
              <polyline
                key={s.id}
                fill="none"
                stroke={s.color}
                strokeWidth={1.5}
                strokeLinejoin="round"
                strokeLinecap="round"
                points={filtered
                  .filter((p) => p.x >= xMin && p.x <= xMax)
                  .map((p) => `${xScale(p.x)},${yScale(p.y, axis)}`)
                  .join(" ")}
              />
            )
          })
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
                r={3}
                fill={it.color}
              />
            ))}
          </g>
        )}
      </svg>

      {tooltip && (
        <div className="pointer-events-none absolute right-2 top-9 max-w-[240px] rounded-[5px] border border-border/60 bg-background/95 backdrop-blur-sm shadow-[var(--panel-shadow)] px-2 py-1 text-[11px] tabular-nums">
          <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
            x = {Math.round(tooltip.x)}
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
                  {fmtNum(it.y)}
                  {it.unit ?? ""}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="flex items-center flex-wrap gap-x-3 gap-y-1 text-[11px]">
        {series.map((s) => {
          const off = !!hidden[s.id]
          const axisHint = s.axis === "right" ? " (右)" : ""
          const visiblePoints = s.points.filter(
            (p) => typeof p.y === "number" && Number.isFinite(p.y),
          ) as Array<{ x: number; y: number }>
          const last =
            visiblePoints.length > 0
              ? visiblePoints[visiblePoints.length - 1]
              : null
          return (
            <button
              key={s.id}
              type="button"
              onClick={() => toggle(s.id)}
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
              <span className={cn(off && "line-through")}>
                {s.label}
                {axisHint}
              </span>
              {!off && last && (
                <span className="text-muted-foreground/70 tabular-nums">
                  {fmtNum(last.y)}
                  {s.unit ?? ""}
                </span>
              )}
            </button>
          )
        })}
        <span className="ml-auto inline-flex items-center gap-2 text-[10px] text-muted-foreground/70">
          {zoomedIn && (
            <span className="inline-flex items-center gap-1">
              <Eye className="size-3" />
              {Math.round(xMin)} – {Math.round(xMax)}
            </span>
          )}
          {zoomedIn && fullExtent && fullExtent.xMax > xMax && (
            <button
              type="button"
              onClick={reset}
              className="inline-flex items-center gap-1 rounded-[3px] border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 text-amber-700 dark:text-amber-300 hover:bg-amber-500/20"
              title="新数据已超出视图,点击跟随到最新"
            >
              <span className="size-1.5 rounded-full bg-amber-500 animate-pulse" />
              +{Math.round(fullExtent.xMax - xMax)} · 跟随
            </button>
          )}
          {xLabel && <span>{xLabel}</span>}
        </span>
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

  // Portal to <body> so the modal isn't trapped in any ancestor's
  // stacking context. Without this, a parent Card with transform /
  // filter / isolate creates a new stacking context, and z-50 only
  // applies *inside* that context — sibling cards rendered later
  // would visually cover the "fullscreen" view.
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
        <div className="flex items-center justify-between mb-2">
          <span className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
            {title ?? "图表"}
          </span>
          <button
            type="button"
            onClick={onClose}
            className="inline-flex size-7 items-center justify-center rounded-[3px] text-muted-foreground hover:bg-muted hover:text-foreground"
            aria-label="关闭全屏"
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
