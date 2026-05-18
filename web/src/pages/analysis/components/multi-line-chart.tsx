/**
 * MultiLineChart — compact single-card SVG chart that overlays multiple
 * series on a shared X axis with optional dual Y axis support.
 *
 * Used by the analysis workbench's secondary charts row to fit "LR +
 * iter time + samples/sec" and "GPU util + VRAM% + temperature" into
 * single panels each, instead of stacking several `<SeriesLineChart>`
 * rows. The dual Y axis is automatic — series declare an `axis` of
 * `"left"` or `"right"`, both auto-scale independently.
 */
import { useMemo, useState } from "react"
import type { CSSProperties } from "react"
import { cn } from "@/lib/utils"

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
const VIEW_H = 220
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
  /** Width in pixels of the card container; only affects label spacing. */
  height?: number
}

export function MultiLineChart({
  series,
  xLabel,
  emptyHint = "暂无数据",
  className,
  height = VIEW_H,
}: Props) {
  const [hidden, setHidden] = useState<Record<string, boolean>>({})
  const visible = useMemo(
    () => series.filter((s) => !hidden[s.id]),
    [series, hidden],
  )

  // Per-axis extent
  const stats = useMemo(() => {
    const left: number[] = []
    const right: number[] = []
    const xs: number[] = []
    for (const s of visible) {
      for (const p of s.points) {
        if (p.y == null || !Number.isFinite(p.y)) continue
        xs.push(p.x)
        if (s.axis === "right") right.push(p.y)
        else left.push(p.y)
      }
    }
    return {
      xs,
      left,
      right,
      hasLeft: left.length > 0,
      hasRight: right.length > 0,
    }
  }, [visible])

  const hasData = stats.xs.length > 0
  const xMin = hasData ? Math.min(...stats.xs) : 0
  const xMax = hasData ? Math.max(...stats.xs) : 1
  const xSpan = xMax - xMin || 1

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
  const leftR = axisRange(stats.left)
  const rightR = axisRange(stats.right)

  const innerW = VIEW_W - PAD_LEFT - PAD_RIGHT
  const innerH = height - PAD_TOP - PAD_BOTTOM
  const xScale = (x: number) =>
    PAD_LEFT + ((x - xMin) / xSpan) * innerW
  const yScale = (y: number, axis: "left" | "right") => {
    const r = axis === "right" ? rightR : leftR
    const span = r.hi - r.lo || 1
    return PAD_TOP + (1 - (y - r.lo) / span) * innerH
  }

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

  function toggle(id: string) {
    setHidden((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <svg
        viewBox={`0 0 ${VIEW_W} ${height}`}
        className="block w-full h-auto"
        style={{ color: "var(--foreground)" } as CSSProperties}
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
        {/* Y ticks (left) */}
        {stats.hasLeft &&
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
        {/* Y ticks (right) */}
        {stats.hasRight &&
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
        {/* X ticks */}
        {hasData &&
          xTicks.map((v, i) => {
            const x = xScale(v)
            return (
              <g key={`x${i}`}>
                <line
                  x1={x}
                  x2={x}
                  y1={height - PAD_BOTTOM}
                  y2={height - PAD_BOTTOM + 4}
                  stroke="currentColor"
                  strokeOpacity={0.3}
                />
                <text
                  x={x}
                  y={height - PAD_BOTTOM + 16}
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
        {/* Polylines */}
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
                  .map((p) => `${xScale(p.x)},${yScale(p.y, axis)}`)
                  .join(" ")}
              />
            )
          })
        ) : (
          <text
            x={VIEW_W / 2}
            y={height / 2}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={12}
            fill="currentColor"
            opacity={0.55}
          >
            {emptyHint}
          </text>
        )}
      </svg>

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
              <span
                className="inline-block h-[2px] w-3 align-middle"
                style={{ background: off ? "currentColor" : s.color }}
                aria-hidden
              />
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
        {xLabel && (
          <span className="ml-auto text-[10px] text-muted-foreground/70">
            {xLabel}
          </span>
        )}
      </div>
    </div>
  )
}
