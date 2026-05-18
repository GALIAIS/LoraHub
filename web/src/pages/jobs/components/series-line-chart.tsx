/**
 * Single-series line chart used across the analysis + metrics tabs.
 *
 * Rendered as inline SVG with fixed viewBox so it scales cleanly with the
 * card width. Right-side legend block reports the latest / mean / peak /
 * trough values so the user gets numerical context next to the curve.
 *
 * The shape mirrors `ResourceLine` from `analysis-tab.tsx` (which this
 * replaces) but is generic over the X-axis unit so we can use the same
 * primitive for "minutes since start" and "training step".
 */
import type { CSSProperties } from "react"
import { cn } from "@/lib/utils"

export interface SeriesLinePoint {
  x: number
  y: number | null
}

export function SeriesLineChart({
  label,
  unit,
  points,
  color,
  yMax = null,
  yMin = null,
  className,
  hint,
}: {
  label: string
  unit: string
  points: SeriesLinePoint[]
  color: string
  yMax?: number | null
  yMin?: number | null
  className?: string
  hint?: string
}) {
  const valid = points.filter((p): p is { x: number; y: number } =>
    typeof p.y === "number" && Number.isFinite(p.y),
  )
  if (valid.length === 0) {
    return (
      <div className={cn("flex items-center gap-2 text-[11px]", className)}>
        <span className="w-24 shrink-0 text-muted-foreground">{label}</span>
        <span className="text-muted-foreground/60">{hint ?? "未采集到数据"}</span>
      </div>
    )
  }
  const W = 600
  const H = 56
  const xs = valid.map((p) => p.x)
  const ys = valid.map((p) => p.y)
  const xMin = Math.min(...xs)
  const xMax = Math.max(...xs) || 1
  const yLo = yMin != null ? yMin : Math.min(...ys, 0)
  const yHi = yMax != null ? yMax : Math.max(...ys, 1)
  const yRange = yHi - yLo || 1
  const path = valid
    .map((p, i) => {
      const x = ((p.x - xMin) / (xMax - xMin || 1)) * W
      const y = H - ((p.y - yLo) / yRange) * H
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(" ")
  const last = valid[valid.length - 1]
  const min = Math.min(...ys)
  const max = Math.max(...ys)
  const avg = ys.reduce((a, b) => a + b, 0) / ys.length
  return (
    <div className={cn("flex items-center gap-3", className)}>
      <span className="w-24 shrink-0 text-[11px] text-muted-foreground">
        {label}
      </span>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="flex-1 h-[44px]"
        preserveAspectRatio="none"
        style={{ color } as CSSProperties}
      >
        <path d={path} fill="none" stroke="currentColor" strokeWidth={1.5} />
      </svg>
      <span className="w-56 shrink-0 text-right text-[11px] tabular-nums text-muted-foreground">
        当前 {fmt(last.y)}{unit} · 均 {fmt(avg)} · 峰 {fmt(max)} · 谷 {fmt(min)}
      </span>
    </div>
  )
}

function fmt(v: number): string {
  if (!Number.isFinite(v)) return "—"
  if (Math.abs(v) >= 100) return v.toFixed(1)
  if (Math.abs(v) >= 1) return v.toFixed(2)
  if (Math.abs(v) >= 0.01) return v.toFixed(3)
  if (Math.abs(v) === 0) return "0"
  return v.toExponential(2)
}
