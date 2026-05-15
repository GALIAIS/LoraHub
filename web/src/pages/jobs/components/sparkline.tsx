import { useMemo } from "react"
import { cn } from "@/lib/utils"

/**
 * Tiny inline SVG sparkline. Designed for the Overview "实时" tiles where the
 * surrounding card already supplies a label and headline value, so the chart
 * itself is intentionally chrome-less: no axes, no ticks, just the trace plus
 * an optional area fill underneath.
 */
export function Sparkline({
  values,
  width = 160,
  height = 40,
  stroke = "var(--primary)",
  fill,
  strokeWidth = 1.5,
  className,
  emptyHint = "—",
}: {
  values: number[]
  width?: number
  height?: number
  stroke?: string
  fill?: string
  strokeWidth?: number
  className?: string
  emptyHint?: string
}) {
  const finite = useMemo(
    () => values.filter((v) => Number.isFinite(v)),
    [values],
  )

  const path = useMemo(() => {
    if (finite.length === 0) return null
    let lo = Infinity
    let hi = -Infinity
    for (const v of finite) {
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null
    if (lo === hi) {
      const pad = Math.max(0.001, Math.abs(lo) * 0.05)
      lo -= pad
      hi += pad
    }
    const padY = 2
    const innerH = height - padY * 2
    const stepX = finite.length === 1 ? 0 : width / (finite.length - 1)
    const pts: { x: number; y: number }[] = finite.map((v, i) => {
      const t = (v - lo) / (hi - lo || 1)
      return { x: i * stepX, y: padY + (1 - t) * innerH }
    })
    const linePath = pts
      .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(2)},${p.y.toFixed(2)}`)
      .join(" ")
    const areaPath = `${linePath} L${pts[pts.length - 1].x.toFixed(2)},${height} L0,${height} Z`
    return { line: linePath, area: areaPath, last: pts[pts.length - 1] }
  }, [finite, width, height])

  if (!path) {
    return (
      <div
        className={cn(
          "flex items-center justify-center text-[10px] text-muted-foreground/70",
          className,
        )}
        style={{ width, height }}
      >
        {emptyHint}
      </div>
    )
  }

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
      className={cn("block", className)}
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      {fill && <path d={path.area} fill={fill} stroke="none" />}
      <path
        d={path.line}
        fill="none"
        stroke={stroke}
        strokeWidth={strokeWidth}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <circle cx={path.last.x} cy={path.last.y} r={1.8} fill={stroke} />
    </svg>
  )
}
