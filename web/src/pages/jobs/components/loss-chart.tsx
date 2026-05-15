import { useMemo, useState } from "react"
import type { CSSProperties } from "react"
import { cn } from "@/lib/utils"
import { downsamplePoints } from "../utils"

export interface LossSeries {
  id: string
  label: string
  color: string
  points: { step: number; loss: number }[]
}

const VIEW_W = 800
const VIEW_H = 280
const PAD_LEFT = 48
const PAD_RIGHT = 16
const PAD_TOP = 16
const PAD_BOTTOM = 28
const MAX_POINTS = 1000

function formatLoss(v: number): string {
  if (!Number.isFinite(v)) return "—"
  if (Math.abs(v) >= 100) return v.toFixed(1)
  if (Math.abs(v) >= 1) return v.toFixed(3)
  return v.toFixed(4)
}

export function LossChart({
  series,
  className,
  emptyHint = "暂无损失数据。",
}: {
  series: LossSeries[]
  className?: string
  emptyHint?: string
}) {
  // Resample each series independently so multi-series compare doesn't blow up
  // for jobs with very long histories.
  const prepared = useMemo(() => {
    return series.map((s) => ({
      ...s,
      points: downsamplePoints(s.points, MAX_POINTS),
    }))
  }, [series])

  const allPoints = prepared.flatMap((s) => s.points)
  const hasData = allPoints.length > 0

  const [stepMin, stepMax, lossMin, lossMax] = useMemo(() => {
    if (!hasData) return [0, 1, 0, 1] as const
    let xMin = Infinity
    let xMax = -Infinity
    let yMin = Infinity
    let yMax = -Infinity
    for (const p of allPoints) {
      if (p.step < xMin) xMin = p.step
      if (p.step > xMax) xMax = p.step
      if (p.loss < yMin) yMin = p.loss
      if (p.loss > yMax) yMax = p.loss
    }
    if (xMin === xMax) xMax = xMin + 1
    if (yMin === yMax) {
      const pad = Math.max(0.001, Math.abs(yMin) * 0.05)
      yMin -= pad
      yMax += pad
    } else {
      const pad = (yMax - yMin) * 0.08
      yMin -= pad
      yMax += pad
    }
    return [xMin, xMax, yMin, yMax] as const
  }, [allPoints, hasData])

  const innerW = VIEW_W - PAD_LEFT - PAD_RIGHT
  const innerH = VIEW_H - PAD_TOP - PAD_BOTTOM

  const xScale = (step: number) =>
    PAD_LEFT + ((step - stepMin) / (stepMax - stepMin || 1)) * innerW
  const yScale = (loss: number) =>
    PAD_TOP + (1 - (loss - lossMin) / (lossMax - lossMin || 1)) * innerH

  const yTicks = useMemo(() => {
    const out: number[] = []
    for (let i = 0; i <= 4; i += 1) {
      out.push(lossMax - ((lossMax - lossMin) * i) / 4)
    }
    return out
  }, [lossMin, lossMax])

  const xTicks = useMemo(() => {
    const out: number[] = []
    for (let i = 0; i <= 4; i += 1) {
      out.push(stepMin + ((stepMax - stepMin) * i) / 4)
    }
    return out
  }, [stepMin, stepMax])

  const [hover, setHover] = useState<
    | null
    | {
        seriesId: string
        label: string
        color: string
        step: number
        loss: number
        cx: number
        cy: number
      }
  >(null)

  function onPointerMove(e: React.PointerEvent<SVGSVGElement>) {
    if (!hasData) return
    const svg = e.currentTarget
    const rect = svg.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * VIEW_W
    if (px < PAD_LEFT || px > VIEW_W - PAD_RIGHT) {
      setHover(null)
      return
    }
    const targetStep =
      stepMin + ((px - PAD_LEFT) / innerW) * (stepMax - stepMin)
    let best:
      | null
      | {
          seriesId: string
          label: string
          color: string
          step: number
          loss: number
          cx: number
          cy: number
          d: number
        } = null
    for (const s of prepared) {
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
      const cx = xScale(cand.step)
      const cy = yScale(cand.loss)
      const d = Math.abs(cx - px)
      if (best === null || d < best.d) {
        best = {
          seriesId: s.id,
          label: s.label,
          color: s.color,
          step: cand.step,
          loss: cand.loss,
          cx,
          cy,
          d,
        }
      }
    }
    setHover(best)
  }

  return (
    <div className={cn("w-full overflow-x-auto", className)}>
      <svg
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        className="block min-w-[640px] w-full h-auto"
        style={{ color: "var(--primary)" } as CSSProperties}
        onPointerMove={onPointerMove}
        onPointerLeave={() => setHover(null)}
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
                strokeOpacity={0.08}
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
        {hasData ? (
          prepared.map((s) =>
            s.points.length === 0 ? null : (
              <polyline
                key={s.id}
                fill="none"
                stroke={s.color}
                strokeWidth={1.5}
                strokeLinejoin="round"
                strokeLinecap="round"
                points={s.points
                  .map((p) => `${xScale(p.step)},${yScale(p.loss)}`)
                  .join(" ")}
              />
            ),
          )
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
        {hover && (
          <g pointerEvents="none">
            <line
              x1={hover.cx}
              x2={hover.cx}
              y1={PAD_TOP}
              y2={VIEW_H - PAD_BOTTOM}
              stroke="currentColor"
              strokeOpacity={0.25}
              strokeDasharray="2 4"
            />
            <circle cx={hover.cx} cy={hover.cy} r={3.5} fill={hover.color} />
          </g>
        )}
      </svg>
      {hover && (
        <div className="mt-1 px-1 text-[11px] text-muted-foreground tabular-nums flex flex-wrap gap-x-4 gap-y-0.5">
          <span>
            <span
              className="inline-block size-2 rounded-full mr-1.5 align-middle"
              style={{ background: hover.color }}
            />
            {hover.label}
          </span>
          <span>步 {hover.step}</span>
          <span>损失 {formatLoss(hover.loss)}</span>
        </div>
      )}
    </div>
  )
}
