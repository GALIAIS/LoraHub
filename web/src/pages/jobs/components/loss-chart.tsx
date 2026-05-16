import { useMemo, useState } from "react"
import type { CSSProperties } from "react"
import { AlertTriangle } from "lucide-react"
import { cn } from "@/lib/utils"
import type { OverfitSignal } from "@/lib/api"
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

export function LossChart({
  series,
  className,
  emptyHint = "暂无损失数据。",
  overfitSignal,
}: {
  series: LossSeries[]
  className?: string
  emptyHint?: string
  overfitSignal?: OverfitSignal | null
}) {
  // Resample each series independently so multi-series compare doesn't blow up
  // for jobs with very long histories.
  const prepared = useMemo(() => {
    return series.map((s) => ({
      ...s,
      points: downsamplePoints(s.points, MAX_POINTS),
    }))
  }, [series])

  // The first two prepared series (when both train + val are present) are
  // shaded with a thin amber band between them so the gap is visible at a
  // glance. We assume the convention from `metrics-tab.tsx`: train is index 0,
  // validation is index 1. If only one series is provided we just skip the
  // band — nothing to compare against.
  const gapBand = useMemo(() => {
    if (prepared.length < 2) return null
    const train = prepared[0]
    const val = prepared[1]
    if (!train.points.length || !val.points.length) return null
    return val.points.map((vp) => {
      // Closest train point by x — train series is much denser than val so a
      // simple linear scan stays cheap and avoids extrapolation surprises.
      let bestStep = train.points[0].step
      let bestLoss = train.points[0].loss
      let bestDist = Math.abs(bestStep - vp.step)
      for (const tp of train.points) {
        const d = Math.abs(tp.step - vp.step)
        if (d < bestDist) {
          bestDist = d
          bestStep = tp.step
          bestLoss = tp.loss
        }
      }
      return { step: vp.step, train: bestLoss, val: vp.loss, _trainStep: bestStep }
    })
  }, [prepared])

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

  const trend = overfitSignal ? trendCopy(overfitSignal.trend) : null

  return (
    <div className={cn("w-full overflow-x-auto", className)}>
      {trend ? (
        <div className="mb-2 flex items-center gap-2 text-[11px]">
          <span
            className={cn(
              "inline-flex items-center gap-1 rounded-[3px] border px-1.5 py-0.5 font-medium",
              trend.tone === "danger"
                ? "border-destructive/40 bg-destructive/10 text-destructive"
                : trend.tone === "ok"
                  ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                  : "border-border bg-muted text-muted-foreground",
            )}
          >
            {trend.tone === "danger" ? (
              <AlertTriangle className="size-3" aria-hidden />
            ) : null}
            {trend.label}
          </span>
          {overfitSignal?.gap !== null && overfitSignal?.gap !== undefined ? (
            <span className="text-muted-foreground/70 tabular-nums">
              gap {formatLoss(overfitSignal.gap)}
            </span>
          ) : null}
        </div>
      ) : null}
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
        {hasData && gapBand && gapBand.length >= 2 ? (
          <polygon
            points={[
              ...gapBand.map(
                (g) => `${xScale(g.step)},${yScale(g.val)}`,
              ),
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
        ) : null}
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
