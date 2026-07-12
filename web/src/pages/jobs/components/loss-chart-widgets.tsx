import { AlertTriangle, Eye, EyeOff } from "lucide-react"

import { cn } from "@/lib/utils"
import {
  formatLoss,
  type ChartBand,
  type LossSeries,
} from "./loss-chart-model"

type TrendTone = "ok" | "muted" | "danger"

export function TrendBadge({
  trend,
  gap,
}: {
  trend: { label: string; tone: TrendTone } | null
  gap?: number | null
}) {
  if (!trend) return null
  return (
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
      {gap != null && (
        <span className="ml-1 tabular-nums text-muted-foreground/80">
          Δ{formatLoss(gap)}
        </span>
      )}
    </span>
  )
}

export interface LossTooltip {
  step: number
  items: Array<{
    seriesId: string
    label: string
    color: string
    step: number
    loss: number
    cy: number
  }>
  anchorX: number
}

export function TooltipCard({ tooltip }: { tooltip: LossTooltip | null }) {
  if (!tooltip) return null
  return (
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
  )
}

export function LegendRow({
  bands,
  series,
  hidden,
  markersCount,
  xLabel,
  zoomedIn,
  xMin,
  xMax,
  fullXMax,
  onToggleSeries,
  onReset,
}: {
  bands: ChartBand[]
  series: LossSeries[]
  hidden: Record<string, boolean>
  markersCount: number
  xLabel?: string
  zoomedIn: boolean
  xMin: number
  xMax: number
  fullXMax?: number
  onToggleSeries: (id: string) => void
  onReset: () => void
}) {
  return (
    <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5 px-1 text-[11px]">
      {series.map((s) => {
        const off = !!hidden[s.id]
        return (
          <button
            key={s.id}
            type="button"
            onClick={() => onToggleSeries(s.id)}
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
                className="inline-block w-3 border-t-2 align-middle"
                style={{
                  borderColor: s.color,
                  borderTopStyle: s.dashed ? "dashed" : "solid",
                }}
                aria-hidden
              />
            )}
            <span className={cn(off && "line-through")}>{s.label}</span>
          </button>
        )
      })}
      {bands.map((band) =>
        band.label ? (
          <span
            className="inline-flex items-center gap-1.5 text-muted-foreground"
            key={band.id}
          >
            <span
              aria-hidden
              className="h-2 w-3 rounded-[2px] border border-border/50"
              style={{ background: band.color }}
            />
            {band.label}
          </span>
        ) : null,
      )}
      {markersCount > 0 && (
        <span className="text-[10px] text-muted-foreground/70">
          · {markersCount} 个检查点标记
        </span>
      )}
      {xLabel && (
        <span className="text-[10px] text-muted-foreground/70">
          · X: {xLabel}
        </span>
      )}
      {zoomedIn && (
        <span className="ml-auto inline-flex items-center gap-1 text-[10.5px] text-muted-foreground">
          <Eye className="size-3" />
          视图 {Math.round(xMin)} – {Math.round(xMax)}
        </span>
      )}
      {zoomedIn && fullXMax != null && fullXMax > xMax && (
        <button
          type="button"
          onClick={onReset}
          className="inline-flex items-center gap-1 rounded-[3px] border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 text-[10.5px] text-amber-700 dark:text-amber-300 hover:bg-amber-500/20"
          title="新数据已超出视图，点击跟随到最新"
        >
          <span className="size-1.5 rounded-full bg-amber-500 animate-pulse" />
          +{Math.round(fullXMax - xMax)} 步未显示 · 跟随
        </button>
      )}
    </div>
  )
}
