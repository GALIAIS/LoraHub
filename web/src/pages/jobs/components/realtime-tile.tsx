import type { ReactNode } from "react"
import { useState } from "react"
import { cn } from "@/lib/utils"
import type { SystemGpu } from "@/lib/api"
import { Sparkline } from "./sparkline"
import { fmtBytes } from "../utils"

/**
 * Compact card surface used by the Overview "实时" row. Unlike the global Card
 * we keep these tiles intentionally small (label + headline value + extras)
 * so four can sit side-by-side on a 12-col grid without crowding.
 */
function TileShell({
  label,
  hint,
  state,
  children,
  className,
}: {
  label: string
  hint?: string
  state?: "live" | "idle" | "warn"
  children: ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        "rounded-[6px] border border-border/60 bg-card/70 p-3 flex flex-col gap-2 min-w-0",
        "shadow-[var(--panel-shadow)]",
        className,
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          {state && (
            <span
              className={cn(
                "size-2 rounded-full shrink-0",
                state === "live"
                  ? "bg-emerald-500 animate-pulse"
                  : state === "warn"
                    ? "bg-amber-500"
                    : "bg-muted-foreground/50",
              )}
              aria-hidden="true"
            />
          )}
          <span className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground/80 truncate">
            {label}
          </span>
        </div>
        {hint && (
          <span className="text-[10px] text-muted-foreground/70 tabular-nums shrink-0">
            {hint}
          </span>
        )}
      </div>
      <div className="flex-1 min-w-0">{children}</div>
    </div>
  )
}

function Bar({
  percent,
  toneClass,
}: {
  percent: number
  toneClass: string
}) {
  const v = Math.max(0, Math.min(100, percent))
  return (
    <div className="h-1.5 w-full bg-muted/50 rounded-full overflow-hidden">
      <div
        className={cn("h-full rounded-full transition-[width]", toneClass)}
        style={{ width: `${v.toFixed(1)}%` }}
      />
    </div>
  )
}

function toneFor(percent: number | null): string {
  if (percent === null || !Number.isFinite(percent)) return "bg-muted-foreground/40"
  if (percent >= 90) return "bg-red-500"
  if (percent >= 70) return "bg-amber-500"
  if (percent >= 40) return "bg-blue-500"
  return "bg-emerald-500"
}

// =================================================== GPU tile ===============

export function GpuLiveTile({
  gpu,
  active,
}: {
  gpu: SystemGpu | null | undefined
  active: boolean
}) {
  if (!gpu) {
    return (
      <TileShell label="当前 GPU" state="idle" hint="未检测">
        <div className="text-xs text-muted-foreground/80">
          未检测到 GPU 设备。
        </div>
      </TileShell>
    )
  }
  const memPercent =
    typeof gpu.memory_used_bytes === "number" && gpu.memory_total_bytes
      ? Math.max(
          0,
          Math.min(100, (gpu.memory_used_bytes / gpu.memory_total_bytes) * 100),
        )
      : null
  const utilPercent =
    typeof gpu.utilization_percent === "number" ? gpu.utilization_percent : null
  return (
    <TileShell
      label={`当前 GPU · #${gpu.index}`}
      state={active ? "live" : "idle"}
      hint={gpu.name}
    >
      <div className="space-y-2">
        <div className="space-y-1">
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-muted-foreground/80">显存</span>
            <span className="tabular-nums text-foreground/90">
              {typeof gpu.memory_used_bytes === "number" && gpu.memory_total_bytes
                ? `${fmtBytes(gpu.memory_used_bytes)} / ${fmtBytes(gpu.memory_total_bytes)}`
                : "—"}
            </span>
          </div>
          <Bar percent={memPercent ?? 0} toneClass={toneFor(memPercent)} />
        </div>
        <div className="space-y-1">
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-muted-foreground/80">计算</span>
            <span className="tabular-nums text-foreground/90">
              {utilPercent === null ? "—" : `${utilPercent.toFixed(0)}%`}
            </span>
          </div>
          <Bar percent={utilPercent ?? 0} toneClass={toneFor(utilPercent)} />
        </div>
        <div className="flex items-center gap-1.5 flex-wrap text-[10px]">
          {typeof gpu.temperature_c === "number" && (
            <span className="rounded-[2px] border border-border/60 px-1.5 py-0.5 text-muted-foreground tabular-nums">
              {gpu.temperature_c.toFixed(0)}°C
            </span>
          )}
          {typeof gpu.power_w === "number" && (
            <span className="rounded-[2px] border border-border/60 px-1.5 py-0.5 text-muted-foreground tabular-nums">
              {gpu.power_w.toFixed(0)} W
              {typeof gpu.power_limit_w === "number"
                ? ` / ${gpu.power_limit_w.toFixed(0)}`
                : ""}
            </span>
          )}
          {typeof gpu.fan_percent === "number" && (
            <span className="rounded-[2px] border border-border/60 px-1.5 py-0.5 text-muted-foreground tabular-nums">
              风扇 {gpu.fan_percent.toFixed(0)}%
            </span>
          )}
        </div>
      </div>
    </TileShell>
  )
}

// =================================================== Throughput tile ========

type ThroughputUnit = "it_per_sec" | "s_per_step"

/** Format an it/s rate in the chosen display unit. The two are
 * mathematical inverses (s/step = 1 / it/s); we show one at a time
 * and let the user click the headline to flip — keeps the tile
 * compact while still surfacing both perspectives on demand.
 */
function _formatThroughput(v: number | null, unit: ThroughputUnit): string {
  if (v === null || !Number.isFinite(v) || v <= 0) return "—"
  if (unit === "it_per_sec") return `${v.toFixed(2)} it/s`
  return `${(1 / v).toFixed(2)} s/step`
}

// Two-phase fade-swap for unit toggling. We keep the value visually
// looking like plain text — no border / no chip — but on click we:
//   1. fade-out + slide up the current value (160ms)
//   2. swap the unit + reset transform
//   3. fade-in + slide down to original position (160ms via CSS transition)
// Total ~320ms; long enough to feel intentional, short enough not to
// block reading the new number.
const _FADE_MS = 160

interface ThroughputClickableProps {
  itPerSec: number | null
  className?: string
}

function _ThroughputClickable({
  itPerSec,
  className,
}: ThroughputClickableProps) {
  const [unit, setUnit] = useState<ThroughputUnit>("it_per_sec")
  const [fading, setFading] = useState(false)
  const interactive =
    itPerSec !== null && Number.isFinite(itPerSec) && itPerSec > 0

  function toggle() {
    if (!interactive || fading) return
    setFading(true)
    window.setTimeout(() => {
      setUnit((u) => (u === "it_per_sec" ? "s_per_step" : "it_per_sec"))
      setFading(false)
    }, _FADE_MS)
  }

  return (
    <button
      type="button"
      onClick={toggle}
      // Strip every default button look — preflight already drops
      // border / bg, but a few user agents still paint focus rings
      // that read as "this is a button". We want a plain text feel
      // with the cursor as the only affordance.
      className={cn(
        "appearance-none bg-transparent p-0 m-0 border-0 outline-none",
        "tabular-nums truncate text-left",
        "transition duration-[160ms] ease-out",
        interactive
          ? "cursor-pointer hover:text-primary focus-visible:text-primary"
          : "cursor-default",
        fading && "opacity-0 -translate-y-0.5",
        className,
      )}
      title={interactive ? "点击切换吞吐单位" : undefined}
      aria-label={
        interactive
          ? `吞吐 ${_formatThroughput(itPerSec, unit)}，点击切换单位`
          : undefined
      }
    >
      {_formatThroughput(itPerSec, unit)}
    </button>
  )
}

/**
 * Click-to-toggle throughput display with a fade-swap animation.
 * Looks like plain text; click anywhere on the value flips between
 * ``X.XX it/s`` and ``Y.YY s/step``. Used by the post-run "最终统计"
 * Stat. The runtime tile uses the same animation directly inline so
 * the avg-hint can stay in sync.
 */
export function ThroughputSwitcher({
  itPerSec,
  className,
}: {
  itPerSec: number | null
  className?: string
}) {
  return <_ThroughputClickable itPerSec={itPerSec} className={className} />
}

export function ThroughputTile({
  itPerSecRecent,
  itPerSecAvg,
  history,
}: {
  itPerSecRecent: number | null
  itPerSecAvg: number | null
  history: number[]
}) {
  const [unit, setUnit] = useState<ThroughputUnit>("it_per_sec")
  const [fading, setFading] = useState(false)
  const interactive =
    itPerSecRecent !== null &&
    Number.isFinite(itPerSecRecent) &&
    itPerSecRecent > 0

  function toggle() {
    if (!interactive || fading) return
    setFading(true)
    window.setTimeout(() => {
      setUnit((u) => (u === "it_per_sec" ? "s_per_step" : "it_per_sec"))
      setFading(false)
    }, _FADE_MS)
  }

  return (
    <TileShell
      label="训练吞吐"
      state={itPerSecRecent === null ? "idle" : "live"}
      hint={
        itPerSecAvg !== null && Number.isFinite(itPerSecAvg) && itPerSecAvg > 0
          ? `均值 ${_formatThroughput(itPerSecAvg, unit)}`
          : undefined
      }
    >
      <div className="flex items-end justify-between gap-3">
        <button
          type="button"
          onClick={toggle}
          className={cn(
            "appearance-none bg-transparent p-0 m-0 border-0 outline-none",
            "text-base font-semibold tabular-nums truncate text-left",
            "transition duration-[160ms] ease-out",
            interactive
              ? "cursor-pointer hover:text-primary focus-visible:text-primary"
              : "cursor-default",
            fading && "opacity-0 -translate-y-0.5",
          )}
          title={interactive ? "点击切换吞吐单位" : undefined}
          aria-label={
            interactive
              ? `吞吐 ${_formatThroughput(itPerSecRecent, unit)}，点击切换单位`
              : undefined
          }
        >
          {_formatThroughput(itPerSecRecent, unit)}
        </button>
        <Sparkline
          values={history}
          width={120}
          height={36}
          stroke="var(--chart-1)"
          fill="color-mix(in oklch, var(--chart-1) 22%, transparent)"
          emptyHint="等待更多步…"
        />
      </div>
      <div className="mt-1 text-[10px] text-muted-foreground/70">
        最近 {history.length} 步窗口
      </div>
    </TileShell>
  )
}

// =================================================== ETA tile ===============

export function EtaTile({
  etaSeconds,
  step,
  totalSteps,
}: {
  etaSeconds: number | null
  step: number | null
  totalSteps: number | null
}) {
  const pct =
    typeof step === "number" && typeof totalSteps === "number" && totalSteps > 0
      ? Math.max(0, Math.min(100, (step / totalSteps) * 100))
      : null
  const headline = formatEta(etaSeconds)
  const stepLine =
    step !== null && totalSteps !== null
      ? `${step} / ${totalSteps}`
      : step !== null
        ? `${step} 步`
        : "—"
  return (
    <TileShell
      label="预计剩余"
      state={etaSeconds === null ? "idle" : "live"}
      hint={pct !== null ? `${pct.toFixed(0)}%` : undefined}
    >
      <div className="text-xl font-semibold tabular-nums truncate">
        {headline}
      </div>
      <div className="mt-1 text-[11px] text-muted-foreground/80 tabular-nums">
        {stepLine}
      </div>
      {pct !== null && (
        <div className="mt-2">
          <Bar percent={pct} toneClass="bg-primary/80" />
        </div>
      )}
    </TileShell>
  )
}

function formatEta(secs: number | null): string {
  if (secs === null || !Number.isFinite(secs) || secs < 0) return "—"
  const total = Math.round(secs)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  if (h > 0) return `${h} 时 ${m} 分`
  if (m > 0) return `${m} 分 ${s} 秒`
  return `${s} 秒`
}

// =================================================== Loss trend tile ========

export function LossTrendTile({
  history,
  latest,
}: {
  history: number[]
  latest: number | null
}) {
  const headline = latest === null ? "—" : latest.toFixed(4)
  const delta =
    history.length >= 2
      ? history[history.length - 1] - history[0]
      : null
  return (
    <TileShell
      label="损失趋势"
      state={history.length === 0 ? "idle" : "live"}
      hint={
        delta !== null && Number.isFinite(delta)
          ? `${delta >= 0 ? "+" : ""}${delta.toFixed(4)}`
          : undefined
      }
    >
      <div className="flex items-end justify-between gap-3">
        <div className="text-xl font-semibold tabular-nums truncate">
          {headline}
        </div>
        <Sparkline
          values={history}
          width={120}
          height={36}
          stroke="var(--chart-2)"
          fill="color-mix(in oklch, var(--chart-2) 22%, transparent)"
          emptyHint="等待损失数据…"
        />
      </div>
      <div className="mt-1 text-[10px] text-muted-foreground/70">
        最近 {history.length} 步窗口
      </div>
    </TileShell>
  )
}
