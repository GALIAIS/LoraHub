import { memo } from "react"
import { Cpu, MemoryStick } from "lucide-react"
import type { SystemSnapshot } from "@/lib/api"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { cn } from "@/lib/utils"
import { UsageBar, fmtBytes, formatFrequency, toneForPercent } from "./_shared"

export const CpuMemoryCard = memo(function CpuMemoryCard({
  snapshot,
  variant = "both",
}: {
  snapshot: SystemSnapshot
  variant?: "both" | "cpu" | "memory"
}) {
  const cpu = snapshot.cpu
  const mem = snapshot.memory
  const memPercent = Math.max(0, Math.min(100, mem.percent))
  const cpuPercent =
    typeof cpu.usage_percent === "number"
      ? Math.max(0, Math.min(100, cpu.usage_percent))
      : null
  const swapPercent =
    mem.swap_total_bytes && typeof mem.swap_used_bytes === "number"
      ? Math.max(
          0,
          Math.min(100, (mem.swap_used_bytes / Math.max(mem.swap_total_bytes, 1)) * 100),
        )
      : null

  // CPU model goes in the description; "current / min-max" frequency, temp,
  // load average follow as space-separated meta. Each piece is optional so
  // older snapshots still render cleanly.
  const coreText = (() => {
    const parts: string[] = [`${cpu.cores_logical} 逻辑核`]
    if (cpu.cores_physical) parts.unshift(`${cpu.cores_physical} 物理核`)
    return parts.join(" / ")
  })()
  const description: string[] = [coreText]
  const freqStr = formatFreqRange(
    cpu.frequency_mhz ?? null,
    cpu.frequency_min_mhz ?? null,
    cpu.frequency_max_mhz ?? null,
  )
  if (freqStr) description.push(freqStr)
  if (typeof cpu.cpu_temperature_c === "number") {
    description.push(`温度 ${cpu.cpu_temperature_c.toFixed(0)}°C`)
  }
  if (cpu.load_average) {
    description.push(`负载 ${cpu.load_average.map((n) => n.toFixed(2)).join(" / ")}`)
  }

  const perCoreFreq = cpu.frequency_per_core_mhz ?? []

  return (
    <div className={cn("grid h-full grid-cols-1 gap-4", variant === "both" && "lg:grid-cols-2")}>
      {variant !== "memory" && (
        <Card className="flex h-full min-h-0 flex-col">
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Cpu className="size-4 text-muted-foreground" />
            CPU
          </CardTitle>
          <CardDescription className="text-xs space-y-0.5">
            {cpu.model ? (
              <div className="font-mono text-foreground/80 truncate" title={cpu.model}>
                {cpu.model}
              </div>
            ) : null}
            <div>{description.join(" · ")}</div>
          </CardDescription>
        </CardHeader>
        <CardContent className="min-h-0 flex-1 space-y-4">
          <UsageBar
            label="总体利用率"
            percent={cpuPercent}
            valueText={typeof cpuPercent === "number" ? `${cpuPercent.toFixed(1)}%` : "—"}
          />
          {cpu.per_core_percent.length > 0 && (
            <details className="group" open={cpu.per_core_percent.length <= 8}>
              <summary className="flex items-center gap-2 cursor-pointer select-none list-none text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
                <span className="size-3 grid place-items-center transition-transform group-open:rotate-90">
                  ›
                </span>
                <span>每核利用率（{cpu.per_core_percent.length} 核）</span>
              </summary>
              <div className="mt-2 max-h-40 overflow-y-auto rounded-[4px] border border-border/40 bg-muted/20 p-2">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  {cpu.per_core_percent.map((p, i) => (
                    <CoreBar
                      key={i}
                      index={i}
                      percent={p}
                      freqMhz={perCoreFreq[i] ?? null}
                    />
                  ))}
                </div>
              </div>
            </details>
          )}
        </CardContent>
        </Card>
      )}

      {variant !== "cpu" && (
        <Card className="h-full">
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <MemoryStick className="size-4 text-muted-foreground" />
            内存
          </CardTitle>
          <CardDescription className="text-xs">
            {fmtBytes(mem.used_bytes)} / {fmtBytes(mem.total_bytes)} · 可用 {fmtBytes(mem.available_bytes)}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <UsageBar
            label="物理内存"
            percent={memPercent}
            valueText={`${memPercent.toFixed(1)}%`}
          />
          {typeof swapPercent === "number" && (
            <UsageBar
              label="交换分区"
              percent={swapPercent}
              valueText={`${fmtBytes(mem.swap_used_bytes ?? 0)} / ${fmtBytes(
                mem.swap_total_bytes ?? 0,
              )}`}
            />
          )}
        </CardContent>
        </Card>
      )}
    </div>
  )
})

const CoreBar = memo(function CoreBar({
  index,
  percent,
  freqMhz,
}: {
  index: number
  percent: number
  freqMhz?: number | null
}) {
  const tone = toneForPercent(percent)
  return (
    <div className="flex items-center gap-2 text-[11px]">
      <span className="w-8 text-muted-foreground tabular-nums">#{index}</span>
      <div className="shiro-progress-track flex-1 h-1.5">
        <div
          className={cn("shiro-progress-fill", tone.bar)}
          style={{ width: `${Math.max(0, Math.min(100, percent))}%` }}
        />
      </div>
      <span className="w-10 text-right font-mono tabular-nums">{percent.toFixed(0)}%</span>
      {typeof freqMhz === "number" && freqMhz > 0 && (
        <span
          className="w-16 text-right font-mono tabular-nums text-muted-foreground/70"
          title="当前频率"
        >
          {formatFrequency(freqMhz)}
        </span>
      )}
    </div>
  )
})

/**
 * Format CPU frequency as "current · min-max GHz" when range is known,
 * else fall back to just the current value. Returns null when there is
 * nothing meaningful to show so the caller can hide the line entirely.
 */
function formatFreqRange(
  current: number | null,
  min: number | null,
  max: number | null,
): string | null {
  const hasCurrent = typeof current === "number" && current > 0
  const hasMin = typeof min === "number" && min > 0
  const hasMax = typeof max === "number" && max > 0
  if (!hasCurrent && !hasMin && !hasMax) return null
  const fmt = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(2)}` : `${v.toFixed(0)}`)
  const unit = (current ?? max ?? min ?? 0) >= 1000 ? "GHz" : "MHz"
  if (hasCurrent && (hasMin || hasMax)) {
    const lo = hasMin ? fmt(min) : "—"
    const hi = hasMax ? fmt(max) : "—"
    return `${fmt(current)} ${unit} · ${lo}-${hi}`
  }
  if (hasCurrent) return `${formatFrequency(current)}`
  // Range only.
  const lo = hasMin ? fmt(min) : "—"
  const hi = hasMax ? fmt(max) : "—"
  return `${lo}-${hi} ${unit}`
}
