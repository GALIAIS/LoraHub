/**
 * 仪表盘子组件共用的工具函数与小组件。
 *
 * 跨多个 section 复用的内容才会落到这里；只在单个 section 用到的助手保留在
 * 那个文件里，避免过度集中。
 */
import { memo } from "react"
import { Progress, ProgressTrack, ProgressIndicator } from "@/components/ui/progress"
import { cn } from "@/lib/utils"

export function toneForPercent(percent: number | null): { text: string; bar: string } {
  if (percent === null) return { text: "text-muted-foreground", bar: "bg-muted-foreground/40" }
  if (percent >= 90) return { text: "text-destructive", bar: "bg-destructive" }
  if (percent >= 70) {
    return {
      text: "text-amber-700 dark:text-amber-400",
      bar: "bg-amber-500",
    }
  }
  if (percent >= 40) return { text: "text-primary", bar: "bg-primary" }
  return {
    text: "text-emerald-700 dark:text-emerald-400",
    bar: "bg-emerald-500",
  }
}

export function fmtBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0 B"
  const units = ["B", "KB", "MB", "GB", "TB", "PB"]
  let i = 0
  let v = n
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i += 1
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : v >= 10 ? 1 : 2)} ${units[i]}`
}

export function formatRate(bps: number): string {
  // Network/disk per-second rate. fmtBytes already does adaptive units.
  if (!Number.isFinite(bps) || bps <= 0) return "0 B/s"
  return `${fmtBytes(bps)}/s`
}

export function formatFrequency(mhz: number): string {
  if (!Number.isFinite(mhz) || mhz <= 0) return "—"
  if (mhz >= 1000) return `${(mhz / 1000).toFixed(2)} GHz`
  return `${mhz.toFixed(0)} MHz`
}

export const UsageBar = memo(function UsageBar({
  label,
  percent,
  valueText,
}: {
  label: string
  percent: number | null
  valueText: string
}) {
  const tone = toneForPercent(percent)
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="text-muted-foreground truncate">{label}</span>
        <span
          className={cn(
            "font-mono tabular-nums shrink-0 text-right min-w-[6ch]",
            tone.text,
          )}
        >
          {valueText}
        </span>
      </div>
      <Progress value={percent ?? 0}>
        <ProgressTrack>
          <ProgressIndicator className={tone.bar} />
        </ProgressTrack>
      </Progress>
    </div>
  )
})

export const Metric = memo(function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">{label}</dt>
      <dd className="mt-0.5 font-mono tabular-nums">{value}</dd>
    </div>
  )
})
