import { memo } from "react"
import { BatteryCharging, BatteryFull } from "lucide-react"
import type { SystemBattery } from "@/lib/api"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { cn } from "@/lib/utils"

export const BatteryCard = memo(function BatteryCard({ battery }: { battery: SystemBattery }) {
  const percent = Math.max(0, Math.min(100, battery.percent))
  const Icon = battery.plugged ? BatteryCharging : BatteryFull
  // Battery uses inverted tone semantics: 满 = 绿 / 中 = 黄 / 低 = 红.
  // (Don't reuse toneForPercent — that one tints "high" red because it's
  // built for utilisation metrics where high is bad.)
  const batteryTone = batteryToneForPercent(percent)
  const tone = battery.plugged
    ? "text-emerald-700 dark:text-emerald-400"
    : batteryTone.text
  const description = (() => {
    if (battery.plugged) return "电源已连接"
    if (typeof battery.secs_left === "number") return `预计剩余 ${formatSecs(battery.secs_left)}`
    return "未连接电源"
  })()
  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="text-base flex items-center gap-2">
              <Icon className={cn("size-4", tone)} />
              电池
            </CardTitle>
            <CardDescription className="text-xs">{description}</CardDescription>
          </div>
          <div className={cn("text-2xl font-semibold tabular-nums", tone)}>
            {percent.toFixed(0)}%
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="shiro-progress-track h-1.5">
          <div
            className={cn("shiro-progress-fill", batteryTone.bar)}
            style={{ width: `${percent}%` }}
          />
        </div>
      </CardContent>
    </Card>
  )
})

/**
 * Battery-flavoured percent → tone: full reads green, mid amber, low red.
 * Thresholds match the macOS / Windows convention (≤ 20% low warning).
 */
function batteryToneForPercent(percent: number): { text: string; bar: string } {
  if (percent <= 20) {
    return { text: "text-destructive", bar: "bg-destructive" }
  }
  if (percent <= 50) {
    return {
      text: "text-amber-700 dark:text-amber-400",
      bar: "bg-amber-500",
    }
  }
  return {
    text: "text-emerald-700 dark:text-emerald-400",
    bar: "bg-emerald-500",
  }
}

function formatSecs(secs: number): string {
  if (!Number.isFinite(secs) || secs <= 0) return "—"
  const hours = Math.floor(secs / 3600)
  const minutes = Math.floor((secs % 3600) / 60)
  if (hours > 0) return `${hours} 小时 ${minutes} 分钟`
  return `${minutes} 分钟`
}
