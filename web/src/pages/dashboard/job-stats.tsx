import { memo } from "react"
import type React from "react"
import { CheckCircle2, CircleX, Loader2, Pause } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"

export const JobStatGrid = memo(function JobStatGrid({
  stats,
}: {
  stats: { running: number; queued: number; succeeded: number; failed: number }
}) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
      <StatCard
        icon={<Loader2 className="size-3.5" />}
        label="运行中"
        value={String(stats.running)}
        tone="primary"
      />
      <StatCard
        icon={<Pause className="size-3.5" />}
        label="排队中"
        value={String(stats.queued)}
      />
      <StatCard
        icon={<CheckCircle2 className="size-3.5" />}
        label="已完成"
        value={String(stats.succeeded)}
      />
      <StatCard
        icon={<CircleX className="size-3.5" />}
        label="失败 / 中断"
        value={String(stats.failed)}
        tone={stats.failed > 0 ? "warning" : "default"}
      />
    </div>
  )
})

function StatCard({
  icon,
  label,
  value,
  tone = "default",
}: {
  icon: React.ReactNode
  label: string
  value: string
  tone?: "default" | "primary" | "destructive" | "warning"
}) {
  const toneStyle = {
    default: "text-foreground",
    primary: "text-primary",
    destructive: "text-destructive",
    warning: "text-amber-700 dark:text-amber-400",
  }[tone]
  return (
    <Card>
      <CardContent className="px-4 py-3">
        <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
          {icon}
          {label}
        </div>
        <div className={cn("mt-1.5 text-2xl font-semibold tracking-tight tabular-nums", toneStyle)}>
          {value}
        </div>
      </CardContent>
    </Card>
  )
}
