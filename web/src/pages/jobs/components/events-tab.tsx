import { useState } from "react"
import type { TrainingEvent } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import { EventTimeline } from "./event-timeline"
import { TerminalLog } from "./terminal-log"

type SubView = "timeline" | "raw"

export function EventsTab({
  events,
  status,
  jobId,
  fallbackTotalSteps = null,
}: {
  events: TrainingEvent[]
  status: "idle" | "open" | "closed"
  jobId: string | null
  fallbackTotalSteps?: number | null
}) {
  const [sub, setSub] = useState<SubView>("raw")
  return (
    <Card className="overflow-hidden flex flex-col h-full min-h-0">
      <CardHeader className="py-3 px-4 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
        <div className="flex items-center gap-3 min-w-0">
          <CardTitle className="text-xs uppercase tracking-[0.18em] text-muted-foreground shrink-0">
            事件流
          </CardTitle>
          <div className="inline-flex rounded-[4px] border border-border/60 bg-background/60 p-[2px]">
            {(
              [
                { value: "raw", label: "原始日志" },
                { value: "timeline", label: "时间轴" },
              ] as { value: SubView; label: string }[]
            ).map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() => setSub(opt.value)}
                className={cn(
                  "h-6 px-2.5 text-[11px] rounded-[3px] transition-colors",
                  sub === opt.value
                    ? "bg-primary/15 text-foreground border border-primary/30"
                    : "text-muted-foreground hover:text-foreground border border-transparent",
                )}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
        <span className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
          WS{" "}
          {status === "open"
            ? "已连接"
            : status === "closed"
              ? "已断开"
              : "等待中"}
        </span>
      </CardHeader>
      <CardContent className="p-0 flex-1 min-h-0 flex flex-col">
        {sub === "timeline" ? (
          <div className="flex-1 min-h-0 p-3">
            <EventTimeline
              events={events}
              jobId={jobId}
              fallbackTotalSteps={fallbackTotalSteps}
            />
          </div>
        ) : (
          <div className="flex-1 min-h-0 flex flex-col p-3">
            <TerminalLog events={events} fallbackTotalSteps={fallbackTotalSteps} />
          </div>
        )}
      </CardContent>
    </Card>
  )
}
