import { useState } from "react"
import type { TrainingEvent } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { EventOperations } from "./event-operations"
import { TerminalLog } from "./terminal-log"

type SubView = "operations" | "raw"

export function EventsTab({
  events,
  status,
  historyStatus = "idle",
  jobId,
  fallbackTotalSteps = null,
}: {
  events: TrainingEvent[]
  status: "idle" | "open" | "closed"
  historyStatus?: "idle" | "loading" | "ready" | "error"
  jobId: string | null
  fallbackTotalSteps?: number | null
}) {
  const [sub, setSub] = useState<SubView>("raw")
  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <div className="mb-2 flex shrink-0 items-center justify-between gap-2">
        <div className="flex items-center gap-3 min-w-0">
          <div className="inline-flex rounded-[4px] border border-border/60 bg-background/60 p-[2px]">
            {(
              [
                { value: "raw", label: "原始日志" },
                { value: "operations", label: "诊断事件" },
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
        <span className="shrink-0 text-[11px] text-muted-foreground">
          {status === "open"
            ? "已连接"
            : status === "closed"
              ? "已断开"
              : "等待中"}
        </span>
      </div>
      <div className="min-h-0 flex-1">
        {sub === "raw" ? (
          <TerminalLog events={events} fallbackTotalSteps={fallbackTotalSteps} />
        ) : (
          <div className="flex h-full min-h-0 flex-col gap-2">
            <div className="flex shrink-0 items-center gap-2 rounded-[6px] border border-border/60 bg-muted/25 px-3 py-2 text-xs text-muted-foreground">
              <span className="flex-1">
                诊断事件用于排错，默认请看原始日志。
              </span>
              <Button
                type="button"
                size="sm"
                variant="outline"
                className="h-7 px-2 text-[11px]"
                onClick={() => setSub("raw")}
              >
                返回日志
              </Button>
            </div>
            <EventOperations
              events={events}
              status={status}
              historyStatus={historyStatus}
              jobId={jobId}
              fallbackTotalSteps={fallbackTotalSteps}
            />
          </div>
        )}
      </div>
    </div>
  )
}
