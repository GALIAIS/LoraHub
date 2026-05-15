import type { TrainingEvent } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { EventRow } from "./event-row"

export function EventsTab({
  events,
  status,
}: {
  events: TrainingEvent[]
  status: "idle" | "open" | "closed"
}) {
  return (
    <Card className="rounded-[6px] border-border/60 shadow-[var(--panel-shadow)] overflow-hidden flex flex-col h-full min-h-0">
      <CardHeader className="py-3 px-4 border-b border-border/60 bg-muted/40 flex-row items-center justify-between gap-2">
        <CardTitle className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
          事件流
        </CardTitle>
        <span className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
          WS{" "}
          {status === "open"
            ? "已连接"
            : status === "closed"
              ? "已断开"
              : "等待中"}
        </span>
      </CardHeader>
      <CardContent className="p-0 flex-1 min-h-0">
        <ScrollArea className="h-full">
          <ul className="font-mono text-[12px] divide-y divide-border/30">
            {events.length === 0 && (
              <li className="px-4 py-6 text-muted-foreground text-center">
                正在等待事件…
              </li>
            )}
            {events.map((e, i) => (
              <EventRow key={i} event={e} />
            ))}
          </ul>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
