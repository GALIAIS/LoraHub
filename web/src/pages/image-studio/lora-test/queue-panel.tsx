/**
 * 右侧队列面板 — 当前生成任务进度与最近事件。
 */
import { Loader2, Square } from "lucide-react"
import type { TaskSessionRecord } from "@/lib/api/tasks"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"

export function QueuePanel({
  session,
  loading,
  onCancel,
  canceling,
}: {
  session: TaskSessionRecord | null
  loading: boolean
  onCancel: () => void
  canceling: boolean
}) {
  const active =
    session?.status === "queued" ||
    session?.status === "running" ||
    session?.status === "stop_requested"
  const stopping = session?.status === "stop_requested" || canceling
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>队列</CardTitle>
        <CardDescription>刷新后仍可恢复最近任务。</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {!session && (
          <p className="text-xs text-muted-foreground">
            暂无生成任务。填写 prompt 后点击生成。
          </p>
        )}
        {session && (
          <>
            <div className="flex items-center justify-between gap-2">
              <Badge variant={session.status === "failed" ? "destructive" : "outline"}>
                {session.status}
              </Badge>
              {loading && <Loader2 className="size-3 animate-spin text-muted-foreground" />}
            </div>
            <Progress value={session.percent} />
            <div className="text-xs tabular-nums text-muted-foreground">
              {Math.round(session.percent)}%
            </div>
            {session.error && (
              <div className="rounded-[6px] border border-destructive/30 bg-destructive/5 p-2 text-xs text-destructive">
                {session.error}
              </div>
            )}
            {active && (
              <Button
                variant="destructive"
                size="sm"
                onClick={onCancel}
                disabled={stopping}
              >
                {stopping ? <Loader2 className="animate-spin" /> : <Square />}
                {stopping ? "正在停止" : "停止"}
              </Button>
            )}
            <div className="flex flex-col gap-2">
              {session.events.slice(-8).map((event, index) => (
                <div
                  key={`${event.ts}-${index}`}
                  className="rounded-[6px] bg-muted/35 px-2 py-1.5 text-[11px]"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate">{event.message}</span>
                    {event.percent != null && (
                      <span className="tabular-nums text-muted-foreground">
                        {Math.round(event.percent)}%
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}
