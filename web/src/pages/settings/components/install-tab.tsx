import { useEffect, useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Download, Loader2 } from "lucide-react"
import {
  api,
  useBootstrapStream,
  type BackendDescriptor,
  type BackendId,
  type BootstrapEvent,
} from "@/lib/api"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { cn } from "@/lib/utils"
import { BackendStatusCard } from "./backend-status-card"

function StatusBadge({ status }: { status: string }) {
  const tone =
    status === "succeeded"
      ? "text-emerald-600 dark:text-emerald-400 border-emerald-500/40 bg-emerald-500/5"
      : status === "failed"
        ? "text-destructive border-destructive/40 bg-destructive/5"
        : status === "running"
          ? "text-amber-600 dark:text-amber-400 border-amber-500/40 bg-amber-500/5"
          : "text-muted-foreground border-border/60 bg-muted/30"
  return (
    <span
      className={cn(
        "px-2 py-0.5 text-[10px] uppercase tracking-[0.18em] rounded-[3px] border font-mono",
        tone,
      )}
    >
      {status}
    </span>
  )
}

function EventLog({ events }: { events: BootstrapEvent[] }) {
  return (
    <div className="rounded-[4px] border border-border/60 bg-muted/30 max-h-64 overflow-y-auto">
      <ol className="divide-y divide-border/40 font-mono text-[11px]">
        {events.map((ev, idx) => (
          <li
            key={`${ev.ts}-${idx}`}
            className={cn(
              "px-3 py-1.5 flex items-start gap-2",
              ev.level === "error" && "text-destructive",
              ev.level === "done" && "text-emerald-600 dark:text-emerald-400",
            )}
          >
            <span className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground/70 w-12 shrink-0 pt-0.5">
              {ev.level}
            </span>
            <span className="break-all">{ev.message}</span>
          </li>
        ))}
      </ol>
    </div>
  )
}

/**
 * One-click install panel. The user picks a backend, hits 安装, and the
 * server kicks off the registry-driven bootstrap runner. Because the server
 * keeps a single bootstrap session at a time, while one is in flight we
 * lock the selector to whatever backend is actually running.
 */
export function InstallTab() {
  const qc = useQueryClient()

  const backendsQuery = useQuery({
    queryKey: ["backends"],
    queryFn: api.listBackends,
  })

  // Poll status as a fallback so the panel survives a page reload while an
  // install is mid-flight, and so we never miss a terminal frame the WS may
  // have sent before we attached.
  const statusQuery = useQuery({
    queryKey: ["backend-bootstrap-status"],
    queryFn: api.getBootstrapStatus,
    refetchInterval: (query) =>
      query.state.data?.status === "running" ? 1500 : false,
  })

  const status = statusQuery.data?.status ?? "idle"
  const isRunning = status === "running"
  const sessionBackend = statusQuery.data?.backend

  // The user's currently-selected backend — we initialize it from the
  // settings default once both queries land.
  const [selected, setSelected] = useState<BackendId | null>(null)
  useEffect(() => {
    if (selected || !backendsQuery.data) return
    setSelected(backendsQuery.data.default)
  }, [backendsQuery.data, selected])

  // While a session is running we always show the running backend, no matter
  // what the user previously picked — that's also what the action button
  // operates on, so the UX is unambiguous.
  const effective: BackendId | null = isRunning
    ? (sessionBackend ?? selected)
    : selected

  // Stream live events while a session is active. Prefer the streamed
  // events when the WS has produced any (lower latency than the poll);
  // otherwise fall back to the buffered events from the polling query.
  const { events: streamedEvents } = useBootstrapStream(isRunning)
  const polled = statusQuery.data?.events ?? []
  const events: BootstrapEvent[] =
    streamedEvents.length > 0 ? streamedEvents : polled

  const start = useMutation({
    mutationFn: (backend: BackendId) => api.startBootstrap({ backend }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["backend-bootstrap-status"] })
    },
  })

  // When an install transitions to a terminal state, refresh the backend
  // catalog so the user sees their new checkout immediately.
  useEffect(() => {
    if (status === "succeeded" || status === "failed") {
      qc.invalidateQueries({ queryKey: ["settings"] })
      qc.invalidateQueries({ queryKey: ["backends"] })
      qc.invalidateQueries({ queryKey: ["health"] })
    }
  }, [status, qc])

  const startError = start.error as Error | null
  const lastError = events.find((e) => e.level === "error")

  const descriptor: BackendDescriptor | undefined = useMemo(() => {
    if (!backendsQuery.data || !effective) return undefined
    return backendsQuery.data.backends.find((b) => b.id === effective)
  }, [backendsQuery.data, effective])

  const isOtherSessionRunning =
    isRunning && sessionBackend !== undefined && sessionBackend !== selected

  return (
    <div className="space-y-5">
      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">安装训练后端</CardTitle>
          <CardDescription>
            克隆仓库、创建 venv，并安装 PyTorch 与依赖。与命令行 {" "}
            <code className="text-foreground">lorahub bootstrap-*</code> 等价；
            安装在后台运行，请保持本页打开。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-xs text-muted-foreground">后端</span>
            <Select
              value={effective ?? undefined}
              onValueChange={(v) => setSelected(v as BackendId)}
              disabled={isRunning || !backendsQuery.data}
            >
              <SelectTrigger className="w-64 text-xs font-mono h-8">
                <SelectValue placeholder="选择要安装的后端" />
              </SelectTrigger>
              <SelectContent>
                {(backendsQuery.data?.backends ?? []).map((b) => (
                  <SelectItem key={b.id} value={b.id}>
                    {b.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              size="sm"
              disabled={
                isRunning ||
                start.isPending ||
                !effective ||
                !backendsQuery.data
              }
              onClick={() => effective && start.mutate(effective)}
            >
              {isRunning ? (
                <Loader2 className="size-3 animate-spin" />
              ) : (
                <Download className="size-3" />
              )}
              {isRunning ? "安装中…" : "安装"}
            </Button>
            <StatusBadge status={status} />
            {isOtherSessionRunning && (
              <span className="text-xs text-amber-600 dark:text-amber-400">
                另一个后端（
                <code className="font-mono">{sessionBackend}</code>
                ）正在安装，请等待完成。
              </span>
            )}
            {startError && (
              <span className="text-xs text-destructive font-mono truncate max-w-md">
                {startError.message}
              </span>
            )}
            {!startError && status === "failed" && lastError && (
              <span className="text-xs text-destructive font-mono truncate max-w-md">
                {lastError.message}
              </span>
            )}
          </div>

          {descriptor && (
            <BackendStatusCard
              descriptor={descriptor}
              isDefault={descriptor.id === backendsQuery.data?.default}
              compact
            />
          )}

          {events.length > 0 && <EventLog events={events} />}
        </CardContent>
      </Card>
    </div>
  )
}
