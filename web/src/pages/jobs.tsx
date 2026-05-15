import { useState, useMemo } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { api, useJobStream, type TrainingEvent } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { ScrollArea } from "@/components/ui/scroll-area"
import { StateBadge } from "./dashboard"
import { Square, RefreshCw, FolderOpen, Archive } from "lucide-react"
import { cn } from "@/lib/utils"

const TERMINAL_STATES = new Set([
  "succeeded",
  "failed",
  "canceled",
  "interrupted",
])

const EVENT_TYPE_LABELS: Record<string, string> = {
  step: "训练步",
  epoch_end: "回合结束",
  checkpoint_saved: "保存检查点",
  sample_ready: "样本生成",
  done: "完成",
  error: "错误",
  log: "日志",
  start: "启动",
  cancel: "取消",
}

export function JobsPage() {
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.listJobs, refetchInterval: 2000 })
  const list = jobs.data?.jobs ?? []
  const selected = selectedId ? list.find((j) => j.id === selectedId) : null

  return (
    <div className="grid grid-cols-[minmax(360px,420px)_1fr] h-full">
      <aside className="border-r border-border/60 flex flex-col min-h-0">
        <header className="px-5 py-4 border-b border-border/60">
          <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
            训练任务
          </div>
          <div className="text-xs text-muted-foreground mt-0.5">
            共 {list.length} 个 · 每 2 秒刷新
          </div>
        </header>
        <ScrollArea className="flex-1">
          <ul className="divide-y divide-border/40">
            {list.length === 0 && (
              <li className="px-5 py-10 text-sm text-muted-foreground text-center">
                还没有训练任务。
              </li>
            )}
            {list.slice().reverse().map((j) => {
              const active = j.id === selectedId
              return (
                <li
                  key={j.id}
                  onClick={() => setSelectedId(j.id)}
                  className={cn(
                    "px-5 py-3 cursor-pointer transition-colors",
                    active
                      ? "bg-accent/70 border-l-2 border-l-primary"
                      : "border-l-2 border-l-transparent hover:bg-muted/40",
                  )}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <StateBadge state={j.state} />
                    <code className="text-[11px] font-mono text-muted-foreground">
                      {j.id.slice(-8)}
                    </code>
                    {j.pid !== null && (
                      <span className="text-[10px] text-muted-foreground/60">PID {j.pid}</span>
                    )}
                  </div>
                  <div className="text-xs text-muted-foreground truncate font-mono">
                    {j.workspace}
                  </div>
                  <div className="text-[10px] text-muted-foreground/70 mt-0.5">
                    {new Date(j.created_at).toLocaleString()}
                  </div>
                </li>
              )
            })}
          </ul>
        </ScrollArea>
      </aside>

      <section className="min-w-0 flex flex-col bg-background/60">
        {selected ? (
          <JobDetail jobId={selected.id} onSelectJob={setSelectedId} />
        ) : (
          <div className="flex-1 grid place-items-center text-sm text-muted-foreground">
            从列表中选择一个任务以查看事件流。
          </div>
        )}
      </section>
    </div>
  )
}

function JobDetail({
  jobId,
  onSelectJob,
}: {
  jobId: string
  onSelectJob: (id: string | null) => void
}) {
  const queryClient = useQueryClient()
  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.getJob(jobId),
    refetchInterval: 2000,
  })
  const stream = useJobStream(jobId)
  const [busy, setBusy] = useState<null | "rerun" | "reveal" | "archive">(null)
  const [actionError, setActionError] = useState<string | null>(null)

  const data = job.data
  const events = stream.events
  const lastStep = useMemo(
    () => [...events].reverse().find((e) => e.type === "step"),
    [events],
  )
  const isLive = data?.state === "running"
  const isTerminal = data ? TERMINAL_STATES.has(data.state) : false

  async function onCancel() {
    if (!data) return
    await api.cancelJob(data.id)
    job.refetch()
  }

  async function onRerun() {
    if (!data) return
    setBusy("rerun")
    setActionError(null)
    try {
      const fresh = await api.rerunJob(data.id)
      await queryClient.invalidateQueries({ queryKey: ["jobs"] })
      onSelectJob(fresh.id)
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  async function onReveal() {
    if (!data) return
    setBusy("reveal")
    setActionError(null)
    try {
      await api.revealJob(data.id)
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  async function onArchive() {
    if (!data) return
    if (!window.confirm(
      `确定要归档任务 ${data.id.slice(-8)} 吗？工作区将被移到 _archive/，记录会从列表中移除。`,
    )) {
      return
    }
    setBusy("archive")
    setActionError(null)
    try {
      const result = await api.archiveJob(data.id)
      if (result.warnings.length > 0) {
        setActionError(`归档完成，有警告：${result.warnings.join("；")}`)
      }
      await queryClient.invalidateQueries({ queryKey: ["jobs"] })
      onSelectJob(null)
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="flex flex-col min-h-0 h-full">
      <header className="px-7 py-5 border-b border-border/60 flex items-start gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5">
            {data && <StateBadge state={data.state} />}
            <span className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
              WS {stream.status === "open" ? "已连接" : stream.status === "closed" ? "已断开" : "等待中"}
            </span>
          </div>
          <div className="text-base font-semibold tracking-tight font-mono truncate">
            {jobId}
          </div>
          {data && (
            <div className="text-xs text-muted-foreground mt-1 font-mono truncate">
              {data.workspace}
            </div>
          )}
          {actionError && (
            <div className="text-[11px] text-destructive mt-2 break-all">
              {actionError}
            </div>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Button
            variant="outline"
            size="sm"
            onClick={onReveal}
            disabled={!data || busy !== null}
            title="在文件管理器中打开工作区"
          >
            <FolderOpen className="size-3" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={onRerun}
            disabled={!data || busy !== null}
          >
            <RefreshCw className={cn("size-3", busy === "rerun" && "animate-spin")} /> 再次运行
          </Button>
          {isTerminal && (
            <Button
              variant="outline"
              size="sm"
              onClick={onArchive}
              disabled={busy !== null}
            >
              <Archive className="size-3" /> 归档
            </Button>
          )}
          {isLive && (
            <Button variant="destructive" size="sm" onClick={onCancel}>
              <Square className="size-3" /> 取消
            </Button>
          )}
        </div>
      </header>

      <div className="grid grid-cols-3 gap-3 px-7 py-4 border-b border-border/60">
        <Stat label="状态" value={data?.state ? stateLabel(data.state) : "—"} />
        <Stat
          label="进度"
          value={
            lastStep
              ? `${lastStep.payload.step ?? "?"} / ${lastStep.payload.total_steps ?? "?"}`
              : "—"
          }
        />
        <Stat
          label="损失"
          value={
            typeof lastStep?.payload.loss === "number"
              ? (lastStep.payload.loss as number).toFixed(4)
              : "—"
          }
        />
      </div>

      <Card className="m-4 mb-0 rounded-[6px] border-border/60 shadow-[var(--panel-shadow)] overflow-hidden flex-1 min-h-0 flex flex-col">
        <CardHeader className="py-3 px-4 border-b border-border/60 bg-muted/40">
          <CardTitle className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
            事件流
          </CardTitle>
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

      <Separator className="opacity-0" />
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[4px] border border-border/60 bg-card/70 px-3 py-2">
      <div className="text-[9px] uppercase tracking-[0.2em] text-muted-foreground/70">
        {label}
      </div>
      <div className="text-sm font-semibold tabular-nums truncate">{value}</div>
    </div>
  )
}

function EventRow({ event }: { event: TrainingEvent }) {
  const time = new Date(event.timestamp * 1000).toLocaleTimeString()
  const summary = renderPayload(event)
  const tone = {
    error: "text-destructive",
    done: "text-emerald-600 dark:text-emerald-400",
    checkpoint_saved: "text-cyan-700 dark:text-cyan-400",
    sample_ready: "text-fuchsia-700 dark:text-fuchsia-400",
    epoch_end: "text-primary",
  }[event.type] ?? "text-foreground"
  const label = EVENT_TYPE_LABELS[event.type] ?? event.type

  return (
    <li className="px-4 py-1.5 flex gap-3 items-baseline hover:bg-muted/30">
      <span className="text-muted-foreground/60 shrink-0 text-[11px]">{time}</span>
      <span className={cn("shrink-0 w-[120px] text-[11px] tracking-wide", tone)}>
        {label}
      </span>
      <span className="text-foreground/80 truncate">{summary}</span>
    </li>
  )
}

function renderPayload(e: TrainingEvent): string {
  const p = e.payload
  switch (e.type) {
    case "step":
      return `第 ${p.step}/${p.total_steps} 步${p.loss !== undefined ? ` · 损失 ${(p.loss as number).toFixed(4)}` : ""}`
    case "epoch_end":
      return `第 ${p.epoch}/${p.total_epochs} 回合结束`
    case "checkpoint_saved":
      return String(p.path ?? "")
    case "sample_ready":
      return String(p.path ?? "")
    case "done":
      return `返回码 ${p.returncode} · 用时 ${(p.duration_s as number)?.toFixed?.(1) ?? "?"}s`
    case "log":
      return String(p.message ?? "")
    default:
      return JSON.stringify(p)
  }
}

const STATE_LABELS: Record<string, string> = {
  running: "运行中",
  succeeded: "已完成",
  failed: "失败",
  canceled: "已取消",
  canceling: "取消中",
  queued: "排队中",
  interrupted: "已中断",
}

function stateLabel(state: string): string {
  return STATE_LABELS[state] ?? state
}
