import { useState } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { api, useJobStream } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Square, RefreshCw, FolderOpen, Archive } from "lucide-react"
import { cn } from "@/lib/utils"

import { StateBadge } from "../../dashboard"
import { TERMINAL_STATES } from "../utils"
import { OverviewTab } from "./overview-tab"
import { EventsTab } from "./events-tab"
import { MetricsTab } from "./metrics-tab"
import { FilesTab } from "./files-tab"
import { CompareTab } from "./compare-tab"

type TabKey = "overview" | "events" | "metrics" | "files" | "compare"

export function JobDetail({
  jobId,
  onSelectJob,
  compareMode,
  compareIds,
}: {
  jobId: string
  onSelectJob: (id: string | null) => void
  compareMode: boolean
  compareIds: string[]
}) {
  const queryClient = useQueryClient()
  const [tab, setTab] = useState<TabKey>("overview")
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
  const isLive = data?.state === "running"
  const isTerminal = data ? TERMINAL_STATES.has(data.state) : false
  const showCompare = compareMode && compareIds.length >= 2

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
    if (
      !window.confirm(
        `确定要归档任务 ${data.id.slice(-8)} 吗？工作区将被移到 _archive/，记录会从列表中移除。`,
      )
    ) {
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
              WS{" "}
              {stream.status === "open"
                ? "已连接"
                : stream.status === "closed"
                  ? "已断开"
                  : "等待中"}
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
            <RefreshCw
              className={cn("size-3", busy === "rerun" && "animate-spin")}
            />{" "}
            再次运行
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

      <Tabs
        value={tab}
        onValueChange={(v) => setTab(v as TabKey)}
        className="flex-1 min-h-0 flex flex-col"
      >
        <div className="px-7 pt-4 pb-2 border-b border-border/60 bg-background/40">
          <TabsList variant="line">
            <TabsTrigger value="overview">概览</TabsTrigger>
            <TabsTrigger value="events">事件</TabsTrigger>
            <TabsTrigger value="metrics">指标曲线</TabsTrigger>
            <TabsTrigger value="files">产物文件</TabsTrigger>
            {showCompare && <TabsTrigger value="compare">对比</TabsTrigger>}
          </TabsList>
        </div>

        <div className="flex-1 min-h-0 overflow-hidden">
          <TabsContent value="overview" className="h-full">
            <ScrollArea className="h-full">
              <div className="px-7 py-5">
                <OverviewTab jobId={jobId} job={data} events={events} />
              </div>
            </ScrollArea>
          </TabsContent>
          <TabsContent value="events" className="h-full">
            <div className="px-7 py-5 h-full min-h-0 flex flex-col">
              <EventsTab events={events} status={stream.status} />
            </div>
          </TabsContent>
          <TabsContent value="metrics" className="h-full">
            <ScrollArea className="h-full">
              <div className="px-7 py-5">
                <MetricsTab jobId={jobId} jobState={data?.state} />
              </div>
            </ScrollArea>
          </TabsContent>
          <TabsContent value="files" className="h-full">
            <ScrollArea className="h-full">
              <div className="px-7 py-5">
                <FilesTab jobId={jobId} jobState={data?.state} />
              </div>
            </ScrollArea>
          </TabsContent>
          {showCompare && (
            <TabsContent value="compare" className="h-full">
              <ScrollArea className="h-full">
                <div className="px-7 py-5">
                  <CompareTab compareIds={compareIds} />
                </div>
              </ScrollArea>
            </TabsContent>
          )}
        </div>
      </Tabs>
    </div>
  )
}
