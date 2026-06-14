import { useMemo, useState } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { api, useJobStream } from "@/lib/api"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { toast } from "sonner"
import { Archive, Skull } from "lucide-react"

import { TERMINAL_STATES } from "../utils"
import { expectedTotalSteps } from "../utils"
import { OverviewTab } from "./overview-tab"
import { EventsTab } from "./events-tab"
import { FilesTab } from "./files-tab"
import { RunSummaryCard } from "./run-summary-card"
import { ResumeWithEditDialog } from "./resume-with-edit-dialog"
import { CloneWithStateDialog } from "./clone-with-state-dialog"
import { JobDetailHeader } from "./job-detail-header"

type TabKey = "overview" | "events" | "files"

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
  const navigate = useNavigate()
  const [tab, setTab] = useState<TabKey>("overview")
  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.getJob(jobId),
    // Stop polling once the job lands in a terminal state. Without
    // this the detail panel kept hitting /api/jobs/<id> every 2s
    // forever for a finished run; the SSE stream covers any remaining
    // race-window updates anyway.
    refetchInterval: (query) => {
      const j = query.state.data as { state?: string } | undefined
      if (j?.state && TERMINAL_STATES.has(j.state)) return false
      return 2000
    },
    staleTime: 1_500,
  })
  // Drive the run-summary card on the job detail header. Mirrors the
  // refresh cadence of the metrics tab — short while running, off when
  // the job has reached a terminal state.
  const summaryMetrics = useQuery({
    queryKey: ["job-metrics", jobId],
    queryFn: () => api.getJobMetrics(jobId),
    refetchInterval: () => {
      const j = job.data
      const terminal = j ? TERMINAL_STATES.has(j.state) : false
      if (terminal) return false
      return 4000
    },
    staleTime: 2_000,
  })
  const stream = useJobStream(jobId)
  const [busy, setBusy] = useState<null | "rerun" | "reveal" | "archive" | "kill" | "pause" | "resume">(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [archiveOpen, setArchiveOpen] = useState(false)
  const [killOpen, setKillOpen] = useState(false)
  const [resumeEditOpen, setResumeEditOpen] = useState(false)
  const [cloneOpen, setCloneOpen] = useState(false)

  // Fall back to a config-derived total step count when the backend hasn't
  // yet emitted a `total_steps` payload. We need the dataset image count,
  // which we lazily fetch via /datasets/scan against `config.dataset.source`.
  const datasetSource = useMemo(() => {
    const cfg = (job.data as { config_snapshot?: Record<string, unknown> } | undefined)
      ?.config_snapshot
    const ds = cfg?.["dataset"] as Record<string, unknown> | undefined
    const src = ds?.["source"]
    return typeof src === "string" ? src : null
  }, [job.data])
  const datasetScan = useQuery({
    // Recursive walk: anima / dp datasets often nest images under
    // sub-folders (chapters, scene groups). Without it the count
    // collapses to whatever sits in the top dir, which is usually
    // empty, and the progress bar never gets a denominator.
    queryKey: ["dataset-scan", datasetSource, true],
    queryFn: () => api.scanDataset(datasetSource!, true, 0),
    enabled: !!datasetSource,
    refetchOnWindowFocus: false,
    staleTime: 5 * 60_000,
  })
  const fallbackTotalSteps = useMemo(() => {
    const cfg = (job.data as { config_snapshot?: Record<string, unknown> } | undefined)
      ?.config_snapshot
    return expectedTotalSteps(cfg, datasetScan.data?.image_files ?? null)
  }, [job.data, datasetScan.data])

  const data = job.data
  const events = stream.events
  // "Live" = the job slot is held by us — running, preparing
  // (anima_lora preprocess phase), queued (waiting for the slot),
  // or canceling (we already asked it to stop). All four states
  // need a visible "取消" button so the user can recover from a
  // bug-stuck job without restarting uvicorn. Without this the
  // queued / preparing rows had no way out and the user had to ssh
  // in to flip them.
  const isLive =
    data?.state === "running"
    || data?.state === "preparing"
    || data?.state === "queued"
    || data?.state === "canceling"
  const isTerminal = data ? TERMINAL_STATES.has(data.state) : false
  // "Paused" is purely a metadata signal stamped by ``DELETE
  // /jobs/<id>?paused=true`` so the UI can flip the next render
  // from "取消" to "恢复训练". The job is in canceled state on the
  // backend; ``isPaused`` just changes which button we show.
  const isPaused = Boolean(
    data?.metadata && (data.metadata as Record<string, unknown>).paused === true,
  )
  // Resume eligibility: the route accepts canceled / failed /
  // interrupted (see _RESUMABLE_STATES). We additionally show the
  // button on paused jobs because that's the workflow we set up
  // explicitly.
  const isResumable =
    data?.state === "canceled"
    || data?.state === "failed"
    || data?.state === "interrupted"
  // Compare-mode in the jobs page is a *jumping-off point*: we keep the
  // checkbox UX in the sidebar, but the actual compare panels live on
  // the analysis workbench so the detail view stays focused on running
  // / inspecting one job.
  const showCompareJumpButton = compareMode && compareIds.length >= 2

  async function onCancel() {
    if (!data) return
    await api.cancelJob(data.id)
    job.refetch()
  }

  async function onPause() {
    if (!data) return
    setBusy("pause")
    setActionError(null)
    try {
      await api.pauseJob(data.id)
      await job.refetch()
      toast.success("已请求暂停，等待训练写出最新 state…", {
        description: "随后点「恢复训练」从此处继续",
      })
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setActionError(msg)
      toast.error("暂停失败", { description: msg })
    } finally {
      setBusy(null)
    }
  }

  async function onResume() {
    if (!data) return
    setBusy("resume")
    setActionError(null)
    try {
      const fresh = await api.resumeJob(data.id)
      await queryClient.invalidateQueries({ queryKey: ["jobs"] })
      await job.refetch()
      toast.success("训练已从断点恢复", {
        description: `任务 ID ${fresh.id.slice(-8)}`,
      })
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setActionError(msg)
      toast.error("恢复失败", { description: msg })
    } finally {
      setBusy(null)
    }
  }

  async function onKill() {
    if (!data) return
    setBusy("kill")
    setActionError(null)
    try {
      await api.killJob(data.id)
      await job.refetch()
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
      setKillOpen(false)
    }
  }

  async function onRerun() {
    if (!data) return
    setBusy("rerun")
    setActionError(null)
    try {
      const fresh = await api.rerunJob(data.id)
      await queryClient.invalidateQueries({ queryKey: ["jobs"] })
      onSelectJob(fresh.id)
      toast.success("任务已再次启动", {
        description: `新任务 ID ${fresh.id.slice(-8)}`,
      })
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setActionError(msg)
      toast.error("再次启动失败", { description: msg })
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
      toast.success("已在文件管理器中打开工作区")
    } catch (e) {
      // Headless / remote API: server can't open a file manager. Surface
      // the resolved workspace path instead so the user can copy it
      // straight from the toast. The detail payload is a JSON object
      // with `code`, `message`, and `workspace`; older fastapi
      // serialisations stringify it, so handle both shapes.
      const raw = e instanceof Error ? e.message : String(e)
      let parsed: { workspace?: string; message?: string } | null = null
      const match = raw.match(/\{.*\}$/)
      if (match) {
        try {
          parsed = JSON.parse(match[0].replace(/'/g, '"'))
        } catch {
          /* ignore */
        }
      }
      if (parsed?.workspace) {
        const ws = parsed.workspace
        try {
          await navigator.clipboard.writeText(ws)
          toast.info("服务器无桌面环境", {
            description: `工作区路径已复制到剪贴板：${ws}`,
            duration: 8000,
          })
        } catch {
          toast.warning("服务器无桌面环境", {
            description: `工作区路径：${ws}`,
            duration: 8000,
          })
        }
      } else {
        toast.error("打开工作区失败", { description: raw })
      }
    } finally {
      setBusy(null)
    }
  }

  async function onArchive() {
    if (!data) return
    setArchiveOpen(false)
    setBusy("archive")
    setActionError(null)
    try {
      const result = await api.archiveJob(data.id)
      if (result.warnings.length > 0) {
        toast.warning("归档完成，有警告", {
          description: result.warnings.join("；"),
          duration: 8000,
        })
      } else {
        toast.success("已归档到 _archive/")
      }
      await queryClient.invalidateQueries({ queryKey: ["jobs"] })
      onSelectJob(null)
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setActionError(msg)
      toast.error("归档失败", { description: msg })
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="flex flex-col min-h-0 h-full">
      <JobDetailHeader
        actionError={actionError}
        busy={busy}
        compareIds={compareIds}
        data={data}
        isLive={isLive}
        isPaused={isPaused}
        isResumable={isResumable}
        isTerminal={isTerminal}
        jobId={jobId}
        showCompareJumpButton={showCompareJumpButton}
        streamStatus={stream.status}
        onArchiveOpen={() => setArchiveOpen(true)}
        onCancel={onCancel}
        onCloneOpen={() => setCloneOpen(true)}
        onKillOpen={() => setKillOpen(true)}
        onNavigateAnalysis={navigate}
        onPause={onPause}
        onResume={onResume}
        onResumeEditOpen={() => setResumeEditOpen(true)}
        onReveal={onReveal}
        onRerun={onRerun}
      />
      <div className="px-7 pt-4">
        <RunSummaryCard
          job={data}
          metrics={summaryMetrics.data}
          fallbackTotalSteps={fallbackTotalSteps}
        />
      </div>

      <Tabs
        value={tab}
        onValueChange={(v) => setTab(v as TabKey)}
        className="flex-1 min-h-0 flex flex-col"
      >
        <div className="px-7 pt-4 pb-2 border-b border-border/60 bg-background/40">
          <TabsList variant="line">
            <TabsTrigger value="overview">概览</TabsTrigger>
            <TabsTrigger value="events">事件</TabsTrigger>
            <TabsTrigger value="files">产物文件</TabsTrigger>
          </TabsList>
        </div>

        <div className="flex-1 min-h-0 overflow-hidden">
          <TabsContent value="overview" className="h-full">
            <ScrollArea className="h-full">
              <div className="px-7 py-5">
                <OverviewTab
                  jobId={jobId}
                  job={data}
                  events={events}
                  fallbackTotalSteps={fallbackTotalSteps}
                />
              </div>
            </ScrollArea>
          </TabsContent>
          <TabsContent value="events" className="h-full">
            <div className="px-7 py-5 h-full min-h-0 flex flex-col">
              <EventsTab
                events={events}
                status={stream.status}
                jobId={jobId}
                fallbackTotalSteps={fallbackTotalSteps}
              />
            </div>
          </TabsContent>
          <TabsContent value="files" className="h-full">
            <ScrollArea className="h-full">
              <div className="px-7 py-5">
                <FilesTab jobId={jobId} jobState={data?.state} />
              </div>
            </ScrollArea>
          </TabsContent>
        </div>
      </Tabs>
      <AlertDialog open={archiveOpen} onOpenChange={setArchiveOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>归档训练任务</AlertDialogTitle>
            <AlertDialogDescription>
              工作区将被移动到 <code className="font-mono">_archive/</code>，
              记录会从列表中移除。
              {data && (
                <>
                  {" "}任务 ID:{" "}
                  <code className="font-mono">{data.id.slice(-8)}</code>。
                </>
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              onClick={(e) => {
                e.preventDefault()
                onArchive()
              }}
            >
              <Archive className="size-3" /> 确认归档
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      <AlertDialog open={killOpen} onOpenChange={setKillOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>强制终止训练任务</AlertDialogTitle>
            <AlertDialogDescription>
              将向 PID <code className="font-mono">{data?.pid ?? "—"}</code> 及其
              进程组发送 SIGKILL。常用于训练僵死、取消按钮无响应的场景。
              任务状态会被标记为 <code className="font-mono">interrupted</code>，
              checkpoint 不受影响。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={(e) => {
                e.preventDefault()
                onKill()
              }}
            >
              <Skull className="size-3" /> 确认强制终止
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      {resumeEditOpen && data && (
        <ResumeWithEditDialog
          job={data}
          onClose={() => setResumeEditOpen(false)}
          onResumed={() => {
            setResumeEditOpen(false)
            job.refetch()
          }}
        />
      )}
      {cloneOpen && data && (
        <CloneWithStateDialog
          job={data}
          onClose={() => setCloneOpen(false)}
          onCloned={(newId) => {
            setCloneOpen(false)
            queryClient.invalidateQueries({ queryKey: ["jobs"] })
            onSelectJob(newId)
          }}
        />
      )}
    </div>
  )
}
