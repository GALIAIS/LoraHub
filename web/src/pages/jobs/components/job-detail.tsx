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
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { PathDisplay } from "@/components/path-display"
import { toast } from "sonner"
import {
  BarChart3,
  Square,
  RefreshCw,
  FolderOpen,
  Archive,
  Skull,
  Pause,
  Play,
  Pencil,
} from "lucide-react"

import { StateBadge } from "../../dashboard"
import { TERMINAL_STATES } from "../utils"
import { expectedTotalSteps } from "../utils"
import { OverviewTab } from "./overview-tab"
import { EventsTab } from "./events-tab"
import { FilesTab } from "./files-tab"
import { RunSummaryCard } from "./run-summary-card"
import { ResumeWithEditDialog } from "./resume-with-edit-dialog"

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
    refetchInterval: 2000,
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
  })
  const stream = useJobStream(jobId)
  const [busy, setBusy] = useState<null | "rerun" | "reveal" | "archive" | "kill" | "pause" | "resume">(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [archiveOpen, setArchiveOpen] = useState(false)
  const [killOpen, setKillOpen] = useState(false)
  const [resumeEditOpen, setResumeEditOpen] = useState(false)

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
      toast.success("已请求暂停,等待训练写出最新 state...", {
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
            description: `工作区路径已复制到剪贴板: ${ws}`,
            duration: 8000,
          })
        } catch {
          toast.warning("服务器无桌面环境", {
            description: `工作区路径: ${ws}`,
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
        toast.warning("归档完成,有警告", {
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
      <header className="px-7 py-5 border-b border-border/60 flex items-start gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5">
            {data && <StateBadge state={data.state} paused={isPaused} />}
            <span className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
              {stream.status === "open"
                ? "实时已连接"
                : stream.status === "closed"
                  ? "已断开"
                  : "等待中"}
            </span>
          </div>
          <div className="text-base font-semibold tracking-tight font-mono truncate">
            {jobId}
          </div>
          {data && (
            <PathDisplay
              path={data.workspace}
              tailSegments={3}
              block
              className="text-xs text-muted-foreground mt-1"
            />
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
            onClick={() =>
              navigate(
                showCompareJumpButton
                  ? `/analysis/compare?ids=${compareIds.join(",")}`
                  : `/analysis/${jobId}`,
              )
            }
            title={
              showCompareJumpButton
                ? "在分析工作台中对比所选任务"
                : "打开训练分析工作台"
            }
          >
            <BarChart3 className="size-3" />{" "}
            {showCompareJumpButton ? "对比分析" : "深入分析"}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={onReveal}
            disabled={!data || busy !== null}
            title="在文件管理器中打开工作区" aria-label="在文件管理器中打开工作区"
          >
            {busy === "reveal" ? (
              <Spinner className="size-3" />
            ) : (
              <FolderOpen className="size-3" />
            )}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={onRerun}
            disabled={!data || busy !== null}
          >
            {busy === "rerun" ? (
              <Spinner className="size-3" />
            ) : (
              <RefreshCw className="size-3" />
            )}{" "}
            再次运行
          </Button>
          {isTerminal && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setArchiveOpen(true)}
              disabled={busy !== null}
            >
              {busy === "archive" ? (
                <Spinner className="size-3" />
              ) : (
                <Archive className="size-3" />
              )}{" "}
              归档
            </Button>
          )}
          {isResumable && (
            <>
              <Button
                variant="default"
                size="sm"
                onClick={onResume}
                disabled={busy !== null}
                title="从最新 state + safetensors 续训(保留 optimizer / lr 进度)"
              >
                {busy === "resume" ? (
                  <Spinner className="size-3" />
                ) : (
                  <Play className="size-3" />
                )}{" "}
                {isPaused ? "继续训练" : "恢复训练"}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setResumeEditOpen(true)}
                disabled={busy !== null || !data?.config_snapshot}
                title="先编辑 lr / dropTokens / 数据集等再续训(权重相关字段会被锁)"
              >
                <Pencil className="size-3" /> 编辑
              </Button>
            </>
          )}
          {isLive && (
            <>
              {/* Pause is only meaningful once the trainer subprocess
                  is actually stepping — preparing means anima_lora is
                  still in resize / cache_latents / cache_text_embeddings,
                  with no checkpoint to resume from yet. SIGINT-ing
                  during preprocess just kills the worker without leaving
                  resumable state on disk; users should "取消" instead.
                  Gate the button on state === "running" only. */}
              {data?.state === "running" ? (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onPause}
                  disabled={busy !== null}
                  title="发送 SIGINT,等待训练写出最新 state 后停止;之后点「继续训练」从此处续"
                >
                  {busy === "pause" ? (
                    <Spinner className="size-3" />
                  ) : (
                    <Pause className="size-3" />
                  )}{" "}
                  暂停
                </Button>
              ) : null}
              <Button variant="destructive" size="sm" onClick={onCancel}>
                <Square className="size-3" /> 取消
              </Button>
              <Button
                variant="destructive"
                size="sm"
                onClick={() => setKillOpen(true)}
                title="强制 SIGKILL 进程组(用于卡死的训练任务)"
                disabled={busy !== null || !data?.pid}
              >
                {busy === "kill" ? (
                  <Spinner className="size-3" />
                ) : (
                  <Skull className="size-3" />
                )}{" "}
                强制终止
              </Button>
            </>
          )}
          {!isLive && data?.state === "interrupted" && data.pid && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setKillOpen(true)}
              title="任务标记为 interrupted 但 PID 仍可能存活,可强制清理"
              disabled={busy !== null}
            >
              <Skull className="size-3" /> 强制终止
            </Button>
          )}
        </div>
      </header>

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
              任务状态会被标记为 <code className="font-mono">interrupted</code>,
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
    </div>
  )
}
