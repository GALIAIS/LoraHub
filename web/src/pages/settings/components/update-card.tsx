/**
 * Self-update card — version check + one-click upgrade.
 *
 * The card mirrors ShiroManager's update-status panel:
 *  - Shows current vs latest tag/commit on the chosen channel.
 *  - Lets the user pick channel: ``tag`` (release cuts, default) or
 *    ``main`` (bleeding edge).
 *  - "立即更新" runs the SSE-streamed upgrade (git → deps → build) and
 *    by default re-execs the daemon at the end so the new code wins
 *    without the user having to touch the shell.
 *  - Renders the streaming log line-by-line (same shape as the
 *    bootstrap installer panel).
 */
import { useEffect, useRef, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  AlertTriangle,
  CheckCircle,
  Download,
  ExternalLink,
  Loader2,
  RefreshCw,
} from "lucide-react"
import { api, type UpdateEvent } from "@/lib/api"
import {
  useSystemVersion,
  type UpdateChannel,
} from "@/hooks/use-system-version"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
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
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Switch } from "@/components/ui/switch"
import {
  Tabs,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"
import { cn } from "@/lib/utils"

const PHASE_LABEL: Record<UpdateEvent["phase"], string> = {
  git: "拉取代码",
  deps: "安装 Python 依赖",
  build: "构建前端",
  done: "完成",
  restart: "重启服务",
  error: "失败",
}

const ACTIVE_UPDATE_TASK_STATUSES = new Set(["pending", "running"])

function isUpdateEvent(value: unknown): value is UpdateEvent {
  if (!value || typeof value !== "object") return false
  const obj = value as Partial<UpdateEvent>
  return (
    typeof obj.phase === "string" &&
    typeof obj.level === "string" &&
    typeof obj.message === "string"
  )
}

function taskEventsToUpdateEvents(
  events: Array<{ message: string; level: string; payload?: unknown }>,
): UpdateEvent[] {
  return events.map((event) => {
    if (isUpdateEvent(event.payload)) return event.payload
    return {
      phase: event.level === "error" ? "error" : "git",
      level:
        event.level === "error" || event.level === "warn"
          ? event.level
          : "info",
      message: event.message,
    }
  })
}

function channelVersionLabel(
  info: ReturnType<typeof useSystemVersion>["data"],
  channel: UpdateChannel,
): string {
  if (!info) return "检查中"
  if (channel === "tag") return info.tag_name ?? (info.latest ? `v${info.latest}` : "—")
  return info.latest_commit?.slice(0, 7) ?? info.latest ?? "—"
}

function channelVersionTone(channel: UpdateChannel): string {
  return channel === "tag"
    ? "text-emerald-700 dark:text-emerald-400"
    : "text-sky-700 dark:text-sky-400"
}

function currentDetailLabel(
  info: ReturnType<typeof useSystemVersion>["data"],
): string {
  return info?.current ?? "—"
}

function remoteDetailLabel(
  info: ReturnType<typeof useSystemVersion>["data"],
): string {
  return info?.latest_commit?.slice(0, 7) ?? info?.tag_name ?? info?.latest ?? "—"
}

export function UpdateCard() {
  const qc = useQueryClient()
  const [channel, setChannel] = useState<UpdateChannel>("tag")
  const tagVersion = useSystemVersion("tag")
  const devVersion = useSystemVersion("dev")
  const version = channel === "tag" ? tagVersion : devVersion
  const [restart, setRestart] = useState(true)
  const [build, setBuild] = useState(true)
  // ``force`` discards local changes (git reset --hard + clean -fd)
  // before checkout. Destructive — guarded by the AlertDialog below.
  const [force, setForce] = useState(false)
  // Two-phase confirm: when force is on, the apply button instead
  // pops a dialog asking the user to acknowledge what they're about
  // to lose. Cancel rolls force back to false; confirm runs the
  // upgrade.
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [events, setEvents] = useState<UpdateEvent[]>([])
  const [running, setRunning] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const logRef = useRef<HTMLDivElement | null>(null)

  const latestUpdateTask = useQuery({
    queryKey: ["tasks", "latest", "system_update"],
    queryFn: () => api.getLatestTask("system_update"),
    retry: false,
    staleTime: 10_000,
    refetchInterval: (query) =>
      query.state.data?.status === "running" ? 3000 : false,
    throwOnError: false,
  })

  // Auto-scroll the log to the latest line as events stream in.
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [events.length])

  useEffect(() => {
    if (events.length > 0 || running) return
    const task = latestUpdateTask.data
    if (!task) return
    if (!ACTIVE_UPDATE_TASK_STATUSES.has(task.status)) return
    if (task.events.length === 0) return
    setEvents(taskEventsToUpdateEvents(task.events))
    setRunning(task.status === "running" || task.status === "pending")
  }, [events.length, latestUpdateTask.data, running])

  const recheck = useMutation({
    mutationFn: () => api.getSystemVersion(channel, true),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["system-version", channel] })
    },
  })

  const runUpdate = async () => {
    setEvents([])
    setRunning(true)
    const ac = new AbortController()
    abortRef.current = ac
    try {
      await api.applySystemUpdate(
        { channel, build, restart, force },
        (ev) => setEvents((prev) => [...prev, ev]),
        ac.signal,
      )
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : String(exc)
      setEvents((prev) => [
        ...prev,
        { phase: "error", level: "error", message },
      ])
    } finally {
      setRunning(false)
      abortRef.current = null
      qc.invalidateQueries({ queryKey: ["system-version"] })
      qc.invalidateQueries({ queryKey: ["tasks", "latest", "system_update"] })
    }
  }

  // Apply-button click: force=on sends the user through the
  // destructive-confirm dialog first; force=off goes straight to
  // runUpdate. The dialog's Confirm calls runUpdate after closing.
  const onApplyClick = () => {
    if (force) {
      setConfirmOpen(true)
      return
    }
    void runUpdate()
  }

  const cancel = () => {
    abortRef.current?.abort()
  }

  const info = version.data
  const versionTone = channelVersionTone(channel)
  const checkedAt = info?.checked_at
    ? new Date(info.checked_at).toLocaleString()
    : "—"
  const isDockerInstall = info?.install_kind === "docker"

  const headerStatus = (() => {
    if (!info) return null
    if (info.error) {
      return (
        <Badge variant="outline" className="rounded-[2px] gap-1 text-amber-700 dark:text-amber-400 border-amber-500/40">
          <AlertTriangle className="size-3" />
          网络异常
        </Badge>
      )
    }
    if (info.update_available) {
      return (
        <Badge className="rounded-[2px] gap-1 bg-primary/15 text-primary border-primary/40">
          <Download className="size-3" />
          有可用更新
        </Badge>
      )
    }
    return (
      <Badge variant="outline" className="rounded-[2px] gap-1 text-emerald-700 dark:text-emerald-400 border-emerald-500/40">
        <CheckCircle className="size-3" />
        已是最新
      </Badge>
    )
  })()

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start gap-3">
          <div className="flex-1 min-w-0">
            <CardTitle className="text-base flex items-center gap-2">
              <Download className="size-4 text-muted-foreground" />
              软件更新
            </CardTitle>
            <CardDescription>
              通过 GitHub Releases 检查并升级 LoraHub。镜像加速可在「网络加速」配置。
            </CardDescription>
          </div>
          {headerStatus}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <Tabs value={channel} onValueChange={(v) => setChannel(v as UpdateChannel)}>
          <TabsList variant="line" className="h-8">
            <TabsTrigger value="tag" className="text-xs">
              <span>正式版</span>
              <span className="font-mono text-[11px] text-emerald-700 dark:text-emerald-400">
                {channelVersionLabel(tagVersion.data, "tag")}
              </span>
            </TabsTrigger>
            <TabsTrigger value="dev" className="text-xs">
              <span>Dev</span>
              <span className="font-mono text-[11px] text-sky-700 dark:text-sky-400">
                {channelVersionLabel(devVersion.data, "dev")}
              </span>
            </TabsTrigger>
          </TabsList>
        </Tabs>

        <div className="grid grid-cols-[7rem_1fr] gap-x-4 gap-y-2 text-sm">
          <span className="text-muted-foreground">当前版本</span>
          <span className="flex items-center gap-2 flex-wrap">
            <code className={cn("font-mono text-[12px] font-medium", versionTone)}>
              {currentDetailLabel(info)}
            </code>
            {info?.current_commit && (
              <code className={cn("font-mono text-[11px]", versionTone)}>
                {info.current_commit.slice(0, 7)}
              </code>
            )}
            {info?.version_source && info.version_source !== "hatch-vcs" && (
              <Badge
                variant="outline"
                className="rounded-[2px] text-[10px] tracking-wide border-amber-500/40 text-amber-700 dark:text-amber-400"
                title={
                  info.version_source === "fallback"
                    ? "无法读取版本元数据,使用回退占位"
                    : info.version_source === "changelog"
                      ? "从 CHANGELOG.md 读取的最近发布版本(可能落后于 commit)"
                      : "从 dist 元数据读取(非 git 检出)"
                }
              >
                {info.version_source === "fallback"
                  ? "未知"
                  : info.version_source === "changelog"
                    ? "估算"
                    : "已安装"}
              </Badge>
            )}
          </span>

          <span className="text-muted-foreground">远端版本</span>
          <span className="flex items-center gap-2 flex-wrap">
            <code className={cn("font-mono text-[12px] font-medium", versionTone)}>
              {remoteDetailLabel(info)}
            </code>
          </span>

          <span className="text-muted-foreground">最近检查</span>
          <span className="text-[12px] text-muted-foreground">{checkedAt}</span>

          {info && info.git_checkout === false && (
            <>
              <span className="text-amber-700 dark:text-amber-400 text-[12px]">
                安装方式
              </span>
              <span className="text-amber-700 dark:text-amber-400 text-[12px]">
                {isDockerInstall ? (
                  <>
                    Docker 安装不在容器内改写源码。请在宿主机执行
                    <code className="font-mono text-[11px] mx-1">git pull</code>
                    后重新
                    <code className="font-mono text-[11px] mx-1">docker compose up -d --build</code>
                    。
                  </>
                ) : (
                  <>
                    未检测到 .git 目录。在线更新无法运行,请改用 git clone 安装。
                  </>
                )}
              </span>
            </>
          )}

          {info?.is_dirty && (
            <>
              <span className="text-amber-700 dark:text-amber-400 text-[12px]">
                工作树状态
              </span>
              <span className="text-amber-700 dark:text-amber-400 text-[12px]">
                检测到本地源码修改 — 升级时会自动 stash + pop,冲突需手动解决
              </span>
            </>
          )}

          {info?.error && (
            <>
              <span className="text-amber-700 dark:text-amber-400 text-[12px]">
                提示
              </span>
              <span className="text-amber-700 dark:text-amber-400 text-[12px] font-mono">
                {info.error}
              </span>
            </>
          )}
        </div>

        {info?.release_notes && (
          <details className="rounded-[4px] border border-border/60 bg-muted/30 px-3 py-2">
            <summary className="text-[12px] cursor-pointer select-none text-muted-foreground hover:text-foreground">
              更新说明
            </summary>
            <pre className="mt-2 max-h-48 overflow-y-auto text-[11px] font-mono whitespace-pre-wrap leading-relaxed">
              {info.release_notes}
            </pre>
          </details>
        )}

        <div className="flex flex-wrap gap-x-6 gap-y-2 text-[12px]">
          <label className="flex items-center gap-2 cursor-pointer">
            <Switch checked={build} onCheckedChange={setBuild} disabled={running} />
            <span>升级时重新构建前端</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <Switch checked={restart} onCheckedChange={setRestart} disabled={running} />
            <span>升级完成后自动重启服务</span>
          </label>
          <label
            className={cn(
              "flex items-center gap-2 cursor-pointer",
              force && "text-destructive",
            )}
          >
            <Switch
              checked={force}
              onCheckedChange={setForce}
              disabled={running}
            />
            <span className="flex items-center gap-1">
              忽略冲突强制升级
              {force && <AlertTriangle className="size-3" />}
            </span>
          </label>
        </div>

        <div className="flex items-center gap-2">
          <Button
            size="sm"
            disabled={
              running ||
              !info?.update_available ||
              info?.git_checkout === false
            }
            onClick={onApplyClick}
            variant={force ? "destructive" : "default"}
            title={
              info?.git_checkout === false
                ? isDockerInstall
                  ? "Docker 安装请在宿主机拉取代码后重建容器"
                  : "非 git 安装无法在线更新"
                : undefined
            }
          >
            {running ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <Download className="size-3" />
            )}
            {running ? "正在升级…" : force ? "强制升级 (危险)" : "立即更新"}
          </Button>
          {running && (
            <Button size="sm" variant="outline" onClick={cancel}>
              取消
            </Button>
          )}
          <Button
            size="sm"
            variant="outline"
            disabled={recheck.isPending || running}
            onClick={() => recheck.mutate()}
          >
            <RefreshCw className={cn("size-3", recheck.isPending && "animate-spin")} />
            重新检查
          </Button>
          <a
            href={info?.release_url ?? "https://github.com/GALIAIS/LoraHub/releases"}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-[12px] text-muted-foreground hover:text-foreground"
          >
            <ExternalLink className="size-3" />
            打开 GitHub
          </a>
        </div>

        {events.length > 0 && (
          <div
            ref={logRef}
            className="rounded-[4px] border border-border/60 bg-muted/30 max-h-64 overflow-y-auto p-2 text-[11px] font-mono leading-relaxed"
          >
            {events.map((ev, i) => {
              const tone =
                ev.level === "error"
                  ? "text-destructive"
                  : ev.level === "warn"
                    ? "text-amber-700 dark:text-amber-400"
                    : ev.phase === "done" || ev.phase === "restart"
                      ? "text-emerald-700 dark:text-emerald-400"
                      : "text-muted-foreground"
              return (
                <div key={i} className={cn("flex gap-2", tone)}>
                  <span className="shrink-0 text-[10px] uppercase tracking-[0.1em] opacity-70">
                    {PHASE_LABEL[ev.phase] ?? ev.phase}
                  </span>
                  <span className="break-all">{ev.message}</span>
                </div>
              )
            })}
          </div>
        )}
      </CardContent>

      <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center gap-2">
              <AlertTriangle className="size-5 text-destructive" />
              确认强制升级
            </AlertDialogTitle>
            <AlertDialogDescription>
              此操作会执行 <code className="font-mono">git reset --hard</code> +{" "}
              <code className="font-mono">git clean -fd</code>,
              <strong className="text-destructive">
                丢弃工作树内全部本地修改 (含未跟踪文件)
              </strong>
              ,然后切到目标版本。已 commit 的提交不会丢失,但未提交修改不可恢复。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <div className="space-y-2 text-sm leading-relaxed">
            <p className="text-muted-foreground">
              受影响的典型路径:
            </p>
            <ul className="list-disc pl-5 text-xs text-muted-foreground space-y-0.5">
              <li>
                <code className="font-mono">configs/</code> 自定义训练配置
              </li>
              <li>
                <code className="font-mono">.env</code> 本地凭据 / 覆盖
              </li>
              <li>未跟踪的临时脚本、调试代码</li>
            </ul>
          </div>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setConfirmOpen(false)}>
              取消
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                setConfirmOpen(false)
                void runUpdate()
              }}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              我已了解,继续强制升级
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Card>
  )
}
