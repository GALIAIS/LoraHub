/**
 * Self-update card — version check + one-click upgrade.
 *
 * The card mirrors ShiroManager's update-status panel:
 *  - Shows the running version plus formal and development update targets.
 *  - Update targets are one-shot actions, not a persisted branch preference.
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
  GitBranch,
  History,
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
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
const MAX_VISIBLE_UPDATE_EVENTS = 2000

interface RequestedUpdate {
  channel: UpdateChannel
  label: string
  targetTag?: string
}

interface UpdateTargetRowProps {
  title: string
  description: string
  version: string
  detail: string
  tone: string
  updateAvailable: boolean
  loading: boolean
  error?: string | null
  disabled: boolean
  buttonLabel: string
  onApply: () => void
}

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
  return events.slice(-MAX_VISIBLE_UPDATE_EVENTS).map((event) => {
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

function appendUpdateEvent(events: UpdateEvent[], event: UpdateEvent): UpdateEvent[] {
  return [...events, event].slice(-MAX_VISIBLE_UPDATE_EVENTS)
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

function currentVersionTone(
  info: ReturnType<typeof useSystemVersion>["data"],
): string {
  return /(^|[-.])dev([-.]|$)/i.test(info?.current ?? "")
    ? channelVersionTone("dev")
    : channelVersionTone("tag")
}

function remoteDetailLabel(
  info: ReturnType<typeof useSystemVersion>["data"],
): string {
  return info?.latest_commit?.slice(0, 7) ?? info?.tag_name ?? info?.latest ?? "—"
}

function UpdateTargetRow({
  title,
  description,
  version,
  detail,
  tone,
  updateAvailable,
  loading,
  error,
  disabled,
  buttonLabel,
  onApply,
}: UpdateTargetRowProps) {
  const actionLabel = loading
    ? "检查中…"
    : error
      ? "检查失败"
      : updateAvailable
        ? buttonLabel
        : "已是最新"
  const showDetail = detail !== "—" && detail !== version

  return (
    <div className="grid gap-3 px-3 py-3 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-center">
      <div className="min-w-0">
        <div className="flex min-w-0 flex-wrap items-baseline gap-x-2 gap-y-1">
          <span className="text-[13px] font-medium">{title}</span>
          <code className={cn("font-mono text-[11px] font-medium", tone)}>
            {version}
          </code>
          {showDetail && (
            <span className="font-mono text-[11px] text-muted-foreground">
              {detail}
            </span>
          )}
        </div>
        <p className="mt-0.5 text-[11px] text-muted-foreground">{description}</p>
      </div>
      <Button
        size="sm"
        variant={updateAvailable && !error ? "default" : "outline"}
        className="w-full sm:w-auto"
        disabled={disabled || loading || Boolean(error) || !updateAvailable}
        onClick={onApply}
        title={error ?? undefined}
      >
        {loading ? (
          <Loader2 className="size-3 animate-spin" />
        ) : updateAvailable && !error ? (
          <Download className="size-3" />
        ) : error ? (
          <AlertTriangle className="size-3" />
        ) : (
          <CheckCircle className="size-3" />
        )}
        {actionLabel}
      </Button>
    </div>
  )
}

export function UpdateCard() {
  const qc = useQueryClient()
  const tagVersion = useSystemVersion("tag")
  const devVersion = useSystemVersion("dev")
  const version = tagVersion
  const [restart, setRestart] = useState(true)
  const [build, setBuild] = useState(true)
  // ``force`` discards local changes (git reset --hard + clean -fd)
  // before checkout. Destructive — guarded by the AlertDialog below.
  const [force, setForce] = useState(false)
  // Force updates and release rollbacks require explicit confirmation.
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [requestedUpdate, setRequestedUpdate] = useState<RequestedUpdate | null>(null)
  const [rollbackTag, setRollbackTag] = useState("")
  const [events, setEvents] = useState<UpdateEvent[]>([])
  const [running, setRunning] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const logRef = useRef<HTMLDivElement | null>(null)

  useEffect(
    () => () => {
      // Disconnecting the browser stream does not cancel the transactional
      // updater. Its durable task record lets this page reconnect later.
      abortRef.current?.abort()
    },
    [],
  )

  const latestUpdateTask = useQuery({
    queryKey: ["tasks", "latest", "system_update"],
    queryFn: () => api.getLatestTask("system_update"),
    retry: false,
    staleTime: 10_000,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status && ACTIVE_UPDATE_TASK_STATUSES.has(status) ? 3000 : false
    },
    throwOnError: false,
  })

  const releases = useQuery({
    queryKey: ["system-releases"],
    queryFn: () => api.listSystemReleases(6),
    staleTime: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
  })

  useEffect(() => {
    if (!rollbackTag && releases.data?.releases[0]) {
      setRollbackTag(releases.data.releases[0].tag_name)
    }
  }, [releases.data?.releases, rollbackTag])

  // Auto-scroll the log to the latest line as events stream in.
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [events.length])

  useEffect(() => {
    // The active page-owned SSE resolves through runUpdate.finally. This
    // effect only reconciles a durable task recovered after navigation or a
    // full page reload.
    if (abortRef.current) return
    const task = latestUpdateTask.data
    if (!task) return
    const active = ACTIVE_UPDATE_TASK_STATUSES.has(task.status)
    if (active) {
      if (task.events.length > 0) {
        setEvents(taskEventsToUpdateEvents(task.events))
      }
      if (!running) setRunning(true)
      return
    }
    if (running) {
      if (task.events.length > 0) {
        setEvents(taskEventsToUpdateEvents(task.events))
      }
      setRunning(false)
    }
  }, [events.length, latestUpdateTask.data, running])

  const recheck = useMutation({
    mutationFn: async () => {
      const [tag, dev] = await Promise.all([
        api.getSystemVersion("tag", true),
        api.getSystemVersion("dev", true),
      ])
      return { tag, dev }
    },
    onSuccess: ({ tag, dev }) => {
      qc.setQueryData(["system-version", "tag"], tag)
      qc.setQueryData(["system-version", "dev"], dev)
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ["system-releases"] })
    },
  })

  const runUpdate = async (target: RequestedUpdate) => {
    setEvents([])
    setRunning(true)
    const ac = new AbortController()
    abortRef.current = ac
    try {
      await api.applySystemUpdate(
        {
          channel: target.channel,
          build,
          restart,
          force,
          target_tag: target.targetTag,
        },
        (ev) => setEvents((prev) => appendUpdateEvent(prev, ev)),
        ac.signal,
      )
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : String(exc)
      if (!(exc instanceof DOMException && exc.name === "AbortError")) {
        setEvents((prev) =>
          appendUpdateEvent(prev, {
            phase: "error",
            level: "error",
            message,
          }),
        )
      }
    } finally {
      setRunning(false)
      setRequestedUpdate(null)
      abortRef.current = null
      qc.invalidateQueries({ queryKey: ["system-version"] })
      qc.invalidateQueries({ queryKey: ["tasks", "latest", "system_update"] })
    }
  }

  // Force updates and release rollbacks require explicit confirmation.
  const onApplyClick = (target: RequestedUpdate) => {
    setRequestedUpdate(target)
    if (force || target.targetTag) {
      setConfirmOpen(true)
      return
    }
    void runUpdate(target)
  }

  const info = version.data
  const versionTone = currentVersionTone(info)
  const checkedAt = info?.checked_at
    ? new Date(info.checked_at).toLocaleString()
    : "—"
  const isDockerInstall = info?.install_kind === "docker"
  const selectedRelease = releases.data?.releases.find(
    (release) => release.tag_name === rollbackTag,
  )
  const rollbackIsCurrent = Boolean(
    selectedRelease?.commit &&
      info?.current_commit === selectedRelease.commit,
  )
  const confirmIsRollback = Boolean(requestedUpdate?.targetTag)

  const headerStatus = (() => {
    if (!info) return null
    if (info.error || devVersion.data?.error) {
      return (
        <Badge variant="outline" className="rounded-[2px] gap-1 text-amber-700 dark:text-amber-400 border-amber-500/40">
          <AlertTriangle className="size-3" />
          网络异常
        </Badge>
      )
    }
    if (info.update_available || devVersion.data?.update_available) {
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
              选择本次更新目标，或切换到近期正式版本。
            </CardDescription>
          </div>
          {headerStatus}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-[7rem_1fr] gap-x-4 gap-y-2 text-sm">
          <span className="text-muted-foreground">当前版本</span>
          <span className="flex items-center gap-2 flex-wrap">
            <code className={cn("font-mono text-[12px] font-medium", versionTone)}>
              {currentDetailLabel(info)}
            </code>
            {info?.version_source && info.version_source !== "hatch-vcs" && (
              <Badge
                variant="outline"
                className="rounded-[2px] text-[10px] tracking-wide border-amber-500/40 text-amber-700 dark:text-amber-400"
                title={
                  info.version_source === "fallback"
                    ? "无法读取版本元数据,使用回退占位"
                    : info.version_source === "env"
                      ? "从运行环境读取的版本"
                      : info.version_source === "changelog"
                      ? "从 CHANGELOG.md 读取的最近发布版本(可能落后于 commit)"
                      : "从 dist 元数据读取(非 git 检出)"
                }
              >
                {info.version_source === "fallback"
                  ? "未知"
                  : info.version_source === "changelog"
                    ? "估算"
                    : info.version_source === "env"
                      ? "镜像"
                      : "已安装"}
              </Badge>
            )}
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
                    <code className="font-mono text-[11px] mx-1">docker compose pull</code>
                    后重新
                    <code className="font-mono text-[11px] mx-1">docker compose up -d</code>
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

        <section className="space-y-2">
          <div className="flex items-center gap-2">
            <GitBranch className="size-3.5 text-muted-foreground" />
            <h3 className="text-[12px] font-medium">本次更新目标</h3>
          </div>
          <div className="divide-y overflow-hidden rounded-[6px] border border-border/70">
            <UpdateTargetRow
              title="正式更新"
              description="稳定发布与后续修复"
              version={channelVersionLabel(tagVersion.data, "tag")}
              detail={remoteDetailLabel(tagVersion.data)}
              tone="text-emerald-700 dark:text-emerald-400"
              updateAvailable={Boolean(tagVersion.data?.update_available)}
              loading={tagVersion.isLoading || recheck.isPending}
              error={tagVersion.data?.error}
              disabled={running || info?.git_checkout === false}
              buttonLabel="更新到正式版本"
              onApply={() =>
                onApplyClick({ channel: "tag", label: "正式版本" })
              }
            />
            <UpdateTargetRow
              title="Dev 预览"
              description="最新开发提交，仅用于提前测试"
              version={channelVersionLabel(devVersion.data, "dev")}
              detail={remoteDetailLabel(devVersion.data)}
              tone="text-sky-700 dark:text-sky-400"
              updateAvailable={Boolean(devVersion.data?.update_available)}
              loading={devVersion.isLoading || recheck.isPending}
              error={devVersion.data?.error}
              disabled={running || info?.git_checkout === false}
              buttonLabel="更新到 Dev"
              onApply={() =>
                onApplyClick({ channel: "dev", label: "Dev 预览" })
              }
            />
          </div>
        </section>

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

        <section className="rounded-[6px] border border-border/70 p-3">
          <div className="flex items-start gap-2">
            <History className="mt-0.5 size-3.5 text-muted-foreground" />
            <div className="min-w-0 flex-1">
              <h3 className="text-[12px] font-medium">版本回退</h3>
              <p className="mt-0.5 text-[11px] text-muted-foreground">
                切换到近期正式版本，并重新安装依赖、构建前端和重启服务。
              </p>
            </div>
          </div>
          <div className="mt-3 grid gap-2 sm:grid-cols-[minmax(0,14rem)_auto] sm:items-center">
            <Select
              value={rollbackTag}
              disabled={running || releases.isFetching}
              onValueChange={(value) => value && setRollbackTag(value)}
            >
              <SelectTrigger
                size="sm"
                className="w-full font-mono text-xs"
                aria-label="选择回退版本"
              >
                <SelectValue placeholder={releases.isLoading ? "读取版本…" : "选择版本"} />
              </SelectTrigger>
              <SelectContent>
                {(releases.data?.releases ?? []).map((release) => (
                  <SelectItem key={release.tag_name} value={release.tag_name}>
                    {release.tag_name} · {release.commit?.slice(0, 7) ?? "未知"}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              size="sm"
              variant="outline"
              className="w-full sm:w-auto"
              disabled={
                running ||
                releases.isFetching ||
                !selectedRelease ||
                rollbackIsCurrent ||
                info?.git_checkout === false
              }
              onClick={() =>
                selectedRelease &&
                onApplyClick({
                  channel: "tag",
                  label: selectedRelease.tag_name,
                  targetTag: selectedRelease.tag_name,
                })
              }
            >
              <History className="size-3" />
              {rollbackIsCurrent ? "当前版本" : "切换到此版本"}
            </Button>
          </div>
          {releases.isError && (
            <p className="mt-2 text-[11px] text-amber-700 dark:text-amber-400">
              无法读取版本历史，请重新检查网络后重试。
            </p>
          )}
        </section>

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

        <div className="flex flex-wrap items-center gap-2">
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

      <AlertDialog
        open={confirmOpen}
        onOpenChange={(open) => {
          setConfirmOpen(open)
          if (!open) setRequestedUpdate(null)
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center gap-2">
              {force ? (
                <AlertTriangle className="size-5 text-destructive" />
              ) : (
                <History className="size-5 text-muted-foreground" />
              )}
              {force
                ? `确认强制更新到${requestedUpdate?.label ?? "目标版本"}`
                : `确认切换到 ${requestedUpdate?.label ?? "目标版本"}`}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {force ? (
                <>
                  将重置未提交的源码修改后切换版本。训练配置、数据集、模型、运行记录、
                  输出和环境配置会保留。
                </>
              ) : (
                <>
                  将代码、Python 依赖和前端恢复到该正式版本，完成后按当前设置重启服务。
                  用户数据与训练配置不会回退。
                </>
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          {force && (
            <p className="text-xs leading-relaxed text-muted-foreground">
              未提交且不在受保护数据目录内的源码、脚本和临时文件将被删除，无法恢复。
            </p>
          )}
          <AlertDialogFooter>
            <AlertDialogCancel>
              取消
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                const target = requestedUpdate
                if (target) void runUpdate(target)
              }}
              className={cn(
                force &&
                  "bg-destructive text-destructive-foreground hover:bg-destructive/90",
              )}
            >
              {force
                ? "强制更新"
                : confirmIsRollback
                  ? "确认回退"
                  : "确认更新"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Card>
  )
}
