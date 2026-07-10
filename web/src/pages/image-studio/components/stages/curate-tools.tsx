/**
 * 整理类工具 — EXIF 自动旋转 / 批量缩放 / 隔离区 / 备份回滚。
 *
 * 隔离区直接复用 QuarantinePanel；其他三个轻量自实现。所有写入操作都
 * 自动备份到 .workbench/backups/，UI 上只在按钮 hover / 描述里提一下。
 */
import { useEffect, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Calendar,
  Loader2,
  RefreshCw,
  RotateCw,
  Trash2,
  Wrench,
} from "lucide-react"
import { toast } from "sonner"
import {
  imageStudioBackupsList,
  getImageStudioAutoRotateSession,
  getImageStudioBatchResizeSession,
  getLatestTask,
  imageStudioRestoreBackup,
  startImageStudioAutoRotate,
  startImageStudioBatchResize,
  stopImageStudioAutoRotateSession,
  stopImageStudioBatchResizeSession,
  type BackupEntry,
} from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { QuarantinePanel } from "../quarantine-panel"
import { cn } from "@/lib/utils"

// --------------------------------------------------------------------------- //
// curate-auto-rotate
// --------------------------------------------------------------------------- //

export function CurateAutoRotateTool({ datasetPath }: { datasetPath: string }) {
  const qc = useQueryClient()
  const [recursive, setRecursive] = useState(true)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [dismissedSessionId, setDismissedSessionId] = useState<string | null>(null)

  const latestRotateTask = useQuery({
    queryKey: ["tasks", "latest", "image_studio_auto_rotate"],
    queryFn: () => getLatestTask("image_studio_auto_rotate"),
    retry: false,
    staleTime: 10_000,
  })

  useEffect(() => {
    if (sessionId != null) return
    const latest = latestRotateTask.data
    if (
      latest?.metadata?.dataset_path === datasetPath &&
      latest.id !== dismissedSessionId
    ) {
      setSessionId(latest.id)
    }
  }, [datasetPath, dismissedSessionId, latestRotateTask.data, sessionId])

  const sessionQuery = useQuery({
    queryKey: ["image-studio", "auto-rotate-session", sessionId],
    queryFn: () => getImageStudioAutoRotateSession(sessionId!),
    enabled: sessionId != null,
    refetchInterval: (query) =>
      query.state.data?.status === "running" ||
      query.state.data?.status === "stop_requested"
        ? 1000
        : false,
  })

  const session = sessionQuery.data
  const running =
    session?.status === "running" || session?.status === "stop_requested"

  const mutation = useMutation({
    mutationFn: () =>
      startImageStudioAutoRotate({
        dataset_path: datasetPath,
        recursive,
      }),
    onSuccess: (data) => {
      setDismissedSessionId(null)
      setSessionId(data.session_id)
      toast.success(
        `已启动自动旋转：${data.total} 张`,
        { description: "后台进行，刷新后可恢复进度" },
      )
    },
    onError: (err) =>
      toast.error("自动旋转失败", {
        description: err instanceof Error ? err.message : String(err),
      }),
  })

  const stopMutation = useMutation({
    mutationFn: () => stopImageStudioAutoRotateSession(session!.session_id),
    onSuccess: () => {
      void sessionQuery.refetch()
    },
    onError: (err) =>
      toast.error("停止失败", {
        description: err instanceof Error ? err.message : String(err),
      }),
  })
  const stopping = session?.status === "stop_requested" || stopMutation.isPending

  return (
    <div className="h-full overflow-y-auto p-4 max-w-xl">
      <section className="rounded-md border border-border/60 bg-card flex flex-col">
        <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
          <RotateCw className="size-3.5" />
          <span className="text-xs font-medium">EXIF 自动旋转</span>
        </div>
        <div className="p-3 space-y-3 text-xs">
          <p className="text-muted-foreground">
            按 EXIF orientation 把像素旋转到正向，并清空 orientation 标记。
            原图自动备份到 <code>.workbench/backups/</code>，可在「备份回滚」工具还原。
          </p>
          <label className="inline-flex items-center gap-1.5 select-none">
            <Switch checked={recursive} onCheckedChange={setRecursive} />
            递归子目录
          </label>
          <Button
            size="sm"
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending || running}
            className="w-full gap-1"
          >
            {mutation.isPending || running ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <RotateCw className="size-3" />
            )}
            {running ? "自动旋转中…" : "对整个数据集应用 EXIF 旋转"}
          </Button>
          {session && (
            <div className="rounded-[4px] border border-border/60 bg-muted/25 p-2 text-[11px]">
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="font-medium">
                  {session.status === "succeeded"
                    ? "自动旋转完成"
                    : session.status === "failed"
                      ? "自动旋转失败"
                      : session.status === "stop_requested"
                        ? "正在停止自动旋转"
                        : session.status === "canceled"
                          ? "自动旋转已停止"
                          : "自动旋转进行中"}
                </span>
                <span className="font-mono text-muted-foreground">
                  {session.processed} / {session.total}
                </span>
              </div>
              <div className="shiro-progress-track h-1.5 border-0 bg-muted">
                <div
                  className={cn(
                    "shiro-progress-fill",
                    session.status === "failed"
                      ? "bg-destructive"
                      : "bg-primary",
                  )}
                  style={{ width: `${Math.max(0, Math.min(100, session.percent))}%` }}
                />
              </div>
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-muted-foreground">
                <span>已旋转 {session.rotated_count}</span>
                <span>跳过 {session.skipped_count}</span>
                <span>失败 {session.failed.length}</span>
                {session.last_image && <span>最近 {session.last_image}</span>}
              </div>
              {session.error && (
                <div className="mt-1 text-destructive">{session.error}</div>
              )}
              {session.status === "running" || session.status === "stop_requested" ? (
                <Button
                  size="sm"
                  variant="outline"
                  className="mt-2 h-7 text-[11px] text-destructive"
                  onClick={() => stopMutation.mutate()}
                  disabled={stopping}
                >
                  {stopping ? "停止中..." : "停止"}
                </Button>
              ) : (
                <Button
                  size="sm"
                  variant="outline"
                  className="mt-2 h-7 text-[11px]"
                  onClick={() => {
                    setDismissedSessionId(session.session_id)
                    setSessionId(null)
                    qc.invalidateQueries({ queryKey: ["image-studio"] })
                    qc.invalidateQueries({
                      queryKey: ["tasks", "latest", "image_studio_auto_rotate"],
                    })
                  }}
                >
                  关闭结果
                </Button>
              )}
            </div>
          )}
        </div>
      </section>
    </div>
  )
}

// --------------------------------------------------------------------------- //
// curate-batch-resize
// --------------------------------------------------------------------------- //

export function CurateBatchResizeTool({ datasetPath }: { datasetPath: string }) {
  const qc = useQueryClient()
  const [shortEdge, setShortEdge] = useState("1024")
  const [filter, setFilter] = useState<"lanczos" | "bicubic" | "bilinear">("lanczos")
  const [upscale, setUpscale] = useState(false)
  const [recursive, setRecursive] = useState(true)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [dismissedSessionId, setDismissedSessionId] = useState<string | null>(null)

  const latestResizeTask = useQuery({
    queryKey: ["tasks", "latest", "image_studio_batch_resize"],
    queryFn: () => getLatestTask("image_studio_batch_resize"),
    retry: false,
    staleTime: 10_000,
  })

  useEffect(() => {
    if (sessionId != null) return
    const latest = latestResizeTask.data
    if (
      latest?.metadata?.dataset_path === datasetPath &&
      latest.id !== dismissedSessionId
    ) {
      setSessionId(latest.id)
    }
  }, [datasetPath, dismissedSessionId, latestResizeTask.data, sessionId])

  const sessionQuery = useQuery({
    queryKey: ["image-studio", "batch-resize-session", sessionId],
    queryFn: () => getImageStudioBatchResizeSession(sessionId!),
    enabled: sessionId != null,
    refetchInterval: (query) =>
      query.state.data?.status === "running" ||
      query.state.data?.status === "stop_requested"
        ? 1000
        : false,
  })

  const session = sessionQuery.data
  const running =
    session?.status === "running" || session?.status === "stop_requested"

  const mutation = useMutation({
    mutationFn: () =>
      startImageStudioBatchResize({
        dataset_path: datasetPath,
        target_short_edge: Number(shortEdge),
        filter,
        upscale,
        recursive,
      }),
    onSuccess: (data) => {
      setDismissedSessionId(null)
      setSessionId(data.session_id)
      toast.success(
        `已启动批量缩放：${data.total} 张`,
        { description: "后台进行，刷新后可恢复进度" },
      )
    },
    onError: (err) =>
      toast.error("批量缩放失败", {
        description: err instanceof Error ? err.message : String(err),
      }),
  })

  const targetOk = Number(shortEdge) >= 128 && Number(shortEdge) <= 4096

  const stopMutation = useMutation({
    mutationFn: () => stopImageStudioBatchResizeSession(session!.session_id),
    onSuccess: () => {
      void sessionQuery.refetch()
    },
    onError: (err) =>
      toast.error("停止失败", {
        description: err instanceof Error ? err.message : String(err),
      }),
  })
  const stopping = session?.status === "stop_requested" || stopMutation.isPending

  return (
    <div className="h-full overflow-y-auto p-4 max-w-xl">
      <section className="rounded-md border border-border/60 bg-card flex flex-col">
        <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
          <Wrench className="size-3.5" />
          <span className="text-xs font-medium">批量缩放</span>
        </div>
        <div className="p-3 space-y-3 text-xs">
          <p className="text-muted-foreground">
            按目标短边重采样保持原长宽比。原图自动备份到{" "}
            <code>.workbench/backups/</code>。短于目标短边的图默认跳过 —
            勾「向上采样」才会放大。
          </p>
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground w-16">目标短边</span>
            <Input
              type="number"
              value={shortEdge}
              onChange={(e) => setShortEdge(e.target.value)}
              className="h-8 w-28 text-xs font-mono"
              min={128}
              max={4096}
            />
            <span className="text-muted-foreground">px</span>
            {!targetOk && (
              <span className="text-red-600 text-[10px] ml-auto">
                需在 128–4096 之间
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground w-16">重采样</span>
            <Select
              value={filter}
              onValueChange={(v) => v && setFilter(v as typeof filter)}
            >
              <SelectTrigger className="h-8 w-44 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="lanczos">Lanczos（默认 · 锐）</SelectItem>
                <SelectItem value="bicubic">Bicubic（更柔）</SelectItem>
                <SelectItem value="bilinear">Bilinear（最快）</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            <label className="inline-flex items-center gap-1.5 select-none">
              <Switch checked={upscale} onCheckedChange={setUpscale} />
              向上采样（短于目标的也放大）
            </label>
            <label className="inline-flex items-center gap-1.5 select-none">
              <Switch checked={recursive} onCheckedChange={setRecursive} />
              递归子目录
            </label>
          </div>
          <Button
            size="sm"
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending || running || !targetOk}
            className="w-full gap-1"
          >
            {mutation.isPending || running ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <Wrench className="size-3" />
            )}
            {running ? "批量缩放中…" : "应用批量缩放"}
          </Button>
          {session && (
            <div className="rounded-[4px] border border-border/60 bg-muted/25 p-2 text-[11px]">
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="font-medium">
                  {session.status === "succeeded"
                    ? "批量缩放完成"
                    : session.status === "failed"
                      ? "批量缩放失败"
                      : session.status === "stop_requested"
                        ? "正在停止批量缩放"
                        : session.status === "canceled"
                          ? "批量缩放已停止"
                          : "批量缩放进行中"}
                </span>
                <span className="font-mono text-muted-foreground">
                  {session.processed} / {session.total}
                </span>
              </div>
              <div className="shiro-progress-track h-1.5 border-0 bg-muted">
                <div
                  className={cn(
                    "shiro-progress-fill",
                    session.status === "failed"
                      ? "bg-destructive"
                      : "bg-primary",
                  )}
                  style={{ width: `${Math.max(0, Math.min(100, session.percent))}%` }}
                />
              </div>
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-muted-foreground">
                <span>已重采样 {session.resampled_count}</span>
                <span>跳过 {session.skipped_count}</span>
                <span>失败 {session.failed.length}</span>
                {session.last_image && <span>最近 {session.last_image}</span>}
              </div>
              {session.error && (
                <div className="mt-1 text-destructive">{session.error}</div>
              )}
              {session.status === "running" || session.status === "stop_requested" ? (
                <Button
                  size="sm"
                  variant="outline"
                  className="mt-2 h-7 text-[11px] text-destructive"
                  onClick={() => stopMutation.mutate()}
                  disabled={stopping}
                >
                  {stopping ? "停止中..." : "停止"}
                </Button>
              ) : (
                <Button
                  size="sm"
                  variant="outline"
                  className="mt-2 h-7 text-[11px]"
                  onClick={() => {
                    setDismissedSessionId(session.session_id)
                    setSessionId(null)
                    qc.invalidateQueries({ queryKey: ["image-studio"] })
                    qc.invalidateQueries({
                      queryKey: ["tasks", "latest", "image_studio_batch_resize"],
                    })
                  }}
                >
                  关闭结果
                </Button>
              )}
            </div>
          )}
        </div>
      </section>
    </div>
  )
}

// --------------------------------------------------------------------------- //
// curate-quarantine — 直接套用 QuarantinePanel(默认折叠的 drawer，给独立页加个抬头)
// --------------------------------------------------------------------------- //

export function CurateQuarantineTool({ datasetPath }: { datasetPath: string }) {
  return (
    <div className="h-full overflow-y-auto p-4 space-y-3">
      <div className="rounded-md border bg-muted/30 px-3 py-2 text-[11px] text-muted-foreground">
        这里管理已经移到 <code>.workbench/quarantine/</code> 的图片。
        要把图<span className="font-medium">移入</span>隔离区，去「数据集体检」工具里
        按异常类型批量隔离，或在「整理总览」工具的网格右键单张隔离。
      </div>
      <QuarantinePanel datasetPath={datasetPath} />
    </div>
  )
}

// --------------------------------------------------------------------------- //
// curate-restore-backup
// --------------------------------------------------------------------------- //

export function CurateRestoreBackupTool({
  datasetPath,
}: {
  datasetPath: string
}) {
  const qc = useQueryClient()
  const [selected, setSelected] = useState<Set<string>>(new Set())

  const listQuery = useQuery({
    queryKey: ["image-studio-backups", datasetPath],
    queryFn: () => imageStudioBackupsList(datasetPath),
    enabled: Boolean(datasetPath),
  })

  const restoreMutation = useMutation({
    mutationFn: (paths: string[]) =>
      imageStudioRestoreBackup({ dataset_path: datasetPath, backup_paths: paths }),
    onSuccess: (data) => {
      toast.success(`已恢复 ${data.restored_count} 个文件`, {
        description:
          data.failed.length > 0 ? `失败 ${data.failed.length} 个` : undefined,
      })
      setSelected(new Set())
      qc.invalidateQueries({ queryKey: ["image-studio-backups", datasetPath] })
      qc.invalidateQueries({ queryKey: ["image-studio"] })
    },
    onError: (err) =>
      toast.error("恢复失败", {
        description: err instanceof Error ? err.message : String(err),
      }),
  })

  const entries = listQuery.data?.entries ?? []
  const toggle = (p: string) =>
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(p)) next.delete(p)
      else next.add(p)
      return next
    })

  return (
    <div className="h-full overflow-hidden p-4">
      <section className="rounded-md border border-border/60 bg-card flex flex-col h-full min-h-0">
        <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
          <Calendar className="size-3.5" />
          <span className="text-xs font-medium">备份回滚</span>
          <span className="text-[11px] text-muted-foreground tabular-nums">
            {entries.length} 项备份
          </span>
          <div className="ml-auto flex items-center gap-1">
            <Button
              size="sm"
              variant="ghost"
              className="h-6 px-2 text-[11px]"
              onClick={() => listQuery.refetch()}
              disabled={listQuery.isFetching}
            >
              {listQuery.isFetching ? (
                <Loader2 className="size-3 animate-spin" />
              ) : (
                <RefreshCw className="size-3" />
              )}
            </Button>
            {selected.size > 0 && (
              <Button
                size="sm"
                variant="outline"
                className="h-6 px-2 text-[11px] gap-1"
                onClick={() => {
                  if (
                    !window.confirm(
                      `确认恢复这 ${selected.size} 个备份？\n会覆盖当前同路径文件。`,
                    )
                  )
                    return
                  restoreMutation.mutate(Array.from(selected))
                }}
                disabled={restoreMutation.isPending}
              >
                {restoreMutation.isPending ? (
                  <Loader2 className="size-3 animate-spin" />
                ) : null}
                恢复选中 ({selected.size})
              </Button>
            )}
          </div>
        </div>
        <div className="flex-1 overflow-y-auto">
          {listQuery.isLoading ? (
            <div className="flex items-center justify-center h-32 text-muted-foreground">
              <Loader2 className="size-4 animate-spin mr-2" />
              加载备份列表…
            </div>
          ) : entries.length === 0 ? (
            <div className="flex items-center justify-center h-32 text-muted-foreground text-xs">
              暂无备份。整理 / caption 写操作后会自动生成。
            </div>
          ) : (
            <ul className="divide-y divide-border/30">
              {entries.map((e) => (
                <BackupRow
                  key={e.backup_path}
                  entry={e}
                  selected={selected.has(e.backup_path)}
                  onToggle={() => toggle(e.backup_path)}
                />
              ))}
            </ul>
          )}
        </div>
      </section>
    </div>
  )
}

function BackupRow({
  entry,
  selected,
  onToggle,
}: {
  entry: BackupEntry
  selected: boolean
  onToggle: () => void
}) {
  const filename = entry.relative_path.split(/[\\/]/).pop() ?? "?"
  const sizeKb = (entry.size / 1024).toFixed(1)
  return (
    <li
      className={cn(
        "flex items-center gap-2 px-3 py-1.5 text-[11px] cursor-pointer hover:bg-muted/40",
        selected && "bg-amber-50/40 dark:bg-amber-950/15",
      )}
      onClick={onToggle}
    >
      <input
        type="checkbox"
        checked={selected}
        onChange={onToggle}
        className="size-3"
        onClick={(e) => e.stopPropagation()}
      />
      <span className="font-mono truncate flex-1" title={entry.relative_path}>
        {filename}
        <span className="text-muted-foreground ml-1.5 text-[10px]">
          {entry.relative_path !== filename ? entry.relative_path : ""}
        </span>
      </span>
      <span className="tabular-nums text-muted-foreground text-[10px]">
        {sizeKb} KB
      </span>
      <span className="tabular-nums text-muted-foreground text-[10px]">
        {new Date(entry.mtime * 1000).toLocaleString()}
      </span>
    </li>
  )
}

// 占位避免 ESLint 警告（一些 icon 暂未用，但保留以便后续扩展）
void Trash2
