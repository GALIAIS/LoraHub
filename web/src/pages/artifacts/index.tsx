/**
 * 产物管理 — flat list of every job's training artifacts with bulk
 * download / delete. Sister page to /jobs (job-level lifecycle) and
 * /analysis (per-job metrics drill-down); this one focuses on the
 * shippable LoRA / checkpoint files.
 *
 * Per row:
 *   * checkpoint count + sample count + total size
 *   * "下载全部 (zip)" — streams `/api/artifacts/{id}/zip?include=...`
 *   * Per-checkpoint download / delete buttons
 *   * "删除整个工作区" — destructive, refuses non-terminal jobs
 *     server-side (we still gate the button client-side).
 */
import { useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Download,
  FileArchive,
  FolderOpen,
  Loader2,
  Package,
  RefreshCw,
  Trash2,
} from "lucide-react"
import { toast } from "sonner"
import { api, ApiError, type ArtifactRow } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
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
import { Input } from "@/components/ui/input"

const TERMINAL_STATES = new Set([
  "succeeded",
  "failed",
  "canceled",
  "interrupted",
])

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B"
  const units = ["B", "KB", "MB", "GB", "TB"]
  let v = bytes
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : v >= 10 ? 1 : 2)} ${units[i]}`
}

function formatTime(iso: string | null | undefined): string {
  if (!iso) return "—"
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

export function ArtifactsPage() {
  const qc = useQueryClient()
  const list = useQuery({
    queryKey: ["artifacts"],
    queryFn: api.listArtifacts,
    staleTime: 5_000,
  })
  const [filter, setFilter] = useState("")
  const [confirmDelete, setConfirmDelete] = useState<ArtifactRow | null>(null)

  const rows = useMemo(() => {
    const all = list.data?.jobs ?? []
    if (!filter.trim()) return all
    const q = filter.trim().toLowerCase()
    return all.filter(
      (r) =>
        r.job_id.toLowerCase().includes(q) ||
        (r.output_name ?? "").toLowerCase().includes(q) ||
        r.workspace.toLowerCase().includes(q),
    )
  }, [list.data, filter])

  const totalBytes = useMemo(
    () => rows.reduce((acc, r) => acc + (r.total_bytes ?? 0), 0),
    [rows],
  )
  const totalCheckpoints = useMemo(
    () => rows.reduce((acc, r) => acc + (r.checkpoint_count ?? 0), 0),
    [rows],
  )

  const refresh = () => qc.invalidateQueries({ queryKey: ["artifacts"] })

  const deleteWorkspace = useMutation({
    mutationFn: (jobId: string) => api.deleteArtifactWorkspace(jobId),
    onSuccess: (data, jobId) => {
      if (data.deleted) {
        toast.success(`已删除工作区:${data.workspace ?? jobId}`)
      } else {
        toast.info(data.reason ?? "工作区已不存在,记录已清理")
      }
      qc.invalidateQueries({ queryKey: ["artifacts"] })
      qc.invalidateQueries({ queryKey: ["jobs"] })
    },
    onError: (e) => {
      const detail = e instanceof ApiError ? e.message : String(e)
      toast.error("删除失败", { description: detail })
    },
  })

  const deleteFile = useMutation({
    mutationFn: ({ jobId, path }: { jobId: string; path: string }) =>
      api.deleteArtifactFile(jobId, path),
    onSuccess: (data, vars) => {
      toast.success(`已删除 ${vars.path} (${formatBytes(data.size_bytes)})`)
      qc.invalidateQueries({ queryKey: ["artifacts"] })
      qc.invalidateQueries({ queryKey: ["job-files", vars.jobId] })
    },
    onError: (e) => {
      const detail = e instanceof ApiError ? e.message : String(e)
      toast.error("删除文件失败", { description: detail })
    },
  })

  return (
    <div className="h-full overflow-y-auto">
      <div className="px-8 py-7 space-y-5 w-full">
        <header className="space-y-1">
          <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
            产物归档
          </div>
          <h1 className="text-2xl font-semibold tracking-tight flex items-center gap-2">
            <Package className="size-5 text-muted-foreground" />
            训练产物
          </h1>
          <p className="text-sm text-muted-foreground">
            跨任务汇总 LoRA / checkpoint 文件,支持单独下载、批量打包,或删除整个工作区。
          </p>
        </header>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div className="space-y-0.5">
                <CardTitle className="text-base">总览</CardTitle>
                <CardDescription>
                  {rows.length} 个 job · {totalCheckpoints} 个 checkpoint ·{" "}
                  {formatBytes(totalBytes)}
                </CardDescription>
              </div>
              <div className="flex items-center gap-2">
                <Input
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  placeholder="按 job id / output 名 / 路径过滤"
                  className="h-8 w-64 font-mono text-xs"
                />
                <Button
                  size="sm"
                  variant="outline"
                  onClick={refresh}
                  disabled={list.isFetching}
                  className="gap-1"
                >
                  {list.isFetching ? (
                    <Loader2 className="size-3 animate-spin" />
                  ) : (
                    <RefreshCw className="size-3" />
                  )}
                  刷新
                </Button>
              </div>
            </div>
          </CardHeader>
        </Card>

        {list.isError && (
          <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-4 py-3 text-xs font-mono text-destructive">
            加载产物列表失败:{(list.error as Error).message}
          </div>
        )}

        {!list.isLoading && rows.length === 0 && (
          <div className="rounded-[4px] border border-dashed border-border/70 bg-muted/30 px-4 py-12 text-center text-sm text-muted-foreground">
            没有匹配的训练任务产物。
          </div>
        )}

        <div className="space-y-3">
          {rows.map((row) => (
            <ArtifactCard
              key={row.job_id}
              row={row}
              onZipDownload={(include) => {
                const url = api.artifactZipUrl(row.job_id, include)
                window.open(url, "_blank", "noopener,noreferrer")
              }}
              onFileDownload={(path) => {
                const url = api.artifactSingleUrl(row.job_id, path)
                window.open(url, "_blank", "noopener,noreferrer")
              }}
              onFileDelete={(path) =>
                deleteFile.mutate({ jobId: row.job_id, path })
              }
              onWorkspaceDelete={() => setConfirmDelete(row)}
              fileBusy={deleteFile.isPending}
            />
          ))}
        </div>
      </div>

      <AlertDialog
        open={confirmDelete !== null}
        onOpenChange={(open) => !open && setConfirmDelete(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>删除整个工作区</AlertDialogTitle>
            <AlertDialogDescription className="space-y-2">
              <span className="block">
                即将物理删除 <code className="font-mono text-xs">{confirmDelete?.workspace}</code>。
              </span>
              <span className="block">
                此操作不可恢复 —— 不同于「归档」,这里不会移动到 _archive/,
                而是直接 rmtree 整个目录,并从任务列表中移除该记录。
              </span>
              <span className="block text-amber-700 dark:text-amber-400">
                ⚠ {confirmDelete?.checkpoint_count ?? 0} 个 checkpoint
                共计 {formatBytes(confirmDelete?.total_bytes ?? 0)} 将被删除。
              </span>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleteWorkspace.isPending}>
              取消
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (confirmDelete) deleteWorkspace.mutate(confirmDelete.job_id)
                setConfirmDelete(null)
              }}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              disabled={deleteWorkspace.isPending}
            >
              {deleteWorkspace.isPending ? "删除中…" : "永久删除"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}

interface ArtifactCardProps {
  row: ArtifactRow
  onZipDownload: (include: string[]) => void
  onFileDownload: (path: string) => void
  onFileDelete: (path: string) => void
  onWorkspaceDelete: () => void
  fileBusy: boolean
}

function ArtifactCard({
  row,
  onZipDownload,
  onFileDownload,
  onFileDelete,
  onWorkspaceDelete,
  fileBusy,
}: ArtifactCardProps) {
  const isTerminal = TERMINAL_STATES.has(row.state)
  const stateBadgeTone =
    row.state === "succeeded"
      ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border-emerald-500/40"
      : row.state === "failed"
        ? "bg-destructive/15 text-destructive border-destructive/40"
        : row.state === "running" || row.state === "preparing"
          ? "bg-blue-500/15 text-blue-700 dark:text-blue-400 border-blue-500/40"
          : "bg-muted text-muted-foreground border-border"

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <CardTitle className="text-sm flex items-center gap-2 flex-wrap">
              <code className="font-mono text-[12px]">
                {row.output_name ?? row.job_id.slice(-8)}
              </code>
              <Badge
                variant="outline"
                className={`rounded-[2px] text-[10px] tracking-wide ${stateBadgeTone}`}
              >
                {row.state}
              </Badge>
              {!row.exists && (
                <Badge
                  variant="outline"
                  className="rounded-[2px] text-[10px] tracking-wide border-amber-500/40 text-amber-700 dark:text-amber-400"
                >
                  目录已丢失
                </Badge>
              )}
            </CardTitle>
            <CardDescription className="font-mono text-[11px] break-all mt-1">
              {row.workspace}
            </CardDescription>
            <CardDescription className="text-[11px] mt-0.5">
              结束于 {formatTime(row.finished_at)} · {row.checkpoint_count} 个
              checkpoint · {formatBytes(row.total_bytes)}
            </CardDescription>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <Button
              size="sm"
              variant="outline"
              onClick={() => onZipDownload(["checkpoints"])}
              disabled={!row.exists || row.checkpoint_count === 0}
              className="gap-1 h-8"
            >
              <FileArchive className="size-3" />
              下载 checkpoints (zip)
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => onZipDownload(["checkpoints", "samples"])}
              disabled={!row.exists || row.checkpoint_count + row.sample_count === 0}
              className="gap-1 h-8"
              title="包含 sample 预览图"
            >
              <FileArchive className="size-3" />
              + 样本图
            </Button>
            <Button
              size="sm"
              variant="destructive"
              onClick={onWorkspaceDelete}
              disabled={!isTerminal}
              className="gap-1 h-8"
              title={
                !isTerminal
                  ? "任务未结束 —— 取消后才能删除工作区"
                  : "永久删除整个 workspace 目录"
              }
            >
              <FolderOpen className="size-3" />
              删除工作区
            </Button>
          </div>
        </div>
      </CardHeader>
      {row.checkpoints.length > 0 && (
        <CardContent className="pt-0">
          {/* 单卡片内按 ~7 行高度限高 — checkpoint 多时(每 epoch 存一份的
              长跑训练 LoRA 常常 20-50 份)避免单个 job 把整页顶到屏幕外面去。
              内层滚,外层"产物管理"主滚动条不动。 */}
          <div className="rounded-[4px] border border-border/60 bg-muted/30 divide-y divide-border/60 max-h-[18rem] overflow-y-auto">
            {row.checkpoints.map((ckpt) => (
              <div
                key={ckpt.path}
                className="flex items-center justify-between gap-2 px-3 py-2 text-xs"
              >
                <div className="min-w-0 flex-1">
                  <code className="font-mono text-[11px] block truncate">
                    {ckpt.path}
                  </code>
                  <span className="text-muted-foreground text-[10px]">
                    {formatBytes(ckpt.size_bytes)} ·{" "}
                    {new Date(ckpt.modified_at * 1000).toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center gap-1 shrink-0">
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => onFileDownload(ckpt.path)}
                    className="h-7 px-2 gap-1"
                  >
                    <Download className="size-3" />
                    下载
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => onFileDelete(ckpt.path)}
                    disabled={fileBusy}
                    className="h-7 px-2 gap-1 text-destructive hover:text-destructive"
                  >
                    <Trash2 className="size-3" />
                    删除
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      )}
    </Card>
  )
}
