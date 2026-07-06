import { useEffect, useRef, type ReactNode } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { Loader2, X } from "lucide-react"

import { api } from "@/lib/api"
import { cancelTask, removeTask } from "@/lib/studio-task-store"
import { useStudioTasksFor } from "@/hooks/use-studio-tasks"

// ai-bulk-modal.tsx 里同款 fallback。WD14 模型清单从 /tagging/wd14/models
// 来；首次还没拿到时先用一个真实存在的默认值，避免 401。
export const FALLBACK_DEFAULT_MODEL = "SmilingWolf/wd-eva02-large-tagger-v3"

const CAPTION_WRITING_TASKS = new Set(["caption", "smart-caption", "wd14"])

export interface TaggerDownloadDisplay {
  label: string
  percent: number
}

export function useTaggerDownloadDisplay(active: boolean): TaggerDownloadDisplay | null {
  const query = useQuery({
    queryKey: ["tagging-download-status"],
    queryFn: api.getTaggerDownloadStatus,
    enabled: active,
    refetchInterval: active ? 1000 : false,
  })
  if (!active) return null
  const job =
    query.data?.jobs.find((j) => j.status === "running") ??
    query.data?.jobs.find((j) => j.status === "done" || j.status === "error")
  if (!job) return null
  const pct = job.percent == null ? 0 : Math.max(0, Math.min(100, job.percent))
  const name = `${job.repo_id.split("/").pop() ?? job.repo_id}/${job.filename}`
  if (job.status === "error") return { label: `模型下载失败：${name}`, percent: pct }
  if (job.status === "done") return { label: `模型下载完成：${name}`, percent: 100 }
  return { label: `正在下载标注模型：${name} · ${Math.round(pct)}%`, percent: pct }
}

export function useRefreshCaptionViewsOnTaskDone(datasetPath: string) {
  const qc = useQueryClient()
  const tasks = useStudioTasksFor(datasetPath)
  const seen = useRef(new Set<string>())

  useEffect(() => {
    for (const task of tasks) {
      if (task.status === "running" || !CAPTION_WRITING_TASKS.has(task.kind)) continue
      const key = `${task.id}:${task.status}`
      if (seen.current.has(key)) continue
      seen.current.add(key)
      qc.invalidateQueries({ queryKey: ["image-studio-captions-vocab", datasetPath] })
      qc.invalidateQueries({ queryKey: ["image-studio-audit-report", datasetPath] })
      qc.invalidateQueries({ queryKey: ["image-studio"] })
    }
  }, [datasetPath, qc, tasks])
}

export function TaskBanner({ datasetPath }: { datasetPath: string }) {
  useRefreshCaptionViewsOnTaskDone(datasetPath)
  const tasks = useStudioTasksFor(datasetPath)
  const newest = [...tasks].sort((a, b) => b.startedAt - a.startedAt)[0]
  const running = newest?.status === "running"
  const download = useTaggerDownloadDisplay(
    Boolean(newest && running && (newest.kind === "wd14" || newest.kind === "smart-caption")),
  )
  if (!newest) return null
  const lastImageName = newest.lastImage
    ? newest.lastImage.split(/[/\\]/).pop() ?? ""
    : ""
  const label = download?.label ?? (running
    ? newest.kind === "caption"
      ? `${newest.label}中…${lastImageName ? ` · ${lastImageName}` : ""}`
      : newest.kind === "smart-caption"
        ? `${newest.label}中…${lastImageName ? ` · ${lastImageName}` : ""}`
        : newest.kind === "quality-score"
          ? `${newest.label}中…${lastImageName ? ` · ${lastImageName}` : ""}`
          : newest.kind === "trigger-words"
            ? `${newest.label}中…${lastImageName ? ` · ${lastImageName}` : ""}`
            : newest.kind === "wd14"
            ? `${newest.label}中… ${newest.processed ?? 0}/${newest.total ?? "?"}`
            : `${newest.label}中…`
    : newest.label)
  return (
    <div className="flex items-center gap-3 rounded-md border bg-muted/30 px-3 py-2 mb-3">
      {running && <Loader2 className="size-4 animate-spin text-primary" />}
      <span className="text-xs font-medium">{label}</span>
      {download && (
        <div className="shiro-progress-track h-1.5 w-24 border-0 bg-muted">
          <div
            className="shiro-progress-fill bg-primary"
            style={{ width: `${download.percent}%` }}
          />
        </div>
      )}
      {newest.processed != null && newest.kind !== "wd14" && (
        <span className="text-xs text-muted-foreground">
          {newest.processed}
          {newest.total ? ` / ${newest.total}` : ""} 张
        </span>
      )}
      {newest.errorMsg && (
        <span className="text-xs text-destructive truncate flex-1">
          {newest.errorMsg}
        </span>
      )}
      {running ? (
        <button
          type="button"
          onClick={() => void cancelTask(newest)}
          className="ml-auto text-xs text-muted-foreground hover:text-destructive"
          title="停止"
        >
          停止
        </button>
      ) : (
        <button
          type="button"
          onClick={() => removeTask(newest.id)}
          className="ml-auto text-xs text-muted-foreground hover:text-foreground"
          title="关闭"
        >
          <X className="size-3.5" />
        </button>
      )}
    </div>
  )
}

export function Row({
  label,
  children,
}: {
  label: string
  children: ReactNode
}) {
  return (
    <div className="grid grid-cols-[5rem_1fr] items-center gap-2">
      <span className="text-muted-foreground">{label}</span>
      <div className="min-w-0">{children}</div>
    </div>
  )
}

export function Field({
  label,
  children,
}: {
  label: string
  children: ReactNode
}) {
  return (
    <div className="space-y-0.5">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
      <div>{children}</div>
    </div>
  )
}
