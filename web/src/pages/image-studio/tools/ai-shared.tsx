import type { ReactNode } from "react"
import { Loader2, X } from "lucide-react"

import { cancelTask, removeTask } from "@/lib/studio-task-store"
import { useStudioTasksFor } from "@/hooks/use-studio-tasks"

// ai-bulk-modal.tsx 里同款 fallback。WD14 模型清单从 /tagging/wd14/models
// 来；首次还没拿到时先用一个真实存在的默认值，避免 401。
export const FALLBACK_DEFAULT_MODEL = "SmilingWolf/wd-eva02-large-tagger-v3"

export function TaskBanner({ datasetPath }: { datasetPath: string }) {
  const tasks = useStudioTasksFor(datasetPath)
  if (tasks.length === 0) return null
  const newest = [...tasks].sort((a, b) => b.startedAt - a.startedAt)[0]
  if (!newest) return null
  const running = newest.status === "running"
  const lastImageName = newest.lastImage
    ? newest.lastImage.split(/[/\\]/).pop() ?? ""
    : ""
  const label = running
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
    : newest.label
  return (
    <div className="flex items-center gap-3 rounded-md border bg-muted/30 px-3 py-2 mb-3">
      {running && <Loader2 className="size-4 animate-spin text-primary" />}
      <span className="text-xs font-medium">{label}</span>
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
