import type { ReactNode } from "react"
import { ChevronLeft, ChevronRight, Loader2 } from "lucide-react"

import {
  cancelTask,
  isStudioTaskActive,
  type StudioTaskRecord,
} from "@/lib/studio-task-store"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { cn } from "@/lib/utils"
import { toastApiError } from "@/lib/toast-api-error"

export function ViewChip({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        "px-2.5 h-full transition-colors",
        active
          ? "bg-muted font-medium text-foreground"
          : "text-muted-foreground hover:bg-muted/60 hover:text-foreground",
      )}
    >
      {children}
    </button>
  )
}

const SORT_OPTIONS = [
  { value: "name", label: "名称" },
  { value: "mtime", label: "修改时间" },
  { value: "size", label: "大小" },
] as const

export function SortSelect({
  value,
  onChange,
}: {
  value: string
  onChange: (v: string) => void
}) {
  return (
    <Select
      items={SORT_OPTIONS}
      value={value}
      onValueChange={(v) => onChange(v as string)}
    >
      <SelectTrigger className="h-7 text-[11px] min-w-[6.5rem]">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {SORT_OPTIONS.map((o) => (
          <SelectItem key={o.value} value={o.value}>
            {o.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

export function Pagination({
  page,
  total,
  onChange,
}: {
  page: number
  total: number
  onChange: (p: number) => void
}) {
  return (
    <div className="flex items-center justify-center gap-1.5 pt-4 text-[12px]">
      <Button
        size="sm"
        variant="outline"
        type="button"
        disabled={page <= 1}
        onClick={() => onChange(page - 1)}
        aria-label="上一页"
        className="h-7 w-8 shrink-0 px-0"
      >
        <ChevronLeft className="size-3.5" />
      </Button>
      <span className="min-w-[4.75rem] shrink-0 rounded-[6px] border border-border/60 bg-muted/30 px-2 py-1 text-center font-mono text-[11px] tabular-nums text-muted-foreground">
        {page} / {total}
      </span>
      <Button
        size="sm"
        variant="outline"
        type="button"
        disabled={page >= total}
        onClick={() => onChange(page + 1)}
        aria-label="下一页"
        className="h-7 w-8 shrink-0 px-0"
      >
        <ChevronRight className="size-3.5" />
      </Button>
    </div>
  )
}

/**
 * Banner that summarises a single global studio task. The data shape
 * comes from the task store so the banner is identical no matter
 * whether the user just kicked the task off or just got back from
 * another route while it was running in the background.
 */
export function StudioTaskBanner({
  task,
  onDismiss,
}: {
  task: StudioTaskRecord
  onDismiss: () => void
}) {
  const active = isStudioTaskActive(task.status)
  const stopping = task.status === "stopping"
  const lastImageName = task.lastImage
    ? task.lastImage.split(/[/\\]/).pop() ?? ""
    : ""
  const showsLastImage =
    task.kind === "caption" ||
    task.kind === "smart-caption" ||
    task.kind === "quality-score" ||
    task.kind === "trigger-words"
  const runningLabel = showsLastImage
      ? `${task.label}中…${lastImageName ? ` · ${lastImageName}` : ""}`
      : task.kind === "wd14"
        ? `${task.label}中… ${task.processed ?? 0}/${task.total ?? "?"}`
        : `${task.label}中…`
  const label = stopping
    ? `正在停止${task.label}…`
    : active
      ? runningLabel
      : task.label

  return (
    <div className="flex items-center gap-3 border-b px-4 py-2 bg-muted/30">
      {active && <Loader2 className="size-4 animate-spin text-primary" />}
      <span className="text-xs font-medium">{label}</span>
      {task.processed != null && task.kind !== "wd14" && (
        <span className="text-xs text-muted-foreground">
          {task.processed}
          {task.total ? ` / ${task.total}` : ""} 张
        </span>
      )}
      {task.errorMsg && (
        <span className="text-xs text-destructive truncate flex-1">
          {task.errorMsg}
        </span>
      )}
      {active ? (
        <button
          type="button"
          onClick={() => {
            void cancelTask(task).catch((error) => {
              toastApiError(error, { title: "停止任务失败" })
            })
          }}
          disabled={stopping}
          className="ml-auto text-xs text-muted-foreground hover:text-destructive"
        >
          {stopping ? "正在停止" : "停止"}
        </button>
      ) : (
        <button
          type="button"
          onClick={onDismiss}
          className="ml-auto text-xs text-muted-foreground hover:text-foreground"
        >
          关闭
        </button>
      )}
    </div>
  )
}
