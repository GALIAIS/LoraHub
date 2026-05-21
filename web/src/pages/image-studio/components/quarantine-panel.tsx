/**
 * Quarantine review + restore panel.
 *
 * Lives at the bottom of the Audit stage as a collapsible drawer.
 * Lists everything that's been moved to `.workbench/quarantine/`
 * across all curate operations, with a per-row "restore" button.
 *
 * Restored entries stay in the index with a `restored_at` timestamp
 * so the audit trail survives, but the UI hides them by default
 * (toggle reveals).
 */
import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  Archive,
  ArrowDown,
  ArrowUp,
  History,
  Loader2,
  RotateCcw,
} from "lucide-react"
import { toast } from "sonner"
import {
  imageStudioQuarantineList,
  imageStudioRestoreQuarantine,
  type QuarantineEntry,
} from "@/lib/api"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface Props {
  datasetPath: string
}

export function QuarantinePanel({ datasetPath }: Props) {
  const qc = useQueryClient()
  const [expanded, setExpanded] = useState(false)
  const [showRestored, setShowRestored] = useState(false)

  const listQuery = useQuery({
    queryKey: ["image-studio-quarantine", datasetPath],
    queryFn: () => imageStudioQuarantineList(datasetPath),
    enabled: Boolean(datasetPath),
  })

  const restoreMutation = useMutation({
    mutationFn: (qpaths: string[]) =>
      imageStudioRestoreQuarantine({
        dataset_path: datasetPath,
        quarantine_paths: qpaths,
      }),
    onSuccess: (data) => {
      toast.success(`已恢复 ${data.restored_count} 张`)
      qc.invalidateQueries({ queryKey: ["image-studio-quarantine", datasetPath] })
      qc.invalidateQueries({ queryKey: ["image-studio"] })
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err)
      toast.error("恢复失败", { description: msg })
    },
  })

  const all = listQuery.data?.entries ?? []
  const active = all.filter((e) => !e.restored_at)
  const restored = all.filter((e) => e.restored_at)
  const visible = showRestored ? all : active

  return (
    <div className="border-t border-border/60 bg-muted/20">
      <button
        type="button"
        className="flex w-full items-center gap-2 px-4 py-2 text-xs hover:bg-muted/40"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
      >
        <Archive className="size-3.5 text-muted-foreground" />
        <span className="font-medium">隔离区</span>
        <span className="tabular-nums text-muted-foreground">
          {active.length} 项
          {restored.length > 0 && ` · 已恢复 ${restored.length}`}
        </span>
        <span className="ml-auto text-muted-foreground">
          {expanded ? <ArrowDown className="size-3.5" /> : <ArrowUp className="size-3.5" />}
        </span>
      </button>

      {expanded && (
        <div className="border-t border-border/60 bg-background">
          {visible.length === 0 ? (
            <div className="p-4 text-xs text-muted-foreground text-center">
              {active.length === 0
                ? "隔离区为空 — 在审计列表点「隔离」可批量移入。"
                : "无符合条件的条目"}
            </div>
          ) : (
            <>
              <div className="flex items-center gap-2 px-4 py-2 border-b border-border/40 text-[11px] text-muted-foreground">
                <button
                  type="button"
                  className={cn(
                    "inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px]",
                    showRestored ? "bg-muted text-foreground" : "hover:bg-muted/50",
                  )}
                  onClick={() => setShowRestored((v) => !v)}
                >
                  <History className="size-3" />
                  显示已恢复
                </button>
                <span className="ml-auto">显示 {visible.length} 项</span>
              </div>
              <ul className="max-h-64 overflow-y-auto">
                {visible.map((e, i) => (
                  <QuarantineRow
                    key={i}
                    entry={e}
                    onRestore={() => restoreMutation.mutate([e.quarantine_path])}
                    pending={
                      restoreMutation.isPending &&
                      restoreMutation.variables?.includes(e.quarantine_path)
                    }
                  />
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  )
}

function QuarantineRow({
  entry,
  onRestore,
  pending,
}: {
  entry: QuarantineEntry
  onRestore: () => void
  pending: boolean
}) {
  const isRestored = Boolean(entry.restored_at)
  const filename = entry.original_path.split(/[\\/]/).pop() ?? "?"
  return (
    <li
      className={cn(
        "flex items-center gap-3 px-4 py-1.5 text-[11px] border-t first:border-t-0 border-border/30",
        isRestored && "opacity-60",
      )}
    >
      <span
        className={cn(
          "h-1.5 w-1.5 rounded-full",
          isRestored ? "bg-emerald-500" : "bg-amber-500",
        )}
        title={isRestored ? "已恢复" : "在隔离区"}
      />
      <span className="font-mono truncate flex-1" title={entry.original_path}>
        {filename}
      </span>
      {entry.reason && (
        <span className="text-muted-foreground truncate max-w-40 italic" title={entry.reason}>
          {entry.reason}
        </span>
      )}
      <span className="tabular-nums text-muted-foreground text-[10px]">
        {new Date(entry.moved_at).toLocaleString()}
      </span>
      {!isRestored && (
        <Button
          size="sm"
          variant="ghost"
          className="h-6 px-2 text-[11px] gap-1"
          onClick={onRestore}
          disabled={pending}
        >
          {pending ? <Loader2 className="size-3 animate-spin" /> : <RotateCcw className="size-3" />}
          恢复
        </Button>
      )}
    </li>
  )
}
