import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  AlertCircle,
  CheckCircle2,
  Loader2,
  Play,
  Undo2,
} from "lucide-react"
import { toast } from "sonner"
import {
  imageStudioApplyOps,
  imageStudioDeleteOp,
  imageStudioListOps,
} from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { fmtUnixSeconds } from "@/pages/jobs/utils"

/**
 * Pending ops drawer. Image-studio is built around a queue model:
 * "delete this", "rotate that", "replace_caption: ..." are appended
 * to a per-image op list and only flush on imageStudioApplyOps.
 *
 * Until now nothing in the UI exposed that queue, so users couldn't
 * see what was about to happen and couldn't undo a queued op short of
 * applying-and-reversing. This dialog lists every pending op for the
 * currently-open dataset, lets users drop individual ops, and offers
 * a single button that triggers apply across every distinct image
 * with at least one op (the server's queue is per-image so we have
 * to fan out, but it's idempotent and fast).
 */
export interface PendingOpsDialogProps {
  open: boolean
  onOpenChange: (next: boolean) => void
  /** When provided, scopes the listing to a single image / dataset path. */
  path?: string
}

export function PendingOpsDialog({ open, onOpenChange, path }: PendingOpsDialogProps) {
  const queryClient = useQueryClient()
  const [busyApply, setBusyApply] = useState(false)

  const opsQuery = useQuery({
    queryKey: ["image-studio", "ops", path ?? "all"],
    queryFn: () => imageStudioListOps(path),
    enabled: open,
    refetchInterval: open ? 2000 : false,
  })

  const ops = opsQuery.data?.ops ?? []

  const dropMutation = useMutation({
    mutationFn: (id: string) => imageStudioDeleteOp(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["image-studio"] })
    },
    onError: (e) => {
      toast.error("撤销失败", {
        description: e instanceof Error ? e.message : String(e),
      })
    },
  })

  const applyAll = async () => {
    if (ops.length === 0) return
    setBusyApply(true)
    // Group by image — the server applies one image at a time.
    const byImage = new Map<string, number>()
    for (const op of ops) {
      byImage.set(op.imagePath, (byImage.get(op.imagePath) ?? 0) + 1)
    }
    const failures: string[] = []
    try {
      const results = await Promise.allSettled(
        Array.from(byImage.keys()).map((p) => imageStudioApplyOps(p)),
      )
      for (const res of results) {
        if (res.status === "rejected") {
          failures.push(
            res.reason instanceof Error ? res.reason.message : String(res.reason),
          )
        } else if (res.value.errors.length > 0) {
          for (const err of res.value.errors) failures.push(err.error)
        }
      }
      if (failures.length === 0) {
        toast.success(`已应用 ${ops.length} 个操作`, {
          description: `跨 ${byImage.size} 张图片`,
        })
      } else {
        toast.error(`应用时出现 ${failures.length} 个错误`, {
          description: failures[0],
        })
      }
      queryClient.invalidateQueries({ queryKey: ["image-studio"] })
    } finally {
      setBusyApply(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Play className="size-4" /> 待应用的操作
          </DialogTitle>
          <DialogDescription>
            旋转 / 翻转 / 替换 / 删除等修改会先写入队列，点&quot;应用全部&quot;
            才会落到磁盘。在这里可以看到队列状态并撤销单条。
          </DialogDescription>
        </DialogHeader>

        <div className="flex items-center justify-between gap-3 px-1">
          <div className="text-xs text-muted-foreground">
            共 <span className="font-mono text-foreground">{ops.length}</span>{" "}
            条待应用
            {path && (
              <span className="ml-2">
                · 范围 <code className="font-mono">{shortPath(path)}</code>
              </span>
            )}
          </div>
          <Button
            size="sm"
            disabled={ops.length === 0 || busyApply}
            onClick={applyAll}
          >
            {busyApply ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <CheckCircle2 className="size-3" />
            )}
            应用全部
          </Button>
        </div>

        <div className="max-h-[60vh] overflow-y-auto rounded-[4px] border border-border/60">
          {opsQuery.isLoading && (
            <div className="px-3 py-6 text-xs text-muted-foreground text-center">
              <Loader2 className="size-3 animate-spin inline-block mr-1.5" />
              加载…
            </div>
          )}
          {opsQuery.isError && (
            <div className="px-3 py-6 text-xs text-destructive text-center">
              <AlertCircle className="size-3 inline-block mr-1.5" />
              {(opsQuery.error as Error).message}
            </div>
          )}
          {!opsQuery.isLoading && !opsQuery.isError && ops.length === 0 && (
            <div className="px-3 py-8 text-xs text-muted-foreground text-center">
              暂无待应用的操作。修改图片后会自动出现在这里。
            </div>
          )}
          {ops.map((op) => (
            <div
              key={op.id}
              className="flex items-center gap-2 px-3 py-2 border-b border-border/40 last:border-b-0 hover:bg-muted/40 transition-colors"
            >
              <Badge variant="outline" className="rounded-[2px] text-[10px] font-mono shrink-0">
                {op.op}
              </Badge>
              <span
                className="font-mono text-[11px] truncate flex-1 min-w-0"
                title={op.imagePath}
              >
                {shortPath(op.imagePath)}
              </span>
              {summarizePayload(op.payload) && (
                <span className="text-[10px] text-muted-foreground truncate max-w-[10rem] hidden md:inline">
                  {summarizePayload(op.payload)}
                </span>
              )}
              <span className="text-[10px] text-muted-foreground tabular-nums shrink-0 hidden sm:inline">
                {fmtIso(op.createdAt)}
              </span>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => dropMutation.mutate(op.id)}
                disabled={dropMutation.isPending}
                title="撤销 (从队列移除)"
              >
                <Undo2 className="size-3.5" />
              </Button>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  )
}

function shortPath(p: string): string {
  const parts = p.split(/[\\/]/)
  if (parts.length <= 2) return p
  return `…/${parts.slice(-2).join("/")}`
}

function summarizePayload(payload: Record<string, unknown> | undefined): string {
  if (!payload || Object.keys(payload).length === 0) return ""
  if ("degrees" in payload) return `${payload.degrees}°`
  if ("direction" in payload) return String(payload.direction)
  if ("caption" in payload) {
    const cap = String(payload.caption ?? "")
    return cap.length > 40 ? `${cap.slice(0, 40)}…` : cap
  }
  // Fallback: just show the keys so the user can guess what's queued.
  return Object.keys(payload).join(", ")
}

function fmtIso(s: string): string {
  // Best-effort: treat the createdAt string as ISO; degrade to the
  // raw value if Date.parse can't make sense of it.
  const d = new Date(s)
  if (Number.isNaN(d.getTime())) return s
  return fmtUnixSeconds(Math.floor(d.getTime() / 1000))
}

// Re-export under a simpler name, mirroring the other dialog modules.
export { PendingOpsDialog as PendingOpsDrawer }
