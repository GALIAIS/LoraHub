/**
 * CloneWithStateDialog — branch a fresh job from a saved state of the
 * current job. Unlike ResumeWithEditDialog (which restarts the SAME
 * JobRecord on its workspace), this spawns a NEW job whose
 * ``cfg.resume.resume_from`` is pinned to the picked state.
 *
 * Two-step flow:
 *  1. Picker — list every resumable artifact (kohya/anima_lora:
 *     ``*-state*`` dirs; dp: timestamped run dirs) returned by
 *     ``GET /artifacts/{id}/states``. Newest is selected by default.
 *  2. Confirm — show the source's snapshot config in a read-only
 *     summary (rank, arch, backend.type are locked anyway), then
 *     call ``POST /jobs/{id}/clone-with-state``.
 *
 * The "edit config" path is deliberately deferred — the same field-lock
 * applies as in ResumeWithEditDialog, and the typical user wants
 * "more epochs from here" not "different rank from here". Adding a
 * full ConfigForm later is a one-line swap if needed.
 */
import { useEffect, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { toast } from "sonner"
import { GitBranch, X, Layers, Clock } from "lucide-react"
import { api, type JobDetail } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"

interface Props {
  job: JobDetail
  onClose: () => void
  onCloned: (newJobId: string) => void
}

export function CloneWithStateDialog({ job, onClose, onCloned }: Props) {
  const qc = useQueryClient()
  const [selectedPath, setSelectedPath] = useState<string | null>(null)
  const [serverError, setServerError] = useState<string | null>(null)

  const states = useQuery({
    queryKey: ["job-states", job.id],
    queryFn: () => api.listJobStates(job.id),
    refetchOnWindowFocus: false,
  })

  // Default to the newest state once data lands.
  useEffect(() => {
    if (selectedPath !== null) return
    const first = states.data?.states?.[0]?.path
    if (first) setSelectedPath(first)
  }, [states.data, selectedPath])

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onClose])

  const submit = useMutation({
    mutationFn: async () => {
      if (!selectedPath) throw new Error("请先选择一个 state")
      return api.cloneJobWithState(job.id, { statePath: selectedPath })
    },
    onSuccess: (fresh) => {
      qc.invalidateQueries({ queryKey: ["jobs"] })
      toast.success("已从该 state 派生新任务", {
        description: `新任务 ID ${fresh.id.slice(-8)}`,
      })
      onCloned(fresh.id)
    },
    onError: (err) => {
      setServerError(err instanceof Error ? err.message : String(err))
    },
  })

  const items = states.data?.states ?? []
  const backend = states.data?.backend_type ?? null

  return (
    <div
      className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/60 p-6"
      onClick={onClose}
    >
      <div
        className="relative flex max-h-[90vh] w-[600px] max-w-[92vw] flex-col overflow-hidden rounded-[6px] border border-border/60 bg-background shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex shrink-0 items-center justify-between border-b border-border/60 px-5 py-3">
          <div>
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
              从 state 派生新任务
            </div>
            <div className="text-sm font-mono">{job.id.slice(-8)}</div>
            <p className="mt-1 text-[11px] text-muted-foreground">
              新任务保留同一 backend / rank / arch；从所选 state
              恢复 optimizer + lr 进度，原任务保持不动。
            </p>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            disabled={submit.isPending}
            aria-label="关闭"
          >
            <X className="size-4" />
          </Button>
        </header>

        {serverError && (
          <div className="mx-5 mt-3 shrink-0 rounded-[4px] border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs font-mono text-destructive whitespace-pre-wrap break-words">
            {serverError}
          </div>
        )}

        <div className="flex-1 min-h-0 overflow-y-auto px-5 py-4 space-y-2">
          {states.isLoading && (
            <div className="text-sm text-muted-foreground py-6 flex items-center gap-2">
              <Spinner className="size-3" /> 加载 state 列表…
            </div>
          )}
          {!states.isLoading && items.length === 0 && (
            <div className="text-sm text-muted-foreground py-6">
              该任务暂无可恢复的 state。
              {backend === "diffusion-pipe"
                ? "等 diffusion-pipe 写出第一个 global_step* 后再来。"
                : "等 save_state 周期触发后再来。"}
            </div>
          )}
          {items.map((s) => {
            const selected = s.path === selectedPath
            const dt = new Date(s.modified_at * 1000)
            return (
              <button
                key={s.path}
                type="button"
                onClick={() => setSelectedPath(s.path)}
                className={
                  "w-full rounded-[4px] border px-3 py-2.5 text-left transition-colors " +
                  (selected
                    ? "border-primary bg-primary/5"
                    : "border-border/60 hover:border-primary/40 hover:bg-accent/40")
                }
              >
                <div className="flex items-center gap-2 text-sm font-mono break-all">
                  <Layers className="size-3 shrink-0 text-muted-foreground" />
                  {s.basename}
                </div>
                <div className="mt-1 flex flex-wrap items-center gap-3 text-[11px] text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Clock className="size-3" />
                    {dt.toLocaleString()}
                  </span>
                  {s.current_step !== undefined && (
                    <span>step {s.current_step}</span>
                  )}
                  {s.current_epoch !== undefined && (
                    <span>epoch {s.current_epoch}</span>
                  )}
                  {s.latest_step !== undefined && (
                    <span>latest_step {s.latest_step}</span>
                  )}
                  {s.global_step_count !== undefined && (
                    <span>{s.global_step_count} ckpt</span>
                  )}
                </div>
              </button>
            )
          })}
        </div>

        <footer className="flex shrink-0 items-center justify-end gap-2 border-t border-border/60 px-5 py-3">
          <Button variant="outline" onClick={onClose} disabled={submit.isPending}>
            取消
          </Button>
          <Button
            onClick={() => submit.mutate()}
            disabled={!selectedPath || submit.isPending}
          >
            {submit.isPending ? (
              <Spinner className="size-3" />
            ) : (
              <GitBranch className="size-3" />
            )}{" "}
            {submit.isPending ? "派生中…" : "派生新任务"}
          </Button>
        </footer>
      </div>
    </div>
  )
}
