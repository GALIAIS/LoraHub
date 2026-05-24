/**
 * ResumeWithEditDialog — full-screen dialog that lets the user tweak
 * ``cfg`` on a paused / canceled / failed job before resuming. Locked
 * fields (rank, arch, checkpoint paths …) are still editable in the
 * form so the existing schema panels work as-is, but the backend
 * rejects locked-field changes with 409 and we surface the error
 * inline.
 */
import { useEffect, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { toast } from "sonner"
import { Play, X } from "lucide-react"
import { api, type JobDetail, type ValidationFieldError } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { ConfigForm, type ConfigFormValue } from "@/components/config-form"

interface ResumeWithEditDialogProps {
  job: JobDetail
  onClose: () => void
  onResumed: () => void
}

export function ResumeWithEditDialog({
  job,
  onClose,
  onResumed,
}: ResumeWithEditDialogProps) {
  const qc = useQueryClient()
  const [draft, setDraft] = useState<ConfigFormValue | null>(null)
  const [errors, setErrors] = useState<ValidationFieldError[]>([])
  const [serverError, setServerError] = useState<string | null>(null)

  useEffect(() => {
    if (job.config_snapshot) {
      // Snapshot is camelCase: backend dumps with by_alias=True so the
      // shape lines up 1:1 with the form widgets' field names.
      setDraft(job.config_snapshot as unknown as ConfigFormValue)
    }
  }, [job.config_snapshot])

  // Esc closes the dialog (matches the rest of the LoRaHub modal
  // surface).
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onClose])

  const submit = useMutation({
    mutationFn: async () => {
      if (!draft) throw new Error("config not loaded yet")
      // Pre-flight validate so locked-field shape errors surface in
      // the form rather than as opaque 422s. The backend re-validates
      // anyway.
      const validation = await api.validateConfig(
        draft as unknown as Record<string, unknown>,
      )
      if (!validation.valid) {
        setErrors(validation.errors ?? [])
        throw new Error("配置校验未通过")
      }
      setErrors([])
      return api.resumeJob(job.id, draft as unknown as Record<string, unknown>)
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobs"] })
      qc.invalidateQueries({ queryKey: ["job", job.id] })
      toast.success("已用新配置恢复训练", {
        description: "保持任务 ID，从最新 state 续",
      })
      onResumed()
    },
    onError: (err) => {
      setServerError(err instanceof Error ? err.message : String(err))
    },
  })

  return (
    <div
      className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/60 p-6"
      onClick={onClose}
    >
      <div
        className="relative flex h-[90vh] w-[92vw] max-w-[1200px] flex-col overflow-hidden rounded-[6px] border border-border/60 bg-background shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex shrink-0 items-center justify-between border-b border-border/60 px-5 py-3">
          <div>
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
              编辑配置
            </div>
            <div className="text-sm font-mono">{job.id.slice(-8)}</div>
            <p className="mt-1 text-[11px] text-muted-foreground">
              改 lr / dropTokens / 数据集等字段会在续训中生效；改 rank / arch / checkpoint 等会被拒绝（那些会让权重不兼容）。
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

        {/* Native scroll container — base-ui's ScrollArea wraps the
            content in a viewport that ignored ``flex-1 min-h-0`` here
            and let the dialog grow to its content height instead of
            paging. ``overflow-y-auto`` on the flex item with
            ``min-h-0`` is the canonical fix that stops the column
            from expanding past the parent. */}
        <div className="flex-1 min-h-0 overflow-y-auto px-5 py-4">
          {draft === null ? (
            <div className="text-sm text-muted-foreground py-6">加载中…</div>
          ) : (
            <ConfigForm value={draft} onChange={setDraft} errors={errors} />
          )}
        </div>

        <footer className="flex shrink-0 items-center justify-end gap-2 border-t border-border/60 px-5 py-3">
          <Button variant="outline" onClick={onClose} disabled={submit.isPending}>
            取消
          </Button>
          <Button
            onClick={() => submit.mutate()}
            disabled={!draft || submit.isPending}
          >
            {submit.isPending ? (
              <Spinner className="size-3" />
            ) : (
              <Play className="size-3" />
            )}{" "}
            {submit.isPending ? "续训中…" : "用新配置续训"}
          </Button>
        </footer>
      </div>
    </div>
  )
}
