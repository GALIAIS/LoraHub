import { useEffect, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Image as ImageIcon, Loader2, Save } from "lucide-react"
import { api } from "@/lib/api"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"

export interface CaptionEditorModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  /** Image whose sidecar caption (.txt) we're editing. */
  imagePath: string | null
  /** Optional callback after a successful save (e.g. refresh dataset scan). */
  onAfterSave?: () => void
}

/**
 * Minimal caption editor — opens the .txt sidecar for an image, edits in
 * a textarea, persists via /api/datasets/caption. Replaces the older
 * file-browser-coupled editor: caption editing is the only flow we need
 * here, so the UI stays narrow.
 */
export function CaptionEditorModal({
  open,
  onOpenChange,
  imagePath,
  onAfterSave,
}: CaptionEditorModalProps) {
  const qc = useQueryClient()
  const captionQuery = useQuery({
    queryKey: ["caption", imagePath],
    queryFn: () => api.getCaption(imagePath!),
    enabled: open && !!imagePath,
    staleTime: 0,
  })

  const [draft, setDraft] = useState<string | null>(null)
  const [savedAt, setSavedAt] = useState<number | null>(null)

  useEffect(() => {
    if (!open) {
      setDraft(null)
      setSavedAt(null)
    }
  }, [open, imagePath])

  const baseText = captionQuery.data?.caption ?? ""
  const text = draft ?? baseText
  const dirty = draft !== null && draft !== baseText

  const save = useMutation({
    mutationFn: (value: string) => api.putCaption(imagePath!, value),
    onSuccess: () => {
      setDraft(null)
      setSavedAt(Date.now())
      qc.invalidateQueries({ queryKey: ["caption", imagePath] })
      onAfterSave?.()
    },
  })

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <ImageIcon className="size-4" /> 编辑标注
          </DialogTitle>
          <DialogDescription className="font-mono text-[11px] break-all">
            {imagePath ?? "—"}
          </DialogDescription>
        </DialogHeader>

        {captionQuery.isLoading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground py-6 px-1">
            <Loader2 className="size-3.5 animate-spin" /> 读取标注…
          </div>
        ) : captionQuery.isError ? (
          <div className="text-xs text-destructive font-mono">
            {(captionQuery.error as Error).message}
          </div>
        ) : (
          <textarea
            value={text}
            onChange={(e) => setDraft(e.target.value)}
            rows={10}
            className="w-full font-mono text-sm border border-border/60 rounded-[4px] p-2 bg-background resize-y"
            placeholder="逗号分隔的标签…"
          />
        )}

        <DialogFooter className="flex items-center justify-between gap-3">
          <div className="text-[11px] text-muted-foreground">
            {save.isError ? (
              <span className="text-destructive font-mono">
                {(save.error as Error).message}
              </span>
            ) : savedAt ? (
              <span>已保存</span>
            ) : dirty ? (
              <span>未保存的修改</span>
            ) : null}
          </div>
          <div className="flex gap-2">
            <Button variant="ghost" onClick={() => onOpenChange(false)}>
              关闭
            </Button>
            <Button
              disabled={!dirty || save.isPending}
              onClick={() => save.mutate(text)}
            >
              {save.isPending ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <Save className="size-3.5" />
              )}
              保存
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
