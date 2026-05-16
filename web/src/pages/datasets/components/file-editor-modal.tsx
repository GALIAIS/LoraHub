import { useEffect, useRef, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Image as ImageIcon, Loader2, Save, X } from "lucide-react"
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
import { Badge } from "@/components/ui/badge"

type Mode =
  | { kind: "fs" }
  | { kind: "caption"; imagePath: string; onAfterSave?: () => void }

export interface FileEditorModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  /** Path of the file to edit, OR (for caption mode) the path of the .txt file that should be written. */
  path: string | null
  mode?: Mode
}

/**
 * Modal-based editor used for both arbitrary FS files and image caption
 * sidecars. Caption mode uses /api/datasets/caption (which has its own
 * allow-list for image paths); fs mode uses /api/fs/read + /api/fs/write.
 */
export function FileEditorModal({
  open,
  onOpenChange,
  path,
  mode = { kind: "fs" },
}: FileEditorModalProps) {
  const qc = useQueryClient()
  const isCaption = mode.kind === "caption"

  const captionQuery = useQuery({
    queryKey: ["caption", isCaption ? mode.imagePath : "", path],
    queryFn: () => api.getCaption((mode as { imagePath: string }).imagePath),
    enabled: open && isCaption && !!(mode as { imagePath: string }).imagePath,
    staleTime: 0,
  })

  const fsQuery = useQuery({
    queryKey: ["fs-read", path],
    queryFn: () => api.fsRead(path!),
    enabled: open && !isCaption && !!path,
    staleTime: 0,
  })

  const [draft, setDraft] = useState<string | null>(null)
  const [savedAt, setSavedAt] = useState<number | null>(null)
  const taRef = useRef<HTMLTextAreaElement | null>(null)

  // Reset local state when the file or open state changes.
  useEffect(() => {
    if (!open) {
      setDraft(null)
      setSavedAt(null)
    }
  }, [open, path])

  const fsKind = !isCaption ? fsQuery.data?.kind : "text"
  const fsContent = !isCaption ? fsQuery.data?.content ?? "" : ""
  const captionContent = isCaption ? captionQuery.data?.caption ?? "" : ""
  const baseText = isCaption ? captionContent : fsContent
  const text = draft ?? baseText
  const dirty = draft !== null && draft !== baseText

  const save = useMutation({
    mutationFn: async (value: string) => {
      if (isCaption) {
        return api.putCaption((mode as { imagePath: string }).imagePath, value)
      }
      if (!path) throw new Error("missing path")
      return api.fsWrite(path, value)
    },
    onSuccess: async () => {
      setDraft(null)
      setSavedAt(Date.now())
      if (isCaption) {
        const m = mode as { imagePath: string; onAfterSave?: () => void }
        await qc.invalidateQueries({ queryKey: ["caption", m.imagePath] })
        await qc.invalidateQueries({ queryKey: ["dataset-scan"] })
        m.onAfterSave?.()
      } else if (path) {
        await qc.invalidateQueries({ queryKey: ["fs-read", path] })
      }
    },
  })

  const isLoading = isCaption ? captionQuery.isLoading : fsQuery.isLoading
  const errorMessage =
    (save.error instanceof Error && save.error.message) ||
    (!isCaption && fsQuery.error instanceof Error && fsQuery.error.message) ||
    (isCaption && captionQuery.error instanceof Error && captionQuery.error.message) ||
    null

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "s") {
      e.preventDefault()
      if (dirty && !save.isPending) save.mutate(text)
    }
    if (e.key === "Tab") {
      e.preventDefault()
      const ta = e.currentTarget
      const start = ta.selectionStart
      const end = ta.selectionEnd
      const next = text.substring(0, start) + "  " + text.substring(end)
      setDraft(next)
      requestAnimationFrame(() => {
        ta.selectionStart = ta.selectionEnd = start + 2
      })
    }
  }

  const fileName = path ? path.split(/[\\/]/).pop() : ""
  const renderImage = !isCaption && fsKind === "image" && path
  const renderBinary = !isCaption && fsKind === "binary"
  const renderText = isCaption || fsKind === "text" || (!fsQuery.data && !isCaption)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[min(calc(100%-2rem),64rem)] gap-3">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-base">
            {isCaption ? (
              <>
                <ImageIcon className="size-4 text-muted-foreground" />
                编辑 caption
              </>
            ) : (
              <>{fileName || "文件"}</>
            )}
            {dirty && (
              <Badge variant="outline" className="rounded-[2px] text-[10px] py-0 px-1.5">
                未保存
              </Badge>
            )}
          </DialogTitle>
          <DialogDescription className="font-mono break-all text-[11px]">
            {path ?? ""}
          </DialogDescription>
        </DialogHeader>

        {isLoading && (
          <div className="py-10 text-center text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin inline mr-2" /> 加载中…
          </div>
        )}

        {!isLoading && renderImage && (
          <div className="rounded-[4px] border border-border/60 bg-muted/30 p-3 flex items-center justify-center">
            <img
              src={api.datasetThumbUrl(path!, 1024)}
              alt={fileName ?? ""}
              className="max-h-[60vh] max-w-full object-contain"
            />
          </div>
        )}

        {!isLoading && renderBinary && (
          <div className="rounded-[4px] border border-border/60 bg-muted/30 px-4 py-6 text-sm text-muted-foreground text-center">
            该文件无法以文本方式打开
            {fsQuery.data?.reason ? `（${fsQuery.data.reason}）` : ""}
            。
          </div>
        )}

        {!isLoading && renderText && (
          <div className="flex flex-col gap-2 min-h-0">
            <textarea
              ref={taRef}
              value={text}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={handleKeyDown}
              spellCheck={false}
              className="w-full min-h-[40vh] max-h-[60vh] rounded-[3px] border border-input bg-background/80 px-3 py-2 text-[12px] font-mono leading-relaxed resize-y focus:outline-none focus:ring-2 focus:ring-ring/40"
              placeholder={
                isCaption ? "一行一段，或用逗号分隔的 kohya 标签" : "文件内容"
              }
              disabled={save.isPending}
            />
            <div className="flex items-center justify-between text-[10.5px] text-muted-foreground/85">
              <div className="flex items-center gap-3">
                <span>UTF-8 · 换行符 LF</span>
                <span>{text.length} 字符</span>
              </div>
              <span>
                Ctrl/⌘+S 保存 · Tab 缩进
                {savedAt && !dirty ? "  · 已保存" : ""}
              </span>
            </div>
          </div>
        )}

        {errorMessage && (
          <div className="rounded-[3px] border border-destructive/40 bg-destructive/5 px-3 py-2 text-[11px] font-mono text-destructive break-all">
            {errorMessage}
          </div>
        )}

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => onOpenChange(false)}
            disabled={save.isPending}
          >
            <X className="size-3" /> 关闭
          </Button>
          {(renderText || isCaption) && (
            <Button
              type="button"
              size="sm"
              onClick={() => save.mutate(text)}
              disabled={!dirty || save.isPending}
            >
              {save.isPending ? (
                <Loader2 className="size-3 animate-spin" />
              ) : (
                <Save className="size-3" />
              )}
              {save.isPending ? "保存中" : "保存"}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
