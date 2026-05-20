import { useEffect, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { FlipHorizontal, Heart, Maximize2, Pencil, RotateCw, Save, Sparkles, Trash2, X } from "lucide-react"
import { toast } from "sonner"
import {
  api,
  imageStudioSaveAnnotation,
  imageStudioAddOp,
  imageStudioApplyOps,
  imageStudioSmartCaptionSingle,
} from "@/lib/api"
import type { ImageStudioDetailItem } from "@/lib/api"
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
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { TagChipEditor } from "./tag-chip-editor"

interface InspectorProps {
  detail: ImageStudioDetailItem | null
  loading: boolean
  path: string
  onClose: () => void
  onOpenLightbox?: () => void
}

export function Inspector({ detail, loading, path, onClose, onOpenLightbox }: InspectorProps) {
  const queryClient = useQueryClient()
  const [editingCaption, setEditingCaption] = useState(false)
  const [captionDraft, setCaptionDraft] = useState("")
  const [editingNotes, setEditingNotes] = useState(false)
  const [notesDraft, setNotesDraft] = useState("")
  const [optimisticCaption, setOptimisticCaption] = useState<string | null>(null)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)

  // Reset edit drafts whenever the inspector switches to a new image,
  // otherwise the next image inherits the previous draft state.
  useEffect(() => {
    setEditingCaption(false)
    setEditingNotes(false)
    setOptimisticCaption(null)
    setCaptionDraft(detail?.caption ?? "")
    setNotesDraft(detail?.annotation?.userNotes ?? "")
  }, [detail?.path, detail?.caption, detail?.annotation?.userNotes])

  const invalidate = () =>
    queryClient.invalidateQueries({ queryKey: ["image-studio"] })

  const favMutation = useMutation({
    mutationFn: (fav: boolean) =>
      imageStudioSaveAnnotation({ path, favorite: fav }),
    onSuccess: invalidate,
  })

  const qualityMutation = useMutation({
    mutationFn: (label: string | null) =>
      imageStudioSaveAnnotation({ path, userQualityLabel: label }),
    onSuccess: invalidate,
  })

  const notesMutation = useMutation({
    mutationFn: (notes: string) =>
      imageStudioSaveAnnotation({ path, userNotes: notes }),
    onSuccess: () => {
      setEditingNotes(false)
      invalidate()
    },
  })

  const execMutation = useMutation({
    mutationFn: async (body: { op: string; payload?: Record<string, unknown> }) => {
      await imageStudioAddOp({ path, ...body })
      return imageStudioApplyOps(path)
    },
    onSuccess: () => {
      setOptimisticCaption(null)
      invalidate()
    },
    onError: (e) => {
      // Roll back the optimistic value if the save failed; the server
      // is still authoritative.
      setOptimisticCaption(null)
      toast.error("保存失败", {
        description: e instanceof Error ? e.message : String(e),
      })
    },
  })

  const reCaptionMutation = useMutation({
    mutationFn: () => imageStudioSmartCaptionSingle({ path }),
    onSuccess: () => {
      toast.success("已重新生成描述")
      invalidate()
    },
    onError: (e) => {
      toast.error("AI 重标失败", {
        description: e instanceof Error ? e.message : String(e),
      })
    },
  })

  const startCaptionEdit = () => {
    setCaptionDraft((optimisticCaption ?? detail?.caption) || "")
    setEditingCaption(true)
  }

  const saveCaption = () => {
    // Show the new caption immediately so the textarea collapse doesn't
    // reveal the stale value while the op queue applies.
    setOptimisticCaption(captionDraft)
    setEditingCaption(false)
    execMutation.mutate({ op: "replace_caption", payload: { caption: captionDraft } })
  }

  const handleDelete = () => {
    execMutation.mutate({ op: "delete", payload: {} })
    setShowDeleteConfirm(false)
    onClose()
  }

  const displayedCaption = optimisticCaption ?? detail?.caption ?? ""

  return (
    <aside className="shiro-page-aside w-[22rem] shrink-0 overflow-y-auto p-3">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium truncate">{detail?.name ?? "..."}</h3>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onClose}
          aria-label="关闭"
        >
          <X className="size-4" />
        </Button>
      </div>

      {loading && <p className="text-xs text-muted-foreground">加载中...</p>}

      {detail && (
        <div className="flex flex-col gap-3">
          <div className="overflow-hidden rounded-md border">
            <img
              src={api.datasetThumbUrl(detail.path, 1024)}
              alt={detail.name}
              className="w-full"
            />
          </div>

          <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
            <span className="text-muted-foreground">文件大小</span>
            <span>{formatBytes(detail.bytes)}</span>
            {detail.width && detail.height && (
              <>
                <span className="text-muted-foreground">尺寸</span>
                <span>
                  {detail.width} x {detail.height}
                  <span className="ml-1 text-muted-foreground">
                    ({(detail.width / detail.height).toFixed(2)})
                  </span>
                </span>
              </>
            )}
          </div>

          <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium">描述（标签）</span>
              <div className="flex items-center gap-0.5">
                {!editingCaption && onOpenLightbox && (
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={onOpenLightbox}
                    title="全屏查看 (F)"
                  >
                    <Maximize2 className="size-3" />
                  </Button>
                )}
                {!editingCaption && (
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={() => reCaptionMutation.mutate()}
                    disabled={reCaptionMutation.isPending}
                    title="对这张图 AI 重新打标"
                  >
                    <Sparkles className="size-3" />
                  </Button>
                )}
                {!editingCaption && (
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={startCaptionEdit}
                    title="编辑描述"
                  >
                    <Pencil className="size-3" />
                  </Button>
                )}
              </div>
            </div>
            {editingCaption ? (
              <TagChipEditor
                value={captionDraft}
                onChange={setCaptionDraft}
                onSave={saveCaption}
                onCancel={() => setEditingCaption(false)}
                disabled={execMutation.isPending}
              />
            ) : displayedCaption ? (
              <p className="rounded bg-muted/50 p-2 text-xs leading-relaxed whitespace-pre-wrap break-words">
                {displayedCaption}
              </p>
            ) : (
              <p className="text-xs text-muted-foreground italic">无描述文件</p>
            )}
          </div>

          {detail.annotation?.aiTriggerWords && detail.annotation.aiTriggerWords.length > 0 && (
            <div className="flex flex-col gap-1">
              <span className="text-xs font-medium">触发词建议</span>
              <div className="flex flex-wrap gap-1">
                {detail.annotation.aiTriggerWords.map((w) => (
                  <Badge key={w} variant="secondary" className="rounded-[2px] text-[10px]">
                    {w}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {detail.annotation?.aiComposition && (
            <div className="flex flex-col gap-1">
              <span className="text-xs font-medium">构图分析</span>
              <p className="rounded bg-muted/50 p-2 text-[11px] leading-relaxed whitespace-pre-wrap">
                {detail.annotation.aiComposition}
              </p>
            </div>
          )}

          {(detail.annotation?.aiQualityLabel || detail.annotation?.aiQualityReason) && (
            <div className="flex flex-col gap-1">
              <span className="text-xs font-medium">AI 质量评分</span>
              <div className="rounded bg-muted/50 p-2 text-[11px] space-y-1">
                <div className="flex items-center gap-2">
                  {detail.annotation.aiQualityLabel && (
                    <Badge
                      variant="outline"
                      className="rounded-[2px] text-[10px]"
                    >
                      {detail.annotation.aiQualityLabel}
                    </Badge>
                  )}
                  {detail.annotation.aiQualityScore != null && (
                    <span className="font-mono tabular-nums">
                      {(detail.annotation.aiQualityScore * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
                {detail.annotation.aiQualityReason && (
                  <p className="text-muted-foreground leading-relaxed whitespace-pre-wrap">
                    {detail.annotation.aiQualityReason}
                  </p>
                )}
              </div>
            </div>
          )}

          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium">人工质量评级</span>
            <div className="flex gap-1">
              {(["good", "ok", "bad"] as const).map((label) => {
                const active = detail.annotation?.userQualityLabel === label
                return (
                  <Button
                    key={label}
                    variant={active ? "default" : "outline"}
                    size="sm"
                    className="h-7 flex-1 text-[11px]"
                    onClick={() =>
                      qualityMutation.mutate(active ? null : label)
                    }
                    disabled={qualityMutation.isPending}
                  >
                    {label === "good" ? "优" : label === "ok" ? "中" : "差"}
                  </Button>
                )
              })}
            </div>
          </div>

          <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium">备注</span>
              {!editingNotes && (
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={() => setEditingNotes(true)}
                  title="编辑备注"
                >
                  <Pencil className="size-3" />
                </Button>
              )}
            </div>
            {editingNotes ? (
              <div className="flex flex-col gap-1.5">
                <textarea
                  value={notesDraft}
                  onChange={(e) => setNotesDraft(e.target.value)}
                  rows={3}
                  className="w-full rounded border bg-background px-2 py-1.5 text-xs outline-none focus:border-ring focus:ring-1 focus:ring-ring/30 resize-y"
                  placeholder="自己的笔记，不会写入到 caption 文件"
                />
                <div className="flex gap-1.5">
                  <Button
                    size="sm"
                    onClick={() => notesMutation.mutate(notesDraft)}
                    disabled={notesMutation.isPending}
                    className="h-7 text-[11px]"
                  >
                    <Save className="size-3" /> 保存
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setEditingNotes(false)
                      setNotesDraft(detail.annotation?.userNotes ?? "")
                    }}
                    className="h-7 text-[11px]"
                  >
                    <X className="size-3" /> 取消
                  </Button>
                </div>
              </div>
            ) : detail.annotation?.userNotes ? (
              <p className="rounded bg-muted/50 p-2 text-xs leading-relaxed whitespace-pre-wrap break-words">
                {detail.annotation.userNotes}
              </p>
            ) : (
              <p className="text-xs text-muted-foreground italic">无备注</p>
            )}
          </div>

          <div className="flex flex-wrap gap-1.5 pt-2 border-t">
            <Button
              variant={detail.annotation?.favorite ? "default" : "outline"}
              size="sm"
              className="h-7 text-[11px]"
              onClick={() => favMutation.mutate(!detail.annotation?.favorite)}
              disabled={favMutation.isPending}
            >
              <Heart
                className={`size-3 ${detail.annotation?.favorite ? "fill-current" : ""}`}
              />
              {detail.annotation?.favorite ? "取消收藏" : "收藏"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-7 text-[11px]"
              onClick={() => execMutation.mutate({ op: "rotate", payload: { degrees: 90 } })}
              disabled={execMutation.isPending}
            >
              <RotateCw className="size-3" /> 旋转
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-7 text-[11px]"
              onClick={() => execMutation.mutate({ op: "flip", payload: { direction: "horizontal" } })}
              disabled={execMutation.isPending}
            >
              <FlipHorizontal className="size-3" /> 翻转
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-[11px] text-destructive hover:text-destructive"
              onClick={() => setShowDeleteConfirm(true)}
            >
              <Trash2 className="size-3" /> 删除
            </Button>
          </div>
        </div>
      )}

      <AlertDialog
        open={showDeleteConfirm}
        onOpenChange={(next) => !next && setShowDeleteConfirm(false)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>确认删除</AlertDialogTitle>
            <AlertDialogDescription>
              确定要删除 <code className="font-mono">{detail?.name}</code> 吗？
              文件将移入回收站，可在数据集级别恢复。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={(e) => {
                e.preventDefault()
                handleDelete()
              }}
            >
              删除
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </aside>
  )
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
