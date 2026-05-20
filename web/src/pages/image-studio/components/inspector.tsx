import { useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { FlipHorizontal, Heart, Pencil, RotateCw, Save, Star, Trash2, X } from "lucide-react"
import {
  imageStudioSaveAnnotation,
  imageStudioAddOp,
  imageStudioApplyOps,
} from "@/lib/api"
import type { ImageStudioDetailItem } from "@/lib/api"

interface InspectorProps {
  detail: ImageStudioDetailItem | null
  loading: boolean
  path: string
  onClose: () => void
}

export function Inspector({ detail, loading, path, onClose }: InspectorProps) {
  const queryClient = useQueryClient()
  const [editingCaption, setEditingCaption] = useState(false)
  const [captionDraft, setCaptionDraft] = useState("")
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)

  const invalidate = () =>
    queryClient.invalidateQueries({ queryKey: ["image-studio"] })

  const favMutation = useMutation({
    mutationFn: (fav: boolean) =>
      imageStudioSaveAnnotation({ path, favorite: fav }),
    onSuccess: invalidate,
  })

  const execMutation = useMutation({
    mutationFn: async (body: { op: string; payload?: Record<string, unknown> }) => {
      await imageStudioAddOp({ path, ...body })
      return imageStudioApplyOps(path)
    },
    onSuccess: invalidate,
  })

  const startCaptionEdit = () => {
    setCaptionDraft(detail?.caption || "")
    setEditingCaption(true)
  }

  const saveCaption = () => {
    execMutation.mutate({ op: "replace_caption", payload: { caption: captionDraft } })
    setEditingCaption(false)
  }

  const handleDelete = () => {
    execMutation.mutate({ op: "delete", payload: {} })
    setShowDeleteConfirm(false)
    onClose()
  }

  return (
    <aside className="shiro-page-aside w-[22rem] shrink-0 overflow-y-auto p-3">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium truncate">{detail?.name ?? "..."}</h3>
        <button
          type="button"
          onClick={onClose}
          className="rounded p-1 text-muted-foreground hover:bg-muted"
        >
          &times;
        </button>
      </div>

      {loading && <p className="text-xs text-muted-foreground">加载中...</p>}

      {detail && (
        <div className="flex flex-col gap-3">
          <div className="overflow-hidden rounded-md border">
            <img
              src={`/api/datasets/thumb?path=${encodeURIComponent(detail.path)}&size=1024`}
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
                <span>{detail.width} x {detail.height}</span>
              </>
            )}
          </div>

          <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium">描述</span>
              {!editingCaption && (
                <button
                  type="button"
                  onClick={startCaptionEdit}
                  className="rounded p-0.5 text-muted-foreground hover:bg-muted"
                >
                  <Pencil className="size-3" />
                </button>
              )}
            </div>
            {editingCaption ? (
              <div className="flex flex-col gap-1.5">
                <textarea
                  value={captionDraft}
                  onChange={(e) => setCaptionDraft(e.target.value)}
                  rows={4}
                  className="w-full rounded border bg-background px-2 py-1.5 text-xs outline-none focus:border-ring focus:ring-1 focus:ring-ring/30 resize-y"
                />
                <div className="flex gap-1.5">
                  <button
                    type="button"
                    onClick={saveCaption}
                    className="flex items-center gap-1 rounded bg-primary px-2 py-1 text-xs text-primary-foreground"
                  >
                    <Save className="size-3" /> 保存
                  </button>
                  <button
                    type="button"
                    onClick={() => setEditingCaption(false)}
                    className="flex items-center gap-1 rounded px-2 py-1 text-xs hover:bg-muted"
                  >
                    <X className="size-3" /> 取消
                  </button>
                </div>
              </div>
            ) : detail.caption ? (
              <p className="rounded bg-muted/50 p-2 text-xs leading-relaxed">
                {detail.caption}
              </p>
            ) : (
              <p className="text-xs text-muted-foreground italic">无描述文件</p>
            )}
          </div>

          {detail.annotation?.aiQualityLabel && (
            <div className="flex items-center gap-2">
              <Star className="size-3.5 text-amber-500" />
              <span className="text-xs">
                质量: {detail.annotation.aiQualityLabel}
                {detail.annotation.aiQualityScore != null &&
                  ` (${(detail.annotation.aiQualityScore * 100).toFixed(0)}%)`}
              </span>
            </div>
          )}

          <div className="flex flex-wrap gap-1.5 pt-2 border-t">
            <button
              type="button"
              onClick={() => favMutation.mutate(!detail.annotation?.favorite)}
              className={`flex items-center gap-1 rounded px-2 py-1 text-xs transition-colors ${
                detail.annotation?.favorite
                  ? "bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-400"
                  : "hover:bg-muted"
              }`}
            >
              <Heart className="size-3" />
              {detail.annotation?.favorite ? "取消收藏" : "收藏"}
            </button>
            <button
              type="button"
              onClick={() => execMutation.mutate({ op: "rotate", payload: { degrees: 90 } })}
              disabled={execMutation.isPending}
              className="flex items-center gap-1 rounded px-2 py-1 text-xs hover:bg-muted disabled:opacity-50"
            >
              <RotateCw className="size-3" /> 旋转
            </button>
            <button
              type="button"
              onClick={() => execMutation.mutate({ op: "flip", payload: { direction: "horizontal" } })}
              disabled={execMutation.isPending}
              className="flex items-center gap-1 rounded px-2 py-1 text-xs hover:bg-muted disabled:opacity-50"
            >
              <FlipHorizontal className="size-3" /> 翻转
            </button>
            <button
              type="button"
              onClick={() => setShowDeleteConfirm(true)}
              className="flex items-center gap-1 rounded px-2 py-1 text-xs text-destructive hover:bg-destructive/10"
            >
              <Trash2 className="size-3" /> 删除
            </button>
          </div>
        </div>
      )}

      {showDeleteConfirm && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
          onClick={() => setShowDeleteConfirm(false)}
        >
          <div
            className="w-80 rounded-lg border bg-popover p-4 shadow-lg"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-sm font-semibold mb-2">确认删除</h3>
            <p className="text-xs text-muted-foreground mb-4">
              确定要删除 &quot;{detail?.name}&quot; 吗？文件将移入回收站。
            </p>
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowDeleteConfirm(false)}
                className="rounded-md px-3 py-1.5 text-xs hover:bg-muted"
              >
                取消
              </button>
              <button
                type="button"
                onClick={handleDelete}
                className="rounded-md bg-destructive px-3 py-1.5 text-xs font-medium text-destructive-foreground"
              >
                删除
              </button>
            </div>
          </div>
        </div>
      )}
    </aside>
  )
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}