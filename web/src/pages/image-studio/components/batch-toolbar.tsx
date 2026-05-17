import { Heart, Sparkles, Trash2, Download, X } from "lucide-react"

interface BatchToolbarProps {
  count: number
  onDelete: () => void
  onFavorite: () => void
  onAiBulk: () => void
  onExport: () => void
  onClear: () => void
}

export function BatchToolbar({
  count,
  onDelete,
  onFavorite,
  onAiBulk,
  onExport,
  onClear,
}: BatchToolbarProps) {
  if (count === 0) return null

  return (
    <div className="fixed bottom-6 left-1/2 z-40 -translate-x-1/2 flex items-center gap-2 rounded-lg border bg-popover px-4 py-2 shadow-lg">
      <span className="text-xs font-medium mr-2">已选 {count} 张</span>
      <button
        type="button"
        onClick={onFavorite}
        className="flex items-center gap-1 rounded px-2 py-1 text-xs hover:bg-muted"
      >
        <Heart className="size-3" /> 批量收藏
      </button>
      <button
        type="button"
        onClick={onAiBulk}
        className="flex items-center gap-1 rounded px-2 py-1 text-xs hover:bg-muted"
      >
        <Sparkles className="size-3" /> AI标注
      </button>
      <button
        type="button"
        onClick={onExport}
        className="flex items-center gap-1 rounded px-2 py-1 text-xs hover:bg-muted"
      >
        <Download className="size-3" /> 导出
      </button>
      <button
        type="button"
        onClick={onDelete}
        className="flex items-center gap-1 rounded px-2 py-1 text-xs text-destructive hover:bg-destructive/10"
      >
        <Trash2 className="size-3" /> 批量删除
      </button>
      <button
        type="button"
        onClick={onClear}
        className="ml-2 rounded p-1 text-muted-foreground hover:bg-muted"
        title="清除选择 (Esc)"
      >
        <X className="size-3.5" />
      </button>
    </div>
  )
}
