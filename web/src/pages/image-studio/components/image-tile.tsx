import { useState } from "react"
import { Heart, ImageOff, Trash2 } from "lucide-react"
import type { ImageStudioItem } from "@/lib/api"
import {
  ContextMenu,
  ContextMenuTrigger,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
} from "@/components/ui/context-menu"

interface ImageTileProps {
  item: ImageStudioItem
  selected: boolean
  multiSelected: boolean
  onClick: () => void
  onCtrlClick: () => void
  onDoubleClick?: () => void
  onContextAction: (action: string, item: ImageStudioItem) => void
}

export function ImageTile({
  item,
  selected,
  multiSelected,
  onClick,
  onCtrlClick,
  onDoubleClick,
  onContextAction,
}: ImageTileProps) {
  const [broken, setBroken] = useState(false)
  const handleClick = (e: React.MouseEvent) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault()
      onCtrlClick()
    } else {
      onClick()
    }
  }

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <button
          type="button"
          onClick={handleClick}
          onDoubleClick={onDoubleClick}
          className={`group relative flex w-full max-w-full min-w-0 flex-col overflow-hidden rounded-md border transition-colors ${
            selected
              ? "border-primary ring-2 ring-primary/30"
              : multiSelected
                ? "border-blue-500 ring-2 ring-blue-500/30"
                : "border-border hover:border-muted-foreground/40"
          }`}
        >
          <div className="aspect-square w-full overflow-hidden bg-muted shrink-0">
            {broken ? (
              <div className="flex h-full w-full flex-col items-center justify-center gap-1 text-muted-foreground/70">
                <ImageOff className="size-5" />
                <span className="text-[10px]">缩略图未生成</span>
              </div>
            ) : (
              <img
                src={item.thumbUrl}
                alt={item.name}
                loading="lazy"
                draggable={false}
                onError={() => setBroken(true)}
                className="block h-full w-full object-cover"
              />
            )}
          </div>
          <div className="flex min-w-0 items-center gap-1 px-1.5 py-1">
            {multiSelected && (
              <span className="size-3 rounded-sm border border-blue-500 bg-blue-500/20 shrink-0" />
            )}
            <span className="min-w-0 flex-1 truncate text-left text-[11px]">
              {item.name}
            </span>
            {item.annotation?.favorite && (
              <Heart className="size-3 shrink-0 fill-rose-500 text-rose-500" />
            )}
            {item.annotation?.softDeleted && (
              <Trash2 className="size-3 shrink-0 text-muted-foreground" />
            )}
          </div>
          {!item.captionExists && (
            <div className="absolute right-1 top-1 rounded bg-amber-500/80 px-1 py-0.5 text-[9px] font-medium text-white">
              无描述
            </div>
          )}
        </button>
      </ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem onClick={() => onContextAction("inspect", item)}>
          打开详情
        </ContextMenuItem>
        <ContextMenuItem onClick={() => onContextAction("lightbox", item)}>
          全屏查看
        </ContextMenuItem>
        <ContextMenuItem onClick={() => onContextAction("edit-caption", item)}>
          编辑描述
        </ContextMenuItem>
        <ContextMenuItem onClick={() => onContextAction("toggle-fav", item)}>
          {item.annotation?.favorite ? "取消收藏" : "收藏"}
        </ContextMenuItem>
        <ContextMenuItem onClick={() => onContextAction("rotate", item)}>
          旋转 90°
        </ContextMenuItem>
        <ContextMenuItem onClick={() => onContextAction("flip", item)}>
          水平翻转
        </ContextMenuItem>
        <ContextMenuItem onClick={() => onContextAction("copy-path", item)}>
          复制路径
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem
          onClick={() => onContextAction("delete", item)}
          className="text-destructive focus:text-destructive"
        >
          删除
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  )
}
