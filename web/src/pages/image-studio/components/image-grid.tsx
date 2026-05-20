import { useCallback, useRef } from "react"
import { VirtuosoGrid } from "react-virtuoso"
import type { ImageStudioItem } from "@/lib/api"
import { ImageTile } from "./image-tile"

interface ImageGridProps {
  items: ImageStudioItem[]
  selectedPath: string | null
  multiSelected: Set<string>
  onSelect: (path: string) => void
  onMultiToggle: (path: string) => void
  onDoubleSelect?: (path: string) => void
  onContextAction: (action: string, item: ImageStudioItem) => void
}

export function ImageGrid({
  items,
  selectedPath,
  multiSelected,
  onSelect,
  onMultiToggle,
  onDoubleSelect,
  onContextAction,
}: ImageGridProps) {
  const gridRef = useRef<HTMLDivElement>(null)

  const renderItem = useCallback(
    (index: number) => {
      const item = items[index]
      if (!item) return null
      return (
        <ImageTile
          item={item}
          selected={item.path === selectedPath}
          multiSelected={multiSelected.has(item.path)}
          onClick={() => onSelect(item.path)}
          onCtrlClick={() => onMultiToggle(item.path)}
          onDoubleClick={onDoubleSelect ? () => onDoubleSelect(item.path) : undefined}
          onContextAction={onContextAction}
        />
      )
    },
    [items, selectedPath, multiSelected, onSelect, onMultiToggle, onDoubleSelect, onContextAction],
  )

  if (items.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-muted-foreground">
        该目录下未找到图片，拖入文件或压缩包上传
      </div>
    )
  }

  // For smaller datasets, use a simple CSS grid (avoids virtualization overhead)
  if (items.length <= 200) {
    return (
      <div ref={gridRef} className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-2">
        {items.map((item, idx) => (
          <div key={item.path}>{renderItem(idx)}</div>
        ))}
      </div>
    )
  }

  // For large datasets, use virtualized grid
  return (
    <VirtuosoGrid
      totalCount={items.length}
      overscan={200}
      listClassName="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-2"
      itemContent={renderItem}
      style={{ height: "100%", width: "100%" }}
    />
  )
}
