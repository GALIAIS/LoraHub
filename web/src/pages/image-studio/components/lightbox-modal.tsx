import { useCallback, useEffect, useState } from "react"
import {
  ChevronLeft,
  ChevronRight,
  Download,
  ExternalLink,
  Heart,
  ImageOff,
  Loader2,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react"
import { api } from "@/lib/api"
import type { ImageStudioItem } from "@/lib/api"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
} from "@/components/ui/dialog"
import { cn } from "@/lib/utils"

/**
 * Full-screen image viewer for the workbench. Inspector lives in the
 * right sidebar and tops out at 22rem; this is a separate "press F to
 * see the picture full size" mode you can flip through with arrow keys.
 *
 * Zoom is purely view-side: it scales the <img> via CSS transform and
 * never re-fetches at a higher resolution, so it stays snappy even on
 * the large datasets where browsers hate decoding 50 MB PNGs.
 */
export interface LightboxModalProps {
  open: boolean
  items: ImageStudioItem[]
  index: number
  onIndexChange: (next: number) => void
  onClose: () => void
  onToggleFavorite?: (item: ImageStudioItem) => void
}

export function LightboxModal({
  open,
  items,
  index,
  onIndexChange,
  onClose,
  onToggleFavorite,
}: LightboxModalProps) {
  const [zoom, setZoom] = useState(1)
  const item = items[index]

  // Reset zoom whenever the active image changes — staying at 3× on
  // a tiny image after flipping past a huge one looks broken.
  const [loaded, setLoaded] = useState(false)
  const [broken, setBroken] = useState(false)

  useEffect(() => {
    setZoom(1)
    setLoaded(false)
    setBroken(false)
  }, [item?.path])

  const goPrev = useCallback(() => {
    if (index > 0) onIndexChange(index - 1)
  }, [index, onIndexChange])
  const goNext = useCallback(() => {
    if (index < items.length - 1) onIndexChange(index + 1)
  }, [index, items.length, onIndexChange])

  // Arrow-key + zoom shortcuts. Listening only while open keeps these
  // from leaking to other surfaces (image-grid already binds j/k).
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === "ArrowLeft") {
        e.preventDefault()
        goPrev()
      } else if (e.key === "ArrowRight") {
        e.preventDefault()
        goNext()
      } else if (e.key === "+" || e.key === "=") {
        e.preventDefault()
        setZoom((z) => Math.min(z + 0.25, 4))
      } else if (e.key === "-") {
        e.preventDefault()
        setZoom((z) => Math.max(z - 0.25, 0.25))
      } else if (e.key === "0") {
        e.preventDefault()
        setZoom(1)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [open, goPrev, goNext])

  const filename = item?.name ?? ""
  const rawUrl = item ? api.datasetThumbUrl(item.path, 4096) : ""

  return (
    <Dialog open={open} onOpenChange={(next) => !next && onClose()}>
      <DialogContent
        showCloseButton={false}
        compositorSafe
        className="max-w-[min(96vw,1400px)] gap-0 p-0"
      >
        {item && (
          <div className="flex flex-col h-full max-h-[calc(100dvh-2rem)]">
            <div className="flex items-center gap-2 px-3 py-2 border-b border-border/60 bg-background/40 shrink-0">
              <span className="font-mono text-[12px] truncate flex-1" title={item.path}>
                {filename}
              </span>
              <span className="text-[10px] text-muted-foreground tabular-nums">
                {index + 1} / {items.length}
              </span>
              <div className="inline-flex items-center rounded-[3px] border border-border/60 overflow-hidden text-[11px]">
                <button
                  type="button"
                  onClick={() => setZoom((z) => Math.max(z - 0.25, 0.25))}
                  className="px-1.5 h-7 hover:bg-muted/60 transition-colors"
                  title="缩小 (-)"
                >
                  <ZoomOut className="size-3.5" />
                </button>
                <span className="px-2 h-7 inline-flex items-center font-mono tabular-nums border-x border-border/60">
                  {Math.round(zoom * 100)}%
                </span>
                <button
                  type="button"
                  onClick={() => setZoom((z) => Math.min(z + 0.25, 4))}
                  className="px-1.5 h-7 hover:bg-muted/60 transition-colors"
                  title="放大 (+)"
                >
                  <ZoomIn className="size-3.5" />
                </button>
              </div>
              {onToggleFavorite && (
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={() => onToggleFavorite(item)}
                  title={item.annotation?.favorite ? "取消收藏" : "收藏"}
                >
                  <Heart
                    className={cn(
                      "size-4",
                      item.annotation?.favorite && "fill-rose-500 text-rose-500",
                    )}
                  />
                </Button>
              )}
              <Button
                variant="ghost"
                size="icon-sm"
                render={<a href={rawUrl} download={filename} />}
                title="下载"
              >
                <Download className="size-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => window.open(rawUrl, "_blank")}
                title="在新标签页打开"
              >
                <ExternalLink className="size-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={onClose}
                title="关闭 (Esc)"
              >
                <X className="size-4" />
              </Button>
            </div>

            <div className="relative flex-1 min-h-0 grid place-items-center bg-black/86 overflow-auto">
              {index > 0 && (
                <button
                  type="button"
                  onClick={goPrev}
                  className="absolute left-2 top-1/2 -translate-y-1/2 rounded-full bg-background/70 p-1.5 text-foreground border border-border/60 hover:bg-background transition-colors z-10"
                  title="上一张 (←)"
                >
                  <ChevronLeft className="size-5" />
                </button>
              )}
              {index < items.length - 1 && (
                <button
                  type="button"
                  onClick={goNext}
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full bg-background/70 p-1.5 text-foreground border border-border/60 hover:bg-background transition-colors z-10"
                  title="下一张 (→)"
                >
                  <ChevronRight className="size-5" />
                </button>
              )}
              {!loaded && !broken && (
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
                  <Loader2 className="size-7 animate-spin text-muted-foreground/70" />
                </div>
              )}
              {broken ? (
                <div className="flex flex-col items-center justify-center gap-2 text-muted-foreground/70 px-6 py-10">
                  <ImageOff className="size-10" />
                  <span className="text-xs">无法加载该图片</span>
                </div>
              ) : (
                <img
                  src={rawUrl}
                  alt={item.name}
                  style={{
                    transform: `scale(${zoom})`,
                    transformOrigin: "center center",
                    transition: "transform 120ms ease-out",
                  }}
                  className={cn(
                    "max-w-full max-h-[calc(100dvh-10rem)] object-contain select-none",
                    !loaded && "opacity-0",
                  )}
                  draggable={false}
                  onLoad={() => setLoaded(true)}
                  onError={() => {
                    setBroken(true)
                    setLoaded(true)
                  }}
                />
              )}
            </div>

            <div className="flex items-center gap-3 px-3 py-1.5 border-t border-border/60 bg-background/40 text-[11px] text-muted-foreground shrink-0">
              {item.width && item.height && (
                <span className="tabular-nums">
                  {item.width}×{item.height}
                </span>
              )}
              <span className="tabular-nums">{formatBytes(item.bytes)}</span>
              {item.annotation?.aiQualityLabel && (
                <span>· 质量：{item.annotation.aiQualityLabel}</span>
              )}
              {item.captionExists ? (
                <span className="ml-auto truncate flex-1 max-w-[60%] text-right">
                  {item.caption || ""}
                </span>
              ) : (
                <span className="ml-auto text-amber-600 dark:text-amber-400">
                  无描述
                </span>
              )}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
