/**
 * ChartToolbar — top-right floating control rail for SVG charts.
 *
 * Buttons are stateless from the toolbar's perspective; the parent owns
 * zoom / view state and reacts to callbacks. Buttons hide automatically
 * when they don't apply (e.g. zoom-out is disabled at full extent).
 *
 * Used by `LossChart` and any future chart that wants the same gesture
 * vocabulary. Kept small so it doesn't crowd the chart at default
 * (220-280 px) heights.
 */
import {
  Crosshair,
  Download,
  Maximize2,
  RotateCcw,
  ZoomIn,
  ZoomOut,
} from "lucide-react"
import { cn } from "@/lib/utils"

interface ChartToolbarProps {
  zoomedIn: boolean
  // Box-select mode: when true, mouse drag draws a selection rectangle
  // that becomes the new visible range.
  selectMode?: boolean
  onZoomIn?: () => void
  onZoomOut?: () => void
  onReset?: () => void
  onToggleSelect?: () => void
  onFullscreen?: () => void
  onDownload?: () => void
  className?: string
}

export function ChartToolbar({
  zoomedIn,
  selectMode = false,
  onZoomIn,
  onZoomOut,
  onReset,
  onToggleSelect,
  onFullscreen,
  onDownload,
  className,
}: ChartToolbarProps) {
  return (
    <div
      className={cn(
        "inline-flex items-center gap-0.5 rounded-[4px] border border-border/60 bg-background/85 backdrop-blur-sm px-0.5 py-0.5 shadow-[var(--panel-shadow)]",
        className,
      )}
    >
      {onZoomIn && (
        <ToolBtn label="放大" onClick={onZoomIn}>
          <ZoomIn className="size-3.5" />
        </ToolBtn>
      )}
      {onZoomOut && (
        <ToolBtn
          label={zoomedIn ? "缩小" : "已是全量视图"}
          onClick={onZoomOut}
          disabled={!zoomedIn}
        >
          <ZoomOut className="size-3.5" />
        </ToolBtn>
      )}
      {onToggleSelect && (
        <ToolBtn
          label={selectMode ? "退出框选" : "框选缩放"}
          onClick={onToggleSelect}
          active={selectMode}
        >
          <Crosshair className="size-3.5" />
        </ToolBtn>
      )}
      {onReset && (
        <ToolBtn
          label={zoomedIn ? "复位视图" : "已复位"}
          onClick={onReset}
          disabled={!zoomedIn}
        >
          <RotateCcw className="size-3.5" />
        </ToolBtn>
      )}
      {onFullscreen && (
        <ToolBtn label="全屏" onClick={onFullscreen}>
          <Maximize2 className="size-3.5" />
        </ToolBtn>
      )}
      {onDownload && (
        <ToolBtn label="下载/导出" onClick={onDownload}>
          <Download className="size-3.5" />
        </ToolBtn>
      )}
    </div>
  )
}

function ToolBtn({
  label,
  onClick,
  disabled,
  active,
  children,
}: {
  label: string
  onClick: () => void
  disabled?: boolean
  active?: boolean
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "inline-flex items-center justify-center size-6 rounded-[3px] transition-colors",
        active
          ? "bg-primary/15 text-primary"
          : "text-muted-foreground hover:bg-muted/60 hover:text-foreground",
        disabled && "opacity-40 cursor-not-allowed hover:bg-transparent hover:text-muted-foreground",
      )}
    >
      {children}
    </button>
  )
}
