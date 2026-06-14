import { useEffect, type ReactNode } from "react"
import { createPortal } from "react-dom"
import { X } from "lucide-react"

export function FullscreenModal({
  children,
  onClose,
}: {
  children: ReactNode
  onClose: () => void
}) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onClose])

  // Portal to <body> so the modal escapes any ancestor stacking context
  // (parent Cards / Tabs panels often create one via transform / isolate
  // / will-change). Without this a sibling chart Card rendered later in
  // the DOM can paint on top of our "fullscreen" view.
  if (typeof document === "undefined") return null
  return createPortal(
    <div
      className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/60 p-6"
      onClick={onClose}
    >
      <div
        className="relative w-[92vw] max-w-[1400px] rounded-[6px] border border-border/60 bg-background p-4 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          onClick={onClose}
          className="absolute right-3 top-3 z-20 inline-flex size-7 items-center justify-center rounded-[3px] text-muted-foreground hover:bg-muted hover:text-foreground"
          aria-label="关闭全屏"
        >
          <X className="size-4" />
        </button>
        {children}
      </div>
    </div>,
    document.body,
  )
}
