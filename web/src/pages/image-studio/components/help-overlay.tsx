import { X } from "lucide-react"

export function HelpOverlay({ onClose }: { onClose: () => void }) {
  const shortcuts = [
    { key: "j / k", desc: "在网格中向下/向上导航" },
    { key: "Space", desc: "预览选中图片" },
    { key: "x", desc: "切换多选" },
    { key: "e", desc: "编辑描述" },
    { key: "q", desc: "质量评分" },
    { key: "d", desc: "软删除" },
    { key: "Ctrl+A", desc: "全选当前页" },
    { key: "Escape", desc: "清除选择/关闭面板" },
    { key: "?", desc: "切换帮助面板" },
  ]

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={onClose}
    >
      <div
        className="w-80 rounded-lg border bg-popover p-4 shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold">键盘快捷键</h3>
          <button type="button" onClick={onClose} className="rounded p-1 hover:bg-muted">
            <X className="size-4" />
          </button>
        </div>
        <div className="flex flex-col gap-1.5">
          {shortcuts.map((s) => (
            <div key={s.key} className="flex items-center justify-between text-xs">
              <kbd className="rounded border bg-muted px-1.5 py-0.5 font-mono text-[11px]">
                {s.key}
              </kbd>
              <span className="text-muted-foreground">{s.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
