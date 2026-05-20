import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"

const SHORTCUTS: Array<{ key: string; desc: string; group: string }> = [
  { group: "导航", key: "j / k", desc: "网格中向下 / 向上选择" },
  { group: "导航", key: "Space", desc: "预览选中图片 (Inspector)" },
  { group: "导航", key: "F", desc: "全屏 lightbox 查看选中图" },
  { group: "导航", key: "← / →", desc: "lightbox 内翻页" },
  { group: "导航", key: "+ / - / 0", desc: "lightbox 缩放" },
  { group: "选择", key: "x", desc: "切换当前图的多选" },
  { group: "选择", key: "Ctrl+A", desc: "全选当前页" },
  { group: "选择", key: "Escape", desc: "清空选择 / 关闭面板" },
  { group: "操作", key: "e", desc: "编辑描述" },
  { group: "操作", key: "d", desc: "删除（弹确认）" },
  { group: "操作", key: "?", desc: "显示 / 隐藏此帮助面板" },
]

export function HelpOverlay({ onClose }: { onClose: () => void }) {
  return (
    <Dialog open onOpenChange={(next) => !next && onClose()}>
      <DialogContent className="sm:max-w-[420px]">
        <DialogHeader>
          <DialogTitle>键盘快捷键</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-1">
          {Array.from(new Set(SHORTCUTS.map((s) => s.group))).map((group) => (
            <div key={group} className="space-y-0.5">
              <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground/70 mt-2 first:mt-0">
                {group}
              </div>
              {SHORTCUTS.filter((s) => s.group === group).map((s) => (
                <div
                  key={`${group}-${s.key}`}
                  className="flex items-center justify-between text-xs gap-3"
                >
                  <kbd className="rounded border border-border/70 bg-muted/60 px-1.5 py-0.5 font-mono text-[11px] shrink-0">
                    {s.key}
                  </kbd>
                  <span className="text-muted-foreground text-right truncate">
                    {s.desc}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  )
}
