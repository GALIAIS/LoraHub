/**
 * 轻量级"从工具库选触发词"弹层 — 给 ai-bulk-modal 的触发词字段用。
 *
 * 设计选择:
 *  - 不用 dialog (会跟 ai-bulk-modal 自身的 fixed 蒙层抢 z-index),
 *    用 absolute 浮层贴在按钮右下方;
 *  - 简单的搜索框 + 列表, 不做分类 / 标签 等高级筛选(picker 不是主面板);
 *  - 选完 onSelect(triggerWord) 后由调用方关闭。
 */
import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Search, X, BookOpen } from "lucide-react"
import { libraryListTriggers } from "@/lib/api"
import { cn } from "@/lib/utils"

interface Props {
  /** 选中后回调,把 trigger word 填回到调用方的输入框里。 */
  onSelect: (triggerWord: string) => void
  /** 关闭弹层。 */
  onClose: () => void
}

export function TriggerPicker({ onSelect, onClose }: Props) {
  const [search, setSearch] = useState("")
  const triggersQuery = useQuery({
    queryKey: ["library", "triggers", "picker", search],
    queryFn: () => libraryListTriggers({ search: search || undefined }),
  })
  const triggers = triggersQuery.data?.triggers ?? []

  return (
    <div
      className={cn(
        "absolute right-0 top-full mt-1 z-10 w-72 max-h-72 flex flex-col",
        "rounded-md border bg-popover shadow-lg",
      )}
    >
      <div className="flex items-center gap-1.5 border-b px-2 py-1.5">
        <BookOpen className="size-3.5 text-muted-foreground" />
        <span className="text-[11px] font-medium">从工具库选触发词</span>
        <button
          type="button"
          onClick={onClose}
          className="ml-auto rounded p-1 hover:bg-muted"
        >
          <X className="size-3" />
        </button>
      </div>
      <div className="border-b px-2 py-1.5">
        <div className="relative">
          <Search className="size-3 absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="搜索 trigger / 角色 / 概念"
            className="w-full rounded border bg-background pl-7 pr-2 py-1 text-xs outline-none focus:border-ring"
            autoFocus
          />
        </div>
      </div>
      <div className="flex-1 overflow-y-auto">
        {triggersQuery.isLoading ? (
          <p className="px-3 py-4 text-center text-[11px] text-muted-foreground">
            加载中…
          </p>
        ) : triggers.length === 0 ? (
          <p className="px-3 py-4 text-center text-[11px] text-muted-foreground">
            {search
              ? "未找到匹配的触发词"
              : "工具库还没有触发词,先去 工具库 → 触发词索引 添加。"}
          </p>
        ) : (
          <ul>
            {triggers.map((t) => (
              <li key={t.triggerWord}>
                <button
                  type="button"
                  onClick={() => {
                    onSelect(t.triggerWord)
                    onClose()
                  }}
                  className="w-full flex items-start gap-2 px-2 py-1.5 text-left hover:bg-accent/50 border-b last:border-b-0"
                >
                  <div className="flex-1 min-w-0">
                    <div className="font-mono text-xs truncate">
                      {t.triggerWord}
                    </div>
                    {(t.characterName || t.concept) && (
                      <div className="text-[10px] text-muted-foreground truncate">
                        {[t.characterName, t.concept].filter(Boolean).join(" · ")}
                      </div>
                    )}
                  </div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
