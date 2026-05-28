/**
 * 图像工作台"全部工具"广场页面。
 *
 * 入口在主区顶部 — 当 URL 没有 ?stage 参数（或显式 ?stage=tools）时显示。
 * 卡片网格按 5 大类聚合 13 个工具；每张卡片点击后写 URL `?stage=<stage>&tool=<id>`，
 * 让 stage 子页可以高亮对应面板。
 *
 * 行为约束：
 *  - 不强制选数据集 — 没选时灰显 requiresDataset=true 的卡片，不直接 disable。
 *  - 不在这里弹对话框 — 跳到 stage 子页让现成面板处理。
 *  - 卡片右下角的小标用来透露"异步会话"或"会写文件 / 自动备份"语义。
 */
import { Link2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { TOOL_CATEGORIES, TOOLS, type ToolInfo } from "../tools-catalog"

interface Props {
  datasetPath: string
  /** 选中工具后回调；调用方写 URL ?stage=<tool.stage>&tool=<tool.id>。 */
  onSelect: (tool: ToolInfo) => void
}

export function ToolsGrid({ datasetPath, onSelect }: Props) {
  const hasDataset = Boolean(datasetPath)
  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="border-b px-6 py-4">
        <h1 className="text-base font-semibold">全部工具</h1>
        <p className="mt-1 text-xs text-muted-foreground">
          {hasDataset
            ? "点任意工具直接跳到对应面板，无需走完整流程。"
            : "先在左侧选一个数据集，工具会进入对应面板。"}
        </p>
      </header>
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
        {TOOL_CATEGORIES.map((cat) => {
          const tools = TOOLS.filter((t) => t.category === cat.id)
          if (tools.length === 0) return null
          const CatIcon = cat.icon
          return (
            <section key={cat.id} className="space-y-2">
              <div className="flex items-center gap-2">
                <CatIcon className="size-4 text-muted-foreground" />
                <h2 className="text-sm font-medium">{cat.label}</h2>
                <span className="text-[10px] text-muted-foreground">
                  {cat.description}
                </span>
              </div>
              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {tools.map((tool) => (
                  <ToolCard
                    key={tool.id}
                    tool={tool}
                    disabled={tool.requiresDataset && !hasDataset}
                    onClick={() => onSelect(tool)}
                  />
                ))}
              </div>
            </section>
          )
        })}
      </div>
    </div>
  )
}

function ToolCard({
  tool,
  disabled,
  onClick,
}: {
  tool: ToolInfo
  disabled: boolean
  onClick: () => void
}) {
  const Icon = tool.icon
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "group relative flex flex-col items-start gap-1.5 rounded-md border bg-card px-3 py-2.5 text-left transition-colors",
        disabled
          ? "opacity-50 cursor-not-allowed"
          : "hover:border-primary/40 hover:bg-accent/40",
      )}
      title={disabled ? "请先在左侧选择一个数据集" : tool.description}
    >
      <div className="flex w-full items-center gap-2">
        <Icon className="size-4 text-muted-foreground group-hover:text-foreground shrink-0" />
        <span className="text-sm font-medium truncate flex-1">{tool.label}</span>
        <Link2 className="size-3 text-muted-foreground/60 shrink-0" />
      </div>
      <p className="text-[11px] text-muted-foreground line-clamp-2">
        {tool.description}
      </p>
      <div className="flex items-center gap-1 text-[10px]">
        {tool.async && (
          <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-amber-700 dark:text-amber-400">
            异步
          </span>
        )}
        {tool.writes && (
          <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-emerald-700 dark:text-emerald-400">
            写入
          </span>
        )}
      </div>
    </button>
  )
}
