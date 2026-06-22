/**
 * 图像工作台"全部工具"广场页面。
 *
 * 入口在主区顶部 — 当 URL 是 /image-studio (或 ?stage=tools) 时显示。
 * 卡片直接跳到 /image-studio/tools/<id>?path=<datasetPath>，每个工具有独立页面。
 *
 * 行为约束：
 *  - 不强制选数据集 — 没选时灰显 requiresDataset=true 的卡片，不直接 disable。
 *  - 用 <Link> 而不是按钮，方便右键"在新标签页打开"。
 *  - 卡片右下角的小标用来透露"异步会话"或"会写文件 / 自动备份"语义。
 */
import { Link } from "react-router-dom"
import { Link2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { TOOL_CATEGORIES, TOOLS, type ToolInfo } from "../tools-catalog"

interface Props {
  datasetPath: string
}

export function ToolsGrid({ datasetPath }: Props) {
  const hasDataset = Boolean(datasetPath)
  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto px-4 py-4 md:px-6 md:py-5 space-y-5">
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
                    datasetPath={datasetPath}
                    disabled={tool.requiresDataset && !hasDataset}
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
  datasetPath,
  disabled,
}: {
  tool: ToolInfo
  datasetPath: string
  disabled: boolean
}) {
  const Icon = tool.icon
  const to = datasetPath
    ? `/image-studio/tools/${tool.id}?path=${encodeURIComponent(datasetPath)}`
    : `/image-studio/tools/${tool.id}`

  const inner = (
    <>
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
    </>
  )

  const baseCls =
    "shiro-surface group relative flex flex-col items-start gap-1.5 px-3 py-2.5 text-left transition-colors"

  if (disabled) {
    return (
      <div
        className={cn(baseCls, "opacity-50 cursor-not-allowed")}
        title="请先在左侧选择一个数据集"
        aria-disabled
      >
        {inner}
      </div>
    )
  }
  return (
    <Link
      to={to}
      className={cn(baseCls, "hover:border-primary/40 hover:bg-accent/40")}
      title={tool.description}
    >
      {inner}
    </Link>
  )
}
