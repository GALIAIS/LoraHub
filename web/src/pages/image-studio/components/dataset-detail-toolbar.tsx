import type { ReactNode } from "react"
import {
  Filter,
  FolderOpen,
  HelpCircle,
  ListChecks,
  Sparkles,
  Tag,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"
import { SortSelect, ViewChip } from "./dataset-detail-widgets"

type DatasetDetailToolbarProps = {
  datasetName: string
  path: string
  total?: number
  sort: string
  view: string
  recursive: boolean
  params: URLSearchParams
  setParams: (next: URLSearchParams) => void
  showFilters: boolean
  showTagging: boolean
  pendingOpsCount: number
  onBack: () => void
  onToggleFilters: () => void
  onToggleTagging: () => void
  onOpenAiBulk: () => void
  onOpenOpsQueue: () => void
  onOpenHelp: () => void
}

export function DatasetDetailToolbar({
  datasetName,
  path,
  total,
  sort,
  view,
  recursive,
  params,
  setParams,
  showFilters,
  showTagging,
  pendingOpsCount,
  onBack,
  onToggleFilters,
  onToggleTagging,
  onOpenAiBulk,
  onOpenOpsQueue,
  onOpenHelp,
}: DatasetDetailToolbarProps) {
  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-border/60 px-4 py-2">
      <div className="flex min-w-[220px] flex-1 items-center gap-2">
        <Button
          variant="ghost"
          size="sm"
          className="size-7 p-0"
          onClick={onBack}
          aria-label="返回数据集列表"
          title="返回数据集列表"
        >
          <FolderOpen className="size-4" />
        </Button>
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-medium">{datasetName}</span>
            {total != null && (
              <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
                {total} 张
              </span>
            )}
          </div>
          <div className="truncate font-mono text-[11px] text-muted-foreground">
            {path}
          </div>
        </div>
      </div>
      <div className="no-scrollbar flex max-w-full items-center gap-2 overflow-x-auto">
        <SortSelect
          value={sort}
          onChange={(s) => {
            const next = new URLSearchParams(params)
            next.set("sort", s)
            next.set("page", "1")
            setParams(next)
          }}
        />
        <div
          role="group"
          aria-label="视图模式"
          className="inline-flex h-7 shrink-0 items-center overflow-hidden rounded-[4px] border border-border/60 bg-background text-[11px]"
        >
          <ViewChip
            active={view === "grid"}
            onClick={() => {
              const next = new URLSearchParams(params)
              next.set("view", "grid")
              setParams(next)
            }}
          >
            网格
          </ViewChip>
          <span className="h-full w-px bg-border/60" aria-hidden />
          <ViewChip
            active={view === "duplicates"}
            onClick={() => {
              const next = new URLSearchParams(params)
              next.set("view", "duplicates")
              setParams(next)
            }}
          >
            去重
          </ViewChip>
        </div>
        <label className="inline-flex shrink-0 cursor-pointer select-none items-center gap-1.5 text-[11px] text-muted-foreground">
          <Switch
            size="sm"
            checked={recursive}
            onCheckedChange={(checked) => {
              const next = new URLSearchParams(params)
              if (checked) next.set("recursive", "1")
              else next.delete("recursive")
              next.set("page", "1")
              setParams(next)
            }}
          />
          递归
        </label>
        <ToolbarIconButton
          active={showFilters}
          label="筛选面板"
          onClick={onToggleFilters}
        >
          <Filter className="size-4" />
        </ToolbarIconButton>
        <ToolbarIconButton
          active={showTagging}
          label="WD14 标注"
          onClick={onToggleTagging}
        >
          <Tag className="size-4" />
        </ToolbarIconButton>
        <ToolbarIconButton label="AI 批量操作" onClick={onOpenAiBulk}>
          <Sparkles className="size-4" />
        </ToolbarIconButton>
        <Button
          variant="ghost"
          size="sm"
          onClick={onOpenOpsQueue}
          className={cn(
            "relative size-7 p-0",
            pendingOpsCount > 0 && "text-primary",
          )}
          aria-label={`待应用操作 (${pendingOpsCount})`}
          title={
            pendingOpsCount > 0
              ? `${pendingOpsCount} 个待应用操作`
              : "待应用操作 (空)"
          }
        >
          <ListChecks className="size-4" />
          {pendingOpsCount > 0 && (
            <span className="absolute -right-0.5 -top-0.5 inline-flex h-3.5 min-w-3.5 items-center justify-center rounded-full bg-primary px-1 text-[9px] font-semibold leading-none text-primary-foreground">
              {pendingOpsCount > 99 ? "99+" : pendingOpsCount}
            </span>
          )}
        </Button>
        <ToolbarIconButton label="键盘快捷键 (?)" onClick={onOpenHelp}>
          <HelpCircle className="size-4" />
        </ToolbarIconButton>
      </div>
    </div>
  )
}

function ToolbarIconButton({
  active,
  children,
  label,
  onClick,
}: {
  active?: boolean
  children: ReactNode
  label: string
  onClick: () => void
}) {
  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={onClick}
      className={cn("size-7 p-0", active && "bg-muted text-primary")}
      aria-label={label}
      aria-pressed={active}
      title={label}
    >
      {children}
    </Button>
  )
}
