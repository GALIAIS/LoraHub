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
    <div className="flex items-center gap-2 border-b border-border/60 px-4 py-2">
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
      <span className="font-medium text-sm">{datasetName}</span>
      <span className="font-mono text-xs text-muted-foreground truncate flex-1">
        {path}
      </span>
      {total != null && (
        <span className="text-xs text-muted-foreground">{total} 张图片</span>
      )}
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
        className="inline-flex h-7 items-center rounded-[4px] border border-border/60 bg-background overflow-hidden text-[11px]"
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
      <label className="inline-flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer select-none">
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
      <Button
        variant="ghost"
        size="sm"
        onClick={onToggleFilters}
        className={cn("size-7 p-0", showFilters && "bg-muted text-primary")}
        aria-label="筛选面板"
        aria-pressed={showFilters}
        title="筛选面板"
      >
        <Filter className="size-4" />
      </Button>
      <Button
        variant="ghost"
        size="sm"
        onClick={onToggleTagging}
        className={cn("size-7 p-0", showTagging && "bg-muted text-primary")}
        aria-label="WD14 标注"
        aria-pressed={showTagging}
        title="WD14 标注"
      >
        <Tag className="size-4" />
      </Button>
      <Button
        variant="ghost"
        size="sm"
        onClick={onOpenAiBulk}
        className="size-7 p-0"
        aria-label="AI 批量操作"
        title="AI 批量操作"
      >
        <Sparkles className="size-4" />
      </Button>
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
          <span className="absolute -top-0.5 -right-0.5 inline-flex h-3.5 min-w-3.5 items-center justify-center rounded-full bg-primary px-1 text-[9px] font-semibold leading-none text-primary-foreground">
            {pendingOpsCount > 99 ? "99+" : pendingOpsCount}
          </span>
        )}
      </Button>
      <Button
        variant="ghost"
        size="sm"
        onClick={onOpenHelp}
        className="size-7 p-0"
        aria-label="键盘快捷键 (?)"
        title="键盘快捷键 (?)"
      >
        <HelpCircle className="size-4" />
      </Button>
    </div>
  )
}
