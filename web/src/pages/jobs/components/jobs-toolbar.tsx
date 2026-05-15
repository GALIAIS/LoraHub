import { Search } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { STATUS_FILTER_OPTIONS, type StatusFilter } from "../utils"

export function JobsToolbar({
  total,
  visibleCount,
  query,
  onQueryChange,
  status,
  onStatusChange,
  hideCompleted,
  onHideCompletedChange,
  compareMode,
  onCompareModeChange,
}: {
  total: number
  visibleCount: number
  query: string
  onQueryChange: (next: string) => void
  status: StatusFilter
  onStatusChange: (next: StatusFilter) => void
  hideCompleted: boolean
  onHideCompletedChange: (next: boolean) => void
  compareMode: boolean
  onCompareModeChange: (next: boolean) => void
}) {
  return (
    <header className="px-5 py-4 border-b border-border/60 space-y-3">
      <div className="text-xs text-muted-foreground">
        共 {total} 个 · 显示 {visibleCount} 个 · 每 2 秒刷新
      </div>

      <div className="relative">
        <Search className="absolute left-2 top-1/2 -translate-y-1/2 size-3.5 text-muted-foreground/70 pointer-events-none" />
        <Input
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          placeholder="按 ID 末 8 位或工作区路径搜索…"
          className="h-8 pl-7 text-xs"
        />
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        <Select
          value={status}
          onValueChange={(v) => onStatusChange(v as StatusFilter)}
        >
          <SelectTrigger className="h-8 text-xs flex-1 min-w-[7rem]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {STATUS_FILTER_OPTIONS.map((o) => (
              <SelectItem key={o.value} value={o.value}>
                {o.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <label className="inline-flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer select-none">
          <Switch
            size="sm"
            checked={hideCompleted}
            onCheckedChange={onHideCompletedChange}
          />
          隐藏已完成
        </label>
        <label className="inline-flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer select-none">
          <Switch
            size="sm"
            checked={compareMode}
            onCheckedChange={onCompareModeChange}
          />
          对比模式
        </label>
      </div>
    </header>
  )
}
