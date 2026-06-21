import type { ReactNode } from "react"
import { GitCompare, Search, Trash2 } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
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
  selectMode,
  onSelectModeChange,
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
  selectMode: boolean
  onSelectModeChange: (next: boolean) => void
}) {
  return (
    <header className="space-y-3 border-b border-border/60 px-4 py-3">
      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="font-medium text-foreground">任务列表</span>
        <span className="tabular-nums text-muted-foreground">
          {visibleCount} / {total}
        </span>
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

      <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-2">
        <Select
          items={STATUS_FILTER_OPTIONS}
          value={status}
          onValueChange={(v) => onStatusChange(v as StatusFilter)}
        >
          <SelectTrigger className="h-8 min-w-0 text-xs">
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
      </div>

      <div className="grid grid-cols-2 gap-2">
        <ModeButton
          active={compareMode}
          disabled={selectMode}
          icon={<GitCompare className="size-3.5" />}
          label="对比"
          onClick={() => onCompareModeChange(!compareMode)}
        />
        <ModeButton
          active={selectMode}
          disabled={compareMode}
          icon={<Trash2 className="size-3.5" />}
          label="批量"
          onClick={() => onSelectModeChange(!selectMode)}
        />
      </div>
    </header>
  )
}

function ModeButton({
  active,
  disabled,
  icon,
  label,
  onClick,
}: {
  active: boolean
  disabled?: boolean
  icon: ReactNode
  label: string
  onClick: () => void
}) {
  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      disabled={disabled}
      aria-pressed={active}
      onClick={onClick}
      className={cn(
        "h-8 justify-center gap-1.5 text-xs",
        active && "border-primary/40 bg-primary/10 text-primary",
      )}
    >
      {icon}
      {label}
    </Button>
  )
}
