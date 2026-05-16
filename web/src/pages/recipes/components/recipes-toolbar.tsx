import { Folder, Plus, Search, Upload } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { shortenPath } from "../utils"
import type { ArchFilter, SortOrder } from "../types"

const ARCH_OPTIONS: { value: ArchFilter; label: string }[] = [
  { value: "all", label: "全部架构" },
  { value: "sdxl", label: "SDXL" },
  { value: "sd15", label: "SD 1.5" },
  { value: "flux", label: "FLUX" },
  { value: "sd3", label: "SD3" },
]

const SORT_OPTIONS: { value: SortOrder; label: string }[] = [
  { value: "name-asc", label: "名称 升序" },
  { value: "name-desc", label: "名称 降序" },
  { value: "modified-desc", label: "修改时间 新→旧" },
]

export function RecipesToolbar({
  dir,
  total,
  visibleCount,
  query,
  onQueryChange,
  arch,
  onArchChange,
  sort,
  onSortChange,
  onCreate,
  onImport,
}: {
  dir: string | null
  total: number
  visibleCount: number
  query: string
  onQueryChange: (next: string) => void
  arch: ArchFilter
  onArchChange: (next: ArchFilter) => void
  sort: SortOrder
  onSortChange: (next: SortOrder) => void
  onCreate: () => void
  onImport: () => void
}) {
  const filtered = visibleCount !== total
  return (
    <header className="px-5 py-4 border-b border-border/60 space-y-3">
      <div className="text-xs text-muted-foreground flex items-center gap-1.5 min-w-0">
        <Folder className="size-3 shrink-0" />
        <span className="font-mono truncate" title={dir ?? ""}>
          {dir ? shortenPath(dir) : "加载中…"}
        </span>
      </div>

      <div>
        <div className="text-xs text-muted-foreground tabular-nums">
          {filtered
            ? `共 ${total} 个 · 显示 ${visibleCount} 个`
            : `共 ${total} 个`}
        </div>
      </div>

      <div className="space-y-2">
        <div className="relative">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 size-3.5 text-muted-foreground/70 pointer-events-none" />
          <Input
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder="按名称搜索…"
            className="h-8 pl-7 text-xs"
          />
        </div>

        <div className="flex items-center gap-2">
          <Select items={ARCH_OPTIONS} value={arch} onValueChange={(v) => onArchChange(v as ArchFilter)}>
            <SelectTrigger className="h-8 text-xs flex-1 min-w-[7rem]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {ARCH_OPTIONS.map((o) => (
                <SelectItem key={o.value} value={o.value}>
                  {o.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-2">
          <Select items={SORT_OPTIONS} value={sort} onValueChange={(v) => onSortChange(v as SortOrder)}>
            <SelectTrigger className="h-8 text-xs flex-1 min-w-[7rem]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {SORT_OPTIONS.map((o) => (
                <SelectItem key={o.value} value={o.value}>
                  {o.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button size="sm" variant="outline" onClick={onCreate}>
            <Plus className="size-3" /> 新建
          </Button>
          <Button size="sm" variant="outline" onClick={onImport}>
            <Upload className="size-3" /> 导入
          </Button>
        </div>
      </div>
    </header>
  )
}
