import { ChevronLeft, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { cn } from "@/lib/utils"

export interface PaginationProps {
  /** Total item count across all pages. */
  total: number
  /** Items per page. */
  pageSize: number
  /** 1-based current page index. */
  page: number
  onPageChange: (page: number) => void
  pageSizeOptions?: number[]
  onPageSizeChange?: (size: number) => void
  className?: string
}

/**
 * Compact "page X / Y" pagination strip with prev/next buttons and an
 * optional page-size dropdown. Hides itself when there's nothing to
 * paginate (total <= pageSize and page == 1).
 */
export function Pagination({
  total,
  pageSize,
  page,
  onPageChange,
  pageSizeOptions,
  onPageSizeChange,
  className,
}: PaginationProps) {
  const safeSize = Math.max(1, pageSize)
  const totalPages = Math.max(1, Math.ceil(total / safeSize))
  const safePage = Math.min(Math.max(1, page), totalPages)
  const start = total === 0 ? 0 : (safePage - 1) * safeSize + 1
  const end = Math.min(total, safePage * safeSize)

  if (total <= safeSize && safePage === 1 && !pageSizeOptions) return null

  return (
    <div
      className={cn(
        "flex flex-col gap-2 text-[12px] sm:flex-row sm:items-center sm:justify-between",
        className,
      )}
    >
      <div className="min-w-0 text-center text-muted-foreground tabular-nums sm:text-left">
        {total > 0 ? `${start}–${end} / ${total}` : "0 / 0"}
      </div>
      <div className="flex min-w-0 items-center justify-center gap-1.5 sm:justify-end">
        {pageSizeOptions && onPageSizeChange && (
          <div className="flex shrink-0 items-center gap-1 text-muted-foreground">
            <span className="hidden sm:inline">每页</span>
            <Select
              value={String(safeSize)}
              onValueChange={(value) => onPageSizeChange(Number(value))}
            >
              <SelectTrigger
                size="sm"
                aria-label="每页数量"
                className="h-7 w-[4.25rem] px-2 font-mono text-[11px]"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="min-w-[4.25rem]">
                {pageSizeOptions.map((opt) => (
                  <SelectItem key={opt} value={String(opt)} className="font-mono text-[12px]">
                    {opt}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
        <Button
          size="sm"
          variant="outline"
          className="h-7 w-8 shrink-0 px-0"
          disabled={safePage <= 1}
          aria-label="上一页"
          onClick={() => onPageChange(safePage - 1)}
        >
          <ChevronLeft className="size-3.5" />
        </Button>
        <span className="min-w-[4.75rem] shrink-0 rounded-[6px] border border-border/60 bg-muted/30 px-2 py-1 text-center font-mono text-[11px] tabular-nums text-muted-foreground">
          {safePage} / {totalPages}
        </span>
        <Button
          size="sm"
          variant="outline"
          className="h-7 w-8 shrink-0 px-0"
          disabled={safePage >= totalPages}
          aria-label="下一页"
          onClick={() => onPageChange(safePage + 1)}
        >
          <ChevronRight className="size-3.5" />
        </Button>
      </div>
    </div>
  )
}
