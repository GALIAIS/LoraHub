import { ChevronLeft, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"
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
        "flex items-center justify-between gap-3 flex-wrap text-[12px]",
        className,
      )}
    >
      <div className="text-muted-foreground tabular-nums">
        {total > 0 ? `${start}–${end} / ${total}` : "0 / 0"}
      </div>
      <div className="flex items-center gap-2">
        {pageSizeOptions && onPageSizeChange && (
          <label className="flex items-center gap-1.5 text-muted-foreground">
            每页
            <select
              value={safeSize}
              onChange={(e) => onPageSizeChange(Number(e.target.value))}
              className="h-7 rounded-[3px] border border-border/60 bg-background px-1.5 font-mono text-[11px]"
            >
              {pageSizeOptions.map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          </label>
        )}
        <Button
          size="sm"
          variant="outline"
          className="h-7 px-2"
          disabled={safePage <= 1}
          onClick={() => onPageChange(safePage - 1)}
        >
          <ChevronLeft className="size-3.5" />
        </Button>
        <span className="font-mono tabular-nums text-muted-foreground">
          {safePage} / {totalPages}
        </span>
        <Button
          size="sm"
          variant="outline"
          className="h-7 px-2"
          disabled={safePage >= totalPages}
          onClick={() => onPageChange(safePage + 1)}
        >
          <ChevronRight className="size-3.5" />
        </Button>
      </div>
    </div>
  )
}
