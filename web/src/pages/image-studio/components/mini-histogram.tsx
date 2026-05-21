/**
 * Tiny CSS bar chart — no external dep.
 *
 * Used by the audit stage's resolution / AR / filesize / caption
 * histograms. Pure flex + width %, accessible via aria-label.
 */
import { cn } from "@/lib/utils"

interface Props {
  buckets: { bucket: string; count: number }[]
  title: string
  /** Total used as the histogram denominator; defaults to max count
   *  across buckets so the tallest bar fills the row. */
  total?: number
  className?: string
}

export function MiniHistogram({ buckets, title, total, className }: Props) {
  const max = total ?? Math.max(1, ...buckets.map((b) => b.count))
  return (
    <div
      className={cn("rounded-md border border-border/60 bg-card p-3", className)}
      role="figure"
      aria-label={title}
    >
      <div className="text-xs font-medium text-foreground mb-2">{title}</div>
      <div className="space-y-1">
        {buckets.map((b) => {
          const pct = max > 0 ? (b.count / max) * 100 : 0
          return (
            <div key={b.bucket} className="flex items-center gap-2 text-[11px]">
              <span className="w-20 shrink-0 text-muted-foreground tabular-nums truncate">
                {b.bucket}
              </span>
              <div className="flex-1 h-3 rounded-sm bg-muted/40 overflow-hidden">
                <div
                  className="h-full bg-primary/70 transition-all"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="w-10 shrink-0 tabular-nums text-right text-muted-foreground">
                {b.count}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
