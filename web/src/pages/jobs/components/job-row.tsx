import type { JobSummary } from "@/lib/api"
import { PathDisplay } from "@/components/path-display"
import { StateBadge } from "../../dashboard"
import { cn } from "@/lib/utils"
import { Circle, CircleCheck } from "lucide-react"

export function JobRow({
  job,
  active,
  compareMode,
  checked,
  checkboxDisabled,
  onSelect,
  onToggleCompare,
}: {
  job: JobSummary
  active: boolean
  compareMode: boolean
  checked: boolean
  checkboxDisabled: boolean
  onSelect: () => void
  onToggleCompare: () => void
}) {
  const shortId = formatJobId(job.id)
  return (
    <li
      onClick={onSelect}
      className={cn(
        "mx-1.5 my-1 flex cursor-pointer items-start gap-2.5 rounded-[6px] border px-3 py-2.5 transition-colors",
        active
          ? "border-primary/35 bg-accent text-accent-foreground"
          : "border-transparent text-muted-foreground hover:border-border/60 hover:bg-muted/45 hover:text-foreground",
      )}
    >
      {compareMode && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation()
            if (!checkboxDisabled || checked) onToggleCompare()
          }}
          disabled={checkboxDisabled && !checked}
          aria-label={checked ? "取消对比" : "加入对比"}
          className={cn(
            "mt-0.5 shrink-0 transition-colors",
            checked
              ? "text-primary"
              : "text-muted-foreground/60 hover:text-foreground",
            checkboxDisabled && !checked && "opacity-40 cursor-not-allowed",
          )}
        >
          {checked ? (
            <CircleCheck className="size-4" />
          ) : (
            <Circle className="size-4" />
          )}
        </button>
      )}
      <div className="min-w-0 flex-1">
        <div className="mb-1 flex items-center gap-2">
          <StateBadge
            state={job.state}
            paused={
              Boolean(job.metadata && (job.metadata as Record<string, unknown>).paused === true)
            }
          />
          <code className="truncate text-[11px] font-mono text-muted-foreground">
            {shortId}
          </code>
          {job.pid !== null && (
            <span className="text-[10px] text-muted-foreground/60">
              PID {job.pid}
            </span>
          )}
        </div>
        <PathDisplay
          path={job.workspace}
          tailSegments={3}
          block
          className="text-[11px]"
        />
        <div className="mt-1 text-[10px] text-muted-foreground/70">
          {new Date(job.created_at).toLocaleString()}
        </div>
      </div>
    </li>
  )
}

function formatJobId(id: string): string {
  if (id.length <= 18) return id
  const parts = id.split("-")
  const maybeTimestamp = parts.find((part) => /^\d{8}$/.test(part))
  if (maybeTimestamp) {
    const prefix = id.slice(0, id.indexOf(maybeTimestamp)).replace(/-$/, "")
    if (prefix.length > 0 && prefix.length <= 24) return prefix
  }
  return `${id.slice(0, 10)}…${id.slice(-6)}`
}
