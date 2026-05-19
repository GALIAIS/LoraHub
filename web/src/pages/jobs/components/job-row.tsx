import type { JobSummary } from "@/lib/api"
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
  return (
    <li
      onClick={onSelect}
      className={cn(
        "px-5 py-3 cursor-pointer transition-colors flex items-start gap-2.5",
        active
          ? "bg-accent/70 border-l-2 border-l-primary"
          : "border-l-2 border-l-transparent hover:bg-muted/40",
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
        <div className="flex items-center gap-2 mb-1">
          <StateBadge
            state={job.state}
            paused={
              Boolean(job.metadata && (job.metadata as Record<string, unknown>).paused === true)
            }
          />
          <code className="text-[11px] font-mono text-muted-foreground">
            {job.id.slice(-8)}
          </code>
          {job.pid !== null && (
            <span className="text-[10px] text-muted-foreground/60">
              PID {job.pid}
            </span>
          )}
        </div>
        <div className="text-xs text-muted-foreground truncate font-mono">
          {job.workspace}
        </div>
        <div className="text-[10px] text-muted-foreground/70 mt-0.5">
          {new Date(job.created_at).toLocaleString()}
        </div>
      </div>
    </li>
  )
}
