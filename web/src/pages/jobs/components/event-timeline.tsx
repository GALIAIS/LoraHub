/**
 * Event timeline (A 方案): two-pane layout for the events tab.
 *
 * Left rail compresses the entire run into a vertical stack of milestone
 * markers (spawn / cache progress bars / per-epoch / validation /
 * checkpoint / sample / error / oom / done). Step/log/gpu_sample noise is
 * elided from the rail but still indexed so the right-hand context panel
 * can replay the surrounding lines around any milestone the user clicks.
 *
 * The right pane has three modes:
 *   - milestone selected: header + structured details + ±N log context
 *   - filter "all" / "log" with no selection: virtualised raw stream
 *   - error tab: jump-list of every error / oom event
 *
 * No backend changes needed; we feed off the same events array
 * useJobStream already streams.
 */
import { useEffect, useMemo, useState } from "react"
import { AlertTriangle, ServerCog, Zap } from "lucide-react"
import { cn } from "@/lib/utils"
import type { TrainingEvent } from "@/lib/api"
import { Input } from "@/components/ui/input"
import {
  ALL_KINDS,
  FILTER_CHIPS,
  buildMilestones,
  milestoneTitle,
} from "./event-timeline-model"
import { DetailPane, TimelineRail } from "./event-timeline-panes"

export function EventTimeline({
  events,
  jobId,
  fallbackTotalSteps = null,
}: {
  events: TrainingEvent[]
  jobId: string | null
  fallbackTotalSteps?: number | null
}) {
  const milestones = useMemo(() => buildMilestones(events), [events])
  const [filterId, setFilterId] = useState<(typeof FILTER_CHIPS)[number]["id"]>(
    "all",
  )
  const filter = FILTER_CHIPS.find((f) => f.id === filterId)?.kinds ?? new Set(ALL_KINDS)
  const filteredMilestones = useMemo(
    () => milestones.filter((m) => filter.has(m.kind)),
    [milestones, filter],
  )

  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [followLatest, setFollowLatest] = useState(true)
  const [query, setQuery] = useState("")

  // Auto-follow latest milestone unless the user has explicitly clicked one.
  useEffect(() => {
    if (!followLatest) return
    if (filteredMilestones.length === 0) return
    setSelectedId(filteredMilestones[filteredMilestones.length - 1].id)
  }, [filteredMilestones, followLatest])

  const selected = useMemo(
    () => milestones.find((m) => m.id === selectedId) ?? null,
    [milestones, selectedId],
  )

  const counts = useMemo(() => {
    const c = { all: milestones.length, errors: 0, milestones: 0 }
    for (const m of milestones) {
      if (FILTER_CHIPS[1].kinds.has(m.kind)) c.milestones += 1
      if (FILTER_CHIPS[2].kinds.has(m.kind)) c.errors += 1
    }
    return c
  }, [milestones])

  // Search across milestones (title + payload string).
  const railMilestones = useMemo(() => {
    if (!query) return filteredMilestones
    const q = query.toLowerCase()
    return filteredMilestones.filter((m) => {
      const t = milestoneTitle(m).toLowerCase()
      const data = JSON.stringify(m.data).toLowerCase()
      return t.includes(q) || data.includes(q)
    })
  }, [filteredMilestones, query])

  return (
    <div className="grid h-full min-h-0 grid-rows-[minmax(180px,38%)_1fr] overflow-hidden rounded-[6px] border border-border/60 bg-background md:grid-cols-[300px_1fr] md:grid-rows-none">
      {/* LEFT: rail */}
      <div className="flex min-h-0 flex-col border-b border-border/60 bg-muted/20 md:border-b-0 md:border-r">
        <div className="border-b border-border/60 px-3 py-2">
          <div className="no-scrollbar mb-1.5 flex items-center gap-1 overflow-x-auto">
            {FILTER_CHIPS.map((c) => {
              const active = filterId === c.id
              const n =
                c.id === "all"
                  ? counts.all
                  : c.id === "milestones"
                    ? counts.milestones
                    : counts.errors
              return (
                <button
                  key={c.id}
                  type="button"
                  onClick={() => {
                    setFilterId(c.id)
                    setFollowLatest(true)
                  }}
                  className={cn(
                    "flex h-6 items-center gap-1.5 rounded-[4px] border px-2 text-[11px] transition-colors",
                    active
                      ? "border-primary/40 bg-primary/15 text-foreground"
                      : "border-border/50 bg-background/60 text-muted-foreground hover:text-foreground",
                  )}
                >
                  {c.id === "errors" && (
                    <AlertTriangle
                      className={cn(
                        "size-3",
                        n > 0 ? "text-red-500" : "text-muted-foreground",
                      )}
                    />
                  )}
                  {c.id === "milestones" && <ServerCog className="size-3" />}
                  {c.id === "all" && <Zap className="size-3" />}
                  {c.label}
                  <span className="font-mono text-[10px] tabular-nums text-muted-foreground/80">
                    {n}
                  </span>
                </button>
              )
            })}
          </div>
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="搜索里程碑…"
            className="h-7 text-[11px]"
          />
        </div>
        <div className="min-h-0 flex-1">
          <TimelineRail
            milestones={railMilestones}
            selectedId={selectedId}
            onSelect={(id) => {
              setSelectedId(id)
              setFollowLatest(false)
            }}
            filter={filter}
          />
        </div>
        {!followLatest && (
          <div className="border-t border-border/60 px-3 py-1.5">
            <button
              type="button"
              onClick={() => setFollowLatest(true)}
              className="text-[10px] uppercase tracking-[0.18em] text-primary hover:underline"
            >
              返回最新事件 →
            </button>
          </div>
        )}
      </div>
      {/* RIGHT: detail */}
      <div className="min-h-0 overflow-hidden">
        <DetailPane
          milestone={selected}
          events={events}
          jobId={jobId}
          fallbackTotalSteps={fallbackTotalSteps}
        />
      </div>
    </div>
  )
}
