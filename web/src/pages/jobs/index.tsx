import { useEffect, useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { api, type JobSummary } from "@/lib/api"
import { ScrollArea } from "@/components/ui/scroll-area"
import { JobsToolbar } from "./components/jobs-toolbar"
import { JobRow } from "./components/job-row"
import { JobDetail } from "./components/job-detail"
import { COMPARE_LIMIT, type StatusFilter } from "./utils"

const STATUS_GROUPS: Record<StatusFilter, (state: string) => boolean> = {
  all: () => true,
  running: (s) => s === "running",
  succeeded: (s) => s === "succeeded",
  failed: (s) => s === "failed",
  canceled: (s) => s === "canceled" || s === "canceling" || s === "interrupted",
}

const COMPLETED_STATES = new Set([
  "succeeded",
  "failed",
  "canceled",
  "interrupted",
])

function matchesQuery(job: JobSummary, query: string): boolean {
  if (!query) return true
  const q = query.trim().toLowerCase()
  if (!q) return true
  return (
    job.id.toLowerCase().endsWith(q) ||
    job.id.slice(-8).toLowerCase().includes(q) ||
    job.workspace.toLowerCase().includes(q)
  )
}

export function JobsPage() {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [query, setQuery] = useState("")
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all")
  const [hideCompleted, setHideCompleted] = useState(false)
  const [compareMode, setCompareMode] = useState(false)
  const [compareIds, setCompareIds] = useState<string[]>([])

  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: api.listJobs,
    refetchInterval: 2000,
  })
  const list = jobs.data?.jobs ?? []

  // Most recent first; the API gives us creation order, so reverse a copy.
  const visibleJobs = useMemo(() => {
    const filtered = list.filter((j) => {
      if (!STATUS_GROUPS[statusFilter](j.state)) return false
      if (hideCompleted && COMPLETED_STATES.has(j.state)) return false
      if (!matchesQuery(j, query)) return false
      return true
    })
    filtered.sort((a, b) => {
      const ta = new Date(a.created_at).getTime()
      const tb = new Date(b.created_at).getTime()
      return tb - ta
    })
    return filtered
  }, [list, statusFilter, hideCompleted, query])

  // When compare mode is turned off, drop the selection so re-enabling it
  // doesn't surprise the user with stale ticks.
  useEffect(() => {
    if (!compareMode) setCompareIds([])
  }, [compareMode])

  // Prune compare ids that no longer exist (e.g. archived).
  useEffect(() => {
    if (compareIds.length === 0) return
    const known = new Set(list.map((j) => j.id))
    const pruned = compareIds.filter((id) => known.has(id))
    if (pruned.length !== compareIds.length) setCompareIds(pruned)
  }, [list, compareIds])

  const selected = selectedId
    ? list.find((j) => j.id === selectedId) ?? null
    : null

  function toggleCompare(id: string) {
    setCompareIds((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id)
      if (prev.length >= COMPARE_LIMIT) return prev
      return [...prev, id]
    })
  }

  return (
    <div className="grid grid-cols-[minmax(360px,420px)_1fr] h-full">
      <aside className="border-r border-border/60 flex flex-col min-h-0">
        <JobsToolbar
          total={list.length}
          visibleCount={visibleJobs.length}
          query={query}
          onQueryChange={setQuery}
          status={statusFilter}
          onStatusChange={setStatusFilter}
          hideCompleted={hideCompleted}
          onHideCompletedChange={setHideCompleted}
          compareMode={compareMode}
          onCompareModeChange={setCompareMode}
        />
        {compareMode && (
          <div className="px-5 py-2 border-b border-border/60 bg-muted/30 text-[11px] text-muted-foreground">
            已选 {compareIds.length} / {COMPARE_LIMIT}
            {compareIds.length >= COMPARE_LIMIT && " · 已达上限"}
          </div>
        )}
        <ScrollArea className="flex-1">
          <ul className="divide-y divide-border/40">
            {visibleJobs.length === 0 && (
              <li className="px-5 py-10 text-sm text-muted-foreground text-center">
                {list.length === 0 ? "还没有训练任务。" : "没有匹配的任务。"}
              </li>
            )}
            {visibleJobs.map((j) => {
              const checked = compareIds.includes(j.id)
              const checkboxDisabled =
                !checked && compareIds.length >= COMPARE_LIMIT
              return (
                <JobRow
                  key={j.id}
                  job={j}
                  active={j.id === selectedId}
                  compareMode={compareMode}
                  checked={checked}
                  checkboxDisabled={checkboxDisabled}
                  onSelect={() => setSelectedId(j.id)}
                  onToggleCompare={() => toggleCompare(j.id)}
                />
              )
            })}
          </ul>
        </ScrollArea>
      </aside>

      <section className="min-w-0 flex flex-col bg-background/60">
        {selected ? (
          <JobDetail
            jobId={selected.id}
            onSelectJob={setSelectedId}
            compareMode={compareMode}
            compareIds={compareIds}
          />
        ) : (
          <div className="flex-1 grid place-items-center text-sm text-muted-foreground">
            从列表中选择一个任务以查看事件流。
          </div>
        )}
      </section>
    </div>
  )
}
