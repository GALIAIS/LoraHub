import { useEffect, useMemo, useRef, useState } from "react"
import { useNavigate, useSearchParams } from "react-router-dom"
import { ArrowRight, PanelLeftClose, PanelLeftOpen } from "lucide-react"
import { type JobSummary } from "@/lib/api"
import { useJobsList } from "@/lib/queries/jobs"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"
import { JobsToolbar } from "./components/jobs-toolbar"
import { JobRow } from "./components/job-row"
import { JobDetail } from "./components/job-detail"
import { COMPARE_LIMIT, type StatusFilter } from "./utils"

const STATUS_GROUPS: Record<StatusFilter, (state: string) => boolean> = {
  all: () => true,
  running: (s) => s === "running" || s === "preparing",
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

const SIDEBAR_KEY = "lorahub.jobs.sidebar"

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
  const navigate = useNavigate()
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(() => {
    if (typeof window === "undefined") return true
    return window.localStorage.getItem(SIDEBAR_KEY) !== "closed"
  })

  // Honor ?id=<jobId> and ?compare=<id1>,<id2>,... handed in by the sweeps
  // page (or any external link). We consume the query string exactly once
  // and then strip it so a refresh doesn't fight the user's later choices.
  const [searchParams, setSearchParams] = useSearchParams()
  const consumedQueryRef = useRef(false)
  useEffect(() => {
    if (consumedQueryRef.current) return
    const idParam = searchParams.get("id")
    const compareParam = searchParams.get("compare")
    let touched = false
    if (idParam) {
      setSelectedId(idParam)
      touched = true
    }
    if (compareParam) {
      const ids = compareParam
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean)
        .slice(0, COMPARE_LIMIT)
      if (ids.length > 0) {
        setCompareMode(true)
        setCompareIds(ids)
        if (!idParam) setSelectedId(ids[0])
        touched = true
      }
    }
    if (touched) {
      consumedQueryRef.current = true
      const next = new URLSearchParams(searchParams)
      next.delete("id")
      next.delete("compare")
      setSearchParams(next, { replace: true })
    } else {
      consumedQueryRef.current = true
    }
  }, [searchParams, setSearchParams])

  useEffect(() => {
    if (typeof window === "undefined") return
    window.localStorage.setItem(SIDEBAR_KEY, sidebarOpen ? "open" : "closed")
  }, [sidebarOpen])

  const jobs = useJobsList()
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

  // Auto-select the most recent job on first load when no URL param
  // pre-selected one — opening the page with no selection feels broken
  // ("从列表中选择一个任务" is technically correct but unhelpful when
  // there obviously is a newest run the user wants to look at). Runs
  // exactly once after the jobs list resolves; manual deselects later
  // are respected.
  const autoSelectedRef = useRef(false)
  useEffect(() => {
    if (autoSelectedRef.current) return
    if (!consumedQueryRef.current) return
    if (!jobs.isSuccess) return
    autoSelectedRef.current = true
    if (selectedId) return
    if (visibleJobs.length === 0) return
    setSelectedId(visibleJobs[0].id)
  }, [jobs.isSuccess, selectedId, visibleJobs])

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
    <div
      className={cn(
        "grid h-full min-h-0 overflow-hidden grid-rows-[1fr] transition-[grid-template-columns] duration-200",
        sidebarOpen
          ? "grid-cols-[minmax(280px,340px)_1fr]"
          : "grid-cols-[0px_1fr]",
      )}
    >
      <aside
        className={cn(
          "shiro-page-aside flex flex-col min-h-0 min-w-0 overflow-hidden",
          !sidebarOpen && "pointer-events-none opacity-0",
        )}
        aria-hidden={!sidebarOpen}
      >
        <div className="flex items-center justify-between px-4 pt-3">
          <span className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
            训练任务
          </span>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setSidebarOpen(false)}
            title="收起侧栏"
          >
            <PanelLeftClose className="size-4" />
          </Button>
        </div>
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
          <div className="px-5 py-2 border-b border-border/60 bg-muted/30 text-[11px] text-muted-foreground flex items-center gap-2">
            <span className="flex-1">
              已选 {compareIds.length} / {COMPARE_LIMIT}
              {compareIds.length >= COMPARE_LIMIT && " · 已达上限"}
            </span>
            {compareIds.length >= 2 && (
              <Button
                size="sm"
                variant="outline"
                className="h-6 text-[11px]"
                onClick={() =>
                  navigate(
                    `/analysis/compare?ids=${compareIds.join(",")}`,
                  )
                }
                title="到分析工作台对比 loss / 指标"
              >
                对比 <ArrowRight className="size-3" />
              </Button>
            )}
          </div>
        )}
        <ScrollArea className="flex-1 min-h-0">
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

      <section className="min-w-0 min-h-0 flex flex-col bg-background/60 overflow-hidden relative">
        {!sidebarOpen && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => setSidebarOpen(true)}
            className="absolute left-3 top-3 z-10 shadow-[var(--panel-shadow)]"
            title="展开侧栏"
          >
            <PanelLeftOpen className="size-4" />
            <span className="ml-1 text-xs">{list.length} 个任务</span>
          </Button>
        )}
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
