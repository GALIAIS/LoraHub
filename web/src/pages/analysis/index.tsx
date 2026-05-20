/**
 * Analysis Workbench
 *
 * Standalone home for everything that's not about *running* a job: loss
 * curves, AI commentary, multi-job compare. The job-detail page now
 * focuses on overview / events / artifacts only; users jump here for
 * deeper inspection.
 *
 * Routing:
 *   /analysis                    → overview (recent runs grid + quick links)
 *   /analysis/:jobId             → metrics + AI analysis for a single job
 *   /analysis/compare?ids=a,b,c  → compare mode (≥2 jobs overlaid)
 *
 * The job picker on the left is shared across all sub-modes; switching
 * jobs preserves the active sub-tab where it makes sense.
 */
import { useMemo, useState } from "react"
import { useNavigate, useParams, useSearchParams } from "react-router-dom"
import { useQuery } from "@tanstack/react-query"
import {
  ArrowLeftRight,
  BarChart3,
  Inbox,
  Search,
  Sparkles,
  X,
} from "lucide-react"
import { api, type JobSummary } from "@/lib/api"
import { useJobsList } from "@/lib/queries/jobs"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { PathDisplay } from "@/components/path-display"
import { cn } from "@/lib/utils"
import {
  COMPARE_LIMIT,
  STATE_LABELS,
  STATUS_FILTER_OPTIONS,
  TERMINAL_STATES,
  expectedTotalSteps,
  type StatusFilter,
} from "../jobs/utils"
import { CompareTab } from "../jobs/components/compare-tab"
import { StateBadge } from "../dashboard"
import { AnalysisWorkbench } from "./components/analysis-workbench"

const STATUS_GROUPS: Record<StatusFilter, (state: string) => boolean> = {
  all: () => true,
  running: (s) => s === "running",
  succeeded: (s) => s === "succeeded",
  failed: (s) => s === "failed",
  canceled: (s) => s === "canceled" || s === "canceling" || s === "interrupted",
}

export function AnalysisPage() {
  const params = useParams<{ jobId?: string }>()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()

  // The route may carry a jobId in the path or a `compare=...` query string.
  // We resolve both up front so the rest of the component reads from a
  // single normalised state shape.
  const compareParam = searchParams.get("ids") ?? searchParams.get("compare")
  const compareIds = useMemo(() => {
    if (!compareParam) return [] as string[]
    return compareParam
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean)
      .slice(0, COMPARE_LIMIT)
  }, [compareParam])
  const isCompareRoute = compareIds.length >= 2

  const activeJobId = params.jobId ?? null

  // Job list driving the picker on the left. Polling cadence + live
  // semantics are owned by `useJobsList` so this page no longer
  // duplicates the `["jobs"]` observer with a divergent interval.
  const jobs = useJobsList()
  const jobList: JobSummary[] = jobs.data?.jobs ?? []

  const [query, setQuery] = useState("")
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all")

  const visibleJobs = useMemo(() => {
    const filtered = jobList.filter((j) => {
      if (!STATUS_GROUPS[statusFilter](j.state)) return false
      if (query) {
        const q = query.trim().toLowerCase()
        if (
          !(
            j.id.toLowerCase().endsWith(q) ||
            j.id.slice(-8).toLowerCase().includes(q) ||
            j.workspace.toLowerCase().includes(q)
          )
        ) {
          return false
        }
      }
      return true
    })
    filtered.sort((a, b) => {
      const ta = new Date(a.created_at).getTime()
      const tb = new Date(b.created_at).getTime()
      return tb - ta
    })
    return filtered
  }, [jobList, statusFilter, query])

  const activeJob = activeJobId
    ? jobList.find((j) => j.id === activeJobId) ?? null
    : null

  function pickJob(id: string) {
    navigate(`/analysis/${id}`)
  }

  function startCompare(ids: string[]) {
    const safe = ids.slice(0, COMPARE_LIMIT)
    navigate(`/analysis/compare?ids=${safe.join(",")}`)
  }

  function exitCompare() {
    setSearchParams(new URLSearchParams(), { replace: true })
    navigate(activeJobId ? `/analysis/${activeJobId}` : "/analysis")
  }

  return (
    <div className="grid h-full min-h-0 grid-cols-[minmax(260px,300px)_1fr] overflow-hidden">
      <aside className="shiro-page-aside flex flex-col min-h-0">
        <header className="px-4 pt-4 pb-2 space-y-2.5">
          <div className="flex items-center gap-2">
            <BarChart3 className="size-4 text-primary" />
            <span className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
              分析工作台
            </span>
          </div>
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 size-3.5 text-muted-foreground/70" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="按 ID 末段 / 工作区搜索"
              className="h-7 pl-7 text-[12px]"
            />
          </div>
          <div className="flex flex-wrap gap-1">
            {STATUS_FILTER_OPTIONS.map((o) => {
              const active = statusFilter === o.value
              return (
                <button
                  key={o.value}
                  type="button"
                  onClick={() => setStatusFilter(o.value)}
                  className={cn(
                    "h-6 rounded-[3px] border px-1.5 text-[10.5px] tracking-wide transition-colors",
                    active
                      ? "border-primary/40 bg-primary/15 text-foreground"
                      : "border-border/50 bg-background/70 text-muted-foreground hover:text-foreground",
                  )}
                >
                  {o.label}
                </button>
              )
            })}
          </div>
        </header>

        {isCompareRoute && (
          <div className="px-4 py-2 border-y border-border/60 bg-amber-500/5 text-[11px] flex items-center justify-between gap-2">
            <span className="flex items-center gap-1.5">
              <ArrowLeftRight className="size-3 text-amber-600" />
              对比模式 · {compareIds.length} 个任务
            </span>
            <button
              type="button"
              onClick={exitCompare}
              className="inline-flex items-center gap-1 text-muted-foreground hover:text-foreground"
              title="退出对比模式"
            >
              <X className="size-3" /> 退出
            </button>
          </div>
        )}

        <ScrollArea className="flex-1 min-h-0">
          <ul className="divide-y divide-border/40">
            {visibleJobs.length === 0 && (
              <li className="px-4 py-10 text-[12px] text-center text-muted-foreground">
                {jobList.length === 0
                  ? "还没有训练任务。"
                  : "没有匹配的任务。"}
              </li>
            )}
            {visibleJobs.map((j) => (
              <JobPickerRow
                key={j.id}
                job={j}
                active={j.id === activeJobId}
                inCompare={compareIds.includes(j.id)}
                compareMode={isCompareRoute}
                onSelect={() => pickJob(j.id)}
              />
            ))}
          </ul>
        </ScrollArea>
      </aside>

      <section className="min-w-0 min-h-0 flex flex-col bg-background/60 overflow-hidden">
        {!activeJob && !isCompareRoute && (
          <EmptyState
            jobList={jobList}
            onPick={pickJob}
            onCompare={(ids) => {
              if (ids.length >= 2) startCompare(ids)
            }}
          />
        )}

        {(activeJob || isCompareRoute) && (
          <>
            <header className="px-7 py-4 border-b border-border/60 flex items-start justify-between gap-4">
              <div className="min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  {activeJob && <StateBadge state={activeJob.state} />}
                  <span className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
                    {isCompareRoute
                      ? `对比 · ${compareIds.length} 个任务`
                      : activeJob?.state
                        ? STATE_LABELS[activeJob.state] ?? activeJob.state
                        : ""}
                  </span>
                </div>
                <div className="font-mono text-[14px] truncate">
                  {isCompareRoute
                    ? compareIds.map((id) => id.slice(-8)).join(" · ")
                    : activeJob?.id}
                </div>
                {activeJob?.workspace && !isCompareRoute && (
                  <PathDisplay
                    path={activeJob.workspace}
                    tailSegments={3}
                    block
                    className="text-xs text-muted-foreground mt-1"
                  />
                )}
              </div>
              {activeJob && !isCompareRoute && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => navigate(`/jobs?id=${activeJob.id}`)}
                  className="shrink-0 text-[11px]"
                >
                  打开训练详情
                </Button>
              )}
            </header>

            {activeJob && !isCompareRoute && (
              <ScrollArea className="flex-1 min-h-0">
                <SingleJobView job={activeJob} />
              </ScrollArea>
            )}

            {isCompareRoute && (
              <ScrollArea className="flex-1 min-h-0">
                <div className="px-7 py-5">
                  <CompareTab compareIds={compareIds} />
                </div>
              </ScrollArea>
            )}
          </>
        )}
      </section>
    </div>
  )
}

function SingleJobView({ job }: { job: JobSummary }) {
  // Pull config_snapshot so the KPI strip can derive a total-steps
  // estimate without forcing the user to wait for the trainer to
  // emit one. dp's parser never emits `total_steps`; without an
  // image-count-driven derivation the progress chip stays as
  // `step / ?` for the whole run.
  const detail = useQuery({
    queryKey: ["job", job.id],
    queryFn: () => api.getJob(job.id),
    refetchInterval: 4000,
  })
  const datasetSource = useMemo(() => {
    const cfg = detail.data?.config_snapshot as
      | Record<string, unknown>
      | undefined
    const ds = cfg?.["dataset"] as Record<string, unknown> | undefined
    const src = ds?.["source"]
    return typeof src === "string" ? src : null
  }, [detail.data])
  const datasetScan = useQuery({
    queryKey: ["dataset-scan", datasetSource, true],
    queryFn: () => api.scanDataset(datasetSource!, true, 0),
    enabled: !!datasetSource,
    refetchOnWindowFocus: false,
    staleTime: 5 * 60_000,
  })
  const fallbackTotalSteps = useMemo(() => {
    const cfg = detail.data?.config_snapshot as
      | Record<string, unknown>
      | undefined
    return expectedTotalSteps(
      cfg ?? null,
      datasetScan.data?.image_files ?? null,
    )
  }, [detail.data, datasetScan.data])
  return <AnalysisWorkbench job={job} fallbackTotalSteps={fallbackTotalSteps} />
}

// ---------------------------------------------------------------------------
// Subcomponents
// ---------------------------------------------------------------------------

function JobPickerRow({
  job,
  active,
  inCompare,
  compareMode,
  onSelect,
}: {
  job: JobSummary
  active: boolean
  inCompare: boolean
  compareMode: boolean
  onSelect: () => void
}) {
  const terminal = TERMINAL_STATES.has(job.state)
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
        className={cn(
          "w-full text-left px-4 py-2.5 flex items-start gap-2.5 transition-colors",
          active
            ? "bg-primary/10"
            : "hover:bg-muted/50",
          compareMode && inCompare && "ring-1 ring-amber-500/40 bg-amber-500/5",
        )}
      >
        <StateBadge state={job.state} />
        <div className="min-w-0 flex-1">
          <div className="text-[12px] font-mono truncate">
            {job.id.slice(-8)}
          </div>
          <div className="text-[10px] text-muted-foreground truncate font-mono">
            {job.workspace.split(/[\\/]/).pop() ?? job.workspace}
          </div>
          <div className="text-[10px] text-muted-foreground/70 mt-0.5">
            {new Date(job.created_at).toLocaleString()}
            {terminal && job.finished_at ? " · 已结束" : ""}
          </div>
        </div>
      </button>
    </li>
  )
}

function EmptyState({
  jobList,
  onPick,
  onCompare,
}: {
  jobList: JobSummary[]
  onPick: (id: string) => void
  onCompare: (ids: string[]) => void
}) {
  const recent = useMemo(() => {
    return [...jobList]
      .sort((a, b) => {
        const ta = new Date(a.created_at).getTime()
        const tb = new Date(b.created_at).getTime()
        return tb - ta
      })
      .slice(0, 6)
  }, [jobList])

  const [picks, setPicks] = useState<string[]>([])

  function togglePick(id: string) {
    setPicks((prev) =>
      prev.includes(id)
        ? prev.filter((x) => x !== id)
        : prev.length >= COMPARE_LIMIT
          ? prev
          : [...prev, id],
    )
  }

  if (jobList.length === 0) {
    return (
      <div className="flex-1 grid place-items-center text-sm text-muted-foreground">
        <div className="flex flex-col items-center gap-2 py-12">
          <Inbox className="size-8 opacity-40" />
          <div>还没有训练任务可以分析。</div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-auto">
      <div className="max-w-3xl mx-auto px-7 py-8 space-y-6">
        <div className="space-y-1">
          <div className="text-[11px] uppercase tracking-[0.22em] text-muted-foreground/80">
            分析工作台
          </div>
          <h2 className="text-[18px] font-semibold tracking-tight">
            选一个任务开始分析
          </h2>
          <p className="text-sm text-muted-foreground">
            最近的 6 个任务已列在下方；勾选两个或更多进入对比模式。也可以直接在左侧搜索具体的 ID。
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {recent.map((j) => {
            const checked = picks.includes(j.id)
            return (
              <div
                key={j.id}
                className={cn(
                  "rounded-[6px] border bg-card p-3 transition-colors",
                  checked
                    ? "border-amber-500/40 bg-amber-500/5"
                    : "border-border/60 hover:border-border",
                )}
              >
                <div className="flex items-start gap-2">
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => togglePick(j.id)}
                    className="mt-1 size-3.5"
                    title="加入对比"
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5 mb-0.5">
                      <StateBadge state={j.state} />
                      <span className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
                        {STATE_LABELS[j.state] ?? j.state}
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={() => onPick(j.id)}
                      className="block text-left text-[13px] font-mono hover:underline"
                    >
                      {j.id.slice(-8)}
                    </button>
                    <div className="text-[11px] text-muted-foreground/80 truncate font-mono mt-0.5">
                      {j.workspace}
                    </div>
                    <div className="text-[10.5px] text-muted-foreground/70 mt-1">
                      {new Date(j.created_at).toLocaleString()}
                    </div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>

        <div className="flex items-center justify-between pt-2 border-t border-border/60">
          <div className="text-[11px] text-muted-foreground">
            {picks.length === 0
              ? "勾选 2 个或更多任务以进入对比模式"
              : `已选 ${picks.length} / ${COMPARE_LIMIT} 个`}
          </div>
          <Button
            size="sm"
            disabled={picks.length < 2}
            onClick={() => onCompare(picks)}
            className="gap-1.5"
          >
            <Sparkles className="size-3" /> 进入对比
          </Button>
        </div>
      </div>
    </div>
  )
}
