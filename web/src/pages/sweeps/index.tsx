import { useEffect, useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import {
  PanelLeftClose,
  PanelLeftOpen,
  Activity,
  ExternalLink,
  GitCompareArrows,
  SlidersHorizontal,
  Trophy,
} from "lucide-react"
import {
  api,
  type SweepDetail,
  type SweepJobSummary,
  type SweepParetoResponse,
  type SweepSummary,
} from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { cn } from "@/lib/utils"
import { StateBadge } from "../dashboard"

const SIDEBAR_KEY = "lorahub.sweeps.sidebar"

const ACTIVE_STATES = new Set(["queued", "running", "canceling"])

// State color tokens for the sweep distribution mini-bar — kept literal here
// so we don't have to spin up a tailwind plugin entry just for one widget.
const STATE_COLORS: Record<string, string> = {
  succeeded: "bg-emerald-500/85",
  running: "bg-sky-500/85",
  queued: "bg-muted-foreground/40",
  failed: "bg-rose-500/85",
  canceled: "bg-amber-500/70",
  canceling: "bg-amber-500/70",
  interrupted: "bg-rose-500/60",
}

// Color tones for the search-strategy badge. TPE gets a brighter tone so
// users immediately notice when an adaptive sweep is running — that's
// where they care about pareto / convergence.
const MODE_BADGE: Record<string, { label: string; toneClass: string }> = {
  grid: {
    label: "grid",
    toneClass: "border-zinc-500/40 bg-zinc-500/10 text-zinc-700 dark:text-zinc-300",
  },
  random: {
    label: "random",
    toneClass: "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300",
  },
  tpe: {
    label: "TPE",
    toneClass: "border-cyan-500/40 bg-cyan-500/10 text-cyan-700 dark:text-cyan-300",
  },
}

function ModeBadge({ mode }: { mode: string | undefined | null }) {
  const meta = MODE_BADGE[mode ?? "grid"] ?? MODE_BADGE.grid
  return (
    <Badge
      variant="outline"
      className={cn(
        "rounded-[2px] uppercase text-[10px] tracking-[0.1em]",
        meta.toneClass,
      )}
    >
      {meta.label}
    </Badge>
  )
}

function fmtRelativeTime(iso: string | null | undefined): string {
  if (!iso) return "—"
  const ts = new Date(iso).getTime()
  if (!Number.isFinite(ts)) return "—"
  const delta = (Date.now() - ts) / 1000
  if (delta < 60) return "刚刚"
  if (delta < 3600) return `${Math.floor(delta / 60)} 分钟前`
  if (delta < 86400) return `${Math.floor(delta / 3600)} 小时前`
  if (delta < 86400 * 30) return `${Math.floor(delta / 86400)} 天前`
  return new Date(ts).toLocaleDateString()
}

function formatAxisValue(value: unknown): string {
  if (value === null) return "null"
  if (typeof value === "number" || typeof value === "boolean") return String(value)
  if (typeof value === "string") return value
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

export function SweepsPage() {
  const sweeps = useQuery({
    queryKey: ["sweeps"],
    queryFn: api.listSweeps,
    refetchInterval: 4000,
    // refetchInterval 已经每 4s 拉一次,挂载/可见性变化时不需要再
    // 立刻补一发 — staleTime 顶住短时窗口的重复请求。
    staleTime: 2_000,
  })
  const list = sweeps.data?.sweeps ?? []

  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(() => {
    if (typeof window === "undefined") return true
    return window.localStorage.getItem(SIDEBAR_KEY) !== "closed"
  })

  useEffect(() => {
    if (typeof window === "undefined") return
    window.localStorage.setItem(SIDEBAR_KEY, sidebarOpen ? "open" : "closed")
  }, [sidebarOpen])

  // Auto-pick first sweep when nothing is selected and the list arrives;
  // also self-heal when the active sweep gets dropped from the registry.
  useEffect(() => {
    if (list.length === 0) {
      if (selectedId !== null) setSelectedId(null)
      return
    }
    const known = new Set(list.map((s) => s.sweep_id))
    if (selectedId === null || !known.has(selectedId)) {
      setSelectedId(list[0].sweep_id)
    }
  }, [list, selectedId])

  return (
    <div
      className={cn(
        "grid h-full min-h-0 overflow-hidden grid-rows-[1fr] transition-[grid-template-columns] duration-200",
        sidebarOpen
          ? "grid-cols-[minmax(240px,300px)_1fr]"
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
            参数搜索
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
        <div className="px-4 pt-1 pb-3 text-[11px] text-muted-foreground">
          {sweeps.isLoading
            ? "加载中…"
            : `${list.length} 个 sweep`}
        </div>
        <ScrollArea className="flex-1 min-h-0">
          <ul className="divide-y divide-border/40">
            {!sweeps.isLoading && list.length === 0 && (
              <li className="px-5 py-10 text-sm text-muted-foreground text-center">
                还没有 sweep。运行
                <code className="text-foreground"> lorahub sweep config.yaml --axis ...</code>
                {" "}或 POST {" "}
                <code className="text-foreground">/api/sweeps</code> 触发一次。
              </li>
            )}
            {list.map((s) => (
              <SweepListRow
                key={s.sweep_id}
                sweep={s}
                active={s.sweep_id === selectedId}
                onSelect={() => setSelectedId(s.sweep_id)}
              />
            ))}
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
            <span className="ml-1 text-xs">{list.length} 个 sweep</span>
          </Button>
        )}
        {selectedId ? (
          <SweepDetailPanel sweepId={selectedId} />
        ) : (
          <div className="flex-1 grid place-items-center text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <SlidersHorizontal className="size-4" />
              选择左侧的 sweep 以查看详情。
            </div>
          </div>
        )}
      </section>
    </div>
  )
}

function SweepListRow({
  sweep,
  active,
  onSelect,
}: {
  sweep: SweepSummary
  active: boolean
  onSelect: () => void
}) {
  const isActive = sweep.queued + sweep.running + sweep.canceling > 0
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
        className={cn(
          "w-full text-left px-4 py-3 transition-colors flex flex-col gap-1.5",
          active
            ? "bg-sidebar-accent/70 text-foreground"
            : "hover:bg-muted/40 text-muted-foreground",
        )}
      >
        <div className="flex items-center justify-between gap-2">
          <span className="text-[13px] font-medium text-foreground truncate">
            {sweep.name_prefix || sweep.sweep_id.slice(-8)}
          </span>
          <ModeBadge mode={sweep.mode} />
        </div>
        <DistributionBar sweep={sweep} />
        <div className="flex items-center justify-between text-[11px] text-muted-foreground">
          <span>
            {sweep.total} 个变体
            {isActive && (
              <span className="ml-1.5 inline-flex items-center gap-1 text-foreground/80">
                <Activity className="size-3" />
                进行中
              </span>
            )}
          </span>
          <span className="tabular-nums">
            {fmtRelativeTime(sweep.latest_modified_at)}
          </span>
        </div>
      </button>
    </li>
  )
}

function DistributionBar({ sweep }: { sweep: SweepSummary }) {
  // Render in priority order so finished/active states sit at the front.
  const segments: Array<{ key: string; count: number; label: string }> = [
    { key: "succeeded", count: sweep.succeeded, label: "已完成" },
    { key: "running", count: sweep.running, label: "运行中" },
    { key: "queued", count: sweep.queued, label: "排队中" },
    { key: "failed", count: sweep.failed, label: "失败" },
    {
      key: "canceled",
      count: sweep.canceled + sweep.canceling,
      label: "已取消",
    },
    { key: "interrupted", count: sweep.interrupted, label: "中断" },
  ]
  const total = sweep.total || 1
  return (
    <div
      className="shiro-progress-track h-1.5 w-full flex"
      title={segments
        .filter((s) => s.count > 0)
        .map((s) => `${s.label} ${s.count}`)
        .join(" · ")}
    >
      {segments.map((seg) =>
        seg.count > 0 ? (
          <div
            key={seg.key}
            className={cn("shiro-progress-fill", STATE_COLORS[seg.key])}
            style={{ width: `${(seg.count / total) * 100}%` }}
          />
        ) : null,
      )}
    </div>
  )
}

function SweepDetailPanel({ sweepId }: { sweepId: string }) {
  const detail = useQuery({
    queryKey: ["sweep", sweepId],
    queryFn: () => api.getSweep(sweepId),
    // Poll only while there's still pending work; once the sweep is fully
    // settled we drop to manual refresh to avoid pointless network noise.
    refetchInterval: (query) => {
      const data = query.state.data as SweepDetail | undefined
      if (!data) return 3000
      const pending = data.queued + data.running + data.canceling
      return pending > 0 ? 3000 : false
    },
    staleTime: 1_500,
  })

  if (detail.isLoading && !detail.data) {
    return (
      <div className="flex-1 grid place-items-center text-sm text-muted-foreground">
        加载中…
      </div>
    )
  }
  if (detail.isError || !detail.data) {
    return (
      <div className="flex-1 grid place-items-center text-sm text-destructive">
        加载失败：{(detail.error as Error | undefined)?.message ?? "未知错误"}
      </div>
    )
  }

  const sweep = detail.data
  return (
    <ScrollArea className="flex-1 min-h-0">
      <div className="p-6 space-y-5 max-w-5xl">
        <SweepSummaryHeader sweep={sweep} />
        <ParetoCard sweepId={sweep.sweep_id} mode={sweep.plan?.mode} />
        <AxisMatrix jobs={sweep.jobs} />
        <VariantTable jobs={sweep.jobs} />
        <SweepActions sweep={sweep} />
      </div>
    </ScrollArea>
  )
}

function SweepSummaryHeader({ sweep }: { sweep: SweepDetail }) {
  // Pull the variant prefix off whichever job carries a recognisable name —
  // we don't ship name_prefix on the detail endpoint to avoid duplicating
  // the heuristic, so the header derives it locally for display only.
  const namePrefix = useMemo(() => {
    const names = sweep.jobs
      .map((j) => j.metadata?.variant_name)
      .filter((n): n is string => typeof n === "string" && n.length > 0)
    if (names.length === 0) return sweep.sweep_id.slice(-8)
    if (names.length === 1) return names[0]
    const shortest = names.reduce((a, b) => (a.length <= b.length ? a : b))
    let cutoff = shortest.length
    for (let i = 0; i < shortest.length; i++) {
      if (names.some((n) => n[i] !== shortest[i])) {
        cutoff = i
        break
      }
    }
    let prefix = shortest.slice(0, cutoff)
    while (prefix && (/[\d\-_./]/.test(prefix[prefix.length - 1]))) {
      prefix = prefix.slice(0, -1)
    }
    return prefix || shortest
  }, [sweep])

  const segments: Array<{ key: string; count: number; label: string }> = [
    { key: "succeeded", count: sweep.succeeded, label: "已完成" },
    { key: "running", count: sweep.running, label: "运行中" },
    { key: "queued", count: sweep.queued, label: "排队中" },
    { key: "failed", count: sweep.failed, label: "失败" },
    {
      key: "canceled",
      count: sweep.canceled + sweep.canceling,
      label: "已取消",
    },
    { key: "interrupted", count: sweep.interrupted, label: "中断" },
  ]
  const total = sweep.total || 1

  return (
    <header className="space-y-3">
      <div>
        <div className="flex items-center gap-2">
          <SlidersHorizontal className="size-4 text-muted-foreground" />
          <h1 className="text-base font-semibold tracking-tight">
            {namePrefix}
          </h1>
          <ModeBadge mode={sweep.plan?.mode} />
          {sweep.plan?.n_trials != null && (
            <span className="text-[11px] text-muted-foreground">
              n_trials={sweep.plan.n_trials}
            </span>
          )}
          {sweep.plan?.seed != null && (
            <span className="text-[11px] text-muted-foreground font-mono">
              seed={sweep.plan.seed}
            </span>
          )}
          <span className="font-mono text-[11px] text-muted-foreground">
            #{sweep.sweep_id.slice(-8)}
          </span>
        </div>
        <div className="text-xs text-muted-foreground mt-1">
          共 {sweep.total} 个变体 · sweep_id{" "}
          <code className="text-foreground/80">{sweep.sweep_id}</code>
        </div>
      </div>
      <div className="space-y-1.5">
        <div
          className="shiro-progress-track h-2.5 w-full flex"
        >
          {segments.map((seg) =>
            seg.count > 0 ? (
              <div
                key={seg.key}
                className={cn("shiro-progress-fill", STATE_COLORS[seg.key])}
                style={{ width: `${(seg.count / total) * 100}%` }}
                title={`${seg.label} ${seg.count}`}
              />
            ) : null,
          )}
        </div>
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
          {segments.map((seg) => (
            <span
              key={seg.key}
              className="inline-flex items-center gap-1 tabular-nums"
            >
              <span
                className={cn(
                  "inline-block size-2 rounded-full",
                  STATE_COLORS[seg.key],
                )}
              />
              {seg.label}
              <span className="text-foreground/80">{seg.count}</span>
            </span>
          ))}
        </div>
      </div>
    </header>
  )
}

function AxisMatrix({ jobs }: { jobs: SweepJobSummary[] }) {
  // Reverse-engineer the axis-value sets from each job's metadata. Order is
  // dictated by the first job that exposes a particular axis path so the UI
  // matches the order the operator declared on the request — Set+Map both
  // preserve insertion order in JS.
  const matrix = useMemo(() => {
    const order: string[] = []
    const buckets = new Map<string, Set<string>>()
    for (const job of jobs) {
      const axes = job.metadata?.axis_values
      if (!axes) continue
      for (const [path, raw] of Object.entries(axes)) {
        if (!buckets.has(path)) {
          buckets.set(path, new Set())
          order.push(path)
        }
        buckets.get(path)!.add(formatAxisValue(raw))
      }
    }
    return order.map((path) => ({
      path,
      values: Array.from(buckets.get(path) ?? []),
    }))
  }, [jobs])

  if (matrix.length === 0) {
    return (
      <section className="rounded-[6px] border border-border/60 bg-muted/20 p-4 text-xs text-muted-foreground">
        没有 axis_values 元数据 — 这些任务可能不是通过 sweep 接口发起的。
      </section>
    )
  }

  return (
    <section className="rounded-[6px] border border-border/60 shadow-[var(--panel-shadow)] overflow-hidden">
      <header className="px-4 py-2 border-b border-border/60 bg-muted/40 flex items-center justify-between">
        <span className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          轴矩阵
        </span>
        <span className="text-[11px] text-muted-foreground/70">
          {matrix.length} 个轴 ·{" "}
          {matrix.reduce((acc, r) => acc * Math.max(1, r.values.length), 1)} 种组合
        </span>
      </header>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[32%] font-mono text-[11px]">axis path</TableHead>
            <TableHead>取值</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {matrix.map((row) => (
            <TableRow key={row.path}>
              <TableCell className="font-mono text-xs align-top">
                {row.path}
              </TableCell>
              <TableCell>
                <div className="flex flex-wrap gap-1.5">
                  {row.values.map((v) => (
                    <code
                      key={v}
                      className="rounded-[2px] border border-border/60 bg-muted/40 px-1.5 py-0.5 text-[11px] text-foreground/90"
                    >
                      {v}
                    </code>
                  ))}
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </section>
  )
}

function VariantTable({ jobs }: { jobs: SweepJobSummary[] }) {
  const navigate = useNavigate()
  if (jobs.length === 0) {
    return null
  }
  return (
    <section className="rounded-[6px] border border-border/60 shadow-[var(--panel-shadow)] overflow-hidden">
      <header className="px-4 py-2 border-b border-border/60 bg-muted/40">
        <span className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          变体 · {jobs.length}
        </span>
      </header>
      {/* Cap the table at ~12 rows so a 256-cell grid sweep doesn't
          push the Pareto card and bulk actions off-screen. The
          ScrollArea is local — outer page scroll is unaffected. */}
      <div className="max-h-[28rem] overflow-y-auto">
        <Table>
          <TableHeader className="sticky top-0 bg-background z-[1]">
            <TableRow>
              <TableHead className="w-[24%]">变体</TableHead>
              <TableHead className="w-[110px]">状态</TableHead>
              <TableHead>取值</TableHead>
              <TableHead className="w-[150px] whitespace-nowrap">起止</TableHead>
              <TableHead className="w-[70px] text-right">操作</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
          {jobs.map((job) => {
            const name = job.metadata?.variant_name ?? job.id.slice(-8)
            const axes = job.metadata?.axis_values ?? {}
            return (
              <TableRow key={job.id}>
                <TableCell className="align-top">
                  <div className="flex flex-col gap-0.5 min-w-0">
                    <span className="font-medium text-[13px] truncate" title={name}>
                      {name}
                    </span>
                    <span className="font-mono text-[10px] text-muted-foreground">
                      #{job.id.slice(-8)}
                    </span>
                  </div>
                </TableCell>
                <TableCell className="align-top">
                  <StateBadge state={job.state} />
                </TableCell>
                <TableCell className="align-top">
                  <div className="flex flex-wrap gap-1">
                    {Object.entries(axes).map(([path, value]) => (
                      <span
                        key={path}
                        className="inline-flex items-center gap-1 rounded-[2px] border border-border/60 bg-muted/40 px-1.5 py-0.5 text-[11px]"
                        title={path}
                      >
                        <span className="font-mono text-muted-foreground">
                          {path.split(".").pop()}
                        </span>
                        <span className="text-foreground/90 font-mono">
                          {formatAxisValue(value)}
                        </span>
                      </span>
                    ))}
                  </div>
                </TableCell>
                <TableCell className="align-top text-[11px] text-muted-foreground tabular-nums">
                  <div>开始 {fmtRelativeTime(job.started_at)}</div>
                  <div>结束 {fmtRelativeTime(job.finished_at)}</div>
                </TableCell>
                <TableCell className="align-top text-right">
                  <Button
                    variant="ghost"
                    size="sm"
                    title="打开任务详情" aria-label="打开任务详情"
                    onClick={() =>
                      navigate(`/jobs?id=${encodeURIComponent(job.id)}`)
                    }
                  >
                    <ExternalLink className="size-3.5" />
                  </Button>
                </TableCell>
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
      </div>
    </section>
  )
}

function SweepActions({ sweep }: { sweep: SweepDetail }) {
  const navigate = useNavigate()
  const succeededIds = sweep.jobs
    .filter((j) => j.state === "succeeded")
    .map((j) => j.id)

  const canCompare = succeededIds.length >= 2
  const activeCount =
    sweep.jobs.filter((j) => ACTIVE_STATES.has(j.state)).length

  return (
    <section className="flex flex-wrap items-center gap-2 pt-1">
      <Button
        size="sm"
        variant="default"
        disabled={!canCompare}
        onClick={() =>
          navigate(`/jobs?compare=${succeededIds.join(",")}`)
        }
        title={
          canCompare
            ? "在任务页对比已完成变体的 loss 曲线"
            : "至少需要 2 个已完成变体"
        }
      >
        <GitCompareArrows className="size-4" />
        比较已完成变体的 loss 曲线
        <span className="ml-1 text-[10px] text-muted-foreground">
          ({succeededIds.length})
        </span>
      </Button>
      {activeCount > 0 && (
        <span className="text-[11px] text-muted-foreground inline-flex items-center gap-1">
          <Activity className="size-3" />
          {activeCount} 个变体仍在进行中，详情会自动刷新
        </span>
      )}
    </section>
  )
}


// --------------------------------------------------------------------------- //
// Pareto card — best trial + completed-trial leaderboard.
//
// Visible for every mode, but the value really shines on TPE / random where
// the user wants to see "which axis values landed the lowest loss". Grid
// sweeps without a per-trial metric will show the same data, just less
// interesting because every cell of the cartesian product runs anyway.
// --------------------------------------------------------------------------- //

function ParetoCard({
  sweepId,
  mode,
}: {
  sweepId: string
  mode: string | undefined
}) {
  const pareto = useQuery({
    queryKey: ["sweep-pareto", sweepId],
    queryFn: () => api.getSweepPareto(sweepId),
    refetchInterval: (query) => {
      const data = query.state.data as SweepParetoResponse | undefined
      // Keep polling while there are pending trials so the leaderboard
      // updates as each one finishes; once everything's done, stop.
      return data && data.pending > 0 ? 4000 : false
    },
    staleTime: 2_000,
  })

  if (pareto.isLoading || !pareto.data) {
    return (
      <section className="space-y-2">
        <h2 className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground/80 inline-flex items-center gap-1">
          <Trophy className="size-3" />
          Pareto / 最佳 trial
        </h2>
        <div className="rounded-[6px] border border-border/60 px-4 py-6 text-center text-xs text-muted-foreground">
          加载中…
        </div>
      </section>
    )
  }

  const { best, completed_trials, pending } = pareto.data
  const finiteTrials = completed_trials
    .filter((t) => Number.isFinite(t.score))
    .sort((a, b) => a.score - b.score)
    .slice(0, 10)
  const noFinite = finiteTrials.length === 0

  return (
    <section className="space-y-2">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <h2 className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground/80 inline-flex items-center gap-1">
          <Trophy className="size-3" />
          Pareto / 最佳 trial
        </h2>
        <span className="text-[11px] text-muted-foreground">
          {completed_trials.length} 个 trial 已完成
          {pending > 0 && <> · {pending} 个进行中</>}
        </span>
      </div>

      {best ? (
        <div className="rounded-[6px] border border-emerald-500/30 bg-emerald-500/5 px-4 py-3 space-y-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[10px] uppercase tracking-[0.18em] text-emerald-600 dark:text-emerald-400 font-semibold">
              最佳
            </span>
            <span className="text-sm font-mono">
              loss = <span className="text-foreground">{best.score.toFixed(6)}</span>
            </span>
            <code className="text-[10px] text-muted-foreground/70">
              job: {best.job_id.slice(-12)}
            </code>
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
            {Object.entries(best.axis_values).map(([k, v]) => (
              <span key={k}>
                <span className="text-muted-foreground/70">{k}=</span>
                <span className="font-mono text-foreground/80">
                  {formatAxisValue(v)}
                </span>
              </span>
            ))}
          </div>
        </div>
      ) : (
        <div className="rounded-[6px] border border-border/60 px-4 py-3 text-xs text-muted-foreground">
          {noFinite
            ? "暂无成功完成的 trial。失败 / 缺指标的 trial 不计入 best。"
            : "等待 trial 完成…"}
        </div>
      )}

      {finiteTrials.length > 0 && (
        <div className="rounded-[6px] border border-border/60 overflow-hidden">
          <Table className="text-xs">
            <TableHeader>
              <TableRow>
                <TableHead className="w-12">#</TableHead>
                <TableHead className="w-32">loss</TableHead>
                <TableHead>axis 值</TableHead>
                <TableHead className="w-32">job</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {finiteTrials.map((t, idx) => (
                <TableRow
                  key={t.job_id}
                  className={cn(idx === 0 && "bg-emerald-500/5")}
                >
                  <TableCell className="font-mono text-muted-foreground">
                    {idx + 1}
                  </TableCell>
                  <TableCell className="font-mono">
                    {t.score.toFixed(6)}
                  </TableCell>
                  <TableCell className="font-mono text-[11px] truncate max-w-[26rem]">
                    {Object.entries(t.axis_values)
                      .map(([k, v]) => `${k}=${formatAxisValue(v)}`)
                      .join("  ")}
                  </TableCell>
                  <TableCell className="font-mono text-muted-foreground">
                    {t.job_id.slice(-12)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
      {mode === "grid" && completed_trials.length === 0 && pending === 0 && (
        <div className="text-[11px] text-muted-foreground">
          网格 sweep 通常不上报 per-trial loss;若有需要可在配置里开 validation。
        </div>
      )}
    </section>
  )
}
