import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useNavigate } from "react-router-dom"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Archive, ArrowRight, Loader2, PanelLeftClose, PanelLeftOpen } from "lucide-react"
import { toast } from "sonner"
import { api, type JobSummary } from "@/lib/api"
import { useJobsList } from "@/lib/queries/jobs"
import { readBool, readList, useUrlState } from "@/lib/url-state"
import { Button } from "@/components/ui/button"
import { WorkbenchSplitLayout } from "@/components/workbench-split-layout"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
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
  // URL-backed UI state — keeps selection / filters / compare baskets
  // alive across route switches and survives deep links from sweeps,
  // analysis, and the gallery.
  const { params, update } = useUrlState()
  const selectedId = params.get("id")
  const query = params.get("q") ?? ""
  const rawStatus = params.get("status") as StatusFilter | null
  const statusFilter: StatusFilter =
    rawStatus && STATUS_GROUPS[rawStatus] ? rawStatus : "all"
  const hideCompleted = readBool(params, "hide_completed")
  const compareMode = readBool(params, "compare")
  const compareIds = useMemo(
    () => readList(params, "compare_ids").slice(0, COMPARE_LIMIT),
    [params],
  )

  const setSelectedId = useCallback(
    (id: string | null) => {
      update({ id: id ?? null })
    },
    [update],
  )
  const setQuery = useCallback(
    (next: string) => {
      update({ q: next || null })
    },
    [update],
  )
  const setStatusFilter = useCallback(
    (next: StatusFilter) => {
      update({ status: next === "all" ? null : next })
    },
    [update],
  )
  const setHideCompleted = useCallback(
    (next: boolean) => {
      update({ hide_completed: next ? "1" : null })
    },
    [update],
  )
  const setCompareMode = useCallback(
    (next: boolean) => {
      // Turning compare mode off also drops the basket — the previous
      // useState version did this via effect; doing both writes in one
      // patch keeps the URL clean (no transient ?compare_ids=… without
      // ?compare=1).
      update({
        compare: next ? "1" : null,
        compare_ids: next ? undefined : null,
      })
    },
    [update],
  )
  const setCompareIds = useCallback(
    (ids: string[]) => {
      update({ compare_ids: ids.length ? ids.join(",") : null })
    },
    [update],
  )

  // Bulk-archive selection lives outside the URL — it's a transient
  // action, not something users want to share or restore on reload.
  const [selectMode, setSelectMode] = useState(false)
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [bulkArchiveOpen, setBulkArchiveOpen] = useState(false)
  const navigate = useNavigate()
  const qc = useQueryClient()
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(() => {
    if (typeof window === "undefined") return true
    return window.localStorage.getItem(SIDEBAR_KEY) !== "closed"
  })

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
    if (!jobs.isSuccess) return
    autoSelectedRef.current = true
    if (selectedId) return
    if (visibleJobs.length === 0) return
    setSelectedId(visibleJobs[0].id)
  }, [jobs.isSuccess, selectedId, visibleJobs, setSelectedId])

  // Mirror the compareMode → drop basket lifecycle for select mode.
  // (Compare-side cleanup already happens inline in setCompareMode so
  // both writes share a single URL update.)
  useEffect(() => {
    if (!selectMode) setSelectedIds([])
  }, [selectMode])

  // Prune compare ids that no longer exist (e.g. archived).
  useEffect(() => {
    if (compareIds.length === 0) return
    const known = new Set(list.map((j) => j.id))
    const pruned = compareIds.filter((id) => known.has(id))
    if (pruned.length !== compareIds.length) setCompareIds(pruned)
  }, [list, compareIds, setCompareIds])

  // Same prune for batch-selection ids — a successful bulk archive
  // removes the rows, so the checkboxes need to drop along with them.
  useEffect(() => {
    if (selectedIds.length === 0) return
    const known = new Set(list.map((j) => j.id))
    const pruned = selectedIds.filter((id) => known.has(id))
    if (pruned.length !== selectedIds.length) setSelectedIds(pruned)
  }, [list, selectedIds])

  const selected = selectedId
    ? list.find((j) => j.id === selectedId) ?? null
    : null

  function toggleCompare(id: string) {
    if (compareIds.includes(id)) {
      setCompareIds(compareIds.filter((x) => x !== id))
      return
    }
    if (compareIds.length >= COMPARE_LIMIT) return
    setCompareIds([...compareIds, id])
  }

  function toggleSelected(id: string) {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    )
  }

  // Selectable = visible jobs in a terminal state. The bulk-archive
  // server-side check enforces this anyway, but pre-filtering on the
  // client keeps the "全选" button honest about what will happen.
  const selectableIds = useMemo(
    () => visibleJobs.filter((j) => COMPLETED_STATES.has(j.state)).map((j) => j.id),
    [visibleJobs],
  )
  const allSelectable = selectableIds.length > 0
    && selectableIds.every((id) => selectedIds.includes(id))

  function selectAllVisible() {
    setSelectedIds((prev) => Array.from(new Set([...prev, ...selectableIds])))
  }
  function clearSelection() {
    setSelectedIds([])
  }

  const bulkArchive = useMutation({
    mutationFn: (ids: string[]) => api.bulkArchiveJobs(ids),
    onSuccess: (data) => {
      const a = data.archived.length
      const s = data.skipped.length
      const f = data.failed.length
      const nf = data.not_found.length
      // Build a single toast summary; details land in the
      // description for skipped/failed reasons.
      const parts: string[] = []
      if (a) parts.push(`成功 ${a}`)
      if (s) parts.push(`跳过 ${s}`)
      if (f) parts.push(`失败 ${f}`)
      if (nf) parts.push(`未找到 ${nf}`)
      const headline = parts.join(" · ") || "无事可做"

      const lines: string[] = []
      for (const row of data.skipped.slice(0, 5)) {
        lines.push(`${row.id.slice(-8)}: ${row.reason}`)
      }
      for (const row of data.failed.slice(0, 5)) {
        lines.push(`${row.id.slice(-8)}: ${row.reason}`)
      }
      const description = lines.length ? lines.join("\n") : undefined

      if (f > 0) {
        toast.error(headline, { description })
      } else if (s > 0 || nf > 0) {
        toast.warning(headline, { description })
      } else {
        toast.success(headline)
      }
      qc.invalidateQueries({ queryKey: ["jobs"] })
      qc.invalidateQueries({ queryKey: ["artifacts"] })
      // Clear selection — successfully-archived ids were removed by the
      // server; pruning effect catches the rest, but emptying here makes
      // the toolbar feedback immediate.
      setSelectedIds([])
    },
    onError: (e) => {
      toast.error("批量归档失败", {
        description: e instanceof Error ? e.message : String(e),
      })
    },
  })

  return (
    <>
      <WorkbenchSplitLayout
        sidebarOpen={sidebarOpen}
        sidebarWidth="minmax(280px,340px)"
        mobileSidebarTitle="训练任务列表"
        mobileSidebarDescription="筛选训练任务、选择任务、进入对比或批量归档。"
        sidebar={
          <>
        <div className="flex items-center justify-between px-4 pt-3">
          <span className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
            训练任务
          </span>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setSidebarOpen(false)}
            title="收起侧栏"
            className="hidden md:inline-flex"
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
          selectMode={selectMode}
          onSelectModeChange={setSelectMode}
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
        {selectMode && (
          <div className="px-5 py-2 border-b border-border/60 bg-muted/30 text-[11px] text-muted-foreground flex items-center gap-2 flex-wrap">
            <span className="flex-1 min-w-0">
              已选 {selectedIds.length}
              {selectedIds.length > 0 && " 个"}
              {selectableIds.length > 0 && (
                <span className="text-muted-foreground/70">
                  {" "}/ 可选 {selectableIds.length}
                </span>
              )}
            </span>
            <Button
              size="sm"
              variant="ghost"
              className="h-6 text-[11px] px-2"
              onClick={allSelectable ? clearSelection : selectAllVisible}
              disabled={selectableIds.length === 0}
              title="只能选择已完成 / 失败 / 取消 / 中断的任务"
            >
              {allSelectable ? "清空" : "全选可选"}
            </Button>
            <Button
              size="sm"
              variant="destructive"
              className="h-6 text-[11px] px-2 gap-1"
              disabled={selectedIds.length === 0 || bulkArchive.isPending}
              onClick={() => setBulkArchiveOpen(true)}
            >
              {bulkArchive.isPending ? (
                <Loader2 className="size-3 animate-spin" />
              ) : (
                <Archive className="size-3" />
              )}
              批量归档
            </Button>
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
              const isCompare = compareMode
              const isSelect = selectMode
              const checked = isCompare
                ? compareIds.includes(j.id)
                : selectedIds.includes(j.id)
              const compareLimitHit =
                isCompare && !checked && compareIds.length >= COMPARE_LIMIT
              const notSelectable =
                isSelect && !COMPLETED_STATES.has(j.state)
              const checkboxDisabled = compareLimitHit || notSelectable
              return (
                <JobRow
                  key={j.id}
                  job={j}
                  active={j.id === selectedId}
                  compareMode={isCompare || isSelect}
                  checked={checked}
                  checkboxDisabled={checkboxDisabled}
                  onSelect={() => setSelectedId(j.id)}
                  onToggleCompare={() =>
                    isCompare ? toggleCompare(j.id) : toggleSelected(j.id)
                  }
                />
              )
            })}
          </ul>
        </ScrollArea>
          </>
        }
      >
        {!sidebarOpen && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => setSidebarOpen(true)}
            className="absolute left-3 top-3 z-10 hidden shadow-[var(--panel-shadow)] md:inline-flex"
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
      </WorkbenchSplitLayout>

      <AlertDialog
        open={bulkArchiveOpen}
        onOpenChange={(open) => !open && setBulkArchiveOpen(false)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>批量归档 {selectedIds.length} 个任务</AlertDialogTitle>
            <AlertDialogDescription className="space-y-2">
              <span className="block">
                即将把所选任务的工作区移动到{" "}
                <code className="font-mono text-xs">_archive/</code> 目录，
                并从任务列表中移除这些记录。
              </span>
              <span className="block text-muted-foreground">
                跟「删除」不同 · 文件不会被销毁，日后可以从 _archive/ 找回；但
                训练详情和 metrics 时间线不再可见。
              </span>
              <span className="block text-amber-700 dark:text-amber-400">
                未完成 / 与活动任务共用工作区的任务会被自动跳过。
              </span>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={bulkArchive.isPending}>
              取消
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                bulkArchive.mutate(selectedIds)
                setBulkArchiveOpen(false)
              }}
              disabled={bulkArchive.isPending}
            >
              {bulkArchive.isPending ? "归档中…" : "确认归档"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}
