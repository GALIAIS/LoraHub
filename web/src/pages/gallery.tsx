import { useEffect, useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { ChevronRight, ExternalLink, Images, X } from "lucide-react"
import {
  api,
  type JobSummary,
  type SampleGalleryItem,
} from "@/lib/api"
import { useJobsList } from "@/lib/queries/jobs"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Pagination } from "@/components/ui/pagination"
import { cn } from "@/lib/utils"
import { fmtBytes, fmtUnixSeconds } from "./jobs/utils"

const PAGE_SIZE_OPTIONS = [24, 48, 96, 192]

export function GalleryPage() {
  const [selectedJobIds, setSelectedJobIds] = useState<string[]>([])
  const [active, setActive] = useState<SampleGalleryItem | null>(null)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(48)

  // Reset to page 1 whenever the filter changes — otherwise the user can
  // be parked on page 5 of an empty filter result.
  useEffect(() => {
    setPage(1)
  }, [selectedJobIds, pageSize])

  const jobs = useJobsList()

  // Refresh the gallery on a slow cadence so freshly produced samples
  // appear without forcing a manual reload, but not so fast that every
  // mouse-move re-fetches the entire feed.
  const offset = (page - 1) * pageSize
  const samples = useQuery({
    queryKey: ["samples", selectedJobIds, pageSize, offset],
    queryFn: () =>
      api.listSamples({
        limit: pageSize,
        offset,
        jobIds: selectedJobIds.length > 0 ? selectedJobIds : undefined,
      }),
    refetchInterval: 6000,
  })

  const items = samples.data?.items ?? []
  const total = samples.data?.total ?? 0
  const allJobs = jobs.data?.jobs ?? []

  return (
    <div className="h-full overflow-y-auto">
      <div className="px-8 py-7 space-y-6 w-full">
        <header className="space-y-1">
          <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
            训练产物
          </div>
          <h1 className="text-2xl font-semibold tracking-tight inline-flex items-center gap-2">
            <Images className="size-5 text-muted-foreground" />
            样图画廊
          </h1>
          <p className="text-sm text-muted-foreground">
            汇总所有任务工作区里的样图，按修改时间倒序展示。
          </p>
        </header>

        <Card>
          <CardContent className="px-4 py-3 space-y-3">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <Pagination
                total={total}
                pageSize={pageSize}
                page={page}
                onPageChange={setPage}
                pageSizeOptions={PAGE_SIZE_OPTIONS}
                onPageSizeChange={setPageSize}
                className="flex-1"
              />
              {selectedJobIds.length > 0 && (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setSelectedJobIds([])}
                  className="h-7 text-[11px]"
                >
                  <X className="size-3" /> 清除筛选
                </Button>
              )}
            </div>
            <JobFilterChips
              jobs={allJobs}
              selectedJobIds={selectedJobIds}
              onToggle={(id) =>
                setSelectedJobIds((prev) =>
                  prev.includes(id)
                    ? prev.filter((p) => p !== id)
                    : [...prev, id],
                )
              }
            />
          </CardContent>
        </Card>

        {samples.isLoading && (
          <div className="text-center text-sm text-muted-foreground py-12">
            正在加载样图…
          </div>
        )}

        {samples.isError && (
          <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-4 py-3 text-xs font-mono text-destructive">
            {(samples.error as Error).message}
          </div>
        )}

        {!samples.isLoading && items.length === 0 && (
          <div className="rounded-[6px] border border-dashed border-border/70 bg-muted/30 px-6 py-16 text-center">
            <Images className="size-8 mx-auto text-muted-foreground/60" />
            <div className="mt-3 text-sm text-muted-foreground">
              {selectedJobIds.length > 0
                ? "选中的任务还没有样图。"
                : "还没有任何任务产出样图。"}
            </div>
          </div>
        )}

        {items.length > 0 && (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
              {items.map((item) => (
                <SampleTile
                  key={`${item.job_id}:${item.path}`}
                  item={item}
                  onOpen={() => setActive(item)}
                />
              ))}
            </div>
            <Pagination
              total={total}
              pageSize={pageSize}
              page={page}
              onPageChange={setPage}
              pageSizeOptions={PAGE_SIZE_OPTIONS}
              onPageSizeChange={setPageSize}
            />
          </>
        )}
      </div>

      <SampleLightbox
        item={active}
        onClose={() => setActive(null)}
      />
    </div>
  )
}

function JobFilterChips({
  jobs,
  selectedJobIds,
  onToggle,
}: {
  jobs: JobSummary[]
  selectedJobIds: string[]
  onToggle: (id: string) => void
}) {
  if (jobs.length === 0) {
    return (
      <div className="text-[11px] text-muted-foreground/70">
        当前还没有任务记录。
      </div>
    )
  }
  // Newest jobs first so a fresh run is one click away from the dashboard.
  const ordered = useMemo(
    () =>
      [...jobs].sort((a, b) => (a.created_at < b.created_at ? 1 : -1)),
    [jobs],
  )
  return (
    <div className="flex flex-wrap gap-1.5">
      {ordered.map((job) => {
        const active = selectedJobIds.includes(job.id)
        const shortId = job.id.slice(-8)
        const wsName = job.workspace.split(/[\\/]/).filter(Boolean).pop() ?? ""
        return (
          <button
            key={job.id}
            type="button"
            onClick={() => onToggle(job.id)}
            className={cn(
              "rounded-[3px] border px-2 py-1 text-[11px] font-mono transition-colors",
              active
                ? "border-primary bg-primary/10 text-primary"
                : "border-border/60 bg-background/60 text-muted-foreground hover:bg-muted/40",
            )}
            title={job.workspace}
          >
            <span>{shortId}</span>
            {wsName && (
              <span className="ml-1.5 text-muted-foreground/70">{wsName}</span>
            )}
          </button>
        )
      })}
    </div>
  )
}

function SampleTile({
  item,
  onOpen,
}: {
  item: SampleGalleryItem
  onOpen: () => void
}) {
  const filename = item.path.split(/[\\/]/).pop() ?? item.path
  const shortId = item.job_id.slice(-8)
  return (
    <button
      type="button"
      onClick={onOpen}
      className="group block rounded-[4px] border border-border/60 overflow-hidden bg-card/50 hover:border-primary/50 transition-colors text-left"
      title={`${shortId} · ${item.path}`}
    >
      <div className="aspect-square bg-muted/40 grid place-items-center overflow-hidden">
        <img
          src={item.raw_url}
          loading="lazy"
          alt={filename}
          className="w-full h-full object-cover group-hover:scale-[1.02] transition-transform"
        />
      </div>
      <div className="px-2 py-1.5 text-[11px] space-y-0.5">
        <div className="flex items-center justify-between gap-1.5">
          <span className="font-mono">{shortId}</span>
          {item.config_name && (
            <span
              className="text-muted-foreground/80 truncate"
              title={item.config_name}
            >
              {item.config_name}
            </span>
          )}
        </div>
        <div className="text-muted-foreground/70 tabular-nums truncate">
          {fmtBytes(item.size_bytes)} · {fmtRelativeTime(item.modified_at)}
        </div>
      </div>
    </button>
  )
}

function SampleLightbox({
  item,
  onClose,
}: {
  item: SampleGalleryItem | null
  onClose: () => void
}) {
  const navigate = useNavigate()
  return (
    <Dialog open={!!item} onOpenChange={(next) => !next && onClose()}>
      <DialogContent className="max-w-[min(calc(100%-2rem),64rem)]">
        {item && (
          <>
            <DialogHeader>
              <DialogTitle className="font-mono text-sm break-all">
                {item.path}
              </DialogTitle>
            </DialogHeader>
            <div className="rounded-[4px] border border-border/60 bg-muted/40 overflow-hidden grid place-items-center max-h-[70vh]">
              <img
                src={item.raw_url}
                alt={item.path}
                className="max-h-[70vh] w-auto object-contain"
              />
            </div>
            <dl className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
              <Stat label="任务" value={item.job_id.slice(-8)} />
              <Stat label="工作区" value={item.job_name} />
              <Stat
                label="配置"
                value={item.config_name ?? "—"}
              />
              <Stat
                label="大小"
                value={fmtBytes(item.size_bytes)}
              />
              <Stat
                label="修改时间"
                value={fmtUnixSeconds(item.modified_at)}
              />
            </dl>
            <div className="flex items-center justify-end gap-2 pt-1">
              <Button
                variant="outline"
                size="sm"
                onClick={() => window.open(item.raw_url, "_blank")}
              >
                <ExternalLink className="size-3" /> 在新标签页打开
              </Button>
              <Button
                size="sm"
                onClick={() => {
                  onClose()
                  navigate(`/jobs?focus=${encodeURIComponent(item.job_id)}`)
                }}
              >
                <ChevronRight className="size-3" /> 前往任务
              </Button>
            </div>
            {item.config_name && (
              <Badge
                variant="outline"
                className="rounded-[2px] mt-1 self-start text-[10px]"
              >
                {item.config_name}
              </Badge>
            )}
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[3px] border border-border/60 bg-background/45 px-2 py-1.5 min-w-0">
      <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
        {label}
      </div>
      <div className="mt-0.5 font-mono tabular-nums text-[12px] truncate" title={value}>
        {value}
      </div>
    </div>
  )
}

function fmtRelativeTime(ts: number): string {
  if (!Number.isFinite(ts) || ts <= 0) return "—"
  const now = Date.now() / 1000
  const delta = now - ts
  if (delta < 60) return "刚刚"
  if (delta < 3600) return `${Math.floor(delta / 60)} 分钟前`
  if (delta < 86400) return `${Math.floor(delta / 3600)} 小时前`
  if (delta < 86400 * 30) return `${Math.floor(delta / 86400)} 天前`
  return new Date(ts * 1000).toLocaleDateString()
}
