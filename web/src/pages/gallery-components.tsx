import { useMemo, useState } from "react"
import { useNavigate } from "react-router-dom"
import {
  ChevronLeft,
  ChevronRight,
  Download,
  ExternalLink,
  Images,
} from "lucide-react"
import type { JobSummary, SampleGalleryItem } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { cn } from "@/lib/utils"
import { fmtBytes, fmtUnixSeconds } from "./jobs/utils"

export function JobFilterChips({
  jobs,
  selectedJobIds,
  onToggle,
  onSelectAll,
  onClear,
}: {
  jobs: JobSummary[]
  selectedJobIds: string[]
  onToggle: (id: string) => void
  onSelectAll: () => void
  onClear: () => void
}) {
  const ordered = useMemo(
    () =>
      [...jobs].sort((a, b) => (a.created_at < b.created_at ? 1 : -1)),
    [jobs],
  )
  if (ordered.length === 0) {
    return (
      <div className="text-[11px] text-muted-foreground/70">
        当前还没有任务记录。
      </div>
    )
  }
  const allSelected = selectedJobIds.length === ordered.length
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.16em] text-muted-foreground/70">
        <span>按任务筛选</span>
        <button
          type="button"
          onClick={allSelected ? onClear : onSelectAll}
          className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground hover:text-foreground transition-colors"
        >
          {allSelected ? "反选" : "全选"}
        </button>
      </div>
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
              title={`${job.workspace} · ${new Date(job.created_at).toLocaleString()}`}
            >
              <span>{shortId}</span>
              {wsName && (
                <span className="ml-1.5 text-muted-foreground/70">
                  {wsName}
                </span>
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}

export function SampleTile({
  item,
  onOpen,
}: {
  item: SampleGalleryItem
  onOpen: () => void
}) {
  const [broken, setBroken] = useState(false)
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
        {broken ? (
          <div className="flex flex-col items-center gap-1 text-muted-foreground/70 text-[10px]">
            <Images className="size-5" />
            <span>样图已不可用</span>
          </div>
        ) : (
          <img
            src={item.raw_url}
            loading="lazy"
            alt={filename}
            className="w-full h-full object-cover group-hover:scale-[1.02] transition-transform"
            onError={() => setBroken(true)}
          />
        )}
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

export function SampleLightbox({
  item,
  onClose,
  onPrev,
  onNext,
  hasPrev,
  hasNext,
}: {
  item: SampleGalleryItem | null
  onClose: () => void
  onPrev: () => void
  onNext: () => void
  hasPrev: boolean
  hasNext: boolean
}) {
  const navigate = useNavigate()
  const filename = item ? item.path.split(/[\\/]/).pop() ?? item.path : ""
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
            <div className="relative rounded-[4px] border border-border/60 bg-muted/40 overflow-hidden grid place-items-center max-h-[70vh]">
              <img
                src={item.raw_url}
                alt={item.path}
                className="max-h-[70vh] w-auto object-contain"
              />
              {hasPrev && (
                <button
                  type="button"
                  onClick={onPrev}
                  className="absolute left-2 top-1/2 -translate-y-1/2 rounded-full bg-background/80 p-1.5 text-foreground border border-border/60 hover:bg-background transition-colors"
                  title="上一张 (←)"
                >
                  <ChevronLeft className="size-4" />
                </button>
              )}
              {hasNext && (
                <button
                  type="button"
                  onClick={onNext}
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full bg-background/80 p-1.5 text-foreground border border-border/60 hover:bg-background transition-colors"
                  title="下一张 (→)"
                >
                  <ChevronRight className="size-4" />
                </button>
              )}
            </div>
            <dl className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
              <Stat label="任务" value={item.job_id.slice(-8)} />
              <Stat label="工作区" value={item.job_name} />
              <Stat label="配置" value={item.config_name ?? "—"} />
              <Stat label="大小" value={fmtBytes(item.size_bytes)} />
              <Stat label="修改时间" value={fmtUnixSeconds(item.modified_at)} />
            </dl>
            <div className="flex items-center justify-end gap-2 pt-1">
              <Button
                variant="outline"
                size="sm"
                render={<a href={item.raw_url} download={filename} />}
              >
                <Download className="size-3" /> 下载
              </Button>
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
      <div
        className="mt-0.5 font-mono tabular-nums text-[12px] truncate"
        title={value}
      >
        {value}
      </div>
    </div>
  )
}

function fmtRelativeTime(ts: number): string {
  if (!Number.isFinite(ts) || ts <= 0) return "—"
  const now = Date.now() / 1000
  const delta = now - ts
  if (delta < 5) return "刚刚"
  if (delta < 60) return `${Math.floor(delta)} 秒前`
  if (delta < 3600) return `${Math.floor(delta / 60)} 分钟前`
  if (delta < 86400) return `${Math.floor(delta / 3600)} 小时前`
  if (delta < 86400 * 30) return `${Math.floor(delta / 86400)} 天前`
  return new Date(ts * 1000).toLocaleDateString()
}
