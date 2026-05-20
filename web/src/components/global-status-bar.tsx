/**
 * Compact horizontal status bar pinned across the whole shell.
 *
 * Sources its data from useSystemStream so it stays in sync with the
 * dashboard's real-time payload. Falls back to a 10-second poll when the
 * live channel isn't open. Always visible above the page header so the
 * user never has to leave the current page to know "is the GPU busy / is
 * the network downloading / am I online?".
 */
import { useQuery } from "@tanstack/react-query"
import {
  Activity,
  ArrowDown,
  ArrowUp,
  Cpu,
  Download,
  MemoryStick,
  Wifi,
  WifiOff,
  Zap,
} from "lucide-react"
import { Link } from "react-router-dom"
import { api, useSystemStream, type SystemSnapshot } from "@/lib/api"
import { useJobsList } from "@/lib/queries/jobs"
import { useSystemVersion } from "@/hooks/use-system-version"
import { cn } from "@/lib/utils"

const POLL_MS = 10_000

export function GlobalStatusBar() {
  const stream = useSystemStream(true)
  const polled = useQuery({
    queryKey: ["system-stats-statusbar"],
    queryFn: api.getSystemStats,
    refetchInterval: stream.status === "open" ? false : POLL_MS,
  })
  const jobsQuery = useJobsList()

  const snapshot: SystemSnapshot | null =
    stream.snapshot ?? polled.data ?? null
  const running = (jobsQuery.data?.jobs ?? []).filter(
    (j) => j.state === "running",
  ).length

  return (
    <div className="shrink-0 border-b border-border/60 bg-background/80 backdrop-blur px-4 py-1.5 flex items-center gap-x-5 gap-y-1 flex-wrap text-[11px]">
      <ConnectionDot live={stream.status === "open"} />
      {snapshot ? (
        <>
          <Chip
            icon={<Cpu className="size-3" />}
            label="CPU"
            value={
              typeof snapshot.cpu.usage_percent === "number"
                ? `${snapshot.cpu.usage_percent.toFixed(0)}%`
                : "—"
            }
            tone={toneForPercent(snapshot.cpu.usage_percent)}
          />
          <Chip
            icon={<MemoryStick className="size-3" />}
            label="内存"
            value={`${snapshot.memory.percent.toFixed(0)}%`}
            tone={toneForPercent(snapshot.memory.percent)}
            sub={`${fmtBytes(snapshot.memory.used_bytes)} / ${fmtBytes(snapshot.memory.total_bytes)}`}
          />
          {snapshot.gpus[0] && (
            <Chip
              icon={<Zap className="size-3" />}
              label={`GPU0`}
              value={
                typeof snapshot.gpus[0].utilization_percent === "number"
                  ? `${snapshot.gpus[0].utilization_percent.toFixed(0)}%`
                  : "—"
              }
              tone={toneForPercent(snapshot.gpus[0].utilization_percent)}
              sub={
                snapshot.gpus[0].memory_total_bytes &&
                typeof snapshot.gpus[0].memory_used_bytes === "number"
                  ? `${fmtBytes(snapshot.gpus[0].memory_used_bytes)} / ${fmtBytes(snapshot.gpus[0].memory_total_bytes)}`
                  : undefined
              }
            />
          )}
          {snapshot.network && (
            <>
              <Chip
                icon={<ArrowDown className="size-3 text-emerald-600 dark:text-emerald-400" />}
                label="下载"
                value={fmtRate(snapshot.network.bytes_recv_per_sec)}
                tone="text-emerald-700 dark:text-emerald-400"
              />
              <Chip
                icon={<ArrowUp className="size-3 text-primary" />}
                label="上传"
                value={fmtRate(snapshot.network.bytes_sent_per_sec)}
                tone="text-primary"
              />
            </>
          )}
          <Chip
            icon={<Activity className="size-3" />}
            label="训练"
            value={running > 0 ? `${running} 个` : "空闲"}
            tone={running > 0 ? "text-primary" : "text-muted-foreground"}
          />
        </>
      ) : (
        <span className="text-muted-foreground/70">正在连接系统监控…</span>
      )}
      <UpdateBadge />
    </div>
  )
}

function ConnectionDot({ live }: { live: boolean }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 text-[10px] font-mono uppercase tracking-[0.15em]",
        live ? "text-emerald-700 dark:text-emerald-400" : "text-muted-foreground",
      )}
      title={live ? "实时事件流连接中" : "实时通道未连接，回退为 10 秒轮询"}
    >
      {live ? <Wifi className="size-3" /> : <WifiOff className="size-3" />}
      <span>{live ? "实时" : "轮询"}</span>
    </span>
  )
}

function Chip({
  icon,
  label,
  value,
  sub,
  tone,
}: {
  icon: React.ReactNode
  label: string
  value: string
  sub?: string
  tone: string
}) {
  return (
    <span
      className="inline-flex items-center gap-1.5 min-w-0 shrink-0"
      title={sub ?? undefined}
    >
      <span className="text-muted-foreground/70 shrink-0">{icon}</span>
      <span className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground/70">
        {label}
      </span>
      <span
        className={cn(
          "font-mono tabular-nums font-semibold text-right min-w-[5ch]",
          tone,
        )}
      >
        {value}
      </span>
    </span>
  )
}

function toneForPercent(p: number | null | undefined): string {
  if (typeof p !== "number") return "text-muted-foreground"
  if (p >= 90) return "text-destructive"
  if (p >= 70) return "text-amber-700 dark:text-amber-400"
  if (p >= 40) return "text-primary"
  return "text-emerald-700 dark:text-emerald-400"
}

function fmtBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0 B"
  const u = ["B", "KB", "MB", "GB", "TB", "PB"]
  let i = 0
  let v = n
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : v >= 10 ? 1 : 2)} ${u[i]}`
}

function fmtRate(b: number): string {
  if (!Number.isFinite(b) || b <= 0) return "0 B/s"
  if (b < 1024) return `${b.toFixed(0)} B/s`
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB/s`
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB/s`
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB/s`
}

/**
 * Right-aligned dot + label that lights up when the version-check
 * background job has spotted a release ahead of the current install.
 * Clicking deeplinks to ``/settings?tab=environment`` so the user
 * lands on the update card without hunting.
 */
function UpdateBadge() {
  const tag = useSystemVersion("tag")
  if (!tag.data) return null
  if (!tag.data.update_available) {
    // Stay quiet when up-to-date; the row already shows version info
    // on the maintenance page when the user actually wants it.
    return null
  }
  const label = tag.data.tag_name ?? tag.data.latest ?? "新版本"
  return (
    <Link
      to="/settings?tab=environment"
      className={cn(
        "ml-auto inline-flex items-center gap-1.5 rounded-[2px] border border-primary/40",
        "bg-primary/10 px-2 py-0.5 text-[11px] text-primary",
        "hover:bg-primary/15 transition-colors",
      )}
      title={`已检测到新版本 ${label}（点击查看详情）`}
    >
      <span className="relative flex size-1.5">
        <span className="absolute inline-flex size-full animate-ping rounded-full bg-primary/50" />
        <span className="relative inline-flex size-1.5 rounded-full bg-primary" />
      </span>
      <Download className="size-3" />
      新版本 {label}
    </Link>
  )
}
