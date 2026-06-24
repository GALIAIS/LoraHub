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
  Gauge,
  MemoryStick,
  Sparkles,
  Zap,
} from "lucide-react"
import { Link, useNavigate } from "react-router-dom"
import { api, useSystemStream, type SystemSnapshot } from "@/lib/api"
import type { JobSummary, SystemGpu } from "@/lib/api"
import { useJobsList } from "@/lib/queries/jobs"
import { useSystemVersion } from "@/hooks/use-system-version"
import { useStudioTasks } from "@/hooks/use-studio-tasks"
import type { StudioTaskRecord } from "@/lib/studio-task-store"
import { cn } from "@/lib/utils"

const POLL_MS = 10_000

export function GlobalStatusBar() {
  const stream = useSystemStream(true)
  const polled = useQuery({
    queryKey: ["system", "stats"],
    queryFn: api.getSystemStats,
    refetchInterval: stream.status === "open" ? false : POLL_MS,
    staleTime: 3_000,
  })
  const jobsQuery = useJobsList()
  const studioTasks = useStudioTasks()
  const studioRunning = studioTasks.filter((t) => t.status === "running")

  const snapshot: SystemSnapshot | null =
    stream.snapshot ?? polled.data ?? null
  const runningJobs = (jobsQuery.data?.jobs ?? []).filter(
    (j) => j.state === "running",
  )
  const gpuSummary = snapshot ? summarizeGpus(snapshot.gpus) : null

  return (
    <div className="shrink-0 border-b border-border/60 bg-background/92 px-3 py-1.5 text-[11px] backdrop-blur-xl md:px-4">
      <div className="no-scrollbar flex min-h-8 items-center gap-2 overflow-x-auto md:overflow-visible">
      <ConnectionDot live={stream.status === "open"} />
      {snapshot ? (
        <>
          <MetricChip
            className="hidden md:inline-flex"
            icon={<Cpu className="size-3" />}
            label="CPU"
            value={
              typeof snapshot.cpu.usage_percent === "number"
                ? `${snapshot.cpu.usage_percent.toFixed(0)}%`
                : "—"
            }
            tone={toneForPercent(snapshot.cpu.usage_percent)}
            percent={snapshot.cpu.usage_percent}
          />
          <MetricChip
            className="hidden sm:inline-flex"
            icon={<MemoryStick className="size-3" />}
            label="内存"
            value={`${snapshot.memory.percent.toFixed(0)}%`}
            tone={toneForPercent(snapshot.memory.percent)}
            sub={`${fmtBytes(snapshot.memory.used_bytes)} / ${fmtBytes(snapshot.memory.total_bytes)}`}
            percent={snapshot.memory.percent}
          />
          {snapshot.gpus.length > 0 && (
            <GpuStatusPanel snapshot={snapshot} runningJobs={runningJobs} />
          )}
          {gpuSummary && (
            <MetricChip
              className="hidden lg:inline-flex"
              icon={<Gauge className="size-3" />}
              label="显存"
              value={`${gpuSummary.memoryPercent.toFixed(0)}%`}
              tone={toneForPercent(gpuSummary.memoryPercent)}
              sub={`${fmtBytes(gpuSummary.memoryUsed)} / ${fmtBytes(gpuSummary.memoryTotal)}`}
              percent={gpuSummary.memoryPercent}
            />
          )}
          {snapshot.network && (
            <>
              <MetricChip
                icon={<ArrowDown className="size-3 text-emerald-600 dark:text-emerald-400" />}
                label="下载"
                value={fmtRate(snapshot.network.bytes_recv_per_sec)}
                tone="text-emerald-700 dark:text-emerald-400"
                valueWidth="min-w-[7ch]"
              />
              <MetricChip
                className="hidden md:inline-flex"
                icon={<ArrowUp className="size-3 text-primary" />}
                label="上传"
                value={fmtRate(snapshot.network.bytes_sent_per_sec)}
                tone="text-primary"
                valueWidth="min-w-[7ch]"
              />
            </>
          )}
          <MetricChip
            icon={<Activity className="size-3" />}
            label="训练"
            value={runningJobs.length > 0 ? `${runningJobs.length} 个` : "空闲"}
            tone={runningJobs.length > 0 ? "text-primary" : "text-muted-foreground"}
            sub={assignedGpuSlots(runningJobs) ?? undefined}
          />
          {studioRunning.length > 0 && (
            <StudioTaskChip tasks={studioRunning} />
          )}
        </>
      ) : (
        <span className="text-muted-foreground/70">正在连接系统监控…</span>
      )}
      <UpdateBadge />
      </div>
    </div>
  )
}

function ConnectionDot({ live }: { live: boolean }) {
  return (
    <span
      className={cn(
        "inline-flex h-7 shrink-0 items-center gap-1.5 rounded-[7px] border px-2",
        "font-mono text-[10px] font-medium tabular-nums tracking-[0.08em]",
        live
          ? "border-emerald-500/25 bg-emerald-500/8 text-emerald-700 dark:text-emerald-400"
          : "border-border/70 bg-muted/35 text-muted-foreground",
      )}
      title={live ? "实时事件流连接中" : "实时通道未连接，回退为 10 秒轮询"}
    >
      <span
        className={cn(
          "size-1.5 rounded-full",
          live ? "bg-emerald-500 shadow-[0_0_0_3px_rgba(16,185,129,0.14)]" : "bg-muted-foreground/45",
        )}
      />
      <span>{live ? "实时" : "轮询"}</span>
    </span>
  )
}

function MetricChip({
  icon,
  label,
  value,
  sub,
  tone,
  percent,
  valueWidth = "min-w-[5ch]",
  className,
}: {
  icon: React.ReactNode
  label: string
  value: string
  sub?: string
  tone: string
  percent?: number | null
  valueWidth?: string
  className?: string
}) {
  const boundedPercent =
    typeof percent === "number" && Number.isFinite(percent)
      ? Math.max(0, Math.min(100, percent))
      : null

  return (
    <span
      className={cn(
        "inline-flex h-7 min-w-0 shrink-0 items-center gap-2 rounded-[7px]",
        "border border-border/55 bg-muted/24 px-2 text-foreground/90",
        className,
      )}
      title={sub ?? undefined}
    >
      <span className="shrink-0 text-muted-foreground/65">{icon}</span>
      <span className="text-[10px] font-medium text-muted-foreground/75">
        {label}
      </span>
      <span className="grid min-w-0 gap-0.5">
        <span
          className={cn(
            "font-mono text-[11px] font-semibold leading-none tabular-nums text-right",
            valueWidth,
            tone,
          )}
        >
          {value}
        </span>
        {boundedPercent !== null && (
          <span className="h-0.5 overflow-hidden rounded-full bg-border/70">
            <span
              className={cn("block h-full rounded-full", barToneForPercent(boundedPercent))}
              style={{ width: `${boundedPercent}%` }}
            />
          </span>
        )}
      </span>
    </span>
  )
}

function GpuStatusPanel({
  snapshot,
  runningJobs,
}: {
  snapshot: SystemSnapshot
  runningJobs: JobSummary[]
}) {
  const gpus = snapshot.gpus
  const active = gpus.filter((g) => (g.utilization_percent ?? 0) >= 5).length
  const avgUtil =
    gpus.reduce((sum, g) => sum + (g.utilization_percent ?? 0), 0) / gpus.length
  const slots = assignedGpuSlots(runningJobs)

  return (
    <details className="group relative shrink-0">
      <summary
        className={cn(
          "inline-flex h-7 cursor-pointer list-none items-center gap-2 rounded-[7px]",
          "border border-border/60 bg-muted/24 px-2 text-foreground/90",
          "hover:border-primary/35 hover:bg-muted/40",
        )}
        title="点击查看多卡状态"
      >
        <Zap className="size-3 text-muted-foreground/70" />
        <span className="text-[10px] font-medium text-muted-foreground/75">
          GPU
        </span>
        <span className={cn("font-mono text-[11px] font-semibold tabular-nums", toneForPercent(avgUtil))}>
          {active}/{gpus.length}
        </span>
        <span className="flex max-w-24 items-center gap-1">
          {gpus.slice(0, 8).map((gpu) => (
            <span
              key={gpu.index}
              className="h-3 w-1.5 overflow-hidden rounded-full bg-border/70"
              title={`GPU${gpu.index} ${gpu.name}`}
            >
              <span
                className={cn("block w-full rounded-full", barToneForPercent(gpu.utilization_percent ?? 0))}
                style={{ height: `${Math.max(3, gpu.utilization_percent ?? 0)}%` }}
              />
            </span>
          ))}
        </span>
        {slots && (
          <span className="hidden font-mono text-[10px] text-muted-foreground sm:inline">
            {slots}
          </span>
        )}
      </summary>
      <div
        className={cn(
          "absolute left-0 top-8 z-50 w-[min(92vw,34rem)] rounded-[8px]",
          "border border-border/70 bg-popover p-3 text-popover-foreground shadow-xl",
        )}
      >
        <div className="mb-2 flex items-center justify-between gap-3">
          <div className="text-xs font-semibold">多卡状态</div>
          <div className="font-mono text-[10px] text-muted-foreground">
            {active}/{gpus.length} active
          </div>
        </div>
        <div className="grid gap-2">
          {gpus.map((gpu) => (
            <GpuRow
              key={gpu.index}
              gpu={gpu}
              processCount={
                snapshot.gpu_processes?.filter((p) => p.gpu_index === gpu.index).length ?? 0
              }
              assigned={runningJobs.filter((job) =>
                jobGpuSlots(job).includes(gpu.index),
              ).length}
            />
          ))}
        </div>
      </div>
    </details>
  )
}

function GpuRow({
  gpu,
  processCount,
  assigned,
}: {
  gpu: SystemGpu
  processCount: number
  assigned: number
}) {
  const util = gpu.utilization_percent ?? 0
  const memPercent =
    gpu.memory_total_bytes && typeof gpu.memory_used_bytes === "number"
      ? (gpu.memory_used_bytes / gpu.memory_total_bytes) * 100
      : null
  return (
    <div className="rounded-[6px] border border-border/55 bg-muted/20 p-2">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs font-semibold">GPU{gpu.index}</span>
            <span className="truncate text-xs text-muted-foreground">{gpu.name}</span>
          </div>
          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[10px] text-muted-foreground">
            <span>util {fmtPercent(gpu.utilization_percent)}</span>
            <span>vram {memPercent === null ? "—" : fmtPercent(memPercent)}</span>
            {gpu.temperature_c !== null && <span>{gpu.temperature_c.toFixed(0)}°C</span>}
            {gpu.power_w !== null && <span>{gpu.power_w.toFixed(0)}W</span>}
            {assigned > 0 && <span>{assigned} job</span>}
            {processCount > 0 && <span>{processCount} proc</span>}
          </div>
        </div>
        <span className={cn("font-mono text-xs font-semibold tabular-nums", toneForPercent(util))}>
          {fmtPercent(util)}
        </span>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2">
        <TinyBar value={util} />
        <TinyBar value={memPercent} />
      </div>
    </div>
  )
}

function TinyBar({ value }: { value: number | null }) {
  const bounded =
    typeof value === "number" && Number.isFinite(value)
      ? Math.max(0, Math.min(100, value))
      : 0
  return (
    <span className="h-1 overflow-hidden rounded-full bg-border/65">
      <span
        className={cn("block h-full rounded-full", barToneForPercent(bounded))}
        style={{ width: `${bounded}%` }}
      />
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

function barToneForPercent(p: number): string {
  if (p >= 90) return "bg-destructive"
  if (p >= 70) return "bg-amber-500"
  if (p >= 40) return "bg-primary"
  return "bg-emerald-500"
}

function summarizeGpus(gpus: SystemGpu[]) {
  const withMem = gpus.filter(
    (g) => g.memory_total_bytes && typeof g.memory_used_bytes === "number",
  )
  if (withMem.length === 0) return null
  const memoryUsed = withMem.reduce((sum, g) => sum + (g.memory_used_bytes ?? 0), 0)
  const memoryTotal = withMem.reduce((sum, g) => sum + (g.memory_total_bytes ?? 0), 0)
  if (memoryTotal <= 0) return null
  return {
    memoryUsed,
    memoryTotal,
    memoryPercent: (memoryUsed / memoryTotal) * 100,
  }
}

function jobGpuSlots(job: JobSummary): number[] {
  const raw = job.metadata?.gpu_slots
  if (!Array.isArray(raw)) return []
  return raw.filter((v): v is number => typeof v === "number" && Number.isFinite(v))
}

function assignedGpuSlots(jobs: JobSummary[]): string | null {
  const slots = Array.from(new Set(jobs.flatMap(jobGpuSlots))).sort((a, b) => a - b)
  return slots.length > 0 ? `GPU ${slots.join(",")}` : null
}

function fmtPercent(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value.toFixed(0)}%`
    : "—"
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
 * Pinned chip that surfaces the count of in-flight image-studio AI
 * batch tasks (smart-caption, WD14, trigger-words, quality scoring).
 *
 * Clicking jumps to the dataset where the most-recently-started task
 * is running — that's the page whose progress banner the user is
 * most likely chasing. If the dataset path can't be expressed as a
 * route param the chip stays as plain text rather than producing a
 * dead link.
 */
function StudioTaskChip({ tasks }: { tasks: StudioTaskRecord[] }) {
  const navigate = useNavigate()
  if (tasks.length === 0) return null
  // Newest first so we prefer "the thing the user just kicked off".
  const newest = [...tasks].sort((a, b) => b.startedAt - a.startedAt)[0]
  const onClick = () => {
    if (!newest?.datasetPath) return
    navigate(
      `/image-studio?path=${encodeURIComponent(newest.datasetPath)}`,
    )
  }
  return (
    <button
      type="button"
      onClick={onClick}
      title={tasks.map((t) => `${t.label} · ${t.datasetPath}`).join("\n")}
      className="inline-flex items-center gap-1.5 min-w-0 shrink-0 hover:text-primary transition-colors cursor-pointer"
    >
      <Sparkles className="size-3 text-primary animate-pulse" />
      <span className="text-[10px] font-medium text-muted-foreground/75">
        AI 任务
      </span>
      <span className="font-mono tabular-nums font-semibold text-primary">
        {tasks.length} 个
      </span>
    </button>
  )
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
        "ml-auto inline-flex shrink-0 items-center gap-1.5 rounded-[6px] border border-primary/40",
        "bg-primary/10 px-2 py-0.5 text-[11px] text-primary",
        "hover:bg-primary/15 transition-colors",
      )}
      title={`已检测到新版本 ${label}（点击查看详情）`}
    >
      <span className="relative flex size-1.5">
        <span className="absolute inline-flex size-full animate-ping rounded-full bg-primary/50" />
        <span className="relative inline-flex size-1.5 rounded-full bg-primary" />
      </span>
      新版本 {label}
    </Link>
  )
}
