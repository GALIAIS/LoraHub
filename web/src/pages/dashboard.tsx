/**
 * 实时数据面板：硬件状态 + 任务概览。
 *
 * 通过 WebSocket /api/system/stream 每秒接收一次系统快照；WS 不可用时回退到
 * 5 秒一次的 REST 轮询。任务统计仍走 /api/jobs（3 秒轮询）。
 */
import { useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import {
  Activity,
  AlertTriangle,
  BatteryCharging,
  BatteryFull,
  CheckCircle2,
  CircleX,
  Cpu,
  HardDrive,
  Loader2,
  MemoryStick,
  Pause,
  Server,
  Thermometer,
  Zap,
} from "lucide-react"
import {
  api,
  useSystemStream,
  type JobSummary,
  type SystemBattery,
  type SystemDisk,
  type SystemGpu,
  type SystemSnapshot,
} from "@/lib/api"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress, ProgressTrack, ProgressIndicator } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { cn } from "@/lib/utils"

const POLL_INTERVAL_JOBS_MS = 3_000
const POLL_INTERVAL_SYSTEM_MS = 5_000

export function DashboardPage() {
  const stream = useSystemStream(true)

  // WS 不通时降级为 5s REST 轮询，保证看板永远有数据。
  const polled = useQuery({
    queryKey: ["system-stats"],
    queryFn: api.getSystemStats,
    refetchInterval: stream.status === "open" ? false : POLL_INTERVAL_SYSTEM_MS,
  })

  const snapshot: SystemSnapshot | null = stream.snapshot ?? polled.data ?? null

  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: api.listJobs,
    refetchInterval: POLL_INTERVAL_JOBS_MS,
  })
  const allJobs = jobs.data?.jobs ?? []

  const stats = useMemo(() => {
    const running = allJobs.filter((j) => j.state === "running").length
    const queued = allJobs.filter((j) => j.state === "queued").length
    const succeeded = allJobs.filter((j) => j.state === "succeeded").length
    const failed = allJobs.filter(
      (j) =>
        j.state === "failed" || j.state === "interrupted" || j.state === "canceled",
    ).length
    return { running, queued, succeeded, failed }
  }, [allJobs])

  const wsLive = stream.status === "open"

  return (
    <div className="h-full overflow-y-auto">
      <div className="px-8 py-7 space-y-6 max-w-[1280px]">
        <header className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
              实时概览
            </div>
            <h1 className="text-2xl font-semibold tracking-tight">数据面板</h1>
            <p className="text-sm text-muted-foreground">
              硬件资源、训练任务、后端连通状态实时聚合。
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge
              variant={wsLive ? "default" : "outline"}
              className="rounded-[2px] uppercase text-[10px] tracking-[0.1em]"
            >
              {wsLive ? "WS 连接" : "轮询模式"}
            </Badge>
            {snapshot && (
              <Badge
                variant="secondary"
                className="rounded-[2px] uppercase text-[10px] tracking-[0.1em]"
              >
                {new Date(snapshot.timestamp * 1000).toLocaleTimeString()}
              </Badge>
            )}
          </div>
        </header>

        {snapshot && <JobStatGrid stats={stats} />}

        {snapshot ? (
          <>
            <HostInfoCard snapshot={snapshot} />
            <CpuMemoryCard snapshot={snapshot} />
            {snapshot.battery && <BatteryCard battery={snapshot.battery} />}
            <GpuSection
              gpus={snapshot.gpus}
              hasNvidiaSmi={snapshot.has_nvidia_smi}
              system={snapshot.host.system}
            />
            <DiskSection disks={snapshot.disks} />
          </>
        ) : (
          <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
            <CardContent className="px-4 py-8 text-center text-sm text-muted-foreground">
              <Loader2 className="mx-auto mb-2 size-5 animate-spin" />
              正在采集系统快照…
            </CardContent>
          </Card>
        )}

        <RecentJobsCard jobs={allJobs} />
      </div>
    </div>
  )
}

// =================================================== 任务计数 ===============

function JobStatGrid({
  stats,
}: {
  stats: { running: number; queued: number; succeeded: number; failed: number }
}) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
      <StatCard
        icon={<Loader2 className="size-3.5" />}
        label="运行中"
        value={String(stats.running)}
        tone="primary"
      />
      <StatCard
        icon={<Pause className="size-3.5" />}
        label="排队中"
        value={String(stats.queued)}
      />
      <StatCard
        icon={<CheckCircle2 className="size-3.5" />}
        label="已完成"
        value={String(stats.succeeded)}
      />
      <StatCard
        icon={<CircleX className="size-3.5" />}
        label="失败 / 中断"
        value={String(stats.failed)}
        tone={stats.failed > 0 ? "warning" : "default"}
      />
    </div>
  )
}

function StatCard({
  icon,
  label,
  value,
  tone = "default",
}: {
  icon: React.ReactNode
  label: string
  value: string
  tone?: "default" | "primary" | "destructive" | "warning"
}) {
  const toneStyle = {
    default: "text-foreground",
    primary: "text-primary",
    destructive: "text-destructive",
    warning: "text-amber-700 dark:text-amber-400",
  }[tone]
  return (
    <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
      <CardContent className="px-4 py-3">
        <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
          {icon}
          {label}
        </div>
        <div className={cn("mt-1.5 text-2xl font-semibold tracking-tight tabular-nums", toneStyle)}>
          {value}
        </div>
      </CardContent>
    </Card>
  )
}

// =================================================== 主机信息 ===============

function HostInfoCard({ snapshot }: { snapshot: SystemSnapshot }) {
  const items: { label: string; value: string }[] = [
    { label: "主机名", value: snapshot.host.hostname || "—" },
    { label: "操作系统", value: `${snapshot.host.system} ${snapshot.host.release}` },
    { label: "Python 版本", value: snapshot.host.python },
    { label: "CPU 架构", value: snapshot.cpu.arch || "—" },
  ]
  return (
    <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Server className="size-4 text-muted-foreground" />
          主机信息
        </CardTitle>
        <CardDescription className="text-xs">
          {snapshot.has_psutil ? "psutil 可用" : "psutil 缺失（部分 CPU 指标降级）"}　·
          {snapshot.has_nvidia_smi ? "nvidia-smi 可用" : "未检测到 nvidia-smi"}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-2 text-xs">
          {items.map((it) => (
            <div key={it.label}>
              <dt className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
                {it.label}
              </dt>
              <dd className="mt-0.5 font-mono truncate" title={it.value}>
                {it.value}
              </dd>
            </div>
          ))}
        </dl>
      </CardContent>
    </Card>
  )
}

// =================================================== CPU / 内存 ============

function CpuMemoryCard({ snapshot }: { snapshot: SystemSnapshot }) {
  const cpu = snapshot.cpu
  const mem = snapshot.memory
  const memPercent = Math.max(0, Math.min(100, mem.percent))
  const cpuPercent =
    typeof cpu.usage_percent === "number"
      ? Math.max(0, Math.min(100, cpu.usage_percent))
      : null
  const swapPercent =
    mem.swap_total_bytes && typeof mem.swap_used_bytes === "number"
      ? Math.max(
          0,
          Math.min(100, (mem.swap_used_bytes / Math.max(mem.swap_total_bytes, 1)) * 100),
        )
      : null

  const cpuDescriptionParts: string[] = [
    `${cpu.cores_logical} 逻辑核`,
  ]
  if (cpu.cores_physical) cpuDescriptionParts.push(`${cpu.cores_physical} 物理核`)
  if (typeof cpu.frequency_mhz === "number") {
    cpuDescriptionParts.push(`${formatFrequency(cpu.frequency_mhz)}`)
  }
  if (typeof cpu.cpu_temperature_c === "number") {
    cpuDescriptionParts.push(`温度 ${cpu.cpu_temperature_c.toFixed(0)}°C`)
  }
  if (cpu.load_average) {
    cpuDescriptionParts.push(`负载 ${cpu.load_average.map((n) => n.toFixed(2)).join(" / ")}`)
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Cpu className="size-4 text-muted-foreground" />
            CPU
          </CardTitle>
          <CardDescription className="text-xs">
            {cpuDescriptionParts.join(" · ")}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <UsageBar
            label="总体利用率"
            percent={cpuPercent}
            valueText={typeof cpuPercent === "number" ? `${cpuPercent.toFixed(1)}%` : "—"}
          />
          {cpu.per_core_percent.length > 0 && (
            <details className="group" open={cpu.per_core_percent.length <= 8}>
              <summary className="flex items-center gap-2 cursor-pointer select-none list-none text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
                <span className="size-3 grid place-items-center transition-transform group-open:rotate-90">
                  ›
                </span>
                <span>每核利用率（{cpu.per_core_percent.length} 核）</span>
              </summary>
              <div className="mt-2 max-h-48 overflow-y-auto rounded-[4px] border border-border/40 bg-muted/20 p-2">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  {cpu.per_core_percent.map((p, i) => (
                    <CoreBar key={i} index={i} percent={p} />
                  ))}
                </div>
              </div>
            </details>
          )}
        </CardContent>
      </Card>

      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <MemoryStick className="size-4 text-muted-foreground" />
            内存
          </CardTitle>
          <CardDescription className="text-xs">
            {fmtBytes(mem.used_bytes)} / {fmtBytes(mem.total_bytes)} · 可用 {fmtBytes(mem.available_bytes)}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <UsageBar
            label="物理内存"
            percent={memPercent}
            valueText={`${memPercent.toFixed(1)}%`}
          />
          {typeof swapPercent === "number" && (
            <UsageBar
              label="交换分区"
              percent={swapPercent}
              valueText={`${fmtBytes(mem.swap_used_bytes ?? 0)} / ${fmtBytes(
                mem.swap_total_bytes ?? 0,
              )}`}
            />
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function UsageBar({
  label,
  percent,
  valueText,
}: {
  label: string
  percent: number | null
  valueText: string
}) {
  const tone = toneForPercent(percent)
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className={cn("font-mono tabular-nums", tone.text)}>{valueText}</span>
      </div>
      <Progress value={percent ?? 0}>
        <ProgressTrack>
          <ProgressIndicator className={tone.bar} />
        </ProgressTrack>
      </Progress>
    </div>
  )
}

function CoreBar({ index, percent }: { index: number; percent: number }) {
  const tone = toneForPercent(percent)
  return (
    <div className="flex items-center gap-2 text-[11px]">
      <span className="w-8 text-muted-foreground tabular-nums">#{index}</span>
      <div className="flex-1 h-1.5 rounded-[1px] bg-muted/40 overflow-hidden">
        <div
          className={cn("h-full transition-[width]", tone.bar)}
          style={{ width: `${Math.max(0, Math.min(100, percent))}%` }}
        />
      </div>
      <span className="w-10 text-right font-mono tabular-nums">{percent.toFixed(0)}%</span>
    </div>
  )
}

// =================================================== GPU =====================

function GpuSection({
  gpus,
  hasNvidiaSmi,
  system,
}: {
  gpus: SystemGpu[]
  hasNvidiaSmi: boolean
  system: string
}) {
  // 已经探测到设备时直接渲染，让多源 GPU 都有展示。
  if (gpus.length > 0) {
    return (
      <div className="space-y-3">
        {gpus.map((gpu) => (
          <GpuCard key={gpu.index} gpu={gpu} />
        ))}
      </div>
    )
  }

  const isMac = system === "Darwin"
  const description = isMac
    ? "Apple Silicon / Apple GPU 不支持详细计量，已切换到仅展示型号。本机未检测到独立 GPU。"
    : "未检测到 nvidia-smi。AMD / Apple 设备或 CPU-only 主机可忽略此项。"

  // 没有任何 GPU 数据 - 给个友好提示。
  if (!hasNvidiaSmi) {
    return (
      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Zap className="size-4 text-muted-foreground" />
            GPU
          </CardTitle>
          <CardDescription className="text-xs">{description}</CardDescription>
        </CardHeader>
      </Card>
    )
  }
  return (
    <Card className="rounded-[6px] border-amber-500/30 bg-amber-500/5 shadow-[var(--panel-shadow)]">
      <CardContent className="px-4 py-3 flex items-center gap-2 text-xs text-amber-700 dark:text-amber-400">
        <AlertTriangle className="size-4" />
        nvidia-smi 已安装但未返回任何设备信息。
      </CardContent>
    </Card>
  )
}

function GpuCard({ gpu }: { gpu: SystemGpu }) {
  const memPercent =
    typeof gpu.memory_used_bytes === "number" && gpu.memory_total_bytes
      ? Math.max(0, Math.min(100, (gpu.memory_used_bytes / gpu.memory_total_bytes) * 100))
      : null
  const utilTone = toneForPercent(gpu.utilization_percent ?? 0)
  const memTone = toneForPercent(memPercent)
  const vendor = vendorBadge(gpu)
  const isAppleSilicon = gpu.vendor === "apple"
  return (
    <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="text-base flex items-center gap-2">
              <Zap className="size-4 text-muted-foreground" />
              GPU #{gpu.index} · {gpu.name}
            </CardTitle>
            <CardDescription className="text-xs">
              {isAppleSilicon
                ? "Apple GPU 暂不开放计量接口，仅显示型号"
                : `驱动 ${gpu.driver ?? "—"}`}
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Badge className={cn("rounded-[2px] uppercase text-[10px] tracking-[0.1em]", vendor.className)}>
              {vendor.label}
            </Badge>
            {typeof gpu.temperature_c === "number" && (
              <Badge variant="outline" className="rounded-[2px] gap-1">
                <Thermometer className="size-3" /> {gpu.temperature_c.toFixed(0)}°C
              </Badge>
            )}
            {typeof gpu.fan_percent === "number" && (
              <Badge variant="outline" className="rounded-[2px]">
                风扇 {gpu.fan_percent.toFixed(0)}%
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <UsageBar
          label="计算利用率"
          percent={gpu.utilization_percent}
          valueText={
            typeof gpu.utilization_percent === "number"
              ? `${gpu.utilization_percent.toFixed(0)}%`
              : "—"
          }
        />
        <UsageBar
          label="显存"
          percent={memPercent}
          valueText={
            typeof gpu.memory_used_bytes === "number" && gpu.memory_total_bytes
              ? `${fmtBytes(gpu.memory_used_bytes)} / ${fmtBytes(gpu.memory_total_bytes)}`
              : gpu.memory_total_bytes
                ? `— / ${fmtBytes(gpu.memory_total_bytes)}`
                : "—"
          }
        />
        <Separator />
        <dl className="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-2 text-xs">
          <Metric
            label="功率"
            value={typeof gpu.power_w === "number" ? `${gpu.power_w.toFixed(0)} W` : "—"}
          />
          <Metric
            label="功率上限"
            value={
              typeof gpu.power_limit_w === "number" ? `${gpu.power_limit_w.toFixed(0)} W` : "—"
            }
          />
          <Metric
            label="可用显存"
            value={typeof gpu.memory_free_bytes === "number" ? fmtBytes(gpu.memory_free_bytes) : "—"}
          />
          <Metric
            label="温度"
            value={
              typeof gpu.temperature_c === "number" ? `${gpu.temperature_c.toFixed(0)} °C` : "—"
            }
          />
        </dl>
        <span className="sr-only">
          {utilTone.text}
          {memTone.text}
        </span>
      </CardContent>
    </Card>
  )
}

function vendorBadge(gpu: SystemGpu): { label: string; className: string } {
  switch (gpu.vendor) {
    case "nvidia":
      return {
        label: "NVIDIA",
        className: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border border-emerald-500/40",
      }
    case "amd":
      return {
        label: "AMD",
        className: "bg-red-500/15 text-red-700 dark:text-red-300 border border-red-500/40",
      }
    case "intel":
      return {
        label: "Intel",
        className: "bg-blue-500/15 text-blue-700 dark:text-blue-300 border border-blue-500/40",
      }
    case "apple":
      return {
        label: "Apple",
        className: "bg-slate-500/15 text-slate-700 dark:text-slate-200 border border-slate-500/40",
      }
    default:
      return {
        label: gpu.vendor ? gpu.vendor.toUpperCase() : "GPU",
        className: "bg-muted text-foreground border border-border/70",
      }
  }
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">{label}</dt>
      <dd className="mt-0.5 font-mono tabular-nums">{value}</dd>
    </div>
  )
}

// =================================================== 电池 ===================

function BatteryCard({ battery }: { battery: SystemBattery }) {
  const percent = Math.max(0, Math.min(100, battery.percent))
  const Icon = battery.plugged ? BatteryCharging : BatteryFull
  const tone = battery.plugged
    ? "text-emerald-700 dark:text-emerald-400"
    : percent <= 15
      ? "text-destructive"
      : percent <= 30
        ? "text-amber-700 dark:text-amber-400"
        : "text-foreground"
  const description = (() => {
    if (battery.plugged) return "电源已连接"
    if (typeof battery.secs_left === "number") return `预计剩余 ${formatSecs(battery.secs_left)}`
    return "未连接电源"
  })()
  const barTone = toneForPercent(percent)
  return (
    <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="text-base flex items-center gap-2">
              <Icon className={cn("size-4", tone)} />
              电池
            </CardTitle>
            <CardDescription className="text-xs">{description}</CardDescription>
          </div>
          <div className={cn("text-2xl font-semibold tabular-nums", tone)}>
            {percent.toFixed(0)}%
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-1.5 rounded-[1px] bg-muted/40 overflow-hidden">
          <div
            className={cn("h-full transition-[width]", barTone.bar)}
            style={{ width: `${percent}%` }}
          />
        </div>
      </CardContent>
    </Card>
  )
}

// =================================================== 磁盘 ===================

function DiskSection({ disks }: { disks: SystemDisk[] }) {
  return (
    <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <HardDrive className="size-4 text-muted-foreground" />
          磁盘
        </CardTitle>
        <CardDescription className="text-xs">工作目录与用户目录所在卷的实时容量。</CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>用途</TableHead>
              <TableHead>路径</TableHead>
              <TableHead className="text-right">已用 / 总量</TableHead>
              <TableHead className="text-right">可用</TableHead>
              <TableHead className="w-[180px]">使用率</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {disks.map((d) => {
              const tone = toneForPercent(d.percent)
              return (
                <TableRow key={d.path}>
                  <TableCell>{d.label}</TableCell>
                  <TableCell className="font-mono text-xs truncate max-w-xs" title={d.path}>
                    {d.path}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">
                    {fmtBytes(d.used_bytes)} / {fmtBytes(d.total_bytes)}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">
                    {fmtBytes(d.free_bytes)}
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-1.5 rounded-[1px] bg-muted/40 overflow-hidden">
                        <div
                          className={cn("h-full transition-[width]", tone.bar)}
                          style={{ width: `${Math.max(0, Math.min(100, d.percent))}%` }}
                        />
                      </div>
                      <span className={cn("text-[11px] font-mono tabular-nums", tone.text)}>
                        {d.percent.toFixed(0)}%
                      </span>
                    </div>
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

// =================================================== 最近任务 ===============

function RecentJobsCard({ jobs }: { jobs: JobSummary[] }) {
  return (
    <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Activity className="size-4 text-muted-foreground" />
          最近任务
        </CardTitle>
        <CardDescription>当前工作区中最新的训练记录。</CardDescription>
      </CardHeader>
      <CardContent>
        {jobs.length === 0 ? (
          <EmptyState />
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[110px]">状态</TableHead>
                <TableHead>任务 ID</TableHead>
                <TableHead>工作区</TableHead>
                <TableHead className="text-right">创建时间</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs.slice(-8).reverse().map((j) => (
                <TableRow key={j.id}>
                  <TableCell>
                    <StateBadge state={j.state} />
                  </TableCell>
                  <TableCell className="font-mono text-xs">{j.id.slice(-8)}</TableCell>
                  <TableCell className="font-mono text-xs truncate max-w-md" title={j.workspace}>
                    {j.workspace}
                  </TableCell>
                  <TableCell className="text-right text-xs text-muted-foreground tabular-nums">
                    {new Date(j.created_at).toLocaleString()}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}

function EmptyState() {
  return (
    <div className="rounded-[4px] border border-dashed border-border/70 bg-muted/30 px-4 py-8 text-center">
      <Pause className="size-5 mx-auto text-muted-foreground/60" />
      <div className="mt-2 text-sm font-medium">还没有训练任务</div>
      <div className="text-xs text-muted-foreground">
        在 <code className="text-foreground">recipes/</code> 选一个配方点击训练，或运行{" "}
        <code className="text-foreground">lorahub train recipe.yaml</code>。
      </div>
    </div>
  )
}

const STATE_LABELS: Record<string, string> = {
  running: "运行中",
  succeeded: "已完成",
  failed: "失败",
  canceled: "已取消",
  canceling: "取消中",
  queued: "排队中",
  interrupted: "已中断",
}

export function StateBadge({ state }: { state: string }) {
  const variant = {
    running: "default",
    succeeded: "secondary",
    failed: "destructive",
    canceled: "outline",
    canceling: "outline",
    queued: "outline",
    interrupted: "destructive",
  }[state] as "default" | "secondary" | "destructive" | "outline" | undefined

  return (
    <Badge
      variant={variant ?? "outline"}
      className="rounded-[2px] uppercase text-[10px] tracking-[0.1em]"
    >
      {STATE_LABELS[state] ?? state}
    </Badge>
  )
}

// =================================================== utils =================

function fmtBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0 B"
  const units = ["B", "KB", "MB", "GB", "TB", "PB"]
  let i = 0
  let v = n
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i += 1
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : v >= 10 ? 1 : 2)} ${units[i]}`
}

function formatFrequency(mhz: number): string {
  if (!Number.isFinite(mhz) || mhz <= 0) return "—"
  if (mhz >= 1000) return `${(mhz / 1000).toFixed(2)} GHz`
  return `${mhz.toFixed(0)} MHz`
}

function formatSecs(secs: number): string {
  if (!Number.isFinite(secs) || secs <= 0) return "—"
  const hours = Math.floor(secs / 3600)
  const minutes = Math.floor((secs % 3600) / 60)
  if (hours > 0) return `${hours} 小时 ${minutes} 分钟`
  return `${minutes} 分钟`
}

function toneForPercent(percent: number | null): { text: string; bar: string } {
  if (percent === null) return { text: "text-muted-foreground", bar: "bg-muted-foreground/40" }
  if (percent >= 90) return { text: "text-destructive", bar: "bg-destructive" }
  if (percent >= 70) {
    return {
      text: "text-amber-700 dark:text-amber-400",
      bar: "bg-amber-500",
    }
  }
  if (percent >= 40) return { text: "text-primary", bar: "bg-primary" }
  return {
    text: "text-emerald-700 dark:text-emerald-400",
    bar: "bg-emerald-500",
  }
}
