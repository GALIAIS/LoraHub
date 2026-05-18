/**
 * 实时数据面板：硬件状态 + 任务概览。
 *
 * 通过 SSE /api/system/sse 每秒接收一次系统快照（旧 WS 端点保留作 fallback），
 * 实时通道不可用时降级为 5 秒一次的 REST 轮询。任务统计仍走 /api/jobs（3 秒轮询）。
 */
import { useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import { Loader2 } from "lucide-react"
import {
  api,
  useSystemStream,
  type SystemSnapshot,
} from "@/lib/api"
import { useJobsList } from "@/lib/queries/jobs"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { JobStatGrid } from "./job-stats"
import { HostInfoCard } from "./host-info"
import { CpuMemoryCard } from "./cpu-memory"
import { GpuSection } from "./gpu"
import { BatteryCard } from "./battery"
import { DiskSection, DiskIoCard } from "./disk"
import { NetworkInterfacesCard, NetworkSummaryCard } from "./network"
import { TopProcessesCard, GpuProcessesCard } from "./processes"
import { RecentJobsCard } from "./recent-jobs"

// Re-export StateBadge so existing imports `from "../dashboard"` keep working
// (sweeps / analysis / jobs pages still use it).
export { StateBadge } from "./recent-jobs"

const POLL_INTERVAL_SYSTEM_MS = 5_000

export function DashboardPage() {
  const stream = useSystemStream(true)

  // 实时通道不通时降级为 5s REST 轮询，保证看板永远有数据。
  const polled = useQuery({
    queryKey: ["system-stats"],
    queryFn: api.getSystemStats,
    refetchInterval: stream.status === "open" ? false : POLL_INTERVAL_SYSTEM_MS,
  })

  const snapshot: SystemSnapshot | null = stream.snapshot ?? polled.data ?? null

  const jobs = useJobsList()
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

  const liveStream = stream.status === "open"

  return (
    <div className="h-full overflow-y-auto">
      <div className="px-8 py-7 space-y-6 w-full">
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
              variant={liveStream ? "default" : "outline"}
              className="rounded-[2px] uppercase text-[10px] tracking-[0.1em]"
            >
              {liveStream ? "实时" : "轮询模式"}
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
            {snapshot.processes !== undefined && (
              <TopProcessesCard processes={snapshot.processes ?? []} />
            )}
            {snapshot.battery && <BatteryCard battery={snapshot.battery} />}
            <GpuSection
              gpus={snapshot.gpus}
              hasNvidiaSmi={snapshot.has_nvidia_smi}
              system={snapshot.host.system}
            />
            {snapshot.gpu_processes !== undefined && (
              <GpuProcessesCard processes={snapshot.gpu_processes ?? []} />
            )}
            <DiskSection disks={snapshot.disks} />
            {snapshot.disk_io !== undefined && (
              <DiskIoCard io={snapshot.disk_io ?? null} />
            )}
            {snapshot.network?.interfaces !== undefined && (
              <NetworkInterfacesCard interfaces={snapshot.network?.interfaces ?? []} />
            )}
            {(snapshot.network?.tcp_connections !== undefined ||
              snapshot.network?.public_ip !== undefined) && (
              <NetworkSummaryCard
                tcp={snapshot.network?.tcp_connections ?? null}
                publicIp={snapshot.network?.public_ip ?? null}
              />
            )}
          </>
        ) : (
          <Card>
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
