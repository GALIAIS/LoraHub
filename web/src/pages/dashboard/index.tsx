/**
 * 实时数据面板：硬件状态 + 任务概览。
 *
 * 通过 SSE /api/system/sse 每秒接收一次系统快照（旧 WS 端点保留作 fallback），
 * 实时通道不可用时降级为 5 秒一次的 REST 轮询。任务统计仍走 /api/jobs（3 秒轮询）。
 */
import { useMemo, useState } from "react"
import { Loader2 } from "lucide-react"
import type { SystemSnapshot } from "@/lib/api"
import { useJobsList } from "@/lib/queries/jobs"
import { useSystemTelemetry } from "@/lib/system-telemetry"
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

export function DashboardPage() {
  const telemetry = useSystemTelemetry()
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false)
  const snapshot: SystemSnapshot | null = telemetry.snapshot

  const jobs = useJobsList()
  const allJobs = jobs.data?.jobs ?? []

  const stats = useMemo(() => {
    const running = allJobs.filter(
      (j) => j.state === "running" || j.state === "preparing",
    ).length
    const queued = allJobs.filter((j) => j.state === "queued").length
    const succeeded = allJobs.filter((j) => j.state === "succeeded").length
    const failed = allJobs.filter(
      (j) =>
        j.state === "failed" || j.state === "interrupted" || j.state === "canceled",
    ).length
    return { running, queued, succeeded, failed }
  }, [allJobs])

  const liveStream = telemetry.live

  return (
    <div className="h-full overflow-y-auto">
      <div className="px-4 py-4 md:px-6 md:py-5 space-y-4 w-full">
        <div className="flex items-center justify-between gap-3">
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
        </div>

        {snapshot && <JobStatGrid stats={stats} />}

        {snapshot ? (
          <>
            <HostInfoCard snapshot={snapshot} />

            <div className="grid items-stretch gap-4 xl:grid-cols-[minmax(360px,0.95fr)_minmax(0,1fr)]">
              <GpuSection
                gpus={snapshot.gpus}
                hasNvidiaSmi={snapshot.has_nvidia_smi}
                system={snapshot.host.system}
              />
              <CpuMemoryCard snapshot={snapshot} variant="cpu" />
            </div>

            <RecentJobsCard jobs={allJobs} />

            <details
              className="rounded-[6px] border border-border/60 bg-background/60"
              open={diagnosticsOpen}
              onToggle={(event) => setDiagnosticsOpen(event.currentTarget.open)}
            >
              <summary className="cursor-pointer px-4 py-3 text-sm font-medium">
                系统诊断详情
              </summary>
              {diagnosticsOpen && (
                <div className="space-y-4 border-t border-border/60 p-4">
                  <CpuMemoryCard snapshot={snapshot} variant="memory" />
                  <DiskSection disks={snapshot.disks} />
                  {snapshot.processes !== undefined && (
                    <TopProcessesCard processes={snapshot.processes ?? []} />
                  )}
                  {snapshot.battery && <BatteryCard battery={snapshot.battery} />}
                  {snapshot.gpu_processes !== undefined && (
                    <GpuProcessesCard processes={snapshot.gpu_processes ?? []} />
                  )}
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
                </div>
              )}
            </details>
          </>
        ) : (
          <Card>
            <CardContent className="px-4 py-8 text-center text-sm text-muted-foreground">
              <Loader2 className="mx-auto mb-2 size-5 animate-spin" />
              正在采集系统快照…
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
