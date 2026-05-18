import { memo } from "react"
import { Server } from "lucide-react"
import type { SystemSnapshot } from "@/lib/api"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

export const HostInfoCard = memo(function HostInfoCard({
  snapshot,
}: {
  snapshot: SystemSnapshot
}) {
  const items: { label: string; value: string }[] = [
    { label: "主机名", value: snapshot.host.hostname || "—" },
    { label: "操作系统", value: `${snapshot.host.system} ${snapshot.host.release}` },
    { label: "Python 版本", value: snapshot.host.python },
    { label: "CPU 架构", value: snapshot.cpu.arch || "—" },
  ]
  return (
    <Card>
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
})
