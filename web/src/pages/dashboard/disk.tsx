import { memo } from "react"
import { HardDrive } from "lucide-react"
import type { DiskIoStats, SystemDisk } from "@/lib/api"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { cn } from "@/lib/utils"
import { fmtBytes, formatRate, toneForPercent } from "./_shared"

export const DiskSection = memo(function DiskSection({ disks }: { disks: SystemDisk[] }) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <HardDrive className="size-4 text-muted-foreground" />
          磁盘
        </CardTitle>
        <CardDescription className="text-xs">工作目录与用户目录所在卷的实时容量。</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="max-h-[16rem] overflow-y-auto">
        <Table>
          <TableHeader className="sticky top-0 bg-card z-10">
            <TableRow>
              <TableHead>用途</TableHead>
              <TableHead>路径</TableHead>
              <TableHead className="text-right whitespace-nowrap min-w-[14ch]">已用 / 总量</TableHead>
              <TableHead className="text-right whitespace-nowrap min-w-[8ch]">可用</TableHead>
              <TableHead className="w-[200px]">使用率</TableHead>
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
                  <TableCell className="text-right font-mono tabular-nums whitespace-nowrap">
                    {fmtBytes(d.used_bytes)} / {fmtBytes(d.total_bytes)}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums whitespace-nowrap">
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
                      <span className={cn("text-[11px] font-mono tabular-nums shrink-0 text-right min-w-[4ch]", tone.text)}>
                        {d.percent.toFixed(0)}%
                      </span>
                    </div>
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
        </div>
      </CardContent>
    </Card>
  )
})

export const DiskIoCard = memo(function DiskIoCard({ io }: { io: DiskIoStats | null }) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="text-base flex items-center gap-2">
              <HardDrive className="size-4 text-muted-foreground" />
              磁盘 IO
            </CardTitle>
            <CardDescription className="text-xs">
              {io
                ? "聚合速率 + 各设备明细。"
                : "未读取到 IO 计数器（容器可能屏蔽）。"}
            </CardDescription>
          </div>
          {io && (
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="rounded-[2px] gap-1 font-mono">
                ↓ {formatRate(io.read_bytes_per_sec)}
              </Badge>
              <Badge variant="outline" className="rounded-[2px] gap-1 font-mono">
                ↑ {formatRate(io.write_bytes_per_sec)}
              </Badge>
            </div>
          )}
        </div>
      </CardHeader>
      {io && (
        <CardContent>
          {io.per_device.length === 0 ? (
            <div className="rounded-[4px] border border-dashed border-border/70 bg-muted/20 px-4 py-6 text-center text-xs text-muted-foreground">
              当前未观察到设备级 IO
            </div>
          ) : (
            <div className="max-h-[14rem] overflow-y-auto">
            <Table>
              <TableHeader className="sticky top-0 bg-card z-10">
                <TableRow>
                  <TableHead>设备</TableHead>
                  <TableHead className="text-right whitespace-nowrap min-w-[10ch]">读</TableHead>
                  <TableHead className="text-right whitespace-nowrap min-w-[10ch]">写</TableHead>
                  <TableHead className="text-right whitespace-nowrap min-w-[10ch]">读 ops</TableHead>
                  <TableHead className="text-right whitespace-nowrap min-w-[10ch]">写 ops</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {io.per_device.map((d) => (
                  <TableRow key={d.device}>
                    <TableCell className="font-mono text-xs">{d.device}</TableCell>
                    <TableCell className="text-right font-mono tabular-nums text-xs whitespace-nowrap">
                      {formatRate(d.read_bytes_per_sec)}
                    </TableCell>
                    <TableCell className="text-right font-mono tabular-nums text-xs whitespace-nowrap">
                      {formatRate(d.write_bytes_per_sec)}
                    </TableCell>
                    <TableCell className="text-right font-mono tabular-nums text-xs">
                      {d.read_ops_per_sec.toFixed(1)}
                    </TableCell>
                    <TableCell className="text-right font-mono tabular-nums text-xs">
                      {d.write_ops_per_sec.toFixed(1)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            </div>
          )}
        </CardContent>
      )}
    </Card>
  )
})
