import { memo } from "react"
import { ListChecks, Zap } from "lucide-react"
import type { GpuProcessInfo, ProcessInfo } from "@/lib/api"
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
import { fmtBytes } from "./_shared"

export const TopProcessesCard = memo(function TopProcessesCard({
  processes,
}: {
  processes: ProcessInfo[]
}) {
  // Backend already sorts by cpu_percent desc; we still slice defensively
  // so a runaway list can't blow the layout.
  const rows = processes.slice(0, 10)
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <ListChecks className="size-4 text-muted-foreground" />
          Top 进程
        </CardTitle>
        <CardDescription className="text-xs">
          按 CPU 占用排序的前 {rows.length || 0} 个进程。
        </CardDescription>
      </CardHeader>
      <CardContent>
        {rows.length === 0 ? (
          <div className="rounded-[4px] border border-dashed border-border/70 bg-muted/20 px-4 py-6 text-center text-xs text-muted-foreground">
            无可显示的进程
          </div>
        ) : (
          <div className="max-h-[14rem] overflow-y-auto">
          <Table>
            <TableHeader className="sticky top-0 bg-card z-10">
              <TableRow>
                <TableHead className="w-[80px]">PID</TableHead>
                <TableHead>名称</TableHead>
                <TableHead className="text-right whitespace-nowrap min-w-[8ch]">CPU%</TableHead>
                <TableHead className="text-right whitespace-nowrap min-w-[8ch]">内存%</TableHead>
                <TableHead className="text-right whitespace-nowrap min-w-[10ch]">RSS</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((p) => (
                <TableRow key={p.pid}>
                  <TableCell className="font-mono text-xs tabular-nums">{p.pid}</TableCell>
                  <TableCell className="font-mono text-xs truncate max-w-md" title={p.name}>
                    {p.name || "—"}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums text-xs">
                    {p.cpu_percent.toFixed(1)}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums text-xs">
                    {p.memory_percent.toFixed(1)}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums text-xs whitespace-nowrap">
                    {fmtBytes(p.memory_rss_bytes)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          </div>
        )}
      </CardContent>
    </Card>
  )
})

export const GpuProcessesCard = memo(function GpuProcessesCard({
  processes,
}: {
  processes: GpuProcessInfo[]
}) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Zap className="size-4 text-muted-foreground" />
          GPU 进程
        </CardTitle>
        <CardDescription className="text-xs">
          nvidia-smi 当前观察到的进程列表。
        </CardDescription>
      </CardHeader>
      <CardContent>
        {processes.length === 0 ? (
          <div className="rounded-[4px] border border-dashed border-border/70 bg-muted/20 px-4 py-6 text-center text-xs text-muted-foreground">
            当前无 GPU 计算进程
          </div>
        ) : (
          <div className="max-h-[12rem] overflow-y-auto">
          <Table>
            <TableHeader className="sticky top-0 bg-card z-10">
              <TableRow>
                <TableHead className="w-[60px]">GPU</TableHead>
                <TableHead className="w-[80px]">PID</TableHead>
                <TableHead>进程</TableHead>
                <TableHead className="text-right whitespace-nowrap min-w-[10ch]">显存 MiB</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {processes.map((p) => (
                <TableRow key={`${p.gpu_index}-${p.pid}`}>
                  <TableCell className="font-mono text-xs tabular-nums">
                    #{p.gpu_index}
                    <Badge
                      variant="outline"
                      className="ml-2 rounded-[2px] uppercase text-[9px] tracking-[0.1em]"
                    >
                      {p.type}
                    </Badge>
                  </TableCell>
                  <TableCell className="font-mono text-xs tabular-nums">{p.pid}</TableCell>
                  <TableCell className="font-mono text-xs truncate max-w-md" title={p.process_name}>
                    {p.process_name || "—"}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums text-xs">
                    {p.used_memory_mib.toFixed(0)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          </div>
        )}
      </CardContent>
    </Card>
  )
})
