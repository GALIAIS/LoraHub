import { memo } from "react"
import { Activity, Pause } from "lucide-react"
import type { JobSummary } from "@/lib/api"
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

export const RecentJobsCard = memo(function RecentJobsCard({ jobs }: { jobs: JobSummary[] }) {
  return (
    <Card>
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
                <TableHead className="w-[120px]">任务 ID</TableHead>
                <TableHead>工作区</TableHead>
                <TableHead className="text-right whitespace-nowrap min-w-[160px]">创建时间</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs.slice(-8).reverse().map((j) => (
                <TableRow key={j.id}>
                  <TableCell>
                    <StateBadge state={j.state} />
                  </TableCell>
                  <TableCell className="font-mono text-xs whitespace-nowrap">{j.id.slice(-8)}</TableCell>
                  <TableCell className="font-mono text-xs truncate max-w-md" title={j.workspace}>
                    {j.workspace}
                  </TableCell>
                  <TableCell className="text-right text-xs text-muted-foreground tabular-nums whitespace-nowrap">
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
})

function EmptyState() {
  return (
    <div className="rounded-[4px] border border-dashed border-border/70 bg-muted/30 px-4 py-8 text-center">
      <Pause className="size-5 mx-auto text-muted-foreground/60" />
      <div className="mt-2 text-sm font-medium">还没有训练任务</div>
      <div className="text-xs text-muted-foreground">
        在 <code className="text-foreground">configs/</code> 选一个配置点击训练，或运行{" "}
        <code className="text-foreground">lorahub train config.yaml</code>。
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
