import { useQuery } from "@tanstack/react-query"
import { api, type JobFile } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Download, ExternalLink } from "lucide-react"
import { fmtBytes, fmtUnixSeconds, TERMINAL_STATES } from "../utils"

function FileTable({
  jobId,
  files,
  actionLabel,
  actionIcon,
  emptyHint,
}: {
  jobId: string
  files: JobFile[]
  actionLabel: string
  actionIcon: "download" | "open"
  emptyHint: string
}) {
  if (files.length === 0) {
    return (
      <div className="px-4 py-6 text-xs text-muted-foreground text-center">
        {emptyHint}
      </div>
    )
  }
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>路径</TableHead>
          <TableHead className="w-24">大小</TableHead>
          <TableHead className="w-44">修改时间</TableHead>
          <TableHead className="w-24 text-right">操作</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {files.map((f) => (
          <TableRow key={f.path}>
            <TableCell
              className="font-mono text-[12px] max-w-[420px] truncate"
              title={f.path}
            >
              {f.path}
            </TableCell>
            <TableCell className="tabular-nums text-[12px]">
              {fmtBytes(f.size_bytes)}
            </TableCell>
            <TableCell className="text-[12px] text-muted-foreground">
              {fmtUnixSeconds(f.modified_at)}
            </TableCell>
            <TableCell className="text-right">
              <Button
                size="sm"
                variant="outline"
                onClick={() => window.open(api.jobFileUrl(jobId, f.path))}
                className="h-7 text-[11px]"
              >
                {actionIcon === "download" ? (
                  <Download className="size-3" />
                ) : (
                  <ExternalLink className="size-3" />
                )}
                {actionLabel}
              </Button>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

function SampleGrid({ jobId, files }: { jobId: string; files: JobFile[] }) {
  if (files.length === 0) {
    return (
      <div className="px-4 py-6 text-xs text-muted-foreground text-center">
        还没有样本图。
      </div>
    )
  }
  return (
    <div className="p-4 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
      {files.map((f) => {
        const url = api.jobFileUrl(jobId, f.path)
        const filename = f.path.split(/[\\/]/).pop() ?? f.path
        return (
          <a
            key={f.path}
            href={url}
            target="_blank"
            rel="noreferrer"
            className="group block rounded-[4px] border border-border/60 overflow-hidden bg-card/50 hover:border-primary/50 transition-colors"
            title={f.path}
          >
            <div className="aspect-square bg-muted/40 grid place-items-center overflow-hidden">
              <img
                src={url}
                loading="lazy"
                alt={filename}
                className="w-full h-full object-cover group-hover:scale-[1.02] transition-transform"
              />
            </div>
            <div className="px-2 py-1.5 text-[11px]">
              <div className="truncate font-mono">{filename}</div>
              <div className="text-muted-foreground/70 tabular-nums">
                {fmtBytes(f.size_bytes)} · {fmtUnixSeconds(f.modified_at)}
              </div>
            </div>
          </a>
        )
      })}
    </div>
  )
}

export function FilesTab({
  jobId,
  jobState,
}: {
  jobId: string
  jobState: string | undefined
}) {
  const isTerminal = jobState ? TERMINAL_STATES.has(jobState) : false
  const files = useQuery({
    queryKey: ["job-files", jobId],
    queryFn: () => api.getJobFiles(jobId),
    refetchInterval: isTerminal ? false : 4000,
  })

  if (files.isLoading) {
    return (
      <div className="px-4 py-10 text-sm text-muted-foreground text-center">
        正在加载产物列表…
      </div>
    )
  }
  if (files.isError || !files.data) {
    return (
      <div className="px-4 py-10 text-sm text-destructive text-center">
        无法加载产物列表。
      </div>
    )
  }

  const data = files.data

  return (
    <div className="space-y-4">
      <div className="text-[11px] text-muted-foreground font-mono truncate">
        工作区 {data.workspace}
      </div>

      <Card className="rounded-[6px] border-border/60 shadow-[var(--panel-shadow)] overflow-hidden">
        <CardHeader className="py-3 px-4 border-b border-border/60 bg-muted/40 flex-row items-center justify-between">
          <CardTitle className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
            检查点
          </CardTitle>
          <span className="text-[10px] text-muted-foreground/70">
            {data.checkpoints.length} 个
          </span>
        </CardHeader>
        <CardContent className="p-0">
          <div className="max-h-[360px] overflow-auto">
            <FileTable
              jobId={jobId}
              files={data.checkpoints}
              actionLabel="下载"
              actionIcon="download"
              emptyHint="还没有保存的检查点。"
            />
          </div>
        </CardContent>
      </Card>

      <Card className="rounded-[6px] border-border/60 shadow-[var(--panel-shadow)] overflow-hidden">
        <CardHeader className="py-3 px-4 border-b border-border/60 bg-muted/40 flex-row items-center justify-between">
          <CardTitle className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
            样本预览
          </CardTitle>
          <span className="text-[10px] text-muted-foreground/70">
            {data.samples.length} 张
          </span>
        </CardHeader>
        <CardContent className="p-0">
          <div className="max-h-[480px] overflow-auto">
            <SampleGrid jobId={jobId} files={data.samples} />
          </div>
        </CardContent>
      </Card>

      <Card className="rounded-[6px] border-border/60 shadow-[var(--panel-shadow)] overflow-hidden">
        <CardHeader className="py-3 px-4 border-b border-border/60 bg-muted/40 flex-row items-center justify-between">
          <CardTitle className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
            日志
          </CardTitle>
          <span className="text-[10px] text-muted-foreground/70">
            {data.logs.length} 个
          </span>
        </CardHeader>
        <CardContent className="p-0">
          <div className="max-h-[300px] overflow-auto">
            <FileTable
              jobId={jobId}
              files={data.logs}
              actionLabel="打开"
              actionIcon="open"
              emptyHint="没有日志文件。"
            />
          </div>
        </CardContent>
      </Card>

      <Card className="rounded-[6px] border-border/60 shadow-[var(--panel-shadow)] overflow-hidden">
        <CardHeader className="py-3 px-4 border-b border-border/60 bg-muted/40 flex-row items-center justify-between">
          <CardTitle className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
            其他
          </CardTitle>
          <span className="text-[10px] text-muted-foreground/70">
            {data.other.length} 个
          </span>
        </CardHeader>
        <CardContent className="p-0">
          <div className="max-h-[300px] overflow-auto">
            <FileTable
              jobId={jobId}
              files={data.other}
              actionLabel="下载"
              actionIcon="download"
              emptyHint="没有其他文件。"
            />
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
