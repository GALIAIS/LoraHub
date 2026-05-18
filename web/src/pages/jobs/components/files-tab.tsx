import { useQuery } from "@tanstack/react-query"
import { api, type JobFile } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { PathDisplay } from "@/components/path-display"
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
  // Plain native table — the workbench-wide `<Table>` component carries
  // its own border + box-shadow (shiro-table-shell) which double-frames
  // a card it's nested inside. Inside files-tab the wrapper card is the
  // chrome owner; a borderless native table tucks neatly under it.
  return (
    <table className="w-full text-[12px] border-collapse">
      <thead className="sticky top-0 z-10 bg-muted/40 backdrop-blur-sm">
        <tr className="text-left text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
          <th className="px-3 py-2 font-medium">路径</th>
          <th className="px-3 py-2 font-medium w-20">大小</th>
          <th className="px-3 py-2 font-medium w-40">修改时间</th>
          <th className="px-3 py-2 font-medium w-20 text-right">操作</th>
        </tr>
      </thead>
      <tbody>
        {files.map((f) => (
          <tr
            key={f.path}
            className="border-t border-border/40 hover:bg-muted/30"
          >
            <td
              className="px-3 py-1.5 font-mono max-w-[420px] truncate"
              title={f.path}
            >
              {f.path}
            </td>
            <td className="px-3 py-1.5 tabular-nums">
              {fmtBytes(f.size_bytes)}
            </td>
            <td className="px-3 py-1.5 text-muted-foreground">
              {fmtUnixSeconds(f.modified_at)}
            </td>
            <td className="px-3 py-1.5 text-right">
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
            </td>
          </tr>
        ))}
      </tbody>
    </table>
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
      <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
        <span className="shrink-0">工作区</span>
        <PathDisplay path={data.workspace} tailSegments={3} block className="min-w-0" />
      </div>

      <Card className="overflow-hidden">
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

      <Card className="overflow-hidden">
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

      <Card className="overflow-hidden">
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

      <Card className="overflow-hidden">
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
