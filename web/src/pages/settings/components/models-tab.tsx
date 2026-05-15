import { useEffect, useMemo, useState } from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Download, ExternalLink, Loader2, Rows3, ServerCog } from "lucide-react"
import { api } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Progress,
  ProgressIndicator,
  ProgressLabel,
  ProgressTrack,
} from "@/components/ui/progress"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

type Source = "huggingface" | "modelscope"

const SOURCE_LABEL: Record<Source, string> = {
  huggingface: "HuggingFace",
  modelscope: "ModelScope",
}

function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0 B"
  const units = ["B", "KB", "MB", "GB", "TB"]
  let i = 0
  let v = n
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i += 1
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : v >= 10 ? 1 : 2)} ${units[i]}`
}

export function ModelsTab() {
  const settings = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  })

  const preferModelscope = settings.data?.settings.modelscope_enabled ?? false
  const [source, setSource] = useState<Source>(
    preferModelscope ? "modelscope" : "huggingface",
  )
  const [repoId, setRepoId] = useState("")
  const [revision, setRevision] = useState("")
  const [targetDir, setTargetDir] = useState("")
  const [threads, setThreads] = useState(4)
  const [sessionId, setSessionId] = useState<string | null>(null)

  useEffect(() => {
    setSource((current) => (current ? current : preferModelscope ? "modelscope" : "huggingface"))
  }, [preferModelscope])

  const startDownload = useMutation({
    mutationFn: () =>
      api.downloadModel({
        source,
        repo_id: repoId.trim(),
        revision: revision.trim() || (source === "modelscope" ? "master" : "main"),
        target_dir: targetDir.trim() || undefined,
        threads,
      }),
    onSuccess: (session) => setSessionId(session.session_id),
  })

  const session = useQuery({
    queryKey: ["model-download", sessionId],
    queryFn: () => api.getModelDownload(sessionId!),
    enabled: !!sessionId,
    refetchInterval: (query) =>
      query.state.data?.status === "running" || !query.state.data ? 800 : false,
  })

  const current = session.data ?? startDownload.data ?? null
  const error = (startDownload.error as Error | undefined) ?? (session.error as Error | undefined)
  const ready = repoId.includes("/") && repoId.trim().length > 2
  const running = current?.status === "running" || startDownload.isPending
  const latest = current?.events.at(-1)
  const percent = Math.max(0, Math.min(100, current?.percent ?? 0))

  const result = current?.result
  const summary = useMemo(() => {
    if (!current) return "尚未开始下载"
    if (current.status === "running") return latest?.message ?? "下载进行中"
    if (current.status === "failed") return current.error ?? "下载失败"
    return "下载完成"
  }, [current, latest])

  return (
    <div className="space-y-5 w-full">
      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Download className="size-4 text-muted-foreground" />
            模型下载
          </CardTitle>
          <CardDescription>
            支持 HuggingFace 与 ModelScope。下载在后端线程中运行，本页会轮询显示进度；
            多线程数量会传给下载器，HuggingFace 使用 `max_workers`，ModelScope 使用线程池。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-[8rem_1fr] gap-x-4 gap-y-3 items-center">
            <Label className="text-xs">来源</Label>
            <Select value={source} onValueChange={(v) => setSource(v as Source)}>
              <SelectTrigger className="w-64">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="huggingface">{SOURCE_LABEL.huggingface}</SelectItem>
                <SelectItem value="modelscope">{SOURCE_LABEL.modelscope}</SelectItem>
              </SelectContent>
            </Select>

            <Label className="text-xs">仓库 ID</Label>
            <Input
              value={repoId}
              placeholder={
                source === "modelscope"
                  ? "AI-ModelScope/stable-diffusion-xl-base-1.0"
                  : "stabilityai/stable-diffusion-xl-base-1.0"
              }
              onChange={(e) => setRepoId(e.target.value)}
              className="font-mono"
            />

            <Label className="text-xs">版本 / 分支</Label>
            <Input
              value={revision}
              placeholder={source === "modelscope" ? "master（默认）" : "main（默认）"}
              onChange={(e) => setRevision(e.target.value)}
              className="font-mono w-64"
            />

            <Label className="text-xs">目标目录</Label>
            <Input
              value={targetDir}
              placeholder="默认 ./models/<owner__name>"
              onChange={(e) => setTargetDir(e.target.value)}
              className="font-mono"
            />

            <Label className="text-xs">下载线程</Label>
            <div className="flex items-center gap-3">
              <Input
                type="number"
                min={1}
                max={16}
                value={threads}
                onChange={(e) => setThreads(clampThreadCount(Number(e.target.value)))}
                className="font-mono w-24"
              />
              <div className="text-xs text-muted-foreground inline-flex items-center gap-1.5">
                <ServerCog className="size-3.5" /> 1-16 个并发 worker
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3 pt-1">
            <Button
              size="sm"
              onClick={() => startDownload.mutate()}
              disabled={!ready || running}
            >
              {running ? (
                <Loader2 className="size-3 animate-spin" />
              ) : (
                <Download className="size-3" />
              )}
              {running ? "下载中" : "开始下载"}
            </Button>
            <a
              href={
                source === "modelscope"
                  ? `https://modelscope.cn/models/${encodeURIComponent(repoId)}`
                  : `https://huggingface.co/${encodeURIComponent(repoId)}`
              }
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1"
            >
              <ExternalLink className="size-3" />
              {ready ? "打开仓库主页" : "先填写 owner/name"}
            </a>
          </div>

          {error && (
            <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs font-mono text-destructive whitespace-pre-wrap break-words">
              {error.message}
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Rows3 className="size-4 text-muted-foreground" />
            下载进度
          </CardTitle>
          <CardDescription>{summary}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Progress value={percent} className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-2">
            <ProgressLabel className="text-xs text-muted-foreground">
              {current ? `${SOURCE_LABEL[current.source]} · ${current.repo_id}` : "等待任务"}
            </ProgressLabel>
            <div className="text-xs font-mono text-muted-foreground">{percent.toFixed(1)}%</div>
            <ProgressTrack className="col-span-2 h-3">
              <ProgressIndicator />
            </ProgressTrack>
          </Progress>

          {current && (
            <dl className="grid grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
              <ProgressStat label="线程" value={`${current.threads}`} />
              <ProgressStat
                label="文件"
                value={`${latest?.files_done ?? 0}/${latest?.files_total ?? result?.files ?? 0}`}
              />
              <ProgressStat
                label="字节"
                value={`${formatBytes(latest?.bytes_done ?? result?.total_bytes ?? 0)} / ${formatBytes(latest?.bytes_total ?? result?.total_bytes ?? 0)}`}
              />
              <ProgressStat label="状态" value={current.status} />
            </dl>
          )}

          {result && (
            <div className="rounded-[4px] border border-emerald-500/40 bg-emerald-500/5 px-4 py-3 space-y-1.5">
              <div className="text-sm font-semibold text-emerald-700 dark:text-emerald-400">
                下载完成
              </div>
              <dl className="grid grid-cols-[6rem_1fr] gap-x-3 gap-y-0.5 text-xs font-mono">
                <dt className="text-muted-foreground">来源</dt>
                <dd>{SOURCE_LABEL[current?.source ?? source]}</dd>
                <dt className="text-muted-foreground">仓库</dt>
                <dd className="truncate">{result.repo_id}</dd>
                <dt className="text-muted-foreground">版本</dt>
                <dd>{result.revision}</dd>
                <dt className="text-muted-foreground">文件数</dt>
                <dd>{result.files}</dd>
                <dt className="text-muted-foreground">总大小</dt>
                <dd>{formatBytes(result.total_bytes)}</dd>
                <dt className="text-muted-foreground">保存路径</dt>
                <dd className="truncate" title={result.target}>
                  {result.target}
                </dd>
              </dl>
            </div>
          )}

          {current?.events.length ? (
            <div className="rounded-[4px] border border-border/60 bg-muted/25 max-h-56 overflow-y-auto">
              <ul className="divide-y divide-border/40">
                {current.events.slice(-12).reverse().map((event) => (
                  <li key={`${event.ts}-${event.message}`} className="px-3 py-2 text-xs">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-muted-foreground">
                        {new Date(event.ts * 1000).toLocaleTimeString()}
                      </span>
                      <span className="font-mono">{event.percent?.toFixed(1) ?? "--"}%</span>
                    </div>
                    <div className="mt-0.5 text-muted-foreground break-words">{event.message}</div>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <div className="text-[11px] text-muted-foreground/80">
              大模型下载可能耗时较长。开始后保持服务运行即可，刷新页面仍可通过会话接口查询最近任务。
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function ProgressStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[4px] border border-border/60 bg-background/45 px-3 py-2">
      <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
        {label}
      </div>
      <div className="mt-1 font-mono text-sm truncate">{value}</div>
    </div>
  )
}

function clampThreadCount(value: number): number {
  if (!Number.isFinite(value)) return 1
  return Math.max(1, Math.min(16, Math.round(value)))
}
