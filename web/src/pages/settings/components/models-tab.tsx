import { useEffect, useMemo, useState, type Dispatch, type SetStateAction } from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import {
  Check,
  Download,
  ExternalLink,
  FileSearch,
  Loader2,
  Rows3,
  ServerCog,
} from "lucide-react"
import { api } from "@/lib/api"
import type { RemoteModelFile } from "@/lib/api/models"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

type Source = "huggingface" | "modelscope"

const SOURCE_LABEL: Record<Source, string> = {
  huggingface: "HuggingFace",
  modelscope: "ModelScope",
}

const SOURCE_OPTIONS: { value: Source; label: string }[] = [
  { value: "modelscope", label: SOURCE_LABEL.modelscope },
  { value: "huggingface", label: SOURCE_LABEL.huggingface },
]

const MODEL_DOWNLOAD_SESSION_KEY = "lorahub:model-download-session-id"
const DEFAULT_ALLOW_PATTERNS = [
  "*.safetensors",
  "*.ckpt",
  "*.pt",
  "*.pth",
  "*.bin",
  "*.gguf",
  "*.onnx",
  "*.json",
  "*.txt",
  "*.model",
  "*.vocab",
  "*.merges",
].join(", ")
const DEFAULT_IGNORE_PATTERNS = [
  ".gitattributes",
  "README*",
  "LICENSE*",
  "*.md",
  "*.png",
  "*.jpg",
  "*.jpeg",
  "*.webp",
  "*.gif",
  "*.mp4",
  "*.zip",
  "*.tar",
  "*.tar.gz",
].join(", ")

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
  const [source, setSource] = useState<Source>("modelscope")
  const [repoId, setRepoId] = useState("")
  const [revision, setRevision] = useState("")
  const [targetDir, setTargetDir] = useState("")
  const [threads, setThreads] = useState(4)
  const [allowPatterns, setAllowPatterns] = useState(DEFAULT_ALLOW_PATTERNS)
  const [ignorePatterns, setIgnorePatterns] = useState(DEFAULT_IGNORE_PATTERNS)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set())
  const [lastListedKey, setLastListedKey] = useState("")
  const parsedAllowPatterns = useMemo(
    () => parsePatterns(allowPatterns),
    [allowPatterns],
  )
  const parsedIgnorePatterns = useMemo(
    () => parsePatterns(ignorePatterns),
    [ignorePatterns],
  )

  const latestDownload = useQuery({
    queryKey: ["model-download-latest"],
    queryFn: api.getLatestModelDownload,
    refetchInterval: (query) =>
      query.state.data?.status === "running" ? 800 : false,
    staleTime: 400,
  })

  useEffect(() => {
    const stored = window.localStorage.getItem(MODEL_DOWNLOAD_SESSION_KEY)
    if (stored) setSessionId(stored)
  }, [])

  useEffect(() => {
    const latest = latestDownload.data
    if (!latest?.session_id) return
    if (latest.status === "running" || !sessionId) {
      setSessionId(latest.session_id)
      window.localStorage.setItem(MODEL_DOWNLOAD_SESSION_KEY, latest.session_id)
    }
  }, [latestDownload.data, sessionId])

  const startDownload = useMutation({
    mutationFn: () =>
      api.downloadModel({
        source,
        repo_id: repoId.trim(),
        revision: revision.trim() || (source === "modelscope" ? "master" : "main"),
        target_dir: targetDir.trim() || undefined,
        threads,
        paths: Array.from(selectedPaths),
        allow_patterns: parsedAllowPatterns,
        ignore_patterns: parsedIgnorePatterns,
      }),
    onSuccess: (session) => {
      setSessionId(session.session_id)
      window.localStorage.setItem(MODEL_DOWNLOAD_SESSION_KEY, session.session_id)
    },
  })

  const fileList = useMutation({
    mutationFn: () =>
      api.listModelFiles({
        source,
        repo_id: repoId.trim(),
        revision: revision.trim() || (source === "modelscope" ? "master" : "main"),
        allow_patterns: parsedAllowPatterns,
        ignore_patterns: parsedIgnorePatterns,
      }),
    onSuccess: (res) => {
      setSelectedPaths(
        new Set(res.files.filter((file) => file.selected).map((file) => file.path)),
      )
      setLastListedKey(
        listKey(source, repoId, revision, parsedAllowPatterns, parsedIgnorePatterns),
      )
    },
  })

  useEffect(() => {
    setSelectedPaths(new Set())
    setLastListedKey("")
    fileList.reset()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source, repoId, revision, allowPatterns, ignorePatterns])

  const session = useQuery({
    queryKey: ["model-download", sessionId],
    queryFn: () => api.getModelDownload(sessionId!),
    enabled: !!sessionId,
    refetchInterval: (query) =>
      query.state.data?.status === "running" || !query.state.data ? 800 : false,
    staleTime: 400,
  })

  const latestCurrent =
    latestDownload.data?.session_id &&
    latestDownload.data.status !== "idle" &&
    (!sessionId || latestDownload.data.session_id === sessionId)
      ? latestDownload.data
      : null
  const current = session.data ?? startDownload.data ?? latestCurrent ?? null
  const error = (startDownload.error as Error | undefined) ?? (session.error as Error | undefined)
  const ready = repoId.includes("/") && repoId.trim().length > 2
  const running = current?.status === "running" || startDownload.isPending
  const listing = fileList.isPending
  const latest = current?.events.at(-1)
  const percent = Math.max(0, Math.min(100, current?.percent ?? 0))
  const currentSource = current?.source
  const listed = fileList.data?.files ?? []
  const selectedFiles = listed.filter((file) => selectedPaths.has(file.path))
  const selectedBytes = selectedFiles.reduce((sum, file) => sum + file.size, 0)
  const staleList =
    lastListedKey !==
    listKey(source, repoId, revision, parsedAllowPatterns, parsedIgnorePatterns)
  const canDownload = ready && selectedPaths.size > 0 && !running && !staleList

  const result = current?.result
  const summary = useMemo(() => {
    if (!current) return "尚未开始下载"
    if (current.status === "running") return latest?.message ?? "下载进行中"
    if (current.status === "failed") return current.error ?? "下载失败"
    return "下载完成"
  }, [current, latest])

  return (
    <div className="space-y-5 w-full">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Download className="size-4 text-muted-foreground" />
            模型下载
          </CardTitle>
          <CardDescription>
            支持 HuggingFace 与 ModelScope。先读取仓库文件清单，只下载勾选的模型权重、
            配置和 tokenizer 文件，避免把仓库里的预览图、文档和无关变体一起拉下来。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-[8rem_1fr] gap-x-4 gap-y-3 items-center">
            <Label className="text-xs">来源</Label>
            <Select items={SOURCE_OPTIONS} value={source} onValueChange={(v) => setSource(v as Source)}>
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

            <Label className="text-xs">包含规则</Label>
            <Input
              value={allowPatterns}
              onChange={(e) => setAllowPatterns(e.target.value)}
              className="font-mono"
              placeholder="*.safetensors, *.json, tokenizer/*"
            />

            <Label className="text-xs">忽略规则</Label>
            <Input
              value={ignorePatterns}
              onChange={(e) => setIgnorePatterns(e.target.value)}
              className="font-mono"
              placeholder="README*, *.png, *.mp4"
            />
          </div>

          <div className="flex items-center gap-3 pt-1">
            <Button
              size="sm"
              variant="outline"
              onClick={() => fileList.mutate()}
              disabled={!ready || running || listing}
            >
              {listing ? (
                <Loader2 className="size-3 animate-spin" />
              ) : (
                <FileSearch className="size-3" />
              )}
              {listing ? "读取中" : "读取文件清单"}
            </Button>
            <Button
              size="sm"
              onClick={() => startDownload.mutate()}
              disabled={!canDownload}
            >
              {running ? (
                <Loader2 className="size-3 animate-spin" />
              ) : (
                <Download className="size-3" />
              )}
              {running ? "下载中" : "开始下载"}
            </Button>
            <div className="text-xs text-muted-foreground">
              已选 {selectedPaths.size} 个文件 · {formatBytes(selectedBytes)}
              {staleList && listed.length > 0 ? " · 清单需刷新" : ""}
            </div>
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
          {fileList.error && (
            <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs font-mono text-destructive whitespace-pre-wrap break-words">
              {(fileList.error as Error).message}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <FileSearch className="size-4 text-muted-foreground" />
            远端文件清单
          </CardTitle>
          <CardDescription>
            默认规则只选择模型运行需要的资产。可以手动调整，下载请求只会包含当前勾选项。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              disabled={listed.length === 0 || running}
              onClick={() => setSelectedPaths(new Set(listed.map((file) => file.path)))}
            >
              全选
            </Button>
            <Button
              size="sm"
              variant="outline"
              disabled={listed.length === 0 || running}
              onClick={() => setSelectedPaths(new Set())}
            >
              清空
            </Button>
            <Button
              size="sm"
              variant="outline"
              disabled={listed.length === 0 || running}
              onClick={() =>
                setSelectedPaths(
                  new Set(
                    listed
                      .filter((file) => file.selected)
                      .map((file) => file.path),
                  ),
                )
              }
            >
              恢复推荐
            </Button>
            {fileList.data && (
              <div className="ml-auto text-xs text-muted-foreground">
                推荐 {fileList.data.selected_count}/{fileList.data.total_count} ·{" "}
                {formatBytes(fileList.data.selected_bytes)} /{" "}
                {formatBytes(fileList.data.total_bytes)}
              </div>
            )}
          </div>

          {listed.length > 0 ? (
            <div className="max-h-[360px] overflow-y-auto rounded-[6px] border border-border/60">
              <Table className="text-xs">
                <TableHeader className="sticky top-0 z-[1]">
                  <TableRow>
                    <TableHead className="w-11"></TableHead>
                    <TableHead>路径</TableHead>
                    <TableHead className="w-28 text-right">大小</TableHead>
                    <TableHead className="w-36">规则</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {listed.map((file) => (
                    <RemoteFileRow
                      key={file.path}
                      file={file}
                      checked={selectedPaths.has(file.path)}
                      disabled={running}
                      onToggle={() => togglePath(file.path, setSelectedPaths)}
                    />
                  ))}
                </TableBody>
              </Table>
            </div>
          ) : (
            <div className="rounded-[6px] border border-dashed border-border/70 px-4 py-8 text-center text-sm text-muted-foreground">
              输入仓库 ID 后读取文件清单。下载前需要至少选择一个文件。
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Rows3 className="size-4 text-muted-foreground" />
            下载进度
          </CardTitle>
          <CardDescription>{summary}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Progress value={percent} className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-2">
            <ProgressLabel className="text-xs text-muted-foreground truncate">
              {currentSource && current?.repo_id
                ? `${SOURCE_LABEL[currentSource]} · ${current.repo_id}`
                : "等待任务"}
            </ProgressLabel>
            <div className="text-xs font-mono tabular-nums text-muted-foreground text-right min-w-[5ch]">
              {percent.toFixed(1)}%
            </div>
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
              <ProgressStat label="选择" value={`${current.paths?.length ?? 0}`} />
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

          {current?.status === "failed" && (
            <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-4 py-3 space-y-1.5">
              <div className="text-sm font-semibold text-destructive">
                下载失败
              </div>
              <div className="text-xs font-mono text-destructive whitespace-pre-wrap break-words">
                {current.error ?? latest?.message ?? "未知错误"}
              </div>
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
                    <div
                      className={`mt-0.5 break-words ${
                        event.message.includes("failed")
                          ? "text-destructive"
                          : "text-muted-foreground"
                      }`}
                    >
                      {event.message}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <div className="text-[11px] text-muted-foreground/80">
              大模型下载可能耗时较长。开始后保持服务运行即可，刷新页面会自动恢复最近任务。
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function ProgressStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[4px] border border-border/60 bg-background/45 px-3 py-2 min-w-0">
      <div className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
        {label}
      </div>
      <div className="mt-1 font-mono tabular-nums text-sm truncate" title={value}>
        {value}
      </div>
    </div>
  )
}

function RemoteFileRow({
  file,
  checked,
  disabled,
  onToggle,
}: {
  file: RemoteModelFile
  checked: boolean
  disabled: boolean
  onToggle: () => void
}) {
  return (
    <TableRow data-state={checked ? "selected" : undefined}>
      <TableCell className="w-11">
        <input
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={onToggle}
          className="size-4 rounded-[4px] border border-border accent-primary"
          aria-label={`选择 ${file.path}`}
        />
      </TableCell>
      <TableCell className="min-w-0">
        <div className="flex items-center gap-2 min-w-0">
          {checked && <Check className="size-3 text-primary shrink-0" />}
          <span className="font-mono truncate" title={file.path}>
            {file.path}
          </span>
        </div>
      </TableCell>
      <TableCell className="text-right font-mono tabular-nums">
        {formatBytes(file.size)}
      </TableCell>
      <TableCell>
        <Badge variant={file.selected ? "secondary" : "outline"} className="text-[10px]">
          {file.reason}
        </Badge>
      </TableCell>
    </TableRow>
  )
}

function togglePath(
  path: string,
  setSelectedPaths: Dispatch<SetStateAction<Set<string>>>,
) {
  setSelectedPaths((current) => {
    const next = new Set(current)
    if (next.has(path)) {
      next.delete(path)
    } else {
      next.add(path)
    }
    return next
  })
}

function clampThreadCount(value: number): number {
  if (!Number.isFinite(value)) return 1
  return Math.max(1, Math.min(16, Math.round(value)))
}

function parsePatterns(value: string): string[] {
  return value
    .split(/[\n,]/)
    .map((part) => part.trim())
    .filter(Boolean)
}

function listKey(
  source: Source,
  repoId: string,
  revision: string,
  allowPatterns: string[],
  ignorePatterns: string[],
): string {
  return [
    source,
    repoId.trim(),
    revision.trim(),
    allowPatterns.join("\0"),
    ignorePatterns.join("\0"),
  ].join(":")
}
