import { useState } from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Download, ExternalLink, Loader2 } from "lucide-react"
import { api } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card"
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
  modelscope: "ModelScope（魔搭）",
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

/**
 * Model downloader: enter a repo id, pick a source, and pull files.
 * Defaults the source to ModelScope when the user has flagged it as
 * preferred in Network tab; otherwise HuggingFace.
 */
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

  const download = useMutation({
    mutationFn: () =>
      api.downloadModel({
        source,
        repo_id: repoId.trim(),
        revision: revision.trim() || (source === "modelscope" ? "master" : "main"),
        target_dir: targetDir.trim() || undefined,
      }),
  })

  const result = download.data
  const error = download.error as Error | undefined
  const ready = repoId.includes("/") && repoId.trim().length > 2

  return (
    <div className="space-y-5">
      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Download className="size-4 text-muted-foreground" />
            下载模型
          </CardTitle>
          <CardDescription>
            支持 HuggingFace 和 ModelScope。下载完成后文件将保存到指定目录或
            <code className="text-foreground"> ./models/&lt;owner__name&gt; </code>。
            HuggingFace 走 huggingface_hub，会自动遵循「网络加速」中的镜像设置。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-[8rem_1fr] gap-x-4 items-center">
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
          </div>

          <div className="grid grid-cols-[8rem_1fr] gap-x-4 items-center">
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
          </div>

          <div className="grid grid-cols-[8rem_1fr] gap-x-4 items-center">
            <Label className="text-xs">版本 / 分支</Label>
            <Input
              value={revision}
              placeholder={source === "modelscope" ? "master（默认）" : "main（默认）"}
              onChange={(e) => setRevision(e.target.value)}
              className="font-mono w-64"
            />
          </div>

          <div className="grid grid-cols-[8rem_1fr] gap-x-4 items-center">
            <Label className="text-xs">目标目录</Label>
            <Input
              value={targetDir}
              placeholder="（默认 ./models/&lt;owner__name&gt;）"
              onChange={(e) => setTargetDir(e.target.value)}
              className="font-mono"
            />
          </div>

          <div className="flex items-center gap-3 pt-1">
            <Button
              size="sm"
              onClick={() => download.mutate()}
              disabled={!ready || download.isPending}
            >
              {download.isPending ? (
                <Loader2 className="size-3 animate-spin" />
              ) : (
                <Download className="size-3" />
              )}
              {download.isPending ? "下载中…" : "开始下载"}
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
              {ready ? "在仓库主页查看" : "需要先填写仓库 ID"}
            </a>
          </div>

          {error && (
            <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs font-mono text-destructive whitespace-pre-wrap break-words">
              {error.message}
            </div>
          )}

          {result && (
            <div className="rounded-[4px] border border-emerald-500/40 bg-emerald-500/5 px-4 py-3 space-y-1.5">
              <div className="text-sm font-semibold text-emerald-700 dark:text-emerald-400">
                下载完成
              </div>
              <dl className="grid grid-cols-[6rem_1fr] gap-x-3 gap-y-0.5 text-xs font-mono">
                <dt className="text-muted-foreground">来源</dt>
                <dd>{SOURCE_LABEL[source]}</dd>
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

          <div className="text-[11px] text-muted-foreground/80 space-y-1">
            <div>
              · HuggingFace：通过 <code>huggingface_hub.snapshot_download</code> 下载，
              支持断点续传与镜像加速。
            </div>
            <div>
              · ModelScope：直接调用魔搭 HTTP API（无需安装 modelscope SDK），
              文档：
              <a
                className="text-foreground hover:underline"
                href="https://modelscope.cn/docs/models/download"
                target="_blank"
                rel="noopener noreferrer"
              >
                modelscope.cn/docs/models/download
              </a>
              。
            </div>
            <div>
              · 大模型下载耗时较长。请保持本页打开直到完成；中断会留下未完成的目录，
              重新触发将重新下载所有文件。
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
