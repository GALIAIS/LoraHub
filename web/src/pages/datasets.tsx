import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import {
  Database,
  FileText,
  Image as ImageIcon,
  Pencil,
  Play,
  Save,
  Search,
  X,
} from "lucide-react"
import { api, type DatasetScanResponse } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"

type Sample = DatasetScanResponse["samples"][number]

const IMAGE_EXTS = new Set([
  "bmp",
  "gif",
  "jpeg",
  "jpg",
  "png",
  "webp",
])

function isImageSample(sample: Sample): boolean {
  const ext = sample.name.split(".").pop()?.toLowerCase()
  return !!ext && IMAGE_EXTS.has(ext)
}

export function DatasetsPage() {
  const [path, setPath] = useState("./datasets")
  const [submitted, setSubmitted] = useState("./datasets")
  const navigate = useNavigate()

  const scan = useQuery({
    queryKey: ["dataset-scan", submitted],
    queryFn: () => api.scanDataset(submitted),
    enabled: submitted.trim().length > 0,
  })

  const data = scan.data
  const canTrain = !!data && data.exists && data.image_files > 0

  return (
    <div className="h-full overflow-y-auto">
      <div className="px-8 py-7 space-y-6 w-full">
        <header className="space-y-1">
          <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
            数据集管理
          </div>
          <h1 className="text-2xl font-semibold tracking-tight">数据集</h1>
          <p className="text-sm text-muted-foreground">
            训练前扫描图片目录，预览缩略图，直接编辑每张图的 kohya caption。
          </p>
        </header>

        <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">扫描目录</CardTitle>
            <CardDescription>
              使用与 <code className="font-mono">dataset.source</code> 相同的路径。
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form
              className="flex gap-2"
              onSubmit={(event) => {
                event.preventDefault()
                setSubmitted(path)
              }}
            >
              <Input
                value={path}
                onChange={(event) => setPath(event.target.value)}
                className="font-mono"
                placeholder="./datasets/my_character"
              />
              <Button type="submit" disabled={scan.isFetching}>
                <Search className="size-3.5" /> {scan.isFetching ? "扫描中" : "扫描"}
              </Button>
            </form>
          </CardContent>
        </Card>

        {scan.isError && (
          <div className="rounded-[4px] border border-destructive/40 bg-destructive/5 px-4 py-3 text-xs font-mono text-destructive">
            {(scan.error as Error).message}
          </div>
        )}

        {data && (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <DatasetStat
                icon={<Database className="size-3.5" />}
                label="目录"
                value={data.exists ? "存在" : "未找到"}
                tone={data.exists ? "default" : "warning"}
              />
              <DatasetStat
                icon={<ImageIcon className="size-3.5" />}
                label="图片数量"
                value={data.image_files.toString()}
              />
              <DatasetStat
                icon={<FileText className="size-3.5" />}
                label="标注覆盖"
                value={`${data.caption_files}/${data.image_files}`}
                tone={data.caption_files === data.image_files ? "default" : "warning"}
              />
            </div>

            <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <CardTitle className="text-base">样本预览</CardTitle>
                    <CardDescription className="font-mono break-all">{data.path}</CardDescription>
                  </div>
                  <Badge
                    variant={data.missing_caption_files.length ? "outline" : "secondary"}
                    className="rounded-[2px]"
                  >
                    缺失标注 {data.missing_caption_files.length} 张
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                {data.samples.length === 0 ? (
                  <div className="rounded-[4px] border border-dashed border-border/70 bg-muted/30 px-4 py-8 text-center text-sm text-muted-foreground">
                    此目录下未发现图片样本。
                  </div>
                ) : (
                  <SampleGallery samples={data.samples} scanKey={submitted} />
                )}
              </CardContent>
            </Card>

            <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
              <CardContent className="px-4 py-3 flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-sm font-medium">用此数据集训练</div>
                  <div className="text-xs text-muted-foreground">
                    跳转到训练配方页，自动预填{" "}
                    <code className="font-mono text-foreground">dataset.source</code>。
                  </div>
                </div>
                <Button
                  disabled={!canTrain}
                  onClick={() =>
                    navigate("/recipes", {
                      state: { overrideDataset: data.path },
                    })
                  }
                >
                  <Play className="size-3.5" /> 训练
                </Button>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}

function SampleGallery({
  samples,
  scanKey,
}: {
  samples: Sample[]
  scanKey: string
}) {
  const [expanded, setExpanded] = useState<string | null>(null)
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
      {samples.map((sample) => (
        <SampleCard
          key={sample.relative_path}
          sample={sample}
          isOpen={expanded === sample.relative_path}
          onToggle={() =>
            setExpanded((cur) =>
              cur === sample.relative_path ? null : sample.relative_path,
            )
          }
          scanKey={scanKey}
        />
      ))}
    </div>
  )
}

function SampleCard({
  sample,
  isOpen,
  onToggle,
  scanKey,
}: {
  sample: Sample
  isOpen: boolean
  onToggle: () => void
  scanKey: string
}) {
  const isImage = isImageSample(sample)
  // When the user opens an editor, lazy-load the canonical caption from the
  // server (the scan response only carries a stripped preview). When the
  // panel is closed we drop the query, so flipping cards never holds caches.
  return (
    <div
      className={`relative rounded-[6px] border bg-card/60 overflow-hidden transition-colors ${
        isOpen ? "border-primary/60 ring-1 ring-primary/30 col-span-2 sm:col-span-3 md:col-span-4 lg:col-span-5 xl:col-span-6" : "border-border/60 hover:border-primary/40"
      }`}
    >
      <div className={isOpen ? "flex flex-col md:flex-row gap-3" : "flex flex-col"}>
        <button
          type="button"
          onClick={onToggle}
          className={`group block bg-muted/40 ${
            isOpen ? "md:w-72 md:flex-shrink-0" : "w-full"
          }`}
          title={sample.caption ?? sample.name}
        >
          {isImage ? (
            <div className="aspect-square overflow-hidden">
              <img
                src={api.datasetThumbUrl(sample.path, 336)}
                loading="lazy"
                alt={sample.name}
                className="w-full h-full object-cover group-hover:scale-[1.02] transition-transform"
                onError={(event) => {
                  // Hide thumbnails for files Pillow refused to decode -- the
                  // caption editor still works, the user just sees a blank
                  // tile instead of the broken-image glyph.
                  ;(event.currentTarget as HTMLImageElement).style.visibility = "hidden"
                }}
              />
            </div>
          ) : (
            <div className="aspect-square grid place-items-center text-[11px] text-muted-foreground/70 font-mono px-3 text-center">
              {sample.name}
            </div>
          )}
        </button>
        <div className="flex-1 min-w-0 px-2 py-2 space-y-1.5">
          <div className="flex items-center gap-2">
            <div
              className="font-mono text-[11px] truncate flex-1"
              title={sample.relative_path}
            >
              {sample.relative_path}
            </div>
            <Badge
              variant={sample.caption_exists ? "secondary" : "outline"}
              className="rounded-[2px] text-[10px] py-0 px-1.5"
            >
              {sample.caption_exists ? "已标注" : "缺 .txt"}
            </Badge>
          </div>
          {!isOpen && (
            <div className="flex items-start gap-2">
              <p className="text-[11px] text-muted-foreground line-clamp-2 flex-1">
                {sample.caption ?? "暂无标注"}
              </p>
              <button
                type="button"
                onClick={onToggle}
                className="text-[11px] text-primary hover:underline shrink-0 inline-flex items-center gap-1"
              >
                <Pencil className="size-3" /> 编辑
              </button>
            </div>
          )}
          {isOpen && (
            <CaptionEditor
              imagePath={sample.path}
              fallback={sample.caption ?? ""}
              scanKey={scanKey}
              onClose={onToggle}
            />
          )}
        </div>
      </div>
    </div>
  )
}

function CaptionEditor({
  imagePath,
  fallback,
  scanKey,
  onClose,
}: {
  imagePath: string
  fallback: string
  scanKey: string
  onClose: () => void
}) {
  const qc = useQueryClient()
  const captionQuery = useQuery({
    queryKey: ["caption", imagePath],
    queryFn: () => api.getCaption(imagePath),
    // Always fetch fresh server-side text so edits don't compete with the
    // stripped preview from the scan response.
    staleTime: 0,
  })

  const [draft, setDraft] = useState<string | null>(null)
  // Seed the textarea once from whichever source returns first.
  const text = draft ?? captionQuery.data?.caption ?? fallback

  const save = useMutation({
    mutationFn: (value: string) => api.putCaption(imagePath, value),
    onSuccess: async (resp) => {
      setDraft(null)
      qc.setQueryData(["caption", imagePath], resp)
      // Re-scan so the dataset stat (caption_files / missing) updates.
      await qc.invalidateQueries({ queryKey: ["dataset-scan", scanKey] })
    },
  })

  const dirty = draft !== null && draft !== (captionQuery.data?.caption ?? fallback)
  const errorMessage =
    save.error instanceof Error
      ? save.error.message
      : captionQuery.error instanceof Error
        ? captionQuery.error.message
        : null

  return (
    <div className="space-y-2">
      <textarea
        value={text}
        onChange={(event) => setDraft(event.target.value)}
        rows={5}
        className="w-full rounded-[3px] border border-input bg-background/80 px-2 py-1.5 text-[12px] font-mono leading-relaxed resize-y focus:outline-none focus:ring-2 focus:ring-ring/40"
        placeholder="一行一段，或用逗号分隔的 kohya 标签"
        spellCheck={false}
        disabled={save.isPending}
      />
      {errorMessage && (
        <div className="rounded-[3px] border border-destructive/40 bg-destructive/5 px-2 py-1 text-[11px] font-mono text-destructive">
          {errorMessage}
        </div>
      )}
      <div className="flex items-center justify-between gap-2">
        <div className="text-[10px] text-muted-foreground/70">
          UTF-8 · 写入 <code className="font-mono">{".txt"}</code> 同名文件
        </div>
        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-7 text-[11px]"
            onClick={() => {
              setDraft(null)
              onClose()
            }}
            disabled={save.isPending}
          >
            <X className="size-3" /> 取消
          </Button>
          <Button
            type="button"
            size="sm"
            className="h-7 text-[11px]"
            disabled={!dirty || save.isPending}
            onClick={() => save.mutate(draft ?? "")}
          >
            <Save className="size-3" /> {save.isPending ? "保存中" : "保存"}
          </Button>
        </div>
      </div>
    </div>
  )
}

function DatasetStat({
  icon,
  label,
  value,
  tone = "default",
}: {
  icon: React.ReactNode
  label: string
  value: string
  tone?: "default" | "warning"
}) {
  const toneStyle = tone === "warning" ? "text-amber-700 dark:text-amber-400" : "text-foreground"
  return (
    <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
      <CardContent className="px-4 py-3">
        <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
          {icon}
          {label}
        </div>
        <div className={`mt-1.5 text-2xl font-semibold tracking-tight tabular-nums ${toneStyle}`}>
          {value}
        </div>
      </CardContent>
    </Card>
  )
}
