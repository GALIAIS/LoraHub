import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import {
  Database,
  FileText,
  Image as ImageIcon,
  Play,
  Search,
  Sparkles,
} from "lucide-react"
import { api, type DatasetScanResponse } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { CaptionEditorModal } from "./components/caption-editor-modal"
import { PathBar } from "./components/path-bar"
import { SampleGallery } from "./components/sample-gallery"
import { TaggingDialog } from "./components/tagging-dialog"

type Sample = DatasetScanResponse["samples"][number]

export function DatasetsPage() {
  const [path, setPath] = useState("./datasets")
  const [submitted, setSubmitted] = useState("./datasets")
  const [recursive, setRecursive] = useState(false)
  const [tagOpen, setTagOpen] = useState(false)
  const [editor, setEditor] = useState<{ imagePath: string } | null>(null)
  const navigate = useNavigate()

  const scan = useQuery({
    queryKey: ["dataset-scan", submitted, recursive],
    queryFn: () => api.scanDataset(submitted, recursive),
    enabled: submitted.trim().length > 0,
  })

  const data = scan.data
  const canTrain = !!data && data.exists && data.image_files > 0
  const canTag = !!data && data.exists && data.image_files > 0

  const navigateTo = (next: string) => {
    setPath(next)
    setSubmitted(next)
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="px-8 py-7 space-y-5 w-full">
        <header className="space-y-1">
          <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
            数据集管理
          </div>
          <h1 className="text-2xl font-semibold tracking-tight">数据集</h1>
          <p className="text-sm text-muted-foreground">
            扫描数据集目录、预览缩略图,点击「编辑」弹窗修改 caption。
          </p>
        </header>

        <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">扫描目录</CardTitle>
            <CardDescription>
              与 <code className="font-mono">dataset.source</code> 用同一路径。
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <form
              className="flex gap-2 flex-wrap"
              onSubmit={(event) => {
                event.preventDefault()
                setSubmitted(path)
              }}
            >
              <Input
                value={path}
                onChange={(event) => setPath(event.target.value)}
                className="font-mono flex-1 min-w-[14rem]"
                placeholder="./datasets/my_character"
              />
              <Button type="submit" disabled={scan.isFetching}>
                <Search className="size-3.5" /> {scan.isFetching ? "扫描中" : "扫描"}
              </Button>
              <Button
                type="button"
                variant={recursive ? "default" : "outline"}
                onClick={() => setRecursive((v) => !v)}
                title="递归扫描所有子目录"
              >
                递归
              </Button>
            </form>
            {data?.exists && (
              <PathBar path={data.path} onNavigate={navigateTo} />
            )}
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
                <div className="flex items-center justify-between gap-3 flex-wrap">
                  <div>
                    <CardTitle className="text-base">样本预览</CardTitle>
                    <CardDescription className="font-mono break-all">
                      {data.path}
                    </CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge
                      variant={
                        data.missing_caption_files.length ? "outline" : "secondary"
                      }
                      className="rounded-[2px]"
                    >
                      缺失标注 {data.missing_caption_files.length} 张
                    </Badge>
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={!canTag}
                      onClick={() => setTagOpen(true)}
                    >
                      <Sparkles className="size-3.5" /> 自动标注
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {data.samples.length === 0 ? (
                  <div className="rounded-[4px] border border-dashed border-border/70 bg-muted/30 px-4 py-8 text-center text-sm text-muted-foreground">
                    此目录下未发现图片样本。
                  </div>
                ) : (
                  <SampleGallery
                    samples={data.samples}
                    onEdit={(sample: Sample) =>
                      setEditor({ imagePath: sample.path })
                    }
                    onPreviewImage={(sample) =>
                      setEditor({ imagePath: sample.path })
                    }
                  />
                )}
              </CardContent>
            </Card>

            <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
              <CardContent className="px-4 py-3 flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-sm font-medium">用此数据集训练</div>
                  <div className="text-xs text-muted-foreground">
                    跳转到训练配置页,自动预填{" "}
                    <code className="font-mono text-foreground">dataset.source</code>。
                  </div>
                </div>
                <Button
                  disabled={!canTrain}
                  onClick={() =>
                    navigate("/configs", {
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

      {data && (
        <TaggingDialog
          open={tagOpen}
          onOpenChange={setTagOpen}
          path={data.path}
          onCompleted={() => scan.refetch()}
        />
      )}

      <CaptionEditorModal
        open={editor !== null}
        onOpenChange={(o) => !o && setEditor(null)}
        imagePath={editor?.imagePath ?? null}
        onAfterSave={() => scan.refetch()}
      />
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
