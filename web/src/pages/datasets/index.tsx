import { useEffect, useState } from "react"
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
import { Pagination } from "@/components/ui/pagination"
import { CaptionEditorModal } from "./components/caption-editor-modal"
import { PathBar } from "./components/path-bar"
import { SampleGallery } from "./components/sample-gallery"
import { TaggingDialog } from "./components/tagging-dialog"

type Sample = DatasetScanResponse["samples"][number]

const PAGE_SIZE_OPTIONS = [24, 48, 96, 192]

export function DatasetsPage() {
  const [path, setPath] = useState("./datasets")
  const [submitted, setSubmitted] = useState("./datasets")
  const [recursive, setRecursive] = useState(false)
  const [tagOpen, setTagOpen] = useState(false)
  const [editor, setEditor] = useState<{ imagePath: string } | null>(null)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(48)
  const navigate = useNavigate()

  // Reset to page 1 whenever the scan target changes — otherwise the user
  // can land on an out-of-range page after switching directories.
  useEffect(() => {
    setPage(1)
  }, [submitted, recursive, pageSize])

  const offset = (page - 1) * pageSize
  const scan = useQuery({
    queryKey: ["dataset-scan", submitted, recursive, pageSize, offset],
    queryFn: () => api.scanDataset(submitted, recursive, pageSize, offset),
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

        <Card>
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
                tone={
                  // Only call the dataset "complete" once it actually
                  // has images. 0/0 used to render as success which
                  // misled users into thinking an empty folder was
                  // already labeled.
                  data.image_files === 0
                    ? "warning"
                    : data.caption_files === data.image_files
                      ? "default"
                      : "warning"
                }
              />
            </div>

            <Card>
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
              <CardContent className="space-y-3">
                {data.samples.length === 0 ? (
                  <div className="rounded-[4px] border border-dashed border-border/70 bg-muted/30 px-4 py-8 text-center text-sm text-muted-foreground">
                    {data.image_files > 0
                      ? "本页没有样本(可能在其它分页)。"
                      : "此目录下未发现图片样本。"}
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
                <Pagination
                  total={data.image_files}
                  pageSize={pageSize}
                  page={page}
                  onPageChange={setPage}
                  pageSizeOptions={PAGE_SIZE_OPTIONS}
                  onPageSizeChange={setPageSize}
                />
              </CardContent>
            </Card>

            <Card>
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
    <Card>
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
