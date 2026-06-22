import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import {
  Database,
  FileText,
  Image as ImageIcon,
  Pencil,
  Play,
  Search,
} from "lucide-react"
import {
  api,
  datasetList,
  type DatasetScanResponse,
} from "@/lib/api"
import { readBool, useUrlState } from "@/lib/url-state"
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { CaptionEditorModal } from "./components/caption-editor-modal"
import { PathBar } from "./components/path-bar"
import { SampleGallery } from "./components/sample-gallery"

type Sample = DatasetScanResponse["samples"][number]

const PAGE_SIZE_OPTIONS = [24, 48, 96, 192]

export function DatasetsPage() {
  const datasetsList = useQuery({
    queryKey: ["image-studio-datasets"],
    queryFn: () => datasetList(),
    staleTime: 30_000,
  })
  // Keep the array reference stable across renders. Without this,
  // ``datasetsList.data?.datasets ?? []`` produces a fresh ``[]``
  // every render, and any useEffect that depends on ``knownDatasets``
  // would re-fire forever — the symptom users hit was "Maximum
  // update depth exceeded" the moment ``datasetsList`` settled with
  // an empty list, which froze the route in place because the render
  // loop starved navigation commits.
  const knownDatasets = useMemo(
    () => datasetsList.data?.datasets ?? [],
    [datasetsList.data],
  )

  // ``submitted`` is the actual scan target — synced into the URL so a
  // navigation away and back keeps the user on the same dataset. ``path``
  // is the free-form text input shown only when "高级模式" is on, and
  // stays purely local because mid-typing values aren't worth shoving in
  // the URL bar.
  const { params, update } = useUrlState()
  const submitted = params.get("path") ?? ""
  const recursive = readBool(params, "recursive")
  const advanced = readBool(params, "advanced")

  const setSubmitted = useCallback(
    (next: string) => update({ path: next || null }),
    [update],
  )
  const setRecursive = useCallback(
    (next: boolean) => update({ recursive: next ? "1" : null }),
    [update],
  )
  const setAdvanced = useCallback(
    (next: boolean) => update({ advanced: next ? "1" : null }),
    [update],
  )

  const pageRaw = Number.parseInt(params.get("page") ?? "1", 10)
  const page = Number.isFinite(pageRaw) && pageRaw >= 1 ? pageRaw : 1
  const setPage = useCallback(
    (next: number) => update({ page: next === 1 ? null : String(next) }),
    [update],
  )

  const pageSizeRaw = Number.parseInt(params.get("page_size") ?? "48", 10)
  const pageSize =
    PAGE_SIZE_OPTIONS.includes(pageSizeRaw) ? pageSizeRaw : 48
  const setPageSize = useCallback(
    (next: number) => update({ page_size: next === 48 ? null : String(next) }),
    [update],
  )

  const [path, setPath] = useState("")
  const [editor, setEditor] = useState<{ imagePath: string } | null>(null)
  const navigate = useNavigate()

  // Keep the advanced text input mirror in sync with the URL-driven
  // scan target — fixes the case where the user lands on the page with
  // ?path=... already set and then opens "高级模式".
  useEffect(() => {
    if (path === "" && submitted) setPath(submitted)
  }, [submitted, path])

  // Auto-select the first dataset on first load so the page isn't
  // blank — same affordance the previous "./datasets" default gave.
  useEffect(() => {
    if (!submitted && knownDatasets.length > 0) {
      setSubmitted(knownDatasets[0].path)
      setPath(knownDatasets[0].path)
    }
  }, [knownDatasets, submitted, setSubmitted])

  // Reset to page 1 whenever the scan target changes — otherwise the user
  // can land on an out-of-range page after switching directories. Skipped
  // on mount so a deep link with ``?page=N`` keeps its page on first load.
  const firstScanResetRef = useRef(true)
  useEffect(() => {
    if (firstScanResetRef.current) {
      firstScanResetRef.current = false
      return
    }
    setPage(1)
  }, [submitted, recursive, pageSize, setPage])

  const offset = (page - 1) * pageSize
  const scan = useQuery({
    queryKey: ["dataset-scan", submitted, recursive, pageSize, offset],
    queryFn: () => api.scanDataset(submitted, recursive, pageSize, offset),
    enabled: submitted.trim().length > 0,
    // 翻页时被命中的页可能已经在缓存里;5s 内的回扫直接用缓存,
    // 减掉重复 IO(扫盘 + caption 文件计数)。
    staleTime: 5_000,
  })

  const data = scan.data
  const canTrain = !!data && data.exists && data.image_files > 0

  // Match-current-path so the dropdown's controlled value stays in
  // sync after navigateTo() walks into a sub-folder via PathBar
  // (sub-folder isn't in the registered-datasets list).
  const dropdownValue = useMemo(() => {
    return knownDatasets.some((d) => d.path === submitted) ? submitted : ""
  }, [knownDatasets, submitted])

  const navigateTo = (next: string) => {
    setPath(next)
    setSubmitted(next)
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="px-4 py-4 md:px-6 md:py-5 space-y-4 w-full">
        <Card size="sm">
          <CardContent className="space-y-3">
            {!advanced && (
              <div className="flex gap-2 flex-wrap items-center">
                <Select
                  value={dropdownValue}
                  onValueChange={(v) => {
                    if (v) {
                      setPath(v)
                      setSubmitted(v)
                    }
                  }}
                >
                  <SelectTrigger className="flex-1 min-w-[16rem] font-mono">
                    <SelectValue
                      placeholder={
                        datasetsList.isLoading
                          ? "加载数据集…"
                          : knownDatasets.length === 0
                            ? "datasets/ 下尚无数据集"
                            : "选择数据集…"
                      }
                    />
                  </SelectTrigger>
                  <SelectContent>
                    {knownDatasets.map((d) => (
                      <SelectItem key={d.path} value={d.path}>
                        {d.name}{" "}
                        <span className="text-muted-foreground text-[10px] ml-2">
                          ({d.imageCount} 张)
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Button
                  type="button"
                  variant={recursive ? "default" : "outline"}
                  onClick={() => setRecursive(!recursive)}
                  title="递归扫描所有子目录"
                >
                  递归
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setAdvanced(true)}
                  className="gap-1 text-[11px]"
                >
                  <Pencil className="size-3" />
                  输入路径
                </Button>
                {!datasetsList.isLoading &&
                  knownDatasets.length === 0 && (
                    <Button size="sm" onClick={() => setAdvanced(true)}>
                      去高级模式
                    </Button>
                  )}
              </div>
            )}

            {advanced && (
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
                  <Search className="size-3.5" />{" "}
                  {scan.isFetching ? "扫描中" : "扫描"}
                </Button>
                <Button
                  type="button"
                  variant={recursive ? "default" : "outline"}
                  onClick={() => setRecursive(!recursive)}
                  title="递归扫描所有子目录"
                >
                  递归
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  type="button"
                  onClick={() => setAdvanced(false)}
                  className="gap-1 text-[11px]"
                >
                  切回下拉
                </Button>
              </form>
            )}

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

            <Card size="sm">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between gap-3 flex-wrap">
                  <div>
                    <CardTitle className="text-sm">样本预览</CardTitle>
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
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                {data.samples.length === 0 ? (
                  <div className="rounded-[4px] border border-dashed border-border/70 bg-muted/30 px-4 py-8 text-center text-sm text-muted-foreground">
                    {data.image_files > 0
                      ? "本页没有样本（可能在其它分页）。"
                      : "此目录下未发现图片样本。"}
                  </div>
                ) : (
                  // pageSize 上限 192 时缩略图网格能撑得很高,
                  // 把分页器和「训练此数据集」按钮挤到屏幕外。
                  // 卡片内部限到 ~70vh 局部滚动。
                  <div className="max-h-[70vh] overflow-y-auto pr-1">
                    <SampleGallery
                      samples={data.samples}
                      onEdit={(sample: Sample) =>
                        setEditor({ imagePath: sample.path })
                      }
                      onPreviewImage={(sample) =>
                        setEditor({ imagePath: sample.path })
                      }
                    />
                  </div>
                )}
                <Pagination
                  total={data.image_files}
                  pageSize={pageSize}
                  page={page}
                  onPageChange={setPage}
                  pageSizeOptions={PAGE_SIZE_OPTIONS}
                  onPageSizeChange={setPageSize}
                />
                <div className="flex items-center justify-between gap-3 border-t border-border/60 pt-3">
                  <div className="min-w-0 text-xs text-muted-foreground">
                    跳转到训练配置页并预填{" "}
                    <code className="font-mono text-foreground">dataset.source</code>
                  </div>
                  <Button
                    size="sm"
                    disabled={!canTrain}
                    onClick={() =>
                      navigate("/configs", {
                        state: { overrideDataset: data.path },
                      })
                    }
                  >
                    <Play className="size-3.5" /> 训练
                  </Button>
                </div>
              </CardContent>
            </Card>
          </>
        )}
      </div>

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
    <Card size="sm">
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
