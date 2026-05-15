import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { Database, FileText, Image, Play, Search } from "lucide-react"
import { api } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"

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
            训练前扫描图片目录，预览样本并核对每张图是否都有 kohya caption 文件。
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
                icon={<Image className="size-3.5" />}
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
                  <ul className="divide-y divide-border/50">
                    {data.samples.map((sample) => (
                      <li key={sample.relative_path} className="py-3 grid grid-cols-[1fr_auto] gap-3">
                        <div className="min-w-0">
                          <div className="font-mono text-xs truncate">{sample.relative_path}</div>
                          <div className="mt-1 text-xs text-muted-foreground truncate">
                            {sample.caption ?? "暂无标注文件"}
                          </div>
                        </div>
                        <Badge
                          variant={sample.caption_exists ? "secondary" : "outline"}
                          className="rounded-[2px] self-start"
                        >
                          {sample.caption_exists ? "已标注" : "缺 .txt"}
                        </Badge>
                      </li>
                    ))}
                  </ul>
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
