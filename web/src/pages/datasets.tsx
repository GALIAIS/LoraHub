import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { Database, FileText, Image, Play, Search } from "lucide-react"
import { api } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"

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
    <div className="px-8 py-7 space-y-6 max-w-[1180px]">
      <header className="space-y-1">
        <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground/80">
          Dataset manager
        </div>
        <h1 className="text-2xl font-semibold tracking-tight">Datasets</h1>
        <p className="text-sm text-muted-foreground">
          Scan an image folder before training and spot missing kohya caption files.
        </p>
      </header>

      <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Scan folder</CardTitle>
          <CardDescription>Use the same path you put in `dataset.source`.</CardDescription>
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
              <Search className="size-3.5" /> {scan.isFetching ? "Scanning..." : "Scan"}
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
              label="Folder"
              value={data.exists ? "found" : "missing"}
              tone={data.exists ? "default" : "warning"}
            />
            <DatasetStat
              icon={<Image className="size-3.5" />}
              label="Images"
              value={data.image_files.toString()}
            />
            <DatasetStat
              icon={<FileText className="size-3.5" />}
              label="Captions"
              value={`${data.caption_files}/${data.image_files}`}
              tone={data.caption_files === data.image_files ? "default" : "warning"}
            />
          </div>

          <Card className="rounded-[6px] border-border/70 shadow-[var(--panel-shadow)]">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <CardTitle className="text-base">Samples</CardTitle>
                  <CardDescription className="font-mono break-all">{data.path}</CardDescription>
                </div>
                <Badge variant={data.missing_caption_files.length ? "outline" : "secondary"} className="rounded-[2px]">
                  {data.missing_caption_files.length} missing captions
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              {data.samples.length === 0 ? (
                <div className="rounded-[4px] border border-dashed border-border/70 bg-muted/30 px-4 py-8 text-center text-sm text-muted-foreground">
                  No image samples found in this folder.
                </div>
              ) : (
                <ul className="divide-y divide-border/50">
                  {data.samples.map((sample) => (
                    <li key={sample.relative_path} className="py-3 grid grid-cols-[1fr_auto] gap-3">
                      <div className="min-w-0">
                        <div className="font-mono text-xs truncate">{sample.relative_path}</div>
                        <div className="mt-1 text-xs text-muted-foreground truncate">
                          {sample.caption ?? "No caption file yet"}
                        </div>
                      </div>
                      <Badge variant={sample.caption_exists ? "secondary" : "outline"} className="rounded-[2px] self-start">
                        {sample.caption_exists ? "captioned" : "missing .txt"}
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
                <div className="text-sm font-medium">Train with this dataset</div>
                <div className="text-xs text-muted-foreground">
                  Jumps to Recipes, picks the first one, and pre-fills{" "}
                  <code className="font-mono text-foreground">dataset.source</code> in the launch dialog.
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
                <Play className="size-3.5" /> Train
              </Button>
            </CardContent>
          </Card>
        </>
      )}
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
