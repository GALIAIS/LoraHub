import { useState } from "react"
import { useNavigate } from "react-router-dom"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  ArrowRight,
  Copy,
  FolderInput,
  Loader2,
  Search,
  Upload,
} from "lucide-react"
import { toast } from "sonner"
import {
  datasetList,
  imageStudioIntakeFromDataset,
  imageStudioIntakeLocalPath,
  imageStudioIntakePreflight,
  type IntakePreflightReport,
} from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

interface Props {
  datasetPath: string
}

export function IntakeStage({ datasetPath }: Props) {
  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto p-4 grid gap-4 lg:grid-cols-2">
        <BrowserUploadHint datasetPath={datasetPath} />
        <LocalPathPanel datasetPath={datasetPath} />
        <FromDatasetPanel datasetPath={datasetPath} />
      </div>
    </div>
  )
}

// --------------------------------------------------------------------------- //
// Hint card — browser upload still lives in Curate stage
// --------------------------------------------------------------------------- //

function BrowserUploadHint({ datasetPath }: { datasetPath: string }) {
  const navigate = useNavigate()
  return (
    <section className="rounded-md border border-border/60 bg-card flex flex-col">
      <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
        <Upload className="size-3.5" />
        <span className="text-xs font-medium">浏览器拖拽上传</span>
      </div>
      <div className="p-3 space-y-2 text-xs text-muted-foreground flex-1 flex flex-col">
        <p>
          单文件 / zip / 文件夹的拖拽上传保留在 整理 阶段顶部 · 那里有原生的
          <code className="text-[11px]"> &lt;DropZone /&gt; </code>。
        </p>
        <p className="opacity-80">
          这里的两个面板针对的是另外两类场景：服务器上已有文件
          （无法走浏览器），以及跨数据集复制。
        </p>
        <Button
          size="sm"
          variant="outline"
          className="mt-auto gap-1 self-start"
          onClick={() => {
            const url = new URL(window.location.href)
            url.searchParams.set("stage", "curate")
            navigate(url.pathname + url.search)
            void datasetPath
          }}
        >
          去 整理 拖拽上传 <ArrowRight className="size-3" />
        </Button>
      </div>
    </section>
  )
}

// --------------------------------------------------------------------------- //
// Local path
// --------------------------------------------------------------------------- //

export function LocalPathPanel({ datasetPath }: { datasetPath: string }) {
  const qc = useQueryClient()
  const [sourcePath, setSourcePath] = useState("")
  const [recursive, setRecursive] = useState(true)
  const [skipDups, setSkipDups] = useState(true)
  const [movInstead, setMoveInstead] = useState(false)
  const [phashThreshold, setPhashThreshold] = useState("4")
  const [preflight, setPreflight] = useState<IntakePreflightReport | null>(null)

  const preflightMut = useMutation({
    mutationFn: () =>
      imageStudioIntakePreflight({
        dataset_path: datasetPath,
        source_path: sourcePath.trim(),
        recursive,
        phash_threshold: Number(phashThreshold),
      }),
    onSuccess: (data) => {
      setPreflight(data)
      if (data.candidate_count === 0) {
        toast.info("未在该路径下找到图片")
      }
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err)
      toast.error("预扫失败", { description: msg })
      setPreflight(null)
    },
  })

  const importMut = useMutation({
    mutationFn: () =>
      imageStudioIntakeLocalPath({
        dataset_path: datasetPath,
        source_path: sourcePath.trim(),
        recursive,
        skip_duplicates: skipDups,
        phash_threshold: Number(phashThreshold),
        move: movInstead,
      }),
    onSuccess: (data) => {
      toast.success(
        `导入完成：${data.imported_count} 新 / ${data.skipped_count} 跳过 / ${data.failed_count} 失败`,
      )
      setPreflight(null)
      qc.invalidateQueries({ queryKey: ["image-studio"] })
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err)
      toast.error("导入失败", { description: msg })
    },
  })

  return (
    <section className="rounded-md border border-border/60 bg-card flex flex-col">
      <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
        <FolderInput className="size-3.5" />
        <span className="text-xs font-medium">从服务器路径导入</span>
      </div>
      <div className="p-3 space-y-2 text-xs">
        <p className="text-muted-foreground">
          指定服务器上的目录或单个文件，自动批量拷入数据集。
          数据集自带 <code className="text-[11px]">.txt</code> caption 也会一起带过来。
        </p>
        <Input
          placeholder="如 /root/raw_images/  或  C:\\images\\char_a"
          value={sourcePath}
          onChange={(e) => setSourcePath(e.target.value)}
          className="h-8 text-xs font-mono"
        />
        <div className="flex items-center gap-3 flex-wrap">
          <label className="inline-flex items-center gap-1.5 select-none">
            <Switch checked={recursive} onCheckedChange={setRecursive} />
            递归子目录
          </label>
          <label className="inline-flex items-center gap-1.5 select-none">
            <Switch checked={skipDups} onCheckedChange={setSkipDups} />
            跳过重复（phash）
          </label>
          <label className="inline-flex items-center gap-1.5 select-none">
            <Switch checked={movInstead} onCheckedChange={setMoveInstead} />
            移动而非复制
          </label>
        </div>
        {skipDups && (
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground">phash 阈值：</span>
            <Select
              value={phashThreshold}
              onValueChange={(v) => v != null && setPhashThreshold(v)}
            >
              <SelectTrigger className="h-7 w-32 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0">0（完全相同）</SelectItem>
                <SelectItem value="2">2（轻微改动）</SelectItem>
                <SelectItem value="4">4（近似 / 重编码）</SelectItem>
                <SelectItem value="8">8（相似 / 同场景）</SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}
        <div className="flex items-center gap-2 pt-1">
          <Button
            size="sm"
            variant="outline"
            disabled={!sourcePath.trim() || preflightMut.isPending}
            onClick={() => preflightMut.mutate()}
            className="h-7 gap-1"
          >
            {preflightMut.isPending ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <Search className="size-3" />
            )}
            预扫
          </Button>
          <Button
            size="sm"
            disabled={!sourcePath.trim() || importMut.isPending}
            onClick={() => {
              if (movInstead) {
                if (
                  !window.confirm(
                    "移动模式：源文件会从原位置消失。确定继续？",
                  )
                )
                  return
              }
              importMut.mutate()
            }}
            className="h-7 gap-1"
          >
            {importMut.isPending ? (
              <Loader2 className="size-3 animate-spin" />
            ) : null}
            导入
          </Button>
          {preflight && (
            <span className="text-[11px] text-muted-foreground ml-auto tabular-nums">
              {preflight.candidate_count} 候选 ·{" "}
              <span className="text-emerald-700">{preflight.new_count} 新</span> ·{" "}
              <span className="text-amber-700">
                {preflight.duplicate_existing_count} 已存在
              </span>{" "}
              · {preflight.duplicate_within_batch_count} 批内重复
            </span>
          )}
        </div>
      </div>
    </section>
  )
}

// --------------------------------------------------------------------------- //
// From-dataset
// --------------------------------------------------------------------------- //

export function FromDatasetPanel({ datasetPath }: { datasetPath: string }) {
  const qc = useQueryClient()
  const datasetsQuery = useQuery({
    queryKey: ["image-studio-datasets"],
    queryFn: () => datasetList(),
  })

  const [sourceDataset, setSourceDataset] = useState<string>("")
  const [pattern, setPattern] = useState("*")
  const [skipDups, setSkipDups] = useState(true)

  // Filter out the current dataset from the candidate list — copying
  // a dataset into itself is rejected by the backend anyway and the
  // dropdown looks cleaner without it.
  const candidates = (datasetsQuery.data?.datasets ?? []).filter(
    (d) => d.path !== datasetPath,
  )

  const importMut = useMutation({
    mutationFn: () =>
      imageStudioIntakeFromDataset({
        dataset_path: datasetPath,
        source_dataset_path: sourceDataset,
        pattern: pattern.trim() || "*",
        skip_duplicates: skipDups,
      }),
    onSuccess: (data) => {
      toast.success(
        `已从 ${candidates.find((d) => d.path === sourceDataset)?.name ?? "源"} 导入 ${data.imported_count} 张`,
        {
          description: `候选 ${data.candidate_count} · 跳过 ${data.skipped_count} · 失败 ${data.failed_count}`,
        },
      )
      qc.invalidateQueries({ queryKey: ["image-studio"] })
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err)
      toast.error("跨数据集导入失败", { description: msg })
    },
  })

  return (
    <section className="rounded-md border border-border/60 bg-card flex flex-col">
      <div className="flex items-center gap-2 border-b border-border/60 px-3 py-2">
        <Copy className="size-3.5" />
        <span className="text-xs font-medium">从已有数据集复制</span>
      </div>
      <div className="p-3 space-y-2 text-xs">
        <p className="text-muted-foreground">
          从其它数据集挑一部分图（支持 fnmatch glob）拷入当前数据集。源数据集不动。
        </p>
        <Select
          value={sourceDataset}
          onValueChange={(v) => v != null && setSourceDataset(v)}
        >
          <SelectTrigger className="h-8 text-xs">
            <SelectValue placeholder="选择源数据集…" />
          </SelectTrigger>
          <SelectContent>
            {candidates.length === 0 ? (
              <SelectItem value="__none__" disabled>
                没有其它数据集
              </SelectItem>
            ) : (
              candidates.map((d) => (
                <SelectItem key={d.path} value={d.path}>
                  {d.name}{" "}
                  <span className="text-muted-foreground text-[10px]">
                    ({d.imageCount ?? 0} 张)
                  </span>
                </SelectItem>
              ))
            )}
          </SelectContent>
        </Select>
        <Input
          placeholder="过滤（fnmatch 例：portrait* / char_a/*）"
          value={pattern}
          onChange={(e) => setPattern(e.target.value)}
          className="h-8 text-xs font-mono"
        />
        <label className="inline-flex items-center gap-1.5 select-none">
          <Switch checked={skipDups} onCheckedChange={setSkipDups} />
          跳过重复 (phash)
        </label>
        <Button
          size="sm"
          disabled={!sourceDataset || importMut.isPending}
          onClick={() => importMut.mutate()}
          className="w-full h-7 gap-1 mt-1"
        >
          {importMut.isPending ? (
            <Loader2 className="size-3 animate-spin" />
          ) : null}
          复制
        </Button>
      </div>
    </section>
  )
}
