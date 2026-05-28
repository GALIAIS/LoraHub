/**
 * 审计类工具 — 数据集体检 / L1 哈希查重 / L2 语义相似。
 *
 * audit-scan 直接复用 AuditStage 的整页面板（含触发词输入 / scan 控件 /
 * 异常列表 / 直方图）。dedupe-l1 / similarity-l2 单独写一个轻量 view，
 * 因为既有 DuplicatesView 仅给 L1 用，L2 走另一组 API（imageStudioSimilarity*）。
 */
import { useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { ImageOff, Loader2, RefreshCw, Trash2 } from "lucide-react"
import { toast } from "sonner"
import {
  api,
  imageStudioBatchDelete,
  imageStudioDedupeClusters,
  imageStudioDedupeScan,
  imageStudioSimilarityBatchDelete,
  imageStudioSimilarityClusters,
  imageStudioSimilarityScan,
  type DedupeCluster,
} from "@/lib/api"
import { AuditStage } from "../components/stages/audit-stage"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

// --------------------------------------------------------------------------- //
// audit-scan — 直接复用 AuditStage 整页
// --------------------------------------------------------------------------- //

export function AuditScanTool({ datasetPath }: { datasetPath: string }) {
  return <AuditStage datasetPath={datasetPath} />
}

// --------------------------------------------------------------------------- //
// dedupe-l1 (phash)
// --------------------------------------------------------------------------- //

export function DedupeL1Tool({ datasetPath }: { datasetPath: string }) {
  return (
    <ClusterView
      datasetPath={datasetPath}
      kind="phash"
      title="L1 感知哈希查重"
      description="phash64 聚类 · 阈值 10 (Hamming 距离 ≤ 10 视为相似)"
      scanLabel="扫描像素重复"
      scanFn={(path, recursive) =>
        imageStudioDedupeScan({ path, recursive, algo: "phash64" })
      }
      clustersFn={(path) => imageStudioDedupeClusters({ path, threshold: 10 })}
      batchDeleteFn={(paths) => imageStudioBatchDelete({ paths })}
    />
  )
}

// --------------------------------------------------------------------------- //
// similarity-l2 (AI embedding)
// --------------------------------------------------------------------------- //

export function SimilarityL2Tool({ datasetPath }: { datasetPath: string }) {
  return (
    <ClusterView
      datasetPath={datasetPath}
      kind="ai"
      title="L2 语义相似"
      description="多模态嵌入 · 同主体 / 同角度的不同噪声 · 走 GPU"
      scanLabel="扫描语义重复（GPU）"
      scanFn={(path, recursive) =>
        imageStudioSimilarityScan({ path, recursive, mode: "embedding" })
      }
      clustersFn={(path) =>
        imageStudioSimilarityClusters({ path, kind: "ai", threshold: 0.92 })
      }
      batchDeleteFn={(paths) => imageStudioSimilarityBatchDelete({ paths })}
    />
  )
}

// --------------------------------------------------------------------------- //
// Shared cluster view — phash / AI 共用同一套 UI
// --------------------------------------------------------------------------- //

interface ClusterViewProps {
  datasetPath: string
  kind: "phash" | "ai"
  title: string
  description: string
  scanLabel: string
  scanFn: (path: string, recursive: boolean) => Promise<unknown>
  clustersFn: (path: string) => Promise<{ clusters: DedupeCluster[] }>
  batchDeleteFn: (paths: string[]) => Promise<unknown>
}

function ClusterView({
  datasetPath,
  kind,
  title,
  description,
  scanLabel,
  scanFn,
  clustersFn,
  batchDeleteFn,
}: ClusterViewProps) {
  const qc = useQueryClient()
  const [recursive, setRecursive] = useState(true)
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set())

  const scanMutation = useMutation({
    mutationFn: () => scanFn(datasetPath, recursive),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["image-studio", "clusters", kind, datasetPath] })
      toast.success("扫描完成")
    },
    onError: (err) =>
      toast.error("扫描失败", {
        description: err instanceof Error ? err.message : String(err),
      }),
  })

  const clustersQuery = useQuery({
    queryKey: ["image-studio", "clusters", kind, datasetPath],
    queryFn: () => clustersFn(datasetPath),
    enabled: Boolean(datasetPath),
  })

  const deleteMutation = useMutation({
    mutationFn: () => batchDeleteFn(Array.from(selectedPaths)),
    onSuccess: () => {
      setSelectedPaths(new Set())
      qc.invalidateQueries({ queryKey: ["image-studio"] })
      toast.success("已删除选中图片")
    },
    onError: (err) =>
      toast.error("删除失败", {
        description: err instanceof Error ? err.message : String(err),
      }),
  })

  const clusters = clustersQuery.data?.clusters ?? []
  const totalDupes = clusters.reduce(
    (s, c) => s + Math.max(0, c.members.length - 1),
    0,
  )

  const togglePath = (p: string) =>
    setSelectedPaths((prev) => {
      const next = new Set(prev)
      if (next.has(p)) next.delete(p)
      else next.add(p)
      return next
    })

  const selectAllSuggested = () => {
    const out = new Set<string>()
    for (const c of clusters)
      for (const m of c.members) if (m.path !== c.suggestedKeep) out.add(m.path)
    setSelectedPaths(out)
  }

  return (
    <div className="flex h-full flex-col overflow-y-auto gap-4 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          onClick={() => scanMutation.mutate()}
          disabled={scanMutation.isPending}
        >
          {scanMutation.isPending ? (
            <Loader2 className="size-3 animate-spin" />
          ) : (
            <RefreshCw className="size-3" />
          )}
          {scanMutation.isPending ? "扫描中…" : scanLabel}
        </Button>
        <label className="inline-flex items-center gap-1.5 text-xs text-muted-foreground select-none">
          <input
            type="checkbox"
            checked={recursive}
            onChange={(e) => setRecursive(e.target.checked)}
            className="size-3"
          />
          递归子目录
        </label>
        {clusters.length > 0 && (
          <>
            <Button
              variant="outline"
              size="sm"
              onClick={selectAllSuggested}
              className="h-7 text-[11px]"
            >
              全选建议删除 ({totalDupes})
            </Button>
            <span className="text-xs text-muted-foreground">
              {clusters.length} 个聚类 · {totalDupes} 张可去重 · 已选{" "}
              <span className="font-mono text-foreground">{selectedPaths.size}</span>
            </span>
            {selectedPaths.size > 0 && (
              <Button
                variant="destructive"
                size="sm"
                onClick={() => deleteMutation.mutate()}
                disabled={deleteMutation.isPending}
                className="h-7 ml-auto text-[11px] gap-1"
              >
                <Trash2 className="size-3" /> 删除选中 ({selectedPaths.size})
              </Button>
            )}
          </>
        )}
      </div>

      <div className="text-xs text-muted-foreground space-y-0.5">
        <div>
          <span className="font-medium text-foreground">{title}</span> · {description}
        </div>
      </div>

      {clustersQuery.isLoading && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-3 animate-spin" /> 加载聚类…
        </div>
      )}
      {clusters.length === 0 && !clustersQuery.isLoading && (
        <div className="rounded-[6px] border border-dashed border-border/70 bg-muted/30 px-6 py-12 text-center text-sm text-muted-foreground">
          {scanMutation.isSuccess
            ? "未发现重复聚类。"
            : "尚未扫描。点上方按钮开始分析。"}
        </div>
      )}

      {clusters.map((c) => (
        <ClusterCard
          key={c.id}
          cluster={c}
          selectedPaths={selectedPaths}
          onToggle={togglePath}
          kind={kind}
        />
      ))}
    </div>
  )
}

function ClusterCard({
  cluster,
  selectedPaths,
  onToggle,
  kind,
}: {
  cluster: DedupeCluster
  selectedPaths: Set<string>
  onToggle: (path: string) => void
  kind: "phash" | "ai"
}) {
  const [expanded, setExpanded] = useState(true)
  const keepMember = useMemo(
    () => cluster.members.find((m) => m.path === cluster.suggestedKeep) ?? null,
    [cluster],
  )
  const selectAllInCluster = () => {
    for (const m of cluster.members) {
      if (m.path !== cluster.suggestedKeep && !selectedPaths.has(m.path)) {
        onToggle(m.path)
      }
    }
  }

  return (
    <div className="rounded-md border bg-card/30">
      <div className="flex w-full items-center gap-2 px-3 py-2 text-sm">
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="flex items-center gap-2 flex-1 text-left hover:bg-muted/50 rounded px-1 -mx-1"
        >
          <span className="font-mono text-[11px]">{cluster.id}</span>
          <Badge variant="outline" className="rounded-[2px] text-[10px]">
            {cluster.members.length} 张
          </Badge>
          <span className="ml-auto text-xs">{expanded ? "▼" : "▶"}</span>
        </button>
        <Button
          variant="outline"
          size="sm"
          onClick={selectAllInCluster}
          className="h-7 text-[10px]"
        >
          选择重复
        </Button>
      </div>
      {expanded && (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-2 border-t p-3">
          {cluster.members.map((m) => {
            const isKeep = m.path === cluster.suggestedKeep
            const isSelected = selectedPaths.has(m.path)
            const distance =
              kind === "phash" && !isKeep && keepMember
                ? hammingDistance(m.hash, keepMember.hash)
                : null
            return (
              <DupeMemberTile
                key={m.path}
                path={m.path}
                isKeep={isKeep}
                isSelected={isSelected}
                distance={distance}
                onToggle={() => onToggle(m.path)}
              />
            )
          })}
        </div>
      )}
    </div>
  )
}

function DupeMemberTile({
  path,
  isKeep,
  isSelected,
  distance,
  onToggle,
}: {
  path: string
  isKeep: boolean
  isSelected: boolean
  distance: number | null
  onToggle: () => void
}) {
  const [broken, setBroken] = useState(false)
  return (
    <div
      className={cn(
        "relative flex flex-col overflow-hidden rounded border",
        isKeep && "border-green-500/50 ring-1 ring-green-500/30",
        isSelected && "border-destructive ring-1 ring-destructive/30",
      )}
    >
      <div className="aspect-square overflow-hidden bg-muted">
        {broken ? (
          <div className="w-full h-full grid place-items-center text-muted-foreground/70 text-[10px] gap-1 flex flex-col">
            <ImageOff className="size-4" />
            <span>缩略图不可用</span>
          </div>
        ) : (
          <img
            src={api.datasetThumbUrl(path, 128)}
            alt=""
            className="h-full w-full object-cover"
            onError={() => setBroken(true)}
          />
        )}
      </div>
      <div className="flex items-center gap-1 px-1.5 py-1">
        <input
          type="checkbox"
          checked={isSelected}
          onChange={onToggle}
          className="size-3"
          disabled={isKeep}
          title={isKeep ? "建议保留这一张" : "勾选后会一并删除"}
        />
        <span className="flex-1 truncate text-[10px]" title={path}>
          {path.split(/[/\\]/).pop()}
        </span>
      </div>
      {isKeep && (
        <div className="absolute right-1 top-1 rounded bg-green-600/80 px-1 py-0.5 text-[9px] font-medium text-white">
          保留
        </div>
      )}
      {!isKeep && distance != null && (
        <div
          className="absolute right-1 bottom-7 rounded bg-background/70 px-1 py-0.5 text-[9px] font-mono text-foreground border border-border/60"
          title={`与保留图的 Hamming 距离: ${distance}`}
        >
          d={distance}
        </div>
      )}
    </div>
  )
}

function hammingDistance(a: string, b: string): number {
  if (!a || !b) return Number.POSITIVE_INFINITY
  const len = Math.min(a.length, b.length)
  let dist = 0
  for (let i = 0; i < len; i++) {
    const x = parseInt(a[i] ?? "0", 16)
    const y = parseInt(b[i] ?? "0", 16)
    if (Number.isNaN(x) || Number.isNaN(y)) continue
    let n = x ^ y
    while (n) {
      dist += n & 1
      n >>= 1
    }
  }
  return dist
}
