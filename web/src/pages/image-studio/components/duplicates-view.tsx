import { useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { ImageOff, Loader2, RefreshCw, Trash2 } from "lucide-react"
import { toast } from "sonner"
import {
  api,
  imageStudioDedupeScan,
  imageStudioDedupeClusters,
  imageStudioBatchDelete,
} from "@/lib/api"
import type { DedupeCluster } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

interface DuplicatesViewProps {
  path: string
  recursive: boolean
}

export function DuplicatesView({ path, recursive }: DuplicatesViewProps) {
  const queryClient = useQueryClient()
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set())

  const scanMutation = useMutation({
    mutationFn: () => imageStudioDedupeScan({ path, recursive, algo: "phash64" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["image-studio", "clusters"] })
      toast.success("扫描完成")
    },
    onError: (e) => {
      toast.error("扫描失败", {
        description: e instanceof Error ? e.message : String(e),
      })
    },
  })

  const clustersQuery = useQuery({
    queryKey: ["image-studio", "clusters", path],
    queryFn: () => imageStudioDedupeClusters({ path, threshold: 10 }),
    enabled: !!path,
  })

  const deleteMutation = useMutation({
    mutationFn: () => imageStudioBatchDelete({ paths: Array.from(selectedPaths) }),
    onSuccess: () => {
      setSelectedPaths(new Set())
      queryClient.invalidateQueries({ queryKey: ["image-studio"] })
      toast.success("已删除选中图片")
    },
    onError: (e) => {
      toast.error("删除失败", {
        description: e instanceof Error ? e.message : String(e),
      })
    },
  })

  const clusters = clustersQuery.data?.clusters ?? []
  const totalDupes = clusters.reduce((sum, c) => sum + Math.max(0, c.members.length - 1), 0)

  const togglePath = (p: string) => {
    setSelectedPaths((prev) => {
      const next = new Set(prev)
      if (next.has(p)) next.delete(p)
      else next.add(p)
      return next
    })
  }

  const selectAllSuggested = () => {
    const paths = new Set<string>()
    for (const c of clusters) {
      for (const m of c.members) {
        if (m.path !== c.suggestedKeep) paths.add(m.path)
      }
    }
    setSelectedPaths(paths)
  }

  return (
    <div className="flex flex-1 flex-col overflow-y-auto p-4 gap-4">
      <div className="flex items-center gap-2 flex-wrap">
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
          {scanMutation.isPending ? "扫描中…" : "扫描重复图片"}
        </Button>
        {clusters.length > 0 && (
          <>
            <Button
              variant="outline"
              size="sm"
              onClick={selectAllSuggested}
              className="h-8 text-[11px]"
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
                className="h-8 ml-auto text-[11px]"
              >
                <Trash2 className="size-3" /> 删除选中 ({selectedPaths.size})
              </Button>
            )}
          </>
        )}
      </div>

      <div className="text-xs text-muted-foreground space-y-0.5">
        <div>
          <span className="font-medium text-foreground">L1 pHash</span> 感知哈希 ·
          阈值 10（Hamming 距离 ≤ 10 视为相似）
        </div>
        <div className="text-muted-foreground/70">
          每张缩略图右下角显示与「保留」张图的 Hamming 距离 · 越小越相似
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
            ? "未发现重复聚类。数据集已经很干净了。"
            : "尚未扫描。点上方按钮开始分析。"}
        </div>
      )}

      {clusters.map((cluster) => (
        <ClusterCard
          key={cluster.id}
          cluster={cluster}
          selectedPaths={selectedPaths}
          onToggle={togglePath}
        />
      ))}
    </div>
  )
}

function ClusterCard({
  cluster,
  selectedPaths,
  onToggle,
}: {
  cluster: DedupeCluster
  selectedPaths: Set<string>
  onToggle: (path: string) => void
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
          onClick={() => setExpanded(!expanded)}
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
            const distance = isKeep
              ? 0
              : keepMember
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
      className={`relative flex w-full max-w-full min-w-0 flex-col overflow-hidden rounded border ${
        isKeep ? "border-green-500/50 ring-1 ring-green-500/30" : ""
      } ${isSelected ? "border-destructive ring-1 ring-destructive/30" : ""}`}
    >
      <div className="aspect-square w-full overflow-hidden bg-muted">
        {broken ? (
          <div className="w-full h-full grid place-items-center text-muted-foreground/70 text-[10px] gap-1 flex-col flex">
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
      <div className="flex min-w-0 items-center gap-1 px-1.5 py-1">
        <input
          type="checkbox"
          checked={isSelected}
          onChange={onToggle}
          className="size-3 shrink-0"
          disabled={isKeep}
          title={isKeep ? "建议保留这一张" : "勾选后会一并删除"}
        />
        <span
          className="min-w-0 flex-1 truncate text-[10px]"
          title={path}
        >
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

/**
 * Hamming distance over two hex-encoded perceptual hashes. The server
 * returns hashes as plain hex strings (e.g. "ffd2c0e0a..."), so we
 * XOR each pair of hex digits and popcount the resulting nibble.
 *
 * Hashes of unequal length fall back to the prefix that overlaps —
 * this only happens if the dedupe scan was re-run with a different
 * algo halfway through, so a "best effort" distance is enough to
 * surface as a hint.
 */
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
