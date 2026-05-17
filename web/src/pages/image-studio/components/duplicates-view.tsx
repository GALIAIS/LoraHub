import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  imageStudioDedupeScan,
  imageStudioDedupeClusters,
  imageStudioBatchDelete,
} from "@/lib/api"
import type { DedupeCluster } from "@/lib/api"

interface DuplicatesViewProps {
  path: string
  recursive: boolean
}

export function DuplicatesView({ path, recursive }: DuplicatesViewProps) {
  const queryClient = useQueryClient()
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set())

  const scanMutation = useMutation({
    mutationFn: () => imageStudioDedupeScan({ path, recursive, algo: "phash64" }),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["image-studio", "clusters"] }),
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
    },
  })

  const clusters = clustersQuery.data?.clusters ?? []

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
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => scanMutation.mutate()}
          disabled={scanMutation.isPending}
          className="rounded bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-50"
        >
          {scanMutation.isPending ? "扫描中..." : "扫描重复图片"}
        </button>
        {clusters.length > 0 && (
          <>
            <button
              type="button"
              onClick={selectAllSuggested}
              className="rounded border px-2 py-1 text-xs hover:bg-muted"
            >
              全选建议删除
            </button>
            <span className="text-xs text-muted-foreground">
              {clusters.length} 个聚类, 已选 {selectedPaths.size} 张
            </span>
            {selectedPaths.size > 0 && (
              <button
                type="button"
                onClick={() => deleteMutation.mutate()}
                disabled={deleteMutation.isPending}
                className="rounded bg-destructive px-3 py-1.5 text-xs font-medium text-destructive-foreground disabled:opacity-50"
              >
                删除选中 ({selectedPaths.size})
              </button>
            )}
          </>
        )}
      </div>

      <div className="text-xs text-muted-foreground">
        <span className="font-medium">L1 pHash</span> 感知哈希去重
        <span className="ml-3 opacity-60">(L2 AI 语义层 — 即将推出)</span>
      </div>

      {clustersQuery.isLoading && (
        <p className="text-sm text-muted-foreground">加载聚类中...</p>
      )}

      {clusters.length === 0 && !clustersQuery.isLoading && (
        <p className="text-sm text-muted-foreground">
          未发现重复聚类。点击&quot;扫描重复图片&quot;开始分析。
        </p>
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

  const selectAllInCluster = () => {
    for (const m of cluster.members) {
      if (m.path !== cluster.suggestedKeep && !selectedPaths.has(m.path)) {
        onToggle(m.path)
      }
    }
  }

  return (
    <div className="rounded-md border">
      <div className="flex w-full items-center gap-2 px-3 py-2 text-sm">
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-2 flex-1 text-left hover:bg-muted/50 rounded px-1 -mx-1"
        >
          <span className="font-medium">{cluster.id}</span>
          <span className="text-xs text-muted-foreground">
            {cluster.members.length} 张图片
          </span>
          <span className="ml-auto text-xs">{expanded ? "▼" : "▶"}</span>
        </button>
        <button
          type="button"
          onClick={selectAllInCluster}
          className="rounded px-1.5 py-0.5 text-[10px] border hover:bg-muted"
        >
          选择重复
        </button>
      </div>
      {expanded && (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(120px,1fr))] gap-2 border-t p-3">
          {cluster.members.map((m) => {
            const isKeep = m.path === cluster.suggestedKeep
            const isSelected = selectedPaths.has(m.path)
            return (
              <div
                key={m.path}
                className={`relative flex flex-col overflow-hidden rounded border ${
                  isKeep ? "border-green-500/50 ring-1 ring-green-500/30" : ""
                } ${isSelected ? "border-destructive ring-1 ring-destructive/30" : ""}`}
              >
                <div className="aspect-square overflow-hidden bg-muted">
                  <img
                    src={`/api/datasets/thumb?path=${encodeURIComponent(m.path)}&size=128`}
                    alt=""
                    className="h-full w-full object-cover"
                  />
                </div>
                <div className="flex items-center gap-1 px-1.5 py-1">
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => onToggle(m.path)}
                    className="size-3"
                  />
                  <span className="flex-1 truncate text-[10px]">
                    {m.path.split(/[/\\]/).pop()}
                  </span>
                </div>
                {isKeep && (
                  <div className="absolute right-1 top-1 rounded bg-green-600/80 px-1 py-0.5 text-[9px] font-medium text-white">
                    保留
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
