import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { FolderOpen, FolderPlus, Grid3x3, LayoutList } from "lucide-react"
import { datasetList, datasetCreate, datasetDelete } from "@/lib/api"
import type { DatasetInfo } from "@/lib/api"
import {
  ContextMenu,
  ContextMenuTrigger,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
} from "@/components/ui/context-menu"

interface DatasetManagerProps {
  onOpen: (path: string) => void
}

export function DatasetManager({ onOpen }: DatasetManagerProps) {
  const queryClient = useQueryClient()
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid")
  const [showCreate, setShowCreate] = useState(false)

  const datasetsQuery = useQuery({
    queryKey: ["datasets"],
    queryFn: datasetList,
  })

  const createMutation = useMutation({
    mutationFn: datasetCreate,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] })
      setShowCreate(false)
      onOpen(data.path)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: datasetDelete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] })
    },
  })

  const datasets = datasetsQuery.data?.datasets ?? []

  const handleContextAction = (action: string, ds: DatasetInfo) => {
    switch (action) {
      case "open":
        onOpen(ds.path)
        break
      case "copy-path":
        navigator.clipboard.writeText(ds.path)
        break
      case "delete":
        if (confirm(`确定要删除数据集 "${ds.name}" 吗？`)) {
          deleteMutation.mutate(ds.name)
        }
        break
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-3 border-b px-4 py-3">
        <h1 className="text-base font-semibold">图像工作台</h1>
        <span className="text-xs text-muted-foreground">
          {datasets.length} 个数据集
        </span>
        <div className="ml-auto flex items-center gap-2">
          <div className="flex rounded border text-xs">
            <button
              type="button"
              onClick={() => setViewMode("grid")}
              className={`px-2 py-1 ${viewMode === "grid" ? "bg-muted font-medium" : ""}`}
              title="图集视图"
            >
              <Grid3x3 className="size-3.5" />
            </button>
            <button
              type="button"
              onClick={() => setViewMode("list")}
              className={`px-2 py-1 ${viewMode === "list" ? "bg-muted font-medium" : ""}`}
              title="列表视图"
            >
              <LayoutList className="size-3.5" />
            </button>
          </div>
          <button
            type="button"
            onClick={() => setShowCreate(true)}
            className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90"
          >
            <FolderPlus className="size-3.5" /> 新建
          </button>
        </div>
      </div>

      {showCreate && (
        <CreateDatasetDialog
          onClose={() => setShowCreate(false)}
          onCreate={(data) => createMutation.mutate(data)}
          loading={createMutation.isPending}
        />
      )}

      <div className="flex-1 overflow-y-auto p-4">
        {datasetsQuery.isLoading && (
          <p className="text-sm text-muted-foreground">加载中...</p>
        )}
        {datasets.length === 0 && !datasetsQuery.isLoading && (
          <div className="flex flex-col items-center justify-center h-48 gap-3 text-muted-foreground">
            <FolderOpen className="size-10 opacity-40" />
            <p className="text-sm">暂无数据集，点击&quot;新建&quot;创建第一个</p>
          </div>
        )}

        {viewMode === "grid" ? (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(200px,1fr))] gap-4">
            {datasets.map((ds) => (
              <DatasetGridCard
                key={ds.name}
                dataset={ds}
                onClick={() => onOpen(ds.path)}
                onContextAction={handleContextAction}
              />
            ))}
          </div>
        ) : (
          <div className="flex flex-col gap-1">
            {datasets.map((ds) => (
              <DatasetListRow
                key={ds.name}
                dataset={ds}
                onClick={() => onOpen(ds.path)}
                onContextAction={handleContextAction}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function DatasetGridCard({
  dataset,
  onClick,
  onContextAction,
}: {
  dataset: DatasetInfo
  onClick: () => void
  onContextAction: (action: string, ds: DatasetInfo) => void
}) {
  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <button
          type="button"
          onClick={onClick}
          className="group flex flex-col overflow-hidden rounded-lg border transition-colors hover:border-primary/50 hover:shadow-sm text-left"
        >
          <div className="aspect-[4/3] w-full overflow-hidden bg-muted">
            {dataset.coverUrl ? (
              <img
                src={dataset.coverUrl}
                alt=""
                className="h-full w-full object-cover transition-transform group-hover:scale-105"
              />
            ) : (
              <div className="flex h-full items-center justify-center">
                <FolderOpen className="size-8 text-muted-foreground/40" />
              </div>
            )}
          </div>
          <div className="px-3 py-2">
            <p className="text-sm font-medium truncate">{dataset.name}</p>
            <p className="text-xs text-muted-foreground">
              {dataset.imageCount} 张图片
              {dataset.meta.triggerWord && (
                <span className="ml-2 rounded bg-muted px-1 py-0.5 text-[10px]">
                  {dataset.meta.triggerWord}
                </span>
              )}
            </p>
          </div>
        </button>
      </ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem onClick={() => onContextAction("open", dataset)}>
          打开工作台
        </ContextMenuItem>
        <ContextMenuItem onClick={() => onContextAction("edit", dataset)}>
          编辑信息
        </ContextMenuItem>
        <ContextMenuItem onClick={() => onContextAction("copy-path", dataset)}>
          复制路径
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem
          onClick={() => onContextAction("delete", dataset)}
          className="text-destructive focus:text-destructive"
        >
          删除数据集
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  )
}

function DatasetListRow({
  dataset,
  onClick,
  onContextAction,
}: {
  dataset: DatasetInfo
  onClick: () => void
  onContextAction: (action: string, ds: DatasetInfo) => void
}) {
  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <button
          type="button"
          onClick={onClick}
          className="flex items-center gap-3 rounded-md px-3 py-2 text-left transition-colors hover:bg-muted/50"
        >
          <div className="size-10 shrink-0 overflow-hidden rounded bg-muted">
            {dataset.coverUrl ? (
              <img src={dataset.coverUrl} alt="" className="h-full w-full object-cover" />
            ) : (
              <div className="flex h-full items-center justify-center">
                <FolderOpen className="size-4 text-muted-foreground/40" />
              </div>
            )}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium truncate">{dataset.name}</p>
            <p className="text-xs text-muted-foreground">
              {dataset.imageCount} 张图片
              {dataset.meta.description && ` · ${dataset.meta.description}`}
            </p>
          </div>
          {dataset.meta.triggerWord && (
            <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
              {dataset.meta.triggerWord}
            </span>
          )}
        </button>
      </ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem onClick={() => onContextAction("open", dataset)}>
          打开工作台
        </ContextMenuItem>
        <ContextMenuItem onClick={() => onContextAction("edit", dataset)}>
          编辑信息
        </ContextMenuItem>
        <ContextMenuItem onClick={() => onContextAction("copy-path", dataset)}>
          复制路径
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem
          onClick={() => onContextAction("delete", dataset)}
          className="text-destructive focus:text-destructive"
        >
          删除数据集
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  )
}

function CreateDatasetDialog({
  onClose,
  onCreate,
  loading,
}: {
  onClose: () => void
  onCreate: (data: {
    name: string
    description?: string
    targetResolution?: string
    triggerWord?: string
  }) => void
  loading: boolean
}) {
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [resolution, setResolution] = useState("")
  const [trigger, setTrigger] = useState("")

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div className="w-96 rounded-lg border bg-popover p-5 shadow-lg" onClick={(e) => e.stopPropagation()}>
        <h3 className="text-sm font-semibold mb-4">新建数据集</h3>
        <div className="flex flex-col gap-3">
          <label className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">名称 *</span>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="my_character"
              className="rounded-md border bg-background px-3 py-1.5 text-sm outline-none focus:border-ring focus:ring-1 focus:ring-ring/30"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">描述</span>
            <input
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="角色/风格描述"
              className="rounded-md border bg-background px-3 py-1.5 text-sm outline-none focus:border-ring focus:ring-1 focus:ring-ring/30"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">目标分辨率</span>
            <input
              type="text"
              value={resolution}
              onChange={(e) => setResolution(e.target.value)}
              placeholder="512x512 / 1024x1024"
              className="rounded-md border bg-background px-3 py-1.5 text-sm outline-none focus:border-ring focus:ring-1 focus:ring-ring/30"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">触发词</span>
            <input
              type="text"
              value={trigger}
              onChange={(e) => setTrigger(e.target.value)}
              placeholder="ohwx, sks, ..."
              className="rounded-md border bg-background px-3 py-1.5 text-sm outline-none focus:border-ring focus:ring-1 focus:ring-ring/30"
            />
          </label>
        </div>
        <div className="flex justify-end gap-2 mt-5">
          <button type="button" onClick={onClose} className="rounded-md px-3 py-1.5 text-xs hover:bg-muted">
            取消
          </button>
          <button
            type="button"
            disabled={!name.trim() || loading}
            onClick={() =>
              onCreate({
                name: name.trim(),
                description: description || undefined,
                targetResolution: resolution || undefined,
                triggerWord: trigger || undefined,
              })
            }
            className="rounded-md bg-primary px-4 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {loading ? "创建中..." : "创建"}
          </button>
        </div>
      </div>
    </div>
  )
}