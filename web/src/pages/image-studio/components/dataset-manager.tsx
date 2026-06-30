import { useEffect, useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { FolderOpen, FolderPlus, Grid3x3, LayoutList } from "lucide-react"
import { toast } from "sonner"
import { datasetList, datasetCreate, datasetDelete, datasetUpdateMeta } from "@/lib/api"
import type { DatasetInfo } from "@/lib/api"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import {
  ContextMenu,
  ContextMenuTrigger,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
} from "@/components/ui/context-menu"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

interface DatasetManagerProps {
  onOpen: (path: string) => void
}

export function DatasetManager({ onOpen }: DatasetManagerProps) {
  const queryClient = useQueryClient()
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid")
  const [showCreate, setShowCreate] = useState(false)
  // Pending delete: pinning the target dataset opens the AlertDialog;
  // confirming fires the mutation, cancelling clears the pin.
  const [pendingDelete, setPendingDelete] = useState<DatasetInfo | null>(null)
  // Pending edit-meta target: opening the dialog displays its current
  // meta and lets the user save changes via PUT /datasets/:name/meta.
  const [pendingEdit, setPendingEdit] = useState<DatasetInfo | null>(null)

  const datasetsQuery = useQuery({
    queryKey: ["image-studio-datasets"],
    queryFn: datasetList,
    staleTime: 5_000,
  })

  const createMutation = useMutation({
    mutationFn: datasetCreate,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["image-studio-datasets"] })
      setShowCreate(false)
      onOpen(data.path)
      toast.success("数据集已创建", { description: data.path })
    },
    onError: (e) => {
      toast.error("创建失败", {
        description: e instanceof Error ? e.message : String(e),
      })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: datasetDelete,
    onSuccess: (_, name) => {
      queryClient.invalidateQueries({ queryKey: ["image-studio-datasets"] })
      toast.success(`数据集 "${name}" 已删除`)
    },
    onError: (e) => {
      toast.error("删除失败", {
        description: e instanceof Error ? e.message : String(e),
      })
    },
  })

  const updateMetaMutation = useMutation({
    mutationFn: ({
      name,
      body,
    }: {
      name: string
      body: { description?: string; targetResolution?: string; triggerWord?: string }
    }) => datasetUpdateMeta(name, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["image-studio-datasets"] })
      setPendingEdit(null)
      toast.success("信息已更新")
    },
    onError: (e) => {
      toast.error("更新失败", {
        description: e instanceof Error ? e.message : String(e),
      })
    },
  })

  const datasets = datasetsQuery.data?.datasets ?? []

  const handleContextAction = (action: string, ds: DatasetInfo) => {
    switch (action) {
      case "open":
        onOpen(ds.path)
        break
      case "edit":
        setPendingEdit(ds)
        break
      case "copy-path":
        navigator.clipboard
          .writeText(ds.path)
          .then(() => toast.success("路径已复制到剪贴板"))
          .catch(() => toast.error("复制失败"))
        break
      case "delete":
        setPendingDelete(ds)
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
          <Button
            type="button"
            onClick={() => setShowCreate(true)}
            size="sm"
          >
            <FolderPlus className="size-3.5" /> 新建
          </Button>
        </div>
      </div>

      {showCreate && (
        <CreateDatasetDialog
          onClose={() => setShowCreate(false)}
          onCreate={(data) => createMutation.mutate(data)}
          loading={createMutation.isPending}
        />
      )}

      <EditDatasetDialog
        dataset={pendingEdit}
        onClose={() => setPendingEdit(null)}
        onSave={(name, body) => updateMetaMutation.mutate({ name, body })}
        loading={updateMetaMutation.isPending}
      />

      <div className="flex-1 overflow-y-auto p-4">
        {datasetsQuery.isLoading && (
          <p className="text-sm text-muted-foreground">加载中…</p>
        )}
        {datasets.length === 0 && !datasetsQuery.isLoading && (
          <div className="flex flex-col items-center justify-center h-48 gap-3 text-muted-foreground">
            <FolderOpen className="size-10 opacity-40" />
            <p className="text-sm">暂无数据集，点击「新建」创建第一个</p>
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

      <AlertDialog
        open={!!pendingDelete}
        onOpenChange={(open) => !open && setPendingDelete(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>删除数据集</AlertDialogTitle>
            <AlertDialogDescription>
              将永久移除数据集{" "}
              <code className="font-mono">{pendingDelete?.name}</code>
              {" "}及其所有图片和标注。此操作无法撤销。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={(e) => {
                e.preventDefault()
                if (pendingDelete) {
                  deleteMutation.mutate(pendingDelete.name)
                  setPendingDelete(null)
                }
              }}
            >
              确认删除
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
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

  const submit = () => {
    if (!name.trim() || loading) return
    onCreate({
      name: name.trim(),
      description: description || undefined,
      targetResolution: resolution || undefined,
      triggerWord: trigger || undefined,
    })
  }

  return (
    <Dialog open onOpenChange={(next) => !next && onClose()}>
      <DialogContent className="sm:max-w-[420px]">
        <DialogHeader>
          <DialogTitle>新建数据集</DialogTitle>
          <DialogDescription>
            会在工作区下创建同名目录，meta 信息保存到 dataset.toml。
          </DialogDescription>
        </DialogHeader>
        <form
          onSubmit={(e) => {
            e.preventDefault()
            submit()
          }}
          className="space-y-3"
        >
          <div className="space-y-1.5">
            <Label className="text-[11px]">名称 <span className="text-destructive">*</span></Label>
            <Input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="my_character"
            />
          </div>
          <div className="space-y-1.5">
            <Label className="text-[11px]">描述</Label>
            <Input
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="角色 / 风格 / 用途备注"
            />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-1.5">
              <Label className="text-[11px]">目标分辨率</Label>
              <Input
                value={resolution}
                onChange={(e) => setResolution(e.target.value)}
                placeholder="1024x1024"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-[11px]">触发词</Label>
              <Input
                value={trigger}
                onChange={(e) => setTrigger(e.target.value)}
                placeholder="ohwx, sks, ..."
              />
            </div>
          </div>
        </form>
        <DialogFooter>
          <Button variant="outline" size="sm" onClick={onClose}>
            取消
          </Button>
          <Button
            size="sm"
            onClick={submit}
            disabled={!name.trim() || loading}
          >
            {loading ? "创建中…" : "创建"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function EditDatasetDialog({
  dataset,
  onClose,
  onSave,
  loading,
}: {
  dataset: DatasetInfo | null
  onClose: () => void
  onSave: (
    name: string,
    body: { description?: string; targetResolution?: string; triggerWord?: string },
  ) => void
  loading: boolean
}) {
  const [description, setDescription] = useState("")
  const [resolution, setResolution] = useState("")
  const [trigger, setTrigger] = useState("")

  // Hydrate fields from the selected dataset every time the dialog opens.
  useEffect(() => {
    if (!dataset) return
    setDescription(dataset.meta.description ?? "")
    setResolution(dataset.meta.targetResolution ?? "")
    setTrigger(dataset.meta.triggerWord ?? "")
  }, [dataset])

  if (!dataset) return null

  const submit = () => {
    if (loading) return
    onSave(dataset.name, {
      description: description.trim() || undefined,
      targetResolution: resolution.trim() || undefined,
      triggerWord: trigger.trim() || undefined,
    })
  }

  return (
    <Dialog open onOpenChange={(next) => !next && onClose()}>
      <DialogContent className="sm:max-w-[420px]">
        <DialogHeader>
          <DialogTitle>编辑数据集信息</DialogTitle>
          <DialogDescription className="font-mono break-all">
            {dataset.path}
          </DialogDescription>
        </DialogHeader>
        <form
          onSubmit={(e) => {
            e.preventDefault()
            submit()
          }}
          className="space-y-3"
        >
          <div className="space-y-1.5">
            <Label className="text-[11px]">描述</Label>
            <Input
              autoFocus
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="角色 / 风格 / 用途备注"
            />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-1.5">
              <Label className="text-[11px]">目标分辨率</Label>
              <Input
                value={resolution}
                onChange={(e) => setResolution(e.target.value)}
                placeholder="1024x1024"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-[11px]">触发词</Label>
              <Input
                value={trigger}
                onChange={(e) => setTrigger(e.target.value)}
                placeholder="ohwx, sks, ..."
              />
            </div>
          </div>
        </form>
        <DialogFooter>
          <Button variant="outline" size="sm" onClick={onClose}>
            取消
          </Button>
          <Button size="sm" onClick={submit} disabled={loading}>
            {loading ? "保存中…" : "保存"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
