import { useState } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  ChevronLeft,
  FolderOpen,
  FolderPlus,
  Images,
  Import,
  ClipboardCheck,
  Scissors,
  Tags,
  PackageCheck,
  Trash2,
  LayoutGrid,
  Library,
} from "lucide-react"
import { toast } from "sonner"
import { datasetList, datasetDelete } from "@/lib/api"
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
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { StageId } from "./stage-stepper"

// 除 stepper 的 5 个 stage 外,还有两个虚拟 stage:
//   - "tools"   — 全部工具广场
//   - "library" — 跨数据集的工具库
type SidebarStage = StageId | "tools" | "library"

const STAGES: { id: StageId; label: string; icon: typeof Import }[] = [
  { id: "intake", label: "导入", icon: Import },
  { id: "audit", label: "审计", icon: ClipboardCheck },
  { id: "curate", label: "整理", icon: Scissors },
  { id: "annotate", label: "标注", icon: Tags },
  { id: "ship", label: "输出", icon: PackageCheck },
]

interface StudioSidebarProps {
  datasetPath: string
  stage: SidebarStage
  onSelectDataset: (path: string) => void
  onSelectStage: (stage: SidebarStage) => void
  onCreateDataset: () => void
}

export function StudioSidebar({
  datasetPath,
  stage,
  onSelectDataset,
  onSelectStage,
  onCreateDataset,
}: StudioSidebarProps) {
  const queryClient = useQueryClient()
  const [collapsed, setCollapsed] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<DatasetInfo | null>(null)

  const datasetsQuery = useQuery({
    queryKey: ["image-studio-datasets"],
    queryFn: datasetList,
    staleTime: 5_000,
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

  const datasets = datasetsQuery.data?.datasets ?? []
  const navButtonClass =
    "w-full justify-start text-left font-normal data-[active=true]:border-sidebar-primary/35 data-[active=true]:bg-sidebar-accent data-[active=true]:text-sidebar-accent-foreground"
  const iconButtonClass =
    "text-muted-foreground data-[active=true]:border-sidebar-primary/35 data-[active=true]:bg-sidebar-accent data-[active=true]:text-sidebar-accent-foreground"

  return (
    <>
      <div
        className={cn(
          "flex h-full flex-col bg-sidebar text-sidebar-foreground transition-[width] duration-150 ease-out overflow-hidden shrink-0",
          collapsed ? "w-12" : "w-56",
        )}
      >
        {/* Header */}
        <div
          className={cn(
            "flex border-b px-2 py-2",
            collapsed ? "flex-col items-center gap-2" : "items-center gap-1",
          )}
        >
          <Button
            type="button"
            onClick={() => setCollapsed(!collapsed)}
            variant="ghost"
            size="icon-xs"
            className="shrink-0"
            title={collapsed ? "展开侧边栏" : "折叠侧边栏"}
          >
            <ChevronLeft
              className={cn(
                "size-3.5 transition-transform",
                collapsed && "rotate-180",
              )}
            />
          </Button>
          {!collapsed && (
            <Button
              type="button"
              onClick={onCreateDataset}
              variant="ghost"
              size="sm"
              className="flex-1 justify-start"
            >
              <FolderPlus className="size-3.5" />
              新建
            </Button>
          )}
          {collapsed && (
            <Tooltip>
              <TooltipTrigger
                render={
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon-sm"
                  />
                }
                onClick={onCreateDataset}
              >
                <FolderPlus className="size-4" />
              </TooltipTrigger>
              <TooltipContent side="right">新建数据集</TooltipContent>
            </Tooltip>
          )}
        </div>

        <div className="flex-1 overflow-y-auto overflow-x-hidden py-1">
          {!collapsed && (
            <div className="px-2 pb-1">
              <p className="px-2 py-1 text-[11px] font-medium text-muted-foreground">数据集</p>
            </div>
          )}
          <nav className="flex flex-col gap-0.5 px-1.5">
            {datasets.map((ds) =>
              collapsed ? (
                <Tooltip key={ds.name}>
                  <TooltipTrigger
                    render={
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-sm"
                        data-active={ds.path === datasetPath ? "true" : undefined}
                        className={iconButtonClass}
                      />
                    }
                    onClick={() => onSelectDataset(ds.path)}
                  >
                    <FolderOpen className="size-4" />
                  </TooltipTrigger>
                  <TooltipContent side="right">
                    {ds.name} ({ds.imageCount})
                  </TooltipContent>
                </Tooltip>
              ) : (
                <div
                  key={ds.name}
                  className={cn(
                    "group/ds relative flex items-center rounded-md text-sm",
                    ds.path === datasetPath
                      ? "bg-accent text-accent-foreground"
                      : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
                  )}
                >
                  <Button
                    type="button"
                    onClick={() => onSelectDataset(ds.path)}
                    variant="ghost"
                    size="sm"
                    className={cn(
                      "min-w-0 flex-1 justify-start gap-2 text-left font-normal",
                      ds.path === datasetPath && "font-medium",
                    )}
                  >
                    <FolderOpen className="size-3.5 shrink-0" />
                    <span className="truncate flex-1">{ds.name}</span>
                    {ds.imageCount > 0 && (
                      <span className="text-[10px] tabular-nums opacity-60 shrink-0">
                        {ds.imageCount}
                      </span>
                    )}
                  </Button>
                  <Tooltip>
                    <TooltipTrigger
                      render={
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon-xs"
                          className="hidden text-muted-foreground hover:text-destructive group-hover/ds:inline-flex"
                          aria-label={`删除数据集 ${ds.name}`}
                        />
                      }
                      onClick={(e) => {
                        e.stopPropagation()
                        setPendingDelete(ds)
                      }}
                    >
                      <Trash2 className="size-3.5" />
                    </TooltipTrigger>
                    <TooltipContent side="right">删除数据集</TooltipContent>
                  </Tooltip>
                </div>
              ),
            )}
            {datasets.length === 0 && !datasetsQuery.isLoading && !collapsed && (
              <p className="px-2 py-3 text-xs text-muted-foreground text-center">
                暂无数据集
              </p>
            )}
          </nav>

          {/* Stage navigation */}
          {datasetPath && (
            <>
              <div className="mx-2 my-2 border-t" />
              {!collapsed && (
                <div className="px-2 pb-1">
                  <p className="px-2 py-1 text-[11px] font-medium text-muted-foreground">处理阶段</p>
                </div>
              )}
              <nav className="flex flex-col gap-0.5 px-1.5">
                {STAGES.map((s) =>
                  collapsed ? (
                    <Tooltip key={s.id}>
                      <TooltipTrigger
                        render={
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon-sm"
                            data-active={s.id === stage ? "true" : undefined}
                            className={iconButtonClass}
                          />
                        }
                        onClick={() => onSelectStage(s.id)}
                      >
                        <s.icon className="size-4" />
                      </TooltipTrigger>
                      <TooltipContent side="right">{s.label}</TooltipContent>
                    </Tooltip>
                  ) : (
                    <Button
                      key={s.id}
                      type="button"
                      onClick={() => onSelectStage(s.id)}
                      variant="ghost"
                      size="sm"
                      data-active={s.id === stage ? "true" : undefined}
                      className={navButtonClass}
                    >
                      <s.icon className="size-3.5 shrink-0" />
                      <span>{s.label}</span>
                    </Button>
                  ),
                )}
              </nav>
            </>
          )}

          <div className="mx-2 my-2 border-t" />
          {!collapsed && (
            <div className="px-2 pb-1">
              <p className="px-2 py-1 text-[11px] font-medium text-muted-foreground">辅助</p>
            </div>
          )}
          <div className="px-1.5 pb-1 space-y-0.5">
            {collapsed ? (
              <>
                <Tooltip>
                  <TooltipTrigger
                    render={
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-sm"
                        data-active={stage === "tools" ? "true" : undefined}
                        className={iconButtonClass}
                      />
                    }
                    onClick={() => onSelectStage("tools")}
                  >
                    <LayoutGrid className="size-4" />
                  </TooltipTrigger>
                  <TooltipContent side="right">工具目录</TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger
                    render={
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-sm"
                        data-active={stage === "library" ? "true" : undefined}
                        className={iconButtonClass}
                      />
                    }
                    onClick={() => onSelectStage("library")}
                  >
                    <Library className="size-4" />
                  </TooltipTrigger>
                  <TooltipContent side="right">工具库</TooltipContent>
                </Tooltip>
              </>
            ) : (
              <>
                <Button
                  type="button"
                  onClick={() => onSelectStage("tools")}
                  variant="ghost"
                  size="sm"
                  data-active={stage === "tools" ? "true" : undefined}
                  className={navButtonClass}
                >
                  <LayoutGrid className="size-3.5 shrink-0" />
                  <span>工具目录</span>
                </Button>
                <Button
                  type="button"
                  onClick={() => onSelectStage("library")}
                  variant="ghost"
                  size="sm"
                  data-active={stage === "library" ? "true" : undefined}
                  className={navButtonClass}
                >
                  <Library className="size-3.5 shrink-0" />
                  <span>工具库</span>
                </Button>
              </>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="border-t px-2 py-2">
          {collapsed ? (
            <Tooltip>
              <TooltipTrigger
                render={
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon-sm"
                    className="w-full text-muted-foreground"
                  />
                }
              >
                <Images className="size-4 text-muted-foreground" />
              </TooltipTrigger>
              <TooltipContent side="right">
                {datasets.length} 个数据集
              </TooltipContent>
            </Tooltip>
          ) : (
            <p className="flex items-center gap-1.5 px-2 text-[11px] text-muted-foreground">
              <Images className="size-3.5" />
              {datasets.length} 个数据集
            </p>
          )}
        </div>
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
    </>
  )
}
