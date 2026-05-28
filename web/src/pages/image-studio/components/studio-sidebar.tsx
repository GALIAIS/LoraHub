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
    queryKey: ["datasets"],
    queryFn: datasetList,
  })

  const deleteMutation = useMutation({
    mutationFn: datasetDelete,
    onSuccess: (_, name) => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] })
      toast.success(`数据集 "${name}" 已删除`)
    },
    onError: (e) => {
      toast.error("删除失败", {
        description: e instanceof Error ? e.message : String(e),
      })
    },
  })

  const datasets = datasetsQuery.data?.datasets ?? []

  return (
    <>
      <aside
        className={cn(
          "flex h-full flex-col border-r bg-sidebar text-sidebar-foreground transition-[width] duration-150 ease-out overflow-hidden shrink-0",
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
          <button
            type="button"
            onClick={() => setCollapsed(!collapsed)}
            className="rounded-md p-1.5 hover:bg-accent shrink-0"
            title={collapsed ? "展开侧边栏" : "折叠侧边栏"}
          >
            <ChevronLeft
              className={cn(
                "size-3.5 transition-transform",
                collapsed && "rotate-180",
              )}
            />
          </button>
          {!collapsed && (
            <button
              type="button"
              onClick={onCreateDataset}
              className="flex flex-1 items-center gap-1.5 rounded-md px-2 py-1.5 text-xs font-medium hover:bg-accent"
            >
              <FolderPlus className="size-3.5" />
              新建
            </button>
          )}
          {collapsed && (
            <Tooltip>
              <TooltipTrigger
                onClick={onCreateDataset}
                className="flex items-center justify-center rounded-md p-1.5 hover:bg-accent"
              >
                <FolderPlus className="size-4" />
              </TooltipTrigger>
              <TooltipContent side="right">新建数据集</TooltipContent>
            </Tooltip>
          )}
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto overflow-x-hidden py-1">
          {/* "全部工具" / "工具库" 两个虚拟 stage 入口 — 独立于数据集,常驻顶部。 */}
          <div className="px-1.5 pb-1 space-y-0.5">
            {collapsed ? (
              <>
                <Tooltip>
                  <TooltipTrigger
                    onClick={() => onSelectStage("tools")}
                    className={cn(
                      "flex w-full items-center justify-center rounded-md p-2",
                      stage === "tools"
                        ? "bg-primary/10 text-primary"
                        : "hover:bg-accent/50 text-muted-foreground",
                    )}
                  >
                    <LayoutGrid className="size-4" />
                  </TooltipTrigger>
                  <TooltipContent side="right">全部工具</TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger
                    onClick={() => onSelectStage("library")}
                    className={cn(
                      "flex w-full items-center justify-center rounded-md p-2",
                      stage === "library"
                        ? "bg-primary/10 text-primary"
                        : "hover:bg-accent/50 text-muted-foreground",
                    )}
                  >
                    <Library className="size-4" />
                  </TooltipTrigger>
                  <TooltipContent side="right">工具库</TooltipContent>
                </Tooltip>
              </>
            ) : (
              <>
                <button
                  type="button"
                  onClick={() => onSelectStage("tools")}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm",
                    stage === "tools"
                      ? "bg-primary/10 text-primary font-medium"
                      : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
                  )}
                >
                  <LayoutGrid className="size-3.5 shrink-0" />
                  <span>全部工具</span>
                </button>
                <button
                  type="button"
                  onClick={() => onSelectStage("library")}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm",
                    stage === "library"
                      ? "bg-primary/10 text-primary font-medium"
                      : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
                  )}
                >
                  <Library className="size-3.5 shrink-0" />
                  <span>工具库</span>
                </button>
              </>
            )}
          </div>
          <div className="mx-2 my-1 border-t" />

          {!collapsed && (
            <div className="px-2 pb-1">
              <p className="px-2 py-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                数据集
              </p>
            </div>
          )}
          <nav className="flex flex-col gap-0.5 px-1.5">
            {datasets.map((ds) =>
              collapsed ? (
                <Tooltip key={ds.name}>
                  <TooltipTrigger
                    onClick={() => onSelectDataset(ds.path)}
                    className={cn(
                      "flex items-center justify-center rounded-md p-2",
                      ds.path === datasetPath
                        ? "bg-accent text-accent-foreground"
                        : "hover:bg-accent/50",
                    )}
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
                  <button
                    type="button"
                    onClick={() => onSelectDataset(ds.path)}
                    className={cn(
                      "flex flex-1 min-w-0 items-center gap-2 px-2 py-1.5 text-left",
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
                  </button>
                  <Tooltip>
                    <TooltipTrigger
                      onClick={(e) => {
                        e.stopPropagation()
                        setPendingDelete(ds)
                      }}
                      className="hidden group-hover/ds:flex items-center justify-center px-1.5 py-1.5 text-muted-foreground hover:text-destructive shrink-0"
                      aria-label={`删除数据集 ${ds.name}`}
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
                  <p className="px-2 py-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                    处理阶段
                  </p>
                </div>
              )}
              <nav className="flex flex-col gap-0.5 px-1.5">
                {STAGES.map((s) =>
                  collapsed ? (
                    <Tooltip key={s.id}>
                      <TooltipTrigger
                        onClick={() => onSelectStage(s.id)}
                        className={cn(
                          "flex items-center justify-center rounded-md p-2",
                          s.id === stage
                            ? "bg-primary/10 text-primary"
                            : "hover:bg-accent/50 text-muted-foreground",
                        )}
                      >
                        <s.icon className="size-4" />
                      </TooltipTrigger>
                      <TooltipContent side="right">{s.label}</TooltipContent>
                    </Tooltip>
                  ) : (
                    <button
                      key={s.id}
                      type="button"
                      onClick={() => onSelectStage(s.id)}
                      className={cn(
                        "flex items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm",
                        s.id === stage
                          ? "bg-primary/10 text-primary font-medium"
                          : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
                      )}
                    >
                      <s.icon className="size-3.5 shrink-0" />
                      <span>{s.label}</span>
                    </button>
                  ),
                )}
              </nav>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="border-t px-2 py-2">
          {collapsed ? (
            <Tooltip>
              <TooltipTrigger className="flex w-full items-center justify-center">
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
      </aside>

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
