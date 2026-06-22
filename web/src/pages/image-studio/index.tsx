import { useState } from "react"
import { useSearchParams } from "react-router-dom"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { toast } from "sonner"
import { ImageIcon, Plus } from "lucide-react"
import { datasetCreate } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { WorkbenchSplitLayout } from "@/components/workbench-split-layout"
import { StudioSidebar } from "./components/studio-sidebar"
import { DatasetDetail } from "./components/dataset-detail"
import { IntakeStage } from "./components/stages/intake-stage"
import { AuditStage } from "./components/stages/audit-stage"
import { AnnotateStage } from "./components/stages/annotate-stage"
import { ShipStage } from "./components/stages/ship-stage"
import { CreateDatasetDialog } from "./components/create-dataset-dialog"
import { ToolsGrid } from "./components/tools-grid"
import { LibraryPage } from "./components/library/library-page"
import type { StageId } from "./components/stage-stepper"

export { ImageStudioPage }

// stage 参数允许的取值；除了 5 个核心 stage，还有两个虚拟 stage：
//  - "tools"   — 全部工具广场
//  - "library" — 跨数据集的工具库（标签词典 / 触发词 / Prompt 模板）
type StageOrTools = StageId | "tools" | "library"

const STAGE_META: Record<StageId, { label: string; hint: string }> = {
  intake: { label: "导入", hint: "补充来源图片，不离开当前数据集网格" },
  audit: { label: "审计", hint: "扫描问题后回到网格处理图片" },
  curate: { label: "整理", hint: "查看、筛选、批量处理图片" },
  annotate: { label: "标注", hint: "维护 caption、标签和触发词" },
  ship: { label: "输出", hint: "检查训练就绪状态并导出" },
}

function isDatasetStage(stage: StageOrTools): stage is StageId {
  return stage !== "tools" && stage !== "library"
}

function ImageStudioPage() {
  const [params, setParams] = useSearchParams()
  const datasetPath = params.get("path") || ""
  const stageParam = (params.get("stage") || (datasetPath ? "curate" : "tools")) as StageOrTools
  const [showCreate, setShowCreate] = useState(false)
  const queryClient = useQueryClient()

  const createMutation = useMutation({
    mutationFn: datasetCreate,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] })
      setShowCreate(false)
      const next = new URLSearchParams(params)
      next.set("path", data.path)
      next.set("stage", "intake")
      setParams(next)
      toast.success("数据集已创建", { description: data.path })
    },
    onError: (e) => {
      toast.error("创建失败", {
        description: e instanceof Error ? e.message : String(e),
      })
    },
  })

  const selectDataset = (path: string) => {
    const next = new URLSearchParams(params)
    next.set("path", path)
    const currentStage = next.get("stage")
    if (!currentStage || currentStage === "tools" || currentStage === "library") {
      next.set("stage", "curate")
    }
    setParams(next)
  }

  const selectStage = (stage: StageOrTools) => {
    const next = new URLSearchParams(params)
    next.set("stage", stage)
    next.delete("tool")
    setParams(next)
  }

  return (
    <>
      <WorkbenchSplitLayout
        sidebarWidth="max-content"
        asideClassName="bg-transparent"
        mobileSidebarTitle="图像工作台导航"
        mobileSidebarDescription="选择数据集、切换导入审计标注输出阶段或打开工具库。"
        sidebar={
          <StudioSidebar
            datasetPath={datasetPath}
            stage={stageParam}
            onSelectDataset={selectDataset}
            onSelectStage={selectStage}
            onCreateDataset={() => setShowCreate(true)}
          />
        }
      >
        <div className="flex-1 min-h-0 overflow-hidden">
          {stageParam === "tools" && (
            <ToolsGrid datasetPath={datasetPath} />
          )}
          {stageParam === "library" && <LibraryPage />}
          {isDatasetStage(stageParam) &&
            !datasetPath && (
              <EmptyState onCreateDataset={() => setShowCreate(true)} />
            )}
          {isDatasetStage(stageParam) &&
            datasetPath && (
              <DatasetWorkspace stage={stageParam} datasetPath={datasetPath} />
            )}
        </div>
      </WorkbenchSplitLayout>

      {showCreate && (
        <CreateDatasetDialog
          onClose={() => setShowCreate(false)}
          onCreate={(data) => createMutation.mutate(data)}
          loading={createMutation.isPending}
        />
      )}
    </>
  )
}

function DatasetWorkspace({
  stage,
  datasetPath,
}: {
  stage: StageId
  datasetPath: string
}) {
  if (stage === "curate") {
    return <DatasetDetail />
  }
  const meta = STAGE_META[stage]
  return (
    <div className="grid h-full min-h-0 grid-rows-[minmax(0,1fr)_minmax(240px,38vh)] 2xl:grid-cols-[minmax(0,1fr)_minmax(560px,640px)] 2xl:grid-rows-[minmax(0,1fr)]">
      <div className="min-h-0 min-w-0 overflow-hidden">
        <DatasetDetail />
      </div>
      <aside className="flex min-h-0 min-w-0 flex-col overflow-hidden border-t border-border/60 bg-background 2xl:border-l 2xl:border-t-0">
        <div className="shrink-0 border-b border-border/60 px-3 py-2">
          <div className="text-sm font-medium">{meta.label}</div>
          <div className="mt-0.5 text-[11px] text-muted-foreground">
            {meta.hint}
          </div>
        </div>
        <div className="min-h-0 flex-1 overflow-hidden">
          <StagePanel stage={stage} datasetPath={datasetPath} />
        </div>
      </aside>
    </div>
  )
}

function StagePanel({
  stage,
  datasetPath,
}: {
  stage: Exclude<StageId, "curate">
  datasetPath: string
}) {
  if (stage === "intake") return <IntakeStage datasetPath={datasetPath} />
  if (stage === "audit") return <AuditStage datasetPath={datasetPath} />
  if (stage === "annotate") return <AnnotateStage datasetPath={datasetPath} />
  return <ShipStage datasetPath={datasetPath} />
}

function EmptyState({ onCreateDataset }: { onCreateDataset: () => void }) {
  return (
    <div className="flex h-full items-center justify-center">
      <div className="flex flex-col items-center gap-4 text-center">
        <div className="grid size-16 place-items-center rounded-[6px] border border-border/60 bg-muted/35 shadow-[var(--panel-shadow)]">
          <ImageIcon className="size-7 text-muted-foreground/70" />
        </div>
        <div>
          <p className="text-sm font-medium">图像工作台</p>
          <p className="mt-1 text-xs text-muted-foreground">
            从侧边栏选择数据集，或创建新数据集开始工作
          </p>
        </div>
        <Button
          type="button"
          onClick={onCreateDataset}
          size="sm"
        >
          <Plus className="size-3.5" />
          新建数据集
        </Button>
      </div>
    </div>
  )
}
