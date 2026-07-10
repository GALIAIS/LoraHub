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
import {
  CurateAutoRotateTool,
  CurateBatchResizeTool,
  CurateQuarantineTool,
  CurateRestoreBackupTool,
} from "./components/stages/curate-tools"
import { CreateDatasetDialog } from "./components/create-dataset-dialog"
import { ToolsGrid } from "./components/tools-grid"
import { LibraryPage } from "./components/library/library-page"
import { LoraTestPage } from "./lora-test"
import type { StageId } from "./components/stage-stepper"

export { ImageStudioPage }

// stage 参数允许的取值；除了 5 个核心 stage，还有两个虚拟 stage：
//  - "tools"   — 全部工具广场
//  - "library" — 跨数据集的工具库（标签词典 / 触发词）
type StageOrTools = StageId | "tools" | "library"
type StageRoute = StageOrTools | "lora-test"

function isDatasetStage(stage: StageRoute): stage is StageId {
  return stage !== "tools" && stage !== "library"
    && stage !== "lora-test"
}

function ImageStudioPage() {
  const [params, setParams] = useSearchParams()
  const datasetPath = params.get("path") || ""
  const stageParam = (params.get("stage") || (datasetPath ? "curate" : "tools")) as StageRoute
  const [showCreate, setShowCreate] = useState(false)
  const queryClient = useQueryClient()

  const createMutation = useMutation({
    mutationFn: datasetCreate,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["image-studio-datasets"] })
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
    next.delete("page")
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
            stage={stageParam === "lora-test" ? "tools" : stageParam}
            onSelectDataset={selectDataset}
            onSelectStage={selectStage}
            onCreateDataset={() => setShowCreate(true)}
          />
        }
      >
        <div className="flex-1 min-h-0 overflow-hidden">
          {stageParam === "lora-test" ? (
            <LoraTestPage />
          ) : stageParam === "tools" ? (
            <ToolsGrid datasetPath={datasetPath} />
          ) : stageParam === "library" ? (
            <LibraryPage />
          ) : isDatasetStage(stageParam) && !datasetPath ? (
            <EmptyState onCreateDataset={() => setShowCreate(true)} />
          ) : isDatasetStage(stageParam) && datasetPath ? (
            <DatasetWorkspace stage={stageParam} datasetPath={datasetPath} />
          ) : null}
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
    return <CurateStage datasetPath={datasetPath} />
  }
  if (stage === "annotate") {
    return <AnnotateStage datasetPath={datasetPath} />
  }
  return <StagePanel stage={stage} datasetPath={datasetPath} />
}

function CurateStage({ datasetPath }: { datasetPath: string }) {
  const [params] = useSearchParams()
  const tool = params.get("tool")
  if (tool === "curate-auto-rotate") {
    return <CurateAutoRotateTool datasetPath={datasetPath} />
  }
  if (tool === "curate-batch-resize") {
    return <CurateBatchResizeTool datasetPath={datasetPath} />
  }
  if (tool === "curate-quarantine") {
    return <CurateQuarantineTool datasetPath={datasetPath} />
  }
  if (tool === "curate-restore-backup") {
    return <CurateRestoreBackupTool datasetPath={datasetPath} />
  }
  return <DatasetDetail />
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
