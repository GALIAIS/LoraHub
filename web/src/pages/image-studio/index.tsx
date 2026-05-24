import { useState } from "react"
import { useSearchParams } from "react-router-dom"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { toast } from "sonner"
import { datasetCreate } from "@/lib/api"
import { StudioSidebar } from "./components/studio-sidebar"
import { DatasetDetail } from "./components/dataset-detail"
import { IntakeStage } from "./components/stages/intake-stage"
import { AuditStage } from "./components/stages/audit-stage"
import { AnnotateStage } from "./components/stages/annotate-stage"
import { ShipStage } from "./components/stages/ship-stage"
import { CreateDatasetDialog } from "./components/create-dataset-dialog"
import type { StageId } from "./components/stage-stepper"

export { ImageStudioPage }

function ImageStudioPage() {
  const [params, setParams] = useSearchParams()
  const datasetPath = params.get("path") || ""
  const stageParam = (params.get("stage") || "audit") as StageId
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
    if (!next.get("stage")) next.set("stage", "audit")
    setParams(next)
  }

  const selectStage = (stage: StageId) => {
    const next = new URLSearchParams(params)
    next.set("stage", stage)
    setParams(next)
  }

  return (
    <div className="flex h-full min-h-0 w-full">
      <StudioSidebar
        datasetPath={datasetPath}
        stage={stageParam}
        onSelectDataset={selectDataset}
        onSelectStage={selectStage}
        onCreateDataset={() => setShowCreate(true)}
      />

      <main className="flex-1 min-w-0 flex flex-col min-h-0">
        {!datasetPath ? (
          <EmptyState onCreateDataset={() => setShowCreate(true)} />
        ) : (
          <div className="flex-1 min-h-0 overflow-hidden">
            {stageParam === "intake" && (
              <IntakeStage datasetPath={datasetPath} />
            )}
            {stageParam === "audit" && (
              <AuditStage datasetPath={datasetPath} />
            )}
            {stageParam === "curate" && <DatasetDetail />}
            {stageParam === "annotate" && (
              <AnnotateStage datasetPath={datasetPath} />
            )}
            {stageParam === "ship" && (
              <ShipStage datasetPath={datasetPath} />
            )}
          </div>
        )}
      </main>

      {showCreate && (
        <CreateDatasetDialog
          onClose={() => setShowCreate(false)}
          onCreate={(data) => createMutation.mutate(data)}
          loading={createMutation.isPending}
        />
      )}
    </div>
  )
}

function EmptyState({ onCreateDataset }: { onCreateDataset: () => void }) {
  return (
    <div className="flex h-full items-center justify-center">
      <div className="flex flex-col items-center gap-4 text-center">
        <div className="rounded-full bg-muted p-4">
          <svg
            className="size-8 text-muted-foreground/60"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z"
            />
          </svg>
        </div>
        <div>
          <p className="text-sm font-medium">图像工作台</p>
          <p className="mt-1 text-xs text-muted-foreground">
            从侧边栏选择数据集，或创建新数据集开始工作
          </p>
        </div>
        <button
          type="button"
          onClick={onCreateDataset}
          className="rounded-md bg-primary px-4 py-2 text-xs font-medium text-primary-foreground hover:bg-primary/90"
        >
          新建数据集
        </button>
      </div>
    </div>
  )
}
