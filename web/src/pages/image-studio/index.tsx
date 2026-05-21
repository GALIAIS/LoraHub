import { useSearchParams } from "react-router-dom"
import { useMemo } from "react"
import { DatasetManager } from "./components/dataset-manager"
import { DatasetDetail } from "./components/dataset-detail"
import { StageStepper, DEFAULT_STAGES } from "./components/stage-stepper"
import type { StageId, StageInfo } from "./components/stage-stepper"
import { IntakeStage } from "./components/stages/intake-stage"
import { AuditStage } from "./components/stages/audit-stage"
import { AnnotateStage } from "./components/stages/annotate-stage"
import { ShipStage } from "./components/stages/ship-stage"

export { ImageStudioPage }

function ImageStudioPage() {
  const [params, setParams] = useSearchParams()
  const datasetPath = params.get("path") || ""
  const stageParam = (params.get("stage") || "curate") as StageId

  // No dataset selected → manager.
  if (!datasetPath) {
    return (
      <DatasetManager
        onOpen={(path) => {
          const next = new URLSearchParams()
          next.set("path", path)
          next.set("stage", "audit")
          setParams(next)
        }}
      />
    )
  }

  // Stage status — for now everything starts as 'idle'. F1 will wire
  // these to the audit report so the badge reflects what's actually
  // done. Curate (the legacy main view) is shown as 'active' when the
  // user is on it; that makes the stepper useful even before audit
  // landing.
  const stages: StageInfo[] = useMemo(
    () =>
      DEFAULT_STAGES.map((s) => ({
        ...s,
        status: s.id === stageParam ? ("active" as const) : ("idle" as const),
      })),
    [stageParam],
  )

  const setStage = (next: StageId) => {
    const n = new URLSearchParams(params)
    n.set("stage", next)
    setParams(n)
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      <StageStepper stages={stages} active={stageParam} onSelect={setStage} />
      <div className="flex-1 min-h-0 overflow-hidden">
        {stageParam === "intake" && <IntakeStage datasetPath={datasetPath} />}
        {stageParam === "audit" && <AuditStage datasetPath={datasetPath} />}
        {stageParam === "curate" && <DatasetDetail />}
        {stageParam === "annotate" && <AnnotateStage datasetPath={datasetPath} />}
        {stageParam === "ship" && <ShipStage datasetPath={datasetPath} />}
      </div>
    </div>
  )
}
