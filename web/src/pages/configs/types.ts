import type { BackendId } from "@/lib/api"

export type Mode =
  | { kind: "preview"; name: string }
  | { kind: "edit"; name: string }
  | { kind: "new"; backend?: BackendId }

export type LaunchOverrides = {
  datasetSource: string
  outputName: string
  batchSize: string
  epochs: string
  maxSteps: string
}

export type ArchFilter = "all" | "sdxl" | "sd15" | "flux" | "sd3"

/**
 * Backend filter for the configs list. ``"default"`` follows the user's
 * Settings → 默认后端 selection — i.e. show only configs whose ``backend``
 * field matches the workbench-level default. ``"all"`` is the explicit
 * escape hatch.
 */
export type BackendFilter =
  | "default"
  | "all"
  | "kohya"
  | "diffusion-pipe"
  | "anima_lora"
  | "ai_toolkit"

export type SortOrder = "name-asc" | "name-desc" | "modified-desc"

export type RowAction = "duplicate" | "rename" | "delete"
