export type Mode =
  | { kind: "preview"; name: string }
  | { kind: "edit"; name: string }
  | { kind: "new" }

export type LaunchOverrides = {
  datasetSource: string
  outputName: string
  batchSize: string
  epochs: string
  maxSteps: string
}

export type ArchFilter = "all" | "sdxl" | "sd15" | "flux" | "sd3"

export type SortOrder = "name-asc" | "name-desc" | "modified-desc"

export type RowAction = "duplicate" | "rename" | "delete"
