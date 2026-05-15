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
