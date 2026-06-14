// Model downloads and on-disk scan results — used by the model picker
// dialog and dataset-tab "browse local checkpoints" panel.

export interface ModelDownloadEvent {
  message: string
  percent: number | null
  files_done: number
  files_total: number
  bytes_done: number
  bytes_total: number
  ts: number
}

export interface ModelDownloadSession {
  session_id: string
  source: "huggingface" | "modelscope"
  repo_id: string
  revision: string
  target_dir: string | null
  threads: number
  paths: string[]
  status: "running" | "succeeded" | "failed"
  percent: number
  events: ModelDownloadEvent[]
  result: {
    source: string
    repo_id: string
    revision: string
    target: string
    files: number
    total_bytes: number
  } | null
  error: string | null
  started_at: number
  finished_at: number | null
}

export interface RemoteModelFile {
  path: string
  size: number
  selected: boolean
  reason: string
}

export interface RemoteModelFilesResponse {
  source: "huggingface" | "modelscope"
  repo_id: string
  revision: string
  files: RemoteModelFile[]
  selected_count: number
  selected_bytes: number
  total_count: number
  total_bytes: number
}

export interface ScannedModel {
  path: string
  relative_path: string
  name: string
  size_bytes: number
  mtime: number
}

export interface ScannedModelsResponse {
  root: string
  files: ScannedModel[]
  elapsed_s: number
}
