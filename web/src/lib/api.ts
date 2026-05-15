import { useEffect, useRef, useState } from "react"

const API_BASE = "/api"

export interface JobSummary {
  id: string
  state: string
  workspace: string
  created_at: string
  started_at: string | null
  finished_at: string | null
  returncode: number | null
  error: string | null
  pid: number | null
}

export interface TrainingEvent {
  type: string
  payload: Record<string, unknown>
  timestamp: number
  job_id: string | null
}

export interface RecipeListEntry {
  name: string
  filename: string
  size: number
  valid: boolean
  arch: string | null
  summary: string | null
  error: string | null
}

export interface RecipeDetail {
  name: string
  filename: string
  path: string
  content: string
  parsed: Record<string, unknown> | null
  error: string | null
}

export interface BackendStatus {
  sd_scripts_path: string
  sd_scripts_ok: boolean
  missing_scripts: string[]
  python: string | null
  python_ok: boolean
  venv_detected: boolean
  source: "env" | "settings" | "default"
}

export interface SettingsState {
  sd_scripts_path: string | null
  python_executable: string | null
  tagger_device: "auto" | "cpu" | "cuda"
  extra: Record<string, unknown>
}

export interface SettingsResponse {
  settings: SettingsState
  backend: BackendStatus
  path: string
}

export interface BootstrapEvent {
  step: string
  level: "info" | "done" | "error" | string
  message: string
  ts: number
}

export interface BootstrapStatus {
  status: "idle" | "running" | "succeeded" | "failed"
  session_id: string | null
  events: BootstrapEvent[]
}

export interface BootstrapStartResponse {
  session_id: string
  status: string
}

export interface BootstrapRequestBody {
  target?: string | null
  cuda?: string
  torch_version?: string
  torchvision_version?: string
  install_xformers?: boolean
  force?: boolean
}

export interface ValidationFieldError {
  loc: (string | number)[]
  msg: string
  type: string
}

export interface ValidateResponse {
  valid: boolean
  normalized?: Record<string, unknown>
  errors?: ValidationFieldError[]
  preflight?: {
    issues: Array<{ severity: string; field: string; message: string }>
    vram: {
      model_mib: number
      optimizer_mib: number
      activations_mib: number
      overhead_mib: number
      total_mib: number
      total_gib: number
    }
    paths: {
      checkpoint_exists: boolean
      dataset_exists: boolean
      image_files: number
      caption_files: number
      missing_caption_files: string[]
      missing_caption_files_truncated: boolean
    }
  }
}

export interface DatasetScanResponse {
  path: string
  exists: boolean
  recursive: boolean
  image_files: number
  caption_files: number
  missing_caption_files: string[]
  missing_caption_files_truncated: boolean
  samples: Array<{
    name: string
    path: string
    relative_path: string
    caption_exists: boolean
    caption: string | null
  }>
}

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "content-type": "application/json" },
    ...init,
  })
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => http<{ status: string; version: string }>("/health"),
  listJobs: () => http<{ jobs: JobSummary[] }>("/jobs"),
  getJob: (id: string) => http<JobSummary>(`/jobs/${id}`),
  getEvents: (id: string, limit = 200) =>
    http<{ events: TrainingEvent[] }>(`/jobs/${id}/events?limit=${limit}`),
  cancelJob: (id: string) =>
    http<JobSummary>(`/jobs/${id}`, { method: "DELETE" }),
  rerunJob: (id: string) =>
    http<JobSummary>(`/jobs/${id}/rerun`, { method: "POST" }),
  revealJob: (id: string) =>
    http<{ opened: string }>(`/jobs/${id}/reveal`, { method: "POST" }),
  archiveJob: (id: string) =>
    http<{
      archived: boolean
      workspace_moved_to: string | null
      warnings: string[]
    }>(`/jobs/${id}?archive=true`, { method: "DELETE" }),
  recipeSchema: () => http<Record<string, unknown>>("/recipes/schema"),
  listRecipes: () =>
    http<{ dir: string; recipes: RecipeListEntry[] }>("/recipes"),
  getRecipe: (name: string) =>
    http<RecipeDetail>(`/recipes/${encodeURIComponent(name)}`),
  validateRecipe: (recipe: Record<string, unknown>) =>
    http<ValidateResponse>("/recipes/validate", {
      method: "POST",
      body: JSON.stringify({ recipe }),
    }),
  saveRecipe: (
    name: string,
    recipe: Record<string, unknown>,
    overwrite = false,
  ) =>
    http<{ name: string; filename: string; path: string }>("/recipes", {
      method: "POST",
      body: JSON.stringify({ name, recipe, overwrite }),
    }),
  createJob: (recipe: Record<string, unknown>, workspace?: string) =>
    http<JobSummary>("/jobs", {
      method: "POST",
      body: JSON.stringify({ recipe, workspace }),
    }),
  scanDataset: (path: string, recursive = false, limit = 40) =>
    http<DatasetScanResponse>(
      `/datasets/scan?path=${encodeURIComponent(path)}&recursive=${recursive ? "true" : "false"}&limit=${limit}`,
    ),
  getSettings: () => http<SettingsResponse>("/settings"),
  updateSettings: (patch: Partial<SettingsState>) =>
    http<SettingsResponse>("/settings", {
      method: "PUT",
      body: JSON.stringify(patch),
    }),
  startBootstrap: (body: BootstrapRequestBody = {}) =>
    http<BootstrapStartResponse>("/backend/bootstrap", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  getBootstrapStatus: () => http<BootstrapStatus>("/backend/bootstrap/status"),
  getSystemStats: () => http<SystemSnapshot>("/system/stats"),
}

/**
 * Live event stream over WebSocket. Returns the latest snapshot of buffered
 * events plus the connection state; reconnects on close while the job is alive.
 */
export function useJobStream(jobId: string | null) {
  const [events, setEvents] = useState<TrainingEvent[]>([])
  const [status, setStatus] = useState<"idle" | "open" | "closed">("idle")
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!jobId) return
    setEvents([])
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
    const host = window.location.host || "127.0.0.1:18765"
    const ws = new WebSocket(`${protocol}//${host}/api/jobs/${jobId}/stream`)
    wsRef.current = ws

    ws.onopen = () => setStatus("open")
    ws.onclose = () => setStatus("closed")
    ws.onmessage = (msg) => {
      try {
        const event = JSON.parse(msg.data) as TrainingEvent
        setEvents((prev) => [...prev, event].slice(-500))
      } catch {
        // ignore malformed frames
      }
    }

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [jobId])

  return { events, status }
}

/**
 * Live bootstrap event stream over WebSocket. Mirrors `useJobStream` but talks
 * to the singleton backend-install session — it doesn't take an id because at
 * most one bootstrap can run at a time.
 */
export function useBootstrapStream(enabled: boolean) {
  const [events, setEvents] = useState<BootstrapEvent[]>([])
  const [status, setStatus] = useState<"idle" | "open" | "closed">("idle")
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!enabled) return
    setEvents([])
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
    const host = window.location.host || "127.0.0.1:18765"
    const ws = new WebSocket(`${protocol}//${host}/api/backend/bootstrap/stream`)
    wsRef.current = ws

    ws.onopen = () => setStatus("open")
    ws.onclose = () => setStatus("closed")
    ws.onmessage = (msg) => {
      try {
        const event = JSON.parse(msg.data) as BootstrapEvent
        setEvents((prev) => [...prev, event].slice(-200))
      } catch {
        // ignore malformed frames
      }
    }

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [enabled])

  return { events, status }
}

// ============================================ system telemetry =============

export interface SystemHost {
  hostname: string
  system: string
  release: string
  python: string
}

export interface SystemCpu {
  cores_logical: number
  cores_physical: number | null
  usage_percent: number | null
  per_core_percent: number[]
  load_average: number[] | null
  arch: string
}

export interface SystemMemory {
  total_bytes: number
  used_bytes: number
  available_bytes: number
  percent: number
  swap_total_bytes: number | null
  swap_used_bytes: number | null
}

export interface SystemDisk {
  path: string
  label: string
  total_bytes: number
  used_bytes: number
  free_bytes: number
  percent: number
}

export interface SystemGpu {
  index: number
  name: string
  driver: string | null
  memory_total_bytes: number | null
  memory_used_bytes: number | null
  memory_free_bytes: number | null
  utilization_percent: number | null
  temperature_c: number | null
  power_w: number | null
  power_limit_w: number | null
  fan_percent: number | null
}

export interface SystemSnapshot {
  timestamp: number
  has_psutil: boolean
  has_nvidia_smi: boolean
  host: SystemHost
  cpu: SystemCpu
  memory: SystemMemory
  disks: SystemDisk[]
  gpus: SystemGpu[]
}

/**
 * Subscribe to /api/system/stream — the server pushes a fresh snapshot
 * every second. The most recent snapshot is returned alongside the WS state
 * so the dashboard can fall back to polling when the socket is down.
 */
export function useSystemStream(enabled = true) {
  const [snapshot, setSnapshot] = useState<SystemSnapshot | null>(null)
  const [status, setStatus] = useState<"idle" | "open" | "closed">("idle")
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!enabled) return
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
    const host = window.location.host || "127.0.0.1:18765"
    const ws = new WebSocket(`${protocol}//${host}/api/system/stream`)
    wsRef.current = ws

    ws.onopen = () => setStatus("open")
    ws.onclose = () => setStatus("closed")
    ws.onmessage = (msg) => {
      try {
        setSnapshot(JSON.parse(msg.data) as SystemSnapshot)
      } catch {
        // ignore malformed frames
      }
    }

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [enabled])

  return { snapshot, status }
}
