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
  modified_at: number
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

export interface KohyaBackendStatus {
  id: "kohya"
  sd_scripts_path: string
  sd_scripts_ok: boolean
  missing_scripts: string[]
  python: string | null
  python_ok: boolean
  venv_detected: boolean
  ready: boolean
  source: "env" | "settings" | "default"
}

export interface DiffusionPipeBackendStatus {
  id: "diffusion-pipe"
  repo_path: string
  repo_ok: boolean
  missing_files: string[]
  python: string | null
  python_ok: boolean
  venv_detected: boolean
  ready: boolean
  source: "env" | "settings" | "default"
}

export type AnyBackendStatus = KohyaBackendStatus | DiffusionPipeBackendStatus

// Legacy alias still used in older components — points at kohya for now.
export type BackendStatus = KohyaBackendStatus

export type BackendId = "kohya" | "diffusion-pipe"

export interface BackendDescriptor {
  id: BackendId
  name: string
  description: string
  repo_url: string
  default_path: string
  ready: boolean
  status: AnyBackendStatus
}

export interface BackendsResponse {
  backends: BackendDescriptor[]
  default: BackendId
}

export interface SettingsState {
  sd_scripts_path: string | null
  python_executable: string | null
  diffusion_pipe_repo_path: string | null
  diffusion_pipe_python: string | null
  default_backend: BackendId
  tagger_device: "auto" | "cpu" | "cuda"
  github_proxy: string | null
  huggingface_endpoint: string | null
  modelscope_enabled: boolean
  modelscope_token: string | null
  pypi_index_url: string | null
  extra: Record<string, unknown>
}

export interface SettingsResponse {
  settings: SettingsState
  backend: AnyBackendStatus
  backends: Record<BackendId, AnyBackendStatus>
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
  // Present when a session has been started; absent on the synthetic
  // idle response the server emits before the first bootstrap.
  backend?: BackendId
}

export interface BootstrapStartResponse {
  session_id: string
  status: string
  backend?: BackendId
}

export interface BootstrapRequestBody {
  backend?: BackendId
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

export interface JobFile {
  path: string
  size_bytes: number
  modified_at: number
}

export interface JobFilesResponse {
  workspace: string
  checkpoints: JobFile[]
  samples: JobFile[]
  logs: JobFile[]
  other: JobFile[]
}

export interface JobMetricPoint {
  step: number
  epoch?: number | null
  loss?: number | null
  ts: number
}

export interface JobMetricsResponse {
  loss: JobMetricPoint[]
  epochs: Array<{ epoch: number; ts: number }>
  checkpoints: Array<{ path: string; step: number; ts: number }>
  samples: Array<{ path: string; ts: number }>
  first_step_ts: number | null
  last_step_ts: number | null
  duration_s: number | null
}

export interface RecipeTemplate {
  id: string
  name: string
  description: string
  arch: string
  recipe: Record<string, unknown>
}

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
  listBackends: () => http<BackendsResponse>("/backends"),
  getRuntimeStatus: () =>
    http<{
      default_version: string
      recommended_versions: string[]
      install_dir: string
      platform: { system: string; machine: string; release: string }
      installed: Array<{
        version: string
        implementation: string
        arch: string
        os: string
        path: string
        key: string
        installed: boolean
      }>
      active: {
        version: string
        path: string
      } | null
    }>("/runtime/python"),
  installRuntime: (version?: string) =>
    http<{
      installed: { version: string; path: string }
      status: {
        default_version: string
        installed: Array<{ version: string; path: string }>
        active: { version: string; path: string } | null
      }
    }>("/runtime/python/install", {
      method: "POST",
      body: JSON.stringify({ version }),
    }),
  downloadModel: (
    body: {
      source: "huggingface" | "modelscope"
      repo_id: string
      revision?: string
      target_dir?: string | null
      threads?: number
    },
  ) =>
    http<ModelDownloadSession>("/models/download", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  getModelDownload: (sessionId: string) =>
    http<ModelDownloadSession>(`/models/download/${sessionId}`),
  getSystemStats: () => http<SystemSnapshot>("/system/stats"),
  listMirrorPresets: () => http<Record<string, MirrorPreset[]>>("/network/presets"),
  probeMirrors: (
    body: {
      category?: string
      urls?: string[]
      timeout_ms?: number
    },
  ) =>
    http<ProbeResult[]>("/network/probe", {
      method: "POST",
      body: JSON.stringify({
        category: body.category,
        urls: body.urls,
        timeout_ms: body.timeout_ms ?? 4000,
      }),
    }),
  getJobFiles: (id: string) => http<JobFilesResponse>(`/jobs/${id}/files`),
  getJobMetrics: (id: string) => http<JobMetricsResponse>(`/jobs/${id}/metrics`),
  jobFileUrl: (id: string, path: string) =>
    `/api/jobs/${id}/files/raw?path=${encodeURIComponent(path)}`,
  duplicateRecipe: (name: string, newName: string) =>
    http<{ name: string; filename: string; path: string }>(
      `/recipes/${encodeURIComponent(name)}/duplicate`,
      { method: "POST", body: JSON.stringify({ new_name: newName }) },
    ),
  renameRecipe: (name: string, newName: string) =>
    http<{ name: string; filename: string; path: string }>(
      `/recipes/${encodeURIComponent(name)}/rename`,
      { method: "POST", body: JSON.stringify({ new_name: newName }) },
    ),
  deleteRecipe: (name: string) =>
    http<{ deleted: boolean; name: string }>(
      `/recipes/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    ),
  listRecipeTemplates: () =>
    http<{ templates: RecipeTemplate[] }>("/recipes/templates"),
  importRecipe: async (name: string, file: File, overwrite = false) => {
    const fd = new FormData()
    fd.append("file", file)
    fd.append("name", name)
    fd.append("overwrite", overwrite ? "true" : "false")
    const res = await fetch(`${API_BASE}/recipes/import`, {
      method: "POST",
      body: fd,
    })
    if (!res.ok) {
      throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`)
    }
    return res.json() as Promise<{ name: string; filename: string; path: string }>
  },
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
  frequency_mhz: number | null
  cpu_temperature_c: number | null
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
  vendor: "nvidia" | "amd" | "intel" | "apple" | "qemu" | "unknown" | string
}

export interface SystemBattery {
  percent: number
  plugged: boolean | null
  secs_left: number | null
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
  battery: SystemBattery | null
  network: {
    bytes_sent_total: number
    bytes_recv_total: number
    bytes_sent_per_sec: number
    bytes_recv_per_sec: number
  } | null
}

export interface MirrorPreset {
  label: string
  value: string
  probe: string
}

export interface ProbeResult {
  label: string
  value: string
  probe: string
  ok: boolean
  status: number | null
  latency_ms: number | null
  error: string | null
}

/**
 * Subscribe to /api/system/stream — the server pushes a fresh snapshot
 * every second. The most recent snapshot is returned alongside the WS state
 * so the dashboard can fall back to polling when the socket is down.
 *
 * The connection is opened lazily on a microtask so React 18 StrictMode's
 * synchronous mount/unmount/mount cycle doesn't leave a half-opened socket
 * behind (which prints a noisy "closed before the connection is established"
 * warning in dev). On unmount we wait for `open` before calling `close()`,
 * for the same reason.
 */
export function useSystemStream(enabled = true) {
  const [snapshot, setSnapshot] = useState<SystemSnapshot | null>(null)
  const [status, setStatus] = useState<"idle" | "open" | "closed">("idle")
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    let ws: WebSocket | null = null

    const timer = window.setTimeout(() => {
      if (cancelled) return
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
      const host = window.location.host || "127.0.0.1:18765"
      ws = new WebSocket(`${protocol}//${host}/api/system/stream`)
      wsRef.current = ws

      ws.onopen = () => setStatus("open")
      ws.onclose = () => setStatus("closed")
      ws.onerror = () => setStatus("closed")
      ws.onmessage = (msg) => {
        try {
          setSnapshot(JSON.parse(msg.data) as SystemSnapshot)
        } catch {
          // ignore malformed frames
        }
      }
    }, 30)

    return () => {
      cancelled = true
      window.clearTimeout(timer)
      const sock = ws
      if (!sock) return
      if (sock.readyState === WebSocket.CONNECTING) {
        sock.addEventListener("open", () => sock.close(), { once: true })
      } else if (sock.readyState === WebSocket.OPEN) {
        sock.close()
      }
      wsRef.current = null
    }
  }, [enabled])

  return { snapshot, status }
}
