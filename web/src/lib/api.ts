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

export interface ConfigListEntry {
  name: string
  filename: string
  size: number
  modified_at: number
  valid: boolean
  arch: string | null
  summary: string | null
  error: string | null
}

export interface ConfigDetail {
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
  requirements_ok: boolean
  missing_requirements: string[]
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
  requirements_ok: boolean
  missing_requirements: string[]
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
  download_proxy: string | null
  allow_filesystem_browse: boolean
  extra: Record<string, unknown>
}

export interface SettingsResponse {
  settings: SettingsState
  backend: AnyBackendStatus
  backends: Record<BackendId, AnyBackendStatus>
  path: string
}

export type FsEntryKind = "dir" | "image" | "text" | "binary"

export interface FsEntry {
  name: string
  path: string
  relative_path: string
  is_dir: boolean
  kind: FsEntryKind
  suffix: string
  size: number
  mtime: number | null
}

export interface FsRoot {
  name: string
  path: string
  kind: "dataset_root" | "drive"
}

export interface FsRootsResponse {
  roots: FsRoot[]
  unrestricted: boolean
}

export interface FsListResponse {
  path: string
  parent: string | null
  entries: FsEntry[]
  truncated: boolean
}

export interface FsSubdir {
  name: string
  path: string
}

export interface FsSubdirsResponse {
  path: string
  subdirs: FsSubdir[]
}

export interface FsReadResponse {
  path: string
  kind: "text" | "image" | "binary"
  size: number
  suffix?: string
  encoding?: string
  content: string | null
  reason?: string
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

export interface JobValLossPoint {
  epoch: number
  val_loss: number
  step?: number | null
  ts: number
}

export type OverfitTrend = "improving" | "flat" | "overfitting"

export interface OverfitSignal {
  latest_train: number | null
  latest_val: number | null
  gap: number | null
  trend: OverfitTrend | null
}

export interface JobMetricsResponse {
  loss: JobMetricPoint[]
  val_loss: JobValLossPoint[]
  epochs: Array<{ epoch: number; ts: number }>
  checkpoints: Array<{ path: string; step: number; ts: number }>
  samples: Array<{ path: string; ts: number }>
  first_step_ts: number | null
  last_step_ts: number | null
  duration_s: number | null
  overfit_signal: OverfitSignal
}

export interface ConfigTemplatePlaceholder {
  key: string
  label: string
  path_field: string
  placeholder: string
}

export interface ConfigTemplate {
  id: string
  name: string
  description: string
  arch: string
  placeholders: ConfigTemplatePlaceholder[]
  config: Record<string, unknown>
}

export interface SampleGalleryItem {
  job_id: string
  job_name: string
  config_name: string | null
  path: string
  size_bytes: number
  modified_at: number
  raw_url: string
}

export interface SampleGalleryResponse {
  items: SampleGalleryItem[]
  total: number
  limit: number
  offset: number
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

export interface DatasetCaptionResponse {
  path: string
  caption: string | null
}

export interface TaggingEvent {
  ts: number
  message: string
  percent: number
  image: string | null
}

export interface TaggingSession {
  session_id: string
  path: string
  model_id: string
  device: "auto" | "cpu" | "cuda"
  general: number
  character: number
  overwrite: boolean
  recursive: boolean
  include_character: boolean
  underscores: boolean
  status: "running" | "succeeded" | "failed"
  percent: number
  events: TaggingEvent[]
  written: number
  total: number | null
  active_provider: string
  error: string | null
  started_at: number
  finished_at: number | null
}

export interface TagDatasetRequest {
  path: string
  model_id?: string
  general?: number
  character?: number
  device?: "auto" | "cpu" | "cuda"
  overwrite?: boolean
  recursive?: boolean
  include_character?: boolean
  underscores?: boolean
}

export interface SweepSummary {
  sweep_id: string
  name_prefix: string
  total: number
  queued: number
  running: number
  succeeded: number
  failed: number
  canceled: number
  interrupted: number
  canceling: number
  earliest_created_at: string
  latest_modified_at: string
}

export interface SweepJobSummary {
  id: string
  state: string
  workspace: string
  created_at: string
  started_at: string | null
  finished_at: string | null
  returncode: number | null
  error: string | null
  pid: number | null
  metadata: {
    sweep_id?: string
    variant_name?: string
    axis_values?: Record<string, unknown>
  } | null
}

export interface SweepDetail {
  sweep_id: string
  total: number
  queued: number
  running: number
  succeeded: number
  failed: number
  canceled: number
  interrupted: number
  canceling: number
  jobs: SweepJobSummary[]
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
  configSchema: () => http<Record<string, unknown>>("/configs/schema"),
  listConfigs: () =>
    http<{ dir: string; configs: ConfigListEntry[] }>("/configs"),
  getConfig: (name: string) =>
    http<ConfigDetail>(`/configs/${encodeURIComponent(name)}`),
  validateConfig: (config: Record<string, unknown>) =>
    http<ValidateResponse>("/configs/validate", {
      method: "POST",
      body: JSON.stringify({ config }),
    }),
  saveConfig: (
    name: string,
    config: Record<string, unknown>,
    overwrite = false,
  ) =>
    http<{ name: string; filename: string; path: string }>("/configs", {
      method: "POST",
      body: JSON.stringify({ name, config, overwrite }),
    }),
  createJob: (config: Record<string, unknown>, workspace?: string) =>
    http<JobSummary>("/jobs", {
      method: "POST",
      body: JSON.stringify({ config, workspace }),
    }),
  scanDataset: (path: string, recursive = false, limit = 40) =>
    http<DatasetScanResponse>(
      `/datasets/scan?path=${encodeURIComponent(path)}&recursive=${recursive ? "true" : "false"}&limit=${limit}`,
    ),
  datasetThumbUrl: (path: string, size = 256) =>
    `/api/datasets/thumb?path=${encodeURIComponent(path)}&size=${size}`,
  getCaption: (path: string) =>
    http<DatasetCaptionResponse>(
      `/datasets/caption?path=${encodeURIComponent(path)}`,
    ),
  putCaption: (path: string, caption: string) =>
    http<DatasetCaptionResponse & { bytes: number }>("/datasets/caption", {
      method: "PUT",
      body: JSON.stringify({ path, caption }),
    }),
  fsRoots: () => http<FsRootsResponse>("/fs/roots"),
  fsList: (path: string, showHidden = false) =>
    http<FsListResponse>(
      `/fs/list?path=${encodeURIComponent(path)}&show_hidden=${showHidden ? "true" : "false"}`,
    ),
  fsSubdirs: (path: string) =>
    http<FsSubdirsResponse>(`/fs/subdirs?path=${encodeURIComponent(path)}`),
  fsRead: (path: string) =>
    http<FsReadResponse>(`/fs/read?path=${encodeURIComponent(path)}`),
  fsWrite: (path: string, content: string, create = false) =>
    http<{ path: string; bytes: number }>("/fs/write", {
      method: "PUT",
      body: JSON.stringify({ path, content, create }),
    }),
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
  installDeps: (backend: BackendId = "diffusion-pipe") =>
    http<BootstrapStartResponse>("/backend/install-deps", {
      method: "POST",
      body: JSON.stringify({ backend }),
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
  tagDataset: (body: TagDatasetRequest) =>
    http<TaggingSession>("/tagging/tag", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  getTaggingSession: (sessionId: string) =>
    http<TaggingSession>(`/tagging/tag/${sessionId}`),
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
  duplicateConfig: (name: string, newName: string) =>
    http<{ name: string; filename: string; path: string }>(
      `/configs/${encodeURIComponent(name)}/duplicate`,
      { method: "POST", body: JSON.stringify({ new_name: newName }) },
    ),
  renameConfig: (name: string, newName: string) =>
    http<{ name: string; filename: string; path: string }>(
      `/configs/${encodeURIComponent(name)}/rename`,
      { method: "POST", body: JSON.stringify({ new_name: newName }) },
    ),
  deleteConfig: (name: string) =>
    http<{ deleted: boolean; name: string }>(
      `/configs/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    ),
  listConfigTemplates: () =>
    http<{ templates: ConfigTemplate[] }>("/configs/templates"),
  instantiateConfigTemplate: (
    templateId: string,
    body: {
      name: string
      values: Record<string, string>
      overwrite?: boolean
    },
  ) =>
    http<{
      name: string
      filename: string
      path: string
      template_id: string
    }>(`/configs/templates/${encodeURIComponent(templateId)}/instantiate`, {
      method: "POST",
      body: JSON.stringify({
        name: body.name,
        values: body.values,
        overwrite: body.overwrite ?? false,
      }),
    }),
  listSamples: (
    params: { limit?: number; offset?: number; jobIds?: string[] } = {},
  ) => {
    const search = new URLSearchParams()
    if (params.limit !== undefined) search.set("limit", String(params.limit))
    if (params.offset !== undefined) search.set("offset", String(params.offset))
    if (params.jobIds && params.jobIds.length > 0) {
      search.set("job_ids", params.jobIds.join(","))
    }
    const qs = search.toString()
    return http<SampleGalleryResponse>(`/samples${qs ? `?${qs}` : ""}`)
  },
  importConfig: async (name: string, file: File, overwrite = false) => {
    const fd = new FormData()
    fd.append("file", file)
    fd.append("name", name)
    fd.append("overwrite", overwrite ? "true" : "false")
    const res = await fetch(`${API_BASE}/configs/import`, {
      method: "POST",
      body: fd,
    })
    if (!res.ok) {
      throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`)
    }
    return res.json() as Promise<{ name: string; filename: string; path: string }>
  },
  listSweeps: () => http<{ sweeps: SweepSummary[] }>("/sweeps"),
  getSweep: (sweep_id: string) =>
    http<SweepDetail>(`/sweeps/${encodeURIComponent(sweep_id)}`),
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
  // Newer fields - all optional so older snapshots still type-check.
  model?: string
  frequency_min_mhz?: number | null
  frequency_max_mhz?: number | null
  frequency_per_core_mhz?: number[]
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
  // PCIe link state (optional - older backends don't emit these).
  pcie_gen_current?: number | null
  pcie_width_current?: number | null
  pcie_gen_max?: number | null
  pcie_width_max?: number | null
  // Clocks (MHz).
  sm_clock_mhz?: number | null
  mem_clock_mhz?: number | null
  sm_clock_max_mhz?: number | null
  mem_clock_max_mhz?: number | null
}

export interface ProcessInfo {
  pid: number
  name: string
  cpu_percent: number
  memory_rss_bytes: number
  memory_percent: number
}

export interface InterfaceAddress {
  family: string
  address: string
  netmask?: string | null
  broadcast?: string | null
}

export type NetworkInterfaceKind = "physical" | "loopback" | "virtual" | "wireless"

export interface NetworkInterfaceStats {
  name: string
  is_up: boolean
  speed_mbps: number | null
  mtu: number | null
  addresses: InterfaceAddress[]
  bytes_sent_total: number
  bytes_recv_total: number
  bytes_sent_per_sec: number
  bytes_recv_per_sec: number
  packets_sent_total: number
  packets_recv_total: number
  errors_in: number
  errors_out: number
  drops_in: number
  drops_out: number
  kind: NetworkInterfaceKind
}

export interface TcpConnectionStats {
  total: number
  established: number
  listen: number
  time_wait: number
  close_wait: number
  other: number
}

export interface PublicIpInfo {
  ip: string | null
  fetched_at: number
  source: "ip.sb" | "ipinfo.io" | "cached" | "unreachable" | string
}

export interface DiskIoDevice {
  device: string
  read_bytes_per_sec: number
  write_bytes_per_sec: number
  read_ops_per_sec: number
  write_ops_per_sec: number
}

export interface DiskIoStats {
  read_bytes_total: number
  write_bytes_total: number
  read_bytes_per_sec: number
  write_bytes_per_sec: number
  read_ops_per_sec: number
  write_ops_per_sec: number
  per_device: DiskIoDevice[]
}

export interface GpuProcessInfo {
  gpu_index: number
  pid: number
  process_name: string
  used_memory_mib: number
  type: "C" | "G" | "C+G" | string
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
    // Newer fields - all optional so older snapshots still parse.
    interfaces?: NetworkInterfaceStats[]
    tcp_connections?: TcpConnectionStats | null
    public_ip?: PublicIpInfo | null
  } | null
  // Newer top-level fields - all optional.
  processes?: ProcessInfo[]
  disk_io?: DiskIoStats | null
  gpu_processes?: GpuProcessInfo[]
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
 * Subscribe to /api/system/stream with smart automatic reconnection.
 *
 * Reconnection strategy:
 *   - First disconnect: immediate retry (0ms)
 *   - Subsequent: exponential backoff 500ms -> 1s -> 2s -> 3s (capped)
 *   - Tab becomes visible: immediate reconnect if socket is closed
 *   - Network comes back online: immediate reconnect
 *   - Successful open resets backoff to 0
 */
export function useSystemStream(enabled = true) {
  const [snapshot, setSnapshot] = useState<SystemSnapshot | null>(null)
  const [status, setStatus] = useState<"idle" | "open" | "closed">("idle")
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    let ws: WebSocket | null = null
    let retryTimer: ReturnType<typeof setTimeout> | null = null
    let backoff = 0

    function connect() {
      if (cancelled) return
      if (!navigator.onLine) return
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
      const host = window.location.host || "127.0.0.1:18765"
      ws = new WebSocket(`${protocol}//${host}/api/system/stream`)
      wsRef.current = ws

      ws.onopen = () => {
        backoff = 0
        setStatus("open")
      }
      ws.onclose = () => {
        setStatus("closed")
        scheduleRetry()
      }
      ws.onerror = () => {}
      ws.onmessage = (msg) => {
        try {
          setSnapshot(JSON.parse(msg.data) as SystemSnapshot)
        } catch {}
      }
    }

    function scheduleRetry() {
      if (cancelled) return
      retryTimer = setTimeout(() => {
        retryTimer = null
        connect()
      }, backoff)
      backoff = backoff === 0 ? 500 : Math.min(backoff * 2, 3000)
    }

    function reconnectNow() {
      if (cancelled) return
      if (ws && ws.readyState === WebSocket.OPEN) return
      if (retryTimer !== null) clearTimeout(retryTimer)
      retryTimer = null
      backoff = 0
      connect()
    }

    function onVisibilityChange() {
      if (document.visibilityState === "visible") reconnectNow()
    }

    function onOnline() {
      reconnectNow()
    }

    document.addEventListener("visibilitychange", onVisibilityChange)
    window.addEventListener("online", onOnline)

    retryTimer = setTimeout(connect, 30)

    return () => {
      cancelled = true
      document.removeEventListener("visibilitychange", onVisibilityChange)
      window.removeEventListener("online", onOnline)
      if (retryTimer !== null) clearTimeout(retryTimer)
      const sock = ws
      if (!sock) return
      sock.onclose = null
      sock.onerror = null
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
