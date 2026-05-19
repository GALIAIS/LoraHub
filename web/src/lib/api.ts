import { useEventStream } from "./use-event-stream"

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
  /** Free-form bag stamped by orchestrators. UI consumers care about
   *  ``paused``: when true, the cancel button flips into "继续训练". */
  metadata?: Record<string, unknown> | null
}

/** /jobs/{id} returns the summary plus the config snapshot. */
export interface JobDetail extends JobSummary {
  config_snapshot?: Record<string, unknown>
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
  /** Which training backend this config targets. Null when load() failed. */
  backend: BackendId | null
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

export interface AnimaLoraBackendStatus {
  id: "anima_lora"
  repo_path: string
  repo_ok: boolean
  missing_files: string[]
  python: string | null
  python_ok: boolean
  venv_detected: boolean
  // Vendored — no LoraHub-managed requirements, always true. Kept on
  // the type for shape parity with the other two probes.
  requirements_ok: boolean
  missing_requirements: string[]
  // Anima base / TE / VAE checkpoints. ``ready`` only covers the venv;
  // models are tracked separately so the install panel can offer a
  // dedicated "Download models" CTA.
  missing_models: string[]
  models_ok: boolean
  ready: boolean
  // anima_lora's source dimension picks "vendored" instead of
  // "default" because the source ships with LoraHub itself, not as a
  // sibling clone.
  source: "env" | "settings" | "vendored"
}

export type AnyBackendStatus =
  | KohyaBackendStatus
  | DiffusionPipeBackendStatus
  | AnimaLoraBackendStatus

// Legacy alias still used in older components — points at kohya for now.
export type BackendStatus = KohyaBackendStatus

export type BackendId = "kohya" | "diffusion-pipe" | "anima_lora"

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

export interface AnimaModelDownloadEvent {
  message: string
  percent: number
  files_done: number
  files_total: number
  ts: number
}

export interface AnimaModelDownloadStatus {
  // ``"idle"`` only appears from the GET endpoint when no session has
  // ever started; POST always returns ``"running"`` (or 409 if another
  // download is in flight).
  status: "idle" | "running" | "succeeded" | "failed"
  session_id?: string
  percent?: number
  files_done?: number
  files_total?: number
  events?: AnimaModelDownloadEvent[]
  error?: string | null
  started_at?: number
  finished_at?: number | null
  /** Files still missing on disk — empty array means all three are present. */
  missing_files: string[]
}

export interface SettingsState {
  sd_scripts_path: string | null
  python_executable: string | null
  diffusion_pipe_repo_path: string | null
  diffusion_pipe_python: string | null
  // anima_lora — vendored, repo path is auto-resolved by default; both
  // fields exist for env / dev override. Most users only set the python.
  anima_lora_repo_path: string | null
  anima_lora_python: string | null
  default_backend: BackendId
  tagger_device: "auto" | "cpu" | "cuda"
  github_proxy: string | null
  huggingface_endpoint: string | null
  modelscope_enabled: boolean
  modelscope_token: string | null
  pypi_index_url: string | null
  download_proxy: string | null
  huggingface_token: string | null
  wandb_api_key: string | null
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

export interface AttentionBackendsResponse {
  /** Compute capability of the first NVIDIA GPU, e.g. "8.9". null when none. */
  compute_capability: string | null
  /** Backends usable on this host (subset of `all`). */
  supported: string[]
  /** Canonical superset of recipe-level attention.training values. */
  all: string[]
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
  limit: number
  offset: number
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
  // Optional per-step metrics forwarded by the diffusion-pipe parser:
  // learning rate from the deepspeed engine line, plus iteration time
  // and samples-per-second from dp's own per-step summary. Absent when
  // the upstream backend doesn't emit them (older kohya releases).
  lr?: number | null
  iter_time_s?: number | null
  samples_per_sec?: number | null
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
  gpu_samples: Array<{
    gpu_index: number | null
    util_percent: number | null
    vram_used_mib: number | null
    vram_total_mib: number | null
    temperature_c: number | null
    ts: number
  }>
  first_step_ts: number | null
  last_step_ts: number | null
  duration_s: number | null
  /**
   * Trainer-reported total step count, taken from the most recent
   * `step` event payload's `total_steps`. Single source of truth for
   * progress denominators across the overview / summary / analysis
   * tabs (kohya + anima emit it; dp leaves it null and the UI falls
   * back to a config-derived estimate).
   */
  total_steps: number | null
  overfit_signal: OverfitSignal
}

export interface JobAnalysis {
  markdown: string
  model: string
  generated_at: string
  summary_payload: Record<string, unknown>
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
  /** Search strategy. "grid" / "random" / "tpe". Older entries default to "grid". */
  mode?: "grid" | "random" | "tpe"
  /** Trial budget for random / tpe (null for grid). */
  n_trials?: number | null
  /** Optional sampler seed (null when unset). */
  seed?: number | null
  /** True when the sweep's child jobs are all archived (store-only entry). */
  archived?: boolean
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
  /** Plan recipe (axes + name_template + mode etc.) when persisted. */
  plan?: {
    axes?: Array<{
      path: string
      kind?: string
      values?: unknown[]
      low?: number | null
      high?: number | null
      step?: number | null
    }>
    name_template?: string
    workspace_root?: string
    mode?: "grid" | "random" | "tpe"
    n_trials?: number | null
    seed?: number | null
    study_path?: string | null
  }
  name?: string
  name_prefix?: string
  created_at?: string
  known_job_ids?: string[]
}

export interface SweepParetoTrial {
  axis_values: Record<string, unknown>
  score: number
  job_id: string
  state: string
}

export interface SweepParetoBest {
  axis_values: Record<string, unknown>
  score: number
  job_id: string
}

export interface SweepParetoResponse {
  sweep_id: string
  completed_trials: SweepParetoTrial[]
  best: SweepParetoBest | null
  pending: number
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
  getJob: (id: string) => http<JobDetail>(`/jobs/${id}`),
  getEvents: (id: string, limit = 200) =>
    http<{ events: TrainingEvent[] }>(`/jobs/${id}/events?limit=${limit}`),
  cancelJob: (id: string) =>
    http<JobSummary>(`/jobs/${id}`, { method: "DELETE" }),
  /** Cancel + stamp ``metadata.paused=true`` so the UI swaps the next
   *  render to "恢复训练". The actual cancel mechanics are identical. */
  pauseJob: (id: string) =>
    http<JobSummary>(`/jobs/${id}?paused=true`, { method: "DELETE" }),
  rerunJob: (id: string) =>
    http<JobSummary>(`/jobs/${id}/rerun`, { method: "POST" }),
  /** Resume (optionally with a new config) — fields that pin checkpoint
   *  shape are locked, others (lr, dropTokens, etc.) take effect on
   *  the resumed run. Pass ``config: undefined`` to replay the original. */
  resumeJob: (id: string, config?: Record<string, unknown>) =>
    http<JobSummary>(`/jobs/${id}/resume`, {
      method: "POST",
      body: JSON.stringify(config !== undefined ? { config } : {}),
    }),
  killJob: (id: string) =>
    http<{
      job_id: string
      pid: number
      killed_process_group: boolean
      killed_pid_only: boolean
      warning: string | null
    }>(`/jobs/${id}/kill`, { method: "POST" }),
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
  scanDataset: (
    path: string,
    recursive = false,
    limit = 40,
    offset = 0,
  ) =>
    http<DatasetScanResponse>(
      `/datasets/scan?path=${encodeURIComponent(path)}&recursive=${
        recursive ? "true" : "false"
      }&limit=${limit}&offset=${offset}`,
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
  getAttentionBackends: () =>
    http<AttentionBackendsResponse>("/system/attention-backends"),
  installFlashAttn: (backend: BackendId, version: "2" | "3" | "4") =>
    http<{
      session_id: string
      status: string
      backend: BackendId
      version?: "2" | "3" | "4"
    }>("/backend/install-flash-attn", {
      method: "POST",
      body: JSON.stringify({ backend, version }),
    }),
  getBootstrapStatus: () => http<BootstrapStatus>("/backend/bootstrap/status"),
  listBackends: () => http<BackendsResponse>("/backends"),
  startAnimaModelDownload: () =>
    http<AnimaModelDownloadStatus>("/backends/anima_lora/download-models", {
      method: "POST",
    }),
  getAnimaModelDownloadStatus: () =>
    http<AnimaModelDownloadStatus>(
      "/backends/anima_lora/download-models/status",
    ),
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
  storageUsage: () =>
    http<{
      filesystem: {
        path: string
        total_bytes: number
        used_bytes: number
        free_bytes: number
      }
      directories: Record<
        "runs" | "runs_archive" | "models" | "huggingface_cache",
        | { path: string | null; exists: boolean; bytes: number; files: number }
        | null
      >
    }>("/storage/usage"),
  storageListArchive: () =>
    http<{
      archive_root: string
      entries: Array<{
        name: string
        path: string
        bytes: number
        files: number
        mtime: number
      }>
    }>("/storage/archive"),
  storageDeleteArchiveEntry: (name: string) =>
    http<{ deleted: string; bytes_freed: number; files_removed: number }>(
      `/storage/archive/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    ),
  storageClearArchive: () =>
    http<{
      deleted: string[]
      bytes_freed: number
      files_removed: number
      failures: Array<{ name: string; error: string }>
    }>("/storage/archive", { method: "DELETE" }),
  storageClearHfCache: () =>
    http<{ deleted: string; bytes_freed: number; files_removed: number }>(
      "/storage/hf-cache",
      { method: "DELETE" },
    ),
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
  // ----- AI subsystem (ShiroManager-shaped) -----
  aiListProviders: () =>
    http<{ providers: AIProviderRecord[] }>("/ai/providers"),
  aiGetProvider: (id: string) =>
    http<AIProviderRecord>(`/ai/providers/${encodeURIComponent(id)}`),
  aiSaveProvider: (draft: AIProviderDraft) =>
    http<{ provider: AIProviderRecord }>("/ai/providers", {
      method: "PUT",
      body: JSON.stringify(draft),
    }),
  aiDeleteProvider: (id: string) =>
    http<{ ok: boolean; providerId: string }>(
      `/ai/providers/${encodeURIComponent(id)}`,
      { method: "DELETE" },
    ),
  aiListModels: (providerId?: string) =>
    http<{ models: AIModelRecord[] }>(
      providerId
        ? `/ai/models?provider_id=${encodeURIComponent(providerId)}`
        : "/ai/models",
    ),
  aiSaveModel: (draft: AIModelDraft) =>
    http<{ model: AIModelRecord }>("/ai/models", {
      method: "PUT",
      body: JSON.stringify(draft),
    }),
  aiDeleteModel: (id: string) =>
    http<{ ok: boolean; modelId: string }>(
      `/ai/models/${encodeURIComponent(id)}`,
      { method: "DELETE" },
    ),
  aiDiscoverModels: (providerId: string) =>
    http<{ models: AIModelRecord[] }>(
      `/ai/providers/${encodeURIComponent(providerId)}/discover-models`,
      { method: "POST" },
    ),
  aiListRoutes: () => http<{ routes: AIRouteRecord[] }>("/ai/routes"),
  aiSaveRoute: (draft: AIRouteDraft) =>
    http<{ route: AIRouteRecord }>("/ai/routes", {
      method: "PUT",
      body: JSON.stringify(draft),
    }),
  aiTestConnection: (input: AIConnectionTestInput) =>
    http<AIConnectionTestResult>("/ai/test", {
      method: "POST",
      body: JSON.stringify(input),
    }),
  aiInvokeTask: (input: AIInvokeTaskInput) =>
    http<AIInvokeTaskResult>("/ai/invoke", {
      method: "POST",
      body: JSON.stringify(input),
    }),
  aiResetKeyRuntime: (keyId: string) =>
    http<{ ok: boolean; keyId: string }>(
      `/ai/keys/${encodeURIComponent(keyId)}/reset-runtime`,
      { method: "POST" },
    ),
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
  getJobAnalysis: (id: string) =>
    http<{ analysis: JobAnalysis | null }>(`/jobs/${id}/analysis`),
  analyzeJob: (id: string) =>
    http<{ analysis: JobAnalysis }>(`/jobs/${id}/analyze`, { method: "POST" }),
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
  getSweepPareto: (sweep_id: string) =>
    http<SweepParetoResponse>(
      `/sweeps/${encodeURIComponent(sweep_id)}/pareto`,
    ),
}

/**
 * Live event stream. Prefers SSE (browser-native reconnect + Last-Event-ID
 * resume) and falls back to WebSocket if EventSource isn't available or
 * the SSE endpoint isn't reachable. The server keeps the legacy WS endpoint
 * alive for compatibility, so old builds keep working too.
 *
 * Resume semantics: the server tags every event with `id: <seq>`. On
 * reconnect EventSource forwards the last seen id back via Last-Event-ID,
 * and the server skips that many entries when replaying. So a tab nap or
 * a proxy-induced drop never loses events from the user's POV.
 */
export function useJobStream(jobId: string | null) {
  const ssePath = jobId
    ? `/api/jobs/${encodeURIComponent(jobId)}/sse`
    : null
  const wsPath = jobId ? `/api/jobs/${jobId}/stream` : null
  const { state, status } = useEventStream<
    TrainingEvent[],
    TrainingEvent | { type: "ping" }
  >({
    ssePath,
    wsPath,
    initialState: [],
    // Cap the in-memory tail so a long-running job can't blow out
    // browser memory. 500 entries comfortably covers the recent
    // window the events tab paints; older context lives in the
    // events.jsonl file on disk.
    reduce: (prev, parsed) =>
      [...prev, parsed as TrainingEvent].slice(-500),
    shouldDrop: (parsed) =>
      (parsed as { type?: string }).type === "ping",
  })
  return { events: state, status }
}

/**
 * Bootstrap event stream. SSE preferred, WS fallback. Same resume
 * semantics as useJobStream.
 */
export function useBootstrapStream(enabled: boolean) {
  const { state, status } = useEventStream<
    BootstrapEvent[],
    BootstrapEvent | { type: "ping" }
  >({
    ssePath: enabled ? "/api/backend/bootstrap/sse" : null,
    wsPath: enabled ? "/api/backend/bootstrap/stream" : null,
    initialState: [],
    reduce: (prev, parsed) =>
      [...prev, parsed as BootstrapEvent].slice(-200),
    shouldDrop: (parsed) =>
      (parsed as { type?: string }).type === "ping",
  })
  return { events: state, status }
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

export type AIReasoningEffort = "low" | "medium" | "high"
export type AIKeySelectionMode = "round_robin" | "random"
export type AIModelSource = "manual" | "discovered"

export const AI_TASK_IDS = [
  "global.default",
  "tagging.assist",
  "caption.rewrite",
  "dataset.analyze",
  "training.diagnose",
  "error.diagnose",
  "quality.score",
  "trigger.suggest",
] as const
export type AITaskId = (typeof AI_TASK_IDS)[number]

export interface AIProviderKeyRuntime {
  requestCount: number
  successCount: number
  failureCount: number
  consecutiveFailures: number
  lastUsedAt: string | null
  lastSucceededAt: string | null
  lastFailedAt: string | null
  lastError: string | null
  cooldownUntil: string | null
}

export interface AIProviderKeyRecord {
  id: string
  preview: string
  createdAt: string
  updatedAt: string
  runtime: AIProviderKeyRuntime
}

export interface AIProviderKeyDraft {
  id?: string | null
  value?: string
  preview?: string
}

export interface AIProviderRecord {
  id: string
  name: string
  kind: "openai-compatible"
  baseUrl: string
  organization: string
  project: string
  headers: Record<string, string>
  enabled: boolean
  hasApiKey: boolean
  apiKeyPreview: string
  apiKeyCount: number
  apiKeySelectionMode: AIKeySelectionMode
  apiKeys: AIProviderKeyRecord[]
  createdAt: string
  updatedAt: string
}

export interface AIProviderDraft {
  id?: string | null
  name: string
  kind?: "openai-compatible"
  baseUrl?: string
  organization?: string
  project?: string
  headers?: Record<string, string>
  enabled?: boolean
  apiKeySelectionMode?: AIKeySelectionMode
  apiKeys?: AIProviderKeyDraft[]
  apiKey?: string
  clearApiKey?: boolean
}

export interface AIModelRecord {
  id: string
  providerId: string
  modelId: string
  displayName: string
  source: AIModelSource
  enabled: boolean
  raw: Record<string, unknown>
  createdAt: string
  updatedAt: string
}

export interface AIModelDraft {
  id?: string | null
  providerId: string
  modelId: string
  displayName: string
  source?: AIModelSource
  enabled?: boolean
  raw?: Record<string, unknown>
}

export interface AIRouteRecord {
  taskId: string
  providerId: string | null
  modelId: string | null
  systemPrompt: string
  stream: boolean | null
  temperature: number | null
  topP: number | null
  frequencyPenalty: number | null
  presencePenalty: number | null
  maxOutputTokens: number | null
  seed: number | null
  reasoningEffort: AIReasoningEffort | null
  thinkingBudgetTokens: number | null
  includeReasoning: boolean | null
  stopSequences: string[]
  extraBodyJson: string
  enabled: boolean
  createdAt: string
  updatedAt: string
}

export interface AIRouteDraft {
  taskId: string
  providerId?: string | null
  modelId?: string | null
  systemPrompt?: string
  stream?: boolean | null
  temperature?: number | null
  topP?: number | null
  frequencyPenalty?: number | null
  presencePenalty?: number | null
  maxOutputTokens?: number | null
  seed?: number | null
  reasoningEffort?: AIReasoningEffort | null
  thinkingBudgetTokens?: number | null
  includeReasoning?: boolean | null
  stopSequences?: string[]
  extraBodyJson?: string
  enabled?: boolean
}

export interface AIConnectionTestInput {
  providerId: string
  modelId?: string | null
  prompt?: string | null
  systemPrompt?: string | null
  stream?: boolean | null
  temperature?: number | null
  topP?: number | null
  frequencyPenalty?: number | null
  presencePenalty?: number | null
  maxOutputTokens?: number | null
  seed?: number | null
  reasoningEffort?: AIReasoningEffort | null
  thinkingBudgetTokens?: number | null
  includeReasoning?: boolean | null
  stopSequences?: string[] | null
  extraBodyJson?: string | null
}

export interface AIInvokeTaskInput {
  taskId: string
  prompt: string
  systemPrompt?: string | null
  stream?: boolean | null
  temperature?: number | null
  topP?: number | null
  frequencyPenalty?: number | null
  presencePenalty?: number | null
  maxOutputTokens?: number | null
  seed?: number | null
  reasoningEffort?: AIReasoningEffort | null
  thinkingBudgetTokens?: number | null
  includeReasoning?: boolean | null
  stopSequences?: string[] | null
  extraBodyJson?: string | null
}

export interface AIUsage {
  promptTokens: number | null
  completionTokens: number | null
  totalTokens: number | null
}

export interface AIInvokeTaskResult {
  taskId: string
  providerId: string
  providerName: string
  modelId: string
  content: string
  reasoning: string | null
  finishReason: string | null
  usage: AIUsage | null
}

export interface AIConnectionTestResult {
  ok: boolean
  providerId: string
  providerName: string
  modelCount: number
  models: Array<{
    id: string | null
    object: string | null
    ownedBy: string | null
  }>
  completion: AIInvokeTaskResult | null
  error: string | null
}

/**
 * Subscribe to /api/system/sse for hardware telemetry. Falls back to the
 * legacy WS endpoint when EventSource isn't available.
 *
 * SSE has the proxy-friendly story we want: no upgrade handshake, no
 * AutoDL idle-kill (we send `: ping` comments), and the browser handles
 * reconnection on its own with the `retry: <ms>` directive the server
 * emits on connect.
 */
export function useSystemStream(enabled = true) {
  const { state, status } = useEventStream<
    SystemSnapshot | null,
    SystemSnapshot
  >({
    ssePath: enabled ? "/api/system/sse" : null,
    wsPath: enabled ? "/api/system/stream" : null,
    initialState: null,
    // Snapshot semantics: every frame replaces the previous state.
    // The history view is built from polled REST plus the live tail,
    // not a buffer here.
    reduce: (_prev, parsed) => parsed,
    reconnectOnVisibility: true,
  })
  return { snapshot: state, status }
}

// --------------------------------------------------------------------------- //
// Image Studio
// --------------------------------------------------------------------------- //

export interface ImageStudioItem {
  path: string
  relativePath: string
  name: string
  width: number | null
  height: number | null
  bytes: number
  mtime: number
  caption: string | null
  captionExists: boolean
  annotation: ImageStudioAnnotation | null
  thumbUrl: string
}

export interface ImageStudioAnnotation {
  aiCaption: string | null
  aiQualityScore: number | null
  aiQualityLabel: string | null
  aiQualityReason: string | null
  aiComposition: string | null
  aiTriggerWords: string[] | null
  userQualityLabel: string | null
  userNotes: string | null
  softDeleted: boolean
  favorite: boolean
}

export interface ImageStudioListResponse {
  path: string
  total: number
  page: number
  limit: number
  items: ImageStudioItem[]
}

export interface ImageStudioDetailItem extends ImageStudioItem {
  phash: Record<string, string>
  pendingOps: Array<{
    id: string
    op: string
    payload: Record<string, unknown>
    createdAt: string
  }>
}

export async function imageStudioList(params: {
  path: string
  recursive?: boolean
  page?: number
  limit?: number
  sort?: string
}): Promise<ImageStudioListResponse> {
  const qs = new URLSearchParams({ path: params.path })
  if (params.recursive) qs.set("recursive", "true")
  if (params.page) qs.set("page", String(params.page))
  if (params.limit) qs.set("limit", String(params.limit))
  if (params.sort) qs.set("sort", params.sort)
  return http<ImageStudioListResponse>(`/image-studio/list?${qs}`)
}

export async function imageStudioGetImage(
  path: string,
): Promise<ImageStudioDetailItem> {
  return http<ImageStudioDetailItem>(
    `/image-studio/image?path=${encodeURIComponent(path)}`,
  )
}

export async function imageStudioSaveAnnotation(body: {
  path: string
  userQualityLabel?: string | null
  userNotes?: string | null
  favorite?: boolean
  softDeleted?: boolean
}): Promise<{ ok: boolean; annotation: ImageStudioAnnotation }> {
  return http<{ ok: boolean; annotation: ImageStudioAnnotation }>(
    "/image-studio/annotations",
    {
      method: "PUT",
      body: JSON.stringify(body),
    },
  )
}

export async function imageStudioDeleteAnnotation(
  path: string,
): Promise<{ ok: boolean }> {
  return http<{ ok: boolean }>(
    `/image-studio/annotations?path=${encodeURIComponent(path)}`,
    { method: "DELETE" },
  )
}

export async function imageStudioAddOp(body: {
  path: string
  op: string
  payload?: Record<string, unknown>
}): Promise<{ id: string; op: string; payload: Record<string, unknown>; createdAt: string }> {
  return http<{
    id: string
    op: string
    payload: Record<string, unknown>
    createdAt: string
  }>("/image-studio/ops", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

export async function imageStudioListOps(
  path?: string,
): Promise<{ ops: Array<{ id: string; imagePath: string; op: string; payload: Record<string, unknown>; createdAt: string }> }> {
  const qs = path ? `?path=${encodeURIComponent(path)}` : ""
  return http<{
    ops: Array<{
      id: string
      imagePath: string
      op: string
      payload: Record<string, unknown>
      createdAt: string
    }>
  }>(`/image-studio/ops${qs}`)
}

export async function imageStudioDeleteOp(
  opId: string,
): Promise<{ ok: boolean }> {
  return http<{ ok: boolean }>(`/image-studio/ops/${opId}`, {
    method: "DELETE",
  })
}

export async function imageStudioApplyOps(
  path: string,
): Promise<{ applied: string[]; errors: Array<{ id: string; error: string }> }> {
  return http<{ applied: string[]; errors: Array<{ id: string; error: string }> }>(
    "/image-studio/ops/apply",
    {
      method: "POST",
      body: JSON.stringify({ path }),
    },
  )
}

// --------------------------------------------------------------------------- //
// Image Studio — Dedupe
// --------------------------------------------------------------------------- //

export interface DedupeClusterMember {
  path: string
  hash: string
}

export interface DedupeCluster {
  id: string
  kind: string
  members: DedupeClusterMember[]
  suggestedKeep: string
}

export async function imageStudioDedupeScan(body: {
  path: string
  recursive?: boolean
  algo?: string
  threshold?: number
}): Promise<{ computed: number; total: number; errors: Array<{ path: string; error: string }> }> {
  return http<{
    computed: number
    total: number
    errors: Array<{ path: string; error: string }>
  }>("/image-studio/dedupe/scan", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

export async function imageStudioDedupeClusters(params: {
  path: string
  kind?: string
  threshold?: number
}): Promise<{ clusters: DedupeCluster[] }> {
  const qs = new URLSearchParams({ path: params.path })
  if (params.kind) qs.set("kind", params.kind)
  if (params.threshold != null) qs.set("threshold", String(params.threshold))
  return http<{ clusters: DedupeCluster[] }>(
    `/image-studio/dedupe/clusters?${qs}`,
  )
}

export async function imageStudioBatchDelete(body: {
  paths: string[]
  forceFavorites?: boolean
}): Promise<{ deletedCount: number; deleted: string[]; bytesFreed: number; errors: Array<{ path: string; error: string }> }> {
  return http<{
    deletedCount: number
    deleted: string[]
    bytesFreed: number
    errors: Array<{ path: string; error: string }>
  }>("/image-studio/dedupe/batch-delete", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

// --------------------------------------------------------------------------- //
// Image Studio — Smart Caption / Tagging from Studio
// --------------------------------------------------------------------------- //

export async function imageStudioSmartCaption(params: {
  path: string
  recursive?: boolean
  device?: string
  mergeStrategy?: string
  captionMode?: "general" | "style" | "character"
  triggerWord?: string
  stripStyleTags?: boolean
  /** Optional progress callback. Fires after each poll with the latest snapshot. */
  onProgress?: (snap: {
    processed: number
    total: number
    percent: number
    last_image: string
    status: string
  }) => void
}): Promise<{ processed: number; results: unknown[]; errors: unknown[] }> {
  // Background-task adapter (uvicorn hang fix): the server now returns
  // 202 with a session_id; we poll status/<id> until it reaches a
  // terminal state, then resolve with the legacy shape so existing
  // callers keep working without code changes.
  const { onProgress, ...body } = params
  const submit = await http<{
    session_id: string
    total: number
    status_url: string
  }>("/image-studio/ai/smart-caption", {
    method: "POST",
    body: JSON.stringify(body),
  })
  const id = submit.session_id
  const poll = async (): Promise<{
    processed: number
    total: number
    percent: number
    last_image: string
    status: string
    results: unknown[]
    errors: unknown[]
    error: string | null
  }> => {
    return http(`/image-studio/ai/smart-caption/status/${encodeURIComponent(id)}`)
  }
  // Simple linear back-off — polls every 1s for first minute, then 3s.
  // Total runtime is bounded by the user explicitly cancelling or by
  // the page being closed (background thread finishes regardless).
  let elapsed = 0
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const snap = await poll()
    if (onProgress) {
      onProgress({
        processed: snap.processed,
        total: snap.total,
        percent: snap.percent,
        last_image: snap.last_image,
        status: snap.status,
      })
    }
    if (snap.status !== "running") {
      if (snap.status === "failed") {
        throw new Error(snap.error ?? "smart-caption batch failed")
      }
      return {
        processed: snap.processed,
        results: snap.results,
        errors: snap.errors,
      }
    }
    const interval = elapsed < 60 ? 1000 : 3000
    await new Promise<void>((resolve) => setTimeout(resolve, interval))
    elapsed += interval / 1000
  }
}

export async function imageStudioSmartCaptionSingle(params: {
  path: string
  device?: string
  captionMode?: "general" | "style" | "character"
  triggerWord?: string
  stripStyleTags?: boolean
  mergeStrategy?: string
}): Promise<{ caption: string; tags: string }> {
  return http<{ caption: string; tags: string }>(
    "/image-studio/ai/smart-caption/single",
    {
      method: "POST",
      body: JSON.stringify(params),
    },
  )
}

export async function startTaggingSession(params: {
  path: string
  tagger?: string
  model_id?: string
  general?: number
  character?: number
  device?: string
  overwrite?: boolean
  recursive?: boolean
}): Promise<{ session_id: string }> {
  return http<{ session_id: string }>("/tagging/tag", {
    method: "POST",
    body: JSON.stringify(params),
  })
}

export async function getTaggingSession(
  sessionId: string,
): Promise<TaggingSession> {
  return http<TaggingSession>(`/tagging/tag/${sessionId}`)
}

export async function imageStudioBatchCaption(params: {
  path: string
  recursive?: boolean
  task?: string
  mergeStrategy?: string
}): Promise<{ processed: number; results: unknown[]; errors: unknown[] }> {
  return http<{ processed: number; results: unknown[]; errors: unknown[] }>(
    "/image-studio/ai/caption",
    {
      method: "POST",
      body: JSON.stringify(params),
    },
  )
}

export async function imageStudioBatchQuality(params: {
  path: string
  recursive?: boolean
  task?: string
}): Promise<{ processed: number; results: unknown[]; errors: unknown[] }> {
  return http<{ processed: number; results: unknown[]; errors: unknown[] }>(
    "/image-studio/ai/quality",
    {
      method: "POST",
      body: JSON.stringify(params),
    },
  )
}

// --------------------------------------------------------------------------- //
// Dataset management
// --------------------------------------------------------------------------- //

export interface DatasetInfo {
  name: string
  path: string
  imageCount: number
  coverPath: string | null
  coverUrl: string | null
  meta: {
    name?: string
    description?: string
    targetResolution?: string
    triggerWord?: string
  }
}

export interface DatasetListResponse {
  root: string
  datasets: DatasetInfo[]
}

export async function datasetList(): Promise<DatasetListResponse> {
  return http<DatasetListResponse>("/image-studio/datasets")
}

export async function datasetCreate(body: {
  name: string
  description?: string
  targetResolution?: string
  triggerWord?: string
}): Promise<{ ok: boolean; path: string; meta: Record<string, string> }> {
  return http<{ ok: boolean; path: string; meta: Record<string, string> }>(
    "/image-studio/datasets",
    {
      method: "POST",
      body: JSON.stringify(body),
    },
  )
}

export async function datasetGetMeta(name: string): Promise<Record<string, string>> {
  return http<Record<string, string>>(
    `/image-studio/datasets/${encodeURIComponent(name)}/meta`,
  )
}

export async function datasetUpdateMeta(
  name: string,
  body: { description?: string; targetResolution?: string; triggerWord?: string },
): Promise<{ ok: boolean; meta: Record<string, string> }> {
  return http<{ ok: boolean; meta: Record<string, string> }>(
    `/image-studio/datasets/${encodeURIComponent(name)}/meta`,
    {
      method: "PUT",
      body: JSON.stringify(body),
    },
  )
}

export async function datasetDelete(name: string): Promise<{ ok: boolean }> {
  return http<{ ok: boolean }>(
    `/image-studio/datasets/${encodeURIComponent(name)}`,
    { method: "DELETE" },
  )
}

export interface UploadProgressEvent {
  file: string
  index: number
  total: number
  status: string
}

export interface UploadCompleteEvent {
  totalExtracted: number
  errors: string[]
}

export function datasetUpload(
  name: string,
  files: File[],
  opts: { keepCaptions?: boolean; onConflict?: string } = {},
): {
  eventSource: ReadableStream<{ event: string; data: unknown }>
  abort: () => void
} {
  const formData = new FormData()
  for (const f of files) {
    formData.append("files", f)
  }
  formData.append("keepCaptions", String(opts.keepCaptions ?? true))
  formData.append("onConflict", opts.onConflict ?? "rename")

  const controller = new AbortController()

  const stream = new ReadableStream<{ event: string; data: unknown }>({
    async start(ctrl) {
      try {
        const r = await fetch(
          `${API_BASE}/image-studio/datasets/${encodeURIComponent(name)}/upload`,
          { method: "POST", body: formData, signal: controller.signal },
        )
        if (!r.ok || !r.body) {
          ctrl.enqueue({ event: "error", data: { message: `upload failed: ${r.status}` } })
          ctrl.close()
          return
        }
        const reader = r.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ""
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split("\n")
          buffer = lines.pop() || ""
          let currentEvent = ""
          for (const line of lines) {
            if (line.startsWith("event: ")) {
              currentEvent = line.slice(7).trim()
            } else if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6))
                ctrl.enqueue({ event: currentEvent, data })
              } catch { /* skip malformed */ }
            }
          }
        }
        ctrl.close()
      } catch (e) {
        if ((e as Error).name !== "AbortError") {
          ctrl.enqueue({ event: "error", data: { message: String(e) } })
        }
        ctrl.close()
      }
    },
  })

  return { eventSource: stream, abort: () => controller.abort() }
}
