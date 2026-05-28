import { http, readSseEvents, ApiError, API_BASE } from "./core"
import type {
  JobSummary,
  JobDetail,
  TrainingEvent,
  JobFilesResponse,
  JobMetricsResponse,
  JobAnalysis,
  JobDiagnosis,
  HyperparamRecommendInput,
  HyperparamRecommendResponse,
  ArtifactRow,
  SampleGalleryResponse,
} from "./jobs"
import type {
  ConfigListEntry,
  ConfigDetail,
  ValidateResponse,
  ConfigTemplate,
} from "./configs"
import type {
  BackendId,
  BackendsResponse,
  BackendUpdateCheck,
  AnimaModelDownloadStatus,
  MsvcInstallStatus,
  BootstrapStartResponse,
  BootstrapStatus,
  BootstrapRequestBody,
  AttentionBackendsResponse,
} from "./backends"
import type { SettingsState, SettingsResponse } from "./settings"
import type { DatasetScanResponse, DatasetCaptionResponse } from "./datasets"
import type { ModelDownloadSession, ScannedModelsResponse } from "./models"
import type { TaggingSession, TagDatasetRequest } from "./tagging"
import type {
  AIProviderRecord,
  AIProviderDraft,
  AIModelRecord,
  AIModelDraft,
  AIRouteRecord,
  AIRouteDraft,
  AIConnectionTestInput,
  AIConnectionTestResult,
  AIInvokeTaskInput,
  AIInvokeTaskResult,
} from "./ai"
import type { SystemSnapshot, UpdateInfo, UpdateEvent } from "./system"
import type { MirrorPreset, ProbeResult } from "./network"
import type {
  SweepSummary,
  SweepDetail,
  SweepParetoResponse,
} from "./sweeps"

export const api = {
  health: () => http<{ status: string; version: string }>("/health"),
  listJobs: () => http<{ jobs: JobSummary[] }>("/jobs"),
  getJob: (id: string) => http<JobDetail>(`/jobs/${id}`),
  getEvents: (id: string, limit = 200) =>
    http<{ events: TrainingEvent[] }>(`/jobs/${id}/events?limit=${limit}`),
  cancelJob: (id: string) =>
    http<JobSummary>(`/jobs/${id}`, { method: "DELETE" }),
  /** Bulk archive: every id is moved to ``_archive/`` if it's in a
   *  terminal state. Server groups outcomes so the UI can render
   *  "成功 N · 跳过 M · 失败 K" in one shot rather than retrying
   *  per-id over the legacy DELETE endpoint. */
  bulkArchiveJobs: (ids: string[]) =>
    http<{
      archived: { id: string; workspace_moved_to: string | null; warnings: string[] }[]
      skipped: { id: string; reason: string }[]
      failed: { id: string; reason: string }[]
      not_found: string[]
    }>("/jobs/archive", {
      method: "POST",
      body: JSON.stringify({ ids }),
    }),
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
  /**
   * LLM-driven config recommendation.
   *
   * **Temporarily disabled for stability.** The upstream proxy fronting
   * the configured AI provider drops multi-thousand-token prompts at
   * the 60-second mark with ``Server disconnected without sending a
   * response``, surfacing as a 422 with a confusing error in the UI.
   * Until the advisor is reworked onto a streaming code path, the
   * client just throws so any stale caller surfaces the disabled
   * state instead of silently spinning a 60-second request.
   *
   * The backend route ``/api/configs/llm-advise`` returns a 503 with
   * the same rationale, so a direct curl from outside the app sees
   * the same shutdown.
   */
  llmAdviseConfig: (_body: {
    currentCfg: Record<string, unknown>
    intent?: string
    vramMib?: number | null
    gpuName?: string | null
    datasetPath?: string | null
    datasetImageCount?: number | null
  }): never => {
    throw new Error(
      "智能推荐已暂时停用 (上游 LLM 流量层超时问题)。后续会切换到 streaming 路径再恢复。",
    )
  },
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
  checkBackendUpdate: (backendId: BackendId) =>
    http<BackendUpdateCheck>(`/backends/${backendId}/check-update`),
  updateBackend: (backendId: BackendId) =>
    http<BackendUpdateCheck>(`/backends/${backendId}/update`, { method: "POST" }),
  startAnimaModelDownload: () =>
    http<AnimaModelDownloadStatus>("/backends/anima_lora/download-models", {
      method: "POST",
    }),
  getAnimaModelDownloadStatus: () =>
    http<AnimaModelDownloadStatus>(
      "/backends/anima_lora/download-models/status",
    ),
  startMsvcInstall: () =>
    http<MsvcInstallStatus>("/backends/anima_lora/install-msvc", {
      method: "POST",
    }),
  getMsvcInstallStatus: () =>
    http<MsvcInstallStatus>("/backends/anima_lora/install-msvc/status"),
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
  scanModels: (root?: string) => {
    const qs = root ? `?root=${encodeURIComponent(root)}` : ""
    return http<ScannedModelsResponse>(`/models/scan${qs}`)
  },
  tagDataset: (body: TagDatasetRequest) =>
    http<TaggingSession>("/tagging/tag", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  getTaggingSession: (sessionId: string) =>
    http<TaggingSession>(`/tagging/tag/${sessionId}`),
  /**
   * Curated SmilingWolf WD14 catalogue. Source of truth lives in
   * ``lorahub/core/tagging/wd14.py`` — the UI dropdowns must NOT
   * hard-code repo ids, or they go stale (and trip a HuggingFace
   * 401/404 when an old short name like ``wd-eva02-large-v3`` is
   * still in flight after the canonical name was tightened).
   */
  listWd14Models: () =>
    http<{ default: string; models: { id: string; label: string }[] }>(
      "/tagging/wd14/models",
    ),
  getTaggerDownloadStatus: () =>
    http<{
      jobs: {
        repo_id: string
        filename: string
        status: "running" | "done" | "error"
        downloaded: number
        total: number | null
        percent: number | null
        started_at: number
        finished_at: number | null
        error: string | null
      }[]
    }>("/tagging/download-status"),
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
  aiListRecommendedPrompts: () =>
    http<{ prompts: Record<string, string> }>("/ai/recommended-prompts"),
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
  getSystemVersion: async (
    channel: "dev" | "tag" = "tag",
    force = false,
  ): Promise<UpdateInfo> => {
    const qs = `?channel=${channel}${force ? "&force=true" : ""}`
    try {
      return await http<UpdateInfo>(`/system/version${qs}`)
    } catch (err) {
      // Backend rolled out the channel rename in v1.0.4 (Literal
      // "main" → "dev"). Older deployments still validate the
      // request body against the original Literal and reject "dev"
      // with a FastAPI 422. Fall back to the pre-rename name and
      // patch the response so the UI's ``channel === "dev"`` checks
      // keep working — the user shouldn't have to upgrade their
      // backend just to see the maintenance card.
      const isLegacyChannelRejection =
        channel === "dev" &&
        err instanceof ApiError &&
        err.status === 422
      if (!isLegacyChannelRejection) {
        throw err
      }
      const legacyQs = `?channel=main${force ? "&force=true" : ""}`
      const legacy = await http<UpdateInfo>(`/system/version${legacyQs}`)
      return { ...legacy, channel: "dev" }
    }
  },
  /**
   * Run a self-update via the SSE stream endpoint. ``onEvent`` fires
   * for every progress line; the returned promise resolves when the
   * stream closes (terminal ``done`` / ``error`` event delivered).
   * Cancel mid-stream by aborting the passed signal.
   */
  applySystemUpdate: async (
    body: {
      channel: "dev" | "tag"
      build: boolean
      restart: boolean
      force?: boolean
    },
    onEvent: (ev: UpdateEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> => {
    const url = `${API_BASE}/system/update`
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    })
    if (!resp.ok) {
      throw new ApiError(
        resp.status,
        resp.statusText,
        await resp.text().catch(() => ""),
        "/system/update",
      )
    }
    await readSseEvents<UpdateEvent>(resp, onEvent)
  },
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
  /**
   * Heuristic failure-mode diagnosis. Reads the job's events.jsonl +
   * trailing log lines on the server, runs a small regex panel, and
   * returns findings + remediation hints. Pure read — cheap to call
   * repeatedly.
   */
  diagnoseJob: (id: string) =>
    http<JobDiagnosis>(`/jobs/${id}/diagnose`),
  /**
   * Hyperparameter recommender: dataset_size + gpu_vram_mb in,
   * concrete config knobs out. Decoupled from any specific job —
   * useful in the "new training run" flow.
   */
  recommendHyperparams: (input: HyperparamRecommendInput) =>
    http<HyperparamRecommendResponse>("/jobs/recommend", {
      method: "POST",
      body: JSON.stringify(input),
    }),
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
  // ── Artifacts ──────────────────────────────────────────────────
  listArtifacts: () => http<{ jobs: ArtifactRow[] }>("/artifacts"),  /**
   * URL of the streaming-zip endpoint for a job's artifacts. Returned
   * as a string so the UI can drop it straight into ``window.open(...)``
   * — letting the browser save-as instead of buffering the whole
   * archive in memory.
   */
  artifactZipUrl: (job_id: string, include: string[] = ["checkpoints"]) =>
    `/api/artifacts/${encodeURIComponent(job_id)}/zip?include=${encodeURIComponent(
      include.join(","),
    )}`,
  artifactSingleUrl: (job_id: string, path: string) =>
    `/api/jobs/${encodeURIComponent(job_id)}/files/raw?path=${encodeURIComponent(path)}`,
  deleteArtifactFile: (job_id: string, path: string) =>
    http<{ deleted: string; size_bytes: number }>(
      `/artifacts/${encodeURIComponent(job_id)}/file?path=${encodeURIComponent(path)}`,
      { method: "DELETE" },
    ),
  deleteArtifactWorkspace: (job_id: string) =>
    http<{ deleted: boolean; workspace?: string; reason?: string }>(
      `/artifacts/${encodeURIComponent(job_id)}/workspace`,
      { method: "DELETE" },
    ),
}
