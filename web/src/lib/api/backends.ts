/**
 * Backend (training engine) status, install, and bootstrap types.
 *
 * The fetcher methods themselves live in `./client` as part of the
 * `api` object — this module owns the type surface and the
 * Bootstrap SSE hook.
 */

import { useEventStream } from "../use-event-stream"

export interface KohyaBackendStatus {
  id: "kohya"
  sd_scripts_path: string
  sd_scripts_ok: boolean
  missing_scripts: string[]
  python: string | null
  python_ok: boolean
  python_error: string | null
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
  python_error: string | null
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
  python_error: string | null
  venv_detected: boolean
  // Vendored — no LoraHub-managed requirements, always true. Kept on
  // the type for shape parity with the other two probes.
  requirements_ok: boolean
  missing_requirements: string[]
  /** Optional add-on needed only for backend.distributed.strategy=deepspeed_zero. */
  deepspeed_ok: boolean
  deepspeed_missing: boolean
  // Anima base / TE / VAE checkpoints. ``ready`` only covers the venv;
  // models are tracked separately so the install panel can offer a
  // dedicated "Download models" CTA.
  missing_models: string[]
  models_ok: boolean
  // MSVC build tools detection. Windows-only — on Linux/macOS the
  // ``platform_relevant`` flag is False and the install panel hides
  // the section entirely. On Windows ``ok=false`` means anima's
  // torch.compile path will crash inside Inductor codegen on the
  // first compile pass; we surface a one-click ``winget install``
  // CTA in that case.
  msvc: {
    platform_relevant: boolean
    ok: boolean
    cl_path: string | null
    msvc_version: string | null
    reason: string | null
    winget_available: boolean
  }
  ready: boolean
  // anima_lora's source dimension picks "vendored" instead of
  // "default" because the source ships with LoraHub itself, not as a
  // sibling clone.
  source: "env" | "settings" | "vendored"
}

export interface AIToolkitBackendStatus {
  id: "ai_toolkit"
  repo_path: string
  repo_ok: boolean
  missing_files: string[]
  python: string | null
  python_ok: boolean
  python_error: string | null
  venv_detected: boolean
  requirements_ok: boolean
  missing_requirements: string[]
  ready: boolean
  source: "env" | "settings" | "vendored" | "default"
}

export type AnyBackendStatus =
  | KohyaBackendStatus
  | DiffusionPipeBackendStatus
  | AnimaLoraBackendStatus
  | AIToolkitBackendStatus

export type BackendId = "kohya" | "diffusion-pipe" | "anima_lora" | "ai_toolkit"

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

export interface BackendUpdateCheck {
  backend_id: BackendId
  repo_path: string
  update_available: boolean
  current_sha: string
  remote_sha: string
  commits_behind: number
  branch: string
  error: string | null
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
  status: "idle" | "running" | "stop_requested" | "succeeded" | "failed" | "canceled" | "interrupted"
  session_id?: string
  source?: "modelscope" | "huggingface"
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

export interface MsvcInstallStatus {
  // Same shape as the model-download session but with a free-form
  // log buffer instead of file counts (winget doesn't expose a clean
  // percent signal). Server detects MSVC presence on every poll, so
  // ``msvc.ok`` flips to True the moment winget finishes — the UI
  // doesn't need to wait for the next backend-list refresh.
  status: "idle" | "running" | "succeeded" | "failed" | "canceled" | "interrupted"
  session_id?: string
  log?: string[]
  error?: string | null
  started_at?: number
  finished_at?: number | null
  msvc: {
    ok: boolean
    cl_path: string | null
    msvc_version: string | null
    reason: string | null
  }
}

export interface BootstrapEvent {
  step: string
  level: "info" | "done" | "error" | string
  message: string
  ts: number
}

export interface BootstrapStatus {
  status: "idle" | "running" | "succeeded" | "failed" | "canceled" | "interrupted"
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
  /** Canonical superset of config-level attention.training values. */
  all: string[]
}

export interface TorchWheelOption {
  cuda: string
  torch_version: string
  torchvision_version: string
  label: string
  min_driver: string
  compatible: boolean
  recommended: boolean
  reason: string
  notes: string
}

export interface TorchOptionsResponse {
  driver_version: string | null
  max_cuda: string | null
  options: TorchWheelOption[]
}

export interface BootstrapRequestBody {
  backend?: BackendId
  target?: string | null
  cuda?: string
  torch_version?: string
  torchvision_version?: string
  install_xformers?: boolean
  install_deepspeed?: boolean
  torch_override?: boolean
  force?: boolean
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
