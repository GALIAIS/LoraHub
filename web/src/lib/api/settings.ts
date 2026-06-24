import type { AnyBackendStatus, BackendId } from "./backends"

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
  max_concurrent_jobs: number
  gpu_dispatch_mode: "one-job-per-gpu" | "distributed"
  gpu_dispatch_num_gpus: number | null
  github_proxy: string | null
  huggingface_endpoint: string | null
  modelscope_enabled: boolean
  modelscope_token: string | null
  pypi_index_url: string | null
  torch_index_url: string | null
  download_proxy: string | null
  huggingface_token: string | null
  wandb_api_key: string | null
  wandb_base_url: string | null
  terminal_unrestricted: boolean
  terminal_command_timeout_s: number
  // Error registry fan-out — see Settings.error_upstream_*
  error_upstream_channel: "off" | "gitlab" | "gitea" | "webhook"
  error_upstream_gitlab_base_url: string
  error_upstream_gitlab_repo: string
  error_upstream_gitlab_token: string
  error_upstream_webhook_url: string
  error_upstream_webhook_auth_header: string
  error_upstream_auto_severity: "off" | "error" | "all"
  extra: Record<string, unknown>
}

export interface SettingsResponse {
  settings: SettingsState
  backend: AnyBackendStatus
  backends: Record<BackendId, AnyBackendStatus>
  path: string
}
