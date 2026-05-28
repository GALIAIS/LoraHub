import { http } from "./core"

// ─── Weights & Biases (read-only proxy) ──────────────────────────────

export interface WandbStatusResponse {
  installed: boolean
  api_key_configured: boolean
  base_url: string | null
}

export interface WandbRunSummary {
  entity: string
  project: string
  run_id: string
  name: string | null
  state: string | null
  url: string
  config: Record<string, unknown>
  summary: Record<string, unknown>
  tags: string[]
}

export interface WandbHistoryResponse {
  keys: string[]
  rows: Array<Record<string, number | string | null>>
  sampled: boolean
  samples_requested: number
}

/**
 * Read-only client for the wandb public API. Backed by
 * `lorahub/api/routers/wandb_routes.py` which calls `wandb.Api()` on
 * the server. Used by the "训练分析 → W&B" tab so the user doesn't
 * have to leave LoraHub to inspect their run.
 */
export const wandbApi = {
  status: () => http<WandbStatusResponse>(`/wandb/status`),
  runSummary: (jobId: string) =>
    http<WandbRunSummary>(`/wandb/runs/${encodeURIComponent(jobId)}/summary`),
  runHistory: (jobId: string, opts?: { keys?: string[]; samples?: number }) => {
    const params = new URLSearchParams()
    if (opts?.samples !== undefined) {
      params.set("samples", String(opts.samples))
    }
    if (opts?.keys && opts.keys.length > 0) {
      for (const k of opts.keys) params.append("keys", k)
    }
    const qs = params.toString()
    return http<WandbHistoryResponse>(
      `/wandb/runs/${encodeURIComponent(jobId)}/history${qs ? `?${qs}` : ""}`,
    )
  },
}
