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
  /** Plan config (axes + name_template + mode etc.) when persisted. */
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
