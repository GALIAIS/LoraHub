import { useEventStream } from "../use-event-stream"
import type { BackendId } from "./backends"

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
  // Optional per-step telemetry normalized across backend parsers. Absent
  // when the trainer does not emit the corresponding signal.
  lr?: number | null
  iter_time_s?: number | null
  samples_per_sec?: number | null
  eta_s?: number | null
  snr?: number | null
  grad_norm?: number | null
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
  /**
   * Side-band SVD summary for every saved LoRA checkpoint. Empty for
   * runs without LoRA adapters or with `sampling.spectrum_analysis=false`.
   */
  lora_spectrum: Array<{
    step: number | null
    checkpoint: string | null
    layers: number | null
    /** Geometric mean of per-layer (Σσ)² / Σσ². */
    effective_rank: number | null
    /** Mean fraction of energy in the top singular value, [0..1]. */
    top1_energy: number | null
    /** Mean Frobenius norm of ΔW = α·B·A. */
    fro_norm: number | null
    ts: number
  }>
  /**
   * Catastrophic-forgetting probe — perceptual similarity of neutral-
   * prompt samples against the earliest seen sample for that prompt.
   * `preserved` is in [0..1]; 1 = identical to baseline, 0 = totally
   * different.
   */
  forgetting_probe: Array<{
    step: number | null
    checkpoint: string | null
    preserved: number | null
    samples: number | null
    image_path: string | null
    ts: number
  }>
  last_step: number | null
  last_nonfinite_loss: { step: number | null; loss: number | null; ts: number | null } | null
  last_nonfinite_val_loss: { step: number | null; loss: number | null; ts: number | null } | null
  nonfinite_loss: Array<{ step: number | null; loss: null; ts: number | null }>
  nonfinite_val_loss: Array<{ step: number | null; loss: null; ts: number | null }>
  cache_progress: Array<{
    phase: string | null
    done: number | null
    total: number | null
    percent: number | null
    rate: string | null
    eta_s: number | null
    ts: number
  }>
  diagnostics: Array<{
    category: string | null
    severity: string | null
    message: string | null
    remediation: string | null
    evidence: string | null
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

// --- Diagnose / recommend (training_assistant) -----------------------

export interface DiagnosisFinding {
  category: string
  severity: "info" | "warn" | "error"
  message: string
  remediation: string
  evidence: string
}

export interface JobDiagnosis {
  findings: DiagnosisFinding[]
  summary: string
  log_excerpt: string
  log_path: string | null
}

export interface HyperparamRecommendInput {
  dataset_size: number
  gpu_vram_mb: number
  backend?: BackendId
  target?: "character" | "style" | "concept"
}

export interface HyperparamSuggestion {
  batch_size: number
  gradient_accumulation_steps: number
  learning_rate: number
  network_dim: number
  network_alpha: number
  max_train_epochs: number
  optimizer_type: string
  extra_flags: Record<string, unknown>
  rationale: string[]
}

export interface HyperparamRecommendResponse {
  suggestion: HyperparamSuggestion
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

export interface ArtifactRow {
  job_id: string
  workspace: string
  exists: boolean
  state: string
  created_at: string | null
  finished_at: string | null
  output_name?: string | null
  checkpoints: JobFile[]
  samples: JobFile[]
  total_bytes: number
  checkpoint_count: number
  sample_count: number
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
    reduce: (prev, parsed) => {
      const ev = parsed as TrainingEvent
      return [...prev, ev].slice(-500)
    },
    shouldDrop: (parsed) =>
      (parsed as { type?: string }).type === "ping",
  })
  return { events: state, status }
}
